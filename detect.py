import os
import torch
from gorilla.config import Config
from utils import *
from utils.utils import hom_matrix_from_transl_rot_matrix

import argparse
import pickle
import math
from tqdm import tqdm
from utils.grasp_data_handler import GraspDataHandlerVae
from dataset import FFHGeneratorDataset
from utils.metrics import maad_for_grasp_distribution
import open3d as o3d
import numpy as np

GUIDE_W = 0.2 # 0.5, in paper is 0.2
DEVICE=torch.device('cuda')


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("--config", default="config/detectiondiffusion.py", help="test config file path")
    parser.add_argument("--checkpoint", default="log/detectiondiffusion_bps_bs64_1e-4",help="path to checkpoint model")
    parser.add_argument("--test_data", help="path to test_data")
    args = parser.parse_args()
    return args


def show_generated_grasp_distribution(pcd_path,
                                      grasps,
                                      highlight_idx=-1,
                                      custom_vis=True,
                                      save_ix=0):
    """Visualizes the object point cloud together with the generated grasp distribution.

    Args:
        path (str): Path to the object pcd
        grasps (dict): contains arrays rot_matrix [n*3*3], palm transl [n*3], joint_conf [n*15]
    """
    n_samples = grasps['rot_matrix'].shape[0]
    frames = []
    for i in range(n_samples):
        rot_matrix = grasps['rot_matrix'][i, :, :]
        transl = grasps['transl'][i, :]
        if np.isnan(rot_matrix).any() or np.isnan(transl).any():
            continue
        palm_pose_centr = hom_matrix_from_transl_rot_matrix(
            transl, rot_matrix)
        if i == highlight_idx:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.065).transform(
                palm_pose_centr)
        else:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.0075).transform(
                palm_pose_centr)
        frames.append(frame)

    # visualize
    ## If you want to add origin, this
    # orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.001)
    # orig = orig.scale(5, center=orig.get_center())
    # frames.append(orig)

    obj = o3d.io.read_point_cloud(pcd_path)
    #obj = obj.voxel_down_sample(0.002)
    obj.paint_uniform_color([230. / 255., 230. / 255., 10. / 255.])
    obj.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=100))
    frames.append(obj)
    if custom_vis:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1)
        frames.append(origin)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for f in frames:
            vis.add_geometry(f)

        vis.run()
        # if save_ix != -1:
        #     key = input("Save image?: ")
        #     if key == 'y':
        #         vis.capture_screen_image("/home/yb/Pictures/ffhflow/{}.png".format(save_ix))

        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries(frames)

def visualize(samples, pcd_path):
    if torch.is_tensor(samples['rot_matrix']):
        samples_copy = {}
        for key, value in samples.items():
            samples_copy[key] = value.cpu().data.numpy()
    else:
        samples_copy = samples
    show_generated_grasp_distribution(pcd_path, samples_copy)

def run_maad(model):
    transl_loss_sum = 0
    rot_loss_sum = 0
    joint_loss_sum = 0
    coverage_sum = 0
    num_nan_out = 0
    num_nan_transl = 0
    num_nan_rot = 0
    num_nan_joint = 0
    num_valid_grasps = 0

    # print("Testing")
    model.eval()
    with torch.no_grad():
        for idx in range(len(batch['obj_name'])):
            # palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
            grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])
            # num_gt_grasps = grasps_gt['transl'].shape[0]
            if cfg.dataset.use_bps:
                xyz = batch['bps_object'][idx].unsqueeze(0).float().cuda() # old: bps.to('cuda')
            else:
                pcd_path = batch['pcd_path'][idx]
                pcd_path = pcd_path.replace('/data/hdd1/qf/hithand_data','/data/net/userstore/qf/hithand_data/data')
                pcd_path = pcd_path.replace('_pcd','_dspcd')
                xyz = o3d.io.read_point_cloud(pcd_path)
                xyz = np.asarray(xyz.points)
                xyz = torch.from_numpy(xyz).unsqueeze(0).float().cuda()
                # xyz = xyz.view(1,-1,3)

            # out = model.detect_and_sample(xyz, n_sample=100, guide_w=GUIDE_W)
            out = model.detect_and_sample_no_classifier_free(xyz, n_sample=100)

            vis=False
            if vis:
                pcd_path = batch['pcd_path'][idx]
                pcd_path = pcd_path.replace('/data/hdd1/qf/hithand_data','/data/net/userstore/qf/hithand_data/data')
                visualize(out, pcd_path)

            transl_loss, rot_loss, joint_loss, coverage, num_valid_grasp = maad_for_grasp_distribution(out, grasps_gt,L1=True)
            num_valid_grasps += num_valid_grasp

            if not math.isnan(transl_loss) and not math.isnan(rot_loss) and not math.isnan(joint_loss):
                transl_loss_sum += transl_loss
                rot_loss_sum += rot_loss
                joint_loss_sum += joint_loss
            else:
                if math.isnan(transl_loss):
                    num_nan_transl += 1
                if math.isnan(rot_loss):
                    num_nan_rot += 1
                if math.isnan(joint_loss):
                    num_nan_joint += 1
                num_nan_out += 1
            coverage_sum += coverage

        coverage_mean = coverage_sum / len(batch['obj_name'])
        num_grasp = 100 * len(batch['obj_name'])
        print(f'transl_loss_sum: {transl_loss_sum:.3f}')
        # print(f'transl_loss_mean per grasp (m): {transl_loss_sum/num_grasp:.6f}')
        print(f'rot_loss_sum: {rot_loss_sum:.3f}')
        # print(f'rot_loss_mean per grasp (rad): {rot_loss_sum/num_grasp:.3f}')
        print(f'joint_loss_sum: {joint_loss_sum:.3f}')
        # print(f'joint_loss_mean per grasp (rad^2): {joint_loss_sum/num_grasp:.3f}')
        print(f'coverage: {coverage_mean:.3f}')
        print(f'num of valid grasps {num_valid_grasps} from total of 6400')
        # print(f'number of nan for transl, rot and joint: ', num_nan_transl, num_nan_rot, num_nan_joint)

if __name__ == "__main__":
    args = parse_args()
    # cfg = Config.fromfile(args.config)

    # filename = "output.cls"
    batch = torch.load('data/eval_batch.pth', map_location="cuda:0")

    # Open the file in write mode
    # with open(filename, 'w') as file:
    model_names = ['detectDif_lr1e-6_bz32_nT100_dp00']
    for model_name in model_names:
        test = [10,20,30,40,50,100,150,200]
        # for idx in test:
        idx = 50
        path2checkpoint = os.path.join('log',model_name,'current_model_' + str(idx)+'.t7')
        if not os.path.exists(path2checkpoint):
            continue
        path2config = os.path.join('log',model_name,'detectiondiffusion.py')
        if not os.path.exists(path2config):
            continue
        cfg = Config.fromfile(path2config)
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
        model = build_model(cfg).to(DEVICE)

        print('test model of:',path2checkpoint)
        if args.checkpoint != None:
            # print("Loading checkpoint....")
            _, exten = os.path.splitext(args.checkpoint)
            if exten == '.t7':
                model.load_state_dict(torch.load(args.checkpoint))
            elif exten == '.pth':
                check = torch.load(args.checkpoint)
                model.load_state_dict(check['model_state_dict'])
        else:
            raise ValueError("Must specify a checkpoint path!")

        # load test data
        grasp_data_path = os.path.join(cfg.dataset.PATH, cfg.dataset.GRASP_DATA_NANE)
        grasp_data = GraspDataHandlerVae(grasp_data_path)
        loader_dict = build_loader(cfg) #, dataset_dict)       # build the loader
        val_dataset = FFHGeneratorDataset(cfg,eval=False)

        # for GUIDE_W in np.linspace(0,1,5):
        # GUIDE_W = 0.5
            # print("GUIDE_W ",GUIDE_W, file=file)
        run_maad(model)
