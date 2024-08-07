import os
import torch
from gorilla.config import Config
from utils import *
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
    parser.add_argument("--checkpoint", default="log/detectiondiffusion_pointnet_bs16/current_model.t7",help="path to checkpoint model")
    parser.add_argument("--test_data", help="path to test_data")
    args = parser.parse_args()
    return args

def run_maad():
    transl_loss_sum = 0
    rot_loss_sum = 0
    joint_loss_sum = 0
    coverage_sum = 0
    num_nan_out = 0
    num_nan_transl = 0
    num_nan_rot = 0
    num_nan_joint = 0

    print("Testing")
    model.eval()
    with torch.no_grad():
        for idx in range(len(batch['obj_name'])):
            palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
            grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])
            num_gt_grasps = grasps_gt['transl'].shape[0]
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

            out = model.detect_and_sample(xyz, n_sample=100, guide_w=GUIDE_W)

            transl_loss, rot_loss, joint_loss, coverage = maad_for_grasp_distribution(out, grasps_gt,L1=True)
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
        print(f'transl_loss_mean per grasp (m): {transl_loss_sum/num_grasp:.6f}')
        print(f'rot_loss_sum: {rot_loss_sum:.3f}')
        print(f'rot_loss_mean per grasp (rad): {rot_loss_sum/num_grasp:.3f}')
        print(f'joint_loss_sum: {joint_loss_sum:.3f}')
        print(f'joint_loss_mean per grasp (rad^2): {joint_loss_sum/num_grasp:.3f}')
        print(f'coverage: {coverage_mean:.3f}')

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = build_model(cfg).to(DEVICE)

    if args.checkpoint != None:
        print("Loading checkpoint....")
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
    # loader_dict = torch.utils.data.DataLoader(dset_gen,
    #                                             batch_size=cfg.training_cfg.batch_size,
    #                                             shuffle=True,
    #                                             drop_last=True,
    #                                             num_workers=cfg.training_cfg.num_worker)

    batch = torch.load('data/eval_batch.pth', map_location="cuda:0")



        # for k, v in loss_per_item.items():
    # print("Testing")
    # model.eval()
    # with torch.no_grad():
    #     for shape in tqdm(shape_data):
    #         xyz = torch.from_numpy(shape['full_shape']['coordinate']).unsqueeze(0).float().cuda()
    #         shape['result'] = {text: [*(model.detect_and_sample(xyz, text, 2000, guide_w=GUIDE_W))] for text in shape['affordance']}

    # with open('./result.pkl', 'wb') as f:
    #     pickle.dump(shape_data, f)
    for GUIDE_W in np.linspace(0,1,5):
        print(GUIDE_W)
        run_maad()