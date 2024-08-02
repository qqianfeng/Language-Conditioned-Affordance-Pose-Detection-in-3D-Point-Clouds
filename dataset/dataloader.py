import numpy as np
import h5py
import os
import pandas as pd
import sys
import torch
from torch.utils import data
import open3d as o3d
import transforms3d

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from utils.grasp_data_handler import GraspDataHandlerVae
from utils import utils#, visualization
from configs import get_config


class FFHGeneratorDataset(data.Dataset):
    def __init__(self, cfg, eval=False, dtype=torch.float32):
        if eval:
            ds_name = "eval"
        else:
            ds_name = "train"

        self.dtype = dtype

        self.ds_path = os.path.join(cfg.DATASETS.PATH, ds_name)
        self.objs_names = self.get_objs_names(self.ds_path)
        self.objs_folder = os.path.join(self.ds_path, 'bps')
        self.grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
        self.gazebo_obj_path = cfg.DATASETS.GAZEBO_OBJ_PATH

        self.grasp_data_handler = GraspDataHandlerVae(self.grasp_data_path)
        df = pd.read_csv(os.path.join(cfg.DATASETS.PATH, 'metadata.csv'))
        df_name_pos = df[df[ds_name] == 'X'].loc[:, ['Unnamed: 0', 'positive']]
        self.num_success_per_object = dict(
            zip(df_name_pos.iloc[:, 0], df_name_pos.iloc[:, 1].astype('int64')))
        df_name_neg = df[df[ds_name] == 'X'].loc[:, ['Unnamed: 0', 'negative']]
        self.num_failure_per_object = dict(
            zip(df_name_neg.iloc[:, 0], df_name_neg.iloc[:, 1].astype('int64')))

        if cfg['DATASETS']['POSITIVE_ONLY']:
            self.bps_paths, self.grasp_idxs = self.get_all_bps_paths_and_grasp_idxs(
                self.objs_folder, self.num_success_per_object)
        elif cfg['DATASETS']['NEGATIVE_ONLY']:
            self.bps_paths, self.grasp_idxs = self.get_all_bps_paths_and_grasp_idxs(
                self.objs_folder, self.num_failure_per_object)
        else:
            raise KeyError("Wrong flag set for nagetiva and positive grasps!")

        self.cfg = cfg
        self.is_debug = False
        if self.is_debug:
            print("The size in KB is: ", sys.getsizeof(self.bps_paths) / 1000)

    def get_objs_names(self, path):
        objs_folder = os.path.join(path, 'pcd')
        return [obj for obj in os.listdir(objs_folder) if '.' not in obj]

    def get_all_bps_paths_and_grasp_idxs(self, objs_folder, success_per_obj_dict):
        """ Creates a long list where each of the N bps per object get repeated as many times
        as there are positive grasps for this object. It also returns a list of indexes with the same length as the bps list
        indicating the grasp index. This way each bps is uniquely belonging to each valid grasp ONCE
        Args:
            objs_folder (str, path): The path to the folder where the bps lie.
            num_success_per_object (dict): A dict with keys being all the object names in the current dataset and values the successful grasps for each object.
        Returns:
            paths (list): List of bps file paths. Each bps occuring as often as there are positive grasps for each object.
            grasp_idxs (list): List of ranges from 0 to n_success_grasps per object and bps.
        """
        paths = []
        grasp_idxs = []
        for obj, n_success in success_per_obj_dict.items():
            obj_path = os.path.join(objs_folder, obj)
            for f_name in os.listdir(obj_path):
                if f_name.split('.')[0].split('_')[-1] == 'single':
                    continue
                elif f_name.split('.')[0].split('_')[-1] == 'obstacle':
                    continue
                f_path = os.path.join(obj_path, f_name)
                if 'bps' in os.path.split(f_name)[1]:
                    paths += n_success * [f_path]
                    grasp_idxs += range(0, n_success)
            # add break for load only one obj
            # break
        assert len(paths) == len(grasp_idxs)
        return paths, grasp_idxs

    def read_pcd_transform(self, bps_path):
        # pcd save path from bps save path
        base_path, bps_name = os.path.split(bps_path)
        pcd_name = bps_name.replace('bps', 'pcd')
        pcd_name = pcd_name.replace('.npy', '.pcd')
        path = os.path.join(base_path, pcd_name)

        # Extract object name from path
        head, pcd_file_name = os.path.split(path)
        pcd_name = pcd_file_name.split('.')[0]
        obj = os.path.split(head)[1]

        # Read the corresponding transform in
        path = os.path.join(os.path.split(self.ds_path)[0], 'pcd_transforms.h5')
        with h5py.File(path, 'r') as hdf:
            # for dataset with multi objects, pcd name ends with 'multi' which has to be removed.
            if pcd_name.find('_multi') != -1:
                pcd_name = pcd_name[:pcd_name.find('_multi')]
            if pcd_name.find('_obstacle') != -1:
                pcd_name = pcd_name[:pcd_name.find('_obstacle')]
            pos_quat_list = hdf[obj][pcd_name + '_mesh_to_centroid'][()]

        # Transform the transform to numpy 4*4 array
        hom_matrix = utils.hom_matrix_from_pos_quat_list(pos_quat_list)
        return hom_matrix

    def get_grasps_from_pcd_path(self, pcd_path,label='positive'):
        base_path, pcd_name = os.path.split(pcd_path)
        base_path = base_path.replace('pcd','bps')
        bps_name = pcd_name.replace('pcd', 'bps')
        bps_name = bps_name.replace('.bps','.npy')
        bps_path = os.path.join(base_path, bps_name)
        obj_name = '_'.join(bps_name.split('_bps')[:-1])
        centr_T_mesh = self.read_pcd_transform(bps_path)
        # bps_path = bps_path.replace('multi','single')
        palm_poses, joint_confs, _ = self.grasp_data_handler.get_grasps_for_object(obj_name=obj_name,outcome=label)

        palm_poses_rot_mat = np.zeros((len(palm_poses),3,3))
        palm_poses_transl = np.zeros((len(palm_poses),3))

        for idx in range(len(palm_poses)):
            palm_pose_hom = utils.hom_matrix_from_pos_quat_list(palm_poses[idx])
            palm_pose_centr = np.matmul(centr_T_mesh, palm_pose_hom)
            palm_poses_rot_mat[idx] = palm_pose_centr[:3,:3]
            palm_poses_transl[idx] = palm_pose_centr[:3,-1]
        grasps = {'rot_matrix':palm_poses_rot_mat, 'transl': palm_poses_transl, 'joint_conf': joint_confs}
        return grasps

    def __getitem__(self, idx):
        """ Batch contains: N random different object bps, each one successful grasp
        Dataset size = total_num_successful_grasps * N_bps_per_object, e.g. 15.000 * 50 = 750k
        Should fetch one bps for an object + a single grasp for that object.
        Returns a dict with palm_position, palm_orientation, finger_configuration and bps encoding of the object.
        """
        bps_path = self.bps_paths[idx]
        # Load the bps encoding
        base_path, bps_name = os.path.split(bps_path)
        obj_name = '_'.join(bps_name.split('_bps')[:-1])

        # Read the corresponding transform between mesh_frame and object_centroid
        centr_T_mesh = self.read_pcd_transform(bps_path)

        bps_path = bps_path.replace('multi','single')
        bps_obj = np.load(bps_path)

        # Read in a grasp for a given object (in mesh frame)
        if self.cfg['DATASETS']['POSITIVE_ONLY']:
            palm_pose, joint_conf, _ = self.grasp_data_handler.get_single_successful_grasp(obj_name,
                                                                                    random=True)
        elif self.cfg['DATASETS']['NEGATIVE_ONLY']:
            palm_pose, joint_conf, world_T_mesh = self.grasp_data_handler.get_single_grasp_of_outcome(
                obj_name, outcome='negative', random=True)

        palm_pose_hom = utils.hom_matrix_from_pos_quat_list(palm_pose)

        # Transform grasp from mesh frame to object centroid
        palm_pose_centr = np.matmul(centr_T_mesh, palm_pose_hom)

        # Before reducing joint conf
        if self.is_debug:
            j = joint_conf
            diffs = np.abs([j[3] - j[2], j[7] - j[6], j[11] - j[10], j[15] - j[14]])
            print(diffs[diffs > 0.09])

        # Turn the full 20 DoF into 15 DoF as every 4th joint is coupled with the third
        joint_conf = utils.reduce_joint_conf(joint_conf)

        # Extract rotmat and transl
        palm_rot_matrix = palm_pose_centr[:3, :3]
        palm_transl = palm_pose_centr[:3, 3]

        # Test restored grasp
        if self.is_debug:
            print(joint_conf)
            print(palm_transl)
            visualization.show_dataloader_grasp(bps_path, obj_name, centr_T_mesh, palm_pose_hom,
                                                palm_pose_centr, self.gazebo_obj_path)

            # Visualize full hand config
            visualization.show_grasp_and_object(bps_path, palm_pose_centr, joint_conf)

        alpha, beta, gamma = transforms3d.euler.mat2euler(palm_rot_matrix)

        # Normalize angles [-pi, pi], [-pi/2,pi/2], [-pi, pi] to [0,1]
        alpha = (alpha + np.pi) / 2 / np.pi
        beta = (beta + np.pi) / 2 / np.pi
        gamma = (gamma + np.pi) / 2 / np.pi

        # Normalize transl if you use positional encoding
        # [ 0.20864879992399918, -0.21115427946708953]
        # [ 0.13591848442440144, -0.3150945039775345 ]
        # [ 0.2628828995958964, -0.1773658852579449]
        # [ 0.2628828995958964, -0.3150945039775345]
        palm_transl_min = -0.3150945039775345
        palm_transl_max = 0.2628828995958964

        palm_transl = (palm_transl - palm_transl_min) / (palm_transl_max - palm_transl_min)

        data_out = {'rot_matrix': palm_rot_matrix,\
                    'angle_vector': np.array([alpha, beta, gamma]),\
                    'transl': palm_transl,\
                    'joint_conf': joint_conf,\
                    'bps_object': bps_obj}

        # If we want to evaluate, also return the pcd path to load from for vis
        # if self.cfg["ds_name"] == 'eval':
        data_out['pcd_path'] = bps_path.replace('bps', 'pcd').replace('npy', 'pcd')
        data_out['obj_name'] = obj_name

        # affpose paper
        # return data_dict['shape_id'], data_dict['semantic class'], data_dict['coordinate'], data_dict['affordance'], \
        #     data_dict['affordance_label'], data_dict['rotation'], data_dict['translation']

        return data_out

    def __len__(self):
        #len of dataset is number of bps per object x num_success_grasps
        return len(self.bps_paths)


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.split(os.path.split(path)[0])[0]

    path = os.path.join(BASE_PATH, "ffhflow/configs/prohmr.yaml")
    cfg = get_config(path)
    gds = FFHGeneratorDataset(cfg)

    # while True:
    #     i = np.random.randint(0, gds.__len__())
    #     gds.__getitem__(i)

    # save angle vector npy
    print(len(gds))
    transl_vectors = np.zeros((len(gds),3))
    for i in range(len(gds)):
        # i = np.random.randint(0, gds.__len__())
        data = gds.__getitem__(i)
        transl_vectors[i] = data['transl']

    np.save('transl_vectors.npy',transl_vectors)