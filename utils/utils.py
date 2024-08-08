import numpy as np
import torch
import torch.nn.functional as F
import random

# newly imported
import transforms3d as tf
import torch

# newly imported
HAND_CFG = {
    'Right_Index_0': 0.2,
    'Right_Index_1': 0.2,
    'Right_Index_2': 0.2,
    'Right_Index_3': 0.2,
    'Right_Little_0': 0.2,
    'Right_Little_1': 0.2,
    'Right_Little_2': 0.2,
    'Right_Little_3': 0.2,
    'Right_Middle_0': 0.2,
    'Right_Middle_1': 0.2,
    'Right_Middle_2': 0.2,
    'Right_Middle_3': 0.2,
    'Right_Ring_0': 0.2,
    'Right_Ring_1': 0.2,
    'Right_Ring_2': 0.2,
    'Right_Ring_3': 0.2,
    'Right_Thumb_0': 0.2,
    'Right_Thumb_1': 0.2,
    'Right_Thumb_2': 0.2,
    'Right_Thumb_3': 0.2,
}

def quat_xyzw2wxyz(quat):
    """   tf transform defines quaternion as xyzw
    transforms3d defines quaternion as wxyz
    so we have to convert quaternion into right form for transforms3d package.
    """
    quat = np.insert(quat, 0, quat[3])
    quat = np.delete(quat, -1)
    return quat

def reduce_joint_conf(jc_full):
    """Turn the 20 DoF input joint array into 15 DoF by either dropping each 3rd or 4th joint value, depending on which is smaller.

    Args:
        jc_full (np array): 20 dimensional array of hand joint values

    Returns:
        jc_red (np array): 15 dimensional array of reduced hand joint values
    """
    idx = 0
    jc_red = np.zeros((15, ))
    for i, _ in enumerate(jc_red):
        if (i + 1) % 3 == 0:
            if jc_full[idx + 1] > jc_full[idx]:
                jc_red[i] = jc_full[idx + 1]
            else:
                jc_red[i] = jc_full[idx]
            idx += 2
        else:
            jc_red[i] = jc_full[idx]
            idx += 1
    return jc_red

def hom_matrix_from_pos_quat_list(rot_quat_list):
    """Get quaternion from ros tf in format of xyzw.

    Args:
        rot_quat_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    p = rot_quat_list[:3]
    q = rot_quat_list[3:]
    q = quat_xyzw2wxyz(q)
    rot = tf.quaternions.quat2mat(q)
    T = np.eye(4, 4)
    T[:3, :3] = rot
    T[:3, 3] = p
    return T



def full_joint_conf_from_vae_joint_conf(vae_joint_conf):
    """Takes in the 15 dimensional joint conf output from VAE and repeats the 3*N-th dimension to turn dim 15 into dim 20.

    Args:
        vae_joint_conf (np array): dim(vae_joint_conf.position) = 15

    Returns:
        full_joint_conf (JointState): Full joint state with dim(full_joint_conf.position) = 20
    """
    # for split to run we have to have even joint dim so 15->16
    if vae_joint_conf.shape[0] == 16:
        vae_joint_conf = vae_joint_conf[:15]
    full_joint_pos = np.zeros(20)
    ix_full_joint_pos = 0
    for i in range(vae_joint_conf.shape[0]):
        if (i + 1) % 3 == 0:
            full_joint_pos[ix_full_joint_pos] = vae_joint_conf[i]
            full_joint_pos[ix_full_joint_pos + 1] = vae_joint_conf[i]
            ix_full_joint_pos += 2
        else:
            full_joint_pos[ix_full_joint_pos] = vae_joint_conf[i]
            ix_full_joint_pos += 1

    return full_joint_pos


def convert_output_to_grasp_mat(samples, return_arr=True):
    """_summary_

    Args:
        samples (dict): pred_angles, pred_pose_transl can be of two types.
        One is after positional encoding, one is original format (mat or vec)
        but ['rot_matrix'] and ['transl'] must be mat or vec for same interface.
        return_arr (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    num_samples = samples['pred_angles'].shape[0]
    pred_rot_matrix = np.zeros((num_samples,3,3))
    pred_transl_all = np.zeros((num_samples,3))

    for idx in range(num_samples):
        pred_angles = samples['pred_angles'][idx].cpu().data.numpy()
        # rescale rotation prediction back
        pred_angles = pred_angles * 2 * np.pi - np.pi
        pred_angles[pred_angles < -np.pi] += 2 * np.pi

        alpha, beta, gamma = pred_angles
        mat = tf.euler.euler2mat(alpha, beta, gamma)
        pred_rot_matrix[idx] = mat

        # rescale transl prediction back

        palm_transl_min = -0.3150945039775345
        palm_transl_max = 0.2628828995958964
        pred_transl = samples['pred_pose_transl'][idx].cpu().data.numpy()
        value_range = palm_transl_max - palm_transl_min
        pred_transl = pred_transl * (palm_transl_max - palm_transl_min) + palm_transl_min

        pred_transl[pred_transl < -value_range / 2] += value_range
        pred_transl[pred_transl > value_range / 2] -= value_range
        pred_transl_all[idx] = pred_transl

    if return_arr:
        samples['rot_matrix'] = pred_rot_matrix
        samples['transl'] = pred_transl_all
        samples['joint_conf'] = samples['pred_joint_conf'].cpu().data.numpy()

    else:
        samples['rot_matrix'] = torch.from_numpy(pred_rot_matrix).cuda()
        samples['transl'] = torch.from_numpy(pred_transl_all).cuda()
        samples['joint_conf'] = samples['pred_joint_conf']

    return samples

def hom_matrix_from_transl_rot_matrix(transl, rot_matrix):
    """Transform rot_matrix and transl vector into 4x4 homogenous transform.

    Args:
        transl (array): Translation array 3x1
        rot_matrix (array): Rotation matrix 3x3

    Returns:
        hom_matrix (array): 4x4 homogenous transform.
    """
    hom_matrix = np.eye(4)
    hom_matrix[:3, :3] = rot_matrix
    hom_matrix[:3, 3] = transl
    return hom_matrix

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


class PN2_Scheduler(object):
    def __init__(self, init_lr, step, decay_rate, min_lr):
        super().__init__()
        self.init_lr = init_lr
        self.step = step
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        return

    def __call__(self, epoch):
        factor = self.decay_rate**(epoch//self.step)
        if self.init_lr*factor < self.min_lr:
            factor = self.min_lr / self.init_lr
        return factor


class PN2_BNMomentum(object):
    def __init__(self, origin_m, m_decay, step):
        super().__init__()
        self.origin_m = origin_m
        self.m_decay = m_decay
        self.step = step
        return

    def __call__(self, m, epoch):
        momentum = self.origin_m * (self.m_decay**(epoch//self.step))
        if momentum < 0.01:
            momentum = 0.01
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
        return


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)