import numpy as np
import torch
import torch.nn.functional as F
import random

# newly imported
import transforms3d as tf

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