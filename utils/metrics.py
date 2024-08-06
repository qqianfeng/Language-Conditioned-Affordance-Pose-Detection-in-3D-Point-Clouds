import numpy as np
import torch
from utils import utils

def geodesic_distance_rotmats_pairwise_tf(r1s, r2s):
    """TensorFlow version of `geodesic_distance_rotmats_pairwise_np`."""
    # These are the traces of R1^T R2
    trace = torch.einsum('aij,bij->ab', r1s, r2s)
    return torch.acos(torch.clip_by_value((trace - 1.0) / 2.0, -1.0, 1.0))


def geodesic_distance_rotmats_pairwise_np(r1s, r2s):
    """Computes pairwise geodesic distances between two sets of rotation matrices.

    Args:
      r1s: [N, 3, 3] numpy array
      r2s: [M, 3, 3] numpy array

    Returns:
      [N, M] angular distances.
    """
    rot_rot_transpose = np.einsum('aij,bkj->abik', r1s, r2s, optimize=True) #[N,M,3,3]
    tr = np.trace(rot_rot_transpose, axis1=-2, axis2=-1) #[N,M]
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def euclidean_distance_points_pairwise_np(pt1, pt2,L1=False):
    """_summary_

    Args:
        pt1 (_type_): [N, 3] numpy array, predicted grasp translation
        pts (_type_): [M, 3] numpy array, ground truth grasp translation

    Returns:
        dist_mat _type_: [N,M]
    """
    dist_mat = np.zeros((pt1.shape[0],pt2.shape[0]))
    for idx in range(pt1.shape[0]):
        deltas = pt2 - pt1[idx]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        if not L1:
            dist_mat[idx] = dist_2
        else:
            dist_mat[idx] = np.sqrt(dist_2)
    return dist_mat


def euclidean_distance_joint_conf_pairwise_np(joint1, joint2,L1=False):
    dist_mat = np.zeros((joint1.shape[0],joint2.shape[0]))
    for idx in range(joint1.shape[0]):
        deltas = joint2 - joint1[idx]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        if not L1:
            dist_mat[idx] = dist_2
        else:
            dist_mat[idx] = np.sqrt(dist_2)
    return dist_mat


def maad_for_grasp_distribution(grasp1, grasp2,L1=False):
    """For each grasp in pt1 set, we find the closest L2 distance to any grasp in pt2 set.

    Args:
        grasp1 (dict): predicted grasp set
        grasp2 (dict): ground truth grasp set

    Returns:
        _type_: _description_
    """

    # Convert tensor to numpy if needed
    if torch.is_tensor(grasp1['rot_matrix']):
        grasp1['rot_matrix'] = grasp1['rot_matrix'].cpu().data.numpy()
        grasp1['transl'] = grasp1['transl'].cpu().data.numpy()
        grasp1['pred_joint_conf'] = grasp1['pred_joint_conf'].cpu().data.numpy()

    # calculate distance matrix
    transl_dist_mat = euclidean_distance_points_pairwise_np(grasp1['transl'], grasp2['transl'],L1)
    rot_dist_mat = geodesic_distance_rotmats_pairwise_np(grasp1['rot_matrix'], grasp2['rot_matrix'])

    # Adapt format of joint conf from 15 dim to 20 dim and numpy array
    grasp2_joint_conf = np.zeros((len(grasp2['joint_conf']),20))
    for idx in range(len(grasp2['joint_conf'])):
        grasp2_joint_conf[idx] = grasp2['joint_conf'][idx]
    pred_joint_conf_full = np.zeros((grasp1['pred_joint_conf'].shape[0], 20))
    for idx in range(grasp1['pred_joint_conf'].shape[0]):
        pred_joint_conf_full[idx] = utils.full_joint_conf_from_vae_joint_conf(grasp1['pred_joint_conf'][idx])
    grasp1['pred_joint_conf'] = pred_joint_conf_full

    joint_dist_mat = euclidean_distance_joint_conf_pairwise_np(grasp1['pred_joint_conf'], grasp2_joint_conf,L1)

    transl_loss = np.min(transl_dist_mat, axis=1)  # [N,1]
    rot_loss = np.zeros_like(transl_loss)
    joint_loss = np.zeros_like(transl_loss)

    cor_grasp_idxs = []
    # find corresponding grasp according to transl dist and add the rot/joint loss
    for idx in range(transl_loss.shape[0]):
        cor_grasp_idx = np.argmin(transl_dist_mat[idx])
        cor_grasp_idxs.append(cor_grasp_idx)
        rot_loss[idx] = rot_dist_mat[idx, cor_grasp_idx]
        joint_loss[idx] = joint_dist_mat[idx, cor_grasp_idx]

    # Calculate coverage. How many grasps are found in grasp2 set.
    unique_cor_grasp_idxs = sorted(set(cor_grasp_idxs), key=cor_grasp_idxs.index)
    coverage = len(unique_cor_grasp_idxs) / len(grasp2['transl'])

    return np.sum(transl_loss), np.sum(rot_loss), np.sum(joint_loss), coverage


def maad_for_grasp_distribution_reversed(grasp1, grasp2):
    """_summary_

    Args:
        grasp1 (dict): predicted grasp set
        grasp2 (dict): ground truth grasp set

    Returns:
        _type_: _description_
    """

    # Convert tensor to numpy if needed
    if torch.is_tensor(grasp1['rot_matrix']):
        grasp1['rot_matrix'] = grasp1['rot_matrix'].cpu().data.numpy()
        grasp1['transl'] = grasp1['transl'].cpu().data.numpy()
        grasp1['pred_joint_conf'] = grasp1['pred_joint_conf'].cpu().data.numpy()

    # calculate distance matrix
    transl_dist_mat = euclidean_distance_points_pairwise_np(grasp2['transl'], grasp1['transl'])
    rot_dist_mat = geodesic_distance_rotmats_pairwise_np(grasp2['rot_matrix'], grasp1['rot_matrix'])

    # Adapt format of joint conf from 15 dim to 20 dim and numpy array
    grasp2_joint_conf = np.zeros((len(grasp2['joint_conf']),20))
    for idx in range(len(grasp2['joint_conf'])):
        grasp2_joint_conf[idx] = grasp2['joint_conf'][idx]
    pred_joint_conf_full = np.zeros((grasp1['pred_joint_conf'].shape[0], 20))
    for idx in range(grasp1['pred_joint_conf'].shape[0]):
        pred_joint_conf_full[idx] = utils.full_joint_conf_from_vae_joint_conf(grasp1['pred_joint_conf'][idx])
    grasp1['pred_joint_conf'] = pred_joint_conf_full

    joint_dist_mat = euclidean_distance_joint_conf_pairwise_np(grasp2_joint_conf, grasp1['pred_joint_conf'])

    transl_loss = np.min(transl_dist_mat, axis=1)  # [N,1]
    rot_loss = np.zeros_like(transl_loss)
    joint_loss = np.zeros_like(transl_loss)

    cor_grasp_idxs = []
    # find corresponding grasp according to transl dist and add the rot/joint loss
    for idx in range(transl_loss.shape[0]):
        cor_grasp_idx = np.argmin(transl_dist_mat[idx])
        cor_grasp_idxs.append(cor_grasp_idx)
        rot_loss[idx] = rot_dist_mat[idx, cor_grasp_idx]
        joint_loss[idx] = joint_dist_mat[idx, cor_grasp_idx]

    # Calculate coverage. How many grasps are found in grasp2 set.
    unique_cor_grasp_idxs = sorted(set(cor_grasp_idxs), key=cor_grasp_idxs.index)
    coverage = len(unique_cor_grasp_idxs) / len(grasp2['transl'])

    return np.sum(transl_loss), np.sum(rot_loss), np.sum(joint_loss), coverage


if __name__ == "__main__":
    a = np.zeros((1,3,3))
    b = np.ones((1,3,3))
    geodesic_distance_rotmats_pairwise_np(a,b)
