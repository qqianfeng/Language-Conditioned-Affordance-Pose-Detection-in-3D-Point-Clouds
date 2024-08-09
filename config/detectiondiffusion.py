import os
import torch
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = 'detectDif_lr1e-6_bz32_nT100_dp00'
seed = 1
log_dir = opj("./log/", exp_name)
try:
    os.makedirs(log_dir)
except:
    print('Logging Dir is already existed!')

dataset = dict(
  PATH="/data/net/userstore/qf/hithand_data/data/ffhnet-data", #/data/hdd1/qf/hithand_data/ffhnet-data
  GAZEBO_OBJ_PATH= "/data/net/userstore/qf/hithand_data/data/gazebo-objects/objects_gazebo", # /home/yb/Projects/gazebo-objects/objects_gazebo/
  GRASP_DATA_NANE= "grasp_data_all.h5",
  POSITIVE_ONLY= True,
  NEGATIVE_ONLY= False,
  POS_AND_NEG_GRASP= False,
  use_bps=True,
)
# TODO: change init_lr here according to optimizier lr
scheduler = dict(
    type='lr_lambda',
    lr_lambda=PN2_Scheduler(init_lr=1e-5, step=20,
                            decay_rate=0.5, min_lr=1e-7)
)

optimizer = dict(
    type='adam',
    lr=1e-6,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5, #1e-5, 1e-4 is in paper
)

model = dict(
    type='detectiondiffusion',
    device=torch.device('cuda'),
    grasp_dim=21,
    scale_down = [6,4,2], #[20,12,4],
    background_text='none',
    betas=[1e-4, 0.02],
    n_T=100,
    drop_prob=0.0, #0.1, 0.05 is in paper
    weights_init='default_init',
)

training_cfg = dict(
    num_worker=10,
    model=model,
    batch_size=32,
    epoch=50,
    gpu='0',
    workflow=dict(
        train=1,
        val=1,
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),
)

data = dict(
    data_dir="/home/qian.feng/data/language-conditioned-affordance-pose-detection",
)