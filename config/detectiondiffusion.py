import os
import torch
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = 'detectiondiffusion_bps'
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

scheduler = dict(
    type='lr_lambda',
    lr_lambda=PN2_Scheduler(init_lr=0.001, step=20,
                            decay_rate=0.5, min_lr=1e-5)
)

optimizer = dict(
    type='adam',
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5,
)

model = dict(
    type='detectiondiffusion',
    device=torch.device('cuda'),
    background_text='none',
    betas=[1e-4, 0.02],
    n_T=1000,
    drop_prob=0.1,
    weights_init='default_init',
)

training_cfg = dict(
    num_worker=10,
    model=model,
    batch_size=32,
    epoch=200,
    gpu='0',
    workflow=dict(
        train=1,
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),
)

data = dict(
    data_dir="/home/qian.feng/data/language-conditioned-affordance-pose-detection",
)