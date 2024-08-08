import os
from os.path import join as opj
from gorilla.config import Config
from utils import *
import argparse
import torch
import shutil


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", default="./config/detectiondiffusion.py", help="train config file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    file_name = os.path.basename(args.config)
    shutil.copyfile(args.config, os.path.join(cfg.log_dir,file_name))

    logger = IOStream(opj(cfg.log_dir, 'run.log'))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))      # number of GPUs to use
    logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))
    if cfg.get('seed') != None:     # set random seed
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
    model = build_model(cfg).cuda()     # build the model from configuration

    print("Training from scratch!")

    # dataset_dict = build_dataset(cfg)       # build the dataset
    loader_dict = build_loader(cfg) #, dataset_dict)       # build the loader
    optim_dict = build_optimizer(cfg, model)        # build the optimizer

    # construct the training process
    training = dict(
        model=model,
        # dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        optim_dict=optim_dict,
        logger=logger
    )

    task_trainer = Trainer(cfg, training)
    continue_train = False
    ckpt_path = 'log/detectiondiffusion_bs32/current_model.t7'
    if continue_train:
        task_trainer.load_checkpoint(ckpt_path)
    task_trainer.run()
