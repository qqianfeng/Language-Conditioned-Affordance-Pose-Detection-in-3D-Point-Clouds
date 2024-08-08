import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, MultiStepLR
from dataset import FFHGeneratorDataset
from models import *
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

# Pools of models, optimizers, weights initialization methods, schedulers
model_pool = {
    'detectiondiffusion': DetectionDiffusion,
}

optimizer_pool = {
    'sgd': SGD,
    'adam': Adam
}

init_pool = {
    'default_init': weights_init
}

scheduler_pool = {
    'step': StepLR,
    'cos': CosineAnnealingLR,
    'lr_lambda': LambdaLR,
    'multi_step': MultiStepLR
}


def build_model(cfg):
    """_summary_
    Function to build the model before training
    """
    if hasattr(cfg, 'model'):
        model_info = cfg.model
        weights_init = model_info.get('weights_init', None)
        background_text = model_info.get('background_text', 'none')
        device = model_info.get('device', torch.device('cuda'))
        model_name = model_info.type
        model_cls = model_pool[model_name]
        if model_name in ['detectiondiffusion']:
            betas = model_info.get('betas', [1e-4, 0.02])
            n_T = model_info.get('n_T', 1000)
            drop_prob = model_info.get('drop_prob', 0.1)
            model = model_cls(cfg.model, betas, n_T, device, background_text, drop_prob,cfg.dataset.use_bps)
        else:
            raise ValueError("The model name does not exist!")
        if weights_init != None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        return model
    else:
        raise ValueError("Configuration does not have model config!")


# def build_dataset(cfg):
#     """_summary_
#     Function to build the dataset
#     """
#     if hasattr(cfg, 'data'):
#         data_info = cfg.data
#         data_dir = data_info.data_dir
#         ffh_datamodule = FFHDataModule(cfg)
#         train_set = ffh_datamodule.train_dataloader
#         test_set = ffh_datamodule.val_dataloader
#         dataset_dict = dict(
#             train_set=train_set,
#             test_set=test_set
#         )
#         return dataset_dict
#     else:
#         raise ValueError("Configuration does not have data config!")


def build_loader(cfg):
    """_summary_
    Function to build the loader
    """
    dset_gen = FFHGeneratorDataset(cfg,eval=False)
    train_loader = torch.utils.data.DataLoader(dset_gen,
                                                batch_size=cfg.training_cfg.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=cfg.training_cfg.num_worker)

    loader_dict = dict(
        train_loader=train_loader,
    )

    return loader_dict


def build_optimizer(cfg, model):
    """_summary_
    Function to build the optimizer
    """
    optimizer_info = cfg.optimizer
    optimizer_type = optimizer_info.type
    optimizer_info.pop('type')
    optimizer_cls = optimizer_pool[optimizer_type]
    optimizer = optimizer_cls(model.parameters(), **optimizer_info)
    scheduler_info = cfg.scheduler
    if scheduler_info:
        scheduler_name = scheduler_info.type
        scheduler_info.pop('type')
        scheduler_cls = scheduler_pool[scheduler_name]
        scheduler = scheduler_cls(optimizer, **scheduler_info)
    else:
        scheduler = None
    optim_dict = dict(
        scheduler=scheduler,
        optimizer=optimizer
    )
    return optim_dict
