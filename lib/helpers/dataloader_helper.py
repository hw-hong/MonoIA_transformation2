import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def paired_collate_fn(batch):
    # batch = [(sample_A_0, sample_B_0), (sample_A_1, sample_B_1), ...]
    samples_A = [item[0] for item in batch]
    samples_B = [item[1] for item in batch]
    return default_collate(samples_A), default_collate(samples_B)


def build_dataloader(cfg, workers=4):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg, data_augmentation=True)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg, data_augmentation=False)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    use_consistency_loss = cfg.get('use_consistency_loss', False)

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=False,
                              collate_fn=paired_collate_fn if use_consistency_loss else None)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, test_loader
