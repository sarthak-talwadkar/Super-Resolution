## Sarthak Uday Talwadkar

import os
import glob
import random
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

"""
        Dataset for unsupervised super-resolution training/testing
        - Handles only LR images (no HR pairs)
        - Supports patch extraction and basic augmentations
        
        Args:
            config: Configuration object with parameters
            train: Boolean flag for training/test mode
"""

class SRUnknownDataset(Dataset):
    def __init__(self, config, train = True):

        self.train = train
        self.scale = config.scale
        self.patch_size = config.patch_size if train else None

        dir_type = 'train' if train else 'test'
        if dir_type == 'train':
            self.hr_dir = os.path.join(config.train_dir if train else config.test_dir, 'DIV2K_train_HR')
            self.lr_dir = os.path.join(config.train_dir if train else config.test_dir, f'DIV2K_train_LR_bicubic/X{config.scale}')
        elif dir_type == 'test':
            self.hr_dir = os.path.join(config.train_dir if train else config.test_dir, 'DIV2K_train_HR')
            self.lr_dir = os.path.join(config.train_dir if train else config.test_dir, f'DIV2K_test_LR_bicubic/X{config.scale}')
        
        self.lr_dir = os.path.join(config.train_dir if train else config.test_dir, f'LR/')
        
        self.lr_files = sorted(glob.glob(os.path.join(self.lr_dir, '*.png')))

        

    def __len__(self):
        return len(self.lr_files)
    
    """
        Returns:
            lr_patch: Low-resolution image tensor [C,H,W] (training)
            lr: Full LR image tensor [C,H,W] (testing)
    """
    def __getitem__(self, id):

        lr = imageio.imread(self.lr_files[id])

        lr = torch.from_numpy(lr).permute(2, 0, 1).float()

        if self.train:

            ih, iw = lr.shape[-2:]
            ix = random.randrange(0, iw - self.patch_size +1)
            iy = random.randrange(0, ih - self.patch_size +1)

            lr_patch = lr[:,
            iy:iy + self.patch_size, 
            ix:ix + self.patch_size
            ]
            
            if random.random() > 0.5:
                lr_patch = TF.hflip(lr_patch)

            if random.random() > 0.5:
                lr_patch = TF.vflip(lr_patch)
            
            return lr_patch
        
        
        return lr