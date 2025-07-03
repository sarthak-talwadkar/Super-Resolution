## Sarthak Uday Talwadkar

import os
import glob
import random
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

'''
    Dataset class for Super-Resolution training and testing
    Handles DIV2K dataset loading with proper HR/LR pairing
    Implements patch extraction and data augmentation for training
'''
class SRDataset(Dataset):
    """
        Args:
            config: Configuration object with parameters
            train: Boolean flag for training/test mode
    """
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
        
        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, '*.png')))
        self.lr_files = sorted(glob.glob(os.path.join(self.lr_dir, '*.png')))

    def __len__(self):
        return len(self.hr_files)
    

    def __getitem__(self, id):

        hr = imageio.imread(self.hr_files[id])
        lr = imageio.imread(self.lr_files[id])

        hr = torch.from_numpy(hr).permute(2, 0, 1).float()  
        lr = torch.from_numpy(lr).permute(2, 0, 1).float()

        if self.train:

            ih, iw = lr.shape[-2:]
            ix = random.randrange(0, iw - self.patch_size +1)
            iy = random.randrange(0, ih - self.patch_size +1)

            lr_patch = lr[:,
            iy:iy + self.patch_size, 
            ix:ix + self.patch_size
            ]
            hr_patch = hr[:,
            iy * self.scale:(iy + self.patch_size)*self.scale,
            ix * self.scale:(ix + self.patch_size)*self.scale]

            if random.random() > 0.5:
                lr_patch = TF.hflip(lr_patch)
                hr_patch = TF.hflip(hr_patch)

            if random.random() > 0.5:
                lr_patch = TF.vflip(lr_patch)
                hr_patch = TF.vflip(hr_patch)
            
            return lr_patch, hr_patch
        
        return lr, hr