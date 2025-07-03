import os
import glob
import random
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

'''

'''
class SRDataset(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.scale = config.scale  # Default scale for testing
        self.patch_size = config.patch_size if train else None
        self.hr_files = sorted(glob.glob(os.path.join(config.train_dir, 'DIV2K_train_HR/*.png')))

    def __getitem__(self, idx):
        hr = imageio.imread(self.hr_files[idx])
        hr = torch.from_numpy(hr).permute(2, 0, 1).float()  # Shape: [C, H, W]

        if self.train:
            # Randomly select scale (e.g., 2x, 3x, 4x)
            scale = random.choice([2, 3, 4])
            # Generate LR by downscaling HR
            lr = TF.resize(
                hr, 
                (hr.shape[1] // scale, hr.shape[2] // scale), 
                interpolation=TF.InterpolationMode.BICUBIC
            )
            # Random crop to patch_size (if training)
            ih, iw = lr.shape[1], lr.shape[2]
            ix = random.randrange(0, iw - self.patch_size + 1)
            iy = random.randrange(0, ih - self.patch_size + 1)
            lr_patch = lr[:, iy:iy+self.patch_size, ix:ix+self.patch_size]
            hr_patch = hr[:, 
                iy*scale : (iy+self.patch_size)*scale,
                ix*scale : (ix+self.patch_size)*scale
            ]
            return lr_patch, hr_patch
        else:
            # For testing, use fixed scale
            lr = TF.resize(
                hr, 
                (hr.shape[1] // self.scale, hr.shape[2] // self.scale),
                interpolation=TF.InterpolationMode.BICUBIC
            )
            return lr, hr

    def __init__(self, config, train=True):
        self.train = train
        self.scale = config.scale  # Default scale for testing
        self.patch_size = config.patch_size if train else None
        self.hr_files = sorted(glob.glob(os.path.join(config.train_dir, 'DIV2K_train_HR/*.png')))
        # Remove fixed LR directory; generate LR on-the-fly

    def __getitem__(self, idx):
        hr = imageio.imread(self.hr_files[idx])
        hr = torch.from_numpy(hr).permute(2, 0, 1).float()  # Shape: [C, H, W]

        if self.train:
            # Randomly select scale (e.g., 2x, 3x, 4x)
            scale = random.choice([2, 3, 4])
            # Generate LR by downscaling HR
            lr = TF.resize(
                hr, 
                (hr.shape[1] // scale, hr.shape[2] // scale), 
                interpolation=TF.InterpolationMode.BICUBIC
            )
            # Random crop to patch_size (if training)
            ih, iw = lr.shape[1], lr.shape[2]
            ix = random.randrange(0, iw - self.patch_size + 1)
            iy = random.randrange(0, ih - self.patch_size + 1)
            lr_patch = lr[:, iy:iy+self.patch_size, ix:ix+self.patch_size]
            hr_patch = hr[:, 
                iy*scale : (iy+self.patch_size)*scale,
                ix*scale : (ix+self.patch_size)*scale
            ]
            return lr_patch, hr_patch
        else:
            # For testing, use fixed scale
            lr = TF.resize(
                hr, 
                (hr.shape[1] // self.scale, hr.shape[2] // self.scale),
                interpolation=TF.InterpolationMode.BICUBIC
            )
            return lr, hr