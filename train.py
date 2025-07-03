## Sarthak Uday Talwadkar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import config
from modelDef.edsr import EDSR
from data_loader import SRDataset

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

'''
    Trains an EDSR super-resolution model using L1 loss and Adam optimization.
    Loads training data (LR-HR pairs) in batches
    For each epoch:
        Generates super-resolved (SR) images from LR inputs
        Calculates reconstruction error between SR and HR
        Updates model weights via backpropagation
    Saves model checkpoints every 50 epochs
'''
def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EDSR(config).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr = config.lr)

    train_set = SRDataset(config, train = True)
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        for lr, hr in train_loader:
            
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

        print(f'Epoch[{epoch + 1}/{config.epochs}], Avg Loss: {avg_loss:.4f}')
        if(epoch +1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'edsr_epochs_{epoch + 1}.pth')


if __name__ == '__main__':
    train()