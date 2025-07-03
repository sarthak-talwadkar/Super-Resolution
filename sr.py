## Sarthak Uday Talwadkar

import os 
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageFont

from modelDef.edsr import EDSR
from data_loader_unknown import SRUnknownDataset
from config import config

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def test(checkpoint_path, output_dir = 'EDSR/result/sr'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EDSR(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    model.eval()

    os.makedirs(output_dir, exist_ok = True)

    test_set = SRUnknownDataset(config, train = False)
    test_loader = DataLoader(test_set, batch_size = 1)

    metrics = []
    font = ImageFont.load_default()

    with torch.no_grad():
        for idx, lr in enumerate(test_loader):
            lr = lr.to(device)
            sr = model(lr)
            print("LR Input Range:", lr.min().item(), lr.max().item())  # Should be 0-255
            print("SR Output Range:", sr.min().item(), sr.max().item())  # Should be ~0-255


            lr_np = lr.squeeze().cpu().numpy().transpose(1, 2, 0)
            sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)

            lr_np = np.clip(lr_np, 0, 255).astype(np.uint8)
            sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)

            sr_pil = Image.fromarray(sr_np)
            sr_pil.save(os.path.join(output_dir, f'sr{idx:04d}.png'))

            lr_pil = Image.fromarray(lr_np)
            bicubic_pil = lr_pil.resize(
                (lr_pil.width*config.scale, lr_pil.height*config.scale),
                Image.Resampling.BICUBIC
            )
            bicubic_np = np.array(bicubic_pil)

            bicubic_pil.save(os.path.join(output_dir, f'bicubic{idx:04d}.png'))

            comparison = np.concatenate([bicubic_np, sr_np], axis=1)
            comparison_pil = Image.fromarray(comparison)
            
            comparison_pil.save(os.path.join(output_dir, f'compare{idx:04d}.png'))

            

if __name__ == '__main__':
    test('/home/sunny/Downloads/edsr_epochs_4_200.pth')
