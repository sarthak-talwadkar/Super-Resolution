## Sarthak Uday Talwadkar

import os 
import imageio
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from modelDef.edsr import EDSR
from data_loader import SRDataset
from config import config

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

'''
    Evaluates a trained EDSR super-resolution model 
    by comparing it with bicubic upscaling using image quality metrics and visual comparisons.
    Saves the bicubic, SR image and comparing all three image to the output folder
    Saves all the image metric to a CSV file in the output folder
'''

def test(checkpoint_path, output_dir = '.'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EDSR(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs(output_dir, exist_ok = True)

    test_set = SRDataset(config, train = False)
    test_loader = DataLoader(test_set, batch_size = 1)

    metrics = []
    font = ImageFont.load_default()

    with torch.no_grad():
        for idx, (lr, hr) in enumerate(test_loader):
            lr = lr.to(device)
            sr = model(lr)
            print("LR Input Range:", lr.min().item(), lr.max().item())  # Should be 0-255
            print("HR Input Range:", hr.min().item(), hr.max().item())  # Should be 0-255
            print("SR Output Range:", sr.min().item(), sr.max().item())  # Should be ~0-255


            lr_np = lr.squeeze().cpu().numpy().transpose(1, 2, 0)
            hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
            sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)

            lr_np = np.clip(lr_np, 0, 255).astype(np.uint8)
            hr_np = np.clip(hr_np, 0, 255).astype(np.uint8)
            sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)

            sr_pil = Image.fromarray(sr_np)
            sr_pil.save(os.path.join(output_dir, f'sr{idx:04d}.png'))

            lr_pil = Image.fromarray(lr_np)
            bicubic_pil = lr_pil.resize(
                (hr_np.shape[1], hr_np.shape[0]),  
                Image.Resampling.BICUBIC  
            )
            bicubic_np = np.array(bicubic_pil)

            bicubic_pil.save(os.path.join(output_dir, f'bicubic{idx:04d}.png'))

            bicubic_psnr = psnr(hr_np, bicubic_np, data_range=255)
            bicubic_ssim = ssim(hr_np, bicubic_np, 
                               win_size=7, channel_axis=-1, data_range=255)
            
            sr_psnr = psnr(hr_np, sr_np, data_range=255)
            sr_ssim = ssim(hr_np, sr_np, 
                          win_size=7, channel_axis=-1, data_range=255)
            
            metrics.append({
                'image_id': idx,
                'bicubic_psnr': bicubic_psnr,
                'bicubic_ssim': bicubic_ssim,
                'sr_psnr': sr_psnr,
                'sr_ssim': sr_ssim
            })

            comparison = np.concatenate([hr_np, bicubic_np, sr_np], axis=1)
            comparison_pil = Image.fromarray(comparison)

            draw = ImageDraw.Draw(comparison_pil)
            text_y = comparison.shape[0] - 20
            draw.text((10, text_y), 
                     f"Original HR", fill=(255,255,255), font=font)
            draw.text((hr_np.shape[1]+10, text_y), 
                     f"Bicubic: {bicubic_psnr:.2f}dB / {bicubic_ssim:.3f}", 
                     fill=(255,255,255), font=font)
            draw.text((hr_np.shape[1]*2+10, text_y), 
                     f"SR: {sr_psnr:.2f}dB / {sr_ssim:.3f}", 
                     fill=(255,255,255), font=font)
            
            comparison_pil.save(os.path.join(output_dir, f'compare_{idx:04d}.png'))

            

            

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print(f"Saved results to {output_dir}")

    # Print summary
    print(f"\nAverage Metrics:")
    print(f"Bicubic PSNR: {df.bicubic_psnr.mean():.2f} dB")
    print(f"Bicubic SSIM: {df.bicubic_ssim.mean():.4f}")
    print(f"SR PSNR: {df.sr_psnr.mean():.2f} dB")
    print(f"SR SSIM: {df.sr_ssim.mean():.4f}")

## Path to the train model
if __name__ == '__main__':
    test('/home/sunny/Downloads/edsr_epochs_150.pth')