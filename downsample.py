# Sarthak Uday Talwadkar

import os
import cv2
import numpy as np
import rawpy
from PIL import Image

'''
    Pre process the images to produce downsample and introduce noise in the images
    Uses PIL and CV to product Bicubic and Blur and noise into images
'''
def process_image(file_path):
    """Load image (RAW or standard) and return as BGR numpy array"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ('.cr2', '.nef', '.arw'):
        # Process RAW file
        with rawpy.imread(file_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR
    else:
        # Process standard image
        img = Image.open(file_path).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def generate_lr_images(input_dir, lr_bicubic_dir, lr_noise_dir, scale=4):
    os.makedirs(lr_bicubic_dir, exist_ok=True)
    os.makedirs(lr_noise_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not os.path.isfile(file_path):
            continue
        
        try:
            # Load image (RAW or standard)
            hr_img = process_image(file_path)
            h, w = hr_img.shape[:2]
            
            # Generate bicubic LR
            lr_bicubic = cv2.resize(hr_img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
            output_name = f"{os.path.splitext(filename)[0]}.png"
            cv2.imwrite(os.path.join(lr_bicubic_dir, output_name), lr_bicubic)
            
            # Generate noisy LR
            blurred = cv2.GaussianBlur(hr_img, (5, 5), 0)
            noise = np.random.normal(0, 1, blurred.shape).astype(np.uint8)
            degraded = cv2.add(blurred, noise)
            lr_degraded = cv2.resize(degraded, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(lr_noise_dir, output_name), lr_degraded)
            
        except Exception as e:
            print(f"Skipped {filename}: {str(e)}")

# Provide the directory for images
# Input image Dir, Output dir for saving the Bicubic and Noise images
# Scale for how much you want to downsample the image
generate_lr_images(
    input_dir="/home/sunny/Projects/EDSR/data/Testing/",
    lr_bicubic_dir="/home/sunny/Projects/EDSR/data/Testing/LR/Bicubic",
    lr_noise_dir="/home/sunny/Projects/EDSR/data/Testing/LR/Noise",
    scale=4
)