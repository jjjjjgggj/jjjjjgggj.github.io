import cv2
import os
import re
import numpy as np
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
#from tensorboardX import SummaryWriter
import os
import numpy as np
import pdb

def normalize_image(image):
    # 归一化图像像素值到 [0, 1]
    return image.astype(np.float32) / 255.0

def super_resolution(image_path, output_dir):
    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    print(f"Image shape: {image.shape}, Image dtype: {image.dtype}")

    # 归一化图像像素值到 [0, 1]
    image = normalize_image(image)

    # 进行4倍图像超分辨率
    # 使用bicubic插值方法
    upscaled_image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    print(f"Upscaled Image shape: {upscaled_image.shape}")
    # 保存超分辨率图像
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, (upscaled_image * 255.0).astype(np.uint8))

# 示例用法
if __name__ == "__main__":
    # 定义输入图像路径和输出目录
    input_directory = '/home/dell/storage/JIANGUOJUN/lwtdm-sr/Dataset/MRI_TEST_28_112'
    output_directory = '/home/dell/storage/JIANGUOJUN/lwtdm-sr/Dataset/bicubic'

    val_set = Data.create_dataset2('/home/dell/storage/JIANGUOJUN/lwtdm-sr/Dataset/MRI_TEST_28_112', 'val')
    val_loader = Data.create_dataloader(
        val_set, '/home/dell/storage/JIANGUOJUN/lwtdm-sr/Dataset/MRI_TEST_28_112', 'val')

    # 创建输出目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    super_resolution(val_loader,output_directory)