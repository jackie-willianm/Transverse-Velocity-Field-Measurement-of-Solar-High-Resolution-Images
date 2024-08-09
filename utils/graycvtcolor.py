import cv2
import numpy as np
import glob
import os
from PIL import Image
import torch
from saveWrapedImage import saveImage


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    else:
        img = img[..., :3]
    img = img.astype(np.int32)
    return img


def save_gray2rgb(original_file, target_file):
    # 首先获取filepath文件夹下所有png和jpg图像的路径名的列表，例:filepatn/Ha_0001.jpg
    # 然后调用存储图像函数，返回图像的数量
    images = glob.glob(os.path.join(original_file, '*.png')) + \
             glob.glob(os.path.join(original_file, '*.jpg'))
    images = sorted(images)
    for imfile in images:
        image = load_image(imfile)
        saveImage(imfile, image, target_file)
    return len(images)


original_file = "filament"
target_file = "solar"
imgs_num = save_gray2rgb(original_file, target_file)
print("imgs_num = ", imgs_num)