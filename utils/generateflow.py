import numpy as np
import torch
from wrap_image import Wrap
import cv2
from PIL import Image
import torch.nn.functional as F

img_path = 'solar/0001.jpg'

# 构建随机流
xx = np.random.normal(loc=-0.006, scale=0.8941, size=(552, 600))
yy = np.random.normal(loc=0.029, scale=1.0035, size=(552, 600))
flow = np.zeros([2, 552, 600]) # shape = (2,552,600)
# print(flow.shape)
flow[0] = xx
flow[1] = yy
flow = torch.from_numpy(flow)
flow = flow.unsqueeze(0)

# 加载对应路径图像，进行格式转换
image = np.array(Image.open(img_path)).astype(np.uint8)
image2 = image
image = torch.from_numpy(image).permute(2, 0, 1).float()
image = image.unsqueeze(0)

# 获取图像尺寸，今天padding（填充）
h, w = image.shape[-2:]
pad_h = (((h // 8) + 1) * 8 - h) % 8  # 这两段的目的就是获取能被n整除，尺寸需要填充的像素数
pad_w = (((w // 8) + 1) * 8 - w) % 8
pad = [pad_w // 2, pad_w - pad_w // 2, 0, pad_h]
image = F.pad(image, pad, mode='replicate')

# 根据输入的image和flow来扭曲图像
warp_image = Wrap(image, flow)
cv2.imshow("original image", image2)
cv2.imshow("warp_image", warp_image)
cv2.waitKey()




