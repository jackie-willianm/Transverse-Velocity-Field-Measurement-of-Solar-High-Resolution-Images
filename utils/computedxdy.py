import sys
sys.path.append('core')
import torch
import torch.nn as nn
from PIL import Image
import cv2
from core.utils import flow_viz
from wrap_image import Wrap # 自己添加的，但改行本身就是空格行
from comp_dif import imgs_dif
import scipy.io as scio
import torch.nn.functional as F
# 导入绘图所需的matplotlib.pyplot库
import matplotlib.pyplot as plt
# 创建随机数所需导入的库 numpy&random
import numpy as np
import random
from core.module27_improve.net_module02 import conv_one
from core.module27_improve.net_module03 import correlate

DEVICE = 'cuda'

image1 = Image.open('datasets/Solar0513/training/img/Ha_r000_20141003_072620_1B.jpg')
image2 = Image.open('datasets/Solar0513/training/img/Ha_r000_20141003_072632_1B.jpg')
# image2 = torch.zeros([3, 3, 3])
img1 = np.array(image1).astype(np.uint8)[..., :3]
img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
img1 = img1.to(DEVICE)
img2 = np.array(image2).astype(np.uint8)[..., :3]
img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
img2 = img2.to(DEVICE)
img1 = img1.contiguous()
img2 = img2.contiguous()
out_img = correlate(img1, img2)
print(out_img.shape)
# conv3_3 = nn.Conv2d(3, 3, kernel_size=3, padding=1).cuda()
# img1 = conv3_3(img1)
# conv_img1 = conv_one(img1, kernel_size=1)
# # print('img1[0, 0, 0:3, 0:3]:\n', img1[0, 0, 0:3, 0:3])
# # print('img1[0, 1, 0:3, 0:3]:\n', img1[0, 1, 0:3, 0:3])
# # print('img1[0, 2, 0:3, 0:3]:\n', img1[0, 2, 0:3, 0:3])
# print('img1[0, 0, 0:3, 0:3]:\n', img1[0, 0, 0:3, 0:3])
# print('img1[0, 1, 0:3, 0:3]:\n', img1[0, 1, 0:3, 0:3])
# print('img1[0, 2, 0:3, 0:3]:\n', img1[0, 2, 0:3, 0:3])
# print('conv_img1.shape:\n', conv_img1.shape)
# print('conv_img1[0, 0, 0:3, 0:3]:\n', conv_img1[0, 0, 0:3, 0:3])
# print('conv_img1[0, 1, 0:3, 0:3]:\n', conv_img1[0, 1, 0:3, 0:3])

# img2 = np.array(image2).astype(np.uint8)[..., :3]
# img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
# img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
# img1 = img1.to(DEVICE)
# img2 = img2.to(DEVICE)
# B, C, H, W = img2.shape
# conv1 = F.conv2d(img1, img2, bias=None, stride=1, padding=((H-1)//2, (W-1)//2))
# print(conv1.shape)
# conv2 = F.conv2d(img2, img1, bias=None, stride=1, padding=((H-1)//2, (W-1)//2))
# print(conv2.shape)

# num1 = np.array([[1, 2, 3, 4, 5],
#                  [6, 7, 8, 9, 10],
#                  [11, 12, 13, 14, 15],
#                  [16, 17, 18, 19, 20],
#                  [21, 22, 23, 24, 25]])
# num2 = np.array([[1, 1, 1, 1, 1],
#                  [2, 2, 2, 2, 2],
#                  [3, 3, 3, 3, 3],
#                  [4, 4, 4, 4, 4],
#                  [5, 5, 5, 5, 5]])
# num1 = num1.astype(np.uint8)
# num1 = torch.from_numpy(num1).unsqueeze(0).unsqueeze(0).float()
# num1 = num1.to(DEVICE)
# B1, C1, H1, W1 = num1.shape
# num2 = num2.astype(np.uint8)
# num2 = torch.from_numpy(num2).unsqueeze(0).unsqueeze(0).float()
# num2 = num2.to(DEVICE)
# B2, C2, H2, W2 = num2.shape
# print('num1:\n', num1)
# num1 = num1.view(B1, C1, H1*W1)
# print('num1.shape:\n', num1)
# kernel1 = torch.ones([1, 1, 1]).unsqueeze(0).float()
# kernel1 = kernel1.to(DEVICE)
# pad_num = 1
# num2 = F.conv2d(num2, kernel1, stride=1, padding=pad_num)
# print('num2.shape:\n', num2.shape)
# print('num2:\n', num2)
# kernel2 = torch.ones([B1, C1, 1, 1]).to(DEVICE)
# corr = []
# for i in range(H1*W1):
#     kernel = kernel2 * num1[:, :, i]
#     # 计算当前行和列
#     row = i // W1  # 行 即 H
#     col = i - row * W1  # 列 即 W
#     tensor1 = num2[:, :, row:row+3, col:col+3]
#     conv = F.conv2d(tensor1, kernel, bias=None, stride=1, padding=0)
#     corr.append(conv)
# corr = torch.cat(corr, dim=1)
# print(corr.shape)



# # 加载从matlab生成的mat文件
# dataFile = 'flow.mat'
# dataf = scio.loadmat(dataFile)
# # 从字典格式中提取数据，成为ndarray数据
# flow_msy = dataf['flow']
# flow_msy = torch.from_numpy(flow_msy)
# # 后面F.pad是对tensor张量进行处理，若张量为3维，则复制/反射填充或只对最后一维填充，4维，则对最后2维填充
# # 所以需要增加一维，变成4维张量，来确保能对最后两维图像size进行填充
# flow_msy = flow_msy.unsqueeze(0)  # 此处不增加一维，会导致调用F.pad函数报错
# # print(flow_msy.shape)
# h, w = flow_msy.shape[-2:]
# # print(h, w)
# pad_h = (((h // 8) + 1) * 8 - h) % 8  # 这两段的目的就是获取能被n整除，尺寸需要填充的像素数
# pad_w = (((w // 8) + 1) * 8 - w) % 8
# # print(pad_h, pad_w)
# pad = [pad_w // 2, pad_w - pad_w // 2, 0, pad_h]
# # print(pad)
# flow_msy = F.pad(flow_msy, pad, mode='replicate')  # 填充完后，h，w均能被8整除
# flow_msy = flow_msy.squeeze(0)  # 降维，方便后续转化成array数组进行数据分析
# # 获取flow数据x方向位移和y方向的位移数据
# flow_x = flow_msy[0].numpy()
# flow_y = flow_msy[1].numpy()
# print("flow_x shape:", flow_x.shape, "\tflow_y shape:", flow_y.shape)
# # 获取均值 x=-0.00616 y=0.02921
# print("flow_x mean:", flow_x.mean(), "\tflow_y mean:", flow_y.mean())
# # 获取运动绝对值的均值,x=0.62218 y=0.70004，可见属于小光流场景
# # 注：直接均值发现，整体运动基本趋向于0，即相反方向运动总体接近
# print("flow_x abs_mean:", abs(flow_x).mean(), "\tflow_y abs_mean:", abs(flow_y).mean())
# # 获取标准差 x=0.894116 y=1.003566
# print("flow_x 标准差:", flow_x.std(), "\tflow_y std:", flow_y.std())
# # 获取方差, x=0.41236 y=0.51793
# print("flow_x 方差:", abs(flow_x).var(), "\tflow_y var:", abs(flow_y).var())
# # 获取最大值 x=7.41030 y=8.73273
# print("flow_x max:", flow_x.max(), "\tflow_y max:", flow_y.max())
# # 获取最小值 x=-6.41792 y=-7.30284
# print("flow_x min:", flow_x.min(), "\tflow_y min:", flow_y.min())
# # 将数据转换为1维数据以方便可视化
# flow_x1 = flow_x.reshape(-1)
# flow_y1 = flow_y.reshape(-1)
# # print(flow_x1.shape) # size=331200
# # plt.figure("flow_x attribution")
# plt.subplot(121)
# n, bins, patches = plt.hist(flow_x1, range=(-7, 9), bins=500, density=1, stacked=False, facecolor='green',
#                             alpha=0.75)
# # plt.show()
#
#
# '''
# 生成正态分布随机数组
# np.random.randn(a,b) 随机生成a*b维的标准正态分布
# np.random.normal(loc, scale, size) loc:float类型，表示均值； scale:float类型，表示标准差，越大越矮胖
#                                   size:输出的shape，可以size=(c,h,w)
# np.random.standard_normal(size) # 返回指定形状的标准正态分布
#
# '''
# xx = np.random.normal(loc=-0.006, scale=0.8941, size=(552, 600))
# xx1 = xx.reshape(-1)
# # plt.figure("生成的随机正态分布")
# plt.subplot(122)
# n2, bins2, patches2 = plt.hist(flow_x1, range=(-7, 9), bins=500, density=1, stacked=False, facecolor='green',
#                             alpha=0.75)
# plt.show()
