import PIL.Image
import numpy
import numpy as np
import torch
import torch.nn
from PIL import Image
import scipy.io as scio
import torch.nn.functional as F
import os
import glob
import cv2
# from d2l import torch as d2l
# import tensorflow as tf

for i in [3, 2, 1, 0]:
    print(i, "\t")

# cus_kernel = torch.zeros(9)
# cus_kernel[3] = 1
# cus_kernel = cus_kernel.reshape(1, 3, 3)
# cus_kernel = torch.repeat_interleave(cus_kernel, repeats=64, dim=0)
# print(cus_kernel)


# image1 = np.array(Image.open('zancun/0003.jpg')).astype(np.uint8)
# image1[100:200, 100:150] = 0
# image2 = image1
# imgpath = 'zancun/0002.jpg'
# cv2.imwrite(imgpath, image2)


'''设置路径代码'''
# par_dir = os.path.dirname(os.path.abspath(__file__))
# print(par_dir)
# os.chdir(par_dir)
# print(os.getcwd())
#
# split='training'
# root='datasets/Solar_single_channel'
# flow_root1 = glob.glob(os.path.join(os.getcwd(), root, split))
# flow_root2 = glob.glob(os.path.join(root, split))
# flow_root3 = glob.glob(os.path.join(root, split, 'flow'))
# flow_root4 = glob.glob(os.path.join(root, split, 'flow', '*.flo_flct'))
# # cpath = cpath + '/rootb'
# print(flow_root1)
# print(flow_root2)
# print(flow_root3)
# print(flow_root4)
# # print(cpath)

# import tkinter as tk
# from tkinter import filedialog
# root = tk.Tk()
# root.withdraw()
# soupath = filedialog.askdirectory()  # askdirectory() 获取选择的文件夹   askopenfilename() 获取选择的文件
#
# # TAG_CHAR = np.array([202021.25], np.float32)
#
#
# sous = glob.glob(os.path.join(soupath, '*.jpg')) + \
#         glob.glob(os.path.join(soupath, '*.png'))
# sous = sorted(sous)


# DEVICE = 'cuda'
#
# # 转置卷积
# def trans_conv(X, K):
#     h, w = K.shape
#     Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             Y[i:i+h, j:j+w] += X[i, j] * K
#     return Y
#
#
# X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(trans_conv(X, K))
#
# X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
# tconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
# tconv.weight.data = K
# print(tconv(X))
#
# X = torch.arange(9.0).reshape(3, 3)
# K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# Y = d2l.corr2d(X, K)
# print(Y)


# im = Image.open('datasets/Solar/training/img/Ha_r000_20141003_072632_1B.jpg')
# im = np.array(im).astype(np.uint8)
# print(im.shape)

# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
# img = np.resize(data, (1, 12, 2))
# print(img)
# img_torch = torch.from_numpy(img).permute(2, 0, 1)
# print(img_torch)

# img = np.array([[1, 2],
#                 [3, 4]])
# print("shape = ", img.shape)
# print("img[1,1] = ", img[1, 1])
# print("img[3] = ", img[3])  # 该行会报错，python没有matlab中二维矩阵的列查找

# 测试torch.meshgrid和np.meshgrid的区别
# coords = torch.meshgrid(torch.arange(5, device=DEVICE), torch.arange(4, device=DEVICE))  # 生成5行4列
# print("coords = ", coords)
# arr3 = torch.stack(coords[::-1], dim=0)
# print("arr3 = ", arr3)
#
# meshg = np.meshgrid(np.arange(4), np.arange(5))  # 生成5行4列
# print("meshg = ", meshg)

# imgname = 'Ha_r000_20141003_072620_1B.jpg'
# no_houzhui = imgname.split(".")[0]
# houzhui = imgname.split(".")[-1]
# num = no_houzhui[-7:-3]
# prores = no_houzhui[:-7]
# print(no_houzhui)
# print(prores)
# print(num)
# print(houzhui)
# num1 = np.int32(num)
# num1 = num1 + 2
# print(num1)
# img1name = prores + str(num1) + houzhui
# print(img1name)


# 尝试获取文件名，去掉路径和后缀名
# matfile = 'H:/solar/filament/Ha_0001.mat'
# matname1 = matfile.split('/')[-1]
# print("matname1 = ", matname1)
# matname2 = matname1.split(".")[0]
# print("matname2 = ", matname2)


# ===================== 测试为什么matlab和python的均值和方差不一样====================
# img1 = np.array(Image.open('solar/0001.jpg')).astype(np.uint8)  # (550,600,3)
# img1 = img1.mean(axis=2)
# img1_mean = np.mean(img1)
#
# print("np.mean(img1).dtype", img1_mean.dtype)
#
# img2 = np.array(Image.open('solar/0002.jpg')).astype(np.uint8)  # (550,600,3)
# img2 = img2.mean(axis=2)
# img2_mean = np.mean(img2)
# print(img2.shape)
# print("np.mean(img2).dtype", img2_mean.dtype)
#
# img_dif = img2 - img1
# print("abs(img2 - img1).shape = ", img_dif.shape)
# print("abs(img2 - img1).dtype = ", img_dif.dtype)
# img_dif_mean = (sum(sum(img_dif))) / (550 * 600)
# print("np.mean(img_dif) = ", img_dif_mean)

# a.update(b) 对字典的更新操作代码验证
# result = {
#     "name": "digital",
#     "age": 50,
# }
# result1 = {
#     "name": "Image",
#     "age": 24,
#     "gender": "male",
# }
# print(result)
# result.update(result1)
# print(result)



# 验证光流中writeFlow是怎么存储二维光流数据的
# nBands = 2
# f = open("aoligei.flo_flct", 'wb')
# f.write(TAG_CHAR)
# width = 3
# height = 2
# u = np.array([1, 3, 5, 7, 9, 11])
# u = torch.from_numpy(u)
# u1 = u.view(-1)
# print((u1 < 8).float().mean().item())
# v = np.array([[2, 4, 6],
#               [8, 10, 12]])
# np.array(width).astype(np.int32).tofile(f)
# np.array(height).astype(np.int32).tofile(f)
# tmp = np.zeros((height, width*nBands))
# tmp[:,np.arange(width)*2] = u
# tmp[:,np.arange(width)*2 + 1] = v
# print(tmp)


# 验证[:,:,::-1]、[:,:,:2]、[] = n * []的效果
# valid = None
# flow = np.array([[[1001,2,-3],
#                   [4,5,6]],
#                  [[2,-3003,4],
#                   [5,-6,7]]]);
# print("flow.shape=", flow.shape)
# print("flow details:\n", flow)
# new_flow = flow[:,:,::-1] # 最后一维倒序
# print("new_flow.shape=", new_flow.shape)
# print("new_flow details:\n", new_flow)
# flow_two = flow[:,:,:2]  # 获取最后一维中前2个元素
# print("flow_two = flow[:,:,:2]=\n", flow_two)
#
# flow1 = np.array([[[1001,2],
#                   [4,5]],
#                  [[2,-3003],
#                   [5,-6001]],
#                   [[6, -100],
#                    [12, -27]]
#                   ]);
# flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
# print("flow1.shape=", flow1.shape)
# mag = torch.sum(flow1**2, dim=0).sqrt()
# print("flow1=\n", flow1)
# print("mag=\n", mag)
# valid = (mag > 0) & (mag < 1024)
# valid2 = (valid[:, None] * mag)
# print("valid2=\n", valid2)
# valid2mean = valid2.mean()
# print("valid2mean=\n", valid2mean)
# print("valid=\n", valid)
# arr = np.array([[1.0,5.5],
#                 [6.5,7.7],
#                 [8.0,2.0]])
# arr = torch.from_numpy(arr)
# res = arr * valid
# print("arr * valid = \n", res)
# flow1 = flow1.clamp(0.0, 255)
# print("flow1=\n", flow1)
# valid = (flow1[0].abs() < 1000) & (flow1[1].abs() < 1000)
# print("valid=\n", valid)
# valid = valid.float()
# print("valid.float=\n", valid)
#
# flow_list = [1, 2, 3, 4]
# flow_list = 4 * flow_list
# msy_list = []
# print(flow_list)
# msy_list += [[flow_list[0], flow_list[1]]]
# print("msy_list=\n", msy_list)


# arr = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ])
# arr1 = np.tile(arr, (2, 1))
# arr2 = np.tile(arr, (1, 2))
# print("(2, 1)的作用：", arr1)
# print("(1, 2)的作用：", arr2)

# image = Image.open('filament/1.jpg')
# image = np.array(image).astype(np.uint8)
# image3 = Image.open('solar/0001.jpg')
# image3 = np.array(image3).astype(np.uint8)
# print("image.shape=", image.shape)
# print("image3.shape=", image3.shape)
# print("image[..., None].shape=", image[..., None].shape)
# if len(image.shape) == 2:
#     print("this is a gray image, convert it to color image")
#     img1 = np.tile(image[..., None], (1, 1, 3))
# else:
#     img1 = image[..., :3]
# print("converted img1.shape=", img1.shape)
#
# img2 = img1[..., :3]
# print(img2.shape)



# 此段代码存在错误，metrics_str中，需要迭代操作，输出len（data）个指定格式数据
# total_steps = 10
# lr = 0.00234
# data = [2.250, 3.250, 4.250, 5.250]
# training_str = "[{:6d}, {:10.7f}] ".format(total_steps+1, lr)
# print(training_str)
# metrics_str = ("{:10.4f}, "*len(data)).format(*data)
# print(metrics_str)

# os.path.join方法的使用测试
# path1 = 'E:'
# path2 = 'raft'
# path3 = 'dataset'
# file_list = ['a', 'b', 'c', 'd']
# new_list = []
# path_ds = os.path.join(path1, path2, path3, '')
# path_test = os.path.join('/aa/', 'bb/', 'cc.txt')
# print("path_ds = " + path_ds)
# print("path_test = ", path_test)
# for i in range(len(file_list)-1):
#     new_list +=  [ [file_list[i], file_list[i+1]] ]
#     print(new_list)




# 指定随机数种子序号方法
# torch.manual_seed(2)
# print(torch.rand(1))
# print(torch.rand(1))

# # 测试读取matlab生成的mat文件
# dataFile = 'flow.mat'
# data = scio.loadmat(dataFile) # 原本matlab存储和python的loadmat是以字典形式存储，标签即原本参数名称
# # print(type(data['flow'])) # data['*']操作获取ndarray数据
# flow = data['flow']
# print(flow.shape)
# flow_msy = torch.from_numpy(flow)
# flow_msy = flow_msy.unsqueeze(0)
# print(flow_msy.shape)
# h, w = flow_msy.shape[-2:]
# print(h, w)
# pad_h = (((h // 8) + 1) * 8 - h) % 8 # 这两段的目的就是获取能被n整除，尺寸需要填充的像素数
# pad_w = (((w // 8) + 1) * 8 - w) % 8
# print(pad_h, pad_w)
# pad = [pad_w//2, pad_w - pad_w//2, 0, pad_h]
# print(pad)
# flow_msy = F.pad(flow_msy, pad, mode='replicate')
# print(flow_msy.shape)


# points = np.random.rand(10, 2)  # 生成[10,2]的随机矩阵
# print(points)

# def load_image(imfile):
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     print(img.shape)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     print(img.size())
#     return img[None].to(DEVICE)
#
# img = load_image('./solar/0001.jpg')
# img2 = Image.open('./solar/2.jpg')
# img3 = Image.open('./filament/1.jpg')
# img3 = np.array(img3).astype(np.uint8)
# print(img3.size)

# 如何创建meshgrid网格矩阵
# x = np.arange(0,4,1)  # np.arange(开始,结束（不含）,间隔)
# y = np.arange(0,3,1)
# xx, yy = np.meshgrid(x, y)
# print(xx)
# print(yy)
# img = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
# u = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
# v = np.array([[0,0,0,0],
#               [0,1,0,0],
#               [0,-1,0,0]])
# xx1 = xx + u
# yy1 = yy + v
# img1 = np.zeros((3,4))
# print("xx1:")
# print(xx1)
# print('yy1:')
# print(yy1)
# for j in range(0,3):
#     for i in range(0,4):
#         img1[j, i] = img[yy1[j, i], xx1[j, i]]
#
# print(img)
# print(img1)

# 使用torch创建的网格矩阵，输出coords包含xx和yy的行列矩阵
# coords = torch.meshgrid(torch.arange(3), torch.arange(4))
# print(coords[0])

# --------------------------------
# 转置的使用
# fmap = np.array([[[1,2],
#                   [1,2]],
#                  [[3,4],
#                   [3,4]]])
# fmap = torch.from_numpy(fmap)
# print(fmap.shape)
# print(fmap)
# fmap1 = fmap.transpose(1,2)
# print(fmap1)

# img = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
# print(img.shape)
# img = torch.from_numpy(img).float()
# print(img.size())
# print(img[:,None,:].to(DEVICE)) # None此处效果类似numpy.newaxis创建新轴，或者说用来对array数组进行维度扩展
#                             # 根据需要扩展的维度添加None，此处相当于img[None,:,:]

