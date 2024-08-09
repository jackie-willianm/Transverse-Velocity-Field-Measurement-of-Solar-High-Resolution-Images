import torch
import numpy as np
from PIL import Image
from scipy import interpolate


def Wrap(img, flo):
    # img:tensor张量，维度为4维
    img = img.squeeze(0)  # 降维
    flo = flo.squeeze(0)
    img = img.cpu().numpy()
    flo = flo.cpu().numpy()
    if len(img.shape) == 3:
        img1 = np.mean(img, axis=0) #以第一维取均值，即转化为灰度图像
    else:
        img1 = img
    # print(img1.shape)
    h, w = img1.shape  # (552, 600)
    dy, dx = flo[0], flo[1]  # flo_flct[0]是水平方向，flo_flct[1]是竖直方向
    y0, x0 = np.meshgrid(np.arange(w), np.arange(h))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    img1 = img1.reshape(-1)

    valid = (x1 >= 0) & (x1 < h) & (y1 >= 0) & (y1 < w)
    # valid = np.array(valid).astype(np.float)
    # print(valid.shape)
    # x1 = x1[valid]
    valid_x = x1 * valid
    # y1 = y1[valid]
    valid_y = y1 * valid
    # print(x1.shape)
    # print(valid_x.shape)
    # print(valid_y.shape)
    # print(img1.shape)

    wrap_img = interpolate.griddata(  # dtype = float64
        (x1, y1), img1, (x0, y0), method='linear', fill_value=0)

    # print(wrap_img)
    # print(wrap_img.shape) (552, 600)
    # wrap_img[invalid] = 0
    # wrap_img = (255 * wrap_img).astype(np.int32)
    wrap_img = wrap_img.astype(np.float32)
    # print(wrap_img.dtype)

    return wrap_img

