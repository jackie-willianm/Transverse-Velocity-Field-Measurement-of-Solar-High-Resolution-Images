import math

import numpy as np
from scipy import interpolate

def Wrap_Rewrite(img, flo, mode = 0):
    # img:tensor张量，维度为4维
    # flo为光流，此处也是4维输入
    # mode表示插值选择，暂时只设置默认0，为双线性插值
    img = img.squeeze(0)  # 降维
    flo = flo.squeeze(0)
    img = img.cpu().numpy()
    flo = flo.cpu().numpy()
    if len(img.shape) == 3:
        img1 = np.mean(img, axis=0)  # 以第一维取均值，即转化为灰度图像
    else:
        img1 = img
    # print(img1.shape)
    h, w = img1.shape
    dy, dx = flo[0], flo[1]
    y0, x0 = np.meshgrid(np.arange(w), np.arange(h))

    x1 = x0 + dx
    y1 = y0 + dy

    # 线性插值中所有的邻居像素
    xBas0 = np.floor(x1)
    yBas0 = np.floor(y1)
    xBas1 = xBas0 + 1
    yBas1 = yBas0 + 1

    # 线性插值常数（百分比）
    xCom = x1 - xBas0
    yCom = y1 - yBas0
    perc0 = (1 - xCom) * (1 - yCom)
    perc1 = (1 - xCom) * yCom
    perc2 = xCom * (1 - yCom)
    perc3 = xCom * yCom

    # 将索引限制为边界
    check_xBas0 = (xBas0 < 0) | (xBas0 > h - 1)
    check_yBas0 = (yBas0 < 0) | (yBas0 > w - 1)
    xBas0[check_xBas0] = 0
    yBas0[check_yBas0] = 0
    check_xBas1 = (xBas1 < 0) | (xBas1 > h - 1)
    check_yBas1 = (yBas1 < 0) | (yBas1 > w - 1)
    xBas1[check_xBas1] = 0
    yBas1[check_yBas1] = 0

    # 输出参数为wrap_img
    wrap_img = np.zeros((h, w), dtype=np.float32)
    # 获取图像对应像素强度
    test00 = (xBas0 + yBas0 * h).astype(np.int32)
    intensity_xy0 = img1[xBas0.astype(np.int32), yBas0.astype(np.int32)]
    intensity_xy1 = img1[xBas0.astype(np.int32), yBas1.astype(np.int32)]
    intensity_xy2 = img1[xBas1.astype(np.int32), yBas0.astype(np.int32)]
    intensity_xy3 = img1[xBas1.astype(np.int32), yBas1.astype(np.int32)]
    # 采用双线性插值方式
    # img_mean = img1.mean()
    if mode == 0:
        intensity_xy0[check_xBas0 | check_yBas0] = 0
        intensity_xy1[check_xBas0 | check_yBas0] = 0
        intensity_xy2[check_xBas1 | check_yBas0] = 0
        intensity_xy3[check_xBas1 | check_yBas1] = 0
    wrap_img = intensity_xy0 * perc0 + intensity_xy1 * perc1 \
        + intensity_xy2 * perc2 + intensity_xy3 * perc3

    # x1 = x1.reshape(-1)
    # y1 = y1.reshape(-1)
    # img1 = img1.reshape(-1)
    #
    # valid = (x1 > 0) & (x1 < w) & (y1 > 0) & (y1 < h)
    # valid = np.array(valid).astype(np.float)
    # print(valid.shape)
    # x1 = x1[valid]
    # valid_x = x1 * valid
    # y1 = y1[valid]
    # valid_y = y1 * valid
    # print(x1.shape)
    # print(valid_x.shape)
    # print(valid_y.shape)
    # print(img1.shape)

    # wrap_img = interpolate.griddata(  # dtype = float64
    #     (x1, y1), img1, (x0, y0), method='linear', fill_value=0)

    # print(wrap_img)
    # print(wrap_img.shape) (552, 600)
    # wrap_img[invalid] = 0
    # wrap_img = (255 * wrap_img).astype(np.int32)
    wrap_img = wrap_img.astype(np.float32)
    # print(wrap_img.dtype)

    return wrap_img

