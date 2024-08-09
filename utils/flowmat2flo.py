import numpy as np
import glob
import os
import scipy.io as scio
import tkinter as tk
from tkinter import filedialog


TAG_CHAR = np.array([202021.25], np.float32)


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def mat2flo(matpath, flopath):
    # 函数输入mat文件路径和转换后输出的flo文件路径
    count = 0
    mats = glob.glob(os.path.join(matpath, '*.mat'))
    mats = sorted(mats)
    for matfile in mats:
        mat = scio.loadmat(matfile)
        # 从字典格式中提取数据，成为ndarray数据
        uv_temp = mat['uv']
        h, w = uv_temp.shape[0], uv_temp.shape[1]
        # print("uv_temp.shape = ",uv_temp.shape)  uv_temp.shape =  (540, 590, 2)
        # uv = np.zeros([uv_temp.shape[0], uv_temp.shape[1], 2])
        uv = np.zeros([h, w, 2])
        # print("uv.shape = ", uv.shape)
        # 已将matlab中生成的mat顺序改变，此处不再需要互换位置
        # uv[:,:,0] = uv_temp[10:-10, 10:-10, 1]
        # uv[:,:,1] = uv_temp[10:-10, 10:-10, 0]
        uv[:, :, 0] = uv_temp[:, :, 0]
        uv[:, :, 1] = uv_temp[:, :, 1]
        # 获取mat文件的名字，设置需要保存flo文件的路径
        matname = (matfile.split("\\")[-1]).split(".")[0]
        fpath = flopath + "\\" + matname + ".flo_flct"
        # writeFlow(fpath, uv_temp)
        writeFlow(fpath, uv)
        count = count + 1
    print("一共将 ", count, " 个mat数据转换成flo数据。")


# 实例化tkinter
root = tk.Tk()
root.withdraw()
matpath = filedialog.askdirectory()  # askdirectory() 获取选择的文件夹   askopenfilename() 获取选择的文件
flopath = filedialog.askdirectory()  # 这种方式获取的绝对路径是用 "/"连接的
mat2flo(matpath, flopath)
