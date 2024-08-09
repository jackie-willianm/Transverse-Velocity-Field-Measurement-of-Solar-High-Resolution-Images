import glob
import os
import scipy.io as scio
import numpy as np
from PIL import Image
import torch
from wrap_image import Wrap
import cv2


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))  # 此处img1[..., None]相当于增加了一个维度
    else:
        img = img[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]



def wrapbyuv(matpath, soupath, wrappath):
    # 函数输入mat文件路径和转换后输出的flo文件路径
    count = 0
    mats = glob.glob(os.path.join(matpath, '*.mat'))
    mats = sorted(mats)
    sous = glob.glob(os.path.join(soupath, '*.jpg')) + \
        glob.glob(os.path.join(soupath, '*.png'))
    sous = sorted(sous)


    for matfile, soufile in zip(mats, sous[:-1]):
        mat = scio.loadmat(matfile)
        sou = load_image(soufile)
        # 从字典格式中提取数据，成为ndarray数据
        uv = mat['uv']
        #  从3维升至4维，方便Wrap函数处理
        uv = torch.from_numpy(uv).permute(2, 0, 1).float()
        uv = uv.unsqueeze(0)
        # 获取mat文件的名字，设置需要保存flo文件的路径
        matname = (matfile.split("\\")[-1]).split(".")[0]
        proname = matname[:-8]
        numname = matname[-8:-3]
        aftname = matname[-3:]
        numname = np.int32(numname)
        numname = numname + 2
        numname_str = str(numname)
        wei = list(numname_str)
        neednum = [-2, -4]
        for i in neednum:
            if (i == -2 and wei[i] == '6'):
                numname += 40
                wei = list(str(numname))
            if (i == -4 and wei[i] == '6'):
                numname += 4000
        numname = str(numname)
        matname = proname + numname + aftname
        # if ((numname+2)%100) >= 60:
        #     numname = numname + 2 - 60 + 100
        #     if ((numname%1000)//100) == 9:

        wpath = wrappath + "\\" + matname + ".jpg"
        wrap_img = Wrap(sou, uv)
        wrap_img = wrap_img.astype(np.int32)
        cv2.imwrite(wpath, wrap_img)
        count = count + 1
        print("已经完成 ", count, " 个文件的转换")

    print("一共将 ", count, " 个mat数据转换成wrap.jpg数据。")


matpath = 'flow_jpg'
soupath = 'filament'
wrappath = 'saveWrapped\\Solar'
wrapbyuv(matpath, soupath, wrappath)
