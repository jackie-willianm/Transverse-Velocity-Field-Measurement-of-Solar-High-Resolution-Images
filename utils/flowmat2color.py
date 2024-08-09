# ------
import sys
sys.path.append('core')
import torch
import cv2
from core.utils import flow_viz
from wrap_image import Wrap # 自己添加的，但改行本身就是空格行
from comp_dif import imgs_dif
import scipy.io as scio
from core.utils import flow_viz

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# # ------
#
# # --------------------------
# dataFile = 'flow.mat'
# dataf = scio.loadmat(dataFile)
# flow_msy = dataf['flow']
# flow_msy = torch.from_numpy(flow_msy)
# flow_msy = flow_msy.unsqueeze(0)
# print(flow_msy.shape)
# h, w = flow_msy.shape[-2:]
# # print(h, w)
# pad_h = (((h // 8) + 1) * 8 - h) % 8  # 这两段的目的就是获取能被n整除，尺寸需要填充的像素数
# pad_w = (((w // 8) + 1) * 8 - w) % 8
# # print(pad_h, pad_w)
# pad = [pad_w // 2, pad_w - pad_w // 2, 0, pad_h]
# # print(pad)
# flow_msy = F.pad(flow_msy, pad, mode='replicate')
# # print(flow_msy.shape)
# flow_visual = flow_msy[0].permute(1, 2, 0).cpu().numpy()
# flow_visual = flow_viz.flow_to_image(flow_visual)
# # cv2.imshow('flow', flow_visual / 255.0)
# cv2.imshow('flow', flow_visual)
# cv2.waitKey()
# # -------------------------


DEVICE = 'cuda'


def viz(flo, filepath):
    # 自己定义的存储图像光流的文件夹路径和名字
    # filepath 是读取的图像的路径，借用该路径获取图像名，
    # 暂时限死flo_viz的存储地址为saveWrapped/flo_viz
    img_name = filepath.split("\\")[-1]
    img_path = "saveWrapped/flo_viz/ours/" + img_name
    # img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)  # [532, 580, 3]
    # flo_flct = flo_flct[50:250, 10:210, :]
    # flo_flct = flo_flct[70:120, 20:70, :]
    fig = plt.figure(1, facecolor='white', dpi=200)
    plt.axis('off')
    plt.imshow(flo)
    f = plt.gcf()
    f.savefig(img_path)
    f.clear()
    fig.clear()
    # cv2.imwrite(img_path, flo_flct)
    # img_flo = np.concatenate([img, flo_flct], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def plt_save(img, name):
    if (isinstance(img, torch.Tensor)):
        img = np.mean(img.squeeze(0).cpu().numpy(), axis=0).astype(np.float32)
    fig = plt.figure(1, facecolor='white', dpi=200)
    ax = plt.axes()
    plt.axis('off')
    plt.imshow(img, cmap="Greys_r")
    f = plt.gcf()
    rgb_path = "datasets/" + name + ".jpg"
    f.savefig(rgb_path)
    f.clear()
    fig.clear()


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_flow(flow_flow):
    flow = frame_utils.read_gen(flow_flow)
    flow = np.array(flow).astype(np.float32)
    flow = torch.from_numpy(flow).permute(2, 0, 1).float()
    return flow[None].to(DEVICE)


def demo(args):

    with torch.no_grad():
        image1s = glob.glob(os.path.join(args.path1, '*.png')) + \
                 glob.glob(os.path.join(args.path1, '*.jpg'))
        image2s = glob.glob(os.path.join(args.path2, '*.png')) + \
                 glob.glob(os.path.join(args.path2, '*.jpg'))
        flows = glob.glob(os.path.join(args.flow_path, '*.flo_flct'))
        image1s = sorted(image1s)
        image2s = sorted(image2s)
        flows = sorted(flows)
        for imfile1, imfile2, flowfile in zip(image1s, image2s, flows):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            flow = load_flow(flowfile)

            viz(flow[:, :, 10:-10, 10:-10], imfile1)
            warp_img = flow_warp(image1, flow)
            saveImage(imfile1, warp_img, "saveWrapped/ours")  # imfile1
            arrow_viz(flow, image1, save_path="saveWrapped/arrow/ours", imgname=imfile1, interval=2)
            img_dif = imgs_dif(warp_img, image2)
            plt_save(img_dif[60:260, 20:220], name="img_dif")
            # saveImage(imfile1, img_dif, "residualImage/ours", 'residual-')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', help="frame1 for evaluation")
    parser.add_argument('--path2', help="frame2 for evaluation")
    args = parser.parse_args()
    demo(args)
