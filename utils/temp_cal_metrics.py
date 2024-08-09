import sys

from core.utils import flow_viz

sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# ------
from utils.evaluationIndicators.utils import ssim, cc, rv_rm

# import scipy.io as scio
# import torch.nn.functional as F
# ------


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = np.array(Image.open(imfile)).astype(np.int16)
    # img = (img + 32768).astype(np.int32)
    # img = ((img + 32768).astype(np.float32) / 256.0).astype(np.uint8)
    # img = ((img * 255.0) // 32767).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(flo, filepath):
    # 自己定义的存储图像光流的文件夹路径和名字
    # filepath 是读取的图像的路径，借用该路径获取图像名，
    # 暂时限死flo_viz的存储地址为saveWrapped/flo_viz
    img_name = filepath.split("/")[-1]
    img_path = "saveWrapped/flo_viz/ours/" + img_name
    # img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)  # [532, 580, 3]
    flo = flo[50:250, 10:210, :]
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


def demo(args):
    # --------------------------

    # -------------------------
    total_res_mean = 0
    total_res_variance = 0
    # 用于统计循环次数，便于统计度量均值大小
    metrics_count = 0
    count = 0

    total_res_cc, res_cc = 0, 0

    total_res_ssim, res_ssim = 0, 0
    # ------------------------

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            metrics_count = metrics_count + 1
            # Demons 均为540 590  Ours为[532 580] [540 590]
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            # image1 = image1[:, :, 4:-4, 5:-5]  # 是Demons则启用该行代码，是Ours则不启用
            image2 = image2[:, :, 4:-4, 5:-5]
            image1 = image1[:, :, 50:250, 10:210]
            image2 = image2[:, :, 50:250, 10:210]

            # 新添加的度量 cc（相关系数）
            res_variance, res_mean = rv_rm(image1, image2)
            res_ssim = ssim(image1, image2)
            res_cc = cc(image1, image2)
            print(metrics_count, "  residual_ssim = ", res_ssim)
            print(metrics_count, "  residual_cc = ", res_cc)
            print(metrics_count, "  residual_mean = ", res_mean)
            print(metrics_count, "  residual_variance = ", res_variance)
            total_res_mean += res_mean
            total_res_variance += res_variance
            total_res_cc += res_cc
            total_res_ssim += res_ssim

    if args.datatest:
        count = (count + 1) / 2
        print(count)  # 49
        total_res_mean = total_res_mean / count
        total_res_variance = total_res_variance / count
        total_res_cc = total_res_cc / count
        total_res_ssim = total_res_ssim / count
    else:
        total_res_mean = total_res_mean / metrics_count
        total_res_variance = total_res_variance / metrics_count
        total_res_cc = total_res_cc / metrics_count
        total_res_ssim = total_res_ssim / metrics_count
    print("\ttotal_res_ssim = ", total_res_ssim)
    print("\ttotal_res_cc = ", total_res_cc)
    print("total_res_mean = ", total_res_mean, "\ttotal_res_variance = ", total_res_variance)
    # if args.small_bw or args.add_bw_flow:
    #     print("total_res_mean = ", total_res_mean_noc, "\ttotal_res_variance_noc = ", total_res_variance_noc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--datatest', action='store_true', help='test dataset need several scene files')
    parser.add_argument('--add_bw_flow', action='store_true', help='add backward flow')
    parser.add_argument('--small_bw', action='store_true', help='add backward flow of small flow module')
    args = parser.parse_args()
    # --model=models/raft-solar22.pth --path=filament_image --add_bw_flow --datatest
    # --model=models/raft-solar22.pth --path=datasets/Solar_demo/train --add_bw_flow
    demo(args)
