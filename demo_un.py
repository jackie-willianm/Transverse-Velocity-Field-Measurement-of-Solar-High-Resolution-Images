import sys

# from core.unet.ablation_experiments.extra_loss.raft import FRAFT
# from core.unet.ablation_experiments.biflow.raft import FRAFT
# from core.unet.ablation_experiments.feature_layer.raft import FRAFT
# from core.unet.ablation_experiments.feature_layer.raft import FRAFTLite as FRAFT
# from core.unet.ablation_experiments.add_resnet.raft import FRAFT
# from core.unet.ablation_experiments.add_resnet.raft import FRAFTLite as FRAFT
# from core.unet.ablation_experiments.context_network.raft import FRAFTLite as FRAFT
# from core.unet.ablation_experiments.multi_output.raft import FRAFT
# from core.unet.ablation_experiments.multi_output.raft import FRAFTLite as FRAFT
# from core.unet.ablation_experiments.extra_loss_lite.raft import FRAFT
# from core.unet.ablation_experiments.ul_lite04.raft import FRAFT
# from core.unet.ablation_experiments.ul_lite04.res_inception.raft import FRAFT
# from core.unet.improve_res_net.module04.raft import FRAFT
from core.unet.ablation_experiments.biflow_ar.raft import FRAFT

# from core.dalunwen.feature_layer.pwcnet import PWCNETH
# from core.dalunwen.residual_block.pwcnet import PWCNETH
# from core.dalunwen.multi_output.pwcnet import PWCNETH
# from core.dalunwen.context_network.pwcnet import PWCNETH

from core.utils import flow_viz
from core.utils.utils import InputPadder
from utils.cal_dif import imgs_dif
from utils.saveWrapedImage import saveImage

sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# ------
from utils.evaluationIndicators.warp import flow_warp
from utils.evaluationIndicators.utils import ssim, cc, rv_rm
from core.utils.vectorArrowViz import arrow_viz

# import scipy.io as scio
# import torch.nn.functional as F
# ------


DEVICE = 'cuda'


def flow_mean(flo):
    flo_mag = torch.sum(flo ** 2, dim=1).sqrt()
    flo_mag = flo_mag[:, 50:250, 10:210]
    flow_m = torch.mean(flo_mag)
    return flow_m


def plt_save(img, name):
    name = name.split("\\")[-1]
    if isinstance(img, torch.Tensor):
        img = np.mean(img.squeeze(0).cpu().numpy(), axis=0).astype(np.float32)
    fig = plt.figure(1, facecolor='white', dpi=200)
    ax = plt.axes()
    plt.axis('off')
    plt.imshow(img, cmap="Greys_r")
    f = plt.gcf()
    rgb_path = "F:/User_Folders/20222104112CL/biflow_un/saveWarpped/residualImage/un/" + name
    f.savefig(rgb_path)
    f.clear()
    fig.clear()


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
    img_name = filepath.split("\\")[-1]
    img_path = "F:/User_Folders/20222104112CL/biflow_un/saveWarpped/flow_viz/un/" + img_name
    # img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)  # [532, 580, 3]
    # flo_flct = flo_flct[50:250, 10:210, :]
    flo = flo[70:120, 20:70, :]
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
    # model = torch.nn.DataParallel(RAFT(args))
    # model = torch.nn.DataParallel(TRAFT(args))
    # model = torch.nn.DataParallel(MRAFT(args))
    model = torch.nn.DataParallel(FRAFT(args))
    # model = torch.nn.DataParallel(PWCNETH(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()  # 不启用BatchNormalization和Dropout,pytorch框架会自动把BN和D固定住
    # --------------------------

    # -------------------------
    total_res_mean = 0
    total_res_variance = 0
    # 用于统计循环次数，便于统计度量均值大小
    metrics_count = 0
    total_sou_mean = 0
    total_sou_variance = 0
    count = 0

    # total_res_mean_noc = 0
    # total_res_variance_noc = 0

    total_res_cc, res_cc = 0, 0
    total_sou_cc, sou_cc = 0, 0

    total_res_ssim, res_ssim = 0, 0
    total_sou_ssim, sou_ssim = 0, 0
    flow_u, flow_v = 0.0, 0.0
    flow_ustd, flow_vstd = 0.0, 0.0
    # ------------------------

    with torch.no_grad():
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg'))
        # images = sorted(images)
        # for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1s = glob.glob(os.path.join(args.path, '1', '*.png')) + \
                  glob.glob(os.path.join(args.path, '1', '*.jpg'))
        image2s = glob.glob(os.path.join(args.path, '2', '*.png')) + \
                  glob.glob(os.path.join(args.path, '2', '*.jpg'))
        for imfile1, imfile2 in zip(image1s, image2s):
            metrics_count += 1  # 本行后添加
            if args.datatest:
                count += 1
                if count % 2 == 1:
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)
                else:
                    continue
            else:
                # for i in range(len(images) // 2):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)  # 对图像进行填充，使得尺寸能够被8整除
            image1, image2 = padder.pad(image1, image2)  # image1.size=[1,3,552,600] torch.float32类型 value=[0,255]

            # flow_low, flow_up, occu_mask = model(image1, image2, iters=12, test_mode=True, add_bw=args.add_bw_flow, small_bw=args.small_bw)  # flow_up[1,2,552,600]
            flow_low, flow_up, occu_mask = model(image1, image2, iters=12, test_mode=True)  # flow_up[1,2,552,600]
            # -----------------------------------
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL) 在这种情况下，用cv2.namedWindow()函数可以指定窗口是否可以调整大小。
            # 在默认情况下，标志为cv2.WINDOW_AUTOSIZE。但是，如果指定标志为cv2.WINDOW_Normal，则可以调整窗口的大小。
            # 计算未被遮挡的像素数量 和 根据计算出的pre_flow进行扭曲的图像warp_img
            viz(flow_up[:, :, 10:-10, 10:-10], imfile1)
            # flo_mean = flow_mean(flow_up[:, :, 10:-10, 10:-10])
            # print("flo_mean = ", flo_mean)
            # occ_sum = occu_mask.sum()
            warp_img = flow_warp(image1, flow_up)
            # plt_save(warp_img[:, :, 60:260, 20:220], name="warp")
            # plt_save(image1[:, :, 60:260, 20:220], name="image1")
            # plt_save(image2[:, :, 60:260, 20:220], name="image2")

            saveImage(imfile1, warp_img, "F:/User_Folders/20222104112CL/biflow_un/saveWarpped/warpImage/un/")  # imfile1
            # print(metrics_count, "  occu_mask_sum = ", occ_sum)
            # occu_mask1 = ((occu_mask > 0.5).float().squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            # saveImage(imfile1, occu_mask1, "Occulation_map")
            # cv2.imshow('wrap_image', occu_mask1 / 255.0)  # 因为此时的wrap_img类型为float32
            # cv2.waitKey()
            # arrow_viz(flow_up, image1, save_path="F:/User_Folders/20222104112CL/biflow_un/saveWarpped/arrow/un/", imgname=imfile1, interval=1)
            img_dif = imgs_dif(warp_img, image2)
            plt_save(img_dif[60:260, 20:220], name=imfile1)
            # saveImage(imfile1, img_dif, "residualImage/ours", 'residual-')
            # res_mean, res_variance = calmetrics(img_dif)
            # sou_mean, sou_variance = calsource(image1, image2)

            # 新添加的度量 cc（相关系数）

            # # 对optical数据集2
            # image1 = image1[:, :, 2:-2, 2:-2]
            # image2 = image2[:, :, 2:-2, 2:-2]
            # warp_img = warp_img[:, :, 2:-2, 2:-2]

            sou_variance, sou_mean = rv_rm(image1, image2, 10)
            res_variance, res_mean = rv_rm(warp_img, image2, 10)
            sou_ssim = ssim(image1, image2, 10)
            res_ssim = ssim(warp_img, image2, 10)
            sou_cc = cc(image1, image2, 10)
            res_cc = cc(warp_img, image2, 10)
            print(metrics_count, "  ", "source_ssim = ", sou_ssim, "\tresidual_ssim = ", res_ssim)
            print(metrics_count, "  ", "source_cc = ", sou_cc, "\tresidual_cc = ", res_cc)
            print(metrics_count, "  ", "source_mean = ", sou_mean, "\tsource_variance = ", sou_variance)
            print(metrics_count, "  ", "residual_mean = ", res_mean, "\tresidual_variance = ", res_variance)
            total_sou_cc += sou_cc
            total_sou_ssim += sou_ssim
            total_sou_mean += sou_mean
            total_sou_variance += sou_variance
            total_res_mean += res_mean
            total_res_variance += res_variance
            total_res_cc += res_cc
            total_res_ssim += res_ssim

            flow_u += torch.mean(flow_up[0, 0, 10:-10, 10:-10])
            flow_v += torch.mean(flow_up[0, 1, 10:-10, 10:-10])
            # flow_ustd += torch.std(flow_up[0, 0, 10:-10, 10:-10])
            # flow_vstd += torch.std(flow_up[0, 1, 10:-10, 10:-10])

            # if args.small_bw or args.add_bw_flow:
            #     img_dif_noc = imgs_dif(warp_img * occu_mask, image2 * occu_mask)
            #     res_mean_noc, res_variance_noc = calmetrics(img_dif_noc)
            #     total_res_mean_noc += res_mean_noc
            #     total_res_variance_noc += res_variance_noc
            #     print(metrics_count, "  ", "residual_mean_noc = ", res_mean_noc, "\tresidual_variance_noc = ", res_variance_noc)
            # cv2.imshow('Difference between images', img_dif * 255)  # 因为img_dif类型为int32，在imshow中会除以256，再映射到[0,255]
            # -----------------------------------
            # viz(image1, flow_up)
            # flo_flct = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # flo_flct = flow_viz.flow_to_image(flo_flct)
            # cv2.imshow('image', flo_flct / 255.0)
            # cv2.waitKey()
    if args.datatest:
        count = (count + 1) / 2
        print(count)  # 49
        total_sou_mean = total_sou_mean / count
        total_sou_variance = total_sou_variance / count
        total_res_mean = total_res_mean / count
        total_res_variance = total_res_variance / count
        total_sou_cc = total_sou_cc / count
        total_res_cc = total_res_cc / count
        total_sou_ssim = total_sou_ssim / count
        total_res_ssim = total_res_ssim / count
        # total_res_mean_noc = total_res_mean_noc / count
        # total_res_variance_noc = total_res_variance_noc / count
    else:
        total_sou_mean = total_sou_mean / metrics_count
        total_sou_variance = total_sou_variance / metrics_count
        total_res_mean = total_res_mean / metrics_count
        total_res_variance = total_res_variance / metrics_count
        total_sou_cc = total_sou_cc / metrics_count
        total_res_cc = total_res_cc / metrics_count
        total_sou_ssim = total_sou_ssim / metrics_count
        total_res_ssim = total_res_ssim / metrics_count
        # total_res_mean_noc = total_res_mean_noc / metrics_count
        # total_res_variance_noc = total_res_variance_noc / metrics_count
    print("total_sou_ssim = ", total_sou_ssim, "\ttotal_res_ssim = ", total_res_ssim)
    print("total_sou_cc = ", total_sou_cc, "\ttotal_res_cc = ", total_res_cc)
    print("total_sou_mean = ", total_sou_mean, "\ttotal_sou_variance = ", total_sou_variance)
    print("total_res_mean = ", total_res_mean, "\ttotal_res_variance = ", total_res_variance)
    print("total_flow_u_mean = ", (flow_u / len(image1s)), "\ttotal_flow_v_mean = ", (flow_v / len(image1s)))
    print("total_flow_ustd_mean = ", (flow_ustd / len(image1s)), "\ttotal_flow_vstd_mean = ",
          (flow_vstd / len(image1s)))
    # if args.small_bw or args.add_bw_flow:
    #     print("total_res_mean = ", total_res_mean_noc, "\ttotal_res_variance_noc = ", total_res_variance_noc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
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
