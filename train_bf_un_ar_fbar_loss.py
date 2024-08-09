from __future__ import print_function, division
import time
import random
import sys

# from core.unet.ablation_experiments.extra_loss.raft import FRAFT
from core.unet.ablation_experiments.biflow_ar.raft import FRAFT
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.utils.augmentor import FlowAugmentorCL, FlowAugmentorMSY_S
from evaluate import evaluate
from core import datasets
from torch.utils.tensorboard import SummaryWriter
from core.utils.warp_occ_utils import flow_warp
from core.losses.loss_blocks import loss_smooth, loss_photometric, loss_gradient, loss_gradient_oc

sys.path.append('../core')
par_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(par_dir)

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremely large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def photo_loss_function(diff, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True):
    if charbonnier_or_abs_robust:
        if if_use_occ:
            p = ((diff) ** 2 + 1e-6).pow(q)
            p = p * mask
            if averge:
                p = p.mean()
                ap = mask.mean()
            else:
                p = p.sum()
                ap = mask.sum()
            loss_mean = p / (ap * 2 + 1e-6)
        else:
            p = ((diff) ** 2 + 1e-8).pow(q)
            if averge:
                p = p.mean()
            else:
                p = p.sum()
            return p
    else:
        if if_use_occ:
            diff = (torch.abs(diff) + 0.01).pow(q)
            diff = diff * mask
            diff_sum = torch.sum(diff)
            loss_mean = diff_sum / (torch.sum(mask) * 2 + 1e-6)
        else:
            diff = (torch.abs(diff) + 0.01).pow(q)
            if averge:
                loss_mean = diff.mean()
            else:
                loss_mean = diff.sum()
    return loss_mean


def census_loss_torch(img1, img1_warp, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True,
                      max_distance=3):
    patch_size = 2 * max_distance + 1

    def _ternary_transform_torch(image):
        R, G, B = torch.split(image, 1, 1)
        intensities_torch = (0.2989 * R + 0.5870 * G + 0.1140 * B)  # * 255  # convert to gray
        # intensities = tf.image.rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c
        w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w
        weight = torch.from_numpy(w_).float()
        if image.is_cuda:
            weight = weight.cuda()
        patches_torch = torch.conv2d(input=intensities_torch, weight=weight, bias=None, stride=[1, 1],
                                     padding=[max_distance, max_distance])
        transf_torch = patches_torch - intensities_torch
        transf_norm_torch = transf_torch / torch.sqrt(0.81 + transf_torch ** 2)
        return transf_norm_torch

    def _hamming_distance_torch(t1, t2):
        dist = (t1 - t2) ** 2
        dist = torch.sum(dist / (0.1 + dist), 1, keepdim=True)
        return dist

    def create_mask_torch(tensor, paddings):
        shape = tensor.shape  # N,c, H,W
        inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
        inner_torch = torch.ones([shape[0], shape[1], inner_width, inner_height]).float()
        if tensor.is_cuda:
            inner_torch = inner_torch.cuda()
        mask2d = F.pad(inner_torch, [paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]])
        return mask2d

    img1 = _ternary_transform_torch(img1)
    img1_warp = _ternary_transform_torch(img1_warp)
    dist = _hamming_distance_torch(img1, img1_warp)
    transform_mask = create_mask_torch(mask, [[max_distance, max_distance],
                                              [max_distance, max_distance]])
    census_loss = photo_loss_function(diff=dist, mask=mask * transform_mask, q=q,
                                      charbonnier_or_abs_robust=charbonnier_or_abs_robust, if_use_occ=if_use_occ,
                                      averge=averge)
    return census_loss


def data_loss(img1, img2, flow):
    # 数据项包含：1 光度损失 2 梯度损失
    # 目前未考虑遮挡
    warped_img = flow_warp(img1, flow)
    # loss_ph = loss_photometric(warped_img, img2)
    loss_gr = loss_gradient(warped_img, img2)
    # return loss_ph, loss_gr
    return loss_gr


def data_loss_msy(img1, img2, flow, img1_org, img2_org, occ=1):
    # 数据项包含：1 光度损失 2 梯度损失
    # 目前未考虑遮挡
    warped_img = flow_warp(img1, flow)
    warped_img_org = flow_warp(img1_org, flow)
    loss_ph = loss_photometric(warped_img_org, img2_org, occ)
    loss_gr = loss_gradient(warped_img, img2, occ)
    return loss_ph, loss_gr


def data_loss_msy_oc(img1, img2, flow, img1_org, img2_org, occ=1):
    # 数据项包含：1 光度损失 2 梯度损失
    # 目前未考虑遮挡
    warped_img = flow_warp(img1, flow)
    warped_img_org = flow_warp(img1_org, flow)
    # loss_ph = loss_photometric(warped_img_org, img2_org, occ)
    loss_ph = census_loss_torch(img2_org, warped_img_org, occ, 0.4, False, True, True)
    loss_gr = loss_gradient_oc(warped_img, img2, occ)
    return loss_ph, loss_gr


def smooth_loss(flow, image, alpha=10):
    loss_1st, loss_2nd = loss_smooth(flow, image, alpha)
    return loss_1st, loss_2nd


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def sequence_loss_msy(flow_preds_fw, flow_gt, image1, image2, valid, image1_org, image2_org, gamma=0.8,
                      max_flow=MAX_FLOW, occu_mask=None, loss_org=None):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds_fw)
    flow_loss = 0.0
    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)  # torch.Size([1,368,768])
    for i in range(n_predictions):
        # exponent = n_predictions - i - 1
        exponent = i  # 此处尝试对低分辨率输出赋予大权重
        i_weight = gamma ** exponent
        b, _, h, w = flow_preds_fw[i].size()
        flow_gt_scaled = F.interpolate((flow_gt / (2 ** exponent)), (h, w), mode='bilinear')
        i_loss = (flow_preds_fw[i] - flow_gt_scaled).abs()  # torch.Size([1,2,368,768])
        valid_scaled = (F.interpolate(valid[:, None].float(), (h, w), mode='bilinear') > 0.8)
        flow_loss += i_weight * (valid_scaled * i_loss).mean()
        # flow_loss += 0.1 * i_weight * (valid_scaled * i_loss).mean()
        # 添加的额外的约束，即非EPE损失
        image1_scaled = F.interpolate(image1, (h, w), mode='bilinear')
        image2_scaled = F.interpolate(image2, (h, w), mode='bilinear')
        image1_org_scaled = F.interpolate(image1_org, (h, w), mode='bilinear')
        image2_org_scaled = F.interpolate(image2_org, (h, w), mode='bilinear')
        flow_preds_fw[i] = flow_preds_fw[i] * valid_scaled
        l_ph, l_gr = data_loss_msy(image1_scaled, image2_scaled, flow_preds_fw[i], image1_org_scaled, image2_org_scaled)
        # l_gr = data_loss(image1_scaled, image2_scaled, flow_preds_fw[i])
        l_1st, l_2nd = smooth_loss(flow_preds_fw[i], image2_scaled)
        # l_bw = 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        # l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        # flow_loss += 0.1 * l_bw * i_weight
        # epe:1  data:1  smooth:0.005
        l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.005 * (l_1st + l_2nd)
        flow_loss += l_bw * i_weight

    epe = torch.sum((flow_preds_fw[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]  # torch.Size([282624])

    flow_loss = loss_org + 0.01 * flow_loss

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss_org': loss_org.item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics


def sequence_loss_msy_un(flow_preds_fw, image1, image2, valid, image1_org, image2_org, gamma=0.8,
                         max_flow=MAX_FLOW, occu_mask=None):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds_fw)
    flow_loss = 0.0
    # exclude invalid pixels and extremely large displacements
    flow_gt = flow_preds_fw[-1]
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)  # torch.Size([1,368,768])
    for i in range(n_predictions):
        # exponent = n_predictions - i - 1
        exponent = i  # 此处尝试对低分辨率输出赋予大权重
        i_weight = gamma ** exponent
        b, _, h, w = flow_preds_fw[i].size()
        # flow_gt_scaled = F.interpolate((flow_gt / (2 ** exponent)), (h, w), mode='bilinear')
        # i_loss = (flow_preds_fw[i] - flow_gt_scaled).abs()  # torch.Size([1,2,368,768])
        valid_scaled = (F.interpolate(valid[:, None].float(), (h, w), mode='bilinear') > 0.8)
        # flow_loss += i_weight * (valid_scaled * i_loss).mean()
        # flow_loss += 0.1 * i_weight * (valid_scaled * i_loss).mean()
        # 添加的额外的约束，即非EPE损失
        image1_scaled = F.interpolate(image1, (h, w), mode='bilinear')
        image2_scaled = F.interpolate(image2, (h, w), mode='bilinear')
        image1_org_scaled = F.interpolate(image1_org, (h, w), mode='bilinear')
        image2_org_scaled = F.interpolate(image2_org, (h, w), mode='bilinear')
        flow_preds_fw[i] = flow_preds_fw[i] * valid_scaled
        l_ph, l_gr = data_loss_msy(image1_scaled, image2_scaled, flow_preds_fw[i], image1_org_scaled, image2_org_scaled)
        # l_gr = data_loss(image1_scaled, image2_scaled, flow_preds_fw[i])
        l_1st, l_2nd = smooth_loss(flow_preds_fw[i], image2_scaled)
        # l_bw = 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        # l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        # flow_loss += 0.1 * l_bw * i_weight
        # epe:1  data:1  smooth:0.005
        l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.005 * (l_1st + l_2nd)
        flow_loss += l_bw * i_weight

    # epe = torch.sum((flow_preds_fw[-1] - flow_gt) ** 2, dim=1).sqrt()
    # epe = epe.view(-1)[valid.view(-1)]  # torch.Size([282624])

    metrics = {
        # 'epe': epe.mean().item(),
        # '1px': (epe < 1).float().mean().item(),
        # '3px': (epe < 3).float().mean().item(),
        # '5px': (epe < 5).float().mean().item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics


def sequence_loss_occ(flow_preds_fw, flow_gt, image1, image2, valid, image1_org, image2_org,
                      flow_preds_bw, occ_fw, occ_bw, gamma=0.8, max_flow=MAX_FLOW, occu_mask=None, loss_org=None):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds_fw)
    flow_loss = 0.0
    eps = 1e-6
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)  # torch.Size([1,368,768])
    for i in range(n_predictions):
        exponent = n_predictions - i - 1
        i_weight = gamma ** exponent
        b, _, h, w = flow_preds_fw[i].size()
        flow_gt_scaled = F.interpolate((flow_gt / (2 ** exponent)), (h, w), mode='bilinear')
        occ_fw_eps = (1 - occ_fw[i]) * eps
        occ_bw_eps = (1 - occ_bw[i]) * eps
        occ_fw_i = occ_fw_eps + occ_fw[i]
        occ_bw_i = occ_bw_eps + occ_bw[i]
        i_loss = ((flow_preds_fw[i] - flow_gt_scaled).abs()) * occ_fw_i  # torch.Size([1,2,368,768])
        valid_scaled = (F.interpolate(valid[:, None].float(), (h, w), mode='bilinear') > 0.8)
        # flow_loss += i_weight * (valid_scaled * i_loss).mean()
        flow_loss += 0.1 * i_weight * (valid_scaled * i_loss).mean()  # Shang
        # 添加的额外的约束，即非EPE损失
        image1_scaled = F.interpolate(image1, (h, w), mode='bilinear')
        image2_scaled = F.interpolate(image2, (h, w), mode='bilinear')
        image1_org_scaled = F.interpolate(image1_org, (h, w), mode='bilinear')
        image2_org_scaled = F.interpolate(image2_org, (h, w), mode='bilinear')
        l_ph_fw, l_gr_fw = data_loss_msy_oc(image1_scaled, image2_scaled, flow_preds_fw[i],
                                            image1_org_scaled, image2_org_scaled, occ=occ_fw_i)
        # l_1st_fw, l_2nd_fw = smooth_loss(flow_preds_fw[i], image2_scaled)
        l_ph_bw, l_gr_bw = data_loss_msy_oc(image2_scaled, image1_scaled, flow_preds_bw[i],
                                            image2_org_scaled, image1_org_scaled, occ=occ_bw_i)
        # l_1st_bw, l_2nd_bw = smooth_loss(flow_preds_bw[i], image1_scaled)
        l_ph = l_ph_fw + l_ph_bw
        l_gr = l_gr_fw + l_gr_bw
        if i == 3:
            l_1st_fw, l_2nd_fw = smooth_loss(flow_preds_fw[i], image2_scaled)
            l_1st_bw, l_2nd_bw = smooth_loss(flow_preds_bw[i], image1_scaled)
        else:
            l_1st_fw = 0
            l_2nd_fw = 0
            l_1st_bw = 0
            l_2nd_bw = 0
        # l_gr = data_loss(image1_scaled, image2_scaled, flow_preds_fw[i])
        # l_bw = 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        l_1st = l_1st_fw + l_1st_bw
        l_2nd = l_2nd_fw + l_2nd_bw
        # l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        # flow_loss += 0.1 * l_bw * i_weight
        # epe:1  data:1  smooth:0.005
        l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.005 * (l_1st + l_2nd)
        flow_loss += l_bw * i_weight

    flow_loss = loss_org + 0.01 * flow_loss

    epe = torch.sum((flow_preds_fw[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]  # torch.Size([282624])

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss_org': loss_org.item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics


def sequence_loss_occ_un(flow_preds_fw, image1, image2, image1_org, image2_org,
                         flow_preds_bw, occ_fw, occ_bw, gamma=0.8, max_flow=MAX_FLOW, occu_mask=None):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds_fw)
    flow_loss = 0.0
    eps = 1e-6
    # exlude invalid pixels and extremely large diplacements
    # mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    # valid = (valid >= 0.5) & (mag < max_flow)  # torch.Size([1,368,768])
    for i in range(n_predictions):
        exponent = n_predictions - i - 1
        i_weight = gamma ** exponent
        b, _, h, w = flow_preds_fw[i].size()
        # flow_gt_scaled = F.interpolate((flow_gt / (2 ** exponent)), (h, w), mode='bilinear')
        occ_fw_eps = (1 - occ_fw[i]) * eps
        occ_bw_eps = (1 - occ_bw[i]) * eps
        occ_fw_i = occ_fw_eps + occ_fw[i]
        occ_bw_i = occ_bw_eps + occ_bw[i]
        # i_loss = ((flow_preds_fw[i] - flow_gt_scaled).abs()) * occ_fw_i  # torch.Size([1,2,368,768])
        # valid_scaled = (F.interpolate(valid[:, None].float(), (h, w), mode='bilinear') > 0.8)
        # flow_loss += i_weight * (valid_scaled * i_loss).mean()
        # flow_loss += 0.1 * i_weight * (valid_scaled * i_loss).mean()  # Shang
        # 添加的额外的约束，即非EPE损失
        image1_scaled = F.interpolate(image1, (h, w), mode='bilinear')
        image2_scaled = F.interpolate(image2, (h, w), mode='bilinear')
        image1_org_scaled = F.interpolate(image1_org, (h, w), mode='bilinear')
        image2_org_scaled = F.interpolate(image2_org, (h, w), mode='bilinear')
        l_ph_fw, l_gr_fw = data_loss_msy_oc(image1_scaled, image2_scaled, flow_preds_fw[i],
                                            image1_org_scaled, image2_org_scaled, occ=occ_fw_i)
        # l_1st_fw, l_2nd_fw = smooth_loss(flow_preds_fw[i], image2_scaled)
        l_ph_bw, l_gr_bw = data_loss_msy_oc(image2_scaled, image1_scaled, flow_preds_bw[i],
                                            image2_org_scaled, image1_org_scaled, occ=occ_bw_i)
        # l_1st_bw, l_2nd_bw = smooth_loss(flow_preds_bw[i], image1_scaled)
        if i == 3:
            l_1st_fw, l_2nd_fw = smooth_loss(flow_preds_fw[i], image2_scaled)
            l_1st_bw, l_2nd_bw = smooth_loss(flow_preds_bw[i], image1_scaled)
        else:
            l_1st_fw = 0
            l_2nd_fw = 0
            l_1st_bw = 0
            l_2nd_bw = 0
        l_ph = l_ph_fw + l_ph_bw
        l_gr = l_gr_fw + l_gr_bw
        # l_gr = data_loss(image1_scaled, image2_scaled, flow_preds_fw[i])
        # l_bw = 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        l_1st = l_1st_fw + l_1st_bw
        l_2nd = l_2nd_fw + l_2nd_bw
        # l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        # flow_loss += 0.1 * l_bw * i_weight
        # epe:1  data:1  smooth:0.005
        l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.005 * (l_1st + l_2nd)
        flow_loss += l_bw * i_weight

    # epe = torch.sum((flow_preds_fw[-1] - flow_gt) ** 2, dim=1).sqrt()
    # epe = epe.view(-1)[valid.view(-1)]  # torch.Size([282624])

    metrics = {
        # 'epe': epe.mean().item(),
        # '1px': (epe < 1).float().mean().item(),
        # '3px': (epe < 3).float().mean().item(),
        # '5px': (epe < 5).float().mean().item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics


def sequence_loss(flow_preds_fw, flow_gt, image1, image2, valid, gamma=0.8, max_flow=MAX_FLOW, occu_mask=None):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds_fw)
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)  # torch.Size([1,368,768])
    for i in range(n_predictions):
        exponent = n_predictions - i - 1
        i_weight = gamma ** exponent
        b, _, h, w = flow_preds_fw[i].size()
        flow_gt_scaled = F.interpolate(flow_gt, (h, w), mode='bilinear')
        i_loss = (flow_preds_fw[i] - flow_gt_scaled).abs()  # torch.Size([1,2,368,768])
        valid_scaled = (F.interpolate(valid[:, None].float(), (h, w), mode='bilinear') > 0.8)
        # flow_loss += i_weight * (valid_scaled * i_loss).mean()
        # 添加的额外的约束，即非EPE损失
        image1_scaled = F.interpolate(image1, (h, w), mode='bilinear')
        image2_scaled = F.interpolate(image2, (h, w), mode='bilinear')
        l_ph, l_gr = data_loss(image1_scaled, image2_scaled, flow_preds_fw[i])
        # l_gr = data_loss(image1_scaled, image2_scaled, flow_preds_fw[i])
        l_1st, l_2nd = smooth_loss(flow_preds_fw[i], image2_scaled)
        # l_bw = 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        flow_loss += 0.1 * l_bw * i_weight

    epe = torch.sum((flow_preds_fw[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]  # torch.Size([282624])

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # numel()返回数组中元素的个数


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:  # metrics中以字典形式存储了epe，1px，3px和5px的数据
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(FRAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)
        # 加载已训练模型参数，strict为False，来自加载模型的keys不需要与torch的state_dict函数方法完全匹配
    model.cuda()
    model.train()  # 将模块设置为训练模式

    if args.stage != 'chairs':  # and args.stage != 'solar':
        model.module.freeze_bn()  # 判断m的类型是否与nn.BatchNorm2d的类型相同，相同执行一个字符串表达式，并返回表达式的值

    train_loader = datasets.fetch_dataloader(args)  # 获取数据加载器
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 10000  # 5000
    add_noise = True

    aug_params = {'crop_size': [384, 448], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

    should_keep_training = True
    while should_keep_training:
        # t_start = time_synchronized()
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            # 此处是考虑到代码内对图像的增强操作，为了避免亮度误差等，引入了亮度无变化的图像对image_org
            _, _, _, _, image1_org_ar, image2_org_ar = [x.cuda() for x in data_blob]

            # image1, image2, flow, valid = [x.to(device) for x in data_blob]

            # 单向流
            if total_steps < 0:
                # run 1st forward
                flow_gt, _, _ = model(image1_org_ar, image2_org_ar)
                flow_final = flow_gt[-1]
                valid = (flow_final[0].abs() < 1000) & (flow_final[1].abs() < 1000)
                loss_org, metrics = sequence_loss_msy_un(flow_gt, image1_org_ar, image2_org_ar, valid, image1_org_ar,
                                                         image2_org_ar, args.gamma)
                # construct augment sample
                augmentor = FlowAugmentorMSY_S(**aug_params)
                flow = flow_gt[-1]

                img1_0 = image1_org_ar[0]
                img1_0 = img1_0.permute(1, 2, 0)
                img1_0 = img1_0.cpu()
                img1_0 = img1_0.numpy()
                img1_0 = (img1_0 * 255).astype(np.uint8)
                img1_1 = image1_org_ar[1]
                img1_1 = img1_1.permute(1, 2, 0)
                img1_1 = img1_1.cpu()
                img1_1 = img1_1.numpy()
                img1_1 = (img1_1 * 255).astype(np.uint8)
                img2_0 = image2_org_ar[0]
                img2_0 = img2_0.permute(1, 2, 0)
                img2_0 = img2_0.cpu()
                img2_0 = img2_0.numpy()
                img2_0 = (img2_0 * 255).astype(np.uint8)
                img2_1 = image2_org_ar[1]
                img2_1 = img2_1.permute(1, 2, 0)
                img2_1 = img2_1.cpu()
                img2_1 = img2_1.numpy()
                img2_1 = (img2_1 * 255).astype(np.uint8)
                flow_0 = flow[0]
                flow_0 = flow_0.permute(1, 2, 0)
                flow_0 = flow_0.cpu()
                flow_0 = flow_0.detach()
                flow_0 = flow_0.numpy()
                flow_1 = flow[1]
                flow_1 = flow_1.permute(1, 2, 0)
                flow_1 = flow_1.cpu()
                flow_1 = flow_1.detach()
                flow_1 = flow_1.numpy()

                img1_0, img2_0, flow_0, img1_org_0, img2_org_0 = augmentor(img1_0, img2_0, flow_0)
                img1_1, img2_1, flow_1, img1_org_1, img2_org_1 = augmentor(img1_1, img2_1, flow_1)

                img1_0 = torch.from_numpy(img1_0).permute(2, 0, 1).float().detach().cuda()
                img1_1 = torch.from_numpy(img1_1).permute(2, 0, 1).float().detach().cuda()
                img2_0 = torch.from_numpy(img2_0).permute(2, 0, 1).float().detach().cuda()
                img2_1 = torch.from_numpy(img2_1).permute(2, 0, 1).float().detach().cuda()
                flow_0 = torch.from_numpy(flow_0).permute(2, 0, 1).float().detach().cuda()
                flow_1 = torch.from_numpy(flow_1).permute(2, 0, 1).float().detach().cuda()
                img1_org_0 = torch.from_numpy(img1_org_0).permute(2, 0, 1).float().detach().cuda()
                img1_org_1 = torch.from_numpy(img1_org_1).permute(2, 0, 1).float().detach().cuda()
                img2_org_0 = torch.from_numpy(img2_org_0).permute(2, 0, 1).float().detach().cuda()
                img2_org_1 = torch.from_numpy(img2_org_1).permute(2, 0, 1).float().detach().cuda()

                image1_org_ar = torch.stack([img1_0, img1_1], dim=0)
                image2_org_ar = torch.stack([img2_0, img2_1], dim=0)
                flow = torch.stack([flow_0, flow_1], dim=0)
                img1_org = torch.stack([img1_org_0, img1_org_1], dim=0)
                img2_org = torch.stack([img2_org_0, img2_org_1], dim=0)
                #
                # if args.add_noise:
                #     stdv = np.random.uniform(0.0, 5.0)
                #     image1_org_ar = (image1_org_ar + stdv * torch.randn(*image1_org_ar.shape).cuda()).clamp(0.0, 255.0)
                #     image2_org_ar = (image2_org_ar + stdv * torch.randn(*image2_org_ar.shape).cuda()).clamp(0.0, 255.0)
                #     img1_org = (img1_org + stdv * torch.randn(*img1_org.shape).cuda()).clamp(0.0, 255.0)
                #     img2_org = (img2_org + stdv * torch.randn(*img2_org.shape).cuda()).clamp(0.0, 255.0)
                # run 2nd pass
                flow_prediction_fw, occu_mask, _ = model(image1_org_ar, image2_org_ar)
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
                #
                loss, metrics = sequence_loss_msy(flow_prediction_fw, flow,
                                                  image1_org_ar, image2_org_ar, valid, img1_org, img2_org, args.gamma,
                                                  occu_mask=occu_mask, loss_org=loss_org)
            # 双向流
            else:
                flow_gt_fw, flow_gt_bw, occu_mask_fw, occu_mask_bw = model(image1_org_ar, image2_org_ar,
                                                                           add_bw=True)
                loss_org, _ = sequence_loss_occ_un(flow_gt_fw, image1_org_ar, image2_org_ar, image1_org_ar,
                                                   image2_org_ar, flow_gt_bw, occu_mask_fw, occu_mask_bw,
                                                   gamma=args.gamma)
                augmentor = FlowAugmentorMSY_S(**aug_params)
                flow = flow_gt_fw[-1]

                img1_0 = image1_org_ar[0]
                img1_0 = img1_0.permute(1, 2, 0)
                img1_0 = img1_0.cpu()
                img1_0 = img1_0.numpy()
                img1_0 = (img1_0 * 255).astype(np.uint8)
                img1_1 = image1_org_ar[1]
                img1_1 = img1_1.permute(1, 2, 0)
                img1_1 = img1_1.cpu()
                img1_1 = img1_1.numpy()
                img1_1 = (img1_1 * 255).astype(np.uint8)
                img2_0 = image2_org_ar[0]
                img2_0 = img2_0.permute(1, 2, 0)
                img2_0 = img2_0.cpu()
                img2_0 = img2_0.numpy()
                img2_0 = (img2_0 * 255).astype(np.uint8)
                img2_1 = image2_org_ar[1]
                img2_1 = img2_1.permute(1, 2, 0)
                img2_1 = img2_1.cpu()
                img2_1 = img2_1.numpy()
                img2_1 = (img2_1 * 255).astype(np.uint8)
                flow_0 = flow[0]
                flow_0 = flow_0.permute(1, 2, 0)
                flow_0 = flow_0.cpu()
                flow_0 = flow_0.detach()
                flow_0 = flow_0.numpy()
                flow_1 = flow[1]
                flow_1 = flow_1.permute(1, 2, 0)
                flow_1 = flow_1.cpu()
                flow_1 = flow_1.detach()
                flow_1 = flow_1.numpy()

                img1_0, img2_0, flow_0, img1_org_0, img2_org_0 = augmentor(img1_0, img2_0, flow_0)
                img1_1, img2_1, flow_1, img1_org_1, img2_org_1 = augmentor(img1_1, img2_1, flow_1)

                img1_0 = torch.from_numpy(img1_0).permute(2, 0, 1).float().detach().cuda()
                img1_1 = torch.from_numpy(img1_1).permute(2, 0, 1).float().detach().cuda()
                img2_0 = torch.from_numpy(img2_0).permute(2, 0, 1).float().detach().cuda()
                img2_1 = torch.from_numpy(img2_1).permute(2, 0, 1).float().detach().cuda()
                flow_0 = torch.from_numpy(flow_0).permute(2, 0, 1).float().detach().cuda()
                flow_1 = torch.from_numpy(flow_1).permute(2, 0, 1).float().detach().cuda()
                img1_org_0 = torch.from_numpy(img1_org_0).permute(2, 0, 1).float().detach().cuda()
                img1_org_1 = torch.from_numpy(img1_org_1).permute(2, 0, 1).float().detach().cuda()
                img2_org_0 = torch.from_numpy(img2_org_0).permute(2, 0, 1).float().detach().cuda()
                img2_org_1 = torch.from_numpy(img2_org_1).permute(2, 0, 1).float().detach().cuda()

                image1_org_ar = torch.stack([img1_0, img1_1], dim=0)
                image2_org_ar = torch.stack([img2_0, img2_1], dim=0)
                flow = torch.stack([flow_0, flow_1], dim=0)
                img1_org = torch.stack([img1_org_0, img1_org_1], dim=0)
                img2_org = torch.stack([img2_org_0, img2_org_1], dim=0)

            flow_prediction_fw, flow_prediction_bw, occu_mask_fw, occu_mask_bw = model(image1_org_ar, image2_org_ar,
                                                                                       add_bw=True)
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            loss, metrics = sequence_loss_occ(flow_prediction_fw, flow, image1_org_ar, image2_org_ar, valid,
                                              img1_org, img2_org, flow_prediction_bw, occu_mask_fw, occu_mask_bw,
                                              gamma=args.gamma, loss_org=loss_org)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # 添加assert来判断更新前参数是否为NaN

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == 'solar':
                        results.update(evaluate.validate_solar_upflow(model.module))
                        # results.update(evaluate.validate_solarzero(model.module))
                        results.update(evaluate.validate_optical_upflow(model.module))
                    elif val_dataset == 'optical':
                        results.update(evaluate.validate_solar_upflow(model.module))
                        results.update(evaluate.validate_optical_upflow(model.module))
                        results.update(evaluate.validate_solarzero(model.module))

                logger.write_dict(results)

                model.train()
                if args.stage != 'chairs':  # and args.stage != 'solar':
                    model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

            # if total_steps % 100 == 99:
            #     t_end = time_synchronized()
            #     print("inference time: {}".format(t_end - t_start))

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    # 训练：--name raft-solar_dis --stage solardis --restore_ckpt checkpoints/raft-solar.pth --validation solardis --gpus 0 --num_steps 60000 --batch_size 2 --lr 0.001 --image_size 384 424 --wdecay 0.0001 --gamma=0.5 --add_bw_flow
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')  # action当参数在命令行中出现时使用的动作基本类型
    parser.add_argument('--validation', type=str, nargs='+')  # type 命令行参数应当被转换成的类型；nargs 命令行参数应当消耗的数目

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=8)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)  # ε
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')  # action保存相应布尔值，store_true表示触发action为真，否则为假
    parser.add_argument('--add_bw_flow', action='store_true', help='add backward flow')
    parser.add_argument('--small_bw', action='store_true', help='add small module backward flow')
    args = parser.parse_args()

    random.seed(1234)
    torch.manual_seed(1234)  # 设置CPU生成随机数的种子，方便下次复现实验结果,能使得调用torch.rand(n)的结果一致，
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)  # 注：是每次运行py文件的输出结果一样，而不是每次随机函数生成的结果一样
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.deterministic = True
    #  np.random.seed(n) 用于生成指定随机数，n代表第n堆种子，想每次都能得到相同的随机数，每次产生随机数之前，都需要调用一次seed()
    if not os.path.isdir('../checkpoints'):
        os.mkdir('../checkpoints')
    # print(torch.backends.cudnn.benchmark)
    # print(torch.backends.cudnn.deterministic)
    # control determine
    # --name unet-solar --stage solar --validation solar --gpus 0 --num_steps 120000 --batch_size 2
    # --lr 0.001 --image_size 384 448 --wdecay 0.0001 --gamma=0.5
    # 继续训练
    # --name unet-optical --stage optical --validation optical --restore_ckpt checkpoints/unet-solar.pth --gpus 0
    # --num_steps 60000 --batch_size 2 --lr 0.0001 --image_size 256 320 --wdecay 0.0001 --gamma=0.5

    # --name unet-solar-ul --stage solar --validation solar --restore_ckpt checkpoints/85000_unet-solar-ul.pth --gpus 0
    # --num_steps 35000 --batch_size 2 --lr 0.0003 --image_size 384 448 --wdecay 0.0001 --gamma=0.5

    train(args)
