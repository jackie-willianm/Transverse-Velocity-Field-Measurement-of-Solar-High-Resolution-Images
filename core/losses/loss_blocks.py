import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2.

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def loss_smooth(flo, image, alpha=10):
    loss_1 = smooth_grad_1st(flo, image, alpha)
    loss_2 = smooth_grad_2nd(flo, image, alpha)
    return loss_1, loss_2


def loss_photometric(image1, image2, occ=1):
    eps = 0.01
    dif = torch.mean(torch.abs(image1 - image2) * occ, 1, keepdim=True) + eps
    exponent = 0.4
    loss = torch.pow(dif, exponent)
    return torch.mean(loss)
    # loss = torch.mean(torch.abs(image1 - image2))
    # return loss


def loss_gradient(image1, image2, occ=1):
    exponent = 0.4
    eps = 0.01
    dx1, dy1 = gradient(image1)
    dx2, dy2 = gradient(image2)
    dif_x = torch.mean(torch.abs(dx1 - dx2) * occ, 1, keepdim=True) + eps
    dif_y = torch.mean(torch.abs(dy1 - dy2) * occ, 1, keepdim=True) + eps
    loss_x = torch.pow(dif_x, exponent)
    loss_y = torch.pow(dif_y, exponent)
    return torch.mean(loss_x) + torch.mean(loss_y)
    # dx1, dy1 = gradient(image1)
    # dx2, dy2 = gradient(image2)
    # loss_x = torch.mean(torch.abs(dx1 - dx2))
    # loss_y = torch.mean(torch.abs(dy1 - dy2))
    # return loss_x + loss_y


def loss_gradient_oc(image1, image2, occ=1):
    exponent = 0.4
    eps = 0.01
    dx1, dy1 = gradient(image1)
    dx2, dy2 = gradient(image2)
    occ_y = occ[:, :, :-1]
    occ_x = occ[:, :, :, :-1]
    dif_x = torch.mean(torch.abs(dx1 - dx2) * occ_x, 1, keepdim=True) + eps
    dif_y = torch.mean(torch.abs(dy1 - dy2) * occ_y, 1, keepdim=True) + eps
    loss_x = torch.pow(dif_x, exponent)
    loss_y = torch.pow(dif_y, exponent)
    return torch.mean(loss_x) + torch.mean(loss_y)


class loss_functions():

    @classmethod
    def photo_loss_function(cls, diff, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True):
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

    @classmethod
    def census_loss_torch(cls, img1, img1_warp, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True,
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
        census_loss = cls.photo_loss_function(diff=dist, mask=mask * transform_mask, q=q,
                                              charbonnier_or_abs_robust=charbonnier_or_abs_robust,
                                              if_use_occ=if_use_occ, averge=averge)
        return census_loss

    @classmethod
    def flow_smooth_delta(cls, flow, if_second_order=False):
        def gradient(x):
            D_dy = x[:, :, 1:] - x[:, :, :-1]
            D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            return D_dx, D_dy

        dx, dy = gradient(flow)
        # dx2, dxdy = gradient(dx)
        # dydx, dy2 = gradient(dy)
        if if_second_order:
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            smooth_loss = dx.abs().mean() + dy.abs().mean() + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        else:
            smooth_loss = dx.abs().mean() + dy.abs().mean()
        # smooth_loss = dx.abs().mean() + dy.abs().mean()  # + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        # 暂时不上二阶的平滑损失，似乎加上以后就太猛了，无法降低photo loss TODO
        return smooth_loss

    @classmethod
    def edge_aware_smoothness_per_pixel(cls, img, pred):
        def gradient_x(img):
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx

        def gradient_y(img):
            gy = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gy

        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)
