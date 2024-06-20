#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.image_utils import erode
import numpy as np
import torchvision.transforms.functional as TF

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)
        weight = erode(weight, erode_size=4)
        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(normal_ref)
    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()
    return loss

def scale_loss(scales):
    min_scale, _ = torch.min(scales, dim=1)
    min_scale = torch.clamp(min_scale, 0, 30)
    flatten_loss = torch.abs(min_scale).mean()
    return flatten_loss

def opacity_loss(opacity, visibility_filter):
    opacity = opacity.clamp(1e-6, 1-1e-6)
    log_opacity = opacity * torch.log(opacity)
    log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
    sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
    return sparse_loss

def delta_normal_loss(delta_normal, alpha=None):
    # To prevent the normal residual from deviating too much from the shortest axis, we add a penalty towards normal residual, making sure it is small enough
    # delta_normal: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device)
    else:
        weight = torch.ones_like(delta_normal)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    l = weight.permute(1,2,0).reshape(-1,3)[...,0]
    loss = (w * l).mean()
    return loss

def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def diffuse_rgb_loss(img1, img2):
    img1 = img1.reshape(-1, 3)
    img2 = img2.reshape(-1, 3)
    # compute the scaling factor s
    numerator = (img1 * img2).sum(dim=0)
    denominator = (img2 * img2).sum(dim=0)
    s = numerator / denominator
    loss = ((img1 - s[None,:] * img2) ** 2).mean()
    return loss

def tv_loss(img):
    pred_vals = img.unsqueeze(0).permute(0,2,3,1) # (B, H, W, 3)
    normal_smooth = (pred_vals[:, 1:, :, :] - pred_vals[:, :-1, :, :]).square().mean() + \
        (pred_vals[:, :, 1:, :] -pred_vals[:, :, :-1, :]).square().mean()
    pred_vals = img.unsqueeze(0).contiguous()  # (B, 3, H, W)
    smoothed_vals = TF.gaussian_blur(pred_vals, kernel_size=9)
    normal_smooth2d = F.mse_loss(pred_vals, smoothed_vals)
    return normal_smooth, normal_smooth2d