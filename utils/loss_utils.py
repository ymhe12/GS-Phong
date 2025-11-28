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
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass
from utils.image_utils import erode
import numpy as np
import torchvision.transforms.functional as TF

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
import torch.nn as nn

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

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


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

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

def opacity_loss(opacity, visibility_filter=None): # From GaussianPro
    opacity = opacity.reshape(-1, 1)
    opacity = opacity.clamp(1e-6, 1-1e-6)
    log_opacity = opacity * torch.log(opacity)
    log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
    if visibility_filter is not None:
        sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
    else:
        sparse_loss = -1 * (log_opacity + log_one_minus_opacity).mean()
    return sparse_loss

def rgb_consistency_loss(img1, img2, mask, epsilon=1e-8):
    img1 = img1 * mask
    img2 = img2 * mask
    img1 = img1.reshape(-1, 3)
    img2 = img2.reshape(-1, 3)
    norm_img1 = img1 / (img1.sum(dim=1, keepdim=True) + epsilon)
    norm_img2 = img2 / (img2.sum(dim=1, keepdim=True) + epsilon)
    loss = torch.mean(torch.abs(norm_img1 - norm_img2).sum(dim=1))
    return loss

def cal_gradient(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient


def bilateral_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad).exp() * mask).mean()

    return smooth_loss

def compute_loss(render_pkg, pred, gt_image, alpha_mask=None, white_bg=True):
    
    losses_extra = {}
    losses_extra["sparse"] = opacity_loss(render_pkg["opacity"], render_pkg['visibility_filter'])
    losses_extra["flatten"] = scale_loss(render_pkg["scales"])
    
    if "diffuse" in render_pkg.keys():
        losses_extra['whole_consistency'] = rgb_consistency_loss(pred, gt_image, render_pkg["alpha"])
        losses_extra["diffuse_consistency"] = rgb_consistency_loss(render_pkg['diffuse'], gt_image, render_pkg['visibility'])

    if "normal" in render_pkg.keys():
        losses_extra['predicted_normal'] = predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"], render_pkg["alpha"])
        losses_extra['delta_reg'] = delta_normal_loss(render_pkg["delta_normal"], render_pkg["alpha"])
        losses_extra['normal_smooth'] = bilateral_smooth_loss(render_pkg['normal'], gt_image, alpha_mask)

        normal_ref = render_pkg['normal_ref'].clamp(0, 1)
        normal_alpha_loss = torch.where(alpha_mask==0, (1-normal_ref) if white_bg else normal_ref, 0)
        alpha_loss = normal_alpha_loss
        losses_extra['alpha_mask'] = torch.sum(alpha_loss) / (alpha_loss.shape[1]*alpha_loss.shape[2])

    return losses_extra