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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn.functional as F
from utils.graphics_utils import normal_from_depth_image
from utils.loss_utils import scale_loss, opacity_loss, zero_one_loss, diffuse_rgb_loss

def render(viewpoint_camera, 
           pc : GaussianModel, 
           pipe, 
           bg_color : torch.Tensor, 
           mode = "gaussian",
           shadow = False,
           only_shadow = False,
           seperate_render = False,
           train_normal = True,
           scaling_modifier = 1.0, 
           override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    out_extras = {}
    losses_extra = {}

    if train_normal:
        ### Calculate Gaussians projected depth as ground truth
        p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1).cuda(), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        depth = depth.repeat(1,3)
        ### Calculate gt normal using the shortesst axis
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True)
        delta_normal = delta_normal.norm(dim=1, keepdim=True)
        ### Get rendered normal image
        render_extras = {"depth": depth}
        normal_normed = 0.5*normal + 0.5  # range (-1, 1) -> (0, 1)
        render_extras.update({"normal": normal_normed})
        render_extras.update({"delta_normal": delta_normal.repeat(1, 3)})
        for k in render_extras.keys():
            image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = render_extras[k],
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)[0]
            out_extras[k] = image
        out_extras["normal"] = (out_extras["normal"] - 0.5) * 2. # range (0, 1) -> (-1, 1)
        ### Rasterize visible Gaussians to alpha mask image. 
        raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=False
        )
        rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
        ones = torch.ones_like(means3D) 
        out_extras['alpha'] =  rasterizer_alpha(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = ones,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        ### Render normal from depth image, and alpha blend with the background. 
        out_extras['normal_ref'] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], bg_color=bg_color, alpha=out_extras['alpha'][0])
        normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])

    if mode == "phong":
        ### phong shading
        gaussian_num = means3D.shape[0]
        light_xyz = viewpoint_camera.light_xyz.cuda().repeat(gaussian_num, 1)
        light_intensity = pc.light_intensity
        # gaussian-to-camera direction v
        v = viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1) - means3D
        v = F.normalize(v, dim = 1)
        # gaussian-to-light direction l
        l = light_xyz - means3D
        l = F.normalize(l, dim = 1)
        # bisector h
        h = F.normalize(v + l, dim = 1)
        # reached intensity 
        r = (light_xyz - means3D).norm(dim=1)
        reached_intensity = light_intensity / r**2
        
        # Ambient color
        ambient_color = colors_precomp
        
        # Diffuse color
        if shadow:
            diffuse_rgb = pc.diffuse_rgb.detach()
        else:
            diffuse_rgb = pc.diffuse_rgb
        losses_extra.update({"diffuse_rgb_loss": diffuse_rgb_loss(diffuse_rgb, ambient_color.detach())})
        zeros = torch.zeros(gaussian_num).cuda()
        cos_l = (normal*l).sum(dim=-1)
        cos_v = (normal*v).sum(dim=-1)
        diffuse_intensity = reached_intensity * torch.where(cos_l*cos_v > 0, torch.abs(cos_l), zeros)
        diffuse_color = diffuse_intensity.unsqueeze(-1).repeat(1, 3)  * diffuse_rgb

        # Specular color
        cos_h = (normal*h).sum(dim=-1)
        specular_intensity = reached_intensity * torch.where(cos_l*cos_v > 0, (torch.abs(cos_h)).pow(pc.cos_p), zeros)
        specular_color = specular_intensity.unsqueeze(-1).repeat(1, 3)
        if only_shadow:
            specular_coef = pc.specular_coef.detach()
        else:
            specular_coef = pc.specular_coef
        specular_color = specular_color * specular_coef
        specular_color = torch.clamp(specular_color, 0., 1.)

        if shadow:
            light_visibility = pc.get_visibility(viewpoint_camera.light_xyz.cuda())
            light_visibility = light_visibility * pc.visibility_coef
            light_visibility = torch.clamp(light_visibility, 0., 1.)

        ambient_color = torch.clamp(ambient_color, 0., 1.)
        diffuse_color = torch.clamp(diffuse_color, 0., 1.)
        specular_color = torch.clamp(specular_color, 0., 1.)

        out_extras['ambient'] = rendered_image

        colors_precomp = ambient_color + diffuse_color + specular_color
        colors_precomp = torch.clamp(colors_precomp, 0., 1.)
        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)
        
        if shadow:
            colors_precomp = ambient_color + (diffuse_color + specular_color) * light_visibility.repeat(1, 3)
            colors_precomp = torch.clamp(colors_precomp, 0., 1.)
            out_extras['pred_shadow'] = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = None)[0]
            
        if seperate_render:
            out_extras['diffuse'] = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = diffuse_color,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = None)[0]
            out_extras['specular'] = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = specular_color,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = None)[0]

    visibility_filter = radii > 0
    out = { "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : visibility_filter,
            "scales": scales,
            "opacity": opacity,
            "radii": radii}
    out.update(out_extras)

    losses_extra.update({
        "flatten": scale_loss(scales), 
        "sparse": opacity_loss(opacity, visibility_filter),
        })
    if shadow:
        losses_extra.update({"sparse_shadow": zero_one_loss(light_visibility)})

    return out, losses_extra


def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)