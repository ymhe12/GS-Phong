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
from utils.graphics_utils import render_normal, normalize_normal_inplace

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, use_phong_model=False, train_normal=False):
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
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
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
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "pred": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "scales": scales,
        "opacity": opacity,
        "depth" : depth_image
        }
    
    if not train_normal:
        return out

    out_extras = {}
    
    # Rasterize visible Gaussians to alpha mask image. 
    bg_alpha = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    raster_settings_alpha = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_alpha,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False,
        antialiasing=pipe.antialiasing
    )
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(means3D) 
    out_extras["alpha"] =  rasterizer_alpha(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = alpha,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)[0]
    
    if train_normal:
        # Calculate Gaussians projected depth as ground truth
        p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1).cuda(), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        depth = depth.repeat(1,3)
        render_extras = {"depth": depth}
        # Calculate induced normal using the shortest axis
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True)
        delta_normal = delta_normal.norm(dim=1, keepdim=True)
        normal_normed = 0.5*normal + 0.5  # range (-1, 1) -> (0, 1)
        render_extras.update({"normal": normal_normed})
        render_extras.update({"delta_normal": delta_normal.repeat(1, 3)})
        # Get rendered normal image
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
        # Render normal from depth image, and alpha blend with the background. 
        out_extras['normal_ref'] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], bg_color=bg_color, alpha=out_extras['alpha'][0])
        normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])

    if use_phong_model:
        ambient_pc = colors_precomp
        out_extras.update({"ambient": rendered_image})
        
        gaussian_num = means3D.shape[0]
        light = viewpoint_camera.light_xyz.cuda().repeat(gaussian_num, 1)
        render_dict = compute_pc_color_with_single_light(viewpoint_camera, normal, pc, means3D, light, pc.cos_p, dir_pp_normalized)
        diffuse_pc = torch.clamp(render_dict["diffuse_pc"], 0., 1.)
        specular_pc = torch.clamp(render_dict["specular_pc"], 0., 1.)
        diffuse_shadow_pc = torch.clamp(render_dict["diffuse_pc"] * render_dict["light_visibility"], 0., 1.)
        specular_shadow_pc = torch.clamp(render_dict["specular_pc"] * render_dict["light_visibility"], 0., 1.)
        rgb_pc = torch.clamp(diffuse_pc + ambient_pc + specular_pc, 0., 1.)
        rgb_shadow_pc = torch.clamp(diffuse_shadow_pc + ambient_pc + specular_shadow_pc, 0., 1.)

        out_extras["diffuse"] = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = diffuse_pc,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        
        out_extras["pred"] = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = rgb_shadow_pc,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        
        out_extras["visibility"] = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = render_dict["light_visibility"].repeat(1, 3),
            opacities = torch.ones_like(opacity),
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
    
    out.update(out_extras)
    
    return out

def compute_pc_color_with_single_light(viewpoint_camera, normal, pc, means3D, light, cos_p, dir_pp_normalized):

    out_dict = {}
    
    cam = viewpoint_camera.camera_center.cuda().repeat(means3D.shape[0], 1)
    v = F.normalize(cam - means3D, dim = 1) # gaussian-to-camera direction v
    l = F.normalize(light - means3D, dim = 1) # gaussian-to-light direction l
    h = F.normalize(v + l, dim = 1) # bisector h
    r = (light - means3D).norm(dim=1) # distance between gaussian and light
    r = torch.clamp(r, min=1e-5) # avoid zero division
    
    light_intensity = pc.light_intensity
    reached_intensity = light_intensity / (r**2)
    reached_intensity = reached_intensity[:, None]
    
    # Diffuse
    diffuse_coef = pc.get_diffuse_coef
    cos_theta = (normal * l).sum(dim=-1)
    diffuse_pc = diffuse_coef * reached_intensity * torch.clamp(cos_theta, min=0.0)[:, None]
    
    # Specular
    specular_coef =  pc.get_specular_coef
    cos_alpha = (normal*h).sum(dim=-1)
    cos_beta = (normal*v).sum(dim=-1)
    legal_specular = torch.where(cos_theta * cos_beta > 0., 1., 0.)[:, None] # light and camera on the same side
    specular_pc = legal_specular * specular_coef * reached_intensity * torch.clamp(cos_alpha, min=0.0).pow(int(cos_p))[:, None]
    
    # Clamp
    diffuse_pc = torch.clamp(diffuse_pc, 0., 1.)
    specular_pc = torch.clamp(specular_pc, 0., 1.)

    # Visibility
    light_visibility = pc.get_visibility(light[0].cuda(), dir_pp_normalized)
    
    out_dict.update({
        "diffuse_coef": diffuse_coef,
        "diffuse_pc": diffuse_pc,
        "specular_pc": specular_pc,
        "light_visibility": light_visibility,
        "specular_coef": specular_coef,
    })
    
    return out_dict
