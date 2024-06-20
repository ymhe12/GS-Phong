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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from bvh import RayTracer
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis, flip_align_view
from utils.sh_utils import eval_sh
from utils.loss_utils import l1_loss, ssim, predicted_normal_loss, delta_normal_loss, tv_loss

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        super(GaussianModel, self).__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._normal = torch.empty(0)
        self.diffuse_rgb = torch.empty(0)
        self.light_intensity = torch.empty(0)
        self.cos_p = torch.empty(0)
        self.visibility_coef = torch.empty(0)
        self.specular_coef = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._normal,
            self.light_intensity,
            self.visibility_coef,
            self.specular_coef,
            self.diffuse_rgb,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._normal,
        self.light_intensity,
        self.visibility_coef,
        self.specular_coef,
        self.diffuse_rgb,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, self.get_rotation)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_normal(self, dir_pp_normalized=None, return_delta=False):
        normal_axis = self.get_minimum_axis
        normal_axis = normal_axis
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        delta_normal1 = self._normal  # (N, 3) 
        delta_normal2 = self._normal # (N, 3) 
        delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) # (N, 3, 2)
        idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) # (N, 3, 1)
        delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) # (N, 3)
        normal = delta_normal + normal_axis 
        normal = normal/normal.norm(dim=1, keepdim=True) # (N, 3)

        if return_delta:
            return normal, delta_normal
        else:
            return normal

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        normals = torch.randn((fused_point_cloud.shape[0], 3)).float().cuda()
        light_intensity = torch.ones((1)).float().cuda()*5
        visibility_coef = torch.ones((1)).float().cuda()
        specular_coef = torch.ones((fused_point_cloud.shape[0], 1)).float().cuda()
        diffuse_rgb = torch.ones((fused_point_cloud.shape[0], 3)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(normals.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.light_intensity = nn.Parameter(light_intensity.requires_grad_(True))
        self.visibility_coef = nn.Parameter(visibility_coef.requires_grad_(True))
        self.specular_coef = nn.Parameter(specular_coef.requires_grad_(True))
        self.diffuse_rgb = nn.Parameter(diffuse_rgb.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.cos_p = training_args.cos_p
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self.light_intensity], 'lr': training_args.light_intensity_lr, "name": "light_intensity"},
            {'params': [self.visibility_coef], 'lr': training_args.visibility_coef_lr, "name": "visibility_coef"},
            {'params': [self.specular_coef], 'lr': training_args.specular_coef_lr, "name": "specular_coef"},
            {'params': [self.diffuse_rgb], 'lr': training_args.diffuse_rgb_lr_init, "name": "diffuse_rgb"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.diffuse_rgb_scheduler_args = get_expon_lr_func(lr_init=training_args.diffuse_rgb_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.diffuse_rgb_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.diffuse_rgb_lr_delay_mult,
                                                        max_steps=training_args.diffuse_rgb_lr_max_steps,
                                                        start_step=training_args.gaussian_iterations+1)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "diffuse_rgb":
                lr = self.diffuse_rgb_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('max_radii2D')
        for i in range(self.diffuse_rgb.shape[1]):
            l.append('diffuse_rgb_{}'.format(i))
        for i in range(self.specular_coef.shape[1]):
            l.append('specular_coef_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        max_radii2D = self.max_radii2D.unsqueeze(-1).detach().cpu().numpy()
        diffuse_rgb = self.diffuse_rgb.detach().cpu().numpy()
        specular_coef = self.specular_coef.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, 
                                     max_radii2D, diffuse_rgb, specular_coef), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_light(self, path):
        mkdir_p(os.path.dirname(path))
        light_intensity = self.light_intensity.detach().cpu().numpy()
        cos_p = self.cos_p
        visibility_coef = self.visibility_coef.detach().cpu().numpy()
        np.savez(path, light_intensity=light_intensity, cos_p=cos_p, visibility_coef=visibility_coef)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                          np.asarray(plydata.elements[0]["ny"]),
                          np.asarray(plydata.elements[0]["nz"])),  axis=1)
        diffuse_rgb = np.stack((np.asarray(plydata.elements[0]["diffuse_rgb_0"]),
                                np.asarray(plydata.elements[0]["diffuse_rgb_1"]),
                                np.asarray(plydata.elements[0]["diffuse_rgb_2"])),  axis=1)
        
        max_radii2D = np.asarray(plydata.elements[0]["max_radii2D"])
        specular_coef = np.asarray(plydata.elements[0]["specular_coef_0"])[..., np.newaxis]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.tensor(max_radii2D, dtype=torch.float, device="cuda")
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self.diffuse_rgb = nn.Parameter(torch.tensor(diffuse_rgb, dtype=torch.float, device="cuda").requires_grad_(True))
        self.specular_coef = nn.Parameter(torch.tensor(specular_coef, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_light(self, path):
        data = np.load(path)
        cos_p = data['cos_p']
        light_intensity = data['light_intensity']
        visibility_coef = data['visibility_coef']
        self.cos_p = torch.tensor(cos_p, dtype=torch.float, device="cuda")
        self.light_intensity = nn.Parameter(torch.tensor(light_intensity, dtype=torch.float, device="cuda").requires_grad_(True))
        self.visibility_coef = nn.Parameter(torch.tensor(visibility_coef, dtype=torch.float, device="cuda").requires_grad_(True))
        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "light_intensity" in group['name']\
                or "visibility_coef" in group['name']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]
        self.specular_coef = optimizable_tensors["specular_coef"]
        self.diffuse_rgb = optimizable_tensors["diffuse_rgb"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group['name'] in tensors_dict.keys():
                continue 
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                              new_normal, new_specular_coef, new_diffuse_rgb):
        d = {"xyz": new_xyz,
        "normal": new_normal,
        "specular_coef": new_specular_coef,
        "diffuse_rgb": new_diffuse_rgb,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]
        self.specular_coef = optimizable_tensors["specular_coef"]
        self.diffuse_rgb = optimizable_tensors["diffuse_rgb"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        big_point_mask = torch.where(self.get_scaling.max(dim=1).values > self.percent_dense*scene_extent, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, big_point_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1)
        new_specular_coef = self.specular_coef[selected_pts_mask].repeat(N,1)
        new_diffuse_rgb = self.diffuse_rgb[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_normal, new_specular_coef, new_diffuse_rgb)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_specular_coef = self.specular_coef[selected_pts_mask]
        new_diffuse_rgb = self.diffuse_rgb[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                   new_normal, new_specular_coef, new_diffuse_rgb)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.01 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)
    
    def get_visibility(self, light_xyz):
        means3D = self.get_xyz
        normal = self.get_normal(dir_pp_normalized=torch.randn_like(means3D))
        opacity = self.get_opacity
        scaling = self.get_scaling
        rotation = self.get_rotation
        cov_inv = self.get_inverse_covariance()
        raytracer = RayTracer(means3D, scaling, rotation)
        rays_o = means3D
        rays_d = light_xyz.unsqueeze(0).repeat(means3D.shape[0], 1) - means3D
        light_visibility = raytracer.trace_visibility(
            rays_o,
            rays_d,
            means3D,
            cov_inv,
            opacity,
            normal
        )["visibility"]
        return light_visibility
 
    def phong_init(self):
        # Init light intensity
        light_intensity = torch.ones((1)).float().cuda() * 10
        self.light_intensity = nn.Parameter(torch.tensor(light_intensity, requires_grad=True))
        # Init diffuse rgb
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp_normalized = torch.randn_like(self.get_xyz)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        ambient_color = torch.clamp_min(sh2rgb + 0.5, 0.0)
        diffuse_detach = ambient_color*2 # diffuse init
        self.diffuse_rgb = nn.Parameter(torch.tensor(diffuse_detach, requires_grad=True))
        # Init specular coef
        specular_coef = torch.ones((self.get_xyz.shape[0], 1)).float().cuda() * 10
        self.specular_coef = nn.Parameter(torch.tensor(specular_coef, requires_grad=True))
        # Init visibility coef
        visibility_coef = torch.ones((1)).float().cuda() * 1#0
        self.visibility_coef = nn.Parameter(torch.tensor(visibility_coef, requires_grad=True))


def compute_loss(render_pkg, gt_image, losses_extra, opt, ambient_restrict=True, return_l1=False):
    
    loss = 0.0
    image = render_pkg["render"]
    if "pred_shadow" in render_pkg.keys():
        image = render_pkg["pred_shadow"]
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    if ambient_restrict:
        image = render_pkg['ambient']
        image = torch.where(image<=gt_image, gt_image, image)
        losses_extra['ambient_restrict'] = (1.0 - opt.lambda_dssim) *  l1_loss(image, gt_image) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    if "normal" in render_pkg.keys():
        losses_extra['predicted_normal'] = predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"], render_pkg["alpha"])
        losses_extra['delta_reg'] = delta_normal_loss(render_pkg["delta_normal"], render_pkg["alpha"])
        # losses_extra['zero_one'] = zero_one_loss(render_pkg["alpha"])
        losses_extra['normal_smooth'], losses_extra['normal_smooth2d'] = tv_loss(render_pkg['normal'])
    if "ambient" in render_pkg.keys():
        losses_extra['ambient_smooth'], losses_extra['ambient_smooth2d'] = tv_loss(render_pkg['ambient'])
    if "diffuse" in render_pkg.keys():
        losses_extra['diffuse_smooth'], losses_extra['diffuse_smooth2d'] = tv_loss(render_pkg['diffuse'])
    if "specular" in render_pkg.keys():
        losses_extra['specular_smooth'], losses_extra['specular_smooth2d'] = tv_loss(render_pkg['specular'])

    for k in losses_extra.keys():
        loss += getattr(opt, f'lambda_{k}')* losses_extra[k]

    if return_l1:
        return loss, Ll1
    else:
        return loss