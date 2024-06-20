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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "no_shadow")
    ambient_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ambient")
    diffuse_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse")
    specular_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    shadow_path = os.path.join(model_path, name, "ours_{}".format(iteration), "shadow")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(ambient_path, exist_ok=True)
    makedirs(diffuse_path, exist_ok=True)
    makedirs(specular_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(shadow_path, exist_ok=True)
    render_images = []
    ambient_images = []
    diffuse_images = []
    specular_images = []
    normal_images = []
    shadow_images = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        result = render(view, gaussians, pipeline, background, seperate_render=True, mode="phong", shadow=True, train_normal=True)[0]
        gt = view.original_image[0:3, :, :]
        rendering = result["render"]
        diffuse_rendering = result["diffuse"]
        specular_rendering = result["specular"]
        ambient_rendering = result["ambient"]
        normal_rendering = 0.5 + 0.5*result["normal"]
        shadow_rendering = result["pred_shadow"]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(ambient_rendering, os.path.join(ambient_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(diffuse_rendering, os.path.join(diffuse_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(specular_rendering, os.path.join(specular_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal_rendering, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(shadow_rendering, os.path.join(shadow_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1,2,0))
        ambient_images.append(to8b(ambient_rendering).transpose(1,2,0))
        diffuse_images.append(to8b(diffuse_rendering).transpose(1,2,0))
        specular_images.append(to8b(specular_rendering).transpose(1,2,0))
        normal_images.append(to8b(normal_rendering).transpose(1,2,0))
        shadow_images.append(to8b(shadow_rendering).transpose(1,2,0))
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_diffuse.mp4'), diffuse_images, fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_specular.mp4'), specular_images, fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_ambient.mp4'), ambient_images, fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_normal.mp4'), normal_images, fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_shadow.mp4'), shadow_images, fps=30, quality=8)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)