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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import compute_loss
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from maml import MAML

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, debug_from, inner_lr):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1)

    first_iter = scene.loaded_iter if scene.loaded_iter else 0
    if first_iter == opt.ambient_iterations:
        gaussians.phong_init()

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    camera_interval = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)[0]["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            if (camera_interval+1)*args.task_fetch_num > len(scene.getTrainCameras()):
                camera_interval = 0
            viewpoint_stack = scene.getTrainCameras().copy()[ camera_interval*args.task_fetch_num : (camera_interval+1)*args.task_fetch_num ]
            camera_interval += 1
        elif len(viewpoint_stack) < opt.task_num:
            if (camera_interval+1)*args.task_fetch_num > len(scene.getTrainCameras()):
                camera_interval = 0
            viewpoint_stack = scene.getTrainCameras().copy()[ camera_interval*args.task_fetch_num : (camera_interval+1)*args.task_fetch_num ]
            camera_interval += 1
            
        viewpoint_cams = []
        for i in range (opt.task_num):
            viewpoint_cams.append(viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        seperate_render = True if iteration % 10 == 1 else False
        cam_nums = len(viewpoint_cams)
        random.shuffle(list(range(cam_nums)))
        viewpoint_cam_train = viewpoint_cams
        viewpoint_cam_test = [viewpoint_cams[i%cam_nums] for i in range (opt.task_num, opt.task_num+cam_nums)]
        
        model = MAML(gaussians, lr=inner_lr, first_order=False, allow_unused=True, allow_nograd=True) # get feature 有grad_fn
        model.train()
        
        loss = 0.0
        Ll1 = 0.0
        for i in range(len(viewpoint_cam_train)):
            learner = model.clone()
            ### Inner Loop
            # update phong
            viewpoint_cam = viewpoint_cam_train[i]
            render_pkg, losses_extra = render(viewpoint_cam, learner, pipe, bg, seperate_render=seperate_render, 
                                            mode="phong", shadow=False, train_normal=True)
            gt_image = viewpoint_cam.original_image.cuda()
            loss_i = compute_loss(render_pkg, gt_image, losses_extra, opt)
            learner.adapt(loss_i)
            torch.cuda.empty_cache()
            # update shadow
            render_pkg, losses_extra = render(viewpoint_cam, learner, pipe, bg, seperate_render=seperate_render, 
                                            mode="phong", shadow=True, train_normal=True, only_shadow=True)
            gt_image = viewpoint_cam.original_image.cuda()
            loss_i = compute_loss(render_pkg, gt_image, losses_extra, opt)
            learner.adapt_only_shadow(loss_i)
            ### Outer Loop
            viewpoint_cam = viewpoint_cam_test[i]
            render_pkg, losses_extra = render(viewpoint_cam, learner, pipe, bg, seperate_render=seperate_render, 
                                            mode="phong", shadow=True, train_normal=True)
            gt_image = viewpoint_cam.original_image.cuda()
            adapt_loss, adapt_Ll1 = compute_loss(render_pkg, gt_image, losses_extra, opt, return_l1=True)
            loss += adapt_loss
            Ll1 += adapt_Ll1
            torch.cuda.empty_cache()

        ### Update the model
        Ll1 = Ll1 / len(viewpoint_cam_test)
        loss = loss / len(viewpoint_cam_test)
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                tb_writer.add_histogram(f'{name}.grad', param.grad, iteration)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"rgb_loss": f"{Ll1.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), 
                            gt_image, render_pkg, losses_extra, mode="phong", shadow=True)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % (opt.densification_interval//20) == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if args.ambient_iterations+5 < iteration < args.reset_opacity_until_iter and iteration % opt.opacity_reset_interval_shadow == 5:
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                    gt_image=None, render_pkg={}, losses_extra={}, mode="gaussian", shadow=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f'train_loss_patches/{k}_loss', losses_extra[k].item(), iteration)
        if iteration % 10 == 1:
            tb_writer.add_images("train/gt", gt_image[None], global_step=iteration)
            tb_writer.add_images("train/pred", render_pkg['render'][None], global_step=iteration)
            if 'normal' in render_pkg.keys():
                tb_writer.add_images("train/normal_pred", render_pkg['normal'][None], global_step=iteration)
                tb_writer.add_images("train/normal_gt", render_pkg['normal_ref'][None], global_step=iteration)
            for key in render_pkg.keys():
                if "shadow" in key or 'ambient' in key or 'specular' in key or 'diffuse' in key:
                    tb_writer.add_images("train/"+key, render_pkg[key][None], global_step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, 
                                            train_normal=True, mode=mode, shadow=shadow, seperate_render=True,)[0]
                    image = render_pkg["pred_shadow"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/pred".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     tb_writer.add_image(config['name'] + "_view_{}/gt".format(viewpoint.image_name), gt_image, global_step=iteration)
                    #     if 'normal' in render_pkg.keys():
                    #         tb_writer.add_images(config['name'] + "_view_{}/normal_pred".format(viewpoint.image_name), render_pkg['normal'][None], global_step=iteration)
                    #         tb_writer.add_images(config['name'] + "_view_{}/normal_gt".format(viewpoint.image_name), render_pkg['normal_ref'][None], global_step=iteration)
                    #     for key in render_pkg.keys():
                    #         if "shadow" in key or 'ambient' in key or 'specular' in key or 'diffuse' in key:
                    #             tb_writer.add_images(config['name'] + "_view_{}/".format(viewpoint.image_name)+key, render_pkg[key][None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(0, 30_000, 50)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(3_000, 30_000, 50)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.debug_from, args.inner_lr)

    # All done
    print("\nTraining complete.")
