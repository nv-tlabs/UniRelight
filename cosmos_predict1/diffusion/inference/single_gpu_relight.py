# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import os
import time
import glob
import torch

from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.visualize.video import save_img_or_video

from cosmos_predict1.diffusion.inference.inference_utils import load_model_by_config, load_network_model
from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader.dataset_inference import VideoGBufferDataset
from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader.dataloader_utils import dict_collation_fn
from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader.utils_env_proj import process_environment_map
from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader.rendering_utils import envmap_vec

from cosmos_predict1.diffusion.training.models.model_relight import MultiviewExtendDiffusionModelRelight

torch.enable_grad(False)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="single gpu only inference script. It does not support tp/sp saved checkpoints!"
    )
    parser.add_argument("--config", type=str, default="", help="inference only config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--config_file", type=str, default="cosmos_predict1/diffusion/training/config/config_relight.py")
    parser.add_argument("--output_path", type=str, default="./", help="Output path")
    parser.add_argument(
        "--inference_passes",
        type=str,
        default=["rgb"],
        nargs="+",
        help="List of inference passes."
    )
    parser.add_argument("--save_gt", type=str2bool, default=True, help="")
    parser.add_argument("--save_condition", type=str2bool, default=True, help="")

    parser.add_argument("--dataset_name", type=str, default=None,
                        help="The name of a dataset, or a path to a folder of video frames")
    parser.add_argument("--sample_n_frames", type=int, default=57, help="Custom data configs")
    parser.add_argument("--image_extensions", type=str, default=None, nargs="+", help="Custom data configs")
    parser.add_argument("--resolution", type=int, default=[480, 848], nargs="+", help="Custom data configs")
    parser.add_argument("--resize_resolution", type=int, default=[486, 864], nargs="+", help="Custom data configs")
    parser.add_argument("--envlight_ind", type=lambda s: list(map(int, s.split(','))), default=None, help="Lighting configs")
    parser.add_argument("--rotate_light", type=str2bool, default=False, help="Lighting configs")
    parser.add_argument("--use_fixed_frame_ind", type=str2bool, default=False, help="")
    parser.add_argument("--fixed_frame_ind", type=int, default=0, help="")
    parser.add_argument("--overwrite_fps", type=float, default=None, help="over write fps in the data batch")
    parser.add_argument("--seed", type=int, default=10007, help="Seed for the generation")
    parser.add_argument("--save_images", type=str2bool, default=False, help="Save images for each video")
    parser.add_argument("--num_steps", type=int, default=35, help="Num of denoising steps")
    parser.add_argument("--use_input_albedo", type=str2bool, default=False, help="Use input albedo as condition")
    parser.add_argument("--env_light_path", type=str, default=None, help="Path to the environment light")
    return parser.parse_args()


args = parse_arguments()

log.info(f"args: {args}")

# instantiate model, config and load checkpoint
model, config = load_model_by_config(
    config_job_name=args.config,
    config_file=args.config_file,
    model_class=MultiviewExtendDiffusionModelRelight
)

load_network_model(model=model, ckpt_path=args.ckpt_path)

misc.set_random_seed(seed=args.seed, by_rank=True)

# Env map path examples
ENV_LIGHT_PATH_LIST = sorted(glob.glob(os.path.join(args.env_light_path, "*.hdr"))) if os.path.isdir(args.env_light_path) else [args.env_light_path]
assert len(ENV_LIGHT_PATH_LIST) > 0, f"No environment light found in {args.env_light_path}"
envlight_ind = list(range(len(ENV_LIGHT_PATH_LIST))) if args.envlight_ind is None else args.envlight_ind

device = torch.device("cuda")

# Create the dataloader instance

gbuf_labels = ['rgb', 'basecolor'] if args.use_input_albedo else ['rgb']
dataset = VideoGBufferDataset(
    root_dir=args.dataset_name,
    sample_n_frames=args.sample_n_frames,
    image_extensions=args.image_extensions,
    resolution=args.resolution,
    resize_resolution=args.resize_resolution,
    gbuf_labels=gbuf_labels,
    group_mode="folder",
    bg_color=(1., 1., 1.),
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=dict_collation_fn)

iter_dataloader = iter(dataloader)

model = model.to(torch.bfloat16)

os.makedirs(args.output_path, exist_ok=True)
success_signal_dir = os.path.join(args.output_path, "TMP_SUCCESS_SIGNAL")
os.makedirs(success_signal_dir, exist_ok=True)

for i in range(len(dataloader)):

    start_time = time.time()
    # Load batch
    data_batch = next(iter_dataloader)
    data_batch = misc.to(data_batch, device="cuda", dtype=torch.bfloat16)   # move to GPU
    data_batch['rgb_ref'] = data_batch['rgb']

    clip_name = data_batch['clip_name'][0].replace("/", "--")

    for env_id in envlight_ind:
        success_signal_path = os.path.join(success_signal_dir, f"{clip_name}_light-{env_id:04d}")
        
        if os.path.exists(success_signal_path):
            log.info(f"Skip! {i:05d}_env{env_id:03d}.")
            continue

        envlight_path = ENV_LIGHT_PATH_LIST[env_id]
        envlight_dict = process_environment_map(
            envlight_path,
            resolution=args.resolution,
            num_frames=args.sample_n_frames,
            fixed_pose=True,
            rotate_envlight=args.rotate_light,
            env_format=['proj', ],
            device=device,
        )  # Tensors are with shape (T, H, W, 3) in [0, 1]
        env_ldr = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        env_log = envlight_dict['env_log'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        env_nrm = envmap_vec(args.resolution, device=device)  # [H, W, 3]
        env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)
        
    
        data_batch['env_ldr'] = env_ldr
        data_batch['env_log'] = env_log
        data_batch['env_nrm'] = env_nrm

        if args.use_fixed_frame_ind:
            # use a static frame
            for attributes in ['rgb_ref', 'basecolor',]:
                if attributes in data_batch:
                    data_batch[attributes] = data_batch[attributes][:, :, args.fixed_frame_ind:args.fixed_frame_ind + 1, ...].expand_as(data_batch[attributes])

        # for placeholder
        data_batch['video'] = data_batch['rgb_ref']

        if args.overwrite_fps is not None:
            data_batch['fps'].fill_(args.overwrite_fps)

        # Generate samples
        log.info("==> processing video for %s" % (data_batch["clip_name"][0]))

        C = model.state_shape[0]
        F = (data_batch['rgb_ref'].shape[2] - 1) // 8 + 1
        H = data_batch['rgb_ref'].shape[3] // 8
        W = data_batch['rgb_ref'].shape[4] // 8
        
        for inference_pass in args.inference_passes:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=0.0,
                seed=args.seed,
                state_shape=(C, F * 3, H, W),
                is_negative_prompt=False,
                use_gt_albedo=args.use_input_albedo, 
                num_steps=args.num_steps,
            )

            video = model.decode(sample)
            clip_name = data_batch['clip_name'][0].replace("/", "--")
            video_save_path = os.path.join(args.output_path, f"{clip_name}_light-{env_id:04d}")
            save_video = torch.cat(torch.chunk(video, 3, dim=2), dim=-1)
            if args.save_condition:
                condition_sample = torch.cat([
                    data_batch[attributes] for attributes in (['basecolor', 'env_ldr'] if args.use_input_albedo else ['env_ldr'])
                    if attributes in data_batch
                ], dim=-1)
                save_video = torch.cat([condition_sample, save_video], dim=-1)
            
            save_img_or_video((1.0 + save_video[0]) / 2, video_save_path, save_images=args.save_images)
            log.info("==> finished saving for %s" % (video_save_path))
        
        os.makedirs(success_signal_path)

    total_time = time.time() - start_time
    log.info(f"Total time for processing batch {i}: {total_time:.4f} seconds")
