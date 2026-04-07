# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_predict1.diffusion.training.callbacks.low_precision import LowPrecisionCallback
from cosmos_predict1.diffusion.training.datasets.dataset_multiview import Dataset
from cosmos_predict1.diffusion.training.models.model_relight import FSDPDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_lvg_multiview_relight import VideoExtendMultiviewGeneralDITRelight
from cosmos_predict1.utils import log
from cosmos_predict1.utils.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

num_frames = 57
num_views = 3


unirelight_cosmos_f57_480p = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_unirelight_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="unirelight",
            name="unirelight_cosmos_f57_480p",
        ),
        optimizer=dict(
            lr=2e-5,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=400,
            broadcast_via_filesystem=False,
            load_path="",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=1000000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=200,
                    hit_thres=5,
                ),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=8,
        ),
        model=dict(
            n_views=num_views,
            # Use 16x16x32x40 latent shape for training
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                88,  # Latent height dim
                160,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=False,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendMultiviewGeneralDITRelight)(
                additional_concat_ch=17 * 3,
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
                n_views=num_views,
                block_x_format="THWBD",
            ),
            conditioner=dict(
                latent_condition=dict(
                    dropout_rate=0.05,
                ),
                video_cond_bool=dict(
                    condition_location="first_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                ),
            ),
            vae=dict(
                video_vae=dict(
                    pixel_chunk_duration=num_frames,
                )
            ),
            condition_keys=['env_ldr', 'env_log', 'env_nrm', ],
            condition_drop_rate=0.05,
            append_condition_mask=True,
            guidance_weights=0.1,
            source_drop_rate=0.3,
        ),
        model_obj=L(FSDPDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
    )
)



log.info(f"Registering experiment: unirelight_cosmos_f57_480p")
cs.store(
    group="experiment",
    package="_global_",
    name="unirelight_cosmos_f57_480p",
    node=unirelight_cosmos_f57_480p,
)
