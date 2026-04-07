# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Callable, Dict, Tuple, Union, Optional


import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
import numpy as np

from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.conditioner import DataType
from cosmos_predict1.diffusion.training.conditioner import VideoExtendCondition, BaseVideoCondition
from cosmos_predict1.diffusion.training.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.training.models.extend_model import (
    ExtendDiffusionModel,
    VideoDenoisePrediction,
    normalize_condition_latent,
)
from cosmos_predict1.diffusion.training.models.model import DiffusionModel, broadcast_condition, _broadcast
from cosmos_predict1.diffusion.training.models.model_image import CosmosCondition, diffusion_fsdp_class_decorator
from cosmos_predict1.utils import log, misc

from cosmos_predict1.diffusion.config.base.conditioner import VideoCondBoolConfig

class MultiviewExtendDiffusionModelRelight(ExtendDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_views = config.n_views

        if hasattr(config, "frame_buffer_max"):
            self.frame_buffer_max = config.frame_buffer_max
        else:
            self.frame_buffer_max = 0

        self.condition_keys = config.condition_keys
        self.condition_drop_rate = config.condition_drop_rate
        self.append_condition_mask = config.append_condition_mask
        self.guidance_weights = config.guidance_weights if config.guidance_weights is not None else None
        self.source_drop_rate = config.source_drop_rate

    @torch.no_grad()
    def encode(self, state: torch.Tensor, n_views: Optional[int] = None) -> torch.Tensor:
        if n_views is None: n_views = self.n_views
        if n_views == 1:
            return self.vae.encode(state) * self.sigma_data
        
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = self.vae.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views) * self.sigma_data
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, n_views: Optional[int] = None) -> torch.Tensor:
        if n_views is None: n_views = self.n_views
        if n_views == 1:
            return self.vae.decode(latent / self.sigma_data)

        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = self.vae.decode(latent / self.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: CosmosCondition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        custom_loss_weights = None
        if self.guidance_weights is not None:
            custom_loss_weights = torch.ones_like(x0)
            VT = custom_loss_weights.shape[2]
            custom_loss_weights[:, :, -VT // self.n_views:] *= self.guidance_weights

        if self.is_image_batch(data_batch):
            # Turn off CP
            self.net.disable_context_parallel()
        else:
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    # Turn on CP
                    cp_group = parallel_state.get_context_parallel_group()
                    self.net.enable_context_parallel(cp_group)
                    log.debug("[CP] Split x0 and epsilon")

                    x0 = rearrange(x0, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
                    epsilon = rearrange(epsilon, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
                    condition.latent_condition = rearrange(condition.latent_condition, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
                    
                    x0 = split_inputs_cp(x=x0, seq_dim=2, cp_group=self.net.cp_group)
                    epsilon = split_inputs_cp(x=epsilon, seq_dim=2, cp_group=self.net.cp_group)
                    condition.get_condition_for_cp(cp_group=self.net.cp_group)

                    x0 = rearrange(x0, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
                    epsilon = rearrange(epsilon, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
                    condition.latent_condition = rearrange(condition.latent_condition, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

                    if custom_loss_weights is not None:
                        custom_loss_weights = rearrange(custom_loss_weights, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
                        custom_loss_weights = split_inputs_cp(x=custom_loss_weights, seq_dim=2, cp_group=self.net.cp_group)
                        custom_loss_weights = rearrange(custom_loss_weights, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        
        output_batch, kendall_loss, pred_mse, edm_loss = super(
            DiffusionModel, self
        ).compute_loss_with_epsilon_and_sigma(data_batch, x0_from_data_batch, x0, condition, epsilon, sigma, custom_loss_weights)
        if not self.is_image_batch(data_batch):
            if self.loss_reduce == "sum" and parallel_state.get_context_parallel_world_size() > 1:
                kendall_loss *= parallel_state.get_context_parallel_world_size()

        return output_batch, kendall_loss, pred_mse, edm_loss

    def prepare_diffusion_renderer_latent_conditions(
            self, data_batch: dict[str, Tensor],
            condition_keys: list[str] = ["rgb"], condition_drop_rate: float = 0, append_condition_mask: bool = True,
            dtype: torch.dtype = None, device: torch.device = None,
            latent_shape: Union[Tuple[int, int, int, int, int], torch.Size] = None,
            mode="train",
            append_strength: bool = True,
    ) -> Tensor:
        if latent_shape is None:
            B, C, T, H, W = data_batch[condition_keys[0]].shape
            latent_shape = (B, 16, T // 8 + 1, H // 8, W // 8)
        if append_condition_mask:
            latent_mask_shape = (latent_shape[0], 1, latent_shape[2], latent_shape[3], latent_shape[4])
        if dtype is None:
            dtype = data_batch[condition_keys[0]].dtype
        if device is None:
            device = data_batch[condition_keys[0]].device

        latent_condition_list = []
        env_latent_condition_list = []

        for cond_key in condition_keys:
            is_condition_dropped = condition_drop_rate > 0 and np.random.rand() < condition_drop_rate
            is_condition_skipped = cond_key not in data_batch
            if is_condition_dropped or is_condition_skipped:
                # Dropped or skipped condition
                condition_state = torch.zeros(latent_shape, dtype=dtype, device=device)

                if 'env' not in cond_key:
                    latent_condition_list.append(condition_state)
                else:
                    env_latent_condition_list.append(condition_state)

                if append_condition_mask:
                    condition_mask = torch.zeros(latent_mask_shape, dtype=dtype, device=device)

                    if 'env' not in cond_key:
                        latent_condition_list.append(condition_mask)
                    else:
                        env_latent_condition_list.append(condition_mask)
            else:
                # Valid condition
                condition_state = data_batch[cond_key].to(device=device, dtype=dtype)
                condition_state = self.encode(condition_state, n_views=1).contiguous()
                if 'env' not in cond_key:
                    latent_condition_list.append(condition_state)
                else:
                    env_latent_condition_list.append(condition_state)

                if append_condition_mask:
                    condition_mask = torch.ones(latent_mask_shape, dtype=dtype, device=device)
                    if 'env' not in cond_key:
                        latent_condition_list.append(condition_mask)
                    else:
                        env_latent_condition_list.append(condition_mask)

            if cond_key.startswith("corrupted_") and append_strength:
                condition_strength = torch.ones(latent_mask_shape, dtype=dtype, device=device) * data_batch[cond_key + "_strength"].to(device=device, dtype=dtype)

                if 'env' not in cond_key:
                    latent_condition_list.append(condition_strength)
                else:
                    env_latent_condition_list.append(condition_strength)
            
        latent_condition = torch.cat(latent_condition_list, dim=1) if len(latent_condition_list) > 0 else None
        env_latent_condition = torch.cat(env_latent_condition_list, dim=1) if len(env_latent_condition_list) > 0 else None
        return latent_condition, env_latent_condition
    
    def get_data_and_condition(self, data_batch: dict[str, Tensor], mode: Union[str, None] = 'default') -> Tuple[Tensor, BaseVideoCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        input_key = self.input_data_key  # by default it is video key
        is_image_batch = self.is_image_batch(data_batch)
        is_video_batch = not is_image_batch

        # Broadcast data and condition across TP and CP groups.
        # sort keys to make sure the order is same, IMPORTANT! otherwise, nccl will hang!
        local_keys = sorted(list(data_batch.keys()))
        for key in local_keys:
            data_batch[key] = _broadcast(data_batch[key], to_tp=True, to_cp=is_video_batch)

        if is_image_batch:
            input_key = self.input_image_key

        if 'rgb_ref' not in data_batch:
            data_batch['rgb_ref'] = torch.zeros_like(data_batch[input_key])
            assert 'basecolor' in data_batch
            mode = 'gt_albedo'
        else:
            if self.source_drop_rate > 0. and np.random.rand() < self.source_drop_rate:
                assert 'basecolor' in data_batch
                if np.random.rand() < 0.4:
                    data_batch['rgb_ref'] = torch.zeros_like(data_batch[input_key])
                    mode = 'gt_albedo'
                else:
                    mode = 'both_cond'

        if 'basecolor' not in data_batch:
            data_batch['basecolor'] = torch.zeros_like(data_batch[input_key])

        # Latent state
        raw_state = torch.cat([data_batch['rgb_ref'], data_batch[input_key], data_batch['basecolor']], dim=2) # shape:[B, C, T, H, W]
        
        latent_state = self.encode(raw_state).contiguous()

        if mode == 'gt_albedo':
            T = latent_state.shape[2]
            latent_state[:, :, :T // 3] *= 0.

        cond_latent_shape = [latent_state.shape[0], latent_state.shape[1], latent_state.shape[2] // 3, latent_state.shape[3], latent_state.shape[4]]
        with torch.no_grad():
            latent_condition, env_latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch,
                condition_keys=self.condition_keys,
                condition_drop_rate=self.condition_drop_rate,
                append_condition_mask=self.append_condition_mask,
                dtype=latent_state.dtype, device=latent_state.device,
                latent_shape=cond_latent_shape,
            )

        zero_env_latent_condition = torch.zeros_like(env_latent_condition)
        data_batch["latent_condition"] = torch.cat([zero_env_latent_condition, env_latent_condition, zero_env_latent_condition], dim=2)

        # Condition
        condition = self.conditioner(data_batch)
        if is_image_batch:
            condition.data_type = DataType.IMAGE
        else:
            condition.data_type = DataType.VIDEO

        # VAE has randomness. CP/TP group should have the same encoded output.

        latent_state = _broadcast(latent_state, to_tp=True, to_cp=is_video_batch)
        condition = broadcast_condition(condition, to_tp=True, to_cp=is_video_batch)

        if condition.data_type == DataType.VIDEO:
            if self.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
                latent_state = self.sample_tokens_start_from_p_or_i(latent_state)
        
        if mode == 'default':
            condition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, condition, num_condition_t=latent_state.shape[2] // 3, to_cp=is_video_batch
            )
        elif mode == 'gt_albedo' or mode == 'both_cond':
            T = latent_state.shape[2]
            condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(latent_state.dtype)  # 1 for condition region
            condition_video_indicator[:, :, :T // 3] += 1.0
            condition_video_indicator[:, :, -T // 3:] += 1.0
            condition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, condition, to_cp=is_video_batch, condition_video_indicator=condition_video_indicator,
            )
        else:
            raise ValueError
        
        if self.config.conditioner.video_cond_bool.add_pose_condition:
            condition = self.add_condition_pose(data_batch, condition)

        log.debug(f"condition.data_type {condition.data_type}")

        return raw_state, latent_state, condition
    
    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        use_gt_albedo: float = False,
    ) -> VideoDenoisePrediction:
        """
        Denoise the noisy input tensor.

        Args:
            noise_x (Tensor): Noisy input tensor.
            sigma (Tensor): Noise level.
            condition (VideoExtendCondition): Condition for denoising.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference

        Returns:
            Tensor: Denoised output tensor.
        """

        assert (
            condition.gt_latent is not None
        ), f"find None gt_latent in condition, likely didn't call self.add_condition_video_indicator_and_video_input_mask when preparing the condition or this is a image batch but condition.data_type is wrong, get {noise_x.shape}"
        gt_latent = condition.gt_latent
        cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

        condition_latent = gt_latent

        if cfg_video_cond_bool.normalize_condition_latent:
            condition_latent = normalize_condition_latent(condition_latent)

        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        condition, augment_latent = self.augment_conditional_latent_frames(
            condition, cfg_video_cond_bool, condition_latent, condition_video_augment_sigma_in_inference, sigma
        )
        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]

        if condition.data_type != DataType.IMAGE and parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()

            condition_video_indicator = rearrange(
                condition_video_indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_views
            )
            augment_latent = rearrange(augment_latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            gt_latent = rearrange(gt_latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)

            condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
            augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
            gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)

            condition_video_indicator = rearrange(
                condition_video_indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_views
            )
            augment_latent = rearrange(augment_latent, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
            gt_latent = rearrange(gt_latent, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        if not condition.video_cond_bool:
            # Unconditional case, drop out the condition region
            augment_latent = self.drop_out_condition_region(augment_latent, noise_x, cfg_video_cond_bool)
        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
        # Call the abse model
        denoise_pred = super(DiffusionModel, self).denoise(new_noise_xt, sigma, condition)

        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0
        if cfg_video_cond_bool.compute_loss_for_condition_region:
            # We also denoise the conditional region
            x0_pred = denoise_pred.x0
        else:
            x0_pred = x0_pred_replaced

        return VideoDenoisePrediction(
            x0=x0_pred,
            eps=batch_mul(noise_x - x0_pred, 1.0 / sigma),
            logvar=denoise_pred.logvar,
            net_in=batch_mul(1.0 / torch.sqrt(self.sigma_data**2 + sigma**2), new_noise_xt),
            net_x0_pred=denoise_pred.x0,
            xt=new_noise_xt,
            x0_pred_replaced=x0_pred_replaced,
        )

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None, to_cp: Union[bool, None] = None, condition_video_indicator: Union[torch.Tensor, None] = None,
    ) -> VideoExtendCondition:
        """Add condition_video_indicator and condition_video_input_mask to the condition object for video conditioning.
        condition_video_indicator is a binary tensor indicating the condition region in the latent state. 1x1xTx1x1 tensor.
        condition_video_input_mask will be concat with the input for the network.
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype

        if condition_video_indicator is None:
            condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(latent_dtype)  # 1 for condition region

            if self.config.conditioner.video_cond_bool.condition_location == "first_n":
                assert num_condition_t is not None, "num_condition_t should be provided"
                assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
                condition_video_indicator[:, :, :num_condition_t] += 1.0

            else:
                raise NotImplementedError(
                    f"condition_location {self.config.conditioner.video_cond_bool.condition_location} not implemented; training={self.training}"
                )

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        if to_cp is None:
            to_cp = self.net.is_context_parallel_enabled
        assert condition.data_type == DataType.VIDEO or not to_cp
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        guidance_other: Union[float, None] = None,
        use_gt_albedo: bool = False,
        forward_render: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.
        Different from the base model, this function support condition latent as input, it will add the condition information into the condition and uncondition object.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
        - condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
        - add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        input_key = self.input_data_key
        mode = None

        if not forward_render:
            mode = 'relight'
            raw_state = torch.cat([data_batch['rgb_ref'], data_batch['rgb_ref'] * 0., data_batch['rgb_ref'] * 0.], dim=2) # shape:[B, C, T, H, W]
            if use_gt_albedo:
                mode = 'gt_albedo'
                raw_state = torch.cat([data_batch['rgb_ref'], data_batch['rgb_ref'] * 0., data_batch['basecolor']], dim=2) # shape:[B, C, T, H, W]
        else:
            mode = 'forward'
            raw_state = torch.cat([data_batch['basecolor'] * 0., data_batch['basecolor'] * 0., data_batch['basecolor']], dim=2) # shape:[B, C, T, H, W]

        latent_state = self.encode(raw_state).contiguous()

        if mode == 'forward':
            T = latent_state.shape[2]
            latent_state[:, :, :T // 3] *= 0.

        cond_latent_shape = [latent_state.shape[0], latent_state.shape[1], latent_state.shape[2] // 3, latent_state.shape[3], latent_state.shape[4]]
        with torch.no_grad():
            latent_condition, env_latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch,
                condition_keys=self.condition_keys,
                condition_drop_rate=0,
                append_condition_mask=self.append_condition_mask,
                dtype=latent_state.dtype, device=latent_state.device,
                latent_shape=cond_latent_shape,
            )

        zero_env_latent_condition = torch.zeros_like(env_latent_condition)
        data_batch["latent_condition"] = torch.cat([zero_env_latent_condition, env_latent_condition, zero_env_latent_condition], dim=2)


        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        if mode == 'relight':
            condition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, condition, num_condition_t=latent_state.shape[2] // 3
            )
        elif mode == 'forward' or mode == 'gt_albedo':
            T = latent_state.shape[2]
            condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(latent_state.dtype)  # 1 for condition region
            condition_video_indicator[:, :, :T // 3] += 1.0
            condition_video_indicator[:, :, -T // 3:] += 1.0
            condition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, condition, condition_video_indicator=condition_video_indicator,
            )

        if self.config.conditioner.video_cond_bool.add_pose_condition:
            condition = self.add_condition_pose(data_batch, condition)

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        if mode == 'relight':
            uncondition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, uncondition, num_condition_t=latent_state.shape[2] // 3
            )
        elif mode == 'forward' or mode == 'gt_albedo':
            T = latent_state.shape[2]
            condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(latent_state.dtype)  # 1 for condition region
            condition_video_indicator[:, :, :T // 3] += 1.0
            condition_video_indicator[:, :, -T // 3:] += 1.0
            uncondition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, uncondition, condition_video_indicator=condition_video_indicator,
            )
            
        if self.config.conditioner.video_cond_bool.add_pose_condition:
            uncondition = self.add_condition_pose(data_batch, uncondition)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        if self.net.is_context_parallel_enabled:
            condition.get_condition_for_cp(cp_group=self.net.cp_group)
            uncondition.get_condition_for_cp(cp_group=self.net.cp_group)

        if guidance_other is not None:  # and guidance_other != guidance:
            import copy

            assert not parallel_state.is_initialized(), "Parallel state not supported with two guidances."
            condition_other = copy.deepcopy(uncondition)
            condition_other.trajectory = condition.trajectory

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                cond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    condition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                    use_gt_albedo=use_gt_albedo,
                ).x0_pred_replaced
                uncond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    uncondition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                    use_gt_albedo=use_gt_albedo,
                ).x0_pred_replaced
                cond_other_x0 = self.denoise(
                    noise_x,
                    sigma,
                    condition_other,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                    use_gt_albedo=use_gt_albedo,
                ).x0_pred_replaced
                return cond_x0 + guidance * (cond_x0 - uncond_x0) + guidance_other * (cond_other_x0 - uncond_x0)

        else:

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                cond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    condition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                    use_gt_albedo=use_gt_albedo,
                ).x0_pred_replaced
                uncond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    uncondition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                    use_gt_albedo=use_gt_albedo,
                ).x0_pred_replaced
                return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        guidance_other: Union[float, None] = None,
        use_gt_albedo: bool = False,
        forward_render: bool = False,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half

            add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
            else:
                log.debug(f"Default Video state shape is used. {self.state_shape}")
                state_shape = self.state_shape

        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
            guidance_other=guidance_other,
            use_gt_albedo=use_gt_albedo,
            forward_render=forward_render,
        )

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        x_sigma_max = (
            torch.randn(n_sample, *state_shape, **self.tensor_kwargs, generator=generator) * self.sde.sigma_max
        )
        if self.net.is_context_parallel_enabled:
            x_sigma_max = rearrange(x_sigma_max, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            x_sigma_max = rearrange(x_sigma_max, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        
        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)

        if self.net.is_context_parallel_enabled:
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        return samples


    @torch.no_grad()
    def validation_step(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        save generated videos
        """
        raw_data, x0, condition = self.get_data_and_condition(data)
        data = misc.to(data, **self.tensor_kwargs)
        sample = self.generate_samples_from_batch(
            data,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
        )
        sample = self.decode(sample)
        gt = raw_data
        return {"gt": gt, "result": sample,}, torch.tensor([0]).to(**self.tensor_kwargs)
    
@diffusion_fsdp_class_decorator
class FSDPDiffusionModel(MultiviewExtendDiffusionModelRelight):
    pass
