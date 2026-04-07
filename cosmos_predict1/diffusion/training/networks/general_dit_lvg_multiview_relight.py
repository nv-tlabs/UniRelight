# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torchvision import transforms

from cosmos_predict1.diffusion.conditioner import DataType
from cosmos_predict1.diffusion.training.context_parallel import split_inputs_cp
from cosmos_predict1.diffusion.training.networks.general_dit_multiview import MultiviewGeneralDIT
from cosmos_predict1.utils import log

from cosmos_predict1.diffusion.training.module.blocks import PatchEmbed


class VideoExtendMultiviewGeneralDITRelight(MultiviewGeneralDIT):
    def __init__(
        self, 
        *args, 
        in_channels,
        additional_concat_ch: int = None,
        **kwargs, 
    ):

        self.additional_concat_ch = additional_concat_ch

        # extra channel for video condition mask
        super().__init__(*args, in_channels=in_channels + 1, **kwargs)
        log.info(f"VideoExtendGeneralDITRelight in_channels: {in_channels + 1}")

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
            view_condition_dim,
            traj_condition_dim,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
            self.view_condition_dim,
            self.traj_condition_dim,
        )

        in_channels = in_channels + self.additional_concat_ch

        in_channels = in_channels + 1 if concat_padding_mask else in_channels

        if self.concat_view_embedding:
            in_channels = in_channels + view_condition_dim if view_condition_dim > 0 else in_channels

        if self.concat_traj_embedding:
            in_channels = in_channels + traj_condition_dim if traj_condition_dim > 0 else in_channels

        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
            keep_spatio=True,
            legacy_patch_emb=self.legacy_patch_emb,
        )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if self.legacy_patch_emb:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the `self.pos_embedder`
                with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """

        ##### Custimized part #####
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, latent_condition], dim=1)
        ###########################

        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        view_indices = torch.arange(self.n_views).to(x_B_C_T_H_W.device)  # View indices [0, 1, ..., V-1]
        view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
        view_embedding = rearrange(view_embedding, "V D -> D V")
        view_embedding = view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # Shape: [1, D, V, 1, 1, 1]

        if self.add_repeat_frame_embedding:
            if frame_repeat is None:
                frame_repeat = (
                    torch.zeros([x_B_C_T_H_W.shape[0], view_embedding.shape[1]])
                    .to(view_embedding.device)
                    .to(view_embedding.dtype)
                )
            frame_repeat_embedding = self.repeat_frame_embedding(frame_repeat.unsqueeze(-1))
            frame_repeat_embedding = rearrange(frame_repeat_embedding, "B V D -> B D V")
            view_embedding = view_embedding + frame_repeat_embedding.unsqueeze(3).unsqueeze(4).unsqueeze(5)

        x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=self.n_views)
        view_embedding = view_embedding.expand(
            x_B_C_V_T_H_W.shape[0],
            view_embedding.shape[1],
            view_embedding.shape[2],
            x_B_C_V_T_H_W.shape[3],
            x_B_C_V_T_H_W.shape[4],
            x_B_C_V_T_H_W.shape[5],
        )  # Shape: [B, V, 3, t, H, W]
        if self.concat_traj_embedding:
            traj_emb = self.traj_embeddings(trajectory)
            traj_emb = traj_emb.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            traj_emb = traj_emb.expand(
                x_B_C_V_T_H_W.shape[0],
                traj_emb.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )  # Shape: [B, V, 3, t, H, W]

            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding, traj_emb], dim=1)
        else:
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)

        x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=self.n_views)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None, extra_pos_emb
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        condition_video_pose: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Args:
        condition_video_augment_sigma: (B) tensor of sigma value for the conditional input augmentation
        condition_video_pose: (B, 1, T, H, W) tensor of pose condition
        [Warning!] crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        """
        B, C, T, H, W = x.shape
        crossattn_emb = crossattn_emb.repeat(1, self.n_views, 1)
        
        if data_type == DataType.VIDEO:
            assert (
                condition_video_input_mask is not None
            ), "condition_video_input_mask is required for video data type; check if your model_obj is extend_model.FSDPDiffusionModel or the base DiffusionModel"
            if self.cp_group is not None:
                condition_video_input_mask = rearrange(
                    condition_video_input_mask, "B C (V T) H W -> B C V T H W", V=self.n_views
                )
                condition_video_input_mask = split_inputs_cp(
                    condition_video_input_mask, seq_dim=3, cp_group=self.cp_group
                )
                condition_video_input_mask = rearrange(
                    condition_video_input_mask, "B C V T H W -> B C (V T) H W", V=self.n_views
                )
                
        input_list = [x, condition_video_input_mask]
        if condition_video_pose is not None:
            if condition_video_pose.shape[2] > T:
                log.warning(
                    f"condition_video_pose has more frames than the input video: {condition_video_pose.shape} > {x.shape}"
                )
                condition_video_pose = condition_video_pose[:, :, :T, :, :].contiguous()
            input_list.append(condition_video_pose)
        x = torch.cat(
            input_list,
            dim=1,
        )

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )
