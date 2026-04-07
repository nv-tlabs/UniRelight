# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Any, List

import attrs
from hydra.core.config_store import ConfigStore

from cosmos_predict1.diffusion.training.config.base.model import MultiviewModelConfigRelight
from cosmos_predict1.diffusion.training.config.text2world.registry import (
    register_configs as register_configs_text2world,
)
from cosmos_predict1.diffusion.training.models.model import DiffusionModel
from cosmos_predict1.utils import config
from cosmos_predict1.utils.config_helper import import_all_modules_from_package
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.utils.trainer import Trainer

from cosmos_predict1.diffusion.config.base.conditioner import VideoUnirelightConditionerConfig

@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": None},
            {"net": None},
            {"conditioner": "add_fps_image_size_padding_mask"},
            {"fsdp": None},
            {"ema": "power"},
            {"vae": "vae1"},
            {"checkpoint": "pbss"},
            {"ckpt_klass": "fsdp"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(DiffusionModel)(
        config=PLACEHOLDER,
    )

def unirelight_register_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_unirelight_cond",
        node=VideoUnirelightConditionerConfig,
    )

def make_config():

    c = Config(
        model=MultiviewModelConfigRelight(),
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_predict1"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    # c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 10
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    c.checkpoint = None

    # Call this function to register config groups.
    register_configs_text2world()

    unirelight_register_configs()
    

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos_predict1.diffusion.training.config.text2world", reload=True)
    import_all_modules_from_package("cosmos_predict1.diffusion.training.config.video2world_relight", reload=True)

    return c
