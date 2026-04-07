# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import random


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, ignore_keys=None):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """

    batched = {key: [] for key in samples[0]}
    # assert isinstance(samples[0][first_key], (list, tuple)), type(samples[first_key])

    for s in samples:
        [batched[key].append(s[key]) for key in batched if not (ignore_keys is not None and key in ignore_keys)]

    result = {}
    for key in batched:
        if ignore_keys and key in ignore_keys:
            continue
        try:
            if isinstance(batched[key][0], bool):
                assert key == "is_preprocessed"
                result[key] = batched[key][0]  # this is a hack to align with cosmos data
            elif isinstance(batched[key][0], (int, float)):
                if combine_scalars:
                    result[key] = torch.from_numpy(np.array(list(batched[key])))
            elif isinstance(batched[key][0], torch.Tensor):
                if combine_tensors:
                    result[key] = torch.stack(list(batched[key]))
            elif isinstance(batched[key][0], np.ndarray):
                if combine_tensors:
                    result[key] = np.array(list(batched[key]))
            elif isinstance(batched[key][0], list) and isinstance(batched[key][0][0], int):
                result[key] = [torch.Tensor(elems).long() for elems in zip(*batched[key])]
            else:
                result[key] = list(batched[key])
        except Exception as e:
            print(key)
            raise e
        # result.append(b)
    del batched
    return result


def dict_collation_fn_concat(
    samples, combine_tensors=True, combine_scalars=True, ignore_keys=None
):
    """
    Take a list of samples (as dictionary) and create a batch by concatenating values, preserving the keys.

    :param samples: list of samples
    :param combine_tensors: whether to concatenate lists of tensors
    :param combine_scalars: whether to concatenate lists of scalars (int/float)
    :param ignore_keys: list of keys to ignore while batching
    :returns: single sample consisting of a batch
    :rtype: dict
    """

    if not samples:
        raise ValueError("The samples list is empty.")

    if not isinstance(samples[0], dict):
        raise ValueError("Each sample must be a dictionary.")

    # Initialize batched dictionary
    batched = {key: [] for key in samples[0]}

    for s in samples:
        for key in batched:
            if ignore_keys is not None and key in ignore_keys:
                continue
            batched[key].append(s[key])

    result = {}
    for key, values in batched.items():
        if ignore_keys and key in ignore_keys:
            continue

        try:
            if isinstance(batched[key][0], bool):
                assert key == "is_preprocessed"
                result[key] = batched[key][0]  # this is a hack to align with cosmos data
            elif isinstance(values[0], torch.Tensor):
                if combine_tensors:
                    result[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], np.ndarray):
                if combine_tensors:
                    result[key] = np.concatenate(values, axis=0)
            elif isinstance(values[0], (int, float)):
                if combine_scalars:
                    result[key] = torch.tensor(values, dtype=torch.float if isinstance(values[0], float) else torch.long)
            elif isinstance(values[0], list):
                result[key] = [item for sublist in values for item in sublist]  # Flatten list of lists
            else:
                # For unsupported types, retain as a list
                result[key] = values
        except Exception as e:
            print(f"Error processing key: {key}")
            raise e

    del batched
    return result


def sample_continuous_keys(
    image_keys,
    depth_keys,
    normal_keys,
    ftex_keys,
    pose,
    N,
    stride=1,
    video_flip=False,
    start_index=None,
    flip_threshold=0.5,
    wrap_video=False
):
    total_frames = len(image_keys)
    if total_frames < N * stride:
        start_index = 0
        # sample_idx = list(range(0, min(N, total_frames)))
        sample_idx = list(range(0, total_frames, stride))
        if len(sample_idx) < N:
            for _ in range(N - len(sample_idx)):
                sample_idx.append(total_frames - 1)
    else:
        if start_index is None:
            if wrap_video:
                start_index = random.randint(0, total_frames)
                sample_idx = list(s % N for s in range(start_index, start_index + N * stride, stride))
            else:
                start_index = random.randint(0, total_frames - N * stride)
                sample_idx = list(range(start_index, start_index + N * stride, stride))
        else:
            if wrap_video:
                sample_idx = list(s % N for s in range(start_index, start_index + N * stride, stride))
            else:
                sample_idx = list(range(start_index, start_index + N * stride, stride))
            # assert False, "Start_index error in sample_continuous_keys"

    if video_flip:
        if random.random() > flip_threshold:
            sample_idx = sample_idx[::-1]

    sampled_image_keys  = [image_keys[idx] for idx in sample_idx]
    sampled_depth_keys  = [depth_keys[idx] for idx in sample_idx]
    sampled_normal_keys = [normal_keys[idx] for idx in sample_idx]
    sampled_ftex_keys   = [ftex_keys[idx] for idx in sample_idx]

    sampled_w2c = pose[sample_idx]
    return sampled_image_keys, sampled_depth_keys, sampled_normal_keys, sampled_ftex_keys, sampled_w2c, start_index, sample_idx


