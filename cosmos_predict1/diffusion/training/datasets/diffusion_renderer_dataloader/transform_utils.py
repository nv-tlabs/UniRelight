# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Callable, Dict, List, Optional, Union
from PIL import Image
import imageio.v3 as imageio

import numpy as np
import torch
import torch.nn.functional as F


def prepare_images_np(paths, use_grayscale=False, mask_np=None, bg_color=(1., 1., 1.)):
    """
    Loads and pre-processes images from a list of file paths.

    Args:
        paths (list of str): List of image file paths.
        use_grayscale (bool): If True, convert images to grayscale (by duplicating the first channel).
        mask_np (np.ndarray, optional): Mask to use in place of an alpha channel.
        bg_color (list or np.ndarray, optional): Background color to use when compositing alpha.

    Returns:
        np.ndarray: A numpy array of shape [T, H, W, C] with processed images.
    """
    # Determine extension from the first path (assumes all images share the same format)
    ext = paths[0].split('.')[-1].lower()
    images = []
    for path in paths:
        if ext in ['jpg', 'jpeg', 'png']:
            img = imageio.imread(path)
        elif ext == 'exr':
            img = imageio.imread(path, flags=-1, plugin='opencv')
        else:
            img = imageio.imread(path)
        images.append(img)

    images = np.stack(images, axis=0)  # shape: [T, H, W, C]

    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0

    # If images are 2D (i.e. missing a channel dimension), add one.
    if images.ndim == 3:
        images = images[..., None]

    # If there is an alpha channel (or a separate mask is provided), composite the image onto a background.
    if images.shape[-1] == 4 or mask_np is not None:
        bg_color = np.array(bg_color).reshape(1, 1, 1, -1)
        if images.shape[-1] == 4:
            alpha = images[..., 3:4]
        else:
            alpha = mask_np[..., 0:1]
            # Resize mask if needed.
            if alpha.shape[1:3] != images.shape[1:3]:
                alpha_tensor = torch.from_numpy(alpha).permute(0, 3, 1, 2).float()  # shape: (T, C, H, W)
                alpha_tensor = F.interpolate(alpha_tensor, size=(images.shape[1], images.shape[2]), mode='nearest')
                alpha = alpha_tensor.permute(0, 2, 3, 1).numpy()
        images = images[..., :3] * alpha + bg_color * (1 - alpha)

    if use_grayscale:
        # Duplicate the first channel to create a 3-channel grayscale image.
        images = np.concatenate([images[..., :1]] * 3, axis=-1)

    return images


def prepare_images(paths,
                   use_grayscale=False,
                   mask_np=None,
                   bg_color=None,
                   resize_transform=lambda x: x,
                   crop_transform=lambda x: x,
                   normalize_cond_img=True,
                   flip_img=False,
                   flip_transform=None,
                   flip_as_normal=False):
    """
    Loads, transforms, and normalizes images from a list of file paths.

    Args:
        paths (list of str): List of image file paths.
        use_grayscale (bool): If True, convert images to grayscale.
        mask_np (np.ndarray, optional): Mask to use in place of an alpha channel.
        bg_color (list or np.ndarray, optional): Background color for compositing.
        resize_transform (callable): Function to resize images. Should accept and return a torch.Tensor.
        crop_transform (callable): Function to crop images. Should accept and return a torch.Tensor.
        normalize_cond_img (bool): If True, normalize image pixel values to [-1, 1].
        flip_img (bool): If True, perform a horizontal flip on the images.
        flip_transform (callable, optional): Function to flip images. Should accept and return a torch.Tensor.
        flip_as_normal (bool): If True, apply an additional transformation to the first channel after flipping.

    Returns:
        torch.Tensor: A tensor of shape [T, C, H, W] containing the processed images.
    """
    # First, load and pre-process the images as a numpy array.
    images_np = prepare_images_np(paths, use_grayscale, mask_np, bg_color)
    # Convert from NHWC to NCHW format.
    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float().contiguous()

    # Apply the provided transforms.
    images = resize_transform(images)
    images = crop_transform(images)

    if normalize_cond_img:
        images = images * 2.0 - 1.0  # Normalize to the range [-1, 1].

    if flip_img:
        if flip_transform is not None:
            images = flip_transform(images)
        else:
            # Default horizontal flip if no flip_transform is provided.
            images = torch.flip(images, dims=[3])
        if flip_as_normal:
            # Adjust the first channel after flipping.
            if normalize_cond_img:
                images[:, 0, ...] *= -1
            else:
                images[:, 0, ...] = 1 - images[:, 0, ...]

    return images


def convert_rgba_to_rgb_pil(image, background_color=(255, 255, 255)):
    """
    Converts an RGBA image to RGB with the specified background color.
    If the image is already in RGB mode, it is returned as is.

    Parameters:
        image (PIL.Image.Image): Input image (RGBA or RGB).
        background_color (tuple): Background color as an RGB tuple. Default is white (255, 255, 255).

    Returns:
        PIL.Image.Image: RGB image.
    """
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

    return image


# source: diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion._resize_with_antialiasing
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


# depth normalizer (source: Marigold.src.util.depth_transform)
class ScaleShiftDepthNormalizer:
    """
    Use near and far plane to linearly normalize depth,
        i.e. d' = d * s + t,
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`
    Near and far planes are determined by taking quantile values.
    """

    is_absolute = False
    far_plane_at_max = True

    def __init__(
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        # Take quantiles as min and max
        if self.min_quantile <= 0.:
            _min = torch.min(depth_linear[valid_mask])
            _max = torch.max(depth_linear[valid_mask])
        else:
            valid_depths = depth_linear[valid_mask]
            THRESHOLD = 100000
            if valid_depths.numel() > THRESHOLD:
                valid_depths = valid_depths[torch.randperm(valid_depths.numel())[:THRESHOLD]]

            _min, _max = torch.quantile(
                valid_depths,
                torch.tensor([self.min_quantile, self.max_quantile]),
            )

        # scale and shift
        depth_norm_linear = (depth_linear - _min) / (
            _max - _min
        ) * self.norm_range + self.norm_min

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        return depth_norm_linear

    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        print(f"{self.__class__} is not revertible without GT")
        return self.scale_back(depth_norm=depth_norm)

