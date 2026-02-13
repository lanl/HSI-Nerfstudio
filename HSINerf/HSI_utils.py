# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from:
#   nerfstudio/nerfstudio/fields/vanilla_nerf_field.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/vanilla_nerf_field.py
#   nerfstudio/nerfstudio/model_components/renders.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/model_components/renderers.py
#
# Copyright (c) 2022-2025 The Nerfstudio Team
# Copyright (c) 2025 Los Alamos National Laboratory/Scout Jarman
#
# Prominent notice of changes (Apache-2.0 §4(b)):
#   [2025-10-03] (Scout Jarman) – Alters NeRFField for use with HSI
#       - Alters RGBRenderer for use with HSI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from typing import Literal, Optional, Tuple, Union, Type, Dict, List
from jaxtyping import Float, Int

from torch import Tensor, nn
import torch
import math
import numpy as np

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.model_components.renderers import RGBRenderer, BackgroundColor, BACKGROUND_COLOR_OVERRIDE
from nerfstudio.utils import colors
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.cameras.rays import RayBundle


"""
HSI NeRF Model Functions
"""


class HSIFieldHead(FieldHead):
    """HSI field head to facilitate Mip usage with HSI"""
    def __init__(self, out_dim=128, field_head_name='hsi', in_dim=None, activation=None):
        super().__init__(in_dim=in_dim, out_dim=out_dim, field_head_name=field_head_name, activation=activation)


class MultiChannelDensityFieldHead(FieldHead):
    """Density output as a vector, one per spectral channel."""

    def __init__(self, num_channels: int, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Softplus()):
        super().__init__(
            in_dim=in_dim,
            out_dim=num_channels,
            field_head_name=FieldHeadNames.DENSITY,
            activation=activation,
        )


class   HSINeRFField(Field):
    """NeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (HSIFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_multi_density: bool = False,
        num_channels: int = 128
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        if use_multi_density:
            self.field_output_density = MultiChannelDensityFieldHead(num_channels=num_channels, in_dim=self.mlp_base.get_out_dim())
        else:
            self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

        if field_heads:
            self.mlp_head = MLP(
                in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
                num_layers=head_mlp_num_layers,
                layer_width=head_mlp_layer_width,
                out_activation=nn.ReLU(),
            )
        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs


class Loss_with_SAM(nn.Module):
    """Adding SAM to a given loss function"""
    def __init__(self, loss, sam_weight=1.0, eps=1e-8):
        super().__init__()
        self.loss = loss
        self.sam_weight = sam_weight
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss(pred, target)

        # Spectral Angle Mapper (SAM)
        pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + self.eps)
        target_norm = target / (target.norm(dim=-1, keepdim=True) + self.eps)
        cos_sim = (pred_norm * target_norm).sum(dim=-1).clamp(-1 + self.eps, 1 - self.eps)
        sam = torch.acos(cos_sim).mean()

        return loss + self.sam_weight * sam
    

class HSIRenderer(RGBRenderer):
    def __init__(self, background_color = "random"):
        super().__init__(background_color)

    @classmethod
    def get_background_color(
        cls, background_color: BackgroundColor, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        assert background_color not in {"last_sample", "random"}
        # assert shape[-1] == 3, "Background color must be RGB."
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            # background_color = colors.COLORS_DICT[background_color]
            # Hard code to be black
            background_color = torch.zeros((shape[1]))
        assert isinstance(background_color, Tensor)

        # Ensure correct shape
        return background_color.expand(shape).to(device)
    
    def blend_background(self, image, background_color = None):
        """ I presume we can just ignore alpha opacity for HSI stuff"""
        return image
    
    @classmethod
    def combine_rgb(
        cls,
        rgb: Float[Tensor, "*bs num_samples C"],
        weights: Float[Tensor, "*bs num_samples C"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs C"]:
        """
        Supports arbitrary channel count (C) for hyperspectral rendering.

        Args:
            rgb: [..., num_samples, C]
            weights: [..., num_samples, C]
        Returns:
            [..., C]
        """
        if ray_indices is not None and num_rays is not None:
            # Packed rays not yet supported for multi-channel rendering
            raise NotImplementedError("Packed sample support for multi-channel HSI not implemented.")
        
        # Composite image by summing over samples
        comp_rgb = torch.sum(weights * rgb, dim=-2)  # [..., C]
        accumulated_weight = torch.sum(weights, dim=-2)  # [..., C]

        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE

        if isinstance(background_color, str) and background_color == "random":
            return comp_rgb  # no background blending
        elif background_color == "last_sample":
            background_color = rgb[..., -1, :]  # [..., C]

        # Get proper background tensor
        background_color = cls.get_background_color(background_color, shape=comp_rgb.shape, device=comp_rgb.device)

        # Alpha blending
        comp_rgb = comp_rgb + background_color * (1.0 - accumulated_weight)
        return comp_rgb

    def forward(self, rgb, weights, ray_indices = None, num_rays = None, background_color = None):
        """ Looks like outside of training all values are clamped between 0 and 1, so got rid of it"""
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            rgb = torch.nan_to_num(rgb)

        # Support multi-channel density
        if weights.ndim == 2:
            # [num_rays, num_samples]
            weights = weights.unsqueeze(-1)  # → [num_rays, num_samples, 1]

        rgb = self.combine_rgb(rgb, weights, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays)

        return rgb
    
    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Override for HSI: Skip background blending.
        Assumes pred_image and gt_image already have compatible shapes and radiometric meaning.
        """
        return pred_image, gt_image


class HSIAccumulationRenderer(nn.Module):
    """Accumulated value along a ray."""

    @classmethod
    def forward(
        cls,
        weights: Float[Tensor, "*bs num_samples C"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs C"]:
        """
        Args:
            weights: [..., num_samples] or [..., num_samples, C]
        Returns:
            Accumulated weights: [..., C]
        """
        if weights.ndim == 2:
            weights = weights.unsqueeze(-1)  # [R, S] -> [R, S, 1]

        if ray_indices is not None and num_rays is not None:
            # Packed not yet supported for HSI
            raise NotImplementedError("Packed accumulation not implemented for multi-channel")
        
        return torch.sum(weights, dim=-2)  # Sum over samples axis


class HSIDepthRenderer(nn.Module):
    """Calculate depth along ray."""

    def __init__(self, method: Literal["median", "expected"] = "median") -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples C"],  # could be [R, S] or [R, S, C]
        ray_samples: RaySamples,
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*batch C"]:
        """Composite samples along ray and calculate depths."""
        if self.method == "median":
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2  # [R, S, 1]

            if ray_indices is not None and num_rays is not None:
                raise NotImplementedError("Packed median not supported yet.")

            if weights.ndim == 2:
                weights = weights.unsqueeze(-1)  # [R, S] → [R, S, 1]

            cum_weights = torch.cumsum(weights, dim=-2)  # [R, S, C]
            threshold = 0.5 * torch.ones_like(cum_weights[:, :1, :])  # [R, 1, C]

            # Flatten for searchsorted compatibility (across batch * channels)
            R, S, C = cum_weights.shape
            cum_flat = cum_weights.transpose(1, 2).reshape(R * C, S)      # [R*C, S]
            thresh_flat = threshold.transpose(1, 2).reshape(R * C, 1)     # [R*C, 1]
            idx_flat = torch.searchsorted(cum_flat, thresh_flat, side="left")  # [R*C, 1]
            idx = idx_flat.view(R, C).unsqueeze(1)  # [R, 1, C]

            # Clamp and gather
            idx = torch.clamp(idx, 0, S - 1)
            steps = steps.expand(-1, -1, weights.shape[-1])  # [R, S, C]
            median_depth = torch.gather(steps, dim=1, index=idx).squeeze(1)  # [R, C]
            return median_depth

        elif self.method == "expected":
            eps = 1e-10
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2  # [R, S, 1]

            if ray_indices is not None and num_rays is not None:
                raise NotImplementedError("Packed expected not yet supported.")

            if weights.ndim == 2:
                weights = weights.unsqueeze(-1)  # [R, S]

            depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, dim=-2) + eps)
            return torch.clip(depth, steps.min(), steps.max())

        raise NotImplementedError(f"Depth method {self.method} not supported")


def get_weights_multi_channel(ray_samples: RaySamples, densities: torch.Tensor, n_channels: int) -> torch.Tensor:
    """
    Args:
        ray_samples: RaySamples object, shape [R, S]
        densities: [R, S, C] already
        n_channels: just for interface
    """
    assert densities.ndim == 3, f"Densities should be [R, S, C], got {densities.shape}"
    assert ray_samples.deltas.shape[:2] == densities.shape[:2], \
        f"Mismatch: deltas {ray_samples.deltas.shape}, densities {densities.shape}"

    deltas = ray_samples.deltas

    # print(densities.shape, deltas.shape)
    alpha = 1.0 - torch.exp(-densities * deltas)  # [R, S, C]

    # transmittance
    shifted_alpha = torch.cat([torch.zeros_like(alpha[:, :1, :]), alpha[:, :-1, :]], dim=1)
    transmittance = torch.cumprod(1.0 - shifted_alpha + 1e-10, dim=1)

    weights = alpha * transmittance  # [R, S, C]
    return weights


def get_annealed_near_far(
    nears: Tensor,
    fars: Tensor,
    step: int,
    total_anneal_steps: int,
    mid_perc: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """
    Linearly anneals near and far bounds between a midpoint and original values over training steps.

    Args:
        nears: Original near bounds per ray. [R]
        fars: Original far bounds per ray. [R]
        step: Current training step.
        total_anneal_steps: Number of steps to complete annealing.
        mid_perc: Midpoint percentage between near and far (0.5 = center).

    Returns:
        Tuple of new_nears and new_fars tensors.
    """
    mid = nears + mid_perc * (fars - nears)
    alpha = min(step / total_anneal_steps, 1.0)
    new_nears = mid + alpha * (nears - mid)
    new_fars = mid + alpha * (fars - mid)
    return new_nears, new_fars


def compute_patch_tv_loss(
    ray_bundle: RayBundle,
    model_outputs: dict,
    depth_key: str = "depth_fine",
    weight: float = 1.0
) -> torch.Tensor:
    """
    Computes total variation (TV) loss on a depth patch.

    Args:
        ray_bundle: The original (unflattened) RayBundle of shape [H, W]
        model_outputs: Output dictionary from model.forward() on ray_bundle.flatten()
        depth_key: Key in model_outputs containing the predicted depth
        weight: Scaling factor (lambda_geometry)

    Returns:
        Scaled TV loss as a torch scalar
    """
    H, W = ray_bundle.directions.shape[:2]

    depth = model_outputs[depth_key]
    if depth.dim() == 2 and depth.shape[1] == 1:
        depth = depth.squeeze(-1)  # [N, 1] → [N]
    elif depth.dim() == 1:
        pass  # [N] → fine
    else:
        raise ValueError(f"Unsupported depth shape {depth.shape}")

    if depth.numel() != H * W:
        raise ValueError(f"Depth size {depth.shape} does not match expected patch size {H}x{W}")

    depth = depth.view(H, W)

    # Total variation
    tv_loss = (
        torch.mean(torch.abs(depth[:-1, :] - depth[1:, :])) +
        torch.mean(torch.abs(depth[:, :-1] - depth[:, 1:]))
    )
    return tv_loss * weight


def compute_batch_tv_loss(
    patch_ray_bundles: List[RayBundle],
    model_outputs: dict,
    patch_size: int,
    weight: float = 1.0,
    depth_key: str = "depth_fine"
) -> torch.Tensor:
    """
    Computes batched TV loss for a list of flattened RayBundles.
    Handles both single-channel and multi-channel depth outputs.
    """
    num_patches = len(patch_ray_bundles)
    H, W = patch_size, patch_size

    # === Step 1: Get and shape depth ===
    depth = model_outputs[depth_key]  # shape [N], [N, 1], or [N, C]

    if depth.ndim == 2 and depth.shape[1] > 1:
        # Multi-channel: sum or average over channels
        depth = depth.mean(dim=-1)  # or .mean(dim=-1) if preferred
    elif depth.ndim == 2:
        depth = depth.squeeze(-1)  # [N, 1] → [N]
    elif depth.ndim != 1:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    expected_pixels = num_patches * H * W
    if depth.shape[0] != expected_pixels:
        raise ValueError(f"Depth shape {depth.shape} does not match expected {expected_pixels} pixels")

    # Reshape to batch of image patches
    depth = depth.view(num_patches, H, W)

    # === Step 2: Compute TV loss over batch ===
    tv_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])
    tv_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
    tv_loss = (tv_y.mean(dim=(1, 2)) + tv_x.mean(dim=(1, 2))).mean()

    return tv_loss * weight


def concat_ray_bundles(bundles: List[RayBundle]) -> RayBundle:
    """Concatenate a list of flattened RayBundles into a single RayBundle."""

    def stack_attr(attr_name):
        vals = [getattr(rb, attr_name) for rb in bundles if getattr(rb, attr_name) is not None]
        return torch.cat(vals, dim=0) if vals else None

    return RayBundle(
        origins=stack_attr("origins"),
        directions=stack_attr("directions"),
        pixel_area=stack_attr("pixel_area"),
        camera_indices=stack_attr("camera_indices"),
        nears=stack_attr("nears"),
        fars=stack_attr("fars"),
        metadata={},  # ignore per-ray metadata for now
        times=stack_attr("times"),
    )



def schedule_geometry_weight(
    step: int,
    start: int,
    end: int,
    min_weight: float,
    max_weight: float,
    mode: str = "linear",
    reverse: bool = True  # <== NEW FLAG
) -> float:
    """Schedule for geometry regularization weight."""
    if step < start:
        return max_weight if reverse else 0.0
    elif step >= end:
        return min_weight if reverse else max_weight

    alpha = (step - start) / (end - start)

    if reverse:
        alpha = 1.0 - alpha

    if mode == "linear":
        return min_weight + alpha * (max_weight - min_weight)
    elif mode == "log":
        return min_weight * ((max_weight / min_weight) ** alpha)
    else:
        raise ValueError(f"Unknown schedule type: {mode}")


"""


Miscelanous functions


"""

def apply_depth_colormap(
    depth: Float[Tensor, "*bs 1"],
    accumulation: Optional[Float[Tensor, "*bs 1"]] = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    colormap_options: ColormapOptions = ColormapOptions(),
) -> Float[Tensor, "*bs rgb=3"]:
    """Safe multi-shape depth-to-color conversion."""

    # Normalize depth to [0, 1]
    near_plane = near_plane if near_plane is not None else float(torch.min(depth))
    far_plane = far_plane if far_plane is not None else float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    # Ensure [H, W] format (2D) for apply_colormap
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    elif depth.ndim == 3 and depth.shape[0] == 1:  # [1, H, W]
        depth = depth.squeeze(0)

    assert depth.ndim == 2, f"Depth must be [H, W], got {depth.shape}"

    # Apply colormap: [H, W] → [H, W, 3]
    colored_image = apply_colormap(depth, colormap_options=colormap_options)

    if accumulation is not None:
        if accumulation.ndim == 3 and accumulation.shape[-1] == 1:
            accumulation = accumulation.squeeze(-1)
        assert accumulation.shape == depth.shape, f"Accum shape mismatch: {accumulation.shape} vs {depth.shape}"
        accumulation = accumulation.unsqueeze(-1)  # [H, W, 1]
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image



def nanstd_mean(x):
    valid = ~torch.isnan(x)
    if valid.sum() == 0:
        return torch.tensor(float('nan')), torch.tensor(float('nan'))
    return torch.std_mean(x[valid])




def ACE_det(img, gt, mean=None, cov=None, regularize_eps=1e-6):
    """
    Compute ACE detection score map.

    Args:
        img: torch.Tensor of shape [H, W, B] — expected in flicks
        gt: numpy array of shape [B,] — target signature (also in flicks)
        mean: optional precomputed background mean (in flicks)
        cov: optional precomputed covariance (in flicks)
        regularize_eps: epsilon for numerical stability
    """
    img_tmp = img.clone().detach()
    H, W, B = img_tmp.shape

    # === Convert to numpy
    pixels = img_tmp.reshape(-1, B).cpu().numpy().astype(np.float32)

    # === Heuristic: detect if values are in microflicks
    max_val = np.percentile(pixels, 99)
    if max_val > 100:  # threshold for microflicks (very conservative)
        pixels = pixels / 1e6
        if mean is not None:
            mean = mean / 1e6
        if cov is not None:
            cov = cov / (1e6 ** 2)

    # === Center the pixels
    if mean is None:
        mean = pixels.mean(axis=0)
    pixels -= mean

    # === Covariance and inverse
    if cov is None:
        cov = np.cov(pixels.T)
    cov += regularize_eps * np.eye(cov.shape[0])
    C_hat = np.linalg.inv(cov)

    # === ACE computation
    num = gt.T @ C_hat @ pixels.T                     # shape: (N,)
    den1 = gt.T @ C_hat @ gt                          # scalar
    den2 = np.sum((pixels @ C_hat) * pixels, axis=1)  # shape: (N,)
    ace = (num ** 2) / (den1 * den2 + 1e-12)

    return ace.reshape(H, W)



def get_train_eval_split_fraction_random(
    image_filenames: List,
    train_split_fraction: Union[float, int],
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    num_images = len(image_filenames)
    
    if train_split_fraction > 1:
        num_train_images = int(min(train_split_fraction, num_images))
    else:
        num_train_images = math.ceil(num_images * train_split_fraction)

    local_rng = np.random.default_rng(seed)
    i_all = np.arange(num_images)
    local_rng.shuffle(i_all)

    i_train = i_all[:num_train_images]
    i_eval = i_all[num_train_images:]
    return i_train, i_eval
 

 
def get_train_eval_split_fixed_eval_random_train(
    image_filenames: List,
    train_split_fraction: Union[float, int] = 1.0,
    seed: Optional[int] = None,
    fixed_eval_fraction: Union[float, int] = 0.3,
    fixed_eval_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image indices into a fixed evaluation set and a random training set (from remaining views).

    Args:
        image_filenames: List of image file paths.
        train_split_fraction: Fraction or count of remaining images (after eval) to use for training.
        seed: Random seed for training set selection.
        fixed_eval_fraction: Fraction or count of total data to use for evaluation.
        fixed_eval_seed: Random seed for evaluation set selection.

    Returns:
        i_train: Indices used for training (random subset of non-eval set).
        i_eval: Indices used for evaluation (fixed across runs).
    """
    num_images = len(image_filenames)
    all_indices = np.arange(num_images)

    # --- Step 1: Fixed evaluation set ---
    if fixed_eval_fraction > 1:
        num_eval_images = int(min(fixed_eval_fraction, num_images))
    else:
        num_eval_images = math.ceil(num_images * fixed_eval_fraction)

    fixed_rng = np.random.default_rng(fixed_eval_seed)
    i_eval = fixed_rng.choice(all_indices, size=num_eval_images, replace=False)

    # --- Step 2: Training set from remaining pool ---
    i_remaining = np.setdiff1d(all_indices, i_eval)
    num_remaining = len(i_remaining)

    if train_split_fraction > 1:
        num_train_images = int(min(train_split_fraction, num_remaining))
    else:
        num_train_images = math.ceil(num_remaining * train_split_fraction)

    train_rng = np.random.default_rng(seed)
    i_train = train_rng.choice(i_remaining, size=num_train_images, replace=False)

    return i_train, i_eval



def get_train_eval_split_farthest_subset(
    image_filenames: List[str],
    poses: Union[List[List[float]], np.ndarray],
    train_fraction: Union[float, int] = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    Splits images into train/eval sets using farthest-point sampling on camera positions.
    
    Args:
        image_filenames: List of all image filenames.
        poses: List or array of (N, 4, 4) camera-to-world transformation matrices.
        train_fraction: Fraction or count of data to use for training.
        seed: Optional random seed.

    Returns:
        Tuple of (i_train, i_eval): Lists of indices for train and eval sets.
    """
    if seed is not None:
        np.random.seed(seed)

    poses = np.array(poses)  # 🔧 convert to ndarray if it's a list

    num_images = len(image_filenames)

    if train_fraction > 1:
        num_train = int(min(train_fraction, num_images))
    else:
        num_train = int(train_fraction * num_images)

    positions = poses[:, :3, 3]  # (N, 3)
    anchor_idx = np.random.randint(num_images)
    selected = [anchor_idx]
    remaining = set(range(num_images)) - {anchor_idx}

    for _ in range(num_train - 1):  # already selected anchor
        max_dist = -np.inf
        farthest_idx = -1
        for idx in remaining:
            pt = positions[idx]
            min_dist = np.min([np.linalg.norm(pt - positions[j]) for j in selected])
            if min_dist > max_dist:
                max_dist = min_dist
                farthest_idx = idx
        selected.append(farthest_idx)
        remaining.remove(farthest_idx)

    i_train = selected
    i_eval = list(remaining)

    return i_train, i_eval



def get_train_eval_split_farthest_biased(
    image_filenames: List[str],
    poses: Union[List[List[float]], np.ndarray],
    train_fraction: Union[int, float] = 0.8,
    top_half_indices: Union[List[int], range] = range(119),
    top_fraction: float = 0.6,
    seed: Optional[int] = None,
    fixed_eval_fraction: Union[int, float] = -1,
    fixed_eval_seed: int = 0,
) -> Tuple[List[int], List[int]]:
    if seed is not None:
        np.random.seed(seed)

    poses = np.array(poses)
    N = len(image_filenames)
    all_indices = np.arange(N)

    # === Fixed Evaluation Strategy ===
    if N == 1101:
        i_eval = list(range(1000, 1101))
        sample_indices = list(range(1000))  # train only from first 1000
        top_half_indices = range(300)
    else:
        # Random fixed evaluation set (if fixed_eval_fraction > 0)
        if fixed_eval_fraction > 0:
            if fixed_eval_fraction > 1:
                num_eval = int(min(fixed_eval_fraction, N))
            else:
                num_eval = math.ceil(N * fixed_eval_fraction)
            # print("num_eval", num_eval)
            eval_rng = np.random.default_rng(fixed_eval_seed)
            i_eval = eval_rng.choice(all_indices, size=num_eval, replace=False).tolist()
        else:
            i_eval = []

        sample_indices = sorted(list(set(all_indices) - set(i_eval)))

    # === Prepare training pool ===
    sample_positions = poses[:, :3, 3]
    sample_positions = sample_positions[sample_indices]

    if train_fraction > 1:
        num_train = int(min(train_fraction, len(sample_indices)))
    else:
        num_train = int(train_fraction * len(sample_indices))
    # print("num_train", num_train)

    top_half_set = set(top_half_indices).intersection(sample_indices)
    bottom_half_set = set(sample_indices) - top_half_set

    num_top = int(np.ceil(num_train * top_fraction))
    num_rest = num_train - num_top

    # --- Stage 1: FPS on top half ---
    top_half_list = sorted(list(top_half_set))
    selected_top = []
    if top_half_list:
        anchor_top = np.random.choice(top_half_list)
        selected_top = [anchor_top]
        remaining_top = set(top_half_list) - {anchor_top}

        for _ in range(num_top - 1):
            if not remaining_top:
                break
            max_dist = -np.inf
            farthest_idx = None
            for idx in remaining_top:
                pt = sample_positions[sample_indices.index(idx)]
                min_dist = np.min([np.linalg.norm(pt - sample_positions[sample_indices.index(j)]) for j in selected_top])
                if min_dist > max_dist:
                    max_dist = min_dist
                    farthest_idx = idx
            if farthest_idx is not None:
                selected_top.append(farthest_idx)
                remaining_top.remove(farthest_idx)

    # --- Stage 2: FPS on bottom, conditioned on top ---
    selected = selected_top.copy()
    remaining_rest = bottom_half_set - set(selected)

    for _ in range(num_rest):
        if not remaining_rest:
            break
        max_dist = -np.inf
        farthest_idx = None
        for idx in remaining_rest:
            pt = sample_positions[sample_indices.index(idx)]
            min_dist = np.min([np.linalg.norm(pt - sample_positions[sample_indices.index(j)]) for j in selected])
            if min_dist > max_dist:
                max_dist = min_dist
                farthest_idx = idx
        if farthest_idx is not None:
            selected.append(farthest_idx)
            remaining_rest.remove(farthest_idx)

    i_train = selected
    if fixed_eval_fraction < 0:
        i_eval = sorted(list(set(all_indices) - set(i_train)))
    return i_train, i_eval




def inverse_standardize(tensor, standardizer):
    """
    tensor: torch.Tensor of shape [1, C, H, W] or [C, H, W]
    standardizer: sklearn StandardScaler with .mean_ and .scale_
    """
    if isinstance(tensor, torch.Tensor):
        mean = torch.tensor(standardizer.mean_, dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(standardizer.scale_, dtype=tensor.dtype, device=tensor.device)
        tensor = tensor * std[None, :, None, None] + mean[None, :, None, None]
        return tensor
    else:
        raise TypeError("Input must be a torch.Tensor")



class HyperspectralUtils:
    @staticmethod
    def hs2rgb_falsecolor(hs_image, band_indices=[59, 7, 15], min_value=0.0001707731, max_value=0.0019809046):
        """
        Convert hyperspectral data to false-color RGB using selected bands.
        
        Args:
            hs_image (Tensor): Hyperspectral image tensor with shape [H, W, C].
            band_indices (List[int]): Indices of the hyperspectral bands to use for R, G, and B.
            min_value (float): Minimum value for normalization.
            max_value (float): Maximum value for normalization.
        
        Returns:
            Tensor: False-color RGB image tensor with shape [H, W, 3].
        """
        # Normalize the hyperspectral image
        normalized_image = (hs_image - min_value) / (max_value - min_value)
        normalized_image = normalized_image.clamp(0, 1)  # Ensure values are in [0, 1]

        # Select bands for false-color RGB
        rgb_image = normalized_image[..., band_indices]
        return rgb_image
    
    def hs2rgb_normalized(hs_image, band_indicies=[59, 7, 15]):
        norm_img = (hs_image - hs_image.min()) / (hs_image.max() - hs_image.min())
        norm_img = norm_img.clamp(0, 1)
        return norm_img[band_indicies, :, :]

    def hs2rgb_grayscale(hs_image, band=60):
        tmp_img = hs_image[..., band]
        tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min())
        tmp_img = (255 * tmp_img).clamp(0, 255)
        return tmp_img.unsqueeze(-1)
    
    def hs2rgb_pca(hs_image, pca_comps):
        h, w, c = hs_image.shape
        img_flat = hs_image.view(-1, c)

        img_flat = torch.matmul(img_flat, pca_comps.T)
        img_rgb = img_flat.view(h, w, 3)

        return img_rgb
    


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss across channels."""

    def __init__(self, weights: np.ndarray):
        super().__init__()
        # Convert to tensor and register as buffer
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input, target shape: (B, C, H, W)  (after renderer output)
        diff = input - target
        squared = diff ** 2
        weighted = squared * self.weights.view(1, -1, 1, 1)  # broadcast properly
        return weighted.mean()



class AdaptiveWeightedMSELoss(nn.Module):
    """Adaptive weighted MSE loss for per-ray spectral vectors of shape (N_rays, C)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("weights", torch.ones(num_channels, dtype=torch.float32) / num_channels)

    def update_weights(self, residuals: torch.Tensor):
        """Update weights using per-channel residuals."""
        total = residuals.sum()
        if total > 0:
            new_weights = residuals / total
        else:
            new_weights = torch.ones_like(residuals)

        self.weights.copy_(new_weights.detach().float())

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input, target: shape [N_rays, C]
        # weights: shape [C]
        diff = input - target                   # shape [N_rays, C]
        squared = diff ** 2                     # shape [N_rays, C]
        weighted = squared * self.weights       # broadcast over rays
        return weighted.mean()



