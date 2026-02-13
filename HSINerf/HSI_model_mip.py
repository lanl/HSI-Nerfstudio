# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from:
#   nerfstudio/nerfstudio/models/mipnerf.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/mipnerf.py
#
# Copyright (c) 2022-2025 The Nerfstudio Team
# Copyright (c) 2025 Los Alamos National Laboratory/Scout Jarman
#
# Prominent notice of changes (Apache-2.0 §4(b)):
#   [2025-10-03] (Scout Jarman) – Alters MipNeRF for HSI
#       - Modifies MipNeRF for HSI, with multi-channel density and geometry regularization
#       - Adds logic for visualizing HSI, and for gas plume detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




from dataclasses import dataclass, field, replace
from typing import Literal, Tuple, Union, Type, Dict, List
from torch import nn
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

from rich.console import Console
CONSOLE = Console(width=120)


from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared, L1Loss
from nerfstudio.utils import misc
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.config_utils import to_immutable_dict

from HSINerf.HSI_utils import HSIRenderer, inverse_standardize, Loss_with_SAM, HSIFieldHead,\
    ACE_det, HSINeRFField, get_weights_multi_channel, HSIAccumulationRenderer, HSIDepthRenderer, \
    get_annealed_near_far, compute_batch_tv_loss, concat_ray_bundles, \
    schedule_geometry_weight, AdaptiveWeightedMSELoss


@dataclass
class HSIModelMipConfig(VanillaModelConfig):
    _target: Type = field(default_factory=lambda: HSIModelMip)
    """Configuration for HSI Mip Nerf model."""
    num_output_channels: int = 128
    """Number of hyperspectral channel outputs"""

    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0})
    num_coarse_samples: int = 64
    num_importance_samples: int = 192
    eval_num_rays_per_chunk: int = 4096
    collider_params: Dict[str, float] = to_immutable_dict({"near_plane": 0, "far_plane": 2})

    background_color: Literal['random', 'last_sample', 'black', 'white'] = "last_sample"
    """Setting to last_sample to be consistent with nerfacto"""
    """number of layers for the color MLP"""
    loss: Literal["mse", "l1"] = "mse"
    """If you want to train using MSE or L1/SAM"""
    lambda_sam: float = 2
    """What weight for SAM"""
    base_mlp_num_layers: int = 8
    """See NeRFField"""
    base_mlp_layer_width: int = 256
    """See NeRFField"""
    head_mlp_num_layers: int = 2
    """See NeRFField"""
    head_mlp_layer_width: int = 128
    """See NeRFField"""
    integrated_encoding: bool = True
    """If you want to use integrated encoding (the MiP nerf conical encoding)"""
    use_multi_channel_density: bool = False
    """If you want to use multi-channel density"""
    use_anneal_sampling: bool = False
    """If you want to ue annealed sampling from Reg-NeRF"""
    anneal_nearfar_steps: int = 2000
    """How many steps to apply annealing for"""
    anneal_mid_perc: float = .85
    """ what percentage of the annealed space to sample. .5 would start with 50% of the space being sampleable. 85% means most of the ray is sampleable. """

    use_geometry_regularization: bool = False
    lambda_geometry: float = 10
    schedule_geomreg: bool = True
    geometry_reg_schedule_start: int = 0
    geometry_reg_schedule_end: int = 6000
    geometry_reg_schedule_min: float = 1
    geometry_reg_schedule_max: float = 10000
    geometry_reg_schedule_type: Literal["linear", "log"] = "linear"

    use_weighted_mse: bool = True
    weighted_mse_eval_steps: int = 5000
    lambda_wmse: float = 0
    schedule_wmse: bool = True
    wmse_schedule_start: int = 5000
    wmse_schedule_end: int = 25000
    wmse_schedule_min: float = .01
    wmse_schedule_max: float = 100
    wmse_schedule_type: Literal["linear", "log"] = "linear"

    use_density_l1: bool = False
    lambda_density_l1: float = .01
    schedule_density_l1: bool = True
    density_l1_schedule_start: int = 2500
    density_l1_schedule_end: int = 3000
    density_l1_schedule_min: float = .000001
    density_l1_schedule_max: float = .001
    density_l1_geomreg_scale: float = 1

class HSIModelMip(MipNerfModel):
    config: HSIModelMipConfig

    def __init__(self, config, scene_box, num_train_data, **kwargs):
        self.metadata = kwargs['metadata']
        super().__init__(config, scene_box=scene_box, num_train_data=num_train_data, **kwargs)


    def populate_modules(self):
        super().populate_modules()

        # Get meta data info stuff
        self.config.num_output_channels = self.metadata.get("n_components", 3)
        self.num_output_channels = self.config.num_output_channels
        self.image_mode = self.metadata['image_mode']
        self.pca = self.metadata['pca']
        self.n_channels = self.metadata["n_channels"]
        self.standardizer = self.metadata.get("standardizer", None)
        self.targ_sig = self.metadata.get("targ_sig", None)
        self.det_thresh = self.metadata.get("det_thresh")
        self.patch_size = self.metadata.get("patch_size")
        # CONSOLE.log(f"patch_size: {self.patch_size}")

        self.out_activation = nn.ReLU()  # Default out activation
        if self.image_mode == 'pca' or self.standardizer is not None:
            self.out_activation = None

        # Choose appropriate loss function
        if self.config.loss == 'mse':
            loss = MSELoss()
        else:
            loss = L1Loss()
        
        if self.config.lambda_sam > 0:
            self.rgb_loss = Loss_with_SAM(loss, self.config.lambda_sam)
        else:
            self.rgb_loss = loss

        if self.config.use_weighted_mse:
            self.wmse = AdaptiveWeightedMSELoss(self.n_channels)
            self.wmse_metrics = AdaptiveWeightedMSELoss(self.n_channels)



        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True,
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True,
        )
        


        self.field = HSINeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding, 
            use_integrated_encoding=self.config.integrated_encoding,
            field_heads=(lambda: HSIFieldHead(out_dim=self.n_channels, activation=self.out_activation),),
            base_mlp_layer_width=self.config.base_mlp_layer_width,
            base_mlp_num_layers=self.config.base_mlp_num_layers,
            head_mlp_layer_width=self.config.head_mlp_layer_width,
            head_mlp_num_layers=self.config.head_mlp_num_layers,
            use_multi_density=self.config.use_multi_channel_density,
            num_channels=self.n_channels
        )


        self.renderer_rgb = HSIRenderer(self.config.background_color)
        self.renderer_accumulation = HSIAccumulationRenderer()
        self.renderer_depth = HSIDepthRenderer()

        from torchmetrics.image import (
            PeakSignalNoiseRatio, SpectralAngleMapper,
            StructuralSimilarityIndexMeasure
        )
        self.psnr = PeakSignalNoiseRatio(data_range=1810.1315)
        self.sam = SpectralAngleMapper()
        self.ssim = StructuralSimilarityIndexMeasure()
        

        if self.image_mode == "pca":
            self.psnr_inv = PeakSignalNoiseRatio()
            self.sam_inv = SpectralAngleMapper()
            self.ssim_inv = StructuralSimilarityIndexMeasure()


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}

        # === Extract GT and prediction ===
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        pred_rgb = outputs["rgb"]

        # === PSNR
        psnr = self.psnr(pred_rgb, gt_rgb)
        metrics_dict["psnr"] = psnr.item() if isinstance(psnr, torch.Tensor) else psnr

        # === Compute L1/L2 and optionally SAM
        if isinstance(self.rgb_loss, Loss_with_SAM):
            base_loss_fn = self.rgb_loss.loss
            loss_type = "l1_loss" if isinstance(base_loss_fn, L1Loss) else "l2_loss"
            loss_val = base_loss_fn(pred_rgb, gt_rgb).item()
            metrics_dict[loss_type] = loss_val

            # Compute SAM
            pred_norm = pred_rgb / (pred_rgb.norm(dim=-1, keepdim=True) + self.rgb_loss.eps)
            target_norm = gt_rgb / (gt_rgb.norm(dim=-1, keepdim=True) + self.rgb_loss.eps)
            cos_sim = (pred_norm * target_norm).sum(dim=-1).clamp(-1 + self.rgb_loss.eps, 1 - self.rgb_loss.eps)
            sam = torch.acos(cos_sim).mean().item()
            metrics_dict["sam_loss"] = sam
        else:
            # Plain loss
            loss_type = "l1_loss" if isinstance(self.rgb_loss, L1Loss) else "l2_loss"
            metrics_dict[loss_type] = self.rgb_loss(pred_rgb, gt_rgb).item()

        # === Optionally compute depth TV loss ===
        if self.config.use_geometry_regularization and "random_ray_patches" in batch:
            patch_bundles = batch["random_ray_patches"]
            flat_bundles = [p.flatten() for p in patch_bundles]
            concat_bundle = concat_ray_bundles(flat_bundles)
            patch_outputs = self.forward(concat_bundle)
            
            tv_loss = compute_batch_tv_loss(
                patch_ray_bundles=patch_bundles,
                model_outputs=patch_outputs,
                patch_size=self.patch_size,
                weight=1.0  # no scaling in metrics
            )
            metrics_dict["geometry_tv_loss"] = tv_loss.item()

        
        if self.config.use_weighted_mse:
            # print(outputs["rgb_fine"].shape)
            wmse_loss = self.wmse_metrics(outputs["rgb_fine"], gt_rgb)
            metrics_dict["wmse"] = wmse_loss

        # if self.config.lambda_density_l1 > 0.0:
        density_fine = outputs["density"]  # shape: [R, S] or [R, S, C]
        
        
        l1_loss = torch.mean(torch.abs(density_fine))  # sum over all channels equally
        metrics_dict["density_l1"] = l1_loss


        return metrics_dict


    def forward(self, ray_bundle: Union[RayBundle, Cameras], step=np.inf) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, step)


    def get_outputs(self, ray_bundle: RayBundle, step=np.inf):
        # print("get_outputs ray_bundle", ray_bundle.shape)
        
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        # === Apply Annealed Sampling if enabled ===
        if self.config.use_anneal_sampling and self.training:
            new_nears, new_fars = get_annealed_near_far(
                nears=ray_bundle.nears,
                fars=ray_bundle.fars,
                step=step,
                total_anneal_steps=self.config.anneal_nearfar_steps,
                mid_perc=self.config.anneal_mid_perc,
            )
            ray_bundle = replace(ray_bundle, nears=new_nears, fars=new_fars)


        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        
        if not self.config.use_multi_channel_density:
            weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        else:
            weights_coarse = get_weights_multi_channel(ray_samples_uniform, field_outputs_coarse[FieldHeadNames.DENSITY], self.n_channels)
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse['hsi'],  # Just have to change this key
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        if self.config.use_multi_channel_density:
            weights_for_pdf = weights_coarse.sum(dim=-1)  # [R, S]
        else:
            weights_for_pdf = weights_coarse[..., 0]      # [R, S]
        weights_for_pdf = weights_for_pdf.unsqueeze(-1)  # [R, S] → [R, S, 1]
        weights_for_pdf = weights_for_pdf.contiguous()
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_for_pdf)

        # Second pass:
        field_outputs_fine = self.field.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        # Compute weights
        if not self.config.use_multi_channel_density:
            weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])  # [R, S]
        else:
            weights_fine = get_weights_multi_channel(
                ray_samples_pdf,
                field_outputs_fine[FieldHeadNames.DENSITY],
                self.n_channels
            )  # [R, S, C]
        # HSI rendering
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine['hsi'],
            weights=weights_fine,
        )
        # Accumulation and depth
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        # Accumulation map colormap
        if self.config.use_multi_channel_density:
            accumulation_mean = accumulation_fine.mean(dim=-1)
        else:
            accumulation_mean = accumulation_fine

        # Depth colormap
        if self.config.use_multi_channel_density:
            depth_map_mean = depth_fine.mean(dim=-1)
        else:
            depth_map_mean = depth_fine

        # Output packaging
        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "rgb": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "accumulation": accumulation_mean,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "depth": depth_map_mean,
            "density": field_outputs_fine[FieldHeadNames.DENSITY]
        }
        return outputs


    def get_loss_dict(self, outputs, batch, metrics_dict=None, step=np.inf):
        image = batch["image"].to(self.device)

        # Background blending
        pred_coarse, image_coarse = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        pred_fine, image_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        rgb_loss_coarse = self.rgb_loss(image_coarse, pred_coarse)
        rgb_loss_fine = self.rgb_loss(image_fine, pred_fine)
        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
        }

        # === Geometry regularization via depth TV loss ===
        if self.config.use_geometry_regularization and "random_ray_patches" in batch:
            patch_bundles = batch["random_ray_patches"]

            # Flatten and concat all patches into one RayBundle
            flat_bundles = [p.flatten() for p in patch_bundles]
            concat_bundle = concat_ray_bundles(flat_bundles)

            model_outputs = self.forward(concat_bundle)

            if self.config.schedule_geomreg:
                geom_weight = schedule_geometry_weight(
                    step=step,
                    start=self.config.geometry_reg_schedule_start,
                    end=self.config.geometry_reg_schedule_end,
                    min_weight=self.config.geometry_reg_schedule_min,
                    max_weight=self.config.geometry_reg_schedule_max,
                    mode=self.config.geometry_reg_schedule_type,
                    
                )
            else:
                geom_weight = self.config.lambda_geometry
            
            tv_loss = compute_batch_tv_loss(
                patch_ray_bundles=patch_bundles,
                model_outputs=model_outputs,
                patch_size=self.patch_size,
                weight=geom_weight
            )

            loss_dict["geometry_tv_loss"] = tv_loss

            # === Additional L1 loss on densities from random patch views ===
            if self.config.use_density_l1:
                if self.config.schedule_density_l1:
                    l1_weight = schedule_geometry_weight(
                        step=step,
                        start=self.config.density_l1_schedule_start,
                        end=self.config.density_l1_schedule_end,
                        min_weight=self.config.density_l1_schedule_min,
                        max_weight=self.config.density_l1_schedule_max,
                        reverse=False
                    )
                else:
                    l1_weight = self.config.lambda_density_l1
                patch_density = model_outputs["density"]  # shape: [R, S] or [R, S, C]
                l1_geom_loss = torch.mean(torch.abs(patch_density))
                loss_dict["density_l1_geom"] = self.config.density_l1_geomreg_scale * l1_weight * l1_geom_loss
        
        if self.config.use_weighted_mse:
            if self.config.schedule_wmse:
                wmse_weight = schedule_geometry_weight(
                    step=step,
                    start=self.config.wmse_schedule_start,
                    end=self.config.wmse_schedule_end,
                    min_weight=self.config.wmse_schedule_min,
                    max_weight=self.config.wmse_schedule_max,
                    mode=self.config.wmse_schedule_type,
                    reverse=False
                )
            else:
                wmse_weight = self.config.lambda_wmse

            wmse_loss = self.wmse(outputs["rgb_fine"], image)
            loss_dict["wmse"] = wmse_weight * wmse_loss
            self.config.lambda_wmse = wmse_weight

        # === Density sparsity regularization (scheduled) ===
        if self.config.use_density_l1:
            if self.config.schedule_density_l1:
                l1_weight = schedule_geometry_weight(
                    step=step,
                    start=self.config.density_l1_schedule_start,
                    end=self.config.density_l1_schedule_end,
                    min_weight=self.config.density_l1_schedule_min,
                    max_weight=self.config.density_l1_schedule_max,
                    reverse=False
                )
            else:
                l1_weight = self.config.lambda_density_l1

            density_fine = outputs["density"]  # shape: [R, S] or [R, S, C]
            l1_loss = torch.mean(torch.abs(density_fine))

            loss_dict["density_l1"] = l1_weight * l1_loss
        return misc.scale_dict(loss_dict, self.config.loss_coefficients)


    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """
        Overrides the function to convert hyperspectral outputs to RGB before rendering.

        Args:
            camera_ray_bundle: The ray bundle generated from the camera.

        Returns:
            Dict[str, torch.Tensor]: Outputs dictionary with hyperspectral image converted to RGB.
        """
        # Call the base method to get the original model outputs
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # Compute detection metrics
        rgb_fine = outputs["rgb_fine"]  # shape [H, W, C]
        rgb_fine = rgb_fine.permute(2, 0, 1).unsqueeze(0)  # → [1, C, H, W]
        rgb_fine = inverse_standardize(rgb_fine, self.standardizer)  # → [1, C, H, W]
        hsi_for_ace = rgb_fine.squeeze(0).permute(1, 2, 0).contiguous()  # → [H, W, C]

        outputs["Falsecolor"] = (hsi_for_ace - 170.7731) / (1980.9046 - 170.7731)
        outputs["Falsecolor"] = outputs["Falsecolor"][:, :, [59, 7, 15]]

        ace_img = ACE_det(hsi_for_ace, self.targ_sig)
        outputs["ACE"] = torch.from_numpy(ace_img).float().to(rgb_fine.device).unsqueeze(-1)

        if self.config.use_multi_channel_density:
            outputs["depth_59"] = outputs["depth_fine"][:, :, [59]]
        return outputs
    

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        assert self.config.collider_params is not None, "mip-NeRF requires collider parameters to be set."

        # === Load predicted and GT images ===
        hsi_fine = outputs["rgb_fine"]
        gt_image = batch["image"].to(self.device)
        gt_image = self.renderer_rgb.blend_background(gt_image)

        # === PCA inverse transform if enabled ===
        if self.image_mode == "pca":
            tmp_img = hsi_fine.cpu().numpy().reshape(-1, self.num_output_channels)
            tmp_img = self.pca.inverse_transform(tmp_img)
            tmp_img = tmp_img.reshape(hsi_fine.shape[0], hsi_fine.shape[1], self.n_channels)
            hsi_fine_inv = torch.from_numpy(tmp_img).to(self.device)
            hsi_fine_inv = torch.moveaxis(hsi_fine_inv, -1, 0)[None, ...]
            gt_image_full = batch["og_image"].to(self.device)
            gt_image_full = torch.moveaxis(gt_image_full, -1, 0)[None, ...]
        else:
            hsi_fine_inv = None
            gt_image_full = None

        # === Convert to [1, C, H, W] ===
        gt_image = torch.moveaxis(gt_image, -1, 0)[None, ...]
        hsi_fine = torch.moveaxis(hsi_fine, -1, 0)[None, ...]
        if self.standardizer is not None:  # Should put images back in full radiance
            gt_image = inverse_standardize(gt_image, self.standardizer)
            hsi_fine = inverse_standardize(hsi_fine, self.standardizer)


        # Compute detection metrics
        hsi_for_ace = hsi_fine.squeeze().permute(1, 2, 0).contiguous()  # [C, H, W] → [H, W, C]
        # tifffile.imwrite("testing/hsi_for_ace.tif", hsi_for_ace.cpu().numpy())
        ace_img = ACE_det(hsi_for_ace, self.targ_sig)
        ace_img_shape = ace_img.shape
        # Basic metrics
        det_mask = ace_img.flatten() > self.det_thresh
        gt_mask = batch["det_mask"].flatten()
        y_score = ace_img.flatten()

        # Confusion matrix (with label fix)
        cm = confusion_matrix(gt_mask, det_mask, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else float("nan")  # Sensitivity, Recall
        TNR = TN / (TN + FP) if (TN + FP) > 0 else float("nan")  # Specificity
        FPR = FP / (FP + TN) if (FP + TN) > 0 else float("nan")
        FNR = FN / (FN + TP) if (FN + TP) > 0 else float("nan")

        # AUC safety
        if len(np.unique(gt_mask)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(gt_mask, y_score)

        # Other scores (also handle edge cases if needed)
        metrics_dict = {
            "rmse": float(torch.mean(torch.sqrt((hsi_fine - gt_image) ** 2)).item()),
            "psnr": float(self.psnr(hsi_fine, gt_image).item()),
            "ssim": float(self.ssim(hsi_fine, gt_image).item()),
            "sam": float(self.sam(hsi_fine, gt_image).item()),
            "prec": precision_score(gt_mask, det_mask, zero_division=np.nan),
            "recall": recall_score(gt_mask, det_mask, zero_division=np.nan),
            "f1": f1_score(gt_mask, det_mask, zero_division=np.nan),
            "auc": auc,
            "acc": accuracy_score(gt_mask, det_mask),
            "tnr": float(TNR),
            "fpr": float(FPR),
            "fnr": float(FNR),
            "tpr": float(TPR)
        }

        


        # === PCA Inverse metrics ===
        if hsi_fine_inv is not None and gt_image_full is not None:
            metrics_dict.update({
                "rmse_invt": float(torch.mean(torch.sqrt((hsi_fine_inv - gt_image_full) ** 2)).item()),
                "psnr_invt": float(self.psnr_inv(hsi_fine_inv, gt_image_full).item()),
                "ssim_invt": float(self.ssim_inv(hsi_fine_inv, gt_image_full).item()),
                "sam_invt": float(self.sam_inv(hsi_fine_inv, gt_image_full).item()),
            })

        # === Prepare images for visualization/logging ===
        accum_map = outputs["accumulation_fine"]
        depth_map = outputs["depth_fine"]
        # print(f"accum shape before: {accum_map.shape}, depth shape: {depth_map.shape}")

        # Reorder if needed (C, H, W) → (H, W, C)
        # if accum_map.shape[0] == self.n_channels:
        #     accum_map = accum_map.permute(1, 2, 0)
        # if depth_map.shape[0] == self.n_channels:
        #     depth_map = depth_map.permute(1, 2, 0)

        # Reduce if multi-channel
        if accum_map.ndim == 3 and accum_map.shape[-1] > 1:
            accum_map = accum_map.mean(dim=-1)
        if depth_map.ndim == 3 and depth_map.shape[-1] > 1:
            depth_map = depth_map.mean(dim=-1)

        # accumulation = colormaps.apply_colormap(accum_map)
        
        # # Reduce accumulation map if it's still 3D
        # accum_for_depth = accum_map
        # if accum_for_depth.ndim == 3 and accum_for_depth.shape[-1] > 1:
        #     accum_for_depth = accum_for_depth.mean(dim=-1)  # [H, W]

        # depth = apply_depth_colormap(
        #     depth_map,
        #     accumulation=accum_for_depth,
        #     near_plane=self.config.collider_params["near_plane"],
        #     far_plane=self.config.collider_params["far_plane"],
        # )

        images_dict = {
            "img": hsi_fine,          # Fine HSI prediction
            "gt": gt_image,           # Ground-truth HSI or PCA image
            "accumulation": accum_map,
            "depth": depth_map,
            "ace_img": torch.tensor(ace_img.reshape(ace_img_shape)),
            "det_mask": torch.tensor(det_mask.reshape(ace_img_shape)),
            "gt_mask": torch.tensor(gt_mask.reshape(ace_img_shape)),
            "ace_gt": torch.tensor(batch["ACE_img"].reshape(ace_img_shape)),
        }

        if hsi_fine_inv is not None:
            images_dict["img_invt"] = hsi_fine_inv

        return metrics_dict, images_dict


        
