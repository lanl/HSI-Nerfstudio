# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from:
#   nerfstudio/nerfstudio/models/nerfacto.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/nerfacto.py
#
# Copyright (c) 2022-2025 The Nerfstudio Team
# Copyright (c) 2025 Los Alamos National Laboratory/Scout Jarman
#
# Prominent notice of changes (Apache-2.0 §4(b)):
#   [2025-10-03] (Scout Jarman) – Alters Nerfacto for HSI
#       - Changes output to be 128 dim, and other modifications to work with HSI
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



from dataclasses import dataclass, field
from typing import Literal, Tuple, Union, Type, Dict, List
from torch import nn
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

from rich.console import Console
CONSOLE = Console(width=120)

from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.losses import MSELoss, L1Loss
from nerfstudio.cameras.cameras import Cameras

from HSINerf.HSI_utils import HSIRenderer, inverse_standardize, Loss_with_SAM, \
    concat_ray_bundles, compute_batch_tv_loss, ACE_det


@dataclass
class HSIModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: HSIModel)
    """Configuration for Hyperspectral Nerfacto Model."""
    num_output_channels: int = 128
    """Number of hyperspectral output channels."""
    num_layers_color: int = 3
    """number of layers for the color MLP"""
    loss: Literal["mse", "l1"] = "l1"
    """If you want to train using MSE or L1/SAM"""
    lambda_sam: float = 1.0



"""
Updates the Nerfacto field so that the mlp_head outputs full hyperspectral channels
Everything else is kept same as nerfacto
"""
class HSIField(NerfactoField):
    def __init__(self, aabb, n_output_channels: int = 128, num_layers_color: int = 3, hidden_dim_color: int = 64, out_activation=None, *args, **kwargs):
        super().__init__(aabb, *args, **kwargs)
        self.n_output_channels = n_output_channels
        self.out_activation = out_activation

        # Replace the RGB head with one that outputs `n_output_channels`
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=n_output_channels,  # Output `n_output_channels` instead of 3 (RGB)
            activation=nn.ReLU(),
            out_activation=out_activation,
            implementation=kwargs.get("implementation", "tcnn"),
        )




class HSIModel(NerfactoModel):

    config: HSIModelConfig

    def __init__(self, config, scene_box, num_train_data, **kwargs):
        self.metadata = kwargs['metadata']
        
        super().__init__(config, scene_box, num_train_data, **kwargs)
        

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


        # Have to copy from original Nerfactomodel, so we can pass it all into our HSI field
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Override the field with HSIField
        self.field = HSIField(
            self.scene_box.aabb,  # Need to manually pass this in
            self.config.num_output_channels,  # This is where we specify our hyperspectral channels
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
            out_activation=self.out_activation,  # Change if using PCA representation
            num_layers_color=self.config.num_layers_color
        )

        # Override the RGBRenderer
        self.renderer_rgb = HSIRenderer(self.config.background_color)

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
        rgb_fine = outputs["rgb"]  # shape [H, W, C]
        rgb_fine = rgb_fine.permute(2, 0, 1).unsqueeze(0)  # → [1, C, H, W]
        rgb_fine = inverse_standardize(rgb_fine, self.standardizer)  # → [1, C, H, W]
        hsi_for_ace = rgb_fine.squeeze(0).permute(1, 2, 0).contiguous()  # → [H, W, C]

        outputs["Falsecolor"] = (hsi_for_ace - 170.7731) / (1980.9046 - 170.7731)
        outputs["Falsecolor"] = outputs["Falsecolor"][:, :, [59, 7, 15]]

        ace_img = ACE_det(hsi_for_ace, self.targ_sig)
        outputs["ACE"] = torch.from_numpy(ace_img).float().to(rgb_fine.device).unsqueeze(-1)


        return outputs



    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        # === Load predicted and GT images ===
        hsi_fine = outputs["rgb"]
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
        accum_map = outputs["accumulation"]
        depth_map = outputs["depth"]
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
        