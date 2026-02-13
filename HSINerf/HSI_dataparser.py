# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from:
#   nerfstudio/nerfstudio/data/dataparsers/nerstudio_dataparaser.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/nerfstudio_dataparser.py
#
# Copyright (c) 2022-2025 The Nerfstudio Team
# Copyright (c) 2025 Los Alamos National Laboratory/Scout Jarman
#
# Prominent notice of changes (Apache-2.0 §4(b)):
#   [2025-10-03] (Scout Jarman) – Makes dataparser for HSI data
#       - Removes extra logic we don't use for HSI
#       - Addes new sampling strategies for train/test
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
from pathlib import Path
from typing import Literal, Optional, Type, Union
import numpy as np
import torch
import spectral.io.envi as envi
from rich.console import Console
import warnings

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction

from HSINerf.HSI_utils import get_train_eval_split_fixed_eval_random_train, get_train_eval_split_fraction_random,\
                              get_train_eval_split_farthest_subset, get_train_eval_split_farthest_biased

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class HSIDataParserConfig(DataParserConfig):
    """ HSI data parser config """
    _target: Type = field(default_factory=lambda: HSIDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """directory to split dataset (just the raw .json file with full image envi file paths)"""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    image_mode: Literal["full", "pca"] = "full"
    """ 'full' for all 128 channels in order, 'sub' to use n_components sorted by SF6 absorption, 'pca' to use PCA with n_components """
    n_components: int = 3
    """ number of components (PCA comps) to use, sorted by spectral variability (or eigenvalues) """
    standardize: bool = True
    """if true, nerf will predict delta from global mean instead of full radiance values"""
    sampling_method: Literal["idx_unif", "fixed_eval_random_train", "random_split", "uniform_spatial", "biased_spatial"] = "biased_spatial"
    """see HSINerf/README.md for sampling details"""
    train_split_fraction: Union[float, int] = .8
    """Percentage of train data to use (see README for details), if int, then that number of images"""
    train_seed: Optional[int] = 789
    """Seed for training split. If none it will be random everytime"""
    fixed_eval_fraction: Union[float, int] = 31
    """Percentage of evaluation data to used (only for fixed_eval_random_train)"""
    fixed_eval_seed: int = 987
    """Seed for fixed eval set. Changing will obviously change the eval set across runs (best to leave the same, only used for fixed_eval_random_train)"""
    detection_threshold: float = 0.6
    """Detection threshold for ACE detection"""
    biased_tophalf_fraction: float = 0.65
    """What proportion of images should come from the top half using biased_spatial sampling"""
    

@dataclass
class HSIDataParser(DataParser):
    """ DataParser for HSI data from Dirsig. Based on the nerfstudio_dataparser.py """

    config: HSIDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split = "train", **kwargs):
        # Make sure directory exists
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        # Read in transforms.json
        meta = load_from_json(self.config.data)
        data_dir = meta['datadir']

        # Values that we need for DataparserOutputs
        image_filenames = []
        poses = []
        cameras = []

        # Sort file names
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        # Loop through frames and save file names and poses (transform matrix)
        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            image_filenames.append(fname)
            poses.append(frame["transform_matrix"])

        # Some image mode and n channel componenet checks
        # set n_channels to work with rendering
        if not hasattr(self, "n_channels"):
            self.n_channels = 128
            self.config.n_channels = self.n_channels

        assert 0 < self.config.n_components <= self.n_channels, f"n_componenets (set to {self.config.n_components}) needs to be between 1 and {self.n_channels}."
        assert self.config.image_mode.lower() in ["full", "pca"], f"Image mode needs to be 'pca' or 'full'."
        self.config.image_mode = self.config.image_mode.lower()
        
        if self.config.image_mode == 'full':
            self.config.n_components = self.n_channels
        self.gas_name = None
        
        if 'sf6' in data_dir.lower():
            self.gas_name = 'sf6'
        elif 'ch4' in data_dir.lower():
            self.gas_name = 'ch4'
        else:
            self.gas_name = 'sf6'
            warnings.warn("Defaulting gas to sf6")

        # CONSOLE.log(f"Using gas {self.gas_name}")
        # assert self.gas_name is not None, f"When using sub, the gas name (either ch4 or sf6) must be present in the `datadir` tag in transforms.json!"

        # Get train/test split indices
        num_images = len(image_filenames)

        # Special-case handling for dataset with 1101 images
        if num_images == 1101:
            i_eval = list(range(1000, 1101))  # last 101 images
            sample_subset_indices = list(range(1000))  # first 1000 images
            poses_subset = [poses[i] for i in sample_subset_indices]
            image_filenames_subset = [image_filenames[i] for i in sample_subset_indices]

            if self.config.sampling_method == "idx_unif":
                i_train_subset, _ = get_train_eval_split_fraction(
                    image_filenames_subset,
                    self.config.train_split_fraction
                )
            elif self.config.sampling_method == "fixed_eval_random_train":
                i_train_subset, _ = get_train_eval_split_fixed_eval_random_train(
                    image_filenames_subset,
                    self.config.train_split_fraction,
                    self.config.train_seed,
                    self.config.fixed_eval_fraction,
                    self.config.fixed_eval_seed,
                )
            elif self.config.sampling_method == "random_split":
                i_train_subset, _ = get_train_eval_split_fraction_random(
                    image_filenames_subset,
                    self.config.train_split_fraction,
                    self.config.train_seed,
                )
            elif self.config.sampling_method == "uniform_spatial":
                i_train_subset, _ = get_train_eval_split_farthest_subset(
                    image_filenames_subset,
                    poses_subset,
                    train_fraction=self.config.train_split_fraction,
                    seed=self.config.train_seed,
                )
            elif self.config.sampling_method == "biased_spatial":
                i_train_subset, _ = get_train_eval_split_farthest_biased(
                    image_filenames_subset,
                    poses_subset,
                    train_fraction=self.config.train_split_fraction,
                    top_half_indices=range(300),  # adjustable
                    top_fraction=self.config.biased_tophalf_fraction,
                    seed=self.config.train_seed,
                    fixed_eval_fraction=self.config.fixed_eval_fraction,
                    fixed_eval_seed=self.config.fixed_eval_seed
                )
            else:
                raise ValueError(f"Sampling method '{self.config.sampling_method}' not supported for 1101-image case.")

            # Map subset indices back to full image indices
            i_train = [sample_subset_indices[i] for i in i_train_subset]


        else:
            # Fallback for all other datasets
            if self.config.sampling_method == "idx_unif":
                i_train, i_eval = get_train_eval_split_fraction(
                    image_filenames,
                    self.config.train_split_fraction,
                )
            elif self.config.sampling_method == "fixed_eval_random_train":
                i_train, i_eval = get_train_eval_split_fixed_eval_random_train(
                    image_filenames,
                    self.config.train_split_fraction,
                    self.config.train_seed,
                    self.config.fixed_eval_fraction,
                    self.config.fixed_eval_seed,
                )
            elif self.config.sampling_method == "random_split":
                i_train, i_eval = get_train_eval_split_fraction_random(
                    image_filenames,
                    self.config.train_split_fraction,
                    self.config.train_seed,
                )
            elif self.config.sampling_method == "uniform_spatial":
                i_train, i_eval = get_train_eval_split_farthest_subset(
                    image_filenames,
                    poses,
                    self.config.train_split_fraction,
                    self.config.train_seed,
                )
            elif self.config.sampling_method == "biased_spatial":
                i_train, i_eval = get_train_eval_split_farthest_biased(
                    image_filenames,
                    poses,
                    self.config.train_split_fraction,
                    top_half_indices=range(119),
                    top_fraction=self.config.biased_tophalf_fraction,
                    seed=self.config.train_seed,
                    fixed_eval_fraction=self.config.fixed_eval_fraction,
                    fixed_eval_seed=self.config.fixed_eval_seed
                )
            else:
                raise ValueError(f"Sampling method {self.config.sampling_method} not recognized")
                
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split '{split}'")

        # Some sort of auto orient stuff (from nerfstudio_dataparser)
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses (from nerfstudio_dataparser)
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames_og = image_filenames.copy()
        image_filenames = [image_filenames_og[i] for i in indices]
        train_filenames = [image_filenames_og[i] for i in i_train]
        eval_filenames  = [image_filenames_og[i] for i in i_eval]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        # === Estimate near/far based on camera distances to scene center ===
        # scene_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        # camera_positions = poses[:, :3, 3]
        # distances = torch.norm(camera_positions - scene_center[None, :], dim=-1)

        # near_plane = distances.min().item() - 0.2
        # far_plane = distances.max().item() + 0.2
        # near_plane = max(0.01, near_plane)  # Clamp to avoid negative values

        # CONSOLE.log(f"[auto] Estimated near_plane: {near_plane:.2f}")
        # CONSOLE.log(f"[auto] Estimated far_plane:  {far_plane:.2f}")




        # Create Camera stuff
        camera_type = CameraType.PERSPECTIVE
        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])
        distortion_params = (
            camera_utils.get_distortion_params (
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        )
        
        metadata = {}
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (fisheye_crop_radius is not None):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # Skipping colmap stuff from nerstudio parser since I don't think we have colmap stuff

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                **metadata,
                'n_channels': self.n_channels, 
                'image_mode': self.config.image_mode, 
                'n_components': self.config.n_components,
                'all_filenames': image_filenames_og,
                'train_filenames': train_filenames,
                'eval_filenames': eval_filenames,
                'gas_name': self.gas_name,
                'standardize': self.config.standardize,
                'det_thresh': self.config.detection_threshold
            },
        )

        return dataparser_outputs

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = envi.open(data_dir / filepath)
                h, w, c = test_img.nrows, test_img.ncols, test_img.nbands
                self.n_channels = c
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2 ** (df + 1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath



