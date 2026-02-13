# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from:
#   nerfstudio/nerfstudio/data/datamanagers/base_datamanager.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/datamanagers/base_datamanager.py
#
# Copyright (c) 2022-2025 The Nerfstudio Team
# Copyright (c) 2025 Los Alamos National Laboratory/Scout Jarman
#
# Prominent notice of changes (Apache-2.0 §4(b)):
#   [2025-10-03] (Scout Jarman) – Makes datamanager for HSI data
#       - Modified vannila datamanager and config for HSI data
#       - Added random pose generation for geometry regularization
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
import math
import random
from typing import Literal, Type, List, Tuple, Dict
import spectral.io.envi as envi
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch
from rich.console import Console
from rich.progress import track
CONSOLE = Console(width=120)

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle

from HSINerf.HSI_dataset import HSIDataset
from HSINerf.HSI_dataparser import HSIDataParserConfig






def visualize_camera_poses(
    random_poses: List[torch.Tensor],
    train_cameras,
    output_path="random_camera_poses.png",
    scene_radius: float = 1.0,
    scene_center: torch.Tensor = None,
    up_vector: torch.Tensor = None
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import torch
    from typing import List
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # === Random camera poses ===
    rand_np = [p.cpu().numpy() for p in random_poses]
    rand_centers = np.stack([pose[:3, 3] for pose in rand_np])
    rand_dirs = np.stack([pose[:3, 2] for pose in rand_np])

    ax.scatter(
        rand_centers[:, 0], rand_centers[:, 1], rand_centers[:, 2],
        c='blue', label='Random Camera Centers', s=10, alpha=0.5
    )

    for origin, direction in zip(rand_centers, rand_dirs):
        ax.quiver(
            origin[0], origin[1], origin[2],
            -direction[0], -direction[1], -direction[2],
            length=0.3 * scene_radius, normalize=True,
            color='red', alpha=0.5
        )

    # === Training cameras ===
    train_poses = train_cameras.camera_to_worlds.cpu()  # [N, 3, 4]
    train_centers = train_poses[:, :3, 3].numpy()
    train_dirs = train_poses[:, :3, 2].numpy()

    ax.scatter(
        train_centers[:, 0], train_centers[:, 1], train_centers[:, 2],
        c='green', label='Training Camera Centers', s=40, alpha=1.0
    )

    for origin, direction in zip(train_centers, train_dirs):
        ax.quiver(
            origin[0], origin[1], origin[2],
            -direction[0], -direction[1], -direction[2],
            length=0.3 * scene_radius, normalize=True, color='black'
        )

    # === Scene center and up vector ===
    if scene_center is not None:
        c = scene_center.cpu().numpy()
        ax.scatter([c[0]], [c[1]], [c[2]], c='black', s=100, label='Scene Center', marker='x')
        
        if up_vector is not None:
            u = up_vector.cpu().numpy()
            ax.quiver(
                c[0], c[1], c[2],
                u[0], u[1], u[2],
                length=0.3 * scene_radius,
                color='magenta', linewidth=2, label='Shared Up'
            )

    ax.set_xlim(-scene_radius, scene_radius)
    ax.set_ylim(-scene_radius, scene_radius)
    ax.set_zlim(0, scene_radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses: Random (blue/red) vs Training (green/black)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)



def generate_rays_from_patch(
    patch_size: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: torch.Tensor,
    x: int,
    y: int,
    device: torch.device
) -> RayBundle:
    """
    Efficiently generate a RayBundle for a patch directly from camera intrinsics + pose.
    All tensors are correctly placed on the target device.
    """
    # Ensure intrinsics are tensors on the correct device
    fx = fx.to(device)
    fy = fy.to(device)
    cx = cx.to(device)
    cy = cy.to(device)

    # Generate pixel coordinates in the patch
    i_coords = torch.arange(y, y + patch_size, device=device)
    j_coords = torch.arange(x, x + patch_size, device=device)
    ii, jj = torch.meshgrid(i_coords, j_coords, indexing='xy')   # [H, W]

    # Convert to normalized camera space directions
    dirs_x = (jj - cx) / fx
    dirs_y = -(ii - cy) / fy
    dirs_z = -torch.ones_like(dirs_x)
    dirs_camera = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # [H, W, 3]

    # Apply camera-to-world transform
    R = c2w[:3, :3]  # [3, 3]
    t = c2w[:3, 3]   # [3]

    dirs_world = torch.matmul(dirs_camera, R.T)  # [H, W, 3]
    origins = t[None, None, :].expand_as(dirs_world)  # [H, W, 3]

    viewdirs = dirs_world / dirs_world.norm(dim=-1, keepdim=True)

    pixel_area_value = float(1.0 / (fx * fy))
    pixel_area = torch.full((*dirs_world.shape[:2], 1), pixel_area_value, device=device)

    return RayBundle(
        origins=origins,
        directions=dirs_world,
        pixel_area=pixel_area,
        camera_indices=torch.zeros((*dirs_world.shape[:2], 1), dtype=torch.long, device=device),
        nears=None,
        fars=None,
        metadata=None,
    )






@dataclass
class HSIDataManagerConfig(VanillaDataManagerConfig):
    """A hyperspectral datamanager - required to use with .setup()"""
    train_num_rays_per_batch: int = 4096
    eval_num_rays_per_batch: int = 4096

    _target: Type = field(default_factory=lambda: HSIDataManager)
    """target class to instantiate"""
    dataparser: AnnotatedDataParserUnion = field(default_factory=HSIDataParserConfig)
    """Makes sure we use our HSI parser not the default blender parser"""
    n_random_views: int = 10000
    """How many random views to generate for geometry regularization"""
    position_std: float = 0.125
    """How much std to apply for random position changes in random views"""
    angle_std: float = 7.5
    """How much angle jitter to apply to random angles"""
    plot_views: bool = False
    """If you want to plot the random views to help tune the random parameters"""
    sampling_method: Literal['from_training', 'bounding_box', 'none'] = 'bounding_box'
    """How to randomly generate views. from_training will add noise to the training locations. bounding_box follows the reg-nerf approach"""
    patch_size_geomreg: int = 8
    """Size of random patch for geometry regularization""" 
    number_random_samples: int = 1024
    """If -1, then sample as many patches to match original batch size, otherwise sample as many patches to reach number_random_samples"""
    use_geometry_regularization: bool = True
    """Turn on random sample generation"""


class HSIDataManager(VanillaDataManager[HSIDataset]):
    """Data manager implementation for data that also requires processing hyperspectral data.
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: HSIDataManagerConfig

    def __init__(self, config, device = "cpu", test_mode = "val", world_size = 1, local_rank = 0, **kwargs):
        # This will get redefine in super().__init__(), but we need it here for file names
        self.config = config
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataparser_outputs.metadata["patch_size"] = self.config.patch_size

        # calculate/load needed transforms
        self.image_mode = self.train_dataparser_outputs.metadata.get("image_mode", 'full')
        self.n_components = self.train_dataparser_outputs.metadata.get("n_components", 3)
        self.n_channels = self.train_dataparser_outputs.metadata.get("n_channels", 128)
        self.gas_name = self.train_dataparser_outputs.metadata.get("gas_name", None)
        self.standardize = self.train_dataparser_outputs.metadata.get("standardize", True)
        self.det_thresh = self.train_dataparser_outputs.metadata.get("det_thresh", .6)

        self.pca = None
        self.standardizer = None
        # load in all images to create standardizer
        if self.standardize:
            CONSOLE.log("loading images to standardize")
            self.images = []
            tmp_files = self.train_dataparser_outputs.metadata.get("train_filenames", [])
            for fname in track(tmp_files, "Reading in images...", transient=True):
                img = envi.open(fname).asarray()
                img = np.array(1e6*img, dtype="float32")
                self.images.append(img.reshape((-1, self.n_channels)))
            self.images = np.asarray(self.images).reshape((-1, self.n_channels))
            self.standardizer = StandardScaler().fit(self.images)
        
        # Find PCA
        if self.image_mode == "pca":
            CONSOLE.log("Finding PCA from training data")
            self.pca = PCA(self.n_components, random_state=789).fit(self.images)

        
        # Load in gas for detection
        if self.gas_name == 'sf6':
            # Read and process the gas path spectral signatures
            gas_path = "data/sf6_ext.txt"
            data = np.loadtxt(gas_path, delimiter="\t")
            wavelengths = 10000 / data[:, 0]
            absorption = data[:, 1]
            # Interpolate to 128 points
            x_new = np.linspace(7.8, 7.8 + 0.044 * 127, 128)
            self.targ_sig = np.interp(x_new, wavelengths, absorption)
        else:
            # Load other gases here
            raise ValueError(f"Can't load {self.gas_name}!")



        # Set everything else up
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

        # This used to be in setup train, we'll just put it here
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size*4
        )

        # Sets up random samples for geometry regularization
        if self.config.use_geometry_regularization:
            self.random_camera_poses: List[torch.Tensor] = []
            self.scene_center = None
            self.shared_up = None

            if self.config.sampling_method == "from_training":
                self.random_camera_poses = self._sample_random_poses_from_training(
                    n=self.config.n_random_views,
                    train_cameras=self.train_dataparser_outputs.cameras,
                    position_noise_std=self.config.position_std,
                    angle_jitter_deg=self.config.angle_std
                )
            elif self.config.sampling_method == "bounding_box":
                self.random_camera_poses = self._sample_random_poses_from_bbox(
                    n=self.config.n_random_views,
                    train_cameras=self.train_dataparser_outputs.cameras,
                    lookat_jitter_std=self.config.position_std
                )


            if self.config.plot_views:
                CONSOLE.log("Plotting random views...")
                visualize_camera_poses(
                    random_poses=self.random_camera_poses,
                    train_cameras=self.train_dataparser_outputs.cameras,
                    output_path="pose_comparison.png",
                    scene_center=self.scene_center,
                    up_vector=self.shared_up
                )


    def setup_random_views(self, num_random_views: int = 50, radius: float = 1.5):
        """Stores random camera poses (CPU only); ray bundles are generated on demand."""
        self.random_camera_poses = []
        poses = self._sample_random_poses(num_random_views, radius=radius).cpu()
        self.random_camera_poses.extend([pose for pose in poses])  # List of [4,4] tensors


    def _sample_random_poses_from_training(
        self,
        n: int,
        train_cameras: Cameras,
        position_noise_std: float = 2.5,
        angle_jitter_deg: float = 10
    ) -> List[torch.Tensor]:
        """Samples augmented versions of training camera poses by adding small noise."""

        device = train_cameras.camera_to_worlds.device
        poses_out = []

        training_poses = train_cameras.camera_to_worlds  # [N, 3, 4]
        num_train = training_poses.shape[0]

        angle_jitter_rad = math.radians(angle_jitter_deg)

        for _ in range(n):
            # === Sample base pose ===
            idx = torch.randint(0, num_train, (1,))
            base_pose = training_poses[idx].squeeze(0).clone()  # [3, 4]

            # === Perturb camera position ===
            translation_noise = torch.randn(3, device=device) * position_noise_std
            base_pose[:3, 3] += translation_noise

            # === Perturb view direction (Z axis) slightly ===
            forward = -base_pose[:3, 2]
            up = base_pose[:3, 1]
            right = base_pose[:3, 0]

            # Generate a small rotation around a random axis in camera space
            axis = torch.randn(3, device=device)
            axis = axis / axis.norm()
            angle = torch.rand(1, device=device) * angle_jitter_rad  # keep as Tensor
            skew = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=device)
            R_delta = torch.eye(3, device=device) + torch.sin(angle)*skew + (1 - torch.cos(angle)) * (skew @ skew)

            # Apply to orientation
            new_rot = R_delta @ base_pose[:3, :3]
            new_pose = torch.eye(4, device=device)
            new_pose[:3, :3] = new_rot
            new_pose[:3, 3] = base_pose[:3, 3]
            poses_out.append(new_pose)

        return poses_out


    def _sample_random_poses_from_bbox(
        self,
        n: int,
        train_cameras: Cameras,
        lookat_jitter_std: float = 0.01
    ) -> List[torch.Tensor]:
        """
        Sample random camera poses within the bounding box of training camera centers,
        using a shared up vector and focusing on the least-squares scene center with jitter.
        """
        device = train_cameras.camera_to_worlds.device
        cam2worlds = train_cameras.camera_to_worlds

        # === Step 1: Compute bounding box of training camera positions
        centers = cam2worlds[:, :3, 3]  # [N, 3]
        min_xyz = centers.min(dim=0).values
        max_xyz = centers.max(dim=0).values

        # === Step 2: Compute shared up vector (mean of R[:, 1])
        rot_mats = cam2worlds[:, :3, :3]     # [N, 3, 3]
        up_vectors = rot_mats[:, :, 1]      # [N, 3]
        shared_up = up_vectors.mean(dim=0)
        shared_up = shared_up / shared_up.norm()
        self.shared_up = shared_up

        # === Step 3: Compute least-squares focus point from optical axes
        def compute_focus_point_least_squares():
            rays_o = centers
            rays_d = -rot_mats[:, :, 2]
            rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
            I = torch.eye(3, device=device)[None]
            proj = I - rays_d[:, :, None] @ rays_d[:, None, :]
            A = proj.sum(dim=0)
            b = (proj @ rays_o[:, :, None]).sum(dim=0)
            return torch.linalg.lstsq(A, b).solution.squeeze(1)  # [3]

        scene_center = compute_focus_point_least_squares()
        self.scene_center = scene_center

        # === Step 4: Generate poses
        poses = []
        for _ in range(n):
            # Random position in bbox
            pos = torch.rand(3, device=device) * (max_xyz - min_xyz) + min_xyz

            # Jittered focus point
            jittered_focus = scene_center + torch.randn(3, device=device) * lookat_jitter_std

            # Camera forward vector
            forward = pos - jittered_focus
            forward = forward / forward.norm()

            # Use shared up to compute orthogonal frame
            right = torch.cross(shared_up, forward)
            right = right / right.norm()
            up = torch.cross(forward, right)

            # Build pose matrix
            rot = torch.stack([right, up, forward], dim=-1)
            pose = torch.eye(4, device=device)
            pose[:3, :3] = rot
            pose[:3, 3] = pos
            poses.append(pose)

        return poses


    def sample_random_patch(self, patch_size: int = 16, device: torch.device = torch.device("cuda")) -> RayBundle:
        """Samples a random patch from a random camera pose."""
        pose = random.choice(self.random_camera_poses).to(device)

        # Camera intrinsics
        cams = self.train_dataparser_outputs.cameras
        fx, fy = cams.fx[0], cams.fy[0]
        cx, cy = cams.cx[0], cams.cy[0]
        H, W = int(cams.height[0]), int(cams.width[0])

        y = torch.randint(0, H - patch_size, (1,)).item()
        x = torch.randint(0, W - patch_size, (1,)).item()

        ray_bundle = generate_rays_from_patch(
            patch_size=patch_size,
            fx=fx, fy=fy, cx=cx, cy=cy,
            c2w=pose,
            x=x, y=y,
            device=device
        )
        return ray_bundle
    

    def sample_random_patches(self, total_rays: int, device: torch.device) -> List[RayBundle]:
        patch_area = self.config.patch_size_geomreg ** 2
        
        if self.config.number_random_samples < 1:
            num_patches = max(1, total_rays // patch_area)
        else:
            num_patches = max(1, self.config.number_random_samples // patch_area)
        return [self.sample_random_patch(self.config.patch_size_geomreg, device) for _ in range(num_patches)]


    def create_train_dataset(self):
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            metadata={
                'image_mode': self.image_mode,
                'n_components': self.n_components,
                'n_channels': self.n_channels,
                'pca': self.pca,
                'standardizer': self.standardizer,
                'targ_sig': self.targ_sig,
                "det_thresh": self.det_thresh,
                "patch_size": self.config.patch_size_geomreg
            }
        )


    def create_eval_dataset(self):
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            metadata={
                'image_mode': self.image_mode,
                'n_components': self.n_components,
                'n_channels': self.n_channels,
                'pca': self.pca,
                'standardizer': self.standardizer,
                'targ_sig': self.targ_sig,
                "det_thresh": self.det_thresh,
                "patch_size": self.config.patch_size_geomreg
            }
        )
        
    
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        ray_bundle, batch = super().next_train(step)  # retains existing behavior
        if self.config.use_geometry_regularization:
            num_rays = ray_bundle.origins.shape[0]
            patches = self.sample_random_patches(total_rays=num_rays, device=self.device)
            batch["random_ray_patches"] = patches
        return ray_bundle, batch
