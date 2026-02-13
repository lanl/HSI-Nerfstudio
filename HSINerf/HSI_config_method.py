from dataclasses import replace

from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from HSINerf.HSI_datamanager import HSIDataManagerConfig
from HSINerf.HSI_dataparser import HSIDataParserConfig
from HSINerf.HSI_model import HSIModelConfig
from HSINerf.HSI_model_mip import HSIModelMipConfig
from HSINerf.HSI_pipeline import HSIPipelineConfig
from HSINerf.HSI_trainer import HSITrainerConfig

# === Shared Components ===

common_optimizer = {
    "fields": {
        "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-8),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=1e-5,
            lr_pre_warmup=1e-5,
            warmup_steps=2000,
            ramp='cosine',
            max_steps=150000),
    }
}

common_viewer = ViewerConfig(
    num_rays_per_chunk=8192,
    websocket_port_default=53435,
    quit_on_train_completion=True,
)

common_datamanager = HSIDataManagerConfig(
    dataparser=HSIDataParserConfig()
)

# Additional ablation study options

hsi_method_mip_L2 = MethodSpecification(
        config=HSITrainerConfig(
            method_name="hsi_mipnerf_L2",
            pipeline=HSIPipelineConfig(
                datamanager=replace(common_datamanager, use_geometry_regularization=False),
                model=HSIModelMipConfig(
                    use_multi_channel_density=False,
                    use_geometry_regularization=False,
                    use_anneal_sampling=False,
                    loss='mse',
                    lambda_sam=0,
                    use_weighted_mse=False
                )
            ),
            optimizers=common_optimizer,
            viewer=common_viewer,
            vis="viewer+tensorboard",
        ),
        description="MipNerf variant with options"
    )

hsi_method_mip_L1 = MethodSpecification(
        config=HSITrainerConfig(
            method_name="hsi_mipnerf_L1",
            pipeline=HSIPipelineConfig(
                datamanager=replace(common_datamanager, use_geometry_regularization=False),
                model=HSIModelMipConfig(
                    use_multi_channel_density=False,
                    use_geometry_regularization=False,
                    use_anneal_sampling=False,
                    loss='l1',
                    lambda_sam=0,
                    use_weighted_mse=False
                )
            ),
            optimizers=common_optimizer,
            viewer=common_viewer,
            vis="viewer+tensorboard",
        ),
        description="MipNerf variant with options"
    )

hsi_method_mip_L1_SAM = MethodSpecification(
        config=HSITrainerConfig(
            method_name="hsi_mipnerf_L1_SAM",
            pipeline=HSIPipelineConfig(
                datamanager=replace(common_datamanager, use_geometry_regularization=False),
                model=HSIModelMipConfig(
                    use_multi_channel_density=False,
                    use_geometry_regularization=False,
                    use_anneal_sampling=False,
                    loss='l1',
                    lambda_sam=2,
                    use_weighted_mse=False
                )
            ),
            optimizers=common_optimizer,
            viewer=common_viewer,
            vis="viewer+tensorboard",
        ),
        description="MipNerf variant with options"
    )

hsi_method_mip_L2_SAM = MethodSpecification(
        config=HSITrainerConfig(
            method_name="hsi_mipnerf_L2_SAM",
            pipeline=HSIPipelineConfig(
                datamanager=replace(common_datamanager, use_geometry_regularization=False),
                model=HSIModelMipConfig(
                    use_multi_channel_density=False,
                    use_geometry_regularization=False,
                    use_anneal_sampling=False,
                    loss='mse',
                    lambda_sam=2,
                    use_weighted_mse=False
                )
            ),
            optimizers=common_optimizer,
            viewer=common_viewer,
            vis="viewer+tensorboard",
        ),
        description="MipNerf variant with options"
    )


hsi_method_mip_MD_GR_L2 = MethodSpecification(
        config=HSITrainerConfig(
            method_name="hsi_mipnerf_MD_GR_L2",
            pipeline=HSIPipelineConfig(
                datamanager=replace(common_datamanager, use_geometry_regularization=True),
                model=HSIModelMipConfig(
                    use_multi_channel_density=True,
                    use_geometry_regularization=True,
                    use_anneal_sampling=True,
                    loss='mse',
                    lambda_sam=0,
                    use_weighted_mse=False
                )
            ),
            optimizers=common_optimizer,
            viewer=common_viewer,
            vis="viewer+tensorboard",
        ),
        description="MipNerf variant with options"
    )

# Main variants

def make_hsi_mipnerf(name: str, use_multidensity=False, use_geomreg=False):
    return MethodSpecification(
        config=HSITrainerConfig(
            method_name=name,
            pipeline=HSIPipelineConfig(
                datamanager=replace(common_datamanager, use_geometry_regularization=use_geomreg),
                model=HSIModelMipConfig(
                    use_multi_channel_density=use_multidensity,
                    use_geometry_regularization=use_geomreg,
                    use_anneal_sampling=use_geomreg,
                )
            ),
            optimizers=common_optimizer,
            viewer=common_viewer,
            vis="viewer+tensorboard",
        ),
        description="MipNerf variant with options"
    )


hsi_method_mip = make_hsi_mipnerf(
    "hsi_mipnerf")

hsi_method_mip_multidensity = make_hsi_mipnerf(
    name="hsi_mipnerf_MD",
    use_multidensity=True
)

hsi_method_mip_geomreg = make_hsi_mipnerf(
    name="hsi_mipnerf_GR",
    use_geomreg=True,
)

hsi_method_mip_geomreg_multichannel = make_hsi_mipnerf(
    name="hsi_mipnerf_MD_GR",
    use_geomreg=True,
    use_multidensity=True,
)


# === Nerfacto-like model (not mipnerf) ===

hsi_method = MethodSpecification(
    config=HSITrainerConfig(
        method_name='hsi_nerf',
        pipeline=HSIPipelineConfig(
            datamanager=replace(common_datamanager, use_geometry_regularization=False),
            model=HSIModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.1,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                lambda_sam=2,
                hidden_dim=128,
                hidden_dim_color=128,
                hidden_dim_transient=128,
                num_layers_color=8,
                collider_params={"near_plane": 0, "far_plane": 2}
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    lr_pre_warmup=1e-5,
                    warmup_steps=2000,
                    ramp='cosine',
                    max_steps=150000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    lr_pre_warmup=1e-5,
                    warmup_steps=2000,
                    ramp='cosine',
                    max_steps=150000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis="viewer+tensorboard",
    ),
    description="HSI Nerfacto Model"
)
