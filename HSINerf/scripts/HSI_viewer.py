#!/usr/bin/env python

"""
hsi-viewer: same as ns-viewer, but with safe checkpoint loading.
"""

from __future__ import annotations

import time
import tyro
from dataclasses import dataclass, field, fields
from pathlib import Path
from threading import Lock
from typing import Literal

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState

from HSINerf.scripts.HSI_eval import hsi_eval_setup


@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Viewer configuration minus num_rays chunk so tyro doesn't override it."""
    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunHSIViewer:
    """Viewer CLI entrypoint."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    vis: Literal["viewer", "viewer_legacy"] = "viewer"
    use_best_model: bool = True

    def main(self) -> None:
        config, pipeline, _, step = hsi_eval_setup(self.load_config, use_best_model=self.use_best_model)


        # Patch config fields as done in ns-viewer
        num_rays = config.viewer.num_rays_per_chunk
        config.vis = self.vis
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays

        _start_viewer(config, pipeline, step)

    def save_checkpoint(self, *args, **kwargs):
        """Needed by viewer_state.update_scene but unused here."""



def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int):
    """Start the viewer UI exactly as in ns-viewer."""
    base_dir = config.get_base_dir()
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    banner_messages = None
    viewer_state = None
    viewer_callback_lock = Lock()

    if config.vis == "viewer_legacy":
        viewer_state = ViewerLegacyState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            train_lock=viewer_callback_lock,
        )
        banner_messages = [f"Legacy viewer at: {viewer_state.viewer_url}"]

    elif config.vis == "viewer":
        viewer_state = ViewerState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            share=config.viewer.make_share_url,
            train_lock=viewer_callback_lock,
        )
        banner_messages = viewer_state.viewer_info

    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

    assert viewer_state and pipeline.datamanager.train_dataset
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    if isinstance(viewer_state, ViewerLegacyState):
        viewer_state.viser_server.set_training_state("completed")
    viewer_state.update_scene(step=step)
    while True:
        time.sleep(0.01)


def entrypoint():
    tyro.extras.set_accent_color("bright_magenta")
    tyro.cli(tyro.conf.FlagConversionOff[RunHSIViewer]).main()


if __name__ == "__main__":
    entrypoint()

get_parser_fn = lambda: tyro.extras.get_parser(tyro.conf.FlagConversionOff[RunHSIViewer])
