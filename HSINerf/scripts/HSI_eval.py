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


#!/usr/bin/env python
"""
Custom evaluation script for HSI pipeline, bypassing eval_setup and eval_load_checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal, Callable, Tuple

import torch
import tyro
import yaml

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.configs.method_configs import all_methods



def hsi_eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
    use_best_model: bool = True,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Custom eval setup for HSI viewer that prioritizes best_eval_psnr.ckpt if available.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass.
        test_mode: Controls which datasets are loaded.
        update_config_callback: Callback to update the config before loading the pipeline.

    Returns:
        Loaded config, pipeline module, checkpoint path, and step number.
    """
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    config.load_dir = config.get_checkpoint_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    pipeline.eval()

    # === Prefer best_eval_psnr.ckpt if it exists
    if use_best_model:
        best_ckpt = config.load_dir / "best_eval_psnr.ckpt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Expected best checkpoint at {best_ckpt} but it was not found.")
        ckpt_path = best_ckpt
    else:
        step_ckpts = sorted(
            [f for f in config.load_dir.glob("step-*.ckpt") if f.stem.split("-")[1].isdigit()],
            key=lambda f: int(f.stem.split("-")[1])
        )
        if not step_ckpts:
            raise FileNotFoundError(f"No valid step-*.ckpt files found in {config.load_dir}")
        ckpt_path = step_ckpts[-1]

    state = torch.load(ckpt_path, map_location="cpu")
    pipeline.load_pipeline(state["pipeline"], state["step"])
    CONSOLE.print(f"[green]Loaded checkpoint:[/green] {ckpt_path}")

    return config, pipeline, ckpt_path, state["step"]


@dataclass
class HSIEval:
    """Evaluate two models in a unified directory and save the results separately."""

    base_dir: Path
    get_std: bool = True
    get_train_metrics: bool = False
    save_keys: List[str] = field(default_factory=lambda: ["img", "ace_img", "det_mask", "gt_mask"])

    def main(self) -> None:
        config_path = self.base_dir / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file at {config_path}")

        # Output dirs
        output_root = self.base_dir / "hsi-eval"
        latest_output_dir = output_root / "latest"
        best_output_dir = output_root / "best"
        latest_output_dir.mkdir(parents=True, exist_ok=True)
        best_output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Evaluate latest step-*.ckpt ----
        CONSOLE.print("[bold blue]Evaluating latest checkpoint...[/bold blue]")
        config_latest, pipeline_latest = load_config_and_setup_pipeline(config_path)
        latest_ckpt = find_latest_checkpoint(self.base_dir)
        load_checkpoint_into_pipeline(pipeline_latest, latest_ckpt)
        metrics_latest = pipeline_latest.get_average_eval_image_metrics(
            output_path=latest_output_dir,
            get_std=self.get_std,
            get_train_metrics=self.get_train_metrics,
            save_keys=self.save_keys,
        )
        self._save_results(latest_output_dir / "metrics.json", config_latest, str(latest_ckpt), metrics_latest)

        # ---- Evaluate best_eval_psnr.ckpt ----
        best_ckpt = self.base_dir / "nerfstudio_models" / "best_eval_psnr.ckpt"
        if best_ckpt.exists():
            CONSOLE.print("[bold green]Evaluating best_eval_psnr.ckpt...[/bold green]")
            config_best, pipeline_best = load_config_and_setup_pipeline(config_path)
            load_checkpoint_into_pipeline(pipeline_best, best_ckpt)
            metrics_best = pipeline_best.get_average_eval_image_metrics(
                output_path=best_output_dir,
                get_std=self.get_std,
                get_train_metrics=self.get_train_metrics,
                save_keys=self.save_keys,
            )
            self._save_results(best_output_dir / "metrics.json", config_best, str(best_ckpt), metrics_best)
        else:
            CONSOLE.print("[yellow]No best_eval_psnr.ckpt found, skipping.[/yellow]")

    @staticmethod
    def _save_results(path: Path, config: TrainerConfig, checkpoint_path: str, results: dict):
        output = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": checkpoint_path,
            "results": results,
        }
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        CONSOLE.print(f"[bold white]Saved results to:[/bold white] {path}")


def load_config_and_setup_pipeline(config_path: Path) -> tuple[TrainerConfig, Pipeline]:
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="val")
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    return config, pipeline


def load_checkpoint_into_pipeline(pipeline: Pipeline, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    pipeline.load_pipeline(state["pipeline"], state["step"])
    CONSOLE.print(f"[green]Loaded checkpoint:[/green] {ckpt_path}")


def find_latest_checkpoint(base_dir: Path) -> Path:
    ckpt_dir = base_dir / "nerfstudio_models"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory found at {ckpt_dir}")

    step_ckpts = sorted(
        [f for f in ckpt_dir.glob("step-*.ckpt") if f.is_file()],
        key=lambda p: int(p.stem.split("-")[1]),
        reverse=True
    )
    if not step_ckpts:
        raise FileNotFoundError(f"No step-*.ckpt files found in {ckpt_dir}")
    return step_ckpts[0]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_magenta")
    tyro.cli(HSIEval).main()


if __name__ == "__main__":
    entrypoint()

get_parser_fn = lambda: tyro.extras.get_parser(HSIEval)
