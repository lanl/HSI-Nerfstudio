# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from:
#   nerfstudio/nerfstudio/engine/trainer.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/engine/trainer.py
#
# Copyright (c) 2022-2025 The Nerfstudio Team
# Copyright (c) 2025 Los Alamos National Laboratory/Scout Jarman
#
# Prominent notice of changes (Apache-2.0 §4(b)):
#   [2025-10-03] (Scout Jarman) – Alters trainer logic
#       - Adds alternative model saving/loading options
#       - Adds adaptive weighted MSE weight calculation
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
from typing import Type, List
import torch
import os
from pathlib import Path
import functools
import random
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils.misc import step_check
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.writer import EventName, TimeWriter

from rich.console import Console
CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600

@dataclass
class HSITrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: HSITrainer)
    best_ckpt_path: str = ""
    get_train_metrics: bool = False
    save_keys: List[str] = field(default_factory=lambda: ["img", "ace_img", "det_mask", "gt_mask"])

    steps_per_eval_batch: int = 1000000
    steps_per_eval_image: int = 1000000
    steps_per_eval_all_images: int = 1000
    steps_per_save: int = 1000
    mixed_precision: bool = True
    max_num_iterations: int = 100000


class HSITrainer(Trainer):
    config: HSITrainerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_eval_psnr = -float("inf")
        self.config.best_ckpt_path = self.checkpoint_dir / "best_eval_psnr.ckpt"
        self.config.best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.set_global_seed(self.config.machine.seed)

    def set_global_seed(self, seed: int):
        print("Setting all seeds...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.use_deterministic_algorithms(True, warn_only=True)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step) 

         # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)


        if step_check(step, self.config.steps_per_eval_all_images):
            # Get latest eval metrics from pipeline
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step, get_train_metrics=False, output_path=None, get_std=False, save_keys=None)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
            
            # You can customize which PSNR key to track
            current_psnr = metrics_dict.get("psnr")
            if current_psnr is not None and current_psnr > self.best_eval_psnr:
                self.best_eval_psnr = current_psnr
                # self.config.best_ckpt_path = self.checkpoint_dir / f"best_psnr_{current_psnr:.6f}_step_{step:06d}.ckpt"
                with open(self.checkpoint_dir / "../best_eval_psnr_txt.txt", "a") as file:
                    file.write(f"step: {step:06d}, PSNR: {self.best_eval_psnr:.4f}\n")
                

                # Save best checkpoint manually
                for f in self.checkpoint_dir.glob("best_eval_*.ckpt"):
                    f.unlink()
                torch.save(
                    {
                        "step": step,
                        "pipeline": self.pipeline.module.state_dict()
                        if hasattr(self.pipeline, "module")
                        else self.pipeline.state_dict(),
                        "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                        "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                        "scalers": self.grad_scaler.state_dict(),
                    },
                    self.config.best_ckpt_path,
                )


        # Update weighted MSE weights
        if step_check(step, self.pipeline.model.config.weighted_mse_eval_steps) and self.pipeline.model.config.use_weighted_mse and self.pipeline.model.config.lambda_wmse > 0:
            self.update_adaptive_loss_weights(step)



    def update_adaptive_loss_weights(self, step: int):
        # CONSOLE.print("[cyan]Updating adaptive loss weights using fixed_indices_train_dataloader...[/cyan]")

        dataloader = self.pipeline.datamanager.fixed_indices_train_dataloader
        device = self.device

        total_rays = 0
        residual_sum = None

        self.pipeline.eval()

        with torch.no_grad(), Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating training images...", total=len(dataloader))

            for camera, batch in dataloader:
                camera = camera.to(device)
                gt = batch["image"].to(device)        # [H, W, C]
                H, W, C = gt.shape
                gt_flat = gt.view(-1, C)              # [N_rays, C]
                num_rays = H * W

                outputs = self.pipeline.model.get_outputs_for_camera(camera)
                pred = outputs["rgb"]                 # [H, W, C]
                pred = pred.view(-1, C)               # [N_rays, C]

                assert pred.shape == gt_flat.shape, f"Shape mismatch: pred {pred.shape}, gt {gt_flat.shape}"

                residual = (pred - gt_flat).pow(2).sum(dim=0)  # [C]
                if residual_sum is None:
                    residual_sum = residual
                else:
                    residual_sum += residual

                total_rays += num_rays
                progress.advance(task)

        if total_rays == 0:
            CONSOLE.print("[red]No rays found. Skipping adaptive update.[/red]")
            return

        mean_residual = residual_sum / total_rays  # [C]

        if hasattr(self.pipeline.model, "wmse"):
            self.pipeline.model.wmse.update_weights(mean_residual)
            CONSOLE.print("[green]Adaptive MSE weights updated successfully.[/green]")

            # === Log top 3 weights ===
            weights = self.pipeline.model.wmse.weights.detach().cpu()
            top_vals, top_indices = torch.topk(weights, 3)

            for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_vals.tolist())):
                writer.put_scalar(f"AdaptiveMSE/top{i+1}_weight_val", val, step=step)
                writer.put_scalar(f"AdaptiveMSE/top{i+1}_weight_idx", idx, step=step)

            # Optional: log the full L2 norm of weights as a sanity check
            writer.put_scalar("AdaptiveMSE/weights_L2_norm", weights.norm().item(), step=step)

        else:
            CONSOLE.print("[yellow]No 'wmse' loss module found on the model.[/yellow]")



    def save_checkpoint(self, step: int) -> None:
        """Override to avoid deleting best_psnr_*.ckpt when saving latest."""
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )

        # Delete only other non-best checkpoints
        if self.config.save_only_latest_checkpoint:
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path and not f.name.startswith("best_eval_"):
                    f.unlink()

    def _load_checkpoint(self) -> None:
        """Override to skip non-step-format checkpoint files and load best PSNR from log."""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint

        # === Auto-fill load_dir if missing ===
        if load_dir is None:
            auto_dir = self.checkpoint_dir
            step_ckpts = sorted(auto_dir.glob("step-*.ckpt"), key=lambda x: int(x.stem.split("-")[1]))
            if step_ckpts:
                load_dir = auto_dir
                CONSOLE.print(f"[yellow]Auto-detected checkpoint dir: {load_dir}[/yellow]")

        # === Try loading best PSNR from log
        try:
            psnr_log_path = self.checkpoint_dir.parent / "best_eval_psnr_txt.txt"
            if psnr_log_path.exists():
                with open(psnr_log_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    last_line = lines[-1]
                    psnr_str = last_line.split("PSNR:")[-1].strip()
                    self.best_eval_psnr = float(psnr_str)
                    CONSOLE.print(f"[green]Loaded best PSNR from log:[/green] {self.best_eval_psnr:.4f}")
        except Exception as e:
            CONSOLE.print(f"[yellow]Warning: Could not parse best PSNR from log. {e}[/yellow]")

        # === Load latest checkpoint if available
        if load_dir is not None:
            step_ckpts = sorted(
                load_dir.glob("step-*.ckpt"),
                key=lambda x: int(x.stem.split("-")[1])
            )
            if not step_ckpts:
                CONSOLE.print(f"[yellow]No valid step-*.ckpt files found in {load_dir}, training from scratch.[/yellow]")
                return
            latest_ckpt = step_ckpts[-1]
            loaded_state = torch.load(latest_ckpt, map_location="cpu")
            self._start_step = loaded_state["step"] + 1

            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"[green]Loaded latest checkpoint from {latest_ckpt}[/green]")

        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1

            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"[green]Loaded specified checkpoint from {load_checkpoint}[/green]")

        else:
            CONSOLE.print("[yellow]No checkpoint found, training from scratch.[/yellow]")




