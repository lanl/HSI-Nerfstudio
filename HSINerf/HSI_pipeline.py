# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from:
#   nerfstudio/nerfstudio/pipelines/base_pipeline.py  (Apache-2.0)
#   https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/pipelines/base_pipeline.py
#
# Copyright (c) 2022-2025 The Nerfstudio Team
# Copyright (c) 2025 Los Alamos National Laboratory/Scout Jarman
#
# Prominent notice of changes (Apache-2.0 §4(b)):
#   [2025-10-03] (Scout Jarman) – Alters pipeline to output HSI and calculate metrics
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
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from time import time
import torch
import tifffile
import json

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from HSINerf.HSI_utils import nanstd_mean


@dataclass
class HSIPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: HSIPipeline)
    get_std: bool = True
    get_train_metrics: bool = False
    save_keys: List[str] = field(default_factory=lambda: ["img", "ace_img", "det_mask", "gt_mask", "depth"])
    output_path: str = ""


class HSIPipeline(VanillaPipeline):

    config: HSIPipelineConfig

    def __init__(self, config, device, test_mode="val", world_size=1, local_rank=0, grad_scaler=None):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step):
        metrics_dict, _ = super().get_eval_image_metrics_and_images(step)
        return metrics_dict, {}  # intentionally omit images
    
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle, step)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, step)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step=None, output_path=None, get_std=None, get_train_metrics=None, save_keys=None):
        output_path = output_path or self.config.output_path
        get_std = self.config.get_std if get_std is None else get_std
        get_train_metrics = self.config.get_train_metrics if get_train_metrics is None else get_train_metrics
        save_keys = self.config.save_keys if save_keys is None else save_keys

        self.eval()

        metrics_eval, per_image_eval = self._evaluate_dataloader(
            self.datamanager.fixed_indices_eval_dataloader,
            self.datamanager.eval_dataset.og_idx,
            output_path,
            prefix="eval",
            save_keys=save_keys,
            get_std=get_std,
        )

        if output_path:
            with open(output_path / "per_image_eval_metrics.json", "w") as f:
                json.dump(per_image_eval, f, indent=2)

        if get_train_metrics:
            metrics_train, per_image_train = self._evaluate_dataloader(
                self.datamanager.fixed_indices_train_dataloader,
                self.datamanager.train_dataset.og_idx,
                output_path,
                prefix="train",
                save_keys=save_keys,
                get_std=get_std,
            )

            if output_path:
                with open(output_path / "per_image_train_metrics.json", "w") as f:
                    json.dump(per_image_train, f, indent=2)

            self.train()
            merged = {f"eval_{k}": v for k, v in metrics_eval.items()}
            merged.update({f"train_{k}": v for k, v in metrics_train.items()})
            return merged

        self.train()
        return metrics_eval


    def _evaluate_dataloader(self, dataloader, og_indices, output_path, prefix, save_keys, get_std):
        metrics_dict_list = []
        if output_path:
            output_path.mkdir(exist_ok=True, parents=True)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"[green]Evaluating {prefix} images...", total=len(dataloader))
            for idx, (camera, batch) in enumerate(dataloader):
                start_time = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width

                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path:
                    og_idx = og_indices[idx]
                    for key in save_keys:
                        if key in image_dict:
                            image = image_dict[key]
                            if key in ["gt", "img"]:
                                image = torch.moveaxis(image.squeeze(), 0, -1)
                            tifffile.imwrite(output_path / f"{prefix}_{key}_t{og_idx:04d}.tif", image.cpu().numpy())

                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - start_time)).item()
                metrics_dict["fps"] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        # Per-image metrics
        per_image_metrics = {
            og_indices[i]: {k: float(v) for k, v in m.items()}
            for i, m in enumerate(metrics_dict_list)
        }

        # Aggregate
        agg_metrics = {}
        for key in metrics_dict_list[0].keys():
            values = torch.tensor([m[key] for m in metrics_dict_list])
            if get_std:
                std, mean = nanstd_mean(values)
                agg_metrics[key] = float(mean)
                agg_metrics[f"{key}_std"] = float(std)
            else:
                agg_metrics[key] = float(torch.nanmean(values))

        return agg_metrics, per_image_metrics
