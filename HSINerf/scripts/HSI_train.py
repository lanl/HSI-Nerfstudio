#!/usr/bin/env python

"""
hsi-train: A simplified wrapper around ns-train that runs only training.
"""

from __future__ import annotations

import dataclasses
import sys
import tyro
import yaml

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts.train import launch, train_loop
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import profiler


def is_dataclass_instance(obj):
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)

def override_config(cli_obj, loaded_obj, overridden_keys, prefix=""):
    """Recursively override nested config fields using dotted CLI keys."""
    for field in dataclasses.fields(cli_obj):
        key_path = f"{prefix}.{field.name}" if prefix else field.name
        
        cli_val = getattr(cli_obj, field.name)
        loaded_val = getattr(loaded_obj, field.name)

        # print(f"{key_path}: cli_val={type(cli_val)}, loaded_val={type(loaded_val)}")

        if is_dataclass_instance(cli_val) and is_dataclass_instance(loaded_val):
            # print("\n")
            override_config(cli_val, loaded_val, overridden_keys, prefix=key_path)
        elif key_path in overridden_keys:
            setattr(loaded_obj, field.name, cli_val)
            CONSOLE.log(f"\tOverriding {key_path}...")


def main(config: TrainerConfig, overridden_keys: set[str]) -> None:
    """Main training function."""

    if config.load_config:
        CONSOLE.log(f"Loading config from: {config.load_config}")
        loaded_config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)
        override_config(config, loaded_config, overridden_keys)
        config = loaded_config

    if config.data:
        CONSOLE.log("Using --data alias for --pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )

    profiler.flush_profiler(config.logging)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    config = tyro.cli(
        AnnotatedBaseConfigUnion,
        description=convert_markup_to_ansi(__doc__),
        skip_unknown=True,
    )

    # Gather CLI-specified dotted key paths
    overridden_keys = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            path = arg.lstrip("-").split("=")[0].replace("-", "_")
            overridden_keys.add(path)

    print(overridden_keys)
    main(config, overridden_keys)


if __name__ == "__main__":
    entrypoint()

get_parser_fn = lambda: tyro.extras.get_parser(AnnotatedBaseConfigUnion)
