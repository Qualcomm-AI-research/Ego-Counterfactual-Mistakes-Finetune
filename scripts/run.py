# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Entrypoint that configures the environment and launches a Hydra/DeepSpeed experiment."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _apply_hf_env() -> None:
    """Set Hugging Face cache env vars from the experiment config.

    This must run before importing any HF library (transformers, datasets,
    huggingface_hub), so the cache locations in the `hf_env` config block take
    effect. We pre-compose the config from the CLI overrides to read `hf_env`.
    """
    overrides = [arg for arg in sys.argv[1:] if "=" in arg and not arg.startswith("hydra.")]
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="train", overrides=overrides)
    except Exception as exc:
        # Best-effort pre-composition: if it fails (e.g. bad override), skip
        # applying hf_env here. The real Hydra entrypoint will surface the
        # underlying config error later; warn so the user knows why HF cache
        # env vars were not set.
        print(
            f"Warning: skipping hf_env setup, config pre-composition failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return
    hf_env = cfg.get("hf_env")
    if not hf_env:
        return
    for key, value in OmegaConf.to_container(hf_env, resolve=True).items():
        if value is not None:
            os.environ[str(key)] = str(value)


_apply_hf_env()

from pipeline import run_experiment


def _strip_launcher_args() -> None:
    """Remove launcher-injected `--local_rank` args from ``sys.argv``.

    Distributed launchers (torchrun, DeepSpeed) inject a ``--local_rank`` flag
    that Hydra does not understand. We drop it (in both ``--local_rank VALUE``
    and ``--local_rank=VALUE`` forms) so Hydra can parse the remaining
    overrides cleanly.
    """
    cleaned_args: list[str] = []
    skip_next = False
    for arg in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--local_rank", "--local-rank"}:
            skip_next = True
            continue
        if arg.startswith("--local_rank=") or arg.startswith("--local-rank="):
            continue
        cleaned_args.append(arg)
    sys.argv = cleaned_args


def _requested_device_count(devices: Any) -> int:
    """Normalize a Hydra ``trainer.devices`` value to an integer count.

    Args:
        devices: The configured device spec, which may be an int, a list/tuple
            of device ids, or a numeric string.

    Returns:
        The number of requested devices, defaulting to 1 for unrecognized specs.
    """
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str) and devices.isdigit():
        return int(devices)
    return 1


def _maybe_relaunch_with_torchrun(cfg: DictConfig) -> None:
    """Re-exec the process under ``torch.distributed.run`` for multi-GPU jobs.

    When more than one device is requested but the process is not already
    running distributed (single-process launch), we relaunch via torchrun with
    one process per device. A ``STREAMING_VLMS_TORCHRUN_REEXEC`` guard and a
    ``WORLD_SIZE > 1`` check prevent infinite relaunch loops. No-op for
    single-device runs.

    Args:
        cfg: The composed Hydra configuration; reads ``trainer.devices`` and
            ``paths.output_dir``.
    """
    if os.environ.get("STREAMING_VLMS_TORCHRUN_REEXEC") == "1":
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return

    device_count = _requested_device_count(cfg.trainer.get("devices", 1))
    if device_count <= 1:
        return

    args = list(sys.argv[1:])
    if not any(arg.startswith("hydra.run.dir=") for arg in args):
        args.append(f"hydra.run.dir={cfg.paths.output_dir}")

    os.environ["STREAMING_VLMS_TORCHRUN_REEXEC"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={device_count}",
        str(Path(sys.argv[0]).resolve()),
        *args,
    ]
    print(f"Relaunching with {device_count} distributed processes: {' '.join(command)}", flush=True)
    os.execv(sys.executable, command)


_strip_launcher_args()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> float:
    """Hydra entrypoint: relaunch under torchrun if needed and run the experiment.

    Args:
        cfg: The composed Hydra configuration.

    Returns:
        The experiment's objective value (e.g. validation/test loss).
    """
    _maybe_relaunch_with_torchrun(cfg)
    return run_experiment(cfg)


if __name__ == "__main__":
    main()
