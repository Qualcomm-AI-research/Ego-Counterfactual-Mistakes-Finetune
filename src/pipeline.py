# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Training/evaluation pipeline: builds the model, data, and DeepSpeed engine, runs the loop, and saves checkpoints/metrics."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoProcessor

from collator import QwenVideoCollator, build_collator_config
from data import build_datasets
from eval import evaluate_loss as _evaluate
from eval import extract_loss as _extract_loss
from eval import extract_optional_scalar as _extract_optional_scalar
from eval import mean_reduce as _mean_reduce
from eval import move_batch_to_device as _move_batch_to_device
from eval import run_eval as _run_eval
from model import QwenVisionLanguageModel


def _to_plain_dict(cfg: Any) -> Any:
    """Recursively convert an OmegaConf/DictConfig tree to plain Python types.

    Args:
        cfg: A config node, list, or scalar.

    Returns:
        The same structure using built-in ``dict``/``list``/scalar types.
    """
    if hasattr(cfg, "items"):
        return {key: _to_plain_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, list):
        return [_to_plain_dict(value) for value in cfg]
    return cfg


def _set_seed(seed: int) -> None:
    """Seed Python, PyTorch, and (if available) CUDA RNGs for reproducibility.

    Args:
        seed: The random seed to apply.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rank() -> int:
    """Return this process's distributed rank, or 0 if not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _world_size() -> int:
    """Return the distributed world size, or 1 if not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _is_rank_zero() -> bool:
    """Return whether this is the rank-zero (primary) process."""
    return _rank() == 0


def _print_rank_zero(message: str) -> None:
    """Write a message to stdout only from the rank-zero process.

    Uses ``tqdm.write`` so output does not corrupt active progress bars.

    Args:
        message: The text to print.
    """
    if _is_rank_zero():
        tqdm.write(message)


def _ensure_distributed_initialized() -> None:
    """Initialize the torch/DeepSpeed distributed process group if needed.

    A no-op when already initialized. Otherwise selects the nccl/gloo backend
    based on CUDA availability, fills in single-process defaults for the standard
    distributed env vars, sets the CUDA device, and calls
    ``deepspeed.init_distributed``.
    """
    if dist.is_available() and dist.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    deepspeed.init_distributed(dist_backend=backend)


def _build_sampler(dataset: Dataset[Any], shuffle: bool) -> DistributedSampler[Any] | None:
    """Build a ``DistributedSampler`` for multi-process runs.

    Args:
        dataset: The dataset to sample from.
        shuffle: Whether the sampler should shuffle indices.

    Returns:
        A ``DistributedSampler`` when the world size is greater than 1, else
        ``None`` (letting the DataLoader handle sampling directly).
    """
    if _world_size() <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=_world_size(),
        rank=_rank(),
        shuffle=shuffle,
        drop_last=False,
    )


def _resolve_train_micro_batch_size(cfg: DictConfig) -> int:
    """Determine the per-GPU micro batch size for training.

    Prefers ``trainer.strategy.config.train_micro_batch_size_per_gpu`` when set,
    falling back to ``data.batch_size``.

    Args:
        cfg: The composed Hydra configuration.

    Returns:
        The resolved micro batch size.
    """
    strategy = cfg.trainer.get("strategy")
    if strategy is not None:
        strategy_config = strategy.get("config", {})
        if "train_micro_batch_size_per_gpu" in strategy_config:
            return int(strategy_config.train_micro_batch_size_per_gpu)
    return int(cfg.data.batch_size)


def _build_dataloader(
    dataset: Dataset[Any],
    cfg: DictConfig,
    collator: QwenVideoCollator,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader[Any]:
    """Build a ``DataLoader`` with a distributed sampler when appropriate.

    Args:
        dataset: The dataset to load from.
        cfg: The ``data`` config (batch size, workers, pin/persistent settings).
        collator: The collate function for batching samples.
        shuffle: Whether to shuffle; ignored when a distributed sampler is used.
        batch_size: Optional override for ``cfg.batch_size``.

    Returns:
        The configured ``DataLoader``.
    """
    sampler = _build_sampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size if batch_size is None else batch_size),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=bool(cfg.persistent_workers) and int(cfg.num_workers) > 0,
        collate_fn=collator,
    )


def _format_progress_value(value: Any) -> str | int:
    """Format a metric value for compact display in a progress bar.

    Ints pass through; floats use scientific notation when very small or large
    and fixed 4-decimal notation otherwise; other types are stringified.

    Args:
        value: The metric value to format.

    Returns:
        The int or formatted string suitable for a tqdm postfix.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value == 0.0:
            return "0.0000"
        if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def _set_progress_postfix(progress_bar: tqdm, payload: dict[str, Any]) -> None:
    """Set a tqdm progress bar's postfix from a metrics payload.

    Args:
        progress_bar: The tqdm bar to update.
        payload: Metric name to value mapping; values are formatted for display.
    """
    progress_bar.set_postfix({key: _format_progress_value(value) for key, value in payload.items()})


def _build_deepspeed_config(cfg: DictConfig) -> dict[str, Any]:
    """Assemble the DeepSpeed engine config from the trainer strategy config.

    Computes the global train batch size from the micro batch size, gradient
    accumulation steps, and world size, and copies through the required and any
    present optional strategy keys.

    Args:
        cfg: The composed Hydra configuration.

    Returns:
        A DeepSpeed config dict ready to pass to ``deepspeed.initialize``.
    """
    strategy_cfg = _to_plain_dict(cfg.trainer.strategy)
    raw_config = strategy_cfg["config"]
    micro_batch_size = int(raw_config.get("train_micro_batch_size_per_gpu", cfg.data.batch_size))
    gradient_accumulation_steps = int(cfg.trainer.accumulate_grad_batches)
    global_batch_size = micro_batch_size * gradient_accumulation_steps * max(1, _world_size())
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "train_batch_size": global_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": float(raw_config.get("gradient_clipping", 1.0)),
        "bf16": raw_config.get("bf16", {"enabled": True}),
        "fp16": raw_config.get("fp16", {"enabled": False}),
        "zero_optimization": raw_config["zero_optimization"],
        "zero_allow_untested_optimizer": True,
    }

    optional_keys = [
        "activation_checkpointing",
        "reduce_bucket_size",
        "allgather_bucket_size",
        "steps_per_print",
        "wall_clock_breakdown",
    ]

    for optional_key in optional_keys:
        if optional_key in raw_config:
            deepspeed_config[optional_key] = raw_config[optional_key]
    return deepspeed_config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dict to ``path`` as pretty-printed, key-sorted JSON.

    Args:
        path: Destination file path.
        payload: The JSON-serializable mapping to write.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_checkpoint_root_and_tag(ckpt_path: str | None) -> tuple[str | None, str | None]:
    """Split a DeepSpeed checkpoint path into a ``(root, tag)`` pair.

    Handles a root directory containing a ``latest`` marker (tag inferred by
    DeepSpeed), a specific tag directory whose parent holds ``latest``, and a
    bare tag directory.

    Args:
        ckpt_path: The checkpoint path to resolve, or falsy for none.

    Returns:
        A ``(root, tag)`` tuple; both are ``None`` when ``ckpt_path`` is falsy,
        and ``tag`` may be ``None`` to let DeepSpeed use its ``latest`` marker.

    Raises:
        FileNotFoundError: If the path does not match a supported layout.
    """
    if not ckpt_path:
        return None, None
    path = Path(ckpt_path)
    if path.is_dir() and (path / "latest").exists():
        return str(path), None
    if path.is_dir() and path.parent.exists() and (path.parent / "latest").exists():
        return str(path.parent), path.name
    if path.is_dir() and path.parent.exists():
        return str(path.parent), path.name
    raise FileNotFoundError(f"Unsupported checkpoint path: {ckpt_path}")


def _load_fp32_state_dict(model: nn.Module, ckpt_path: str) -> None:
    """Load consolidated fp32 weights into ``model`` with key remapping.

    Unwraps common checkpoint containers (``state_dict``/``module``/``model``),
    strips a ``module.`` prefix, remaps legacy ``net.model.``/``model.`` prefixes
    onto ``qwen_model.``, skips unsupported legacy ``vision_encoder.`` weights,
    and loads non-strictly. Logs missing/unexpected key counts on rank zero.

    Args:
        model: The model to load weights into.
        ckpt_path: Path to the fp32 checkpoint file.

    Raises:
        ValueError: If the checkpoint payload is not a supported dict form.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for state_key in ["state_dict", "module", "model"]:
            if state_key in checkpoint and isinstance(checkpoint[state_key], dict):
                checkpoint = checkpoint[state_key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload in {ckpt_path}")

    state_dict = {}
    for key, value in checkpoint.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key.startswith("net.model."):
            normalized_key = "qwen_model." + normalized_key[len("net.model.") :]
        elif normalized_key.startswith("model."):
            normalized_key = "qwen_model." + normalized_key[len("model.") :]
        elif normalized_key.startswith("vision_encoder."):
            # Legacy minimal checkpoints used a TinyVisionEncoder and cannot be
            # loaded into the Qwen3.5 visual tower.
            continue
        state_dict[normalized_key] = value
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if _is_rank_zero():
        _print_rank_zero(
            f"Loaded fp32 checkpoint with {len(missing_keys)} missing and "
            f"{len(unexpected_keys)} unexpected keys: {ckpt_path}"
        )


def _maybe_export_fp32_checkpoint(
    checkpoint_root: Path,
    tag: str,
    export_dir: Path,
) -> None:
    """Export a DeepSpeed ZeRO checkpoint to consolidated fp32 weights.

    Args:
        checkpoint_root: Directory holding the DeepSpeed checkpoint tags.
        tag: The checkpoint tag to export (e.g. ``epoch-6``).
        export_dir: Directory to write the consolidated fp32 weights into
            (created if missing).
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=str(checkpoint_root),
        output_dir=str(export_dir),
        tag=tag,
        safe_serialization=False,
        exclude_frozen_parameters=False,
    )


def run_experiment(cfg: DictConfig) -> float:
    """Run training and/or evaluation for one experiment.

    Initializes distributed state, builds the datasets, model, collator, and the
    DeepSpeed engine, runs the configured train/eval loop, and writes checkpoints,
    metrics, and (optionally) the exported fp32 weights.

    Args:
        cfg: The composed Hydra configuration.

    Returns:
        The objective value used by the sweeper/caller (e.g. final loss).
    """
    _ensure_distributed_initialized()

    output_dir = Path(cfg.paths.output_dir)
    if _is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml", resolve=False)
        (output_dir / "config_resolved.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
        )

    seed = int(cfg.seed)
    _set_seed(seed + _rank())

    processor_name = cfg.model.tokenizer.pretrained_model_name_or_path
    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=True,
        use_fast=bool(cfg.model.tokenizer.get("use_fast", True)),
    )
    tokenizer = processor.tokenizer

    datasets = build_datasets(cfg.data, seed=seed)
    collator = QwenVideoCollator(processor, build_collator_config(cfg.data))
    train_micro_batch_size = _resolve_train_micro_batch_size(cfg)
    train_loader = _build_dataloader(
        datasets["train"],
        cfg.data,
        collator,
        shuffle=True,
        batch_size=train_micro_batch_size,
    )
    val_loader = _build_dataloader(
        datasets["val"],
        cfg.data,
        collator,
        shuffle=False,
        batch_size=train_micro_batch_size,
    )
    test_loader = _build_dataloader(
        datasets["test"],
        cfg.data,
        collator,
        shuffle=False,
        batch_size=train_micro_batch_size,
    )

    model = QwenVisionLanguageModel(cfg.model)
    train_enabled = bool(cfg.get("train", True))
    test_enabled = bool(cfg.get("test", True))
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path and Path(str(ckpt_path)).is_file():
        _load_fp32_state_dict(model, str(ckpt_path))

    if not train_enabled:
        device = torch.device(
            f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"
        )
        model.to(device)
        model.device = device
        generation_enabled = bool(
            cfg.get("evaluator", {}).get("evaluator_kwargs", {}).get("save_predictions", False)
        )
        predictions = []
        if test_enabled and generation_enabled:
            predictions = _run_eval(model, tokenizer, datasets["test"], collator, cfg, device)
            test_metrics = {"num_predictions": float(len(predictions))}
        else:
            test_metrics = _evaluate(model, test_loader) if test_enabled else {}
        if _is_rank_zero():
            metrics = {"test": test_metrics, "predictions_preview": predictions[:10]}
            _write_json(output_dir / "metrics.json", metrics)
            _print_rank_zero(json.dumps(metrics, indent=2, sort_keys=True))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return float(test_metrics.get("loss", 0.0))

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_cfg = _to_plain_dict(cfg.model.optimizer)
    optimizer = DeepSpeedCPUAdam(
        trainable_params,
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(float(value) for value in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg.get("eps", 1.0e-8)),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        adamw_mode=bool(optimizer_cfg.get("adamw_mode", True)),
    )

    deepspeed_config = _build_deepspeed_config(cfg)
    if _is_rank_zero():
        _write_json(output_dir / "deepspeed_config.json", deepspeed_config)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.model.scheduler.T_max),
        eta_min=float(cfg.model.scheduler.eta_min),
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=trainable_params,
        config=deepspeed_config,
        dist_init_required=False,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    resume_root, resume_tag = _resolve_checkpoint_root_and_tag(
        None if ckpt_path and Path(str(ckpt_path)).is_file() else ckpt_path
    )
    if resume_root is not None:
        _, client_state = model_engine.load_checkpoint(
            resume_root,
            tag=resume_tag,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if client_state:
            start_epoch = int(client_state.get("epoch", 0))
            global_step = int(client_state.get("global_step", 0))
            best_val_loss = float(client_state.get("best_val_loss", best_val_loss))

    max_epochs = int(cfg.trainer.max_epochs)
    max_steps = int(cfg.trainer.get("max_steps", -1))
    save_top_k = int(cfg.callbacks.model_checkpoint.get("save_top_k", 1))
    every_n_train_steps = int(cfg.callbacks.model_checkpoint.get("every_n_train_steps", 0) or 0)
    saved_tags: list[str] = []

    for epoch_idx in range(start_epoch, max_epochs):
        sampler = train_loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch_idx)

        model_engine.train()
        running_metrics = {
            "loss": 0.0,
            "language_loss": 0.0,
            "interactive_loss": 0.0,
            "interactive_weighted_loss": 0.0,
        }
        running_counts = {key: 0 for key in running_metrics}
        local_steps = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch_idx + 1}/{max_epochs}",
            disable=not _is_rank_zero(),
            dynamic_ncols=True,
        )

        for batch in progress_bar:
            batch = _move_batch_to_device(batch, model_engine.device)
            outputs = model_engine(**batch)
            loss = _extract_loss(outputs)
            model_engine.backward(loss)
            model_engine.step()

            running_metrics["loss"] += float(loss.detach().item())
            running_counts["loss"] += 1
            for metric_key in ["language_loss", "interactive_loss", "interactive_weighted_loss"]:
                metric_value = _extract_optional_scalar(outputs, metric_key)
                if metric_value is None:
                    continue
                running_metrics[metric_key] += float(metric_value.detach().item())
                running_counts[metric_key] += 1
            local_steps += 1

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                log_payload = {
                    "epoch": epoch_idx + 1,
                    "global_step": global_step,
                    "loss": float(loss.detach().item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                for metric_key in [
                    "language_loss",
                    "interactive_loss",
                    "interactive_weighted_loss",
                    "interactive_loss_weight",
                ]:
                    metric_value = _extract_optional_scalar(outputs, metric_key)
                    if metric_value is not None:
                        log_payload[metric_key] = float(metric_value.detach().item())

                if _is_rank_zero():
                    _set_progress_postfix(progress_bar, log_payload)

                if _is_rank_zero() and global_step % int(cfg.trainer.log_every_n_steps) == 0:
                    tqdm.write(json.dumps(log_payload))

                if every_n_train_steps > 0 and global_step % every_n_train_steps == 0:
                    checkpoint_tag = f"step-{global_step}"
                    client_state = {
                        "epoch": epoch_idx,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "record": log_payload,
                    }
                    model_engine.save_checkpoint(
                        str(checkpoint_root),
                        tag=checkpoint_tag,
                        client_state=client_state,
                        save_latest=True,
                    )

                if max_steps > 0 and global_step >= max_steps:
                    break

        train_metrics = {}
        for metric_key, metric_total in running_metrics.items():
            metric_count = running_counts[metric_key]
            if metric_count == 0:
                continue
            train_metrics[metric_key] = _mean_reduce(
                metric_total / metric_count, model_engine.device
            )
        val_metrics = _evaluate(model_engine, val_loader)
        record = {
            "epoch": epoch_idx + 1,
            "global_step": global_step,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "lr": float(scheduler.get_last_lr()[0]),
        }
        for metric_key in ["language_loss", "interactive_loss", "interactive_weighted_loss"]:
            if metric_key in train_metrics:
                record[f"train_{metric_key}"] = train_metrics[metric_key]
            if metric_key in val_metrics:
                record[f"val_{metric_key}"] = val_metrics[metric_key]
        history.append(record)

        checkpoint_tag = f"epoch-{epoch_idx + 1}"
        client_state = {
            "epoch": epoch_idx + 1,
            "global_step": global_step,
            "best_val_loss": min(best_val_loss, val_metrics["loss"]),
            "record": record,
        }
        model_engine.save_checkpoint(
            str(checkpoint_root), tag=checkpoint_tag, client_state=client_state, save_latest=True
        )
        saved_tags.append(checkpoint_tag)
        while len(saved_tags) > save_top_k and _is_rank_zero():
            oldest_tag = saved_tags.pop(0)
            oldest_path = checkpoint_root / oldest_tag
            if oldest_path.exists():
                for child in oldest_path.iterdir():
                    if child.is_file():
                        child.unlink()
                    else:
                        for nested in child.rglob("*"):
                            if nested.is_file():
                                nested.unlink()
                        for nested in sorted(child.rglob("*"), reverse=True):
                            if nested.is_dir():
                                nested.rmdir()
                        child.rmdir()
                oldest_path.rmdir()

        best_val_loss = min(best_val_loss, val_metrics["loss"])
        if max_steps > 0 and global_step >= max_steps:
            break

    test_metrics = _evaluate(model_engine, test_loader)

    if _is_rank_zero():
        tokenizer.save_pretrained(output_dir / "tokenizer")
        metrics = {
            "train": history[-1] if history else {},
            "val": {"loss": best_val_loss},
            "test": test_metrics,
            "history": history,
        }
        _write_json(output_dir / "metrics.json", metrics)

        if bool(cfg.get("export_fp32_weights", False)) and history:
            _maybe_export_fp32_checkpoint(
                checkpoint_root=checkpoint_root,
                tag=history[-1]["epoch"] and f"epoch-{history[-1]['epoch']}",
                export_dir=output_dir / "fp32_export",
            )

        _print_rank_zero(json.dumps(metrics, indent=2, sort_keys=True))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return best_val_loss
