# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Export a DeepSpeed ZeRO checkpoint to consolidated fp32 weights."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_TAG_RE = re.compile(r"^(?:epoch|step)-\d+$")
_MODEL_STATE_RE = re.compile(r"(?:zero|fp16|bf16)_pp_rank_\d+_mp_rank_\d+_model_states\.pt$")


def _looks_like_tag_dir(path: Path) -> bool:
    """Return whether ``path`` is a DeepSpeed checkpoint tag directory.

    A tag directory is identified either by an ``epoch-N``/``step-N`` name or by
    containing at least one ZeRO model-state file (e.g.
    ``bf16_pp_rank_0_mp_rank_00_model_states.pt``).

    Args:
        path: Candidate directory to inspect.

    Returns:
        True if ``path`` looks like a checkpoint tag directory, else False.
    """
    if not path.is_dir():
        return False
    if _TAG_RE.match(path.name):
        return True
    return any(_MODEL_STATE_RE.match(child.name) for child in path.iterdir() if child.is_file())


def _resolve_checkpoint_root_and_tag(
    checkpoint_path: Path, tag: str | None
) -> tuple[Path, str | None]:
    """Split a checkpoint path into a DeepSpeed root directory and tag.

    Handles three cases: an explicit ``tag`` is passed through unchanged; a root
    containing a ``latest`` file resolves to that root with no tag; and a path
    that is itself a tag directory is split into its parent (root) and its own
    name (tag).

    Args:
        checkpoint_path: The checkpoint root or a specific tag directory.
        tag: An explicit checkpoint tag, or None to infer it.

    Returns:
        A ``(root, tag)`` tuple where ``tag`` may be None (use DeepSpeed's
        ``latest``).
    """
    checkpoint_path = checkpoint_path.expanduser()
    if tag is not None:
        return checkpoint_path, tag
    if (checkpoint_path / "latest").exists():
        return checkpoint_path, None
    if _looks_like_tag_dir(checkpoint_path):
        return checkpoint_path.parent, checkpoint_path.name
    return checkpoint_path, None


def main() -> None:
    """Parse CLI args and convert the ZeRO checkpoint to fp32 weights."""
    parser = argparse.ArgumentParser(
        description="Export a DeepSpeed ZeRO checkpoint to fp32 weights."
    )
    parser.add_argument(
        "checkpoint_root",
        type=Path,
        help="DeepSpeed checkpoint root directory, or a specific tag directory like checkpoints/epoch-6.",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write the consolidated fp32 weights into."
    )
    parser.add_argument("--tag", default=None, help="Optional checkpoint tag, e.g. epoch-1.")
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Write safetensors shards instead of a PyTorch state dict.",
    )
    args = parser.parse_args()

    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    checkpoint_root, tag = _resolve_checkpoint_root_and_tag(args.checkpoint_root, args.tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Exporting DeepSpeed checkpoint root={checkpoint_root} tag={tag or '<latest>'} to {args.output_dir}"
    )
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=str(checkpoint_root),
        output_dir=str(args.output_dir),
        tag=tag,
        max_shard_size=None,
        safe_serialization=args.safe_serialization,
        exclude_frozen_parameters=False,
    )


if __name__ == "__main__":
    main()
