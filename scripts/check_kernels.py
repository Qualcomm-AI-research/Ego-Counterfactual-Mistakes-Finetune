# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Verify the Qwen3.5 hybrid-attention fast path is installed and usable.

The fast path in ``transformers`` gates on four symbols spread over two kernel
packages::

    is_fast_path_available = all((
        causal_conv1d_fn, causal_conv1d_update,           # causal-conv1d
        chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,  # flash-linear-attention
    ))

If any is missing, transformers logs "The fast path is not available because one
of the required library is not installed. Falling back to torch implementation."
and every run silently takes the much slower pure-torch path.

Two modes, because the checks have different requirements:

``--build``
    Package/version checks only, via distribution metadata. Imports nothing from
    ``fla`` or ``triton``, so it runs inside ``docker build`` where no GPU or CUDA
    driver is present. Importing ``fla.ops.*`` there fails with "0 active drivers"
    because its module-level ``@triton.autotune`` decorators initialize the Triton
    driver at import time.

default (no flag)
    The full check: imports the kernels and mirrors the real
    ``is_fast_path_available`` condition. Needs a GPU, so run it inside a
    ``--gpus all`` container (``make check-kernels``).
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

# flash-linear-attention refuses to run the gated ``chunk_bwd_dqkwg`` backward on
# Hopper GPUs with Triton in this window, because Triton miscompiles it and
# produces silently incorrect gradients (fla issue #640). Torch pins Triton
# exactly, so in practice this constrains the base image tag: torch 2.7.1 pins
# 3.3.1 (below the window, OK), torch 2.8.0 pins 3.4.0 (inside, unusable).
TRITON_MIN = (3, 3)
TRITON_BAD_LO = (3, 4, 0)
TRITON_BAD_HI = (3, 7, 1)

# Distribution names Triton ships under, depending on how torch was installed.
TRITON_DISTS = ("triton", "pytorch-triton")


def _parse_version(text: str) -> tuple[int, ...]:
    """Parse a leading dotted numeric version into a tuple.

    Trailing suffixes (``+cu126``, ``.post1``, ``rc1``) are ignored so the
    comparison only considers the numeric release segment.

    Args:
        text: A version string such as ``"3.3.1"`` or ``"3.4.0+git1234"``.

    Returns:
        The leading numeric components, e.g. ``(3, 3, 1)``.
    """
    parts: list[int] = []
    for chunk in text.split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _triton_version() -> tuple[str, str]:
    """Look up the installed Triton distribution name and version.

    Uses distribution metadata rather than ``import triton`` so this is safe to
    call without a CUDA driver.

    Returns:
        A ``(distribution_name, version)`` pair.

    Raises:
        SystemExit: If no known Triton distribution is installed.
    """
    for dist in TRITON_DISTS:
        try:
            return dist, version(dist)
        except PackageNotFoundError:
            continue
    raise SystemExit(
        "FAIL: Triton is not installed. flash-linear-attention's kernels need "
        f"triton >= {'.'.join(map(str, TRITON_MIN))}."
    )


def _check_triton_window(triton_version: str) -> None:
    """Assert the Triton version can run fla's gated backward kernels.

    Args:
        triton_version: The installed Triton version string.

    Raises:
        SystemExit: If Triton is too old, or inside the known-bad window.
    """
    parsed = _parse_version(triton_version)
    if parsed < TRITON_MIN:
        raise SystemExit(
            f"FAIL: triton {triton_version} is older than the "
            f"{'.'.join(map(str, TRITON_MIN))} floor required by flash-linear-attention."
        )
    if TRITON_BAD_LO <= parsed < TRITON_BAD_HI:
        raise SystemExit(
            f"FAIL: triton {triton_version} is inside the known-bad window "
            f"[{'.'.join(map(str, TRITON_BAD_LO))}, {'.'.join(map(str, TRITON_BAD_HI))}), "
            "where Triton miscompiles fla's gated chunk_bwd_dqkwg backward on Hopper "
            "GPUs (fla issue #640). This raises a RuntimeError in the first backward "
            "pass. Pick a base image whose pinned Triton is outside this range: "
            "torch 2.7.1 pins triton 3.3.1, torch 2.8.0 pins the unusable 3.4.0."
        )


def check_build() -> None:
    """Run the GPU-free package and version checks used during ``docker build``."""
    try:
        torch_version = version("torch")
    except PackageNotFoundError as exc:  # pragma: no cover - torch is in the base image
        raise SystemExit("FAIL: torch is not installed.") from exc

    try:
        fla_version = version("flash-linear-attention")
    except PackageNotFoundError as exc:
        raise SystemExit(
            "FAIL: flash-linear-attention is not installed. Without it, "
            "chunk_gated_delta_rule is None and the Qwen3.5 fast path is disabled "
            "(transformers falls back to the slow torch implementation)."
        ) from exc

    triton_dist, triton_version = _triton_version()
    _check_triton_window(triton_version)

    # causal_conv1d is a compiled extension; importing it does not touch a device,
    # so this is safe at build time and confirms the nvcc build actually linked.
    try:
        import causal_conv1d
        import causal_conv1d_cuda  # noqa: F401
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except Exception as exc:
        raise SystemExit(f"FAIL: causal-conv1d is not importable: {type(exc).__name__}: {exc}")

    if not all((causal_conv1d_fn, causal_conv1d_update)):
        raise SystemExit("FAIL: causal-conv1d did not export both fast-path entry points.")

    print(
        f"build check OK: torch {torch_version}, {triton_dist} {triton_version}, "
        f"causal_conv1d {causal_conv1d.__version__}, flash-linear-attention {fla_version}"
    )
    print("note: the full kernel check needs a GPU -- run `make check-kernels`.")


def check_runtime() -> None:
    """Run the full fast-path check, importing the kernels. Requires a GPU."""
    import torch

    triton_dist, triton_version = _triton_version()
    _check_triton_window(triton_version)

    if not torch.cuda.is_available():
        raise SystemExit(
            "FAIL: no CUDA device visible. Run this inside a GPU container, e.g. "
            "`make check-kernels`, or use `--build` for the package-only check."
        )

    import causal_conv1d
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    try:
        import fla
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except Exception as exc:
        raise SystemExit(
            f"FAIL: could not import flash-linear-attention kernels: "
            f"{type(exc).__name__}: {exc}"
        )

    symbols = {
        "causal_conv1d_fn": causal_conv1d_fn,
        "causal_conv1d_update": causal_conv1d_update,
        "chunk_gated_delta_rule": chunk_gated_delta_rule,
        "fused_recurrent_gated_delta_rule": fused_recurrent_gated_delta_rule,
    }
    missing = [name for name, symbol in symbols.items() if symbol is None]
    if missing:
        raise SystemExit(
            "FAIL: Qwen3.5 fast path unavailable, these resolved to None: "
            f"{', '.join(missing)}"
        )

    names = {torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())}
    capabilities = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    is_hopper = any(major == 9 for major, _ in capabilities)

    print(
        f"fast path OK: torch {torch.__version__}, {triton_dist} {triton_version}, "
        f"causal_conv1d {causal_conv1d.__version__}, fla {fla.__version__}"
    )
    print(f"devices: {', '.join(sorted(names))} (compute {sorted(capabilities)})")
    if not is_hopper:
        print(
            "note: no Hopper GPU here, so the Triton-miscompile guard that trips on "
            "H100 would not fire on this machine even with a bad Triton. The version "
            "check above still covers it."
        )


def main() -> None:
    """Parse arguments and run the requested check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--build",
        action="store_true",
        help="package/version checks only; no GPU or kernel imports (for docker build).",
    )
    args = parser.parse_args()
    if args.build:
        check_build()
    else:
        check_runtime()


if __name__ == "__main__":
    sys.exit(main())
