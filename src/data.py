# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Dataset construction: build component datasets and the weighted training mixture from config."""

from __future__ import annotations

from datasets.common import *
from datasets.ego4d.goalstep import Ego4DGoalStepDataset
from datasets.qualcomm_interactive_cooking.ego_comist import EgoCoMist
from datasets.qualcomm_interactive_cooking.ego_mc_bench import EgoMCBench
from datasets.qualcomm_interactive_cooking.neurips2025_non_reactive import (
    CaptainCook4DInstructDataset,
)


def _require_dataset_roots(spec: dict[str, Any], name: str, required_roots: list[str]) -> None:
    """Validate that required dataset-root paths are set (not ``None``).

    The experiment configs leave ``data_paths`` roots ``null`` so they must be
    supplied on the command line (see the README). Missing roots otherwise surface
    later as an opaque ``Path(None)`` ``TypeError``; this raises an actionable
    error instead.

    Args:
        spec: The dataset config spec, with roots already injected.
        name: Human-readable dataset name for the error message.
        required_roots: The ``data_paths`` keys this dataset requires.

    Raises:
        ValueError: If any required root is missing or ``None``.
    """
    missing = [key for key in required_roots if spec.get(key) is None]
    if missing:
        joined = ", ".join(missing)
        overrides = " ".join(f"data_paths.{key}=/path/to/dataset" for key in missing)
        raise ValueError(
            f"{name} requires dataset root(s) {joined}, but they are null. "
            f"Set them on the command line, e.g. python scripts/run.py experiment=train "
            f"{overrides} (see README)."
        )


def _build_component_dataset(spec: dict[str, Any]) -> Dataset[CookingSample]:
    """Instantiate a single dataset from its config spec.

    The dataset class is selected from the spec's ``_target_`` suffix or ``name``.

    Args:
        spec: Config dictionary for one dataset, including ``_target_``/``name``
            and the keyword arguments forwarded to the dataset constructor.

    Returns:
        The constructed dataset.

    Raises:
        ValueError: If the target does not match a known dataset, or a required
            dataset root is missing.
    """
    target = str(spec.get("_target_", ""))
    name = str(spec.get("name", ""))
    if target.endswith("CaptainCook4DInstruct") or name == "captaincook4d_instruct":
        _require_dataset_roots(spec, "CaptainCook4DInstruct", ["captaincook4d_root"])
        return CaptainCook4DInstructDataset(**spec)
    if target.endswith("Ego4DGoalStepInstruct"):
        _require_dataset_roots(spec, "Ego4DGoalStep", ["ego4d_root"])
        return Ego4DGoalStepDataset(**spec)
    if target.endswith("EgoCoMist") or name == "qualcomm_counterfactual_mistakes_hf":
        _require_dataset_roots(
            spec, "EgoCoMist", ["captaincook4d_root", "ego4d_root", "egoexo4d_root"]
        )
        return EgoCoMist(**spec)
    if target.endswith("EgoMCBench") or name == "ego_mc_bench":
        return EgoMCBench(**spec)
    raise ValueError(f"Unsupported dataset target: {target}")


def build_datasets(cfg: Any, seed: int) -> dict[str, Dataset[CookingSample]]:
    """Build the train/val/test datasets from a data config.

    A single dataset spec is shared across all splits. A ``dataset_list`` is
    combined into a :class:`WeightedMixtureDataset` and then split into train,
    validation, and test subsets according to ``create_train_val_test_split``.

    Args:
        cfg: The ``data`` config (Hydra ``DictConfig`` or plain dict).
        seed: Random seed for the mixture sampling and the split.

    Returns:
        Mapping with ``"train"``, ``"val"``, and ``"test"`` datasets.

    Raises:
        ValueError: If the dataset list is empty or the split is malformed.
    """
    plain_cfg = _to_plain_dict(cfg)
    dataset_cfg = plain_cfg.get("dataset", {})
    if "dataset_list" not in dataset_cfg:
        dataset = _build_component_dataset(dict(dataset_cfg))
        return {"train": dataset, "val": dataset, "test": dataset}

    dataset_list = [
        _build_component_dataset(dict(spec)) for spec in dataset_cfg.get("dataset_list", [])
    ]
    if not dataset_list:
        raise ValueError("cfg.data.dataset.dataset_list is empty")

    probabilities = [
        float(value) for value in dataset_cfg.get("probabilities", [1.0] * len(dataset_list))
    ]
    fractions = [
        float(value) for value in plain_cfg.get("create_train_val_test_split", [0.98, 0.01, 0.01])
    ]
    if len(fractions) != 3:
        raise ValueError("create_train_val_test_split must contain train, val, and test fractions")

    train_length = dataset_cfg.get("train_length")
    if train_length is None:
        total_length = int(
            dataset_cfg.get("length") or sum(len(dataset) for dataset in dataset_list)
        )
        train_len = max(1, int(total_length * fractions[0]))
        val_len = max(1, int(total_length * fractions[1]))
        test_len = max(1, total_length - train_len - val_len)
        while train_len + val_len + test_len > total_length:
            if train_len > val_len and train_len > 1:
                train_len -= 1
            elif val_len > 1:
                val_len -= 1
            else:
                test_len -= 1
        while train_len + val_len + test_len < total_length:
            train_len += 1
    else:
        train_len = max(1, int(train_length))
        if fractions[0] <= 0:
            raise ValueError("train fraction must be positive when dataset.train_length is set")
        val_len = (
            max(1, int(round(train_len * fractions[1] / fractions[0]))) if fractions[1] > 0 else 0
        )
        test_len = (
            max(1, int(round(train_len * fractions[2] / fractions[0]))) if fractions[2] > 0 else 0
        )
        total_length = train_len + val_len + test_len

    combined = WeightedMixtureDataset(
        datasets=dataset_list,
        probabilities=probabilities,
        length=total_length,
        seed=seed,
    )

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        combined, [train_len, val_len, test_len], generator=generator
    )
    return {"train": train_set, "val": val_set, "test": test_set}
