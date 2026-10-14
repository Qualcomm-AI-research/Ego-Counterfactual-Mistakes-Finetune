# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Shared dataset types and helpers: the CookingSample record, dataset base classes, frame discovery/extraction, and text utilities."""

from __future__ import annotations

import json
import pickle
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, random_split

NEXT_INSTRUCTION_RESPONSE = "Next Instruction"
SUCCESS_RESPONSE = "Success: 0 seconds ago, you completed that step successfully."


@dataclass
class CookingSample:
    """One streaming cooking example.

    Holds the system prompt, instruction, timed responses, and the frame paths and
    timestamps (both the windowed frames and, optionally, the full-video frames).
    """

    system_prompt: str
    instruction: str
    responses: list[str]
    response_timestamps: list[float]
    frame_paths: list[str]
    frame_timestamps: list[float]
    source: str
    metadata: dict[str, Any]
    video_frame_paths: list[str] = field(default_factory=list)
    video_frame_timestamps: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.video_frame_paths:
            self.video_frame_paths = list(self.frame_paths)
        if not self.video_frame_timestamps:
            self.video_frame_timestamps = list(self.frame_timestamps)

    @property
    def response(self) -> str:
        """Return all responses joined into a single string."""
        return " ".join(self.responses)


class ListDataset(Dataset[CookingSample]):
    """In-memory dataset of :class:`CookingSample` items.

    Optionally applies per-index weighted resampling and attaches temporal-jitter
    config to each returned sample.
    """

    def __init__(
        self,
        samples: list[CookingSample],
        temporal_jitter: dict[str, Any] | None = None,
        sample_weights: list[float] | None = None,
    ) -> None:
        self.samples = samples
        self.temporal_jitter = temporal_jitter
        if sample_weights is not None and len(sample_weights) != len(samples):
            raise ValueError("sample_weights must have the same length as samples")
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CookingSample:
        if self.sample_weights is None:
            sample = self.samples[index]
        else:
            rng = random.Random(index)
            sample_idx = rng.choices(range(len(self.samples)), weights=self.sample_weights, k=1)[0]
            sample = self.samples[sample_idx]
        if self.temporal_jitter is None:
            return sample
        metadata = dict(sample.metadata)
        metadata["temporal_jitter"] = self.temporal_jitter
        return replace(sample, metadata=metadata)


class WeightedMixtureDataset(Dataset[CookingSample]):
    """Virtual dataset that samples component datasets by probability.

    Each index deterministically (seeded) picks a component dataset and a random
    item from it, yielding a fixed-length mixture.
    """

    def __init__(
        self,
        datasets: list[Dataset[CookingSample]],
        probabilities: list[float],
        length: int,
        seed: int,
    ) -> None:
        if len(datasets) != len(probabilities):
            raise ValueError("datasets and probabilities must have the same length")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("all component datasets must be non-empty")
        self.datasets = datasets
        self.probabilities = probabilities
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> CookingSample:
        rng = random.Random(self.seed + index)
        dataset_idx = rng.choices(range(len(self.datasets)), weights=self.probabilities, k=1)[0]
        dataset = self.datasets[dataset_idx]
        sample_idx = rng.randrange(len(dataset))
        data_sample = dataset[sample_idx]
        return data_sample


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Recursively convert a config (e.g. OmegaConf) into plain dicts and lists.

    Args:
        cfg: A mapping, list, or scalar, possibly from a config library.

    Returns:
        The same structure using built-in ``dict``/``list``/scalar types.
    """
    if hasattr(cfg, "items"):
        return {key: _to_plain_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, list):
        return [_to_plain_dict(value) for value in cfg]
    return cfg


def _normalize_text(text: str | None) -> str:
    """Collapse whitespace, capitalize the first letter, and ensure end punctuation.

    Args:
        text: The text to normalize, or ``None``.

    Returns:
        The normalized text, or an empty string if the input is empty/``None``.
    """
    if text is None:
        return ""
    text = " ".join(str(text).strip().split())
    if not text:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _response_prefix(response_type: str | None) -> str | None:
    """Map a raw response type to its canonical display prefix.

    Args:
        response_type: The raw response type string, or ``None``.

    Returns:
        The canonical prefix (e.g. ``"Feedback"``, ``"Success"``), or ``None`` if
        the type is unknown or ``None``.
    """
    if response_type is None:
        return None
    normalized_type = str(response_type).strip().lower().replace("-", "_")
    if normalized_type == "feedback_reply":
        return "Feedback-Reply"
    if normalized_type == "instruction_clarification":
        return "Instruction-Clarification"
    if "success" in normalized_type:
        return "Success"
    if (
        "feedback" in normalized_type
        or "mistake" in normalized_type
        or "failure" in normalized_type
    ):
        return "Feedback"
    return None


def _format_response(response_type: str | None, text: str | None) -> str:
    """Format a response as ``"<Prefix>: <normalized body>"``.

    Args:
        response_type: The response type used to derive the display prefix.
        text: The raw response text.

    Returns:
        The formatted response, or ``None`` if the body indicates all steps are
        finished.
    """
    body = _normalize_text(_strip_eccv_prefix(str(text or "")))
    prefix = _response_prefix(response_type)
    if "you have finished all the steps" in body.lower():
        return None
    else:
        return f"{prefix}: {body}"


def _append_next_instruction(
    responses: list[str], include_next_instruction: bool = True
) -> list[str]:
    """Append the next-instruction marker to responses if not already present.

    Args:
        responses: The response texts to extend (modified in place).
        include_next_instruction: Whether to append the marker at all.

    Returns:
        The (possibly extended) responses list.
    """
    if include_next_instruction and (not responses or responses[-1] != NEXT_INSTRUCTION_RESPONSE):
        responses.append(NEXT_INSTRUCTION_RESPONSE)
    return responses


def _append_next_instruction_timestamped(
    responses: list[str],
    response_timestamps: list[float],
    next_instruction_timestamp: float | None,
    include_next_instruction: bool = True,
) -> tuple[list[str], list[float]]:
    """Append the next-instruction marker and its timestamp to the response lists.

    Uses ``next_instruction_timestamp`` when provided, otherwise reuses the last
    response timestamp.

    Args:
        responses: The response texts to extend (modified in place).
        response_timestamps: The parallel timestamps to extend (modified in place).
        next_instruction_timestamp: Timestamp for the marker, or ``None`` to reuse
            the last timestamp.
        include_next_instruction: Whether to append the marker at all.

    Returns:
        The (possibly extended) responses and timestamps lists.
    """
    if include_next_instruction and next_instruction_timestamp is not None:
        responses.append(NEXT_INSTRUCTION_RESPONSE)
        response_timestamps.append(float(next_instruction_timestamp))
    if include_next_instruction and next_instruction_timestamp is None:
        responses.append(NEXT_INSTRUCTION_RESPONSE)
        response_timestamps.append(response_timestamps[-1])
    return responses, response_timestamps


def _frames_and_timestamps_between(
    video_to_frames: dict[str, list[str]],
    video_id: str,
    fps: float,
    start_time: float,
    end_time: float,
    max_frames: int | None = None,
) -> tuple[list[str], list[float]]:
    """Return frame paths and timestamps within a time window for one video.

    Args:
        video_to_frames: Mapping of video id to its ordered frame paths.
        video_id: The video to select frames from.
        fps: Frame rate used to convert times to frame indices.
        start_time: Window start in seconds.
        end_time: Window end in seconds.
        max_frames: If set, keep only the last ``max_frames`` frames in the window.

    Returns:
        A tuple of the selected frame paths and their timestamps (seconds).
    """
    frames = video_to_frames.get(video_id, [])
    if not frames:
        return [], []
    start_idx = max(0, int(start_time * fps))
    end_idx = min(len(frames), max(start_idx + 1, int(end_time * fps) + 1))
    frame_indices = list(range(start_idx, end_idx))
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[-max_frames:]
    return [frames[frame_idx] for frame_idx in frame_indices], [
        frame_idx / fps for frame_idx in frame_indices
    ]


def _video_frame_timestamps(
    video_to_frames: dict[str, list[str]], fps: float
) -> dict[str, list[float]]:
    """Compute per-frame timestamps for every video from its frame count.

    Args:
        video_to_frames: Mapping of video id to its ordered frame paths.
        fps: Frame rate used to convert indices to timestamps.

    Returns:
        Mapping of video id to a list of frame timestamps (seconds).
    """
    return {
        video_id: [frame_idx / fps for frame_idx in range(len(frame_paths))]
        for video_id, frame_paths in video_to_frames.items()
    }


def _select_frames_between(
    frame_paths: list[str],
    frame_timestamps: list[float],
    start_time: float,
    end_time: float,
    max_frames: int | None = None,
) -> tuple[list[str], list[float]]:
    """Select frames whose timestamps fall within a window from parallel lists.

    Falls back to the single nearest frame when none lie within the window.

    Args:
        frame_paths: Ordered frame paths.
        frame_timestamps: Timestamps parallel to ``frame_paths`` (seconds).
        start_time: Window start in seconds.
        end_time: Window end in seconds.
        max_frames: If set, keep only the first ``max_frames`` frames in the window.

    Returns:
        A tuple of the selected frame paths and their timestamps (seconds).
    """
    frame_indices = [
        idx for idx, timestamp in enumerate(frame_timestamps) if start_time <= timestamp <= end_time
    ]
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices and frame_paths:
        nearest_idx = min(
            range(len(frame_paths)), key=lambda idx: abs(frame_timestamps[idx] - start_time)
        )
        frame_indices = [nearest_idx]
    return [frame_paths[idx] for idx in frame_indices], [
        frame_timestamps[idx] for idx in frame_indices
    ]


def _jitter_window_start(
    start_time: float,
    response_timestamps: list[float],
    cfg: dict[str, Any],
    *,
    amount_key: str,
    random_scale: float,
) -> float:
    """Randomly jitter a segment's start time earlier, subject to config and bounds.

    With probability ``temporal_jitter_prob`` shifts the start forward by a random
    amount, never past a buffer before the first response.

    Args:
        start_time: The original segment start time in seconds.
        response_timestamps: Timestamps of the segment's responses (seconds).
        cfg: Jitter configuration (probabilities, amounts, buffers).
        amount_key: Config key for the jitter amount, with a fallback key.
        random_scale: Scale on the random component of the jitter amount.

    Returns:
        The (possibly jittered) start time in seconds.
    """
    temporal_jitter_prob = float(cfg.get("temporal_jitter_prob", 0.0))
    fallback_amount_key = "temporal_jitter_amount_instructions"
    if amount_key == fallback_amount_key:
        fallback_amount_key = "temporal_jitter_amount"
    temporal_jitter_amount = float(cfg.get(amount_key, cfg.get(fallback_amount_key, 0.0)))
    if (
        temporal_jitter_prob <= 0.0
        or temporal_jitter_amount <= 0.0
        or random.random() >= temporal_jitter_prob
    ):
        return start_time

    first_instruction_time_buffer = float(cfg.get("first_instruction_time_buffer", 2.0))
    first_response_time = (
        min(response_timestamps)
        if response_timestamps
        else start_time + first_instruction_time_buffer
    )
    jitter_amount = (1.0 - random_scale * random.random()) * temporal_jitter_amount
    latest_start_time = first_response_time - first_instruction_time_buffer
    return max(0.0, min(start_time + jitter_amount, latest_start_time))


def _jitter_feedback_timestamps(
    responses: list[str],
    response_timestamps: list[float],
    cfg: dict[str, Any],
) -> list[float]:
    """Randomly delay feedback response timestamps, capped at the last response.

    With probability ``temporal_jitter_prob`` shifts each ``Feedback`` response's
    timestamp later by a random amount, never past the maximum response time.

    Args:
        responses: The response texts (used to detect feedback responses).
        response_timestamps: Timestamps parallel to ``responses`` (seconds).
        cfg: Jitter configuration (probabilities and amounts).

    Returns:
        The (possibly jittered) response timestamps.
    """
    temporal_jitter_prob = float(cfg.get("temporal_jitter_prob", 0.0))
    temporal_jitter_amount = float(cfg.get("temporal_jitter_amount_feedbacks", 0.0))
    if (
        temporal_jitter_prob <= 0.0
        or temporal_jitter_amount <= 0.0
        or not response_timestamps
        or random.random() >= temporal_jitter_prob
    ):
        return response_timestamps

    max_response_time = max(response_timestamps)
    jittered_timestamps = list(response_timestamps)
    for response_idx, response in enumerate(responses):
        if "Feedback" in response:
            jittered_timestamps[response_idx] = min(
                response_timestamps[response_idx] + random.random() * temporal_jitter_amount,
                max_response_time,
            )
    return jittered_timestamps


def _temporal_jitter_config(
    frame_limit: int,
    cfg: dict[str, Any],
    *,
    amount_key: str,
    random_scale: float,
    jitter_feedback_timestamps: bool = False,
) -> dict[str, Any] | None:
    """Build a temporal-jitter spec, or ``None`` when jitter is disabled.

    Args:
        frame_limit: Max frames the jittered window may span.
        cfg: Jitter configuration (probabilities and amounts).
        amount_key: Config key for the jitter amount.
        random_scale: Scale on the random component of the jitter amount.
        jitter_feedback_timestamps: Whether feedback timestamps should be jittered.

    Returns:
        A jitter spec dict, or ``None`` if ``temporal_jitter_prob`` is not positive.
    """
    if float(cfg.get("temporal_jitter_prob", 0.0)) <= 0.0:
        return None
    return {
        "frame_limit": frame_limit,
        "cfg": cfg,
        "amount_key": amount_key,
        "random_scale": random_scale,
        "jitter_feedback_timestamps": jitter_feedback_timestamps,
    }


def _load_pickle(path: str | Path) -> Any:
    """Load and return a pickled object from ``path``."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _frame_sort_key(frame_path: Path) -> tuple[int, str]:
    """Build a sort key that orders frame files by their trailing frame number.

    Args:
        frame_path: The frame file path.

    Returns:
        A ``(trailing_number, stem)`` tuple; the number is ``0`` when the stem has
        no trailing digits.
    """
    stem = frame_path.stem
    digits = ""
    for char in reversed(stem):
        if not char.isdigit():
            break
        digits = char + digits
    if digits:
        return int(digits), stem
    return 0, stem


def _frame_video_id_from_dir(video_dir: Path) -> str:
    """Derive a video id from a frame directory name, stripping known suffixes.

    Args:
        video_dir: The directory containing a video's frames.

    Returns:
        The video id with ``_360p``/``_2fps`` suffixes removed.
    """
    video_id = video_dir.name
    for suffix in ["_360p", "_2fps"]:
        if video_id.endswith(suffix):
            video_id = video_id[: -len(suffix)]
    return video_id


def _jpgs_in_dir(video_dir: Path) -> list[Path]:
    """Return the ``.jpg`` files in a directory, sorted by frame number."""
    return sorted(video_dir.glob("*.jpg"), key=_frame_sort_key)


def _discover_video_frames_from_root(frames_root: Path) -> dict[str, list[str]]:
    """Scan a frames root for per-video frame directories.

    Looks one level down (and one nested level as a fallback) for directories of
    ``.jpg`` frames, keying each by its derived video id.

    Args:
        frames_root: The root directory to scan.

    Returns:
        Mapping of video id to its ordered frame paths; empty if the root is
        missing or contains no frames.
    """
    if not frames_root.exists():
        return {}

    video_to_frames: dict[str, list[str]] = {}
    for video_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        frame_paths = _jpgs_in_dir(video_dir)
        if not frame_paths:
            for nested_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
                frame_paths = _jpgs_in_dir(nested_dir)
                if frame_paths:
                    break
        if not frame_paths:
            continue
        video_to_frames[_frame_video_id_from_dir(video_dir)] = [
            str(frame_path) for frame_path in frame_paths
        ]
    return video_to_frames


def _candidate_frame_roots(dataset_root: Path, dataset_name: str, fps: float) -> list[Path]:
    """Return the candidate frame-root directories to search for a dataset.

    Args:
        dataset_root: The dataset's root directory.
        dataset_name: The dataset identifier (e.g. ``"captaincook4d"``).
        fps: Frame rate, used to build fps-specific directory names.

    Returns:
        An ordered list of candidate frame-root paths to try.
    """
    fps_int = int(fps) if float(fps).is_integer() else fps
    candidates_by_dataset = {
        "captaincook4d": [
            dataset_root / "captain_cook_4d" / "gopro" / f"resolution_360p_{fps_int}fps_frames",
            dataset_root / "captain_cook_4d" / "gopro" / "resolution_360p_2fps_frames",
            dataset_root / "frames",
        ],
        "eccv": [
            dataset_root / "frames",
            dataset_root / f"frames_fps{fps_int}",
        ],
        "ego4d": [
            dataset_root / "ego4d_data" / "v2" / f"frames_fps{fps_int}_short512",
            dataset_root / "ego4d_data" / "v2" / "frames_fps2_short512",
            dataset_root / "frames",
        ],
        "egoexo4d": [
            dataset_root / "data" / f"frames_360p_{fps_int}fps",
            dataset_root / "data" / "frames_360p_2fps",
            dataset_root / "frames",
        ],
    }
    return candidates_by_dataset.get(dataset_name, [dataset_root / "frames"])


def _load_or_build_video_to_frames(
    cache_path: str | Path,
    dataset_root: str | Path,
    dataset_name: str,
    fps: float,
    extra_frame_roots: list[str | Path] | None = None,
) -> dict[str, list[str]]:
    """Load the video-to-frames map from cache, building and caching it if absent.

    Returns the pickled cache if present; otherwise scans candidate (and any
    extra) frame roots, pickles the first non-empty result to ``cache_path``, and
    returns it.

    Args:
        cache_path: Path to the pickle cache to read or write.
        dataset_root: The dataset's root directory.
        dataset_name: The dataset identifier used to pick candidate frame roots.
        fps: Frame rate used to build fps-specific directory names.
        extra_frame_roots: Additional frame roots to search before the defaults.

    Returns:
        Mapping of video id to its ordered frame paths.

    Raises:
        FileNotFoundError: If the cache is absent and no frame root yields frames.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return _load_pickle(cache_path)

    dataset_root = Path(dataset_root)
    frame_roots = [Path(path) for path in extra_frame_roots or []]
    frame_roots.extend(_candidate_frame_roots(dataset_root, dataset_name, fps))
    seen_frame_roots: set[Path] = set()
    for frame_root in frame_roots:
        if frame_root in seen_frame_roots:
            continue
        seen_frame_roots.add(frame_root)
        video_to_frames = _discover_video_frames_from_root(frame_root)
        if not video_to_frames:
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(video_to_frames, handle)
        total_video_hours = (
            sum(len(frames) for frames in video_to_frames.values()) / max(fps, 1.0) / 3600.0
        )
        print(
            f"Built {cache_path} from {frame_root}: "
            f"{len(video_to_frames)} videos, {total_video_hours:.2f} hours."
        )
        return video_to_frames

    searched = ", ".join(str(frame_root) for frame_root in frame_roots)
    raise FileNotFoundError(
        f"Could not find {cache_path} and could not build it for {dataset_name}. "
        f"Searched frame roots: {searched}"
    )


def _frame_roots_from_cfg(cfg: dict[str, Any]) -> list[str | Path] | None:
    """Collect frame-root paths declared under known config keys.

    Args:
        cfg: The dataset config to read frame-root keys from.

    Returns:
        A flattened list of declared frame roots, or ``None`` if none are set.
    """
    frame_roots: list[str | Path] = []
    for key in ["frames_root", "frame_root", "video_frames_root"]:
        value = cfg.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            frame_roots.extend(value)
        else:
            frame_roots.append(value)
    return frame_roots or None


def _load_json(path: str | Path) -> Any:
    """Load and return a JSON document from ``path`` (UTF-8)."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_hf_enabled(value: Any) -> bool:
    """Interpret a config value as a Hugging Face enable flag.

    Args:
        value: A bool, ``None``, or string such as ``"true"``/``"hf"``.

    Returns:
        ``True`` if the value denotes an enabled Hugging Face source.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "hf", "huggingface"}


def _captaincook4d_split_names(split: str, include_validation_in_train: bool) -> list[str]:
    """Resolve a requested split into the underlying CaptainCook4D split names.

    Args:
        split: The requested split (e.g. ``"train"``, ``"val"``).
        include_validation_in_train: Whether the train split should also pull in
            the validation split.

    Returns:
        The list of underlying split names to load.
    """
    normalized_split = str(split).strip().lower()
    if normalized_split == "train":
        split_names = ["train"]
        if include_validation_in_train:
            split_names.append("validation")
        return split_names
    if normalized_split in {"val", "valid", "validation"}:
        return ["validation"]
    return [normalized_split]


def _hf_dataset_server_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Query the Hugging Face datasets-server and return the parsed JSON response.

    Args:
        endpoint: The datasets-server endpoint (e.g. ``"rows"``, ``"size"``).
        params: Query parameters to URL-encode onto the request.

    Returns:
        The decoded JSON response body.
    """
    import urllib.parse
    import urllib.request

    url = (
        "https://datasets-server.huggingface.co/" + endpoint + "?" + urllib.parse.urlencode(params)
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _path_matches(path_entry: str, target: Path) -> bool:
    """Return whether a path string resolves to the same location as ``target``.

    Args:
        path_entry: A filesystem path string (empty is treated as ``"."``).
        target: The already-resolved target path to compare against.

    Returns:
        ``True`` if the resolved paths are equal; ``False`` on resolution error.
    """
    try:
        path = Path(path_entry or ".").resolve()
    except OSError:
        return False
    return path == target


@contextmanager
def _hf_load_dataset():
    """Yield Hugging Face ``load_dataset`` with the local ``src`` dir off the path.

    Temporarily removes the local source directory from ``sys.path`` (and unloads
    any freshly imported ``datasets`` submodules) so the ``datasets`` package
    resolves to the installed library rather than the local ``datasets`` module,
    restoring the original import state on exit.

    Yields:
        The Hugging Face ``load_dataset`` callable.

    Raises:
        ImportError: If the Hugging Face ``datasets`` library is unavailable.
    """
    local_src_dir = Path(__file__).resolve().parent.parent
    original_sys_path = list(sys.path)
    original_datasets_module = sys.modules.pop("datasets", None)
    existing_dataset_modules = set(sys.modules)
    sys.path[:] = [entry for entry in sys.path if not _path_matches(entry, local_src_dir)]
    try:
        from datasets import load_dataset

        yield load_dataset
    except (AttributeError, ImportError) as exc:
        raise ImportError("Hugging Face datasets is not available") from exc
    finally:
        for module_name in list(sys.modules):
            if module_name == "datasets" or module_name.startswith("datasets."):
                if module_name not in existing_dataset_modules:
                    sys.modules.pop(module_name, None)
        if original_datasets_module is not None:
            sys.modules["datasets"] = original_datasets_module
        sys.path[:] = original_sys_path


def _load_hf_dataset_server_rows(
    repo_id: str,
    config_name: str,
    split: str,
) -> list[dict[str, Any]]:
    """Fetch all rows of a dataset split via the Hugging Face datasets-server.

    Determines the split's row count, then pages through the ``rows`` endpoint in
    fixed-size chunks.

    Args:
        repo_id: The dataset repository id.
        config_name: The dataset configuration name.
        split: The split to fetch.

    Returns:
        The split's rows as a list of dicts.

    Raises:
        ValueError: If the row count for the split cannot be determined.
    """
    size_payload = _hf_dataset_server_json(
        "size",
        {"dataset": repo_id, "config": config_name, "split": split},
    )
    num_rows = None
    for split_info in size_payload.get("size", {}).get("splits", []):
        if split_info.get("split") == split:
            num_rows = int(split_info["num_rows"])
            break
    if num_rows is None:
        raise ValueError(f"Could not determine row count for {repo_id}/{config_name}/{split}")

    rows: list[dict[str, Any]] = []
    chunk_size = 100
    for offset in range(0, num_rows, chunk_size):
        rows_payload = _hf_dataset_server_json(
            "rows",
            {
                "dataset": repo_id,
                "config": config_name,
                "split": split,
                "offset": offset,
                "length": min(chunk_size, num_rows - offset),
            },
        )
        rows.extend(dict(row_payload["row"]) for row_payload in rows_payload.get("rows", []))
    return rows


def _is_hf_connectivity_error(exc: Exception) -> bool:
    """Heuristically decide whether an exception is a Hugging Face connectivity issue.

    Inspects the exception, its cause, and its context for connection errors or
    offline-mode markers.

    Args:
        exc: The exception to classify.

    Returns:
        ``True`` if the error looks like an offline/connectivity failure.
    """
    if isinstance(exc, ConnectionError):
        return True
    class_names = {type(exc).__name__}
    cause = exc.__cause__
    context = exc.__context__
    if cause is not None:
        class_names.add(type(cause).__name__)
    if context is not None:
        class_names.add(type(context).__name__)
    if "OfflineModeIsEnabled" in class_names:
        return True
    message = str(exc)
    return "OfflineModeIsEnabled" in message or "Couldn't reach" in message


def _load_hf_dataset_rows(
    repo_id: str,
    config_name: str,
    splits: list[str],
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load dataset rows via the local library, falling back to the datasets-server.

    Tries the installed ``datasets`` library first; on a connectivity error (or
    when it is unavailable) fetches rows through the Hugging Face datasets-server.

    Args:
        repo_id: The dataset repository id.
        config_name: The dataset configuration name.
        splits: The splits to load and concatenate.
        cache_dir: Optional cache directory for the local library path.

    Returns:
        The concatenated rows across the requested splits.
    """
    try:
        with _hf_load_dataset() as load_dataset:
            rows = []
            for split in splits:
                dataset = load_dataset(repo_id, config_name, split=split, cache_dir=cache_dir)
                rows.extend(dict(row) for row in dataset)
            return rows
    except ImportError:
        pass
    except Exception as exc:
        if not _is_hf_connectivity_error(exc):
            raise

    rows: list[dict[str, Any]] = []
    for split in splits:
        rows.extend(_load_hf_dataset_server_rows(repo_id, config_name, split))
    return rows


def _captaincook4d_mistake_flag_by_instruction(annotation: dict[str, Any]) -> dict[int, bool]:
    """Map each instruction segment to whether it contains a mistake.

    Uses the explicit ``mistake_flag`` list when present; otherwise infers per
    instruction from the response ``output_types`` in its segment.

    Args:
        annotation: A CaptainCook4D annotation with ``output_types`` and optional
            ``mistake_flag``.

    Returns:
        Mapping of instruction output index to its mistake flag.
    """
    mistake_flags = annotation.get("mistake_flag") or []
    output_types = list(annotation.get("output_types", []))
    if mistake_flags:
        mistake_flag_by_instruction: dict[int, bool] = {}
        instruction_segment_idx = 0
        for output_idx, output_type in enumerate(output_types):
            if output_type != "instruction":
                continue
            if instruction_segment_idx < len(mistake_flags):
                mistake_flag_by_instruction[output_idx] = bool(
                    mistake_flags[instruction_segment_idx]
                )
            instruction_segment_idx += 1
        return mistake_flag_by_instruction

    mistake_flag_by_instruction = {}
    for output_idx, output_type in enumerate(output_types):
        if output_type != "instruction":
            continue
        next_instruction = len(output_types)
        for follow_idx in range(output_idx + 1, len(output_types)):
            if output_types[follow_idx] == "instruction":
                next_instruction = follow_idx
                break
        response_types = [
            str(response_type).lower()
            for response_type in output_types[output_idx + 1 : next_instruction]
        ]
        if response_types:
            mistake_flag_by_instruction[output_idx] = any(
                "mistake" in response_type or "failure" in response_type
                for response_type in response_types
            )
    return mistake_flag_by_instruction


def _strip_eccv_prefix(text: str) -> str:
    """Strip a known ECCV response prefix (e.g. ``"Feedback:"``) from text.

    Args:
        text: The text to strip.

    Returns:
        The text with a leading known prefix removed and trimmed; unchanged if no
        known prefix is present.
    """
    text = str(text).strip()
    for prefix in [
        "Instruction-Clarification:",
        "Feedback-Reply:",
        "Instruction:",
        "Feedback:",
        "Success:",
    ]:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _download_hf_dataset_file(
    repo_id: str,
    filename: str,
    cache_dir: str | None = None,
) -> Path:
    """Download a file from a Hugging Face dataset repo and return its local path.

    Args:
        repo_id: The dataset repository id.
        filename: The repo-relative file to download.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        The local path to the downloaded file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the Qualcomm cooking eval dataset"
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=cache_dir,
        )
    )


def _resolve_qualcomm_annotations_path(
    repo_id: str,
    cache_dir: str | None = None,
    local_dir: str | None = None,
) -> Path:
    """Resolve the Qualcomm cooking annotations file, locally or via download.

    Args:
        repo_id: The dataset repository id to download from when no local dir.
        cache_dir: Optional Hugging Face cache directory.
        local_dir: Optional local dataset directory to read from instead.

    Returns:
        The path to ``annotations/annotations.json``.

    Raises:
        FileNotFoundError: If ``local_dir`` is given but the file is missing.
    """
    if local_dir:
        path = Path(local_dir).expanduser() / "annotations" / "annotations.json"
        if not path.exists():
            raise FileNotFoundError(f"local Hugging Face annotations file not found: {path}")
        return path
    return _download_hf_dataset_file(
        repo_id=repo_id, filename="annotations/annotations.json", cache_dir=cache_dir
    )


def _resolve_qualcomm_video_path(
    video_id: str,
    repo_id: str,
    cache_dir: str | None = None,
    local_dir: str | None = None,
) -> Path:
    """Resolve a Qualcomm eval video file, locally or via download.

    Tries ``.MP4`` then ``.mp4`` in both the local and download paths.

    Args:
        video_id: The video identifier (filename stem).
        repo_id: The dataset repository id to download from when no local dir.
        cache_dir: Optional Hugging Face cache directory.
        local_dir: Optional local dataset directory to read from instead.

    Returns:
        The path to the resolved video file.

    Raises:
        FileNotFoundError: If ``local_dir`` is given but no matching video exists.
        ImportError: If ``huggingface_hub`` is required but not installed.
    """
    if local_dir:
        root = Path(local_dir).expanduser()
        for suffix in ["MP4", "mp4"]:
            path = root / "videos" / f"{video_id}.{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(
            f"local Hugging Face video file not found for {video_id} in {root / 'videos'}"
        )

    try:
        from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
    except ImportError:
        # huggingface_hub is unavailable; defer to _download_hf_dataset_file,
        # which raises a descriptive ImportError.
        return _download_hf_dataset_file(
            repo_id=repo_id, filename=f"videos/{video_id}.MP4", cache_dir=cache_dir
        )

    try:
        return _download_hf_dataset_file(
            repo_id=repo_id, filename=f"videos/{video_id}.MP4", cache_dir=cache_dir
        )
    except (EntryNotFoundError, HfHubHTTPError):
        return _download_hf_dataset_file(
            repo_id=repo_id, filename=f"videos/{video_id}.mp4", cache_dir=cache_dir
        )


def _extract_video_frames(video_path: Path, output_dir: Path, target_fps: float) -> list[str]:
    """Extract frames from a video at a target fps, caching them as JPEGs.

    Returns previously extracted frames if present; otherwise samples frames from
    the video at ``target_fps`` using OpenCV and writes them to ``output_dir``.

    Args:
        video_path: The source video file.
        output_dir: Directory to read/write extracted ``frame_*.jpg`` files.
        target_fps: Desired sampling frame rate.

    Returns:
        The ordered paths of the extracted frame images.

    Raises:
        ImportError: If OpenCV (``cv2``) is not installed.
        FileNotFoundError: If the video cannot be opened.
        ValueError: If no frames could be extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("frame_*.jpg"))
    if existing_frames:
        return [str(frame_path) for frame_path in existing_frames]

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python/cv2 is required to extract frames from MP4 eval videos"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open eval video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    sample_interval = 1.0 / max(float(target_fps), 1.0e-6)
    next_sample_time = 0.0
    frame_idx = 0
    saved_frame_paths: list[str] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_idx / source_fps
        if timestamp + 1.0e-9 >= next_sample_time:
            frame_path = output_dir / f"frame_{len(saved_frame_paths):06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frame_paths.append(str(frame_path))
            next_sample_time += sample_interval
        frame_idx += 1

    capture.release()
    if not saved_frame_paths:
        raise ValueError(f"No frames extracted from eval video: {video_path}")
    return saved_frame_paths


__all__ = [name for name in globals() if not name.startswith("__")]
