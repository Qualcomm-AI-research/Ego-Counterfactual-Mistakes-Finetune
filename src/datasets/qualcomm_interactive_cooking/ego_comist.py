# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Ego-CoMist dataset: counterfactual mistake/intervention samples reusing CaptainCook4D, Ego4D, and Ego-Exo4D frames."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from datasets.common import (
    CookingSample,
    ListDataset,
    _append_next_instruction_timestamped,
    _download_hf_dataset_file,
    _format_response,
    _frame_roots_from_cfg,
    _frames_and_timestamps_between,
    _jpgs_in_dir,
    _load_json,
    _load_or_build_video_to_frames,
    _normalize_text,
    _temporal_jitter_config,
)


class EgoCoMist(ListDataset):
    """Synthetic counterfactual cooking-mistake samples (Ego-CoMist).

    Loads mistake/intervention annotations from Hugging Face and resolves frames
    from the source datasets (CaptainCook4D, Ego4D, Ego-Exo4D) by ``dataset`` and
    ``video_id``.
    """

    def __init__(self, **cfg: Any) -> None:
        """Build the dataset from a config dict.

        Args:
            **cfg: Dataset config, including ``repo_id``, the source dataset roots
                (``captaincook4d_root``, ``ego4d_root``, ``egoexo4d_root``), fps,
                frame limits, and optional error-type sampling weights.
        """
        repo_id = str(
            cfg.get(
                "repo_id",
                "qualcomm/qualcomm-interactive-cooking-dataset-counterfactual-mistakes",
            )
        )
        split = str(cfg.get("split", "train"))
        fps = float(cfg.get("dataset_extracted_fps", cfg.get("model_input_fps", 2)))
        frame_limit = int(cfg.get("max_number_of_output_frames", 200))
        annotations = self._load_annotations(repo_id, cfg)
        max_videos = cfg.get("max_videos")
        if max_videos is not None:
            annotations = annotations[: int(max_videos)]

        source_roots = {
            "CaptainCook4D": Path(cfg["captaincook4d_root"]).expanduser(),
            "Ego4D": Path(cfg["ego4d_root"]).expanduser(),
            "Ego-Exo4D": Path(cfg["egoexo4d_root"]).expanduser(),
        }
        source_frame_caches = self._load_source_frame_caches(source_roots, fps, cfg)
        discovered_frame_cache: dict[tuple[str, str], list[str]] = {}

        system_prompt = (
            "You are an expert cooking assistant helping a person cook."
            " Watch the video and intervene as soon as a mistake becomes clear."
        )
        samples: list[CookingSample] = []
        for row_idx, annotation in enumerate(annotations):
            source_dataset = str(annotation.get("dataset", ""))
            video_id = str(annotation.get("video_id", ""))
            if not source_dataset or not video_id:
                continue
            frame_paths = self._get_video_frames(
                source_dataset,
                video_id,
                source_roots,
                source_frame_caches,
                discovered_frame_cache,
                fps,
            )
            if not frame_paths:
                continue

            start_time = max(0.0, float(annotation.get("start_time", 0.0)))
            end_time = float(annotation.get("end_time", start_time))
            if end_time < start_time:
                end_time = start_time
            frame_timestamps = [frame_idx / fps for frame_idx in range(len(frame_paths))]
            clip_frame_paths, clip_frame_timestamps = _frames_and_timestamps_between(
                {video_id: frame_paths},
                video_id,
                fps,
                start_time,
                end_time,
                frame_limit,
            )
            if not clip_frame_paths:
                continue

            texts = list(annotation.get("texts", []))
            text_types = list(annotation.get("text_types", []))
            text_timestamps = [
                float(timestamp) # + random.uniform(-5.,5.)
                for timestamp in annotation.get("text_timestamps", [])
            ]
            instruction = ""
            response_texts: list[str] = []
            response_timestamps: list[float] = []
            for text, text_type, timestamp in zip(texts, text_types, text_timestamps):
                if str(text_type).lower() == "instruction" and not instruction:
                    instruction = _normalize_text(text)
                    continue
                formatted_response = _format_response(str(text_type), text)
                if formatted_response:
                    response_texts.append(formatted_response)
                    response_timestamps.append(float(timestamp))
            if not instruction:
                instruction = _normalize_text(texts[0] if texts else "")
            responses, response_timestamps = _append_next_instruction_timestamped(
                response_texts,
                response_timestamps,
                end_time,
                include_next_instruction=True,
            )
            if not instruction or not responses:
                continue
            mistake_type = self._normalize_error_type(annotation.get("mistake_type"))
            samples.append(
                CookingSample(
                    system_prompt=system_prompt,
                    instruction=instruction,
                    responses=responses,
                    response_timestamps=response_timestamps,
                    frame_paths=frame_paths,
                    frame_timestamps=frame_timestamps,
                    source="qualcomm_counterfactual_mistakes_hf",
                    metadata={
                        "row_idx": row_idx,
                        "source_dataset": source_dataset,
                        "video_id": video_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "gt_start_time": start_time,
                        "gt_end_time": end_time,
                        "frame_start_time": start_time,
                        "frame_end_time": end_time,
                        "mistake_type": mistake_type,
                        "error_type": mistake_type,
                        "text_types": text_types,
                        "split": split,
                        "is_interactive": True,
                    },
                    video_frame_paths=frame_paths,
                    video_frame_timestamps=frame_timestamps,
                )
            )

        sample_weights = self._build_error_type_sample_weights(samples, cfg)
        super().__init__(
            samples,
            temporal_jitter=_temporal_jitter_config(
                frame_limit,
                cfg,
                amount_key="temporal_jitter_amount_instructions",
                random_scale=3.0,
                jitter_feedback_timestamps=True,
            ),
            sample_weights=sample_weights,
        )

    @staticmethod
    def _normalize_error_type(error_type: Any) -> str:
        """Normalize an error/mistake type to a trimmed lowercase string."""
        return str(error_type or "").strip().lower()

    @classmethod
    def _build_error_type_sample_weights(
        cls,
        samples: list[CookingSample],
        cfg: dict[str, Any],
    ) -> list[float] | None:
        """Compute per-sample weights that rebalance samples by error type.

        Reads configured target weights per error type and divides each by the
        type's frequency so sampling matches the configured distribution.

        Args:
            samples: The built samples to weight.
            cfg: Config providing ``error_type_sampling_weights`` (or the
                ``mistake_type_sampling_weights`` alias).

        Returns:
            Per-sample weights aligned with ``samples``, or ``None`` if no weights
            are configured.

        Raises:
            ValueError: If the configured weights match none of the samples.
        """
        configured_weights = cfg.get("error_type_sampling_weights") or cfg.get(
            "mistake_type_sampling_weights"
        )
        if not configured_weights:
            return None

        target_weights = {
            cls._normalize_error_type(error_type): max(0.0, float(weight))
            for error_type, weight in dict(configured_weights).items()
        }
        counts = Counter(
            cls._normalize_error_type(sample.metadata.get("error_type")) for sample in samples
        )
        sample_weights = [
            target_weights.get(cls._normalize_error_type(sample.metadata.get("error_type")), 0.0)
            / max(1, counts[cls._normalize_error_type(sample.metadata.get("error_type"))])
            for sample in samples
        ]
        if sum(sample_weights) <= 0.0:
            raise ValueError(
                "error_type_sampling_weights did not match any Qualcomm counterfactual samples"
            )
        return sample_weights

    @staticmethod
    def _load_annotations(repo_id: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Load the counterfactual-mistake annotations from a local path or the hub.

        Prefers ``local_annotations_path``, then a ``local_dir``/``annotations.json``,
        then downloads ``annotations.json`` from the Hugging Face repo.

        Args:
            repo_id: The dataset repository id used for the hub download.
            cfg: Config providing optional ``local_annotations_path``, ``local_dir``,
                and ``cache_dir``.

        Returns:
            The parsed annotation records.

        Raises:
            FileNotFoundError: If ``local_dir`` is set but its annotations are absent.
        """
        local_annotations_path = cfg.get("local_annotations_path")
        if local_annotations_path:
            return _load_json(Path(local_annotations_path).expanduser())
        local_dir = cfg.get("local_dir")
        if local_dir:
            path = Path(local_dir).expanduser() / "annotations.json"
            if not path.exists():
                raise FileNotFoundError(f"local annotations file not found: {path}")
            return _load_json(path)
        annotations_path = _download_hf_dataset_file(
            repo_id=repo_id,
            filename="annotations.json",
            cache_dir=cfg.get("cache_dir"),
        )
        return _load_json(annotations_path)

    @staticmethod
    def _load_source_frame_caches(
        source_roots: dict[str, Path],
        fps: float,
        cfg: dict[str, Any],
    ) -> dict[str, dict[str, list[str]]]:
        """Build per-source video-to-frames maps for the frame source datasets.

        Args:
            source_roots: Mapping of source dataset name to its root directory.
            fps: Frame rate used when discovering/building frame caches.
            cfg: Config providing any extra frame roots to search.

        Returns:
            Mapping of source dataset name to its video-to-frames map.
        """
        extra_frame_roots = _frame_roots_from_cfg(cfg)
        return {
            "CaptainCook4D": _load_or_build_video_to_frames(
                source_roots["CaptainCook4D"] / "video_to_frames.pkl",
                source_roots["CaptainCook4D"],
                "captaincook4d",
                fps,
                extra_frame_roots=extra_frame_roots,
            ),
            "Ego-Exo4D": _load_or_build_video_to_frames(
                source_roots["Ego-Exo4D"] / "video_to_frames.pkl",
                source_roots["Ego-Exo4D"],
                "egoexo4d",
                fps,
                extra_frame_roots=extra_frame_roots,
            ),
        }

    @classmethod
    def _get_video_frames(
        cls,
        source_dataset: str,
        video_id: str,
        source_roots: dict[str, Path],
        source_frame_caches: dict[str, dict[str, list[str]]],
        discovered_frame_cache: dict[tuple[str, str], list[str]],
        fps: float,
    ) -> list[str]:
        """Return a video's frame paths, using caches and discovering on miss.

        Checks the pre-built source cache first, then a per-run discovery cache,
        populating the latter by scanning candidate frame directories.

        Args:
            source_dataset: The source dataset name (e.g. ``"CaptainCook4D"``).
            video_id: The video identifier.
            source_roots: Mapping of source dataset name to its root directory.
            source_frame_caches: Pre-built per-source video-to-frames maps.
            discovered_frame_cache: Per-run cache of discovered frames, keyed by
                ``(source_dataset, video_id)`` (updated in place).
            fps: Frame rate used when discovering frames.

        Returns:
            The video's ordered frame paths, or an empty list if none are found.
        """
        if video_id in source_frame_caches.get(source_dataset, {}):
            return source_frame_caches[source_dataset][video_id]
        cache_key = (source_dataset, video_id)
        if cache_key not in discovered_frame_cache:
            discovered_frame_cache[cache_key] = cls._discover_video_frames(
                source_dataset, video_id, source_roots, fps
            )
        return discovered_frame_cache[cache_key]

    @staticmethod
    def _discover_video_frames(
        source_dataset: str,
        video_id: str,
        source_roots: dict[str, Path],
        fps: float,
    ) -> list[str]:
        """Discover a video's frames by scanning source-specific candidate directories.

        Tries fps-specific and fallback frame directories for the source dataset,
        descending one nested level when a candidate directory holds no JPEGs.

        Args:
            source_dataset: The source dataset name (e.g. ``"Ego4D"``).
            video_id: The video identifier.
            source_roots: Mapping of source dataset name to its root directory.
            fps: Frame rate used to build fps-specific directory names.

        Returns:
            The video's ordered frame paths, or an empty list if none are found.
        """
        fps_int = int(fps) if float(fps).is_integer() else fps
        root = source_roots[source_dataset]
        candidates_by_source = {
            "CaptainCook4D": [
                root
                / "captain_cook_4d"
                / "gopro"
                / f"resolution_360p_{fps_int}fps_frames"
                / f"{video_id}_360p",
                root
                / "captain_cook_4d"
                / "gopro"
                / "resolution_360p_2fps_frames"
                / f"{video_id}_360p",
                root / "frames" / f"{video_id}_360p",
                root / "frames" / video_id,
            ],
            "Ego4D": [
                root / "ego4d_data" / "v2" / f"frames_fps{fps_int}_short512" / video_id,
                root / "ego4d_data" / "v2" / "frames_fps2_short512" / video_id,
                root / "frames" / video_id,
            ],
            "Ego-Exo4D": [
                root / "data" / f"frames_360p_{fps_int}fps" / video_id,
                root / "data" / "frames_360p_2fps" / video_id,
                root / "frames" / video_id,
            ],
        }
        for video_dir in candidates_by_source.get(source_dataset, []):
            frame_paths = _jpgs_in_dir(video_dir)
            if not frame_paths and video_dir.exists():
                for nested_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
                    frame_paths = _jpgs_in_dir(nested_dir)
                    if frame_paths:
                        break
            if frame_paths:
                return [str(frame_path) for frame_path in frame_paths]
        return []
