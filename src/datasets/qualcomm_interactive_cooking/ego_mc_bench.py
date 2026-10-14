# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Ego-MC-Bench dataset: evaluation samples for streaming mistake-correction guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.common import (
    CookingSample,
    ListDataset,
    _append_next_instruction_timestamped,
    _discover_video_frames_from_root,
    _extract_video_frames,
    _format_response,
    _frames_and_timestamps_between,
    _load_json,
    _normalize_text,
    _resolve_qualcomm_annotations_path,
    _resolve_qualcomm_video_path,
    _temporal_jitter_config,
)


class EgoMCBench(ListDataset):
    """Ego-MC-Bench evaluation samples.

    Loads timestamped instruction/feedback annotations from Hugging Face and pairs
    them with frames read from a configured ``frames_root`` or extracted from the
    dataset videos.
    """

    def __init__(self, **cfg: Any) -> None:
        """Build the dataset from a config dict.

        Args:
            **cfg: Dataset config, including ``repo_id``, optional ``local_dir`` or
                ``frames_root``, fps, and frame limits.
        """
        repo_id = str(
            cfg.get(
                "repo_id", "qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections"
            )
        )
        local_dir = cfg.get("local_dir")
        cache_dir = cfg.get("cache_dir")
        frame_cache_dir = Path(
            cfg.get(
                "frame_cache_dir",
                Path.home() / ".cache" / "streaming_vlms" / "qualcomm_interactive_cooking_frames",
            )
        ).expanduser()
        fps = float(cfg.get("dataset_extracted_fps", cfg.get("model_input_fps", 2)))
        frame_limit = int(cfg.get("max_number_of_output_frames", 200))
        configured_frame_roots = self._configured_frame_roots(cfg)
        local_video_to_frames = self._load_local_video_frames(
            configured_frame_roots, frame_cache_dir
        )
        require_local_frames = bool(configured_frame_roots)
        annotations_path = _resolve_qualcomm_annotations_path(
            repo_id=repo_id, cache_dir=cache_dir, local_dir=local_dir
        )
        annotations = _load_json(annotations_path)
        max_videos = cfg.get("max_videos")
        if max_videos is not None:
            annotations = annotations[: int(max_videos)]

        system_prompt = (
            "You are an expert cooking assistant helping a person cook. "
            "Watch the video and provide the appropriate success or failure response. "
            "Try to intervene by providing failure messages as soon as possible."
        )
        assistant_types = {"Feedback", "Feedback-Reply", "Success", "Instruction-Clarification"}

        samples: list[CookingSample] = []
        video_to_frames: dict[str, list[str]] = {}
        video_to_frame_timestamps: dict[str, list[float]] = {}
        for annotation in annotations:
            video_id = str(annotation["video_id"])
            if video_id not in video_to_frames:
                video_to_frames[video_id] = self._resolve_video_frames(
                    video_id=video_id,
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                    frame_cache_dir=frame_cache_dir,
                    local_video_to_frames=local_video_to_frames,
                    fps=fps,
                    require_local_frames=require_local_frames,
                )
                video_to_frame_timestamps[video_id] = [
                    frame_idx / fps for frame_idx in range(len(video_to_frames[video_id]))
                ]

            texts = list(annotation.get("texts", []))
            text_types = list(annotation.get("text_types", []))
            timestamps = [float(timestamp) for timestamp in annotation.get("text_timestamps", [])]
            for idx, text_type in enumerate(text_types):
                if text_type != "Instruction":
                    continue

                next_instruction_idx = len(text_types)
                for follow_idx in range(idx + 1, len(text_types)):
                    if text_types[follow_idx] == "Instruction":
                        next_instruction_idx = follow_idx
                        break

                has_next_instruction = next_instruction_idx < len(text_types)
                response_texts = []
                response_timestamps = []
                for response_text, response_type, response_timestamp in zip(
                    texts[idx + 1 : next_instruction_idx],
                    text_types[idx + 1 : next_instruction_idx],
                    timestamps[idx + 1 : next_instruction_idx],
                ):
                    if response_type not in assistant_types:
                        continue
                    formatted_response = _format_response(response_type, response_text)
                    if formatted_response:
                        response_texts.append(formatted_response)
                        response_timestamps.append(response_timestamp)
                responses, response_timestamps = _append_next_instruction_timestamped(
                    response_texts,
                    response_timestamps,
                    timestamps[next_instruction_idx] if has_next_instruction else None,
                    include_next_instruction=has_next_instruction,
                )
                if not responses:
                    continue

                start_time = timestamps[idx]
                end_time = (
                    timestamps[next_instruction_idx]
                    if next_instruction_idx < len(timestamps)
                    else timestamps[-1]
                )
                frame_paths, frame_timestamps = _frames_and_timestamps_between(
                    video_to_frames, video_id, fps, start_time, end_time, frame_limit
                )
                if not frame_paths:
                    continue

                samples.append(
                    CookingSample(
                        system_prompt=system_prompt,
                        instruction=_normalize_text(texts[idx]),
                        responses=responses,
                        response_timestamps=response_timestamps,
                        frame_paths=video_to_frames.get(video_id) or frame_paths,
                        frame_timestamps=video_to_frame_timestamps.get(video_id)
                        or frame_timestamps,
                        source="qualcomm_interactive_cooking_hf",
                        metadata={
                            "video_id": video_id,
                            "instruction_index": idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "gt_start_time": start_time,
                            "gt_end_time": end_time,
                            "frame_start_time": start_time,
                            "frame_end_time": end_time,
                            "text_types": text_types[idx:next_instruction_idx],
                            "is_interactive": True,
                        },
                        video_frame_paths=video_to_frames.get(video_id),
                        video_frame_timestamps=video_to_frame_timestamps.get(video_id),
                    )
                )

        super().__init__(
            samples,
            temporal_jitter=_temporal_jitter_config(
                frame_limit,
                cfg,
                amount_key="temporal_jitter_amount_instructions",
                random_scale=3.0,
                jitter_feedback_timestamps=True,
            ),
        )

    @staticmethod
    def _configured_frame_roots(cfg: dict[str, Any]) -> list[Path]:
        """Collect the local frame-root directories declared in the config.

        Reads the ``frames_root``/``local_frames_root`` single-value keys and the
        ``frame_roots`` list (accepting a bare string/path), expanding each.

        Args:
            cfg: The dataset config to read frame-root keys from.

        Returns:
            The expanded frame-root paths, in declaration order.
        """
        roots: list[Path] = []
        for key in ["frames_root", "local_frames_root"]:
            value = cfg.get(key)
            if value:
                roots.append(Path(value).expanduser())
        frame_roots = cfg.get("frame_roots", []) or []
        if isinstance(frame_roots, (str, Path)):
            frame_roots = [frame_roots]
        for value in frame_roots:
            roots.append(Path(value).expanduser())
        return roots

    @staticmethod
    def _load_local_video_frames(
        configured_frame_roots: list[Path],
        frame_cache_dir: Path,
    ) -> dict[str, list[str]]:
        """Discover locally available video frames from the configured roots.

        Scans each configured frame root (or the frame cache dir when none are
        configured), merging their discovered video-to-frames maps.

        Args:
            configured_frame_roots: Explicitly configured frame-root directories.
            frame_cache_dir: Fallback frame cache directory when none are configured.

        Returns:
            Mapping of video id to its ordered local frame paths.
        """
        video_to_frames: dict[str, list[str]] = {}
        frame_roots = configured_frame_roots or [frame_cache_dir]
        for frames_root in frame_roots:
            video_to_frames.update(_discover_video_frames_from_root(frames_root))
        return video_to_frames

    @staticmethod
    def _resolve_video_frames(
        video_id: str,
        repo_id: str,
        cache_dir: str | None,
        local_dir: str | None,
        frame_cache_dir: Path,
        local_video_to_frames: dict[str, list[str]],
        fps: float,
        require_local_frames: bool,
    ) -> list[str]:
        """Resolve a video's frames from local caches or by extracting from video.

        Returns cached local frames when present. Otherwise, unless local frames
        are required, resolves the source video (locally or via download) and
        extracts frames at ``fps`` into the frame cache.

        Args:
            video_id: The video identifier.
            repo_id: The dataset repository id for downloads.
            cache_dir: Optional Hugging Face cache directory.
            local_dir: Optional local dataset directory.
            frame_cache_dir: Directory to write/read extracted frames.
            local_video_to_frames: Pre-discovered local video-to-frames map.
            fps: Frame rate used when extracting frames.
            require_local_frames: If ``True``, fail instead of extracting when the
                video is not in the local frame map.

        Returns:
            The video's ordered frame paths.

        Raises:
            FileNotFoundError: If local frames are required but not found.
        """
        if video_id in local_video_to_frames:
            return local_video_to_frames[video_id]
        if require_local_frames:
            raise FileNotFoundError(f"local frames not found for Qualcomm video_id={video_id}")

        video_path = _resolve_qualcomm_video_path(
            video_id=video_id,
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        return _extract_video_frames(video_path, frame_cache_dir / video_id, fps)
