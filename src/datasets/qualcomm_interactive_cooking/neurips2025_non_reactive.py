# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""CaptainCook4D instruction dataset: builds streaming success/feedback samples from CaptainCook4D annotations."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets.common import (
    CookingSample,
    ListDataset,
    _append_next_instruction_timestamped,
    _captaincook4d_mistake_flag_by_instruction,
    _captaincook4d_split_names,
    _format_response,
    _frame_roots_from_cfg,
    _frames_and_timestamps_between,
    _is_hf_enabled,
    _load_hf_dataset_rows,
    _load_json,
    _load_or_build_video_to_frames,
    _normalize_text,
    _temporal_jitter_config,
    _video_frame_timestamps,
)


class CaptainCook4DInstructDataset(ListDataset):
    """Streaming cooking samples derived from CaptainCook4D recordings.

    Pairs timed instruction/feedback annotations with extracted video frames and
    selects a system prompt depending on whether a mistake is present.
    """

    system_prompt = (
        "You are an expert cooking assistant helping a person cook."
        " Watch the video and provide the next success or failure response."
    )

    def __init__(self, **cfg: Any) -> None:
        """Build the dataset from a config dict.

        Args:
            **cfg: Dataset config, including ``captaincook4d_root``, ``split``,
                annotation source/repo settings, fps, and frame limits.
        """
        root = Path(cfg["captaincook4d_root"])
        split = cfg.get("split", "train")
        include_recipe = bool(cfg.get("include_recipe_in_prompt", False))
        fps = float(cfg.get("dataset_extracted_fps", 2))
        frame_limit = int(cfg.get("max_number_of_output_frames", 200))
        temporal_window = frame_limit / max(fps, 1.0)
        feedback_vs_success_pref_weight = float(cfg.get("feedback_vs_success_pref_weight", 1.0))
        annotation_set = str(cfg.get("annotation_set", cfg.get("annotation_config", "main")))
        annotation_source = (
            str(
                cfg.get(
                    "annotation_source",
                    "hf" if cfg.get("annotation_repo_id") or cfg.get("repo_id") else "local",
                )
            )
            .strip()
            .lower()
        )
        use_hf_annotations = annotation_source in {"hf", "huggingface"} or _is_hf_enabled(
            cfg.get("use_hf_annotations")
        )
        video_to_frames = _load_or_build_video_to_frames(
            root / "video_to_frames.pkl",
            root,
            "captaincook4d",
            fps,
            extra_frame_roots=_frame_roots_from_cfg(cfg),
        )
        video_to_frame_timestamps = _video_frame_timestamps(video_to_frames, fps)

        if use_hf_annotations:
            repo_id = str(
                cfg.get(
                    "annotation_repo_id",
                    cfg.get("repo_id", "qualcomm/qualcomm-interactive-cooking-dataset"),
                )
            )
            hf_splits = _captaincook4d_split_names(
                str(split),
                include_validation_in_train=bool(cfg.get("include_validation_in_train", True)),
            )
            annotations = _load_hf_dataset_rows(
                repo_id=repo_id,
                config_name=annotation_set,
                splits=hf_splits,
                cache_dir=cfg.get("hf_cache_dir", cfg.get("cache_dir")),
            )
        elif split == "train":
            annotations = _load_json(root / f"{annotation_set}_train.json") + _load_json(
                root / f"{annotation_set}_val.json"
            )
        elif split in {"val", "valid", "validation"}:
            annotations = _load_json(root / f"{annotation_set}_val.json")
        else:
            annotations = _load_json(root / f"{annotation_set}_{split}.json")

        max_videos = cfg.get("max_videos")
        if max_videos is not None:
            annotations = annotations[: int(max_videos)]

        samples: list[CookingSample] = []
        sample_weights: list[float] = []
        for annotation in annotations:
            video_id = annotation["video_id"]
            outputs = annotation["output_texts"]
            output_types = annotation["output_types"]
            timestamps = annotation["output_timestamps"]
            mistake_flag_by_instruction = _captaincook4d_mistake_flag_by_instruction(annotation)
            recipe_text = ""
            if include_recipe and annotation.get("remaining_plan"):
                recipe_text = " Recipe context: " + " ".join(annotation["remaining_plan"][:5])

            for idx, output_type in enumerate(output_types):
                if output_type != "instruction":
                    continue
                instruction = _normalize_text(outputs[idx])
                next_instruction = len(output_types)
                for follow_idx in range(idx + 1, len(output_types)):
                    if output_types[follow_idx] == "instruction":
                        next_instruction = follow_idx
                        break
                has_next_instruction = next_instruction < len(output_types)
                response_texts = []
                response_timestamps = []
                for response_idx in range(idx + 1, next_instruction):
                    formatted_response = _format_response(
                        output_types[response_idx], outputs[response_idx]
                    )
                    if formatted_response:
                        response_texts.append(formatted_response)
                        response_timestamps.append(float(timestamps[response_idx]))

                responses, response_timestamps = _append_next_instruction_timestamped(
                    response_texts,
                    response_timestamps,
                    float(timestamps[next_instruction]) if has_next_instruction else None,
                    include_next_instruction=has_next_instruction,
                )
                if not responses:
                    continue
                start_time = max(0.0, float(timestamps[idx]))
                if (next_instruction > idx + 1) and (next_instruction < len(timestamps)):
                    end_time = float(timestamps[next_instruction])
                elif (next_instruction > idx + 1) and (next_instruction >= len(timestamps)):
                    end_time = float(timestamps[-1])
                else:
                    end_time = start_time + 5.0
                start_time = max(start_time, start_time - temporal_window)
                frame_paths, frame_timestamps = _frames_and_timestamps_between(
                    video_to_frames, video_id, fps, start_time, end_time, frame_limit
                )
                if not frame_paths:
                    continue
                has_mistake = mistake_flag_by_instruction.get(idx)
                sample_weight = 1.0
                if has_mistake is not None:
                    if feedback_vs_success_pref_weight == 0.0:
                        sample_weight = 0.0 if has_mistake else 1.0
                    elif has_mistake:
                        sample_weight = feedback_vs_success_pref_weight
                samples.append(
                    CookingSample(
                        system_prompt=self.system_prompt,
                        instruction=instruction,
                        responses=responses,
                        response_timestamps=response_timestamps,
                        frame_paths=video_to_frames.get(video_id) or frame_paths,
                        frame_timestamps=video_to_frame_timestamps.get(video_id)
                        or frame_timestamps,
                        source="captaincook4d_instruct",
                        metadata={
                            "video_id": video_id,
                            "instruction_index": idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "gt_start_time": start_time,
                            "gt_end_time": end_time,
                            "frame_start_time": start_time,
                            "frame_end_time": end_time,
                            "response_types": output_types[idx + 1 : next_instruction],
                            "mistake_flag": has_mistake,
                            "annotation_source": "hf" if use_hf_annotations else "local",
                            "annotation_set": annotation_set,
                            "activity_name": annotation.get("activity_name"),
                            "is_interactive": True,
                        },
                        video_frame_paths=video_to_frames.get(video_id),
                        video_frame_timestamps=video_to_frame_timestamps.get(video_id),
                    )
                )
                sample_weights.append(sample_weight)
        if samples and sample_weights and all(weight == 0.0 for weight in sample_weights):
            sample_weights[0] = 1.0
        super().__init__(
            samples,
            temporal_jitter=_temporal_jitter_config(
                frame_limit,
                cfg,
                amount_key="temporal_jitter_amount_instructions",
                random_scale=3.0,
                jitter_feedback_timestamps=True,
            ),
            sample_weights=(
                sample_weights if any(weight != 1.0 for weight in sample_weights) else None
            ),
        )
