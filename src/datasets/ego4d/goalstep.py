# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Ego4D GoalStep dataset: builds streaming success samples from Ego4D GoalStep action instructions."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets.common import (
    SUCCESS_RESPONSE,
    CookingSample,
    ListDataset,
    _append_next_instruction,
    _frame_roots_from_cfg,
    _frames_and_timestamps_between,
    _load_or_build_video_to_frames,
    _load_pickle,
    _normalize_text,
    _temporal_jitter_config,
    _video_frame_timestamps,
)


class Ego4DGoalStepDataset(ListDataset):
    """Streaming cooking samples derived from Ego4D GoalStep actions.

    Each goal-step action becomes an instruction with a success response anchored
    to the action interval and its extracted frames.
    """

    def __init__(self, **cfg: Any) -> None:
        """Build the dataset from a config dict.

        Args:
            **cfg: Dataset config, including ``ego4d_root``, ``split``, fps, the
                frames cache path, and frame limits.
        """
        root = Path(cfg["ego4d_root"])
        split = cfg.get("split", "train")
        actions = _load_pickle(root / f"video_to_actions_instruct_{split}.pkl")
        fps = float(cfg.get("model_input_fps", 2))
        frames_cache_path = Path(
            cfg.get("videos_to_frames_cache_file_path") or root / f"videos_to_frames_{split}.pkl"
        )
        video_to_frames = _load_or_build_video_to_frames(
            frames_cache_path,
            root,
            "ego4d",
            fps,
            extra_frame_roots=_frame_roots_from_cfg(cfg),
        )
        video_to_frame_timestamps = _video_frame_timestamps(video_to_frames, fps)
        frame_limit = int(cfg.get("max_num_frames_to_sample", 200))

        system_prompts = [
            "You are a skilled culinary assistant guiding someone through cooking. Watch the video and step in the moment a mistake is evident.",
            "Act as an expert cooking helper assisting a person in the kitchen. Observe the video and intervene as soon as an error becomes apparent.",
            "You are a professional cooking assistant supporting someone as they cook. Monitor the video and provide guidance immediately when a mistake is clear.",
            "Serve as an experienced kitchen assistant helping a person prepare food. Watch the video carefully and intervene once you can clearly identify a mistake.",
            "You are an expert culinary guide assisting someone with cooking. Review the video and step in as soon as it is obvious that something has gone wrong.",
            "Act as a knowledgeable cooking coach helping a person cook. Watch the video and offer correction the moment a mistake becomes recognizable.",
            "You are a seasoned cooking assistant supporting a person in preparing a dish. Observe the video and intervene immediately when an error is clearly visible.",
            "Take the role of an expert kitchen coach helping someone cook. Watch the video and provide assistance as soon as a mistake becomes clear.",
            "You are a capable culinary assistant guiding a person through the cooking process. Monitor the video and step in right away when an issue is evident.",
            "Act as a professional cooking guide assisting someone in the kitchen. Watch the video and intervene at the earliest point when a mistake is clear.",
        ]

        samples: list[CookingSample] = []
        for video_id, action_map in actions.items():
            ordered_actions = list(action_map.items())
            for action_idx, (_, action) in enumerate(ordered_actions):
                instruction = _normalize_text(action.get("instruction") or action.get("narration"))
                if not instruction:
                    continue
                start_time = float(action["start_time"])
                end_time = float(action["end_time"])
                frame_paths, frame_timestamps = _frames_and_timestamps_between(
                    video_to_frames, video_id, fps, start_time, end_time, frame_limit
                )
                if frame_paths:
                    samples.append(
                        CookingSample(
                            system_prompt=random.choice(system_prompts),  # system_prompt,
                            instruction=instruction,
                            responses=_append_next_instruction([SUCCESS_RESPONSE]),
                            response_timestamps=[end_time, end_time],
                            frame_paths=video_to_frames.get(video_id) or frame_paths,
                            frame_timestamps=video_to_frame_timestamps.get(video_id)
                            or frame_timestamps,
                            source="ego4d_goalstep",
                            metadata={
                                "video_id": video_id,
                                "action_idx": action_idx,
                                "start_time": start_time,
                                "end_time": end_time,
                                "gt_start_time": start_time,
                                "gt_end_time": end_time,
                                "frame_start_time": start_time,
                                "frame_end_time": end_time,
                                "variant": "success",
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
                amount_key="temporal_jitter_amount",
                random_scale=2.0,
            ),
        )
