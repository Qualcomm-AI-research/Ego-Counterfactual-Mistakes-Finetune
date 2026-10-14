# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Collator that turns :class:`CookingSample` items into batched, interleaved
video-text tensors for the Qwen streaming VLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from data import (
    CookingSample,
    _jitter_feedback_timestamps,
    _jitter_window_start,
    _select_frames_between,
)


@dataclass
class CollatorConfig:
    """Resolved collation settings.

    Attributes:
        image_size: Square size (pixels) each frame is resized to.
        max_frames: Maximum frames per sample for non-streaming prompts.
        max_text_length: Maximum number of text tokens (excluding image tokens).
        max_streaming_frames: Memory-safety cap on retained recent frames; ``None``
            or non-positive disables the cap.
    """

    image_size: int
    max_frames: int
    max_text_length: int
    max_streaming_frames: int | None


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


def build_collator_config(data_cfg: Any) -> CollatorConfig:
    """Build a :class:`CollatorConfig` from a data config.

    Args:
        data_cfg: The ``data`` config containing a ``collate_fn`` section.

    Returns:
        The resolved collator configuration.
    """
    collate_cfg = _to_plain_dict(data_cfg.collate_fn)
    frame_transforms = collate_cfg.get("frame_transforms", [])
    image_size = 224
    if frame_transforms:
        image_size = int(frame_transforms[0].get("height", 224))
    return CollatorConfig(
        image_size=image_size,
        max_frames=int(collate_cfg.get("max_frames", 16)),
        max_text_length=int(collate_cfg.get("max_text_length", 1024)),
        max_streaming_frames=(
            int(collate_cfg["max_streaming_frames"])
            if collate_cfg.get("max_streaming_frames") is not None
            else 200
        ),
    )


def _build_streaming_text(sample: CookingSample, frame_count_by_response: list[int]) -> str:
    """Build the interleaved multi-turn streaming text for a sample.

    Emits the system and initial user turn, then for each response interleaves
    the frames observed before it (as vision placeholders) with an assistant
    turn, appending an ``[END]`` marker to non "next instruction" responses and
    opening a fresh user turn between responses.

    Args:
        sample: The cooking sample with ``responses`` to interleave.
        frame_count_by_response: Number of vision placeholder blocks preceding
            each response, aligned with ``sample.responses``.

    Returns:
        The formatted interleaved video-text string.
    """
    text = f"<|im_start|>system\n{sample.system_prompt}<|im_end|>\n"
    text += f"<|im_start|>user\nCurrent instruction: {sample.instruction}\n"
    for response_idx, (response, frame_count) in enumerate(
        zip(sample.responses, frame_count_by_response)
    ):
        text += "<|vision_start|><|image_pad|><|vision_end|>" * frame_count
        if "next instruction" not in response.lower():
            text += f"<|im_end|>\n<|im_start|>assistant\n{response} [END]<|im_end|>\n"
        else:
            text += f"<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
        if response_idx < len(sample.responses) - 1:
            text += "<|im_start|>user\n"
    return text


def _cap_recent_frames(
    frame_paths: list[str],
    frame_timestamps: list[float],
    max_frames: int | None,
) -> tuple[list[str], list[float]]:
    """Keep only the most recent ``max_frames`` frames as a memory-safety cap.

    Args:
        frame_paths: Frame file paths in temporal order.
        frame_timestamps: Timestamps aligned with ``frame_paths``.
        max_frames: Maximum frames to retain; ``None`` or non-positive disables
            the cap.

    Returns:
        The (possibly) truncated ``(frame_paths, frame_timestamps)`` pair,
        keeping the tail (most recent frames).
    """
    if max_frames is None or max_frames <= 0 or len(frame_paths) <= max_frames:
        return frame_paths, frame_timestamps
    return frame_paths[-max_frames:], frame_timestamps[-max_frames:]


class QwenVideoCollator:
    """Collate cooking samples into Qwen-VL streaming batches.

    Builds interleaved system/user/assistant text with vision placeholder tokens,
    loads and resizes the selected frames, and produces input ids, attention and
    multimodal masks, loss masks, and the auxiliary interactive-response targets.
    """

    def __init__(self, processor: Any, cfg: CollatorConfig) -> None:
        """Initialize the collator.

        Args:
            processor: A Hugging Face ``AutoProcessor`` (must expose ``tokenizer``).
            cfg: The resolved collator configuration.

        Raises:
            ValueError: If ``processor`` is a tokenizer-only object.
        """
        if not hasattr(processor, "tokenizer"):
            raise ValueError(
                "QwenVideoCollator requires an AutoProcessor, not a tokenizer-only object"
            )
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.cfg = cfg
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.assistant_token_id = self._resolve_token_id("assistant")
        self.im_start_token_id = self._resolve_token_id("<|im_start|>")
        self.im_end_token_id = self._resolve_token_id("<|im_end|>")
        self.vision_start_token_id = self._resolve_token_id("<|vision_start|>")
        self.vision_end_token_id = self._resolve_token_id("<|vision_end|>")
        self.image_token_id = self._resolve_token_id("<|image_pad|>")

    def _resolve_token_id(self, token: str) -> int | None:
        """Resolve a token string to a single vocabulary id.

        Falls back to encoding the token when direct conversion yields the
        unknown-token id, accepting the result only if it maps to exactly one id.

        Args:
            token: The token text to resolve.

        Returns:
            The token id, or ``None`` if it cannot be resolved to a single id.
        """
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return None
        if (
            hasattr(self.tokenizer, "unk_token_id")
            and token_id == self.tokenizer.unk_token_id
            and token != self.tokenizer.unk_token
        ):
            encoded = self.tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                return int(encoded[0])
            return None
        return int(token_id)

    def _load_frames(self, frame_paths: list[str]) -> tuple[list[Image.Image], torch.Tensor]:
        """Load and resize frames, substituting blank images on failure.

        Args:
            frame_paths: Paths to the frame images to load.

        Returns:
            A tuple of the loaded RGB images (each resized to the configured
            square size) and an all-ones frame attention mask.
        """
        frames: list[Image.Image] = []
        for frame_path in frame_paths:
            try:
                image = Image.open(frame_path).convert("RGB")
                frames.append(image.resize((self.cfg.image_size, self.cfg.image_size)))
            except (FileNotFoundError, OSError, UnidentifiedImageError):
                frames.append(Image.new("RGB", (self.cfg.image_size, self.cfg.image_size), color=0))
        return frames, torch.ones(len(frames), dtype=torch.long)

    def _get_temporal_window(self, sample: CookingSample) -> tuple[float, float, list[float]]:
        """Compute the frame time window and response timestamps for a sample.

        Reads the start/end times from the sample metadata and, when a
        ``temporal_jitter`` spec is present, jitters the window start and
        (optionally) the feedback/response timestamps for augmentation.

        Args:
            sample: The cooking sample to derive the window from.

        Returns:
            A ``(start_time, end_time, response_timestamps)`` tuple.
        """
        metadata = dict(sample.metadata)
        response_timestamps = [float(timestamp) for timestamp in sample.response_timestamps]
        start_time = float(metadata.get("frame_start_time", metadata.get("start_time", 0.0)))
        end_time = float(metadata.get("frame_end_time", metadata.get("end_time", start_time)))
        original_start_time = start_time
        original_window_duration = max(0.0, end_time - original_start_time)

        temporal_jitter = metadata.get("temporal_jitter")
        if temporal_jitter is not None:
            cfg = temporal_jitter["cfg"]
            start_time = _jitter_window_start(
                start_time,
                response_timestamps,
                cfg,
                amount_key=str(temporal_jitter["amount_key"]),
                random_scale=float(temporal_jitter["random_scale"]),
            )
            if temporal_jitter.get("jitter_feedback_timestamps", False):
                response_timestamps = _jitter_feedback_timestamps(
                    sample.responses, response_timestamps, cfg
                )

        return start_time, end_time, response_timestamps

    def _sample_streaming_frames(
        self,
        sample: CookingSample,
        start_time: float,
        end_time: float,
        response_timestamps: list[float],
    ) -> tuple[list[str], list[float], list[int]]:
        """Select the frames observed before each response in the window.

        Restricts the source frames to ``[start_time, end_time]``, applies the
        recent-frame memory cap, then groups frames into the interval preceding
        each response timestamp. Guarantees at least one frame is selected when
        any are available, and pads ``frame_count_by_response`` to cover every
        response.

        Args:
            sample: The cooking sample providing source frames.
            start_time: Window start time (seconds).
            end_time: Window end time (seconds).
            response_timestamps: Response times partitioning the window.

        Returns:
            A tuple of the selected frame paths, their timestamps, and the
            per-response frame counts.

        Raises:
            ValueError: If source frame paths and timestamps differ in length.
        """
        source_frame_paths = sample.video_frame_paths or sample.frame_paths
        source_frame_timestamps = sample.video_frame_timestamps or sample.frame_timestamps
        if len(source_frame_paths) != len(source_frame_timestamps):
            raise ValueError(
                "CookingSample video_frame_paths and video_frame_timestamps must have the same length"
            )

        frame_paths, frame_timestamps = _select_frames_between(
            source_frame_paths,
            source_frame_timestamps,
            start_time,
            end_time,
        )
        # Legacy Qwen3-VL training kept only the most recent 200 frames before
        # constructing the interleaved stream. This is a memory safety cap, not
        # uniform subsampling: all retained frames are still used at the source
        # 2 fps, and older context is dropped only when a window is too long.
        frame_paths, frame_timestamps = _cap_recent_frames(
            frame_paths,
            frame_timestamps,
            self.cfg.max_streaming_frames,
        )

        selected_frame_paths: list[str] = []
        selected_frame_timestamps: list[float] = []
        frame_count_by_response: list[int] = []
        previous_response_timestamp = start_time - 1.0e-6
        for response_timestamp in response_timestamps:
            clip_indices = [
                idx
                for idx, frame_timestamp in enumerate(frame_timestamps)
                if previous_response_timestamp < frame_timestamp <= float(response_timestamp)
            ]
            frame_count_by_response.append(len(clip_indices))
            selected_frame_paths.extend(frame_paths[idx] for idx in clip_indices)
            selected_frame_timestamps.extend(frame_timestamps[idx] for idx in clip_indices)
            previous_response_timestamp = float(response_timestamp)

        if not selected_frame_paths and frame_paths:
            selected_frame_paths = [frame_paths[-1]]
            selected_frame_timestamps = [frame_timestamps[-1]]
            if frame_count_by_response:
                frame_count_by_response[-1] = 1
            else:
                frame_count_by_response = [1]
        while len(frame_count_by_response) < len(sample.responses):
            frame_count_by_response.append(0)
        return selected_frame_paths, selected_frame_timestamps, frame_count_by_response

    def _select_streaming_frames(self, sample: CookingSample) -> tuple[list[str], list[int]]:
        """Select streaming frames and per-response counts for a sample.

        Thin wrapper over :meth:`_get_temporal_window` and
        :meth:`_sample_streaming_frames` that returns only the frame paths and
        per-response frame counts.

        Args:
            sample: The cooking sample to select frames for.

        Returns:
            A tuple of the selected frame paths and the per-response frame counts.

        Raises:
            ValueError: If responses and response timestamps differ in length.
        """
        if len(sample.responses) != len(sample.response_timestamps):
            raise ValueError(
                "CookingSample responses and response_timestamps must have the same length"
            )

        start_time, end_time, response_timestamps = self._get_temporal_window(sample)
        selected_frame_paths, _, frame_count_by_response = self._sample_streaming_frames(
            sample,
            start_time,
            end_time,
            response_timestamps,
        )
        return selected_frame_paths, frame_count_by_response

    def _encode_streaming_sample(self, sample: CookingSample) -> dict[str, torch.Tensor]:
        """Encode one sample into interleaved streaming model inputs.

        Selects streaming frames, rebuilds the sample with the resolved window
        and frames, loads the frame images, builds the interleaved text, runs the
        processor, applies text truncation, and returns the per-sample tensors.

        Args:
            sample: The cooking sample to encode.

        Returns:
            A dict with ``input_ids``, ``attention_mask``, ``mm_token_type_ids``,
            ``pixel_values``, ``image_grid_thw``, and ``frame_attention_mask``.
        """
        start_time, end_time, response_timestamps = self._get_temporal_window(sample)
        responses = list(sample.responses)
        selected_frame_paths, selected_frame_timestamps, frame_count_by_response = (
            self._sample_streaming_frames(
                sample,
                start_time,
                end_time,
                response_timestamps,
            )
        )
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "start_time": start_time,
                "end_time": end_time,
                "frame_start_time": start_time,
                "frame_end_time": end_time,
            }
        )
        sample = CookingSample(
            system_prompt=sample.system_prompt,
            instruction=sample.instruction,
            responses=responses,
            response_timestamps=response_timestamps,
            frame_paths=selected_frame_paths,
            frame_timestamps=selected_frame_timestamps,
            source=sample.source,
            metadata=metadata,
            video_frame_paths=sample.video_frame_paths,
            video_frame_timestamps=sample.video_frame_timestamps,
        )
        if not selected_frame_paths:
            selected_frame_paths = [""]
            if frame_count_by_response:
                frame_count_by_response[0] = 1
            else:
                frame_count_by_response = [1]
        frames, frame_attention_mask = self._load_frames(selected_frame_paths)
        text = _build_streaming_text(sample, frame_count_by_response)
        encoded = self.processor(
            text=[text],
            images=[frames],
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )

        encoded = self._truncate_encoded(encoded)
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "mm_token_type_ids": encoded.get(
                "mm_token_type_ids", torch.zeros_like(encoded["input_ids"])
            )[0],
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
            "frame_attention_mask": frame_attention_mask,
        }

    def _get_loss_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build a mask over assistant-turn content tokens for LM loss.

        For each row, finds assistant turns (an ``<|im_start|>assistant`` opener,
        or bare ``assistant`` tokens when no im-start id is known) and marks the
        tokens between the turn start and the next ``<|im_end|>`` as supervised.

        Args:
            input_ids: Batched token ids, shape ``(batch, seq_len)``.

        Returns:
            A 0/1 mask tensor of the same shape, 1 on supervised assistant tokens.
        """
        loss_mask = torch.zeros_like(input_ids)
        if self.assistant_token_id is None or self.im_end_token_id is None:
            return loss_mask

        for batch_idx in range(input_ids.size(0)):
            row = input_ids[batch_idx]
            if self.im_start_token_id is not None:
                im_start_idxs = torch.where(row[:-1] == self.im_start_token_id)[0]
                assistant_idxs = (
                    im_start_idxs[row[im_start_idxs + 1] == self.assistant_token_id] + 1
                )
            else:
                assistant_idxs = torch.where(row == self.assistant_token_id)[0]
            im_end_idxs = torch.where(row == self.im_end_token_id)[0]
            for start_idx in assistant_idxs:
                end_candidates = im_end_idxs[im_end_idxs > start_idx]
                if end_candidates.numel() == 0:
                    continue
                end_idx = int(end_candidates[0].item())
                content_start_idx = int(start_idx.item()) + 1
                if end_idx > content_start_idx:
                    loss_mask[batch_idx, content_start_idx:end_idx] = 1
        return loss_mask

    def _truncate_encoded(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Truncate encoded inputs from the left to the text-length budget.

        Keeps the most recent tokens up to ``max_text_length`` (plus all image
        tokens and one). Truncation is skipped if it would drop any image tokens,
        since that would desynchronize the vision inputs.

        Args:
            encoded: Processor output containing at least ``input_ids``,
                ``attention_mask``, and optionally ``mm_token_type_ids``.

        Returns:
            The (possibly) left-truncated ``encoded`` dict.
        """
        input_ids = encoded["input_ids"][0]
        image_token_count = (
            int((input_ids == self.image_token_id).sum().item())
            if self.image_token_id is not None
            else 0
        )
        target_full_len = max(2, self.cfg.max_text_length + image_token_count + 1)
        if input_ids.numel() <= target_full_len:
            return encoded

        start_idx = input_ids.numel() - target_full_len
        truncated_input_ids = input_ids[start_idx:]
        truncated_image_token_count = (
            int((truncated_input_ids == self.image_token_id).sum().item())
            if self.image_token_id is not None
            else 0
        )
        if truncated_image_token_count != image_token_count:
            return encoded

        encoded["input_ids"] = truncated_input_ids.unsqueeze(0)
        encoded["attention_mask"] = encoded["attention_mask"][:, start_idx:]
        if "mm_token_type_ids" in encoded:
            encoded["mm_token_type_ids"] = encoded["mm_token_type_ids"][:, start_idx:]
        return encoded

    def _get_interactive_loss_masks(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the interactive-response (action) loss mask and targets.

        Marks every ``<|vision_end|>`` position as supervised for the action
        head, and sets the target to 0 ("do not respond yet") where a
        ``<|vision_end|>`` is immediately followed by ``<|vision_start|>`` (i.e.
        another frame follows with no intervening response); all other supervised
        positions keep target 1.

        Args:
            input_ids: Batched token ids, shape ``(batch, seq_len)``.

        Returns:
            A ``(interactive_loss_mask, interactive_loss_targets)`` pair, each of
            the same shape as ``input_ids``.
        """
        interactive_loss_mask = torch.zeros_like(input_ids)
        interactive_loss_targets = torch.ones_like(input_ids)
        if self.vision_start_token_id is None or self.vision_end_token_id is None:
            return interactive_loss_mask, interactive_loss_targets

        for batch_idx in range(input_ids.size(0)):
            row = input_ids[batch_idx]
            loss_mask_idxs = torch.where(row == self.vision_end_token_id)[0]
            loss_target_idxs = torch.where(
                torch.logical_and(
                    row[:-1] == self.vision_end_token_id,
                    row[1:] == self.vision_start_token_id,
                )
            )[0]
            interactive_loss_mask[batch_idx, loss_mask_idxs] = 1
            interactive_loss_targets[batch_idx, loss_target_idxs] = 0
        return interactive_loss_mask, interactive_loss_targets

    def __call__(self, samples: list[CookingSample]) -> dict[str, torch.Tensor]:
        """Collate a list of samples into a padded training batch.

        Args:
            samples: The cooking samples to encode and batch.

        Returns:
            A dict of batched tensors (input ids, labels, attention/loss masks,
            pixel values, image grids, and interactive-response targets).
        """
        input_id_rows: list[torch.Tensor] = []
        attention_mask_rows: list[torch.Tensor] = []
        mm_token_type_rows: list[torch.Tensor] = []
        pixel_values_rows: list[torch.Tensor] = []
        image_grid_rows: list[torch.Tensor] = []
        frame_attention_masks: list[torch.Tensor] = []
        is_interactive_rows: list[bool] = []

        for sample in samples:
            encoded = self._encode_streaming_sample(sample)
            input_id_rows.append(encoded["input_ids"])
            attention_mask_rows.append(encoded["attention_mask"])
            mm_token_type_rows.append(encoded["mm_token_type_ids"])
            pixel_values_rows.append(encoded["pixel_values"])
            image_grid_rows.append(encoded["image_grid_thw"])
            frame_attention_masks.append(encoded["frame_attention_mask"])
            is_interactive_rows.append(bool(sample.metadata.get("is_interactive", True)))

        max_seq_len = max(int(row.numel()) for row in input_id_rows)
        padded_input_ids: list[torch.Tensor] = []
        padded_attention_mask: list[torch.Tensor] = []
        padded_mm_token_type_ids: list[torch.Tensor] = []
        for input_ids, attention_mask, mm_token_type_ids in zip(
            input_id_rows, attention_mask_rows, mm_token_type_rows
        ):
            pad_len = max_seq_len - int(input_ids.numel())
            padded_input_ids.append(
                torch.cat(
                    [
                        input_ids,
                        torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            padded_attention_mask.append(
                torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_mm_token_type_ids.append(
                torch.cat([mm_token_type_ids, torch.zeros(pad_len, dtype=torch.long)])
            )

        full_input_ids = torch.stack(padded_input_ids)
        full_attention_mask = torch.stack(padded_attention_mask)
        full_mm_token_type_ids = torch.stack(padded_mm_token_type_ids)
        full_loss_mask = self._get_loss_mask(full_input_ids)
        full_loss_weight = full_loss_mask.to(dtype=torch.bfloat16)
        interactive_loss_mask, interactive_loss_targets = self._get_interactive_loss_masks(
            full_input_ids
        )
        max_frame_count = max(int(mask.numel()) for mask in frame_attention_masks)
        padded_frame_attention_masks = [
            torch.cat([mask, torch.zeros(max_frame_count - int(mask.numel()), dtype=torch.long)])
            for mask in frame_attention_masks
        ]
        return {
            "input_ids": full_input_ids[:, :-1],
            "attention_mask": full_attention_mask[:, :-1],
            "labels": full_input_ids[:, 1:],
            "loss_mask": full_loss_mask[:, 1:],
            "loss_weight": full_loss_weight[:, 1:],
            "interactive_loss_mask": interactive_loss_mask[:, :-1],
            "interactive_loss_targets": interactive_loss_targets[:, :-1].to(dtype=torch.bfloat16),
            "is_interactive": torch.tensor(is_interactive_rows, dtype=torch.bool),
            "pixel_values": torch.cat(pixel_values_rows, dim=0),
            "image_grid_thw": torch.cat(image_grid_rows, dim=0),
            "mm_token_type_ids": full_mm_token_type_ids[:, :-1],
            "frame_attention_mask": torch.stack(padded_frame_attention_masks),
        }
