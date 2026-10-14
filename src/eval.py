# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Evaluation utilities: loss aggregation over a loader and streaming turn-based response generation."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from collator import QwenVideoCollator
    from data import CookingSample
    from model import QwenVisionLanguageModel


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


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move every tensor in a batch to the target device.

    Args:
        batch: Mapping of tensor name to tensor.
        device: Target device.

    Returns:
        The batch with all tensors moved to ``device``.
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def extract_loss(outputs: Any) -> torch.Tensor:
    """Return the loss tensor from model outputs (dict, tuple, or object)."""
    if isinstance(outputs, dict):
        return outputs["loss"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs.loss


def extract_optional_scalar(outputs: Any, key: str) -> torch.Tensor | None:
    """Return ``outputs[key]`` as a tensor, or ``None`` if absent.

    Args:
        outputs: Model outputs (dict or object).
        key: Name of the scalar to extract.

    Returns:
        The value as a tensor, or ``None``.
    """
    if isinstance(outputs, dict):
        value = outputs.get(key)
    else:
        value = getattr(outputs, key, None)
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.tensor(float(value))
    return value


def mean_reduce(value: float, device: torch.device) -> float:
    """Average a scalar across distributed ranks (no-op for a single process)."""
    tensor = torch.tensor([value, 1.0], device=device, dtype=torch.float64)
    if _world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor[0] / tensor[1]).item())


def evaluate_loss(model_engine: Any, loader: DataLoader[Any]) -> dict[str, float]:
    """Compute mean losses over a data loader.

    Args:
        model_engine: The (DeepSpeed) model engine in eval mode.
        loader: The evaluation data loader.

    Returns:
        Mapping of metric name to its mean value across the loader (and ranks).
    """
    model_engine.eval()
    metric_keys = ["loss", "language_loss", "interactive_loss", "interactive_weighted_loss"]
    losses: dict[str, list[float]] = {key: [] for key in metric_keys}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, model_engine.device)
            outputs = model_engine(**batch)
            losses["loss"].append(float(extract_loss(outputs).detach().item()))
            for key in metric_keys[1:]:
                value = extract_optional_scalar(outputs, key)
                if value is not None:
                    losses[key].append(float(value.detach().item()))

    metrics = {}
    for key, values in losses.items():
        if not values:
            continue
        local_loss = sum(values) / max(1, len(values))
        metrics[key] = mean_reduce(local_loss, model_engine.device)
    return metrics


def _build_original_monolithic_system_prompt(sample: CookingSample) -> str:
    """Build the system + initial user turn text for a sample.

    Args:
        sample: The sample providing the system prompt and instruction.

    Returns:
        The opening chat text through the start of the user instruction.
    """
    text = f"<|im_start|>system\n{sample.system_prompt}<|im_end|>\n"
    text += f"<|im_start|>user\nCurrent instruction: {sample.instruction}\n"
    return text


def _build_original_monolithic_input_seq(system_prompt: str, vision_token_count: int) -> str:
    """Append vision placeholder blocks to a system prompt.

    Args:
        system_prompt: The base prompt text to extend.
        vision_token_count: Number of vision placeholder blocks to append.

    Returns:
        The prompt with the vision placeholder blocks appended.
    """
    return system_prompt + "<|vision_start|><|image_pad|><|vision_end|>" * vision_token_count


def _generate_turn_based_response(
    model: QwenVisionLanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    sample: CookingSample,
    selected_frame_paths: list[str],
    collator: QwenVideoCollator,
    max_response_length: int,
    response_logit_threshold: float,
    num_frames_per_second: float,
    max_gap_between_responses: int = 360,
) -> tuple[list[str], list[float], list[float]]:
    """Stream frames through the model, generating interactive responses.

    Feeds frames one at a time, consulting the action head's score at each step:
    when it exceeds ``response_logit_threshold`` the model enters response mode
    and decodes text until an ``[END]``/``Next Instruction`` marker, otherwise it
    advances to the next frame. Emits a synthetic success/next-instruction pair
    if ``max_gap_between_responses`` frames elapse without a response.

    Args:
        model: The vision-language model exposing ``generation_step``.
        tokenizer: The tokenizer used to decode generated ids.
        sample: The sample providing the system prompt and instruction.
        selected_frame_paths: The ordered frames to stream over.
        collator: The collator (used for the configured image size).
        max_response_length: Maximum response length (currently advisory).
        response_logit_threshold: Action score above which a response starts.
        num_frames_per_second: Frame rate, used to convert indices to timestamps.
        max_gap_between_responses: Max frames without a response before forcing a
            synthetic success/next-instruction pair.

    Returns:
        A tuple of predicted feedback texts, their timestamps (seconds), and the
        per-frame action scores.

    Raises:
        RuntimeError: If the generation state machine reaches an invalid state.
    """
    if len(selected_frame_paths) <= 1:
        return [], [], []

    curr_input_vision_idx = 1
    current_response_length = 0
    last_response_input_vision_idx = 0
    system_prompt = _build_original_monolithic_system_prompt(sample)
    input_seq = _build_original_monolithic_input_seq(system_prompt, vision_token_count=1)
    generated_token_ids: list[int] = []
    action_scores: list[float] = []
    pred_feedbacks: list[str] = []
    pred_feedback_timestamps: list[float] = []
    response_mode = False

    while curr_input_vision_idx < len(selected_frame_paths):
        generated_text_prefix = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        prompt_for_model = input_seq + generated_text_prefix
        next_token_id, action_score = model.generation_step(
            tokenizer=tokenizer,
            prompt=prompt_for_model,
            frame_paths=selected_frame_paths[:curr_input_vision_idx],
            image_size=int(getattr(collator.cfg, "image_size", 224)),
        )
        action_scores.append(action_score)

        current_gap_between_responses = curr_input_vision_idx - last_response_input_vision_idx

        if current_response_length > 0:
            generated_token_ids.append(next_token_id)
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            decoded_input_seq = input_seq + generated_text
            pred_response_str = decoded_input_seq.split("assistant")[-1].replace("\n", "")

            if "[END]" in generated_text:
                input_seq += pred_response_str
                input_seq += "<|im_end|>\n<|im_start|>user\n"  # "<|im_end|>" #
                pred_feedbacks.append(pred_response_str)
                pred_feedback_timestamps.append(curr_input_vision_idx / num_frames_per_second)
                last_response_input_vision_idx = curr_input_vision_idx
                current_response_length = 0
                generated_token_ids = []
                if "Success" not in pred_response_str:
                    input_seq += "<|vision_start|><|image_pad|><|vision_end|>"
                    curr_input_vision_idx += 1
                else:
                    pred_feedbacks.append("Next Instruction")
                    pred_feedback_timestamps.append(curr_input_vision_idx / num_frames_per_second)
                    break

            if "Next Instruction" in generated_text:
                pred_feedbacks.append(pred_response_str)
                pred_feedback_timestamps.append(curr_input_vision_idx / num_frames_per_second)
                break

        elif current_response_length == 0 and action_score > response_logit_threshold:
            response_mode = True
            input_seq += "<|im_end|>\n<|im_start|>assistant\n"
            current_response_length += 1
            generated_token_ids = []
        elif current_gap_between_responses >= max_gap_between_responses:
            pred_feedbacks.append("Success: You did it.")
            pred_feedback_timestamps.append(curr_input_vision_idx / num_frames_per_second)
            pred_feedbacks.append("Next Instruction")
            pred_feedback_timestamps.append(curr_input_vision_idx / num_frames_per_second)
            last_response_input_vision_idx = curr_input_vision_idx
            break
        elif action_score <= response_logit_threshold and current_response_length == 0:
            input_seq += "<|vision_start|><|image_pad|><|vision_end|>"
            curr_input_vision_idx += 1
        else:
            raise RuntimeError("Model reached an invalid interactive generation state")

        if response_mode:
            continue

    return pred_feedbacks, pred_feedback_timestamps, action_scores


def _clean_legacy_text(text: Any) -> str:
    """Strip chat control tokens and collapse whitespace in a text value.

    Args:
        text: The raw text (or any value coercible to str; ``None`` becomes "").

    Returns:
        The cleaned, whitespace-normalized text.
    """
    return " ".join(str(text or "").replace("<|im_end|>", "").replace("<|im_start|>", "").split())


def _is_next_instruction_text(text: Any) -> bool:
    """Return whether the cleaned text is the "next instruction" marker.

    Args:
        text: The text to test.

    Returns:
        True if the cleaned, lowercased text equals ``"next instruction"``.
    """
    return _clean_legacy_text(text).strip().lower() == "next instruction"


def _strip_legacy_response_prefix(response: Any) -> str:
    """Remove a leading response-type prefix and normalize a response string.

    Drops a leading ``Success:``/``Feedback:``/``Feedback-Reply:``/
    ``Instruction-Clarification:`` prefix and a ``"0 seconds ago,"`` lead-in, then
    capitalizes the first character.

    Args:
        response: The raw response text.

    Returns:
        The stripped and capitalized response text.
    """
    text = _clean_legacy_text(response)
    for prefix in ["Success:", "Feedback:", "Feedback-Reply:", "Instruction-Clarification:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    if text.lower().startswith("0 seconds ago,"):
        text = text[len("0 seconds ago,") :].strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def _legacy_response_types(sample: CookingSample) -> list[str]:
    """Return the per-response type labels for a sample, if any.

    Reads ``response_types``/``text_types`` from the sample metadata, coerces to
    strings, and drops a leading ``instruction`` entry so the list aligns with
    the response entries.

    Args:
        sample: The sample to read metadata from.

    Returns:
        The list of response-type strings (possibly empty).
    """
    metadata = getattr(sample, "metadata", {}) or {}
    response_types = metadata.get("response_types") or metadata.get("text_types") or []
    response_types = [str(response_type) for response_type in response_types]
    if response_types and response_types[0].strip().lower() == "instruction":
        response_types = response_types[1:]
    return response_types


def _legacy_segment_type(sample: CookingSample, response_idx: int, response: Any) -> str:
    """Classify a response into its legacy instruction-segment type.

    Uses the sample's declared response types when available, falling back to the
    response text prefix, to label the segment as an aligned success or an
    (optionally error-typed) aligned mistake.

    Args:
        sample: The sample providing response types and error metadata.
        response_idx: Index of the response among real (non "next instruction")
            responses.
        response: The response text, used as a fallback signal.

    Returns:
        The normalized segment-type label.
    """
    response_types = _legacy_response_types(sample)
    raw_type = response_types[response_idx] if response_idx < len(response_types) else ""
    normalized_type = raw_type.strip().lower().replace("-", "_")
    if normalized_type.startswith("feedback_action"):
        return normalized_type
    if "success" in normalized_type or _clean_legacy_text(response).startswith("Success:"):
        return "feedback_action_aligned_success"
    if (
        "feedback" in normalized_type
        or "mistake" in normalized_type
        or _clean_legacy_text(response).startswith("Feedback:")
    ):
        metadata = getattr(sample, "metadata", {}) or {}
        error_type = (
            str(metadata.get("error_type") or metadata.get("mistake_type") or "").strip().lower()
        )
        if error_type:
            return f"feedback_action_aligned_mistake_{error_type}"
        return "feedback_action_aligned_mistake"
    return normalized_type or "feedback_action_aligned_response"


def _legacy_mistake_flag(sample: CookingSample, target_responses: list[str]) -> bool:
    """Determine whether a sample's segment contains a mistake.

    Prefers an explicit ``mistake_flag`` in metadata; otherwise infers it from
    whether any non "next instruction" response is a feedback/mistake segment.

    Args:
        sample: The sample providing metadata and segment types.
        target_responses: The ground-truth responses for the segment.

    Returns:
        True if the segment is flagged as (or inferred to contain) a mistake.
    """
    metadata = getattr(sample, "metadata", {}) or {}
    if metadata.get("mistake_flag") is not None:
        return bool(metadata.get("mistake_flag"))
    return any(
        (not _is_next_instruction_text(response))
        and (
            _clean_legacy_text(response).startswith("Feedback:")
            or "mistake" in _legacy_segment_type(sample, response_idx, response)
        )
        for response_idx, response in enumerate(target_responses)
    )


def _build_legacy_turn_based_record(
    sample: CookingSample, payload: dict[str, Any]
) -> dict[str, Any]:
    """Assemble a legacy turn-based evaluation record for one segment.

    Builds aligned ground-truth QA pairs/timestamps/segment-types (skipping
    "next instruction" responses) and the cleaned predicted feedback texts and
    timestamps, together with the mistake flag.

    Args:
        sample: The sample providing ground-truth instruction and responses.
        payload: Per-segment data including ``video_id``, ``start_time``,
            ``pred_feedbacks``, and ``pred_feedback_timestamps``.

    Returns:
        A dict record with prediction and ground-truth fields for scoring.
    """
    instruction = _clean_legacy_text(getattr(sample, "instruction", payload.get("instruction", "")))
    start_time = float(payload.get("start_time") or 0.0)
    target_responses = list(getattr(sample, "responses", payload.get("target_responses", [])))
    target_timestamps = [
        float(timestamp)
        for timestamp in getattr(
            sample, "response_timestamps", payload.get("target_response_timestamps", [])
        )
    ]

    qa_pair = [instruction]
    qa_timestamps = [start_time]
    instruction_segment_types = ["instruction"]
    real_response_idx = 0
    for response, timestamp in zip(target_responses, target_timestamps):
        if _is_next_instruction_text(response):
            continue
        qa_pair.append(_strip_legacy_response_prefix(response))
        qa_timestamps.append(float(timestamp))
        instruction_segment_types.append(_legacy_segment_type(sample, real_response_idx, response))
        real_response_idx += 1
    instruction_segment_types.append("end_of_segment")

    pred_texts = [f"Instruct: {instruction}"]
    pred_timestamps = [start_time]
    for pred_text, pred_timestamp in zip(
        payload.get("pred_feedbacks", []),
        payload.get("pred_feedback_timestamps", []),
    ):
        if _is_next_instruction_text(pred_text):
            continue
        pred_texts.append(_clean_legacy_text(pred_text))
        pred_timestamps.append(float(pred_timestamp))

    return {
        "video_id": payload.get("video_id"),
        "start_time": start_time,
        "pred_texts": pred_texts,
        "pred_timestamps": pred_timestamps,
        "mistake_flag": _legacy_mistake_flag(sample, target_responses),
        "qa_pair": qa_pair,
        "qa_timestamps": qa_timestamps,
        "instruction_segment_types": instruction_segment_types,
    }


def _gt_start_time(sample: CookingSample) -> float:
    """Return the ground-truth segment start time (seconds) for a sample.

    Args:
        sample: The sample to read metadata from.

    Returns:
        ``gt_start_time`` if present, else ``start_time``, else 0.0.
    """
    metadata = getattr(sample, "metadata", {}) or {}
    return float(metadata.get("gt_start_time", metadata.get("start_time", 0.0)))


def _gt_end_time(sample: CookingSample) -> float:
    """Return the ground-truth segment end time (seconds) for a sample.

    Args:
        sample: The sample to read metadata from.

    Returns:
        ``gt_end_time`` if present, else ``end_time``, else the start time.
    """
    metadata = getattr(sample, "metadata", {}) or {}
    return float(metadata.get("gt_end_time", metadata.get("end_time", _gt_start_time(sample))))


def _extract_success_skip_back_seconds(text: Any) -> int | None:
    """Parse a ``"Success: N seconds ago"`` back-skip amount from text.

    Args:
        text: The feedback text to search.

    Returns:
        The number of seconds ``N``, or ``None`` if the pattern is absent.
    """
    match = re.search(r"Success: (\d+) seconds ago", str(text or ""))
    if match:
        return int(match.group(1))
    return None


def _postprocess_legacy_generated_feedbacks(
    pred_feedbacks: list[str],
    pred_feedback_timestamps: list[float],
) -> tuple[list[str], list[float], float | None, bool]:
    """Clean generated feedbacks and derive the next segment's start time.

    Strips ``[END]`` markers, applies any ``"Success: N seconds ago"`` back-skip
    to timestamps, and tracks the start time for the following instruction
    segment from the last processed feedback.

    Args:
        pred_feedbacks: The raw generated feedback texts.
        pred_feedback_timestamps: Timestamps aligned with ``pred_feedbacks``.

    Returns:
        A tuple of the processed feedbacks, processed timestamps, and the next
        instruction segment's start time (or ``None``). (A trailing flag value is
        computed internally but not returned by the current implementation.)
    """
    processed_feedbacks: list[str] = []
    processed_timestamps: list[float] = []
    next_instruction_segment_start_time: float | None = None
    ask_for_next_instruction = False

    for pred_feedback, pred_feedback_timestamp in zip(pred_feedbacks, pred_feedback_timestamps):
        skip_back_amount = _extract_success_skip_back_seconds(pred_feedback) or 0
        adjusted_timestamp = float(pred_feedback_timestamp) - float(skip_back_amount)
        if "Next Instruction" not in pred_feedback:
            processed_feedbacks.append(pred_feedback.replace("[END]", "").strip())
            processed_timestamps.append(adjusted_timestamp)
            next_instruction_segment_start_time = adjusted_timestamp
        else:
            next_instruction_segment_start_time = adjusted_timestamp
    ask_for_next_instruction = True
    return processed_feedbacks, processed_timestamps, next_instruction_segment_start_time


def _select_turn_based_eval_frames(
    sample: CookingSample,
    max_eval_frames: int | None,
    num_frames_per_second: float,
    start_time: float | None = None,
) -> list[str]:
    """Select the frames to stream for one turn-based evaluation segment.

    Chooses the frame nearest ``start_time`` (defaulting to the sample's
    ground-truth start) and keeps all subsequent frames, optionally capped at
    ``max_eval_frames``.

    Args:
        sample: The sample providing frame paths and timestamps.
        max_eval_frames: Optional cap on the number of frames returned.
        num_frames_per_second: Frame rate used to map ``start_time`` to an index.
        start_time: Segment start time (seconds); defaults to the sample's
            ground-truth start when ``None``.

    Returns:
        The selected frame paths in temporal order (empty if none available).
    """
    frame_paths = list(
        getattr(sample, "video_frame_paths", []) or getattr(sample, "frame_paths", [])
    )
    frame_timestamps = list(
        getattr(sample, "video_frame_timestamps", []) or getattr(sample, "frame_timestamps", [])
    )
    if not frame_paths:
        return []

    if start_time is None:
        start_time = _gt_start_time(sample)

    start_idx = max(0, int(start_time * num_frames_per_second))
    if start_idx >= len(frame_paths):
        start_idx = min(
            range(len(frame_timestamps)),
            key=lambda idx: abs(float(frame_timestamps[idx]) - start_time),
        )
    selected_paths = frame_paths[start_idx:]

    if max_eval_frames is not None:
        selected_paths = selected_paths[: int(max_eval_frames)]
    return selected_paths


def _load_existing_legacy_predictions(
    predictions_save_path: Path | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Load previously saved predictions to support resuming evaluation.

    Reads the JSON prediction file if present, tolerating a missing file or
    malformed JSON by returning empty results.

    Args:
        predictions_save_path: Path to the saved predictions JSON, or ``None``.

    Returns:
        A tuple of the loaded prediction records and the set of video ids already
        evaluated.
    """
    if predictions_save_path is None or not predictions_save_path.exists():
        return [], set()
    try:
        loaded = json.loads(predictions_save_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], set()
    if not isinstance(loaded, list):
        return [], set()
    predictions = [record for record in loaded if isinstance(record, dict)]
    evaluated_video_ids = {
        str(record.get("video_id")) for record in predictions if record.get("video_id") is not None
    }
    return predictions, evaluated_video_ids


def _video_id_for_sample(sample: CookingSample, fallback_idx: int) -> str:
    """Return a sample's video id, falling back to a synthetic id.

    Args:
        sample: The sample to read metadata from.
        fallback_idx: Index used to build a ``sample-<idx>`` id when none exists.

    Returns:
        The video id string.
    """
    metadata = getattr(sample, "metadata", {}) or {}
    return str(metadata.get("video_id") or f"sample-{fallback_idx}")


def _build_legacy_video_groups(
    dataset: Dataset[CookingSample],
    max_eval_samples: int | None,
) -> list[tuple[str, list[tuple[int, CookingSample]]]]:
    """Group dataset samples by video id for turn-based evaluation.

    Buckets samples by video id, shuffles the video order, optionally limits the
    number of videos, and sorts each video's samples by ground-truth start time.

    Args:
        dataset: The evaluation dataset.
        max_eval_samples: Optional cap on the number of videos to keep.

    Returns:
        A list of ``(video_id, [(sample_idx, sample), ...])`` groups.
    """
    grouped: dict[str, list[tuple[int, CookingSample]]] = {}
    ordered_video_ids: list[str] = []
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        video_id = _video_id_for_sample(sample, sample_idx)
        if video_id not in grouped:
            grouped[video_id] = []
            ordered_video_ids.append(video_id)
        grouped[video_id].append((sample_idx, sample))

    random.shuffle(ordered_video_ids)
    if max_eval_samples is not None:
        ordered_video_ids = ordered_video_ids[:max_eval_samples]

    video_groups: list[tuple[str, list[tuple[int, CookingSample]]]] = []
    for video_id in ordered_video_ids:
        samples = sorted(grouped[video_id], key=lambda item: (_gt_start_time(item[1]), item[0]))
        video_groups.append((video_id, samples))
    return video_groups


def run_eval(
    model: QwenVisionLanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset[CookingSample],
    collator: QwenVideoCollator,
    cfg: DictConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Generate streaming predictions over an evaluation dataset.

    Args:
        model: The vision-language model.
        tokenizer: The tokenizer.
        dataset: The evaluation dataset.
        collator: The collator used to encode samples.
        cfg: The full config (reads ``evaluator.evaluator_kwargs``).
        device: Device to run generation on.

    Returns:
        A list of per-video prediction payloads.
    """
    evaluator_cfg = cfg.get("evaluator", {}).get("evaluator_kwargs", {})
    turn_based_eval = bool(evaluator_cfg.get("turn_based_eval", False))
    max_response_length = int(evaluator_cfg.get("max_response_length", 64))
    max_eval_samples = evaluator_cfg.get("max_eval_samples")
    if max_eval_samples is not None:
        max_eval_samples = int(max_eval_samples)
    response_logit_threshold = float(evaluator_cfg.get("response_logit_threshold", -0.25))
    max_gap_between_responses = int(evaluator_cfg.get("max_gap_between_responses", 360))
    max_eval_frames = evaluator_cfg.get("max_eval_frames")
    num_frames_per_second = float(cfg.model.net.num_of_frames_per_sec)
    predictions_save_path = evaluator_cfg.get("predictions_save_path") or evaluator_cfg.get(
        "resume_from_file"
    )
    if predictions_save_path is not None:
        predictions_save_path = Path(str(predictions_save_path))
        if not predictions_save_path.is_absolute():
            predictions_save_path = Path(cfg.paths.work_dir) / predictions_save_path
        predictions_save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.eval()

    save_pred_dicts, evaluated_video_ids = _load_existing_legacy_predictions(predictions_save_path)
    video_groups = _build_legacy_video_groups(dataset, max_eval_samples)
    progress_bar = tqdm(
        video_groups, desc="Generating", disable=not _is_rank_zero(), dynamic_ncols=True
    )
    for video_id, video_samples in progress_bar:
        if video_id in evaluated_video_ids:
            continue

        pred_feedbacks: list[str] = []
        pred_feedback_timestamps: list[float] = []
        qa_pairs: list[list[str]] = []
        qa_timestamps: list[list[float]] = []
        instruction_segment_types: list[list[str]] = []
        mistake_flags: list[bool] = []
        next_instruction_segment_start_time = _gt_start_time(video_samples[0][1])

        for sample_idx, sample in video_samples:
            gt_start_time = _gt_start_time(sample)
            gt_end_time = _gt_end_time(sample)
            if turn_based_eval:
                next_instruction_segment_start_time = gt_start_time

            selected_frame_paths = _select_turn_based_eval_frames(
                sample,
                max_eval_frames,
                num_frames_per_second,
                start_time=next_instruction_segment_start_time if (not turn_based_eval) else None,
            )
            raw_pred_feedbacks, relative_pred_timestamps, _ = _generate_turn_based_response(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                selected_frame_paths=selected_frame_paths,
                collator=collator,
                max_response_length=max_response_length,
                response_logit_threshold=response_logit_threshold,
                num_frames_per_second=num_frames_per_second,
                max_gap_between_responses=max_gap_between_responses,
            )
            raw_pred_timestamps = [
                next_instruction_segment_start_time + float(timestamp)
                for timestamp in relative_pred_timestamps
            ]
            processed_feedbacks, processed_timestamps, predicted_next_start = (
                _postprocess_legacy_generated_feedbacks(raw_pred_feedbacks, raw_pred_timestamps)
            )
            if turn_based_eval and (len(raw_pred_feedbacks) == 0 or predicted_next_start is None):
                break

            if predicted_next_start is None:
                break
            next_instruction_segment_start_time = predicted_next_start

            segment_record = _build_legacy_turn_based_record(
                sample,
                {
                    "video_id": video_id,
                    "start_time": gt_start_time,
                    "end_time": gt_end_time,
                    "pred_feedbacks": processed_feedbacks,
                    "pred_feedback_timestamps": processed_timestamps,
                },
            )
            pred_feedbacks.extend(segment_record["pred_texts"])
            pred_feedback_timestamps.extend(segment_record["pred_timestamps"])
            qa_pairs.append(segment_record["qa_pair"])
            qa_timestamps.append(segment_record["qa_timestamps"])
            instruction_segment_types.append(segment_record["instruction_segment_types"])
            mistake_flags.append(bool(segment_record["mistake_flag"]))

            if len(raw_pred_feedbacks) == 0 or predicted_next_start is None:
                break

        save_pred_dicts.append(
            {
                "video_id": video_id,
                "pred_texts": pred_feedbacks,
                "pred_timestamps": pred_feedback_timestamps,
                "mistake_flag": mistake_flags,
                "qa_pairs": qa_pairs,
                "qa_timestamps": qa_timestamps,
                "instruction_segment_types": instruction_segment_types,
            }
        )
        evaluated_video_ids.add(video_id)
        if _is_rank_zero() and predictions_save_path is not None:
            predictions_save_path.write_text(
                json.dumps(save_pred_dicts, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    return save_pred_dicts
