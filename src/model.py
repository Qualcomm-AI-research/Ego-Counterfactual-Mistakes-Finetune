# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Qwen3.5 vision-language model with an auxiliary interactive-response (action) head."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "items"):
        return {key: _to_plain_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, list):
        return [_to_plain_dict(value) for value in cfg]
    return cfg


def _enable_gradient_checkpointing(model: nn.Module, use_reentrant: bool) -> None:
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )
        except TypeError:
            model.gradient_checkpointing_enable()


def _disable_gradient_checkpointing(module: nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_disable"):
        try:
            module.gradient_checkpointing_disable()
        except TypeError:
            pass
    for submodule in module.modules():
        if hasattr(submodule, "gradient_checkpointing"):
            submodule.gradient_checkpointing = False


def _enable_input_require_grads(model: nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    if not hasattr(model, "get_input_embeddings"):
        return
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return

    def make_inputs_require_grad(
        _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        """Forward hook that makes input embeddings require grad (for checkpointing)."""
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)


class Qwen3p5ForConditionalGenerationWithActions(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 conditional-generation model with an interactive-response head.

    Adds a linear action head over hidden states to predict when to intervene,
    alongside the standard multimodal language-modeling outputs.
    """

    def __init__(self, config: Any) -> None:
        """Initialize the Qwen3.5 model and a placeholder action head.

        Args:
            config: The Hugging Face model config.
        """
        super().__init__(config)
        self.hidden_size = config.text_config.hidden_size
        self.action_head = None

    def init_action_head(self, num_actions: int = 1) -> None:
        """Create the linear interactive-response head.

        Args:
            num_actions: Number of action logits to predict per position.
        """
        self.action_head = nn.Linear(self.hidden_size, num_actions, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> SimpleNamespace:
        """Run the VLM forward pass and the auxiliary action head.

        Accepts the standard Qwen multimodal inputs and returns language-model
        logits along with the action-head logits.

        Returns:
            A namespace with the language-model outputs and ``action_logits``.
        """
        return_logits = bool(kwargs.pop("return_logits", False))
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :]) if return_logits else None
        if self.action_head is not None:
            action_hidden_states = hidden_states[:, slice_indices, :]
            if hasattr(self.action_head, "weight"):
                action_hidden_states = action_hidden_states.to(dtype=self.action_head.weight.dtype)
            actions = self.action_head(action_hidden_states)
        else:
            actions = None
        loss = None
        if labels is not None and logits is None:
            logits = self.lm_head(hidden_states)
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            actions=actions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
            last_hidden_state=hidden_states,
        )


class QwenVisionLanguageModel(nn.Module):
    """Wrapper module that builds the Qwen VLM, processor, and training loss.

    Loads the pretrained (or fresh) Qwen3.5 model, optionally applies LoRA, exposes
    streaming generation helpers, and computes the combined LM + interactive loss.
    """

    def __init__(self, cfg: Any) -> None:
        """Build the Qwen VLM (optionally LoRA-wrapped) from a model config.

        Args:
            cfg: The ``model`` config, including ``net`` (model name, action head,
                optional PEFT/LoRA settings) and tokenizer options.
        """
        super().__init__()
        plain_cfg = _to_plain_dict(cfg)
        net_cfg = plain_cfg["net"]
        model_cfg = net_cfg["model"]
        pretrained_name = model_cfg["pretrained_model_name_or_path"]
        device_is_cuda = torch.cuda.is_available()
        torch_dtype = torch.bfloat16 if device_is_cuda else torch.float32
        load_pretrained_weights = bool(plain_cfg.get("load_pretrained_weights", True))

        self.processor = AutoProcessor.from_pretrained(
            pretrained_name,
            trust_remote_code=True,
            use_fast=bool(plain_cfg.get("tokenizer", {}).get("use_fast", True)),
        )
        self.tokenizer = self.processor.tokenizer

        if load_pretrained_weights:
            self.qwen_model = Qwen3p5ForConditionalGenerationWithActions.from_pretrained(
                pretrained_name,
                trust_remote_code=True,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            qwen_config = AutoConfig.from_pretrained(pretrained_name, trust_remote_code=True)
            self.qwen_model = Qwen3p5ForConditionalGenerationWithActions(qwen_config)
            self.qwen_model.to(dtype=torch_dtype)

        tokenizer_vocab_size = len(self.tokenizer)
        input_embeddings = self.qwen_model.get_input_embeddings()
        if input_embeddings is not None and input_embeddings.num_embeddings != tokenizer_vocab_size:
            self.qwen_model.resize_token_embeddings(tokenizer_vocab_size)

        action_head_cfg = net_cfg.get("action_head_config") or {"num_actions": 1}
        self.qwen_model.init_action_head(int(action_head_cfg.get("num_actions", 1)))
        if self.qwen_model.action_head is not None:
            self.qwen_model.action_head.to(dtype=torch_dtype)

        if hasattr(self.qwen_model.config, "use_cache"):
            self.qwen_model.config.use_cache = False
        if hasattr(self.qwen_model.config, "text_config") and hasattr(
            self.qwen_model.config.text_config, "use_cache"
        ):
            self.qwen_model.config.text_config.use_cache = False
        gradient_checkpointing_enabled = bool(plain_cfg.get("gradient_checkpointing", False))
        gradient_checkpointing_use_reentrant = bool(
            plain_cfg.get("gradient_checkpointing_use_reentrant", True)
        )

        peft_cfg = net_cfg.get("peft_config")
        if net_cfg.get("is_lora_model", False) and peft_cfg:
            lora_config = LoraConfig(**peft_cfg)
            self.qwen_model.model.language_model = get_peft_model(
                self.qwen_model.model.language_model, lora_config
            )
            self.qwen_model.to(dtype=torch_dtype)

        if gradient_checkpointing_enabled:
            _enable_input_require_grads(self.qwen_model)
            if hasattr(self.qwen_model.model, "language_model"):
                _enable_input_require_grads(self.qwen_model.model.language_model)
            _enable_gradient_checkpointing(self.qwen_model, gradient_checkpointing_use_reentrant)

        if not bool(net_cfg.get("train_vision", False)) and hasattr(
            self.qwen_model.model, "visual"
        ):
            for parameter in self.qwen_model.model.visual.parameters():
                parameter.requires_grad = False
            _disable_gradient_checkpointing(self.qwen_model.model.visual)

        self.config = self.qwen_model.config
        self.criterion_action_weight = float(plain_cfg.get("criterion_action_weight", 1.0))
        self.action_weight_warmup_steps = int(plain_cfg.get("action_weight_warmup_steps", 0))
        self.action_positive_boost_cap = float(plain_cfg.get("action_positive_boost_cap", 100.0))
        self.action_negative_only_weight = float(plain_cfg.get("action_negative_only_weight", 0.01))
        self.action_positive_token_weight = float(
            plain_cfg.get("action_positive_token_weight", 1.0)
        )
        self.action_label_smoothing = float(plain_cfg.get("action_label_smoothing", 0.0))
        self.action_bootstrap_beta = float(plain_cfg.get("action_bootstrap_beta", 1.0))
        self.action_loss_clip = float(plain_cfg.get("action_loss_clip", 0.0))
        self.action_focal_alpha = float(plain_cfg.get("action_focal_alpha", 0.85))
        self.action_focal_gamma = float(plain_cfg.get("action_focal_gamma", 1.5))
        self._train_step = 0

    def _binary_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_weight = (1.0 - p_t) ** self.action_focal_gamma
        alpha_t = targets * self.action_focal_alpha + (1.0 - targets) * (
            1.0 - self.action_focal_alpha
        )
        return alpha_t * focal_weight * bce_loss

    def _current_action_loss_weight(self, reference: torch.Tensor) -> torch.Tensor:
        current_action_weight = self.criterion_action_weight
        if self.action_weight_warmup_steps > 0:
            warmup_step = self._train_step + 1 if self.training else self.action_weight_warmup_steps
            warmup_factor = min(float(warmup_step) / float(self.action_weight_warmup_steps), 1.0)
            current_action_weight = self.criterion_action_weight * warmup_factor
        return reference.new_tensor(float(current_action_weight))

    def _compute_language_loss_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        supervised_positions = loss_mask > 0
        masked_hidden_states = hidden_states[supervised_positions]
        if masked_hidden_states.numel() == 0:
            return hidden_states.sum() * 0.0

        if hasattr(self.qwen_model.lm_head, "weight"):
            masked_hidden_states = masked_hidden_states.to(
                dtype=self.qwen_model.lm_head.weight.dtype
            )
        masked_logits = self.qwen_model.lm_head(masked_hidden_states)
        masked_labels = labels[supervised_positions]

        if loss_weight is None:
            return F.cross_entropy(masked_logits, masked_labels)

        masked_weights = loss_weight[supervised_positions].to(masked_logits)
        token_losses = F.cross_entropy(masked_logits, masked_labels, reduction="none")
        loss_num = (token_losses * masked_weights).sum()
        loss_den = masked_weights.sum()
        return loss_num / (loss_den + 1e-8)

    def _compute_interactive_loss(
        self,
        actions: torch.Tensor | None,
        is_interactive: torch.Tensor | None,
        interactive_loss_mask: torch.Tensor | None,
        interactive_loss_targets: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            actions is None
            or is_interactive is None
            or interactive_loss_mask is None
            or interactive_loss_targets is None
        ):
            return None

        loss_num = reference.new_zeros(())
        loss_den = reference.new_zeros(())
        for batch_idx in range(actions.size(0)):
            if not bool(is_interactive[batch_idx].item()):
                continue
            valid_idxs = torch.where(interactive_loss_mask[batch_idx] == 1)[0]
            if valid_idxs.numel() == 0:
                continue

            hard_targets = interactive_loss_targets[batch_idx, valid_idxs].to(actions)
            action_logits = actions[batch_idx, valid_idxs, :].squeeze(-1)
            action_targets = hard_targets

            if self.action_label_smoothing > 0.0:
                eps = min(max(self.action_label_smoothing, 0.0), 1.0)
                action_targets = action_targets * (1.0 - eps) + 0.5 * eps

            if self.action_bootstrap_beta < 1.0:
                beta = min(max(self.action_bootstrap_beta, 0.0), 1.0)
                pred_probs = torch.sigmoid(action_logits).detach().to(action_targets)
                action_targets = beta * action_targets + (1.0 - beta) * pred_probs

            token_losses = self._binary_focal_loss(action_logits, action_targets)
            if self.action_loss_clip > 0.0:
                token_losses = torch.clamp(token_losses, max=self.action_loss_clip)

            pos_count = hard_targets.sum()
            valid_count = hard_targets.new_tensor(float(hard_targets.numel()))
            neg_count = valid_count - pos_count

            if pos_count > 0:
                pos_token_weight = (
                    torch.clamp(
                        neg_count / (pos_count + 1e-8),
                        min=1.0,
                        max=self.action_positive_boost_cap,
                    )
                    * self.action_positive_token_weight
                )
                sample_scale = hard_targets.new_tensor(1.0)
                token_weights = torch.where(
                    hard_targets > 0.5,
                    pos_token_weight,
                    hard_targets.new_tensor(1.0),
                )
            else:
                sample_scale = hard_targets.new_tensor(self.action_negative_only_weight)
                token_weights = torch.ones_like(hard_targets)

            loss_num = loss_num + (token_losses * token_weights.to(token_losses)).sum().to(
                loss_num
            ) * sample_scale.to(loss_num)
            loss_den = loss_den + token_weights.sum().to(loss_den) * sample_scale.to(loss_den)

        return loss_num / (loss_den + 1e-8)

    @property
    def language_model(self) -> nn.Module:
        """Return the underlying language model module."""
        return self.qwen_model.model.language_model

    @property
    def action_head(self) -> nn.Module | None:
        """Return the interactive-response head, or ``None`` if not initialized."""
        return self.qwen_model.action_head

    def _load_images(self, frame_paths: list[str], image_size: int = 224) -> list[Image.Image]:
        images: list[Image.Image] = []
        output_size = (int(image_size), int(image_size))
        for frame_path in frame_paths:
            try:
                image = Image.open(frame_path).convert("RGB")
                images.append(image.resize(output_size))
            except (FileNotFoundError, OSError, UnidentifiedImageError):
                images.append(Image.new("RGB", output_size, color=0))
        return images

    def _prepare_processor_inputs(
        self,
        prompt: str,
        frame_paths: list[str],
        image_size: int = 224,
    ) -> dict[str, torch.Tensor]:
        images = self._load_images(frame_paths, image_size=image_size)
        inputs = self.processor(text=[prompt], images=[images], return_tensors="pt")
        device = next(self.parameters()).device
        model_dtype = next(self.qwen_model.parameters()).dtype

        prepared: dict[str, torch.Tensor] = {
            "input_ids": inputs["input_ids"].long().to(device),
            "pixel_values": inputs["pixel_values"].to(dtype=model_dtype, device=device),
            "image_grid_thw": inputs["image_grid_thw"].long().to(device),
        }
        if "attention_mask" in inputs:
            prepared["attention_mask"] = inputs["attention_mask"].long().to(device)
        if "mm_token_type_ids" in inputs:
            prepared["mm_token_type_ids"] = inputs["mm_token_type_ids"].long().to(device)
        return prepared

    @torch.no_grad()
    def generation_step(
        self,
        tokenizer: Any,
        prompt: str,
        frame_paths: list[str],
        image_size: int = 224,
    ) -> tuple[int, float]:
        """Predict the next token and action score for one decoding step.

        Args:
            tokenizer: Tokenizer (kept for API symmetry; the model's own is used).
            prompt: The current text prompt with vision placeholders.
            frame_paths: Frames to attend over for this step.
            image_size: Square size each frame is resized to.

        Returns:
            A tuple of (next token id, interactive-response/action score).
        """
        self.eval()
        inputs = self._prepare_processor_inputs(prompt, frame_paths, image_size=image_size)
        outputs = self.qwen_model(
            **inputs,
            return_logits=True,
            use_cache=False,
        )
        next_token_id = int(torch.argmax(outputs.logits[:, -1, :].float(), dim=-1).item())
        action_score = float(outputs.actions[:, -1, 0].detach().float().item())
        return next_token_id, action_score

    @torch.no_grad()
    def generate_response(
        self,
        tokenizer: Any,
        prompt: str,
        frame_paths: list[str],
        max_new_tokens: int = 64,
        stop_at_sentence_end: bool = True,
        image_size: int = 224,
    ) -> str:
        """Greedily decode a response string for the given prompt and frames.

        Args:
            tokenizer: Tokenizer (kept for API symmetry; the model's own is used).
            prompt: The current text prompt with vision placeholders.
            frame_paths: Frames to attend over.
            max_new_tokens: Maximum number of tokens to generate.
            stop_at_sentence_end: Stop early at a sentence boundary or next-instruction marker.
            image_size: Square size each frame is resized to.

        Returns:
            The decoded response text.
        """
        generated_ids: list[int] = []
        current_prompt = prompt
        eos_token_ids = {
            token_id
            for token_id in [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ]
            if token_id is not None
        }
        for _ in range(max_new_tokens):
            next_token_id, _ = self.generation_step(
                self.tokenizer,
                current_prompt,
                frame_paths,
                image_size=image_size,
            )
            if next_token_id in eos_token_ids:
                break
            generated_ids.append(next_token_id)
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            if stop_at_sentence_end:
                if "Next Instruction" in decoded_text:
                    break
                if decoded_text.endswith(".") and "Success:" not in decoded_text:
                    break
            current_prompt = prompt + self.tokenizer.decode(
                generated_ids, skip_special_tokens=False
            )
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        loss_weight: torch.Tensor | None = None,
        is_interactive: torch.Tensor | None = None,
        interactive_loss_mask: torch.Tensor | None = None,
        interactive_loss_targets: torch.Tensor | None = None,
        interactive_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        frame_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        """Compute the training loss for a collated batch.

        Combines the masked language-modeling loss over assistant tokens with the
        auxiliary interactive-response (action) loss.

        Returns:
            A dict of loss tensors and scalar diagnostics for logging.
        """
        qwen_keys = {"image_grid_thw", "video_grid_thw", "pixel_values_videos", "mm_token_type_ids"}
        if not any(key in kwargs for key in qwen_keys):
            raise ValueError(
                "Qwen3.5 forward requires processor outputs such as pixel_values/image_grid_thw. "
                "The legacy TinyVisionEncoder tensor path is no longer supported."
            )
        model_dtype = next(self.qwen_model.parameters()).dtype
        if pixel_values is not None and torch.is_floating_point(pixel_values):
            pixel_values = pixel_values.to(dtype=model_dtype)
        if "pixel_values_videos" in kwargs and torch.is_tensor(kwargs["pixel_values_videos"]):
            pixel_values_videos = kwargs["pixel_values_videos"]
            if torch.is_floating_point(pixel_values_videos):
                kwargs["pixel_values_videos"] = pixel_values_videos.to(dtype=model_dtype)

        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_logits=False,
            **kwargs,
        )

        language_loss = self._compute_language_loss_from_hidden_states(
            outputs.last_hidden_state, labels, loss_mask, loss_weight
        )
        interactive_loss = self._compute_interactive_loss(
            outputs.actions,
            is_interactive,
            interactive_loss_mask,
            interactive_loss_targets,
            language_loss,
        )
        interactive_loss_weight = self._current_action_loss_weight(language_loss)
        if interactive_loss is None:
            interactive_weighted_loss = None
            loss = language_loss
        else:
            interactive_weighted_loss = interactive_loss_weight * interactive_loss
            loss = language_loss + interactive_weighted_loss

        if self.training and torch.is_grad_enabled():
            self._train_step += 1

        return {
            "loss": loss,
            "language_loss": language_loss.detach(),
            "interactive_loss": interactive_loss.detach() if interactive_loss is not None else None,
            "interactive_weighted_loss": (
                interactive_weighted_loss.detach()
                if interactive_weighted_loss is not None
                else None
            ),
            "interactive_loss_weight": (
                interactive_loss_weight.detach() if interactive_loss is not None else None
            ),
            "logits": outputs.logits,
            "last_hidden_state": outputs.last_hidden_state,
            "actions": outputs.actions,
        }
