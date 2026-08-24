# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived in part from Heretic's activation-steering runtime.

"""Model-agnostic chat generation and exact-site residual extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .capture import capture_last_token_layer_outputs
from .prompts import parse_binary_letter


def render_chat_texts(
    tokenizer: Any,
    chats: list[list[dict[str, str]]],
    *,
    chat_template_kwargs: dict[str, Any],
    completions: list[str] | None = None,
) -> list[str]:
    """Render chats at the generation boundary using frozen model kwargs."""

    if not chats:
        return []
    rendered = tokenizer.apply_chat_template(
        chats,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    if not isinstance(rendered, list) or len(rendered) != len(chats):
        raise ValueError("Tokenizer returned an invalid batched chat rendering")
    texts = [str(text) for text in rendered]
    if completions is None:
        return texts
    if len(completions) != len(texts) or any(
        completion not in {"A", "B"} for completion in completions
    ):
        raise ValueError("Forced completions must align and contain only A/B")
    return [
        text + completion for text, completion in zip(texts, completions, strict=True)
    ]


def _tokenize(tokenizer: Any, texts: list[str], device: str) -> Any:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
    )
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise ValueError("Tokenizer must return input_ids and attention_mask")
    return inputs.to(device)


def generate_binary_answers(
    model: Any,
    tokenizer: Any,
    chats: list[list[dict[str, str]]],
    *,
    chat_template_kwargs: dict[str, Any],
    batch_size: int,
    device: str,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> list[dict[str, str | None]]:
    """Greedily generate and strictly parse one A/B answer per chat."""

    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("Generation batch size and token limit must be positive")
    if not eos_token_ids or any(not isinstance(value, int) for value in eos_token_ids):
        raise ValueError("Generation EOS inventory must contain integer token ids")
    if not chats:
        return []
    answers: list[dict[str, str | None]] = []
    for start in range(0, len(chats), batch_size):
        batch_chats = chats[start : start + batch_size]
        prompts = render_chat_texts(
            tokenizer,
            batch_chats,
            chat_template_kwargs=chat_template_kwargs,
        )
        inputs = _tokenize(tokenizer, prompts, device)
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
        if not isinstance(outputs, Tensor) or outputs.shape[0] != len(batch_chats):
            raise ValueError("Generation response count mismatch")
        generated = outputs[:, input_length:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != len(batch_chats):
            raise ValueError("Decoded response count mismatch")
        answers.extend(
            {"text": str(text), "parsed": parse_binary_letter(str(text))}
            for text in decoded
        )
    if len(answers) != len(chats):
        raise ValueError("Final generation response count mismatch")
    return answers


def _last_nonpadding_positions(attention_mask: Tensor) -> Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have batch and sequence dimensions")
    mask = attention_mask.to(dtype=torch.bool)
    if torch.any(~mask.any(dim=1)):
        raise ValueError("Every tokenized item must contain at least one token")
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return torch.where(mask, indices, torch.full_like(indices, -1)).max(dim=1).values


def extract_last_token_residuals(
    model: Any,
    tokenizer: Any,
    layers: Sequence[nn.Module],
    texts: list[str],
    *,
    batch_size: int,
    device: str,
) -> Tensor:
    """Capture post-block final-token states as ``(item, layer, hidden)``."""

    if not texts:
        raise ValueError("Residual extraction requires at least one text")
    if batch_size <= 0:
        raise ValueError("Residual extraction batch size must be positive")
    batches: list[Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = _tokenize(tokenizer, batch_texts, device)
        positions = _last_nonpadding_positions(inputs["attention_mask"])
        with (
            capture_last_token_layer_outputs(layers, positions) as capture,
            torch.inference_mode(),
        ):
            model(**inputs, use_cache=False, return_dict=True)
        batch = capture.stacked()
        if batch.shape[0] != len(batch_texts):
            raise RuntimeError("Residual batch count mismatch")
        batches.append(batch)
    result = torch.cat(batches, dim=0)
    if result.shape[0] != len(texts) or not torch.isfinite(result).all():
        raise RuntimeError("Residual extraction produced invalid output")
    return result
