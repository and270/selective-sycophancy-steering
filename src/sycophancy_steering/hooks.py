# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scoped, phase-aware residual-stream activation steering hooks.

A pre-hook on the verified text-model root reads cache and attention context for
each forward call. The selected decoder-block hook then applies the intervention
to the final non-padding prefill token or to every current cached-decode token.
Model weights are never changed.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn


@dataclass
class SteeringAudit:
    """Runtime evidence that the intervention path was exercised."""

    calls: int = 0
    prefill_calls: int = 0
    decode_calls: int = 0
    modified_batch_rows: int = 0
    modified_token_positions: int = 0


@dataclass
class _ForwardState:
    phase: Literal["prefill", "decode"] | None = None
    prefill_positions: Tensor | None = None
    current_sequence_length: int | None = None
    selected_layer_calls: int = 0

    def clear(self) -> None:
        self.phase = None
        self.prefill_positions = None
        self.current_sequence_length = None
        self.selected_layer_calls = 0


def _current_shape(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, int]:
    input_ids = kwargs.get("input_ids")
    if isinstance(input_ids, Tensor) and input_ids.ndim == 2:
        return int(input_ids.shape[0]), int(input_ids.shape[1])
    inputs_embeds = kwargs.get("inputs_embeds")
    if isinstance(inputs_embeds, Tensor) and inputs_embeds.ndim == 3:
        return int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
    for value in args:
        if isinstance(value, Tensor) and value.ndim in {2, 3}:
            return int(value.shape[0]), int(value.shape[1])
    raise RuntimeError("Could not determine current text-model batch/sequence shape")


def _is_prefill(kwargs: dict[str, Any]) -> bool:
    cache_position = kwargs.get("cache_position")
    if isinstance(cache_position, Tensor) and cache_position.numel():
        return int(cache_position.reshape(-1)[0].item()) == 0
    past = kwargs.get("past_key_values")
    if past is None:
        return True
    get_sequence_length = getattr(past, "get_seq_length", None)
    if callable(get_sequence_length):
        return int(get_sequence_length()) == 0
    return False


def _prefill_positions(
    attention_mask: Any,
    *,
    batch_size: int,
    sequence_length: int,
) -> Tensor:
    if not isinstance(attention_mask, Tensor):
        return torch.full((batch_size,), sequence_length - 1, dtype=torch.long)
    if attention_mask.ndim != 2 or attention_mask.shape[1] < sequence_length:
        raise ValueError("Prefill attention mask is incompatible with current sequence")
    current = attention_mask[:, -sequence_length:]
    if current.shape[0] == 1 and batch_size > 1:
        current = current.expand(batch_size, -1)
    if current.shape[0] != batch_size:
        raise ValueError("Prefill attention-mask batch does not match hidden batch")
    indices = torch.arange(sequence_length, device=current.device).unsqueeze(0)
    positions = torch.where(current.to(dtype=torch.bool), indices, -1).amax(dim=1)
    if (positions < 0).any().item():
        raise ValueError("A prefill row contains no non-padding token")
    return positions.to(dtype=torch.long)


def _validate_direction(direction: Tensor, alpha: float) -> None:
    if not isinstance(direction, Tensor):
        raise TypeError("direction must be a torch.Tensor")
    if direction.ndim != 1:
        raise ValueError("Steering direction must be a one-dimensional vector")
    if not torch.isfinite(direction).all().item():
        raise ValueError("Steering direction must contain only finite values")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")


def _hidden_from_output(output: Any) -> Tensor:
    source = output if isinstance(output, Tensor) else output[0]
    if not isinstance(source, Tensor):
        raise TypeError("Transformer layer output does not contain a tensor")
    return source


def _steer_hidden(
    hidden: Tensor,
    direction: Tensor,
    alpha: float,
    *,
    prefill_positions: Tensor | None,
) -> tuple[Tensor, int]:
    if hidden.ndim != 3:
        raise ValueError(
            "Transformer block hidden state must have shape [batch, sequence, width]"
        )
    if direction.ndim != 1:
        raise ValueError("Steering direction must be a one-dimensional vector")
    if hidden.shape[-1] != direction.shape[0]:
        raise ValueError(
            f"Hidden-state width {hidden.shape[-1]} does not match direction "
            f"width {direction.shape[0]}"
        )

    result = hidden.clone()
    delta = direction.to(device=hidden.device, dtype=hidden.dtype) * alpha
    if prefill_positions is None:
        result = result + delta.view(1, 1, -1)
        return result, int(hidden.shape[0] * hidden.shape[1])
    if prefill_positions.ndim != 1 or prefill_positions.shape[0] != hidden.shape[0]:
        raise ValueError("Prefill positions do not match the hidden-state batch")
    positions = prefill_positions.to(device=hidden.device)
    if (positions < 0).any().item() or (positions >= hidden.shape[1]).any().item():
        raise ValueError("Prefill position lies outside the block sequence")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    result[rows, positions, :] = result[rows, positions, :] + delta
    return result, int(hidden.shape[0])


def _replace_hidden(output: Any, hidden: Tensor) -> Any:
    if isinstance(output, Tensor):
        return hidden
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return (hidden, *output[1:])
    raise TypeError(
        "Transformer layer output must be a Tensor or a tuple whose first "
        "element is a Tensor"
    )


@contextmanager
def steer_transformer_layer(
    text_model: nn.Module,
    layer: nn.Module,
    direction: Tensor,
    *,
    alpha: float,
) -> Iterator[SteeringAudit]:
    """Temporarily steer one block using phase state from its text-model root."""

    if not isinstance(text_model, nn.Module) or not isinstance(layer, nn.Module):
        raise TypeError("text_model and layer must be torch.nn.Module instances")
    if text_model is layer:
        raise ValueError("text_model root and selected layer must be distinct modules")
    _validate_direction(direction, alpha)

    audit = SteeringAudit()
    state = _ForwardState()

    def root_pre_hook(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if state.phase is not None:
            raise RuntimeError("Overlapping text-model forward phases are unsupported")
        batch_size, sequence_length = _current_shape(args, kwargs)
        state.current_sequence_length = sequence_length
        if _is_prefill(kwargs):
            state.phase = "prefill"
            state.prefill_positions = _prefill_positions(
                kwargs.get("attention_mask"),
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        else:
            state.phase = "decode"
            state.prefill_positions = None

    def root_post_hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        _output: Any,
    ) -> None:
        state.clear()

    def layer_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if state.phase is None or state.current_sequence_length is None:
            raise RuntimeError(
                "Selected layer executed without text-model phase context"
            )
        state.selected_layer_calls += 1
        if state.selected_layer_calls != 1:
            raise RuntimeError(
                "Selected decoder layer executed more than once per forward"
            )
        source = _hidden_from_output(output)
        if int(source.shape[1]) != state.current_sequence_length:
            raise ValueError(
                "Block sequence length differs from text-model input length"
            )
        steered, modified_positions = _steer_hidden(
            source,
            direction,
            alpha,
            prefill_positions=(
                state.prefill_positions if state.phase == "prefill" else None
            ),
        )
        audit.calls += 1
        audit.modified_batch_rows += int(source.shape[0])
        audit.modified_token_positions += modified_positions
        if state.phase == "prefill":
            audit.prefill_calls += 1
        else:
            audit.decode_calls += 1
        return _replace_hidden(output, steered)

    root_pre_handle = text_model.register_forward_pre_hook(
        root_pre_hook, with_kwargs=True, prepend=True
    )
    root_post_handle = text_model.register_forward_hook(
        root_post_hook, with_kwargs=True, always_call=True
    )
    layer_handle = layer.register_forward_hook(layer_hook)
    try:
        yield audit
    finally:
        layer_handle.remove()
        root_post_handle.remove()
        root_pre_handle.remove()
        state.clear()


@contextmanager
def steer_trajectory_positions(
    layer: nn.Module,
    direction: Tensor,
    *,
    alpha: float,
    positions_mask: Tensor,
) -> Iterator[SteeringAudit]:
    """Replay deployment steering at all frozen causal source positions."""

    _validate_direction(direction, alpha)
    if positions_mask.ndim != 2 or positions_mask.dtype is not torch.bool:
        raise ValueError(
            "Trajectory positions mask must be a two-dimensional bool tensor"
        )
    if not torch.any(positions_mask):
        raise ValueError("Trajectory positions mask must select at least one position")
    audit = SteeringAudit()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = _hidden_from_output(output)
        if hidden.ndim != 3:
            raise ValueError(
                "Transformer block output must have batch, sequence, width"
            )
        if tuple(positions_mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError(
                "Trajectory positions mask differs from block output shape"
            )
        if hidden.shape[-1] != direction.numel():
            raise ValueError("Direction width does not match transformer hidden width")
        audit.calls += 1
        audit.prefill_calls += 1
        if audit.calls != 1:
            raise RuntimeError("Trajectory replay layer executed more than once")
        mask = positions_mask.to(device=hidden.device)
        delta = direction.to(device=hidden.device, dtype=hidden.dtype) * alpha
        steered = hidden.clone()
        steered[mask] = steered[mask] + delta
        audit.modified_batch_rows += int(mask.any(dim=1).sum().item())
        audit.modified_token_positions += int(mask.sum().item())
        return _replace_hidden(output, steered)

    handle = layer.register_forward_hook(hook)
    try:
        yield audit
    finally:
        handle.remove()
