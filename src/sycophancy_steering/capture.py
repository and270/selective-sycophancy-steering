# SPDX-License-Identifier: AGPL-3.0-or-later

"""Architecture-aware residual capture at the exact steering hook site."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def _hidden_from_output(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(
        "Transformer layer output must be a Tensor or a tuple beginning with one"
    )


@dataclass
class LayerCapture:
    """Captured final-token residuals and per-layer hook call counts."""

    outputs: list[Tensor | None]
    calls: list[int]

    def stacked(self) -> Tensor:
        """Return CPU float32 residuals shaped ``(item, layer, hidden)``."""

        if any(call_count != 1 for call_count in self.calls):
            raise RuntimeError("Every transformer layer must execute exactly once")
        if any(output is None for output in self.outputs):
            raise RuntimeError("A transformer layer produced no captured residual")
        tensors = [output for output in self.outputs if output is not None]
        reference_shape = tensors[0].shape
        if any(tensor.shape != reference_shape for tensor in tensors):
            raise RuntimeError("Captured transformer layer shapes do not match")
        result = torch.stack(tensors, dim=1)
        if not torch.isfinite(result).all():
            raise RuntimeError(
                "Captured transformer residuals contain nonfinite values"
            )
        return result


@contextmanager
def capture_last_token_layer_outputs(
    layers: Sequence[nn.Module],
    last_positions: Tensor,
) -> Iterator[LayerCapture]:
    """Capture the raw post-block state at each row's final non-padding token."""

    if not layers:
        raise ValueError("At least one transformer layer is required")
    if not all(isinstance(layer, nn.Module) for layer in layers):
        raise TypeError("Every transformer layer must be a torch.nn.Module")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("Transformer layer objects must be unique")
    if last_positions.ndim != 1 or last_positions.dtype == torch.bool:
        raise ValueError("last_positions must be a one-dimensional integer tensor")
    if last_positions.is_floating_point() or last_positions.is_complex():
        raise ValueError("last_positions must be a one-dimensional integer tensor")

    capture = LayerCapture(outputs=[None] * len(layers), calls=[0] * len(layers))
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_index: int):  # type: ignore[no-untyped-def]
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = _hidden_from_output(output)
            if hidden.ndim != 3:
                raise ValueError(
                    "Transformer layer hidden state must have shape "
                    "[batch, sequence, width]"
                )
            if hidden.shape[0] != last_positions.shape[0]:
                raise ValueError("Position count does not match hidden-state batch")
            positions = last_positions.to(device=hidden.device, dtype=torch.long)
            if torch.any(positions < 0) or torch.any(positions >= hidden.shape[1]):
                raise ValueError("A requested last-token position is out of range")
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[rows, positions, :]
            capture.calls[layer_index] += 1
            capture.outputs[layer_index] = selected.detach().to(
                dtype=torch.float32, device="cpu"
            )
            return None

        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
        yield capture
    finally:
        for handle in handles:
            handle.remove()
