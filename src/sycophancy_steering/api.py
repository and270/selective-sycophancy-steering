# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small public API for steering an already-loaded transformer model.

The strict study loaders and artifact contracts live in the stage modules.  This
module intentionally does less: it resolves one decoder block on a caller-owned
``torch.nn.Module`` and installs the same temporary, cache-aware runtime hook
used by the experiments.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

from .hooks import SteeringAudit, steer_transformer_layer
from .models import resolve_transformer_layers


@dataclass(frozen=True)
class SteeringTarget:
    """Resolved decoder block used by :func:`steer_model`."""

    layer_path: str
    layer_index: int
    layer_count: int
    text_model: nn.Module = field(repr=False)
    layer: nn.Module = field(repr=False)


def _configured_layer_count(model: nn.Module) -> int:
    config: Any = getattr(model, "config", None)
    candidates = (getattr(config, "text_config", None), config)
    for candidate in candidates:
        for attribute in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    raise ValueError(
        "Could not infer the decoder layer count from model.config; pass "
        "expected_layers explicitly"
    )


def resolve_steering_target(
    model: nn.Module,
    *,
    layer_index: int,
    expected_layers: int | None = None,
) -> SteeringTarget:
    """Resolve one zero-based decoder block on an already-loaded model.

    ``expected_layers`` is normally inferred from the Hugging Face model config.
    Supplying it explicitly provides an additional architecture check and also
    supports custom ``nn.Module`` wrappers without a Transformers-style config.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(layer_index, int) or isinstance(layer_index, bool):
        raise TypeError("layer_index must be an integer")
    if layer_index < 0:
        raise ValueError("layer_index must be zero or greater")
    layer_count = (
        _configured_layer_count(model) if expected_layers is None else expected_layers
    )
    resolved = resolve_transformer_layers(model, expected_layers=layer_count)
    if layer_index >= len(resolved.layers):
        raise ValueError(
            f"layer_index {layer_index} is outside the resolved "
            f"{len(resolved.layers)}-layer model"
        )
    return SteeringTarget(
        layer_path=resolved.path,
        layer_index=layer_index,
        layer_count=len(resolved.layers),
        text_model=resolved.text_model,
        layer=resolved.layers[layer_index],
    )


@contextmanager
def steer_model(
    model: nn.Module,
    direction: Tensor,
    *,
    layer_index: int,
    alpha: float,
    expected_layers: int | None = None,
) -> Iterator[SteeringAudit]:
    """Temporarily steer one layer of an already-loaded model.

    The model remains owned by the caller and its weights are never modified.
    Use this as a context manager around ``model.generate(...)`` or a forward
    pass.  The yielded audit records whether prefill/decode calls were actually
    intercepted.
    """

    target = resolve_steering_target(
        model,
        layer_index=layer_index,
        expected_layers=expected_layers,
    )
    with steer_transformer_layer(
        target.text_model,
        target.layer,
        direction,
        alpha=alpha,
    ) as audit:
        yield audit
