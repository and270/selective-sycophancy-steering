# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-closed transformer block discovery across supported architectures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True)
class ResolvedLayers:
    """The decoder blocks, text-model root, and exact dotted layer path."""

    path: str
    text_model: nn.Module
    layers: tuple[nn.Module, ...]


_CANDIDATE_PATHS = (
    # Verified for Qwen3.5ForConditionalGeneration and
    # Gemma4ForConditionalGeneration under Transformers 5.14.1.
    "model.language_model.layers",
    # Common text-only decoder layouts supported for portability.
    "model.layers",
    "language_model.layers",
    "model.decoder.layers",
    "transformer.h",
)


def _resolve_dotted(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def resolve_transformer_layers(
    model: nn.Module,
    *,
    expected_layers: int,
) -> ResolvedLayers:
    """Resolve decoder blocks and enforce the preregistered layer count."""

    if expected_layers <= 0:
        raise ValueError("expected_layers must be positive")

    candidates: list[tuple[str, nn.Module, Sequence[Any]]] = []
    for path in _CANDIDATE_PATHS:
        value = _resolve_dotted(model, path)
        parent = _resolve_dotted(model, path.rsplit(".", 1)[0])
        if isinstance(value, (nn.ModuleList, list, tuple)) and isinstance(
            parent, nn.Module
        ):
            candidates.append((path, parent, value))

    if not candidates:
        raise ValueError(
            "Could not resolve transformer decoder layers for this architecture"
        )

    matching = [
        (path, parent, layers)
        for path, parent, layers in candidates
        if len(layers) == expected_layers
    ]
    if not matching:
        found = ", ".join(
            f"{path}={len(layers)}" for path, _parent, layers in candidates
        )
        if len(candidates) == 1:
            actual = len(candidates[0][2])
            raise ValueError(
                f"Expected {expected_layers} transformer layers, found {actual} "
                f"at {candidates[0][0]}"
            )
        raise ValueError(
            f"Expected {expected_layers} transformer layers; candidates were {found}"
        )
    if len(matching) > 1:
        paths = ", ".join(path for path, _parent, _layers in matching)
        raise ValueError(f"Ambiguous transformer layer paths: {paths}")

    path, text_model, layers = matching[0]
    if not all(isinstance(layer, nn.Module) for layer in layers):
        raise TypeError(f"Every entry at {path} must be a torch.nn.Module")
    return ResolvedLayers(path=path, text_model=text_model, layers=tuple(layers))
