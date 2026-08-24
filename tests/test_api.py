# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sycophancy_steering import resolve_steering_target, steer_model


class TinyLanguageModel(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Identity() for _ in range(count))

    def forward(
        self, hidden: torch.Tensor, *, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        del attention_mask
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyBackbone(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.language_model = TinyLanguageModel(count)


class TinyConditionalModel(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.model = TinyBackbone(count)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=count)
        )

    def forward(
        self, hidden: torch.Tensor, *, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model.language_model(hidden, attention_mask=attention_mask)


def test_resolves_target_from_transformers_style_config() -> None:
    model = TinyConditionalModel(2)

    target = resolve_steering_target(model, layer_index=1)

    assert target.layer_path == "model.language_model.layers"
    assert target.layer_index == 1
    assert target.layer_count == 2


def test_steers_only_final_prefill_position_without_changing_weights() -> None:
    model = TinyConditionalModel(2)
    hidden = torch.zeros((1, 3, 4))
    attention_mask = torch.ones((1, 3), dtype=torch.long)
    direction = torch.ones(4)

    with steer_model(model, direction, layer_index=0, alpha=2.0) as audit:
        output = model(hidden, attention_mask=attention_mask)

    assert torch.equal(output[:, :2], hidden[:, :2])
    assert torch.equal(output[:, 2], torch.full((1, 4), 2.0))
    assert torch.equal(hidden, torch.zeros_like(hidden))
    assert audit.calls == 1
    assert audit.prefill_calls == 1
    assert audit.modified_token_positions == 1


def test_rejects_out_of_range_layer() -> None:
    with pytest.raises(ValueError, match="outside the resolved"):
        resolve_steering_target(TinyConditionalModel(2), layer_index=2)
