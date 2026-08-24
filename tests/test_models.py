# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pytest
from torch import nn

from sycophancy_steering.models import resolve_transformer_layers


class FakeLanguageModel(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Identity() for _ in range(count))


class FakeBackbone(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.language_model = FakeLanguageModel(count)


class FakeConditionalGeneration(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.model = FakeBackbone(count)


def test_resolves_verified_multimodal_language_model_path() -> None:
    model = FakeConditionalGeneration(4)
    resolved = resolve_transformer_layers(model, expected_layers=4)

    assert resolved.path == "model.language_model.layers"
    assert resolved.text_model is model.model.language_model
    assert len(resolved.layers) == 4


def test_rejects_layer_count_drift() -> None:
    with pytest.raises(ValueError, match="Expected 5 transformer layers, found 4"):
        resolve_transformer_layers(FakeConditionalGeneration(4), expected_layers=5)


def test_rejects_unsupported_architecture() -> None:
    with pytest.raises(ValueError, match="Could not resolve"):
        resolve_transformer_layers(nn.Identity(), expected_layers=1)


def test_rejects_non_module_layer_entries() -> None:
    class BadLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [object(), object()]

    class BadBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = BadLanguageModel()

    class BadRoot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = BadBackbone()

    with pytest.raises(TypeError, match=r"torch\.nn\.Module"):
        resolve_transformer_layers(BadRoot(), expected_layers=2)
