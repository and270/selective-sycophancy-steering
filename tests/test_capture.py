# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pytest
import torch
from torch import nn

from sycophancy_steering.capture import capture_last_token_layer_outputs


class AddLayer(nn.Module):
    def __init__(self, amount: float, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.amount = amount
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor):  # type: ignore[no-untyped-def]
        output = hidden + self.amount
        return (output, "cache") if self.tuple_output else output


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AddLayer(1.0), AddLayer(2.0, tuple_output=True)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            output = layer(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
        return hidden * 10.0  # Deliberate post-layer final normalization surrogate.


def test_captures_raw_post_block_outputs_not_final_model_output() -> None:
    model = TinyDecoder()
    hidden = torch.zeros(2, 3, 2)
    last_positions = torch.tensor([1, 2])

    with capture_last_token_layer_outputs(model.layers, last_positions) as capture:
        final = model(hidden)
    residuals = capture.stacked()

    assert tuple(residuals.shape) == (2, 2, 2)
    torch.testing.assert_close(residuals[:, 0, :], torch.ones(2, 2))
    torch.testing.assert_close(residuals[:, 1, :], torch.full((2, 2), 3.0))
    torch.testing.assert_close(final[:, -1, :], torch.full((2, 2), 30.0))


def test_uses_per_row_last_nonpadding_position() -> None:
    layer = AddLayer(5.0)
    hidden = torch.tensor(
        [
            [[1.0], [2.0], [99.0]],
            [[3.0], [4.0], [5.0]],
        ]
    )

    with capture_last_token_layer_outputs([layer], torch.tensor([1, 2])) as capture:
        layer(hidden)

    torch.testing.assert_close(capture.stacked(), torch.tensor([[[7.0]], [[10.0]]]))


def test_hook_is_removed_after_capture_context() -> None:
    layer = AddLayer(1.0)

    with capture_last_token_layer_outputs([layer], torch.tensor([0])) as capture:
        layer(torch.zeros(1, 1, 1))
    assert capture.calls == [1]

    layer(torch.zeros(1, 1, 1))
    assert capture.calls == [1]


def test_missing_layer_execution_fails_when_stacking() -> None:
    layers = [AddLayer(1.0), AddLayer(2.0)]

    with capture_last_token_layer_outputs(layers, torch.tensor([0])) as capture:
        layers[0](torch.zeros(1, 1, 1))

    with pytest.raises(RuntimeError, match="exactly once"):
        capture.stacked()


def test_invalid_position_fails_at_capture() -> None:
    layer = AddLayer(1.0)

    with (
        capture_last_token_layer_outputs([layer], torch.tensor([2])),
        pytest.raises(ValueError, match="position"),
    ):
        layer(torch.zeros(1, 2, 1))
