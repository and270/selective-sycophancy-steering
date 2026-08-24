# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import torch
from torch import nn

from sycophancy_steering.hooks import (
    SteeringAudit,
    steer_trajectory_positions,
    steer_transformer_layer,
)


class TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Identity()

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        past_key_values: object | None = None,
    ) -> torch.Tensor:
        del attention_mask, cache_position, past_key_values
        return self.layer(hidden)


def test_prefill_modifies_last_nonpadding_position_per_row() -> None:
    root = TinyTextModel()
    direction = torch.tensor([1.0, -2.0])
    hidden = torch.zeros(2, 3, 2)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    with steer_transformer_layer(root, root.layer, direction, alpha=0.5) as audit:
        output = root(
            hidden,
            attention_mask=attention_mask,
            cache_position=torch.tensor([0, 1, 2]),
        )

    expected = hidden.clone()
    expected[0, 1, :] = torch.tensor([0.5, -1.0])
    expected[1, 2, :] = torch.tensor([0.5, -1.0])
    torch.testing.assert_close(output, expected)
    assert audit == SteeringAudit(
        calls=1,
        prefill_calls=1,
        decode_calls=0,
        modified_batch_rows=2,
        modified_token_positions=2,
    )


def test_one_token_prefill_is_not_misclassified_as_decode() -> None:
    root = TinyTextModel()
    hidden = torch.zeros(2, 1, 2)

    with steer_transformer_layer(root, root.layer, torch.ones(2), alpha=1.0) as audit:
        output = root(
            hidden,
            attention_mask=torch.ones(2, 1, dtype=torch.long),
            cache_position=torch.tensor([0]),
        )

    torch.testing.assert_close(output, torch.ones_like(hidden))
    assert audit.prefill_calls == 1
    assert audit.decode_calls == 0


def test_multitoken_cached_decode_modifies_every_current_position() -> None:
    root = TinyTextModel()
    direction = torch.tensor([1.0, -2.0])
    hidden = torch.zeros(2, 2, 2)

    with steer_transformer_layer(root, root.layer, direction, alpha=-1.0) as audit:
        output = root(
            hidden,
            attention_mask=torch.ones(2, 7, dtype=torch.long),
            cache_position=torch.tensor([5, 6]),
            past_key_values=object(),
        )

    expected = torch.tensor([[[-1.0, 2.0], [-1.0, 2.0]], [[-1.0, 2.0], [-1.0, 2.0]]])
    torch.testing.assert_close(output, expected)
    assert audit.decode_calls == 1
    assert audit.modified_batch_rows == 2
    assert audit.modified_token_positions == 4


def test_hook_is_removed_after_context_exit() -> None:
    root = TinyTextModel()
    hidden = torch.zeros(1, 1, 2)

    with steer_transformer_layer(root, root.layer, torch.ones(2), alpha=1.0):
        torch.testing.assert_close(
            root(
                hidden,
                attention_mask=torch.ones(1, 1, dtype=torch.long),
                cache_position=torch.tensor([0]),
            ),
            torch.ones_like(hidden),
        )

    torch.testing.assert_close(root(hidden), hidden)


def test_layer_call_without_root_phase_fails_closed() -> None:
    root = TinyTextModel()

    with steer_transformer_layer(root, root.layer, torch.ones(2), alpha=1.0):
        try:
            root.layer(torch.zeros(1, 1, 2))
        except RuntimeError as error:
            assert "phase" in str(error)
        else:
            raise AssertionError("Expected missing root phase to fail")


def test_full_trajectory_replay_modifies_exact_position_mask() -> None:
    layer = nn.Identity()
    hidden = torch.zeros(2, 4, 2)
    mask = torch.tensor([[False, True, True, False], [False, False, True, True]])
    with steer_trajectory_positions(
        layer, torch.tensor([1.0, -1.0]), alpha=0.5, positions_mask=mask
    ) as audit:
        output = layer(hidden)

    expected = hidden.clone()
    expected[mask] = torch.tensor([0.5, -0.5])
    assert torch.equal(output, expected)
    assert audit == SteeringAudit(
        calls=1,
        prefill_calls=1,
        decode_calls=0,
        modified_batch_rows=2,
        modified_token_positions=4,
    )


def test_invalid_direction_width_fails_during_inference() -> None:
    root = TinyTextModel()
    hidden = torch.zeros(1, 1, 3)

    with steer_transformer_layer(root, root.layer, torch.ones(2), alpha=1.0):
        try:
            root(
                hidden,
                attention_mask=torch.ones(1, 1, dtype=torch.long),
                cache_position=torch.tensor([0]),
            )
        except ValueError as error:
            assert "width" in str(error)
        else:
            raise AssertionError("Expected width mismatch to fail")


def test_nonfinite_alpha_fails_before_installing_hook() -> None:
    root = TinyTextModel()

    try:
        with steer_transformer_layer(
            root, root.layer, torch.ones(2), alpha=float("nan")
        ):
            pass
    except ValueError as error:
        assert "alpha" in str(error)
    else:
        raise AssertionError("Expected non-finite alpha to fail")
