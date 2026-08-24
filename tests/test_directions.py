# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pytest
import torch

from sycophancy_steering.directions import (
    binary_auroc,
    compute_completion_contrast,
    compute_observed_prompt_direction,
    projection_scores,
    random_direction_auroc_max_quantile,
    random_direction_auroc_quantiles,
)


def test_completion_contrast_is_raw_paired_mean_difference() -> None:
    caving = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[4.0, 0.0], [0.0, 8.0]],
        ]
    )
    resisting = torch.zeros_like(caving)

    direction = compute_completion_contrast(caving, resisting)

    torch.testing.assert_close(
        direction,
        torch.tensor([[3.0, 0.0], [0.0, 6.0]]),
    )


def test_observed_direction_is_caved_minus_resisted() -> None:
    residuals = torch.tensor(
        [
            [[1.0, 0.0]],
            [[3.0, 0.0]],
            [[1.0, 0.0]],
            [[5.0, 0.0]],
        ]
    )
    caved = torch.tensor([False, True, False, True])
    modes = ["doubt", "doubt", "authority", "authority"]

    direction, counts = compute_observed_prompt_direction(
        residuals,
        caved,
        modes,
        expected_modes=("doubt", "authority"),
        minimum_overall=2,
        minimum_per_mode=1,
    )

    torch.testing.assert_close(direction, torch.tensor([[3.0, 0.0]]))
    assert counts["overall"] == {"caved": 2, "resisted": 2}
    assert counts["doubt"] == {"caved": 1, "resisted": 1}
    assert counts["authority"] == {"caved": 1, "resisted": 1}


def test_missing_expected_mode_fails_class_count_gate() -> None:
    residuals = torch.tensor([[[0.0]], [[1.0]]])
    caved = torch.tensor([False, True])

    with pytest.raises(ValueError, match="per-mode behavior class"):
        compute_observed_prompt_direction(
            residuals,
            caved,
            ["doubt", "doubt"],
            expected_modes=("doubt", "authority"),
            minimum_overall=1,
            minimum_per_mode=1,
        )


def test_unknown_mode_fails_closed() -> None:
    residuals = torch.tensor([[[0.0]], [[1.0]]])

    with pytest.raises(ValueError, match="unexpected pressure mode"):
        compute_observed_prompt_direction(
            residuals,
            torch.tensor([False, True]),
            ["doubt", "unknown"],
            expected_modes=("doubt", "authority"),
            minimum_overall=1,
            minimum_per_mode=0,
        )


def test_binary_auroc_counts_ties_as_half() -> None:
    scores = torch.tensor([0.0, 1.0, 1.0, 2.0])
    labels = torch.tensor([False, True, False, True])

    assert binary_auroc(scores, labels) == 0.875


def test_projection_scores_align_items_and_layers() -> None:
    residuals = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    directions = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    scores = projection_scores(residuals, directions)

    torch.testing.assert_close(scores, torch.tensor([[1.0, 8.0], [5.0, 16.0]]))


def test_random_control_quantiles_are_seeded_per_layer() -> None:
    residuals = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
            [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]],
            [[0.0, 2.0, 3.0], [2.0, 0.0, 3.0]],
        ]
    )
    labels = torch.tensor([False, False, True, True])
    directions = torch.ones((2, 3))

    first = random_direction_auroc_quantiles(
        residuals,
        labels,
        directions,
        controls=20,
        seed=123,
        quantile=0.95,
    )
    second = random_direction_auroc_quantiles(
        residuals,
        labels,
        directions,
        controls=20,
        seed=123,
        quantile=0.95,
    )

    assert first == second
    assert len(first) == 2
    assert all(0.0 <= value <= 1.0 for value in first)

    maximum = random_direction_auroc_max_quantile(
        residuals,
        labels,
        directions,
        controls=20,
        seed=123,
        quantile=0.95,
    )
    assert 0.0 <= maximum <= 1.0
    assert maximum >= max(first)
