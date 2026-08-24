# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived in part from Heretic's activation-steering research utilities.

"""Direction estimators, linear probes, and null controls."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_nonzero_rows(direction: Tensor, *, label: str) -> None:
    norms = torch.linalg.vector_norm(direction, dim=1)
    if not torch.isfinite(direction).all() or torch.any(norms <= 0):
        raise ValueError(f"{label} must be finite and nonzero at every layer")


def compute_completion_contrast(caving: Tensor, resisting: Tensor) -> Tensor:
    """Return the raw paired caving-minus-resisting mean at every layer."""

    if caving.shape != resisting.shape or caving.ndim != 3 or caving.shape[0] == 0:
        raise ValueError(
            "Completion tensors must share non-empty (item, layer, hidden) shape"
        )
    if not torch.isfinite(caving).all() or not torch.isfinite(resisting).all():
        raise ValueError("Completion tensors must contain only finite values")
    direction = (caving.to(torch.float64) - resisting.to(torch.float64)).mean(dim=0)
    direction = direction.to(torch.float32)
    _validate_nonzero_rows(direction, label="Completion contrast")
    return direction


def compute_observed_prompt_direction(
    residuals: Tensor,
    caved: Tensor,
    modes: list[str],
    *,
    expected_modes: tuple[str, ...],
    minimum_overall: int,
    minimum_per_mode: int,
) -> tuple[Tensor, dict[str, dict[str, int]]]:
    """Return pooled caved-minus-resisted prompt-state directions."""

    if residuals.ndim != 3 or residuals.shape[0] == 0:
        raise ValueError(
            "Prompt residuals must have non-empty (item, layer, hidden) shape"
        )
    if caved.ndim != 1 or caved.shape[0] != residuals.shape[0]:
        raise ValueError("Observed labels must align with prompt residuals")
    if len(modes) != residuals.shape[0]:
        raise ValueError("Observed modes must align with prompt residuals")
    if not expected_modes or len(set(expected_modes)) != len(expected_modes):
        raise ValueError("expected_modes must be unique and non-empty")
    unexpected = set(modes) - set(expected_modes)
    if unexpected:
        raise ValueError(
            f"Observed data contains unexpected pressure mode: {unexpected}"
        )
    if minimum_overall < 1 or minimum_per_mode < 0:
        raise ValueError("Observed class-count thresholds are invalid")
    if not torch.isfinite(residuals).all():
        raise ValueError("Prompt residuals must contain only finite values")

    labels = caved.to(dtype=torch.bool, device=residuals.device)
    counts: dict[str, dict[str, int]] = {
        "overall": {
            "caved": int(labels.sum().item()),
            "resisted": int((~labels).sum().item()),
        }
    }
    for mode in expected_modes:
        mask = torch.tensor(
            [item_mode == mode for item_mode in modes],
            dtype=torch.bool,
            device=residuals.device,
        )
        counts[mode] = {
            "caved": int((labels & mask).sum().item()),
            "resisted": int(((~labels) & mask).sum().item()),
        }

    if min(counts["overall"].values()) < minimum_overall:
        raise ValueError("Observed estimator lacks an overall behavior class")
    if any(min(counts[mode].values()) < minimum_per_mode for mode in expected_modes):
        raise ValueError("Observed estimator lacks a per-mode behavior class")

    direction = residuals[labels].to(torch.float64).mean(dim=0) - residuals[~labels].to(
        torch.float64
    ).mean(dim=0)
    direction = direction.to(torch.float32)
    _validate_nonzero_rows(direction, label="Observed prompt direction")
    return direction, counts


def projection_scores(residuals: Tensor, directions: Tensor) -> Tensor:
    """Project each item onto the corresponding direction at every layer."""

    if residuals.ndim != 3 or directions.ndim != 2:
        raise ValueError(
            "Residuals and directions require (item, layer, hidden) and "
            "(layer, hidden) shapes"
        )
    if residuals.shape[1:] != directions.shape:
        raise ValueError("Residual and direction dimensions do not match")
    if not torch.isfinite(residuals).all() or not torch.isfinite(directions).all():
        raise ValueError("Residuals and directions must be finite")
    return torch.einsum(
        "nlh,lh->nl",
        residuals.to(torch.float64),
        directions.to(torch.float64),
    ).to(torch.float32)


def binary_auroc(scores: Tensor, positive: Tensor) -> float:
    """Compute exact binary AUROC, assigning one half to tied pairs."""

    if scores.ndim != 1 or positive.ndim != 1 or scores.shape != positive.shape:
        raise ValueError("AUROC scores and labels must be aligned vectors")
    if not torch.isfinite(scores).all():
        raise ValueError("AUROC scores must be finite")
    positive = positive.to(dtype=torch.bool, device=scores.device)
    positive_scores = scores[positive]
    negative_scores = scores[~positive]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        raise ValueError("AUROC requires both classes")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    wins = (comparisons > 0).to(torch.float64).sum()
    ties = (comparisons == 0).to(torch.float64).sum()
    return float(((wins + 0.5 * ties) / comparisons.numel()).item())


def _random_direction_auroc_matrix(
    residuals: Tensor,
    positive: Tensor,
    reference_directions: Tensor,
    *,
    controls: int,
    seed: int,
) -> Tensor:
    if residuals.ndim != 3 or reference_directions.shape != residuals.shape[1:]:
        raise ValueError("Random-control residual and direction shapes do not match")
    if positive.ndim != 1 or positive.shape[0] != residuals.shape[0]:
        raise ValueError("Random-control labels do not align")
    if controls <= 0:
        raise ValueError("Random-control count must be positive")
    residuals = residuals.to(dtype=torch.float32, device="cpu")
    positive = positive.to(dtype=torch.bool, device="cpu")
    reference_directions = reference_directions.to(dtype=torch.float32, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.empty(
        (controls, residuals.shape[1]), dtype=torch.float64, device="cpu"
    )
    for layer_index in range(residuals.shape[1]):
        norm = torch.linalg.vector_norm(reference_directions[layer_index])
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Reference direction must be finite and nonzero")
        for control_index in range(controls):
            random_direction = torch.randn(
                residuals.shape[2], generator=generator, dtype=torch.float32
            )
            random_norm = torch.linalg.vector_norm(random_direction)
            if not torch.isfinite(random_norm) or random_norm <= 0:
                raise RuntimeError("Generated an invalid random control")
            random_direction = random_direction * (norm / random_norm)
            scores = residuals[:, layer_index, :] @ random_direction
            matrix[control_index, layer_index] = binary_auroc(scores, positive)
    return matrix


def random_direction_auroc_thresholds(
    residuals: Tensor,
    positive: Tensor,
    reference_directions: Tensor,
    *,
    controls: int,
    seed: int,
    quantile: float,
) -> tuple[list[float], float]:
    """Return per-layer and max-over-layers norm-matched null thresholds."""

    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("Random-control quantile is invalid")
    matrix = _random_direction_auroc_matrix(
        residuals,
        positive,
        reference_directions,
        controls=controls,
        seed=seed,
    )
    per_layer = [
        float(value) for value in torch.quantile(matrix, quantile, dim=0).tolist()
    ]
    maximum = float(torch.quantile(matrix.amax(dim=1), quantile).item())
    return per_layer, maximum


def random_direction_auroc_quantiles(
    residuals: Tensor,
    positive: Tensor,
    reference_directions: Tensor,
    *,
    controls: int,
    seed: int,
    quantile: float,
) -> list[float]:
    """Compute per-layer AUROC quantiles for norm-matched Gaussian axes."""

    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("Random-control quantile is invalid")
    matrix = _random_direction_auroc_matrix(
        residuals,
        positive,
        reference_directions,
        controls=controls,
        seed=seed,
    )
    return [float(value) for value in torch.quantile(matrix, quantile, dim=0).tolist()]


def random_direction_auroc_max_quantile(
    residuals: Tensor,
    positive: Tensor,
    reference_directions: Tensor,
    *,
    controls: int,
    seed: int,
    quantile: float,
) -> float:
    """Family-wise q-quantile of the maximum random AUROC over all layers."""

    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("Random-control quantile is invalid")
    matrix = _random_direction_auroc_matrix(
        residuals,
        positive,
        reference_directions,
        controls=controls,
        seed=seed,
    )
    return float(torch.quantile(matrix.amax(dim=1), quantile).item())
