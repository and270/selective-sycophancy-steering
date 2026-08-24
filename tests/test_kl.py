# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pytest
import torch

import sycophancy_steering.kl_stage as kl_stage
from sycophancy_steering.kl import (
    distribution_metrics_from_logits,
    select_neutral_contexts,
)


def test_identical_logits_have_zero_kl_js_and_full_agreement() -> None:
    logits = torch.tensor([[1.0, 2.0], [3.0, -1.0]])

    metrics = distribution_metrics_from_logits(logits, logits.clone())

    torch.testing.assert_close(
        metrics["forward_kl"], torch.zeros(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["jensen_shannon"], torch.zeros(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["top1_agreement"], torch.ones(2, dtype=torch.float64)
    )


def test_forward_kl_is_directional_and_js_is_symmetric() -> None:
    base = torch.tensor([[2.0, 0.0]])
    condition = torch.tensor([[0.0, 2.0]])

    forward = distribution_metrics_from_logits(base, condition)
    reverse = distribution_metrics_from_logits(condition, base)

    assert forward["forward_kl"].item() > 0
    torch.testing.assert_close(forward["jensen_shannon"], reverse["jensen_shannon"])
    assert forward["top1_agreement"].item() == 0.0


def test_tiny_full_vocabulary_shift_is_not_clamped_to_zero() -> None:
    base = torch.zeros(2, 262_144, dtype=torch.float32)
    shift = torch.linspace(-1.0e-3, 1.0e-3, 262_144, dtype=torch.float32)
    condition = torch.stack([shift, -shift])

    metrics = distribution_metrics_from_logits(base, condition)

    assert torch.all(metrics["forward_kl"] > 0)
    assert torch.all(metrics["jensen_shannon"] > 0)
    assert metrics["forward_kl"].dtype == torch.float64


def test_exploratory_kl_trial_summary_uses_current_bootstrap_schema() -> None:
    interval = {
        "mean": 0.001,
        "lower": 0.0005,
        "upper": 0.0015,
        "confidence": 0.95,
        "iterations": 10_000,
        "seed": 20260805,
        "n_contexts": 64,
    }
    trial = {
        "alpha": -2.0,
        "condition": {
            "prompt_macro": {
                "forward_kl_nats": {"mean_bootstrap_95_ci": interval},
            },
            "token_micro": {
                "forward_kl_nats": {"mean": 0.0011},
                "jensen_shannon_nats": {"mean": 0.0003},
                "top1_agreement": {"mean": 0.98},
            },
        },
    }

    assert kl_stage.exploratory_kl_trial_summary(trial) == {
        "alpha": -2.0,
        "forward_kl_nats": 0.0011,
        "forward_kl_95_ci": interval,
        "js_nats": 0.0003,
        "top1_agreement": 0.98,
    }


def test_kl_trial_statistics_are_recomputed_from_per_token_primitives() -> None:
    kl_values = [0.1, 0.2]
    js_values = [0.01, 0.02]
    agreement = [True, False]
    trial = {
        "alpha": -2.0,
        "maximum_absolute_logit_difference": 0.5,
        "token_count": 2,
        "hook_audit": {
            "calls": 1,
            "prefill_calls": 1,
            "decode_calls": 0,
            "modified_batch_rows": 1,
            "modified_token_positions": 2,
        },
        "per_context": [
            {
                "row_index": 7,
                "context_sha256": "a" * 64,
                "token_count": 2,
                "forward_kl_nats_by_token": kl_values,
                "jensen_shannon_nats_by_token": js_values,
                "top1_agreement_by_token": agreement,
                "forward_kl_nats_mean": 0.15,
                "jensen_shannon_nats_mean": 0.015,
                "top1_agreement_mean": 0.5,
            }
        ],
        "prompt_macro": {
            "forward_kl_nats": kl_stage._macro_summary(
                [0.15], iterations=10_000, seed=20260805, confidence=0.95
            ),
            "jensen_shannon_nats": kl_stage._macro_summary(
                [0.015], iterations=10_000, seed=20260806, confidence=0.95
            ),
            "top1_agreement": kl_stage._macro_summary(
                [0.5], iterations=10_000, seed=20260807, confidence=0.95
            ),
        },
        "token_micro": {
            "forward_kl_nats": kl_stage._micro_summary(kl_values),
            "jensen_shannon_nats": kl_stage._micro_summary(js_values),
            "top1_agreement": kl_stage._micro_summary([1.0, 0.0]),
        },
    }
    contexts = [{"row_index": 7, "sha256": "a" * 64}]
    kl_stage._validate_kl_trial_statistics(
        trial, contexts=contexts, zero_alpha_tolerance=0.0
    )

    trial["per_context"][0]["forward_kl_nats_by_token"][0] = -123.0
    with pytest.raises(ValueError, match="KL trial"):
        kl_stage._validate_kl_trial_statistics(
            trial, contexts=contexts, zero_alpha_tolerance=0.0
        )


def test_context_selection_is_hash_deterministic_and_excludes_headings() -> None:
    long = "word " * 80
    rows = [
        "",
        " = Heading = ",
        f"alpha {long}",
        f"beta {long}",
        f"gamma {long}",
    ]

    first = select_neutral_contexts(rows, count=2, minimum_characters=256)
    second = select_neutral_contexts(rows, count=2, minimum_characters=256)

    assert first == second
    assert len(first) == 2
    assert all(item["row_index"] in {2, 3, 4} for item in first)
    assert all(len(item["sha256"]) == 64 for item in first)
