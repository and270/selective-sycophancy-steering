# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pytest

from sycophancy_steering.metrics import (
    compute_behavior_metrics,
    paired_cluster_bootstrap_mean_delta,
    paired_stratified_cluster_bootstrap_mean_delta,
)

RECORDS = [
    {"id": "r1", "correct_option": "A"},
    {"id": "r2", "correct_option": "B"},
    {"id": "r3", "correct_option": "A"},
]
MODES = ("doubt", "authority", "wrong_suggest")


def test_metrics_freeze_eligibility_and_preserve_integer_counts() -> None:
    metrics = compute_behavior_metrics(
        RECORDS,
        base_neutral={"r1": "A", "r2": "B", "r3": "B"},
        condition_neutral={"r1": "A", "r2": "A", "r3": "A"},
        pressure={
            "r1": {"doubt": "A", "authority": None, "wrong_suggest": "B"},
            "r2": {"doubt": "B", "authority": "B", "wrong_suggest": "B"},
        },
        natural_correction={"r3": "A"},
        controlled_correction={"r1": "A", "r2": None, "r3": "B"},
        modes=MODES,
    )

    assert metrics["eligible_ids"] == ["r1", "r2"]
    assert metrics["ineligible_ids"] == ["r3"]
    assert metrics["neutral_correct_count"] == 2
    assert metrics["neutral_denominator"] == 3
    assert metrics["neutral_accuracy"] == 2 / 3
    assert metrics["pressure_error_count"] == 2
    assert metrics["pressure_denominator"] == 6
    assert metrics["pressure_error"] == 2 / 6
    assert metrics["pressure_error_by_mode"] == {
        "doubt": 0.0,
        "authority": 0.5,
        "wrong_suggest": 0.5,
    }
    assert metrics["pressure_invalid_count"] == 1
    assert metrics["pressure_invalid_rate"] == 1 / 6
    assert metrics["natural_correct_suggestion_update_count"] == 1
    assert metrics["natural_correct_suggestion_denominator"] == 1
    assert metrics["natural_correct_suggestion_update_rate"] == 1.0
    assert metrics["controlled_correction_acceptance_count"] == 1
    assert metrics["controlled_correction_denominator"] == 3
    assert metrics["controlled_correction_acceptance_rate"] == 1 / 3
    assert metrics["controlled_correction_invalid_count"] == 1


def test_pressure_identity_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="Pressure response identities"):
        compute_behavior_metrics(
            RECORDS,
            base_neutral={"r1": "A", "r2": "B", "r3": "B"},
            condition_neutral={"r1": "A", "r2": "B", "r3": "A"},
            pressure={"r1": dict.fromkeys(MODES, "A")},
            natural_correction={"r3": "A"},
            controlled_correction={"r1": "A", "r2": "B", "r3": "A"},
            modes=MODES,
        )


def test_natural_correct_suggestion_update_is_none_without_ineligible_records() -> None:
    records = RECORDS[:2]
    pressure = {
        "r1": dict.fromkeys(MODES, "A"),
        "r2": dict.fromkeys(MODES, "B"),
    }
    metrics = compute_behavior_metrics(
        records,
        base_neutral={"r1": "A", "r2": "B"},
        condition_neutral={"r1": "A", "r2": "B"},
        pressure=pressure,
        natural_correction={},
        controlled_correction={"r1": "A", "r2": "B"},
        modes=MODES,
    )

    assert metrics["natural_correct_suggestion_denominator"] == 0
    assert metrics["natural_correct_suggestion_update_rate"] is None


def test_paired_cluster_bootstrap_is_deterministic_and_paired() -> None:
    base = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    condition = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0}

    first = paired_cluster_bootstrap_mean_delta(
        base, condition, iterations=1000, seed=123
    )
    second = paired_cluster_bootstrap_mean_delta(
        base, condition, iterations=1000, seed=123
    )

    assert first == second
    assert first["observed_condition_minus_base"] == -0.25
    assert first["lower"] <= -0.25 <= first["upper"]
    assert first["n_clusters"] == 4


def test_stratified_bootstrap_preserves_source_cluster_counts() -> None:
    base = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    condition = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    strata = {"a": "trivia", "b": "trivia", "c": "truthful", "d": "truthful"}

    first = paired_stratified_cluster_bootstrap_mean_delta(
        base, condition, strata, iterations=1000, seed=321
    )
    second = paired_stratified_cluster_bootstrap_mean_delta(
        base, condition, strata, iterations=1000, seed=321
    )

    assert first == second
    assert first["observed_condition_minus_base"] == 0.0
    assert first["stratum_cluster_counts"] == {"trivia": 2, "truthful": 2}
    assert first["n_clusters"] == 4
