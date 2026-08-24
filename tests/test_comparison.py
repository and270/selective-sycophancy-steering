# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from sycophancy_steering.behavior import BehaviorRun
from sycophancy_steering.comparison import compare_behavior_runs

RECORDS = [
    {"id": "r1", "correct_option": "A", "source_dataset": "trivia_qa"},
    {"id": "r2", "correct_option": "B", "source_dataset": "truthful_qa"},
]
MODES = ("doubt", "authority", "wrong_suggest")


def run(neutral, pressure, controlled, metrics):  # type: ignore[no-untyped-def]
    return BehaviorRun(
        metrics=metrics,
        neutral_answers={
            key: {"text": value or "bad", "parsed": value}
            for key, value in neutral.items()
        },
        pressure_answers={
            key: {
                mode: {"text": value or "bad", "parsed": value}
                for mode, value in answers.items()
            }
            for key, answers in pressure.items()
        },
        natural_correction_answers={},
        controlled_correction_answers={
            key: {"text": value or "bad", "parsed": value}
            for key, value in controlled.items()
        },
        prompt_hashes={},
        hook_audit=None,
    )


def test_comparison_uses_paired_record_contributions() -> None:
    base = run(
        {"r1": "A", "r2": "B"},
        {"r1": dict.fromkeys(MODES, "A"), "r2": dict.fromkeys(MODES, "B")},
        {"r1": "A", "r2": "B"},
        {
            "eligible_ids": ["r1", "r2"],
            "ineligible_ids": [],
            "neutral_accuracy": 1.0,
            "pressure_error": 0.0,
            "pressure_invalid_rate": 0.0,
            "controlled_correction_acceptance_rate": 1.0,
            "controlled_correction_invalid_rate": 0.0,
            "natural_correct_suggestion_update_rate": None,
            "pressure_error_by_mode": dict.fromkeys(MODES, 0.0),
        },
    )
    condition = run(
        {"r1": "B", "r2": "B"},
        {
            "r1": {"doubt": "B", "authority": "A", "wrong_suggest": "A"},
            "r2": dict.fromkeys(MODES, "B"),
        },
        {"r1": "B", "r2": "B"},
        {
            "eligible_ids": ["r1", "r2"],
            "ineligible_ids": [],
            "neutral_accuracy": 0.5,
            "pressure_error": 1 / 6,
            "pressure_invalid_rate": 0.0,
            "controlled_correction_acceptance_rate": 0.5,
            "controlled_correction_invalid_rate": 0.0,
            "natural_correct_suggestion_update_rate": None,
            "pressure_error_by_mode": {
                "doubt": 0.5,
                "authority": 0.0,
                "wrong_suggest": 0.0,
            },
        },
    )

    comparison = compare_behavior_runs(
        RECORDS,
        base,
        condition,
        modes=MODES,
        bootstrap_iterations=500,
        bootstrap_seed=7,
        confidence=0.95,
    )

    assert comparison["deltas_condition_minus_base"]["neutral_accuracy"] == -0.5
    assert comparison["deltas_condition_minus_base"]["pressure_error"] == 1 / 6
    assert (
        comparison["deltas_condition_minus_base"][
            "controlled_correction_acceptance_rate"
        ]
        == -0.5
    )
    assert comparison["intervals"]["pressure_error"]["n_clusters"] == 2
    assert comparison["intervals"]["natural_correct_suggestion_update_rate"] is None
