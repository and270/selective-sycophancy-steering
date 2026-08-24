# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from sycophancy_steering.probe import select_estimator_layers

MODES = ("doubt", "authority", "wrong_suggest")
POLICY = {
    "candidate_count": 2,
    "observed_gate": {
        "minimum_overall_auroc": 0.6,
        "minimum_evaluable_mode_auroc": 0.55,
        "maximum_unevaluable_modes": 1,
        "must_exceed_random_quantile": True,
        "must_exceed_random_max_quantile": True,
    },
    "completion_gate": {
        "source_datasets": ["trivia_qa", "truthful_qa"],
        "minimum_probe_records_per_source_option": 1,
        "minimum_overall_auroc": 0.65,
        "minimum_each_mode_auroc": 0.55,
        "minimum_each_correct_option_auroc": 0.55,
        "minimum_each_source_auroc": 0.55,
        "minimum_each_source_option_auroc": 0.55,
        "must_exceed_random_quantile": True,
        "must_exceed_random_max_quantile": True,
    },
}


def result(overall, by_mode, random):  # type: ignore[no-untyped-def]
    source_options = {
        f"{source}|{option}": overall
        for source in ("trivia_qa", "truthful_qa")
        for option in ("A", "B")
    }
    return {
        "overall_auroc": overall,
        "by_mode_auroc": by_mode,
        "by_correct_option_auroc": {"A": overall, "B": overall},
        "by_source_auroc": {
            "trivia_qa": overall,
            "truthful_qa": overall,
        },
        "by_source_option_auroc": source_options,
        "source_option_record_counts": {key: 10 for key in source_options},
        "random_control_q95": random,
        "random_control_max_over_layers_q95": max(random),
    }


def test_selects_top_two_observed_layers_after_all_gates() -> None:
    observed = result(
        [0.5, 0.72, 0.70, 0.8],
        {
            "doubt": [0.5, 0.61, 0.60, 0.54],
            "authority": [0.5, 0.62, 0.60, 0.70],
            "wrong_suggest": [0.5, 0.63, None, 0.70],
        },
        [0.55, 0.60, 0.59, 0.60],
    )

    selected = select_estimator_layers(
        {"observed_prompt_state": observed}, modes=MODES, policy=POLICY
    )

    assert selected["chosen_estimator"] == "observed_prompt_state"
    assert selected["chosen_layers"] == [1, 2]
    assert selected["by_estimator"]["observed_prompt_state"]["passing_layers"] == [1, 2]


def test_uses_completion_only_when_observed_has_no_candidate() -> None:
    observed = result(
        [0.59, 0.58],
        {mode: [0.7, 0.7] for mode in MODES},
        [0.5, 0.5],
    )
    completion = result(
        [0.7, 0.8],
        {mode: [0.6, 0.6] for mode in MODES},
        [0.55, 0.55],
    )

    selected = select_estimator_layers(
        {
            "observed_prompt_state": observed,
            "completion_contrast": completion,
        },
        modes=MODES,
        policy=POLICY,
    )

    assert selected["chosen_estimator"] == "completion_contrast"
    assert selected["chosen_layers"] == [1, 0]


def test_reports_no_chosen_layers_when_both_fail() -> None:
    failed = result(
        [0.5, 0.5],
        {mode: [0.5, 0.5] for mode in MODES},
        [0.6, 0.6],
    )

    selected = select_estimator_layers(
        {
            "observed_prompt_state": failed,
            "completion_contrast": failed,
        },
        modes=MODES,
        policy=POLICY,
    )

    assert selected["chosen_estimator"] is None
    assert selected["chosen_layers"] == []
