# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import torch

from sycophancy_steering.fit_probe import (
    DirectionObservations,
    build_direction_chats,
    completion_fit_status,
    completion_probe_result,
    observed_probe_result,
    observed_valid_data,
)
from sycophancy_steering.probe import select_estimator_layers

CONTRACT = {
    "baseline_template": "{question}\nA) {option_a}\nB) {option_b}",
    "pressure_modes": ["doubt", "authority", "wrong_suggest"],
    "pressure_templates": {
        "doubt": ["D", "D", "D"],
        "authority": ["X", "X", "X"],
        "wrong_suggest": ["W {wrong_option}", "W {wrong_option}", "W {wrong_option}"],
        "correct_suggest": [
            "C {correct_option}",
            "C {correct_option}",
            "C {correct_option}",
        ],
    },
}
RECORDS = [
    {
        "id": "r1",
        "source_dataset": "trivia_qa",
        "question": "Q1",
        "options": {"A": "yes", "B": "no"},
        "correct_option": "A",
        "wrong_option": "B",
        "pressure_variant": {"doubt": 0, "authority": 0, "wrong_suggest": 0},
    },
    {
        "id": "r2",
        "source_dataset": "truthful_qa",
        "question": "Q2",
        "options": {"A": "yes", "B": "no"},
        "correct_option": "B",
        "wrong_option": "A",
        "pressure_variant": {"doubt": 0, "authority": 0, "wrong_suggest": 0},
    },
]


def test_build_direction_chats_uses_only_base_eligible_records() -> None:
    baseline = {
        "r1": {"text": "A", "parsed": "A"},
        "r2": {"text": "A", "parsed": "A"},
    }

    descriptors, chats = build_direction_chats(RECORDS, baseline, CONTRACT)

    assert len(descriptors) == 3
    assert len(chats) == 3
    assert {item["record_id"] for item in descriptors} == {"r1"}
    assert {item["mode"] for item in descriptors} == set(CONTRACT["pressure_modes"])
    assert all(item["caving_completion"] == "B" for item in descriptors)
    assert all(item["resisting_completion"] == "A" for item in descriptors)


def test_observed_valid_data_excludes_invalid_and_labels_caving() -> None:
    observations = DirectionObservations(
        baseline_answers={},
        descriptors=[
            {
                "record_id": "a",
                "mode": "doubt",
                "caving_completion": "B",
                "resisting_completion": "A",
            },
            {
                "record_id": "b",
                "mode": "authority",
                "caving_completion": "A",
                "resisting_completion": "B",
            },
            {
                "record_id": "c",
                "mode": "wrong_suggest",
                "caving_completion": "B",
                "resisting_completion": "A",
            },
        ],
        followup_answers=[
            {"text": "B", "parsed": "B"},
            {"text": "B", "parsed": "B"},
            {"text": "bad", "parsed": None},
        ],
        prompt_residuals=torch.tensor([[[1.0]], [[2.0]], [[3.0]]]),
        caving_residuals=torch.empty(0),
        resisting_residuals=torch.empty(0),
    )

    residuals, labels, modes = observed_valid_data(observations)

    torch.testing.assert_close(residuals, torch.tensor([[[1.0]], [[2.0]]]))
    torch.testing.assert_close(labels, torch.tensor([True, False]))
    assert modes == ["doubt", "authority"]


def test_observed_probe_reports_all_modes_and_layer_scores() -> None:
    descriptors = []
    answers = []
    values = []
    for mode in CONTRACT["pressure_modes"]:
        for caved, value in [(False, 0.0), (True, 2.0)]:
            descriptors.append(
                {
                    "record_id": f"{mode}-{caved}",
                    "mode": mode,
                    "caving_completion": "B",
                    "resisting_completion": "A",
                }
            )
            answers.append(
                {"text": "B" if caved else "A", "parsed": "B" if caved else "A"}
            )
            values.append([[value, 0.0], [0.0, value]])
    observations = DirectionObservations(
        baseline_answers={},
        descriptors=descriptors,
        followup_answers=answers,
        prompt_residuals=torch.tensor(values),
        caving_residuals=torch.empty(0),
        resisting_residuals=torch.empty(0),
    )
    direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    probe = observed_probe_result(
        observations,
        direction,
        modes=tuple(CONTRACT["pressure_modes"]),
        random_controls=5,
        random_seed=11,
    )

    assert probe["overall_auroc"] == [1.0, 1.0]
    assert set(probe["by_mode_auroc"]) == set(CONTRACT["pressure_modes"])
    assert all(values == [1.0, 1.0] for values in probe["by_mode_auroc"].values())
    assert len(probe["random_control_q95"]) == 2


def test_completion_gate_rejects_answer_token_confound() -> None:
    descriptors = [
        {
            "record_id": f"{source}-{index}",
            "source_dataset": source,
            "correct_option": "A",
            "mode": "doubt",
            "caving_completion": "B",
            "resisting_completion": "A",
        }
        for source in ("trivia_qa", "truthful_qa")
        for index in range(4)
    ]
    observations = DirectionObservations(
        baseline_answers={},
        descriptors=descriptors,
        followup_answers=[],
        prompt_residuals=torch.empty(0),
        caving_residuals=torch.ones((len(descriptors), 1, 1)),
        resisting_residuals=torch.zeros((len(descriptors), 1, 1)),
    )
    fit_status = completion_fit_status(
        observations,
        source_datasets=("trivia_qa", "truthful_qa"),
        minimum_records_per_source_option=1,
    )
    result = completion_probe_result(
        observations,
        torch.ones((1, 1)),
        modes=("doubt",),
        source_datasets=("trivia_qa", "truthful_qa"),
        random_controls=2,
        random_seed=0,
    )
    policy = {
        "candidate_count": 1,
        "completion_gate": {
            "source_datasets": ["trivia_qa", "truthful_qa"],
            "minimum_overall_auroc": 0.5,
            "minimum_each_mode_auroc": 0.5,
            "minimum_each_correct_option_auroc": 0.5,
            "minimum_each_source_auroc": 0.5,
            "minimum_each_source_option_auroc": 0.5,
            "minimum_probe_records_per_source_option": 1,
            "must_exceed_random_quantile": False,
            "must_exceed_random_max_quantile": False,
        },
    }

    selection = select_estimator_layers(
        {"completion_contrast": result}, modes=("doubt",), policy=policy
    )

    assert fit_status["eligible"] is False
    assert fit_status["fit_source_option_record_counts"]["trivia_qa|B"] == 0
    assert result["source_option_record_counts"]["trivia_qa|B"] == 0
    assert selection["chosen_estimator"] is None
    assert selection["by_estimator"]["completion_contrast"]["passing_layers"] == []
