# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sycophancy_steering.gsm8k_stage as gsm8k_stage
from sycophancy_steering.gsm8k import (
    flexible_extract,
    load_pinned_harness_contract,
    reference_answer,
    score_response,
    select_sample,
    strict_extract,
    wilson_interval,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_pinned_lm_eval_contract_is_loaded_and_used() -> None:
    study = json.loads(
        (REPOSITORY / "configs/studies/multimodel_v1.json").read_text(encoding="utf-8")
    )
    harness = load_pinned_harness_contract(study["sampled_gsm8k"])

    assert harness.version == "0.4.12"
    assert harness.task_name == "gsm8k_cot_zeroshot"
    assert (
        harness.task_yaml_path
        == "sycophancy_steering/contracts/gsm8k-cot-zeroshot-lm-eval-0.4.12.yaml"
    )
    score = score_response(
        "work\nThe answer is 1,234.",
        "solution\n#### 1,234",
        harness=harness,
    )
    assert score["strict_correct"] is True
    assert score["flexible_correct"] is True


def test_local_convenience_scoring_matches_pinned_harness_on_adversarial_cases() -> (
    None
):
    study = json.loads(
        (REPOSITORY / "configs/studies/multimodel_v1.json").read_text(encoding="utf-8")
    )
    harness = load_pinned_harness_contract(study["sampled_gsm8k"])
    cases = [
        ("The answer is 29.", "work\n#### 29"),
        ("The answer is 1,234.", "work\n#### 1,234"),
        ("reason $1,234 then $2,345", "work\n#### 2,345"),
        ("The answer is 5. Later 6", "work\n#### 6"),
        ("no numeric answer", "work\n#### 7"),
        ("The answer is -12.", "work\n#### -12"),
        ("The answer is 3.50.", "work\n#### 3.50"),
        ("first 10; final 11", "work\n#### 11"),
        ("The answer is 8!", "old #### 7\nnew #### 8"),
    ]
    for response, answer in cases:
        local = score_response(response, answer)
        pinned = score_response(response, answer, harness=harness)
        assert local["strict_correct"] == pinned["strict_correct"]
        assert local["flexible_correct"] == pinned["flexible_correct"]


def test_reference_answer_reads_canonical_hash_marker() -> None:
    assert reference_answer("work\n#### 1,234") == "1234"


def test_strict_extract_matches_harness_first_group_semantics() -> None:
    assert strict_extract("reasoning\nThe answer is 29.") == "29"
    assert strict_extract("The answer is 29!") == "29"
    assert strict_extract("The answer is 5. On reflection, The answer is 6.") == "5"
    assert strict_extract("#### 29") is None


def test_flexible_extract_uses_last_numeric_match() -> None:
    assert flexible_extract("First 12, then the answer is $29.00.") == "29.00"
    assert flexible_extract("no number") is None


def test_score_response_reports_both_harness_filters() -> None:
    score = score_response(
        "We calculate 20 + 4 + 5.\nThe answer is 29.",
        "The solution is 29.\n#### 29",
    )

    assert score == {
        "reference": "29",
        "strict_prediction": "29",
        "flexible_prediction": "29",
        "strict_correct": True,
        "flexible_correct": True,
    }


def test_hash_sample_is_deterministic_without_replacement() -> None:
    rows = [
        {"question": f"question {index}", "answer": f"work\n#### {index}"}
        for index in range(10)
    ]

    first = select_sample(rows, count=4)
    second = select_sample(rows, count=4)

    assert first == second
    assert len(first) == 4
    assert len({item["source_index"] for item in first}) == 4
    assert all(len(item["sample_sha256"]) == 64 for item in first)


def test_gsm8k_scores_are_recomputed_from_persisted_examples() -> None:
    scores = {
        "record_count": 1,
        "strict_correct_count": 1,
        "strict_sampled_accuracy": 1.0,
        "strict_wilson_95_ci": wilson_interval(correct=1, total=1, confidence=0.95),
        "flexible_correct_count": 1,
        "flexible_sampled_accuracy": 1.0,
        "flexible_wilson_95_ci": wilson_interval(correct=1, total=1, confidence=0.95),
        "examples": [
            {
                "doc_id": 0,
                "source_index": 7,
                "sample_sha256": "a" * 64,
                "question_sha256": "b" * 64,
                "prompt_sha256": "c" * 64,
                "response": "The answer is 42.",
                "reference": "42",
                "strict_prediction": "42",
                "flexible_prediction": "42",
                "strict_correct": True,
                "flexible_correct": True,
            }
        ],
    }
    gsm8k_stage._validate_scored_condition(scores, expected_records=1)

    scores["flexible_correct_count"] = 999
    with pytest.raises(ValueError, match="GSM8K scores"):
        gsm8k_stage._validate_scored_condition(scores, expected_records=1)


def test_wilson_interval_contains_observed_accuracy() -> None:
    interval = wilson_interval(correct=60, total=100, confidence=0.95)

    assert interval["lower"] <= 0.6 <= interval["upper"]
    assert interval["correct"] == 60
    assert interval["total"] == 100
