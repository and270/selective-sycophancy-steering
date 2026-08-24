# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from sycophancy_steering.data import record_contract_sha256
from sycophancy_steering.materialize import (
    _filter_question_families,
    materialize_study_data,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY.parent
    / "sycophancy-weight-surgery"
    / "data"
    / "raw"
    / "sycophancy-eval"
    / "datasets"
    / "answer.jsonl"
)
LOCK = REPOSITORY / "configs" / "data" / "multimodel_v1_data_lock.json"


def test_question_family_filter_is_symmetric_and_fail_closed() -> None:
    token_prefix = " ".join(f"token{index}" for index in range(18))
    pairs = [
        {
            "source_dataset": "truthful_qa",
            "question": (
                "What are the richest countries in South America by GDP per capita?"
            ),
        },
        {
            "source_dataset": "truthful_qa",
            "question": "What are the richest countries in the EU by GDP per capita?",
        },
        {
            "source_dataset": "trivia_qa",
            "question": f"{token_prefix} left",
        },
        {
            "source_dataset": "trivia_qa",
            "question": f"{token_prefix} right",
        },
        {"source_dataset": "trivia_qa", "question": "A unique safe question?"},
    ]

    filtered, stats = _filter_question_families(pairs)

    assert [record["question"] for record in filtered] == ["A unique safe question?"]
    assert stats["near_duplicate_edges"] == 2
    assert stats["near_duplicate_records"] == 4


@pytest.mark.skipif(not SOURCE.is_file(), reason="Pinned upstream source is not local")
def test_regenerates_frozen_split_hashes(tmp_path: Path) -> None:
    output = tmp_path / "materialized"

    manifest = materialize_study_data(SOURCE, LOCK, output)

    assert manifest["pairwise_disjoint"] is True
    assert manifest["splits"]["direction_fit"]["count"] == 300
    assert manifest["splits"]["direction_probe"]["count"] == 100
    assert manifest["splits"]["evaluation"]["count"] == 1310
    splits = {
        name: [
            json.loads(line)
            for line in (output / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for name in ("direction_fit", "direction_probe", "evaluation")
    }
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for name, records in splits.items():
        assert (
            record_contract_sha256(records)
            == lock["splits"][name]["record_contract_sha256"]
        )
    normalized_questions = {
        name: {
            (
                record["source_dataset"],
                " ".join(
                    unicodedata.normalize("NFKC", record["question"]).casefold().split()
                ),
            )
            for record in records
        }
        for name, records in splits.items()
    }
    names = list(normalized_questions)
    assert all(
        not normalized_questions[left] & normalized_questions[right]
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )
    excluded_ids = set(lock["derived_pairs"]["excluded_question_family_ids"])
    assert {"truthful_qa:1588", "truthful_qa:1032", "truthful_qa:1143"} <= (
        excluded_ids
    )
    assert not excluded_ids & {
        str(record["id"]) for records in splits.values() for record in records
    }
    assert Counter(
        (record["source_dataset"], record["correct_option"])
        for record in splits["direction_fit"]
    ) == {
        ("trivia_qa", "A"): 75,
        ("trivia_qa", "B"): 75,
        ("truthful_qa", "A"): 75,
        ("truthful_qa", "B"): 75,
    }
    assert Counter(
        (record["source_dataset"], record["correct_option"])
        for record in splits["direction_probe"]
    ) == {
        ("trivia_qa", "A"): 25,
        ("trivia_qa", "B"): 25,
        ("truthful_qa", "A"): 25,
        ("truthful_qa", "B"): 25,
    }
    evaluation = splits["evaluation"]
    assert Counter(record["source_dataset"] for record in evaluation) == {
        "trivia_qa": 732,
        "truthful_qa": 578,
    }
    assert Counter(
        (record["source_dataset"], record["correct_option"]) for record in evaluation
    ) == {
        ("trivia_qa", "A"): 366,
        ("trivia_qa", "B"): 366,
        ("truthful_qa", "A"): 289,
        ("truthful_qa", "B"): 289,
    }
    with pytest.raises(FileExistsError):
        materialize_study_data(SOURCE, LOCK, output)
