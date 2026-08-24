# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sycophancy_steering.config import load_study_config

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = REPOSITORY / "configs" / "studies" / "multimodel_v1.json"


def test_loads_current_draft_for_engineering_work() -> None:
    config = load_study_config(STUDY, require_frozen=False)

    assert config["status"] == "draft"
    assert set(config["models"]) == {
        "qwen35_4b",
        "gemma4_e2b_it",
        "gemma4_e4b_it",
    }
    assert config["models"]["gemma4_e2b_it"]["expected_hidden_size"] == 1536
    assert config["models"]["gemma4_e4b_it"]["expected_hidden_size"] == 2560
    assert config["models"]["qwen35_4b"]["generation_eos_token_ids"] == [
        248044,
        248046,
    ]
    assert config["models"]["gemma4_e2b_it"]["generation_eos_token_ids"] == [
        1,
        106,
        50,
    ]
    assert config["runtime"]["maximum_binary_new_tokens"] == 1
    metrics = config["behavioral_evaluation"]["metrics"]
    assert "pressure_error" in metrics
    assert "overall_pressure_error" not in metrics


def test_scientific_access_rejects_draft() -> None:
    with pytest.raises(ValueError, match="not frozen"):
        load_study_config(STUDY, require_frozen=True)


def test_scientific_access_accepts_complete_frozen_copy(tmp_path: Path) -> None:
    payload = json.loads(STUDY.read_text(encoding="utf-8"))
    payload["status"] = "frozen"
    payload["scientific_outputs_allowed"] = True
    payload["freeze_pending"] = []
    for model in payload["models"].values():
        model["binary_generation_batch_size"] = 4
        model["residual_batch_size"] = 4
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_study_config(path, require_frozen=True)

    assert loaded["status"] == "frozen"


def test_invalid_model_revision_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(STUDY.read_text(encoding="utf-8"))
    payload["models"]["qwen35_4b"]["revision"] = "main"
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="revision"):
        load_study_config(path, require_frozen=False)
