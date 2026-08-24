# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sycophancy_steering.exploratory_identity import (
    exploratory_source_identity,
    load_verified_exploratory_fit,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object], Path]:
    repository = tmp_path / "repo"
    fit_dir = tmp_path / "fit"
    repository.mkdir()
    fit_dir.mkdir()
    lock_path = repository / "lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    study_path = repository / "study.json"
    study: dict[str, object] = {"data": {"lock": "lock.json"}}
    study_path.write_text(json.dumps(study) + "\n", encoding="utf-8")
    direction_path = fit_dir / "directions.safetensors"
    direction_path.write_bytes(b"direction")
    fit = {
        "model_key": "model_a",
        "run_kind": "engineering_smoke",
        "scientific_outputs_allowed": False,
        "study_sha256": _digest(study_path),
        "data_lock_sha256": _digest(lock_path),
        "direction_artifact": {
            "path": "directions.safetensors",
            "sha256": _digest(direction_path),
        },
    }
    (fit_dir / "fit_probe.json").write_text(json.dumps(fit) + "\n", encoding="utf-8")
    return repository, study_path, study, fit_dir


def test_exploratory_fit_identity_is_bound(tmp_path: Path) -> None:
    repository, study_path, study, fit_dir = _fixture(tmp_path)
    fit, direction_path = load_verified_exploratory_fit(
        repository=repository,
        study_path=study_path,
        study=study,
        fit_dir=fit_dir,
        model_key="model_a",
    )
    assert direction_path == (fit_dir / "directions.safetensors").resolve()
    assert exploratory_source_identity(fit)["fit_run_kind"] == "engineering_smoke"


def test_portable_fit_evidence_scope_is_accepted(tmp_path: Path) -> None:
    repository, study_path, study, fit_dir = _fixture(tmp_path)
    fit_path = fit_dir / "fit_probe.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit.pop("scientific_outputs_allowed")
    fit["evidence_scope"] = {
        "endpoint_results": "verified_from_persisted_response_and_metric_primitives"
    }
    fit_path.write_text(json.dumps(fit) + "\n", encoding="utf-8")

    loaded, _ = load_verified_exploratory_fit(
        repository=repository,
        study_path=study_path,
        study=study,
        fit_dir=fit_dir,
        model_key="model_a",
    )
    assert loaded["evidence_scope"]["endpoint_results"].startswith("verified_")


def test_executed_reproduction_fit_is_accepted(tmp_path: Path) -> None:
    repository, study_path, study, fit_dir = _fixture(tmp_path)
    fit_path = fit_dir / "fit_probe.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit["run_kind"] = "executed_reproduction"
    fit_path.write_text(json.dumps(fit) + "\n", encoding="utf-8")

    loaded, _ = load_verified_exploratory_fit(
        repository=repository,
        study_path=study_path,
        study=study,
        fit_dir=fit_dir,
        model_key="model_a",
    )
    assert loaded["run_kind"] == "executed_reproduction"


def test_exploratory_fit_rejects_study_drift(tmp_path: Path) -> None:
    repository, study_path, study, fit_dir = _fixture(tmp_path)
    study_path.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Fit/study SHA-256 mismatch"):
        load_verified_exploratory_fit(
            repository=repository,
            study_path=study_path,
            study=study,
            fit_dir=fit_dir,
            model_key="model_a",
        )
