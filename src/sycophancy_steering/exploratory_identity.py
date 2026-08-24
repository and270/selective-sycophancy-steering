# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-closed identity checks for the standalone endpoint runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import sha256_file


def load_verified_exploratory_fit(
    *,
    repository: Path,
    study_path: Path,
    study: dict[str, Any],
    fit_dir: Path,
    model_key: str,
) -> tuple[dict[str, Any], Path]:
    """Load a fit summary only when its model/study/data/direction identities match."""

    repository = repository.resolve()
    study_path = study_path.resolve()
    fit_dir = fit_dir.resolve()
    fit_path = fit_dir / "fit_probe.json"
    if not fit_path.is_file():
        raise FileNotFoundError(f"Missing exploratory fit summary: {fit_path}")
    fit = json.loads(fit_path.read_text(encoding="utf-8"))

    if fit.get("model_key") != model_key:
        found_model_key = fit.get("model_key")
        raise ValueError(
            f"Fit model_key mismatch: expected {model_key}, found {found_model_key}"
        )
    study_digest = sha256_file(study_path)
    if fit.get("study_sha256") != study_digest:
        raise ValueError("Fit/study SHA-256 mismatch")

    lock_relative = Path(str(study["data"]["lock"]))
    lock_path = (repository / lock_relative).resolve()
    try:
        lock_path.relative_to(repository)
    except ValueError as error:
        raise ValueError("Study data-lock path escapes the repository") from error
    lock_digest = sha256_file(lock_path)
    if fit.get("data_lock_sha256") != lock_digest:
        raise ValueError("Fit/data-lock SHA-256 mismatch")

    artifact = fit.get("direction_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Fit direction artifact metadata is missing")
    relative = Path(str(artifact.get("path", "")))
    direction_path = (fit_dir / relative).resolve()
    try:
        direction_path.relative_to(fit_dir)
    except ValueError as error:
        raise ValueError("Direction artifact path escapes the fit directory") from error
    if not direction_path.is_file():
        raise FileNotFoundError(f"Missing direction artifact: {direction_path}")
    if sha256_file(direction_path) != artifact.get("sha256"):
        raise ValueError("Direction artifact SHA-256 mismatch")

    if fit.get("run_kind") not in {
        "engineering_smoke",
        "executed_reproduction",
    }:
        raise ValueError(
            "Completed-study runner requires an executed-reproduction-compatible fit"
        )
    portable_scope = fit.get("evidence_scope")
    legacy_launch_gate = fit.get("scientific_outputs_allowed")
    if portable_scope is None:
        if legacy_launch_gate is not False:
            raise ValueError("Standalone fit source launch gate is invalid")
    elif (
        not isinstance(portable_scope, dict)
        or portable_scope.get("endpoint_results")
        != "verified_from_persisted_response_and_metric_primitives"
    ):
        raise ValueError("Portable fit evidence scope is invalid")
    return fit, direction_path


def exploratory_source_identity(fit: dict[str, Any]) -> dict[str, Any]:
    """Return the non-path identity fields persisted in downstream outputs."""

    return {
        "data_lock_sha256": fit["data_lock_sha256"],
        "direction_file_sha256": fit["direction_artifact"]["sha256"],
        "fit_run_kind": fit["run_kind"],
        "study_sha256": fit["study_sha256"],
    }
