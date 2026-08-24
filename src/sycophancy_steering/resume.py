# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-closed validation for resumable exploratory result checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def _ordered_sample_sha256(examples: list[dict[str, Any]]) -> str:
    hashes: list[str] = []
    for index, example in enumerate(examples):
        digest = example.get("sample_sha256")
        source_index = example.get("source_index")
        if (
            example.get("doc_id") != index
            or not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("GSM8K checkpoint example identity is invalid")
        hashes.append(digest)
    return hashlib.sha256(("\n".join(hashes) + "\n").encode()).hexdigest()


def validate_expanded_gsm8k_resume(
    payload: dict[str, Any],
    *,
    expected_identity: Mapping[str, object],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    """Validate checkpoint identity and return its completed alpha prefix."""

    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(f"GSM8K checkpoint identity differs: {field}")
    base = payload.get("base")
    trials = payload.get("trials")
    if not isinstance(base, dict) or not isinstance(trials, list):
        raise ValueError("GSM8K checkpoint is missing base or trial results")
    record_count = expected_identity["record_count"]
    if not isinstance(record_count, int):
        raise ValueError("GSM8K expected record count is invalid")
    conditions = [base]
    for trial in trials:
        if not isinstance(trial, dict) or not isinstance(trial.get("condition"), dict):
            raise ValueError("GSM8K checkpoint contains an invalid trial")
        conditions.append(trial["condition"])
    if any(
        not isinstance(condition.get("examples"), list)
        or len(condition["examples"]) != record_count
        for condition in conditions
    ):
        raise ValueError("GSM8K checkpoint condition record count differs")
    expected_sample = expected_identity.get("sample_sha256")
    for condition in conditions:
        examples = cast(list[dict[str, Any]], condition["examples"])
        if _ordered_sample_sha256(examples) != expected_sample:
            raise ValueError("GSM8K checkpoint ordered sample identity differs")
    alpha_values = expected_identity["alphas"]
    if not isinstance(alpha_values, (list, tuple)) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in alpha_values
    ):
        raise ValueError("GSM8K expected alpha inventory is invalid")
    numeric_alpha_values = cast(
        "list[int | float] | tuple[int | float, ...]", alpha_values
    )
    alphas = [float(value) for value in numeric_alpha_values]
    completed = [float(trial["alpha"]) for trial in trials]
    if completed != alphas[: len(completed)]:
        raise ValueError("GSM8K checkpoint trials are not an alpha prefix")
    return base, trials, alphas[len(completed) :]


def load_expanded_gsm8k_resume(
    *,
    output_path: Path,
    checkpoint_path: Path,
    resume: bool,
    expected_identity: Mapping[str, object],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]] | None:
    """Load an explicitly requested checkpoint or enforce a fresh-start boundary."""

    if output_path.exists():
        raise FileExistsError("Expanded GSM8K output already exists")
    if not checkpoint_path.exists():
        if resume:
            raise FileNotFoundError("Expanded GSM8K checkpoint does not exist")
        return None
    if not resume:
        raise FileExistsError("Expanded GSM8K checkpoint already exists")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expanded GSM8K checkpoint must contain an object")
    return validate_expanded_gsm8k_resume(
        payload,
        expected_identity=expected_identity,
    )
