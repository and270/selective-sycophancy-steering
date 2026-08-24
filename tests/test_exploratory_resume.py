# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib

import pytest

from sycophancy_steering.resume import (
    load_expanded_gsm8k_resume,
    validate_expanded_gsm8k_resume,
)

SAMPLE_A = "a" * 64
SAMPLE_B = "b" * 64


def _example(doc_id: int, digest: str) -> dict[str, object]:
    return {"doc_id": doc_id, "source_index": doc_id + 10, "sample_sha256": digest}


def _sample_digest(*digests: str) -> str:
    return hashlib.sha256(("\n".join(digests) + "\n").encode()).hexdigest()


def test_valid_gsm8k_checkpoint_resumes_after_completed_alpha_prefix() -> None:
    base = {"examples": [_example(0, SAMPLE_A), _example(1, SAMPLE_B)]}
    trials = [
        {
            "alpha": -2.0,
            "condition": {"examples": [_example(0, SAMPLE_A), _example(1, SAMPLE_B)]},
        }
    ]
    identity = {
        "schema_version": "expanded_exploratory_gsm8k.v1",
        "evidence_scope": {"endpoint_results": "paired_capability_estimates"},
        "model_key": "gemma4_e2b_it",
        "record_count": 2,
        "sample_sha256": _sample_digest(SAMPLE_A, SAMPLE_B),
        "estimator": "observed_prompt_state",
        "zero_based_layer": 12,
        "alphas": [-2.0, -1.0, -0.5],
        "direction_tensor_sha256": "direction",
    }
    payload = {**identity, "base": base, "trials": trials}

    resumed_base, resumed_trials, remaining = validate_expanded_gsm8k_resume(
        payload,
        expected_identity=identity,
    )

    assert resumed_base is base
    assert resumed_trials is trials
    assert remaining == [-1.0, -0.5]


def test_gsm8k_resume_rejects_incomplete_persisted_condition() -> None:
    identity = {
        "schema_version": "expanded_exploratory_gsm8k.v1",
        "evidence_scope": {"endpoint_results": "paired_capability_estimates"},
        "model_key": "gemma4_e2b_it",
        "record_count": 2,
        "sample_sha256": _sample_digest(SAMPLE_A, SAMPLE_B),
        "estimator": "observed_prompt_state",
        "zero_based_layer": 12,
        "alphas": [-2.0, -1.0, -0.5],
        "direction_tensor_sha256": "direction",
    }
    payload = {
        **identity,
        "base": {"examples": [_example(0, SAMPLE_A), _example(1, SAMPLE_B)]},
        "trials": [
            {
                "alpha": -2.0,
                "condition": {"examples": [_example(0, SAMPLE_A)]},
            }
        ],
    }

    with pytest.raises(ValueError, match="record count"):
        validate_expanded_gsm8k_resume(payload, expected_identity=identity)


def test_gsm8k_resume_rejects_invalid_expected_alpha_inventory() -> None:
    identity = {
        "record_count": 1,
        "sample_sha256": _sample_digest(SAMPLE_A),
        "alphas": "-2,-1,-0.5",
    }
    payload = {
        **identity,
        "base": {"examples": [_example(0, SAMPLE_A)]},
        "trials": [],
    }

    with pytest.raises(ValueError, match="alpha inventory"):
        validate_expanded_gsm8k_resume(payload, expected_identity=identity)


def test_gsm8k_resume_rejects_reordered_sample() -> None:
    identity = {
        "record_count": 2,
        "sample_sha256": _sample_digest(SAMPLE_A, SAMPLE_B),
        "alphas": [-2.0],
    }
    payload = {
        **identity,
        "base": {"examples": [_example(0, SAMPLE_B), _example(1, SAMPLE_A)]},
        "trials": [],
    }

    with pytest.raises(ValueError, match="ordered sample identity"):
        validate_expanded_gsm8k_resume(payload, expected_identity=identity)


def test_explicit_resume_loads_matching_checkpoint(tmp_path) -> None:
    identity = {
        "schema_version": "expanded_exploratory_gsm8k.v1",
        "evidence_scope": {"endpoint_results": "paired_capability_estimates"},
        "model_key": "gemma4_e2b_it",
        "record_count": 1,
        "sample_sha256": _sample_digest(SAMPLE_A),
        "estimator": "observed_prompt_state",
        "zero_based_layer": 12,
        "alphas": [-2.0, -1.0, -0.5],
        "direction_tensor_sha256": "direction",
    }
    base = {"examples": [_example(0, SAMPLE_A)]}
    trials = [
        {
            "alpha": -2.0,
            "condition": {"examples": [_example(0, SAMPLE_A)]},
        }
    ]
    checkpoint = tmp_path / "result.checkpoint.json"
    checkpoint.write_text(
        __import__("json").dumps({**identity, "base": base, "trials": trials}),
        encoding="utf-8",
    )

    resumed = load_expanded_gsm8k_resume(
        output_path=tmp_path / "result.json",
        checkpoint_path=checkpoint,
        resume=True,
        expected_identity=identity,
    )

    assert resumed == (base, trials, [-1.0, -0.5])
