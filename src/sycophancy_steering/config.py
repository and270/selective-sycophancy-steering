# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validation for the immutable multi-model study contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STUDY_SCHEMA = "selective_sycophancy_multimodel.v1"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BEHAVIOR_METRICS = [
    "neutral_accuracy",
    "neutral_invalid_rate",
    "pressure_error",
    "pressure_error_by_mode",
    "pressure_invalid_rate",
    "natural_correct_suggestion_update_rate",
    "controlled_correction_acceptance_rate",
    "controlled_correction_invalid_rate",
]


def _positive_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_model(key: str, model: Any, *, require_frozen: bool) -> None:
    if not isinstance(model, dict):
        raise ValueError(f"Model {key} specification must be an object")
    if not isinstance(model.get("id"), str) or "/" not in model["id"]:
        raise ValueError(f"Model {key} has an invalid Hub id")
    revision = model.get("revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError(f"Model {key} revision must be an immutable 40-hex commit")
    _positive_int(
        model.get("expected_checkpoint_file_count"),
        label=f"{key} checkpoint file count",
    )
    checkpoint_hash = model.get("expected_checkpoint_content_tree_sha256")
    if not isinstance(checkpoint_hash, str) or not _SHA256.fullmatch(checkpoint_hash):
        raise ValueError(f"Model {key} checkpoint content tree hash is invalid")
    _positive_int(model.get("expected_transformer_layers"), label=f"{key} layer count")
    _positive_int(model.get("expected_hidden_size"), label=f"{key} hidden size")
    if model.get("expected_layer_path") != "model.language_model.layers":
        raise ValueError(f"Model {key} has an unverified layer path")
    if model.get("dtype") != "bfloat16" or model.get("device") != "cuda:0":
        raise ValueError(f"Model {key} has an unsupported dtype/device policy")
    quantization = model.get("quantization")
    if quantization is not None:
        if (
            not isinstance(quantization, dict)
            or quantization.get("method") != "bitsandbytes"
        ):
            raise ValueError(f"Model {key} has an unsupported quantization policy")
        required = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }
        if any(quantization.get(name) != value for name, value in required.items()):
            raise ValueError(f"Model {key} quantization differs from the frozen policy")
    if not isinstance(model.get("chat_template_kwargs"), dict):
        raise ValueError(f"Model {key} chat_template_kwargs must be an object")
    eos_ids = model.get("generation_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(not isinstance(value, int) or value < 0 for value in eos_ids)
        or len(eos_ids) != len(set(eos_ids))
    ):
        raise ValueError(f"Model {key} generation EOS inventory is invalid")
    if require_frozen:
        _positive_int(
            model.get("binary_generation_batch_size"),
            label=f"{key} binary generation batch size",
        )
        _positive_int(
            model.get("residual_batch_size"),
            label=f"{key} residual batch size",
        )


def load_study_config(path: Path, *, require_frozen: bool) -> dict[str, Any]:
    """Load the study JSON and enforce the scientific freeze boundary."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != STUDY_SCHEMA:
        raise ValueError("Unsupported multi-model study schema")
    if payload.get("status") not in {"draft", "frozen"}:
        raise ValueError("Study status must be draft or frozen")
    if require_frozen:
        pending = payload.get("freeze_pending")
        if (
            payload.get("status") != "frozen"
            or payload.get("scientific_outputs_allowed") is not True
            or pending != []
            or not isinstance(payload.get("runtime", {}).get("required_git_tag"), str)
        ):
            raise ValueError(
                "Study is not frozen and complete; scientific outputs are forbidden"
            )
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Study must specify at least one model")
    for key, model in models.items():
        _validate_model(str(key), model, require_frozen=require_frozen)

    alpha_grid = payload.get("activation_operator", {}).get("alpha_grid")
    if (
        not isinstance(alpha_grid, list)
        or not alpha_grid
        or 0.0 not in alpha_grid
        or any(not isinstance(value, (int, float)) for value in alpha_grid)
    ):
        raise ValueError("Activation alpha grid is invalid")
    completion = payload.get("direction_estimation", {}).get("completion_contrast", {})
    sources = completion.get("source_datasets")
    if sources != ["trivia_qa", "truthful_qa"]:
        raise ValueError("Completion estimator source strata differ from frozen schema")
    _positive_int(
        completion.get("minimum_fit_eligible_records_per_source_option"),
        label="minimum fit eligible records per source-option",
    )
    completion_gate = payload.get("layer_selection", {}).get("completion_gate", {})
    if completion_gate.get("source_datasets") != sources:
        raise ValueError("Completion gate source strata differ from estimator")
    _positive_int(
        completion_gate.get("minimum_probe_records_per_source_option"),
        label="minimum probe records per source-option",
    )
    for field in (
        "minimum_each_correct_option_auroc",
        "minimum_each_source_auroc",
        "minimum_each_source_option_auroc",
    ):
        value = completion_gate.get(field)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Completion subgroup threshold is invalid: {field}")
    if payload.get("behavioral_evaluation", {}).get("metrics") != _BEHAVIOR_METRICS:
        raise ValueError("Behavioral metric inventory differs from frozen schema")
    return payload
