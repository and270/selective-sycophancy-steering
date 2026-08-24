# SPDX-License-Identifier: AGPL-3.0-or-later

"""Descriptive evaluation of every preregistered steering frontier point."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file

from .artifacts import (
    atomic_write_json,
    build_runtime_manifest,
    capture_scientific_launch_identity,
    finalize_artifact_stage,
    tensor_sha256,
    verify_artifact_manifest,
    verify_loaded_study_identity,
    verify_parent_launch_identity,
)
from .behavior import (
    BehaviorRun,
    collect_behavior_run,
    generate_baseline_answers,
)
from .comparison import compare_behavior_runs
from .config import load_study_config
from .data import (
    load_data_lock,
    record_contract,
    record_contract_sha256,
    sha256_file,
    validate_materialized_data,
)
from .directions import compute_completion_contrast, compute_observed_prompt_direction
from .fit_probe import (
    DirectionObservations,
    completion_fit_status,
    completion_probe_result,
    observed_probe_result,
    observed_valid_data,
)
from .loading import load_study_model, unload_study_model
from .metrics import compute_behavior_metrics
from .probe import select_estimator_layers
from .prompts import parse_binary_letter


def verify_loaded_fingerprint(
    loaded: Any, artifact: dict[str, Any], *, label: str
) -> None:
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} artifact has no runtime fingerprint")
    if runtime.get("model_fingerprint") != loaded.model_fingerprint:
        raise ValueError(f"{label} model fingerprint differs from the live model")
    if runtime.get("tokenizer_fingerprint") != loaded.tokenizer_fingerprint:
        raise ValueError(
            f"{label} tokenizer fingerprint differs from the live tokenizer"
        )


def expected_frontier_conditions(
    fit_result: dict[str, Any], study: dict[str, Any]
) -> list[dict[str, Any]]:
    selection = fit_result["layer_selection"]
    estimator = selection["chosen_estimator"]
    layers = [int(layer) for layer in selection["chosen_layers"]]
    if estimator is None:
        if layers:
            raise ValueError("Fit artifact has layers without a chosen estimator")
        return []
    if not layers:
        raise ValueError("Fit artifact has an estimator without chosen layers")
    nonzero = [
        float(alpha)
        for alpha in study["activation_operator"]["alpha_grid"]
        if float(alpha) != 0.0
    ]
    conditions = [
        {"estimator": str(estimator), "zero_based_layer": layers[0], "alpha": 0.0}
    ]
    conditions.extend(
        {"estimator": str(estimator), "zero_based_layer": layer, "alpha": alpha}
        for layer in layers
        for alpha in nonzero
    )
    return conditions


def behavior_run_payload(run: BehaviorRun) -> dict[str, Any]:
    return asdict(run)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OBSERVATION_TENSOR_NAMES = {
    f"{split}_{kind}_residuals"
    for split in ("fit", "probe")
    for kind in ("prompt", "caving", "resisting")
}
_DESCRIPTOR_FIELDS = {
    "record_id",
    "source_dataset",
    "correct_option",
    "mode",
    "caving_completion",
    "resisting_completion",
    "prompt_sha256",
    "caving_text_sha256",
    "resisting_text_sha256",
}


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_answer(answer: Any, *, prompt_hash: bool) -> None:
    expected = {"text", "parsed"}
    if prompt_hash:
        expected.add("prompt_sha256")
    if not isinstance(answer, dict) or set(answer) != expected:
        raise ValueError("Observation answer schema is invalid")
    text = answer.get("text")
    parsed = answer.get("parsed")
    if not isinstance(text, str) or parsed != parse_binary_letter(text):
        raise ValueError("Observation answer text and parsed value differ")
    if prompt_hash and not _valid_sha256(answer.get("prompt_sha256")):
        raise ValueError("Observation prompt hash is invalid")


def _load_verified_tensor_artifact(
    directory: Path,
    metadata: Any,
    *,
    expected_path: str,
    expected_names: set[str],
) -> dict[str, torch.Tensor]:
    path = directory / expected_path
    if (
        not isinstance(metadata, dict)
        or metadata.get("path") != expected_path
        or metadata.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"Tensor artifact metadata mismatch: {expected_path}")
    tensors = load_file(path)
    tensor_metadata = metadata.get("tensors")
    if (
        not isinstance(tensor_metadata, dict)
        or set(tensors) != expected_names
        or set(tensor_metadata) != expected_names
    ):
        raise ValueError(f"Tensor artifact inventory mismatch: {expected_path}")
    for name, tensor in tensors.items():
        entry = tensor_metadata[name]
        if (
            not isinstance(entry, dict)
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
            or entry.get("shape") != list(tensor.shape)
            or entry.get("dtype") != str(tensor.dtype)
            or entry.get("sha256") != tensor_sha256(tensor)
        ):
            raise ValueError(f"Tensor artifact contract mismatch: {name}")
    return tensors


def _reconstruct_observations(
    payload: Any,
    tensors: dict[str, torch.Tensor],
    *,
    prefix: str,
    records: list[dict[str, Any]],
    modes: tuple[str, ...],
    expected_shape: tuple[int, int],
) -> DirectionObservations:
    if not isinstance(payload, dict):
        raise ValueError(f"{prefix} observation payload is invalid")
    baseline = payload.get("baseline_answers")
    examples = payload.get("examples")
    record_by_id = {str(record["id"]): record for record in records}
    if (
        not isinstance(baseline, dict)
        or set(baseline) != set(record_by_id)
        or not isinstance(examples, list)
    ):
        raise ValueError(f"{prefix} observation identities are invalid")
    for answer in baseline.values():
        _validate_answer(answer, prompt_hash=True)

    expected_examples: list[tuple[str, str, str, str, str, str]] = []
    for record_id, record in record_by_id.items():
        answer = baseline[record_id]
        if answer["parsed"] == record["correct_option"]:
            if not answer["text"]:
                raise ValueError("Eligible baseline response is empty")
            expected_examples.extend(
                (
                    record_id,
                    str(record["source_dataset"]),
                    str(record["correct_option"]),
                    mode,
                    str(record["wrong_option"]),
                    str(record["correct_option"]),
                )
                for mode in modes
            )
    if len(examples) != len(expected_examples):
        raise ValueError(f"{prefix} observation example count is invalid")
    descriptors: list[dict[str, str]] = []
    followups: list[dict[str, str | None]] = []
    for example, expected in zip(examples, expected_examples, strict=True):
        if not isinstance(example, dict) or set(example) != {
            *_DESCRIPTOR_FIELDS,
            "followup",
        }:
            raise ValueError(f"{prefix} observation example schema is invalid")
        example_dict = cast(dict[str, Any], example)
        if any(
            not isinstance(example_dict[field], str) for field in _DESCRIPTOR_FIELDS
        ):
            raise ValueError(f"{prefix} observation descriptor fields are invalid")
        descriptor = {
            field: cast(str, example_dict[field]) for field in _DESCRIPTOR_FIELDS
        }
        actual = (
            descriptor["record_id"],
            descriptor["source_dataset"],
            descriptor["correct_option"],
            descriptor["mode"],
            descriptor["caving_completion"],
            descriptor["resisting_completion"],
        )
        if actual != expected or any(
            not _valid_sha256(descriptor[field])
            for field in (
                "prompt_sha256",
                "caving_text_sha256",
                "resisting_text_sha256",
            )
        ):
            raise ValueError(f"{prefix} observation descriptor differs from contract")
        followup = example_dict["followup"]
        _validate_answer(followup, prompt_hash=False)
        descriptors.append(descriptor)
        followups.append(cast(dict[str, str | None], followup))
    if payload.get("n_examples") != len(examples) or payload.get(
        "n_invalid_followups"
    ) != sum(answer["parsed"] is None for answer in followups):
        raise ValueError(f"{prefix} observation summary is invalid")

    residuals = {
        kind: tensors[f"{prefix}_{kind}_residuals"]
        for kind in ("prompt", "caving", "resisting")
    }
    tensor_shape = (len(examples), *expected_shape)
    if any(tuple(tensor.shape) != tensor_shape for tensor in residuals.values()):
        raise ValueError(f"{prefix} residual tensor shape is invalid")
    return DirectionObservations(
        baseline_answers=baseline,
        descriptors=descriptors,
        followup_answers=followups,
        prompt_residuals=residuals["prompt"],
        caving_residuals=residuals["caving"],
        resisting_residuals=residuals["resisting"],
    )


def _verify_fit_semantics(
    result: dict[str, Any],
    directions: dict[str, torch.Tensor],
    observation_tensors: dict[str, torch.Tensor],
    *,
    study: dict[str, Any],
    data_lock: dict[str, Any],
    expected_shape: tuple[int, int],
) -> None:
    contracts = result.get("record_contracts")
    contract_hashes = result.get("record_contract_sha256")
    if (
        not isinstance(contracts, dict)
        or set(contracts) != {"direction_fit", "direction_probe"}
        or not isinstance(contract_hashes, dict)
        or set(contract_hashes) != set(contracts)
    ):
        raise ValueError("Fit/probe record contracts are missing")
    for split, records in contracts.items():
        locked = data_lock["splits"][split]
        digest = record_contract_sha256(records)
        if (
            digest != locked.get("record_contract_sha256")
            or digest != contract_hashes.get(split)
            or [record["id"] for record in records] != locked.get("ids")
        ):
            raise ValueError(f"Fit/probe record contract mismatch: {split}")

    modes = tuple(study["prompt_contract"]["pressure_modes"])
    fit = _reconstruct_observations(
        result.get("fit"),
        observation_tensors,
        prefix="fit",
        records=contracts["direction_fit"],
        modes=modes,
        expected_shape=expected_shape,
    )
    probe = _reconstruct_observations(
        result.get("probe"),
        observation_tensors,
        prefix="probe",
        records=contracts["direction_probe"],
        modes=modes,
        expected_shape=expected_shape,
    )
    completion_spec = study["direction_estimation"]["completion_contrast"]
    source_datasets = tuple(completion_spec["source_datasets"])
    minimum_fit_count = int(
        completion_spec["minimum_fit_eligible_records_per_source_option"]
    )
    completion_status = completion_fit_status(
        fit,
        source_datasets=source_datasets,
        minimum_records_per_source_option=minimum_fit_count,
    )
    completion_fit_eligible = bool(completion_status["eligible"])
    recomputed_directions: dict[str, torch.Tensor] = {}
    if completion_fit_eligible:
        recomputed_directions["completion_contrast"] = compute_completion_contrast(
            fit.caving_residuals, fit.resisting_residuals
        )
    estimator_status: dict[str, dict[str, Any]] = {
        "completion_contrast": completion_status
    }
    observed_spec = study["direction_estimation"]["observed_prompt_state"]
    try:
        residuals, labels, item_modes = observed_valid_data(fit)
        observed, counts = compute_observed_prompt_direction(
            residuals,
            labels,
            item_modes,
            expected_modes=modes,
            minimum_overall=int(observed_spec["minimum_fit_class_count_overall"]),
            minimum_per_mode=int(observed_spec["minimum_fit_class_count_per_mode"]),
        )
        recomputed_directions["observed_prompt_state"] = observed
        estimator_status["observed_prompt_state"] = {
            "eligible": True,
            "fit_class_counts": counts,
        }
    except ValueError as error:
        estimator_status["observed_prompt_state"] = {
            "eligible": False,
            "reason": str(error),
        }
    if set(directions) != set(recomputed_directions) or any(
        not torch.equal(directions[name], tensor)
        for name, tensor in recomputed_directions.items()
    ):
        raise ValueError("Persisted directions differ from primitive residuals")

    control_count = int(study["layer_selection"]["random_direction_controls_per_layer"])
    seed = int(study["layer_selection"]["random_seed"])
    probe_results: dict[str, dict[str, Any]] = {}
    if completion_fit_eligible:
        probe_results["completion_contrast"] = completion_probe_result(
            probe,
            directions["completion_contrast"],
            modes=modes,
            source_datasets=source_datasets,
            random_controls=control_count,
            random_seed=seed,
        )
    if "observed_prompt_state" in directions:
        try:
            probe_results["observed_prompt_state"] = observed_probe_result(
                probe,
                directions["observed_prompt_state"],
                modes=modes,
                random_controls=control_count,
                random_seed=seed + 1,
            )
        except ValueError as error:
            estimator_status["observed_prompt_state"]["probe_error"] = str(error)
    if estimator_status != result.get("estimator_status"):
        raise ValueError("Estimator status differs from primitive observations")
    if probe_results != result.get("probe_results"):
        raise ValueError("Probe results differ from primitive residuals")
    selection = select_estimator_layers(
        probe_results,
        modes=modes,
        policy=study["layer_selection"],
    )
    if selection != result.get("layer_selection"):
        raise ValueError("Layer selection differs from recomputed probe results")


def _verify_fit_artifact(
    fit_probe_dir: Path,
    *,
    model_key: str,
    study_path: Path,
    data_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    result_path = fit_probe_dir / "fit_probe.json"
    status_path = fit_probe_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("stage") != "complete"
        or status.get("run_kind") != "scientific"
        or status.get("model_key") != model_key
    ):
        raise ValueError("Fit/probe status is not a completed scientific run")
    verify_artifact_manifest(
        fit_probe_dir,
        status=status,
        expected_files=(
            "fit_probe.json",
            "directions.safetensors",
            "observations.safetensors",
        ),
    )
    study = load_study_config(study_path, require_frozen=True)
    model_spec = study["models"][model_key]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "selective_sycophancy_fit_probe.v3":
        raise ValueError("Unsupported fit/probe artifact")
    if (
        result.get("run_kind") != "scientific"
        or result.get("scientific_outputs_allowed") is not True
    ):
        raise ValueError("Frontier evaluation requires a scientific fit/probe artifact")
    if result.get("model_key") != model_key:
        raise ValueError("Fit/probe artifact belongs to a different model")
    if result.get("study_sha256") != sha256_file(study_path):
        raise ValueError("Fit/probe artifact does not match the frozen study")
    if result.get("data_lock_sha256") != sha256_file(data_lock_path):
        raise ValueError("Fit/probe artifact does not match the data lock")
    runtime = result.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("stage") != "fit_probe"
        or runtime.get("run_kind") != "scientific"
        or runtime.get("model_key") != model_key
        or runtime.get("model_id") != model_spec["id"]
        or runtime.get("model_revision") != model_spec["revision"]
        or runtime.get("generation_eos_token_ids")
        != model_spec["generation_eos_token_ids"]
        or runtime.get("repository_dirty") is not False
        or runtime.get("study_sha256") != sha256_file(study_path)
        or runtime.get("data_lock_sha256") != sha256_file(data_lock_path)
        or runtime.get("accessed_splits") != ["direction_fit", "direction_probe"]
        or not isinstance(runtime.get("launch_identity"), dict)
        or not isinstance(runtime.get("model_fingerprint"), dict)
        or not isinstance(runtime.get("tokenizer_fingerprint"), dict)
    ):
        raise ValueError("Fit/probe runtime manifest is incomplete or inconsistent")
    fit_launch_identity = cast(dict[str, Any], runtime["launch_identity"])
    verify_loaded_study_identity(fit_launch_identity, study)
    if result.get("fit_record_count") != study["data"]["direction_fit"]["count"]:
        raise ValueError("Fit/probe fit-record count differs from the study")
    if result.get("probe_record_count") != study["data"]["direction_probe"]["count"]:
        raise ValueError("Fit/probe probe-record count differs from the study")
    direction_metadata = result.get("direction_artifact")
    expected = (
        direction_metadata.get("tensors")
        if isinstance(direction_metadata, dict)
        else None
    )
    if not isinstance(expected, dict) or any(
        not isinstance(name, str) for name in expected
    ):
        raise ValueError("Direction tensor inventory mismatch")
    expected_metadata = cast(dict[str, Any], expected)
    expected_names = set(expected_metadata)
    if not expected_names or not expected_names <= {
        "completion_contrast",
        "observed_prompt_state",
    }:
        raise ValueError("Direction tensor inventory mismatch")
    directions = _load_verified_tensor_artifact(
        fit_probe_dir,
        direction_metadata,
        expected_path="directions.safetensors",
        expected_names=expected_names,
    )
    observation_tensors = _load_verified_tensor_artifact(
        fit_probe_dir,
        result.get("observation_artifact"),
        expected_path="observations.safetensors",
        expected_names=_OBSERVATION_TENSOR_NAMES,
    )
    expected_shape = (
        int(model_spec["expected_transformer_layers"]),
        int(model_spec["expected_hidden_size"]),
    )
    for name, tensor in directions.items():
        metadata = cast(dict[str, Any], expected_metadata[name])
        norms = torch.linalg.vector_norm(tensor.to(torch.float32), dim=1)
        if (
            tuple(tensor.shape) != expected_shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
            or torch.any(norms <= 0)
            or metadata.get("shape") != list(expected_shape)
            or metadata.get("dtype") != str(tensor.dtype)
            or tensor_sha256(tensor) != metadata.get("sha256")
        ):
            raise ValueError(f"Direction tensor contract mismatch: {name}")
        layer_norms = metadata.get("layer_norms")
        if not isinstance(layer_norms, list) or len(layer_norms) != expected_shape[0]:
            raise ValueError(f"Direction norm metadata is invalid: {name}")
        recorded_norms = torch.tensor(layer_norms, dtype=torch.float32)
        if recorded_norms.shape != norms.shape or not torch.allclose(
            recorded_norms, norms, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"Direction norm metadata mismatch: {name}")
    data_lock = load_data_lock(
        data_lock_path,
        expected_sha256=str(fit_launch_identity["data_lock_sha256"]),
    )
    _verify_fit_semantics(
        result,
        directions,
        observation_tensors,
        study=study,
        data_lock=data_lock,
        expected_shape=expected_shape,
    )
    selection = result["layer_selection"]
    if (
        status.get("chosen_estimator") != selection["chosen_estimator"]
        or status.get("chosen_layers") != selection["chosen_layers"]
    ):
        raise ValueError("Fit/probe status selection differs from payload")
    estimator = selection["chosen_estimator"]
    layers = [int(layer) for layer in selection["chosen_layers"]]
    if estimator is not None and estimator not in directions:
        raise ValueError("Selected estimator has no direction tensor")
    if any(layer < 0 or layer >= expected_shape[0] for layer in layers):
        raise ValueError("Selected layer is outside the model")
    expected_frontier_conditions(result, study)
    return result, directions


_BEHAVIOR_FIELDS = {
    "metrics",
    "neutral_answers",
    "pressure_answers",
    "natural_correction_answers",
    "controlled_correction_answers",
    "prompt_hashes",
    "hook_audit",
}
_AUDIT_FIELDS = {
    "calls",
    "prefill_calls",
    "decode_calls",
    "modified_batch_rows",
    "modified_token_positions",
}


def _validate_response_map(
    value: Any, expected_ids: list[str], *, label: str
) -> dict[str, dict[str, str | None]]:
    if not isinstance(value, dict) or set(value) != set(expected_ids):
        raise ValueError(f"{label} response inventory is invalid")
    output = cast(dict[str, dict[str, str | None]], value)
    for answer in output.values():
        _validate_answer(answer, prompt_hash=False)
    return output


def _validate_hash_inventory(
    value: Any, expected_ids: list[str], *, label: str
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != set(expected_ids)
        or any(not _valid_sha256(digest) for digest in value.values())
    ):
        raise ValueError(f"{label} prompt-hash inventory is invalid")


def _verify_behavior_payload(
    payload: Any,
    records: list[dict[str, Any]],
    *,
    modes: tuple[str, ...],
    base_neutral: dict[str, dict[str, str | None]] | None,
    expect_hook: bool,
    expect_baseline_generation_hashes: bool,
) -> BehaviorRun:
    if not isinstance(payload, dict) or set(payload) != _BEHAVIOR_FIELDS:
        raise ValueError("Behavior payload schema is invalid")
    ids = [str(record["id"]) for record in records]
    correct = {str(record["id"]): str(record["correct_option"]) for record in records}
    neutral = _validate_response_map(payload["neutral_answers"], ids, label="Neutral")
    eligibility_answers = neutral if base_neutral is None else base_neutral
    if set(eligibility_answers) != set(ids):
        raise ValueError("Behavior base-neutral inventory is invalid")
    eligible = [
        record_id
        for record_id in ids
        if eligibility_answers[record_id]["parsed"] == correct[record_id]
    ]
    ineligible = [record_id for record_id in ids if record_id not in set(eligible)]

    pressure_value = payload["pressure_answers"]
    if not isinstance(pressure_value, dict) or set(pressure_value) != set(eligible):
        raise ValueError("Pressure response inventory is invalid")
    pressure = cast(dict[str, dict[str, dict[str, str | None]]], pressure_value)
    for record_id in eligible:
        if set(pressure[record_id]) != set(modes):
            raise ValueError("Pressure mode inventory is invalid")
        for answer in pressure[record_id].values():
            _validate_answer(answer, prompt_hash=False)
    natural = _validate_response_map(
        payload["natural_correction_answers"],
        ineligible,
        label="Natural correction",
    )
    controlled = _validate_response_map(
        payload["controlled_correction_answers"],
        ids,
        label="Controlled correction",
    )

    prompt_hashes = payload["prompt_hashes"]
    expected_hash_sections = {
        "neutral",
        "pressure",
        "natural_correction",
        "controlled_correction",
    }
    if expect_baseline_generation_hashes:
        expected_hash_sections.add("baseline_generation")
    if (
        not isinstance(prompt_hashes, dict)
        or set(prompt_hashes) != expected_hash_sections
    ):
        raise ValueError("Behavior prompt-hash sections are invalid")
    _validate_hash_inventory(prompt_hashes["neutral"], ids, label="Neutral")
    _validate_hash_inventory(
        prompt_hashes["natural_correction"], ineligible, label="Natural correction"
    )
    _validate_hash_inventory(
        prompt_hashes["controlled_correction"], ids, label="Controlled correction"
    )
    if expect_baseline_generation_hashes:
        _validate_hash_inventory(
            prompt_hashes["baseline_generation"], ids, label="Baseline generation"
        )
    pressure_hashes = prompt_hashes["pressure"]
    if not isinstance(pressure_hashes, dict) or set(pressure_hashes) != set(eligible):
        raise ValueError("Pressure prompt-hash record inventory is invalid")
    for mode_hashes in pressure_hashes.values():
        if (
            not isinstance(mode_hashes, dict)
            or set(mode_hashes) != set(modes)
            or any(not _valid_sha256(digest) for digest in mode_hashes.values())
        ):
            raise ValueError("Pressure prompt-hash mode inventory is invalid")

    parsed_base = {
        record_id: answer["parsed"] for record_id, answer in eligibility_answers.items()
    }
    parsed_neutral = {
        record_id: answer["parsed"] for record_id, answer in neutral.items()
    }
    parsed_pressure = {
        record_id: {mode: answer["parsed"] for mode, answer in answers.items()}
        for record_id, answers in pressure.items()
    }
    parsed_natural = {
        record_id: answer["parsed"] for record_id, answer in natural.items()
    }
    parsed_controlled = {
        record_id: answer["parsed"] for record_id, answer in controlled.items()
    }
    recomputed_metrics = compute_behavior_metrics(
        records,
        parsed_base,
        parsed_neutral,
        parsed_pressure,
        parsed_natural,
        parsed_controlled,
        modes=modes,
    )
    if recomputed_metrics != payload["metrics"]:
        raise ValueError("Behavior metrics differ from raw responses")

    audit = payload["hook_audit"]
    if expect_hook:
        if (
            not isinstance(audit, dict)
            or set(audit) != _AUDIT_FIELDS
            or any(
                not isinstance(audit[field], int) or isinstance(audit[field], bool)
                for field in _AUDIT_FIELDS
            )
            or audit["calls"] <= 0
            or audit["prefill_calls"] + audit["decode_calls"] != audit["calls"]
            or audit["modified_batch_rows"] <= 0
            or audit["modified_token_positions"] < audit["modified_batch_rows"]
        ):
            raise ValueError("Behavior hook audit is invalid")
    elif audit is not None:
        raise ValueError("Unsteered base unexpectedly contains a hook audit")

    return BehaviorRun(
        metrics=payload["metrics"],
        neutral_answers=neutral,
        pressure_answers=pressure,
        natural_correction_answers=natural,
        controlled_correction_answers=controlled,
        prompt_hashes=prompt_hashes,
        hook_audit=audit,
    )


def verify_frontier_artifact(
    frontier_dir: Path,
    *,
    model_key: str,
    study_path: Path,
    data_lock_path: Path,
    fit_probe_dir: Path,
    fit_result: dict[str, Any],
    directions: dict[str, torch.Tensor],
) -> dict[str, Any]:
    status = json.loads((frontier_dir / "status.json").read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("stage") != "complete"
        or status.get("run_kind") != "scientific"
        or status.get("model_key") != model_key
    ):
        raise ValueError("Frontier status is not complete")
    verify_artifact_manifest(
        frontier_dir,
        status=status,
        expected_files=("frontier.json",),
    )
    study = load_study_config(study_path, require_frozen=True)
    frontier = json.loads((frontier_dir / "frontier.json").read_text(encoding="utf-8"))
    if frontier.get("schema_version") != "selective_sycophancy_frontier.v2":
        raise ValueError("Unsupported frontier artifact")
    if frontier.get("reporting") != "descriptive_no_accept_reject_verdict":
        raise ValueError("Frontier reporting contract differs")
    if frontier.get("model_key") != model_key:
        raise ValueError("Frontier artifact belongs to a different model")
    if frontier.get("study_sha256") != sha256_file(study_path):
        raise ValueError("Frontier artifact does not match the frozen study")
    if frontier.get("data_lock_sha256") != sha256_file(data_lock_path):
        raise ValueError("Frontier artifact does not match the data lock")
    runtime = frontier.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("stage") != "frontier"
        or runtime.get("run_kind") != "scientific"
        or runtime.get("model_key") != model_key
        or runtime.get("model_id") != study["models"][model_key]["id"]
        or runtime.get("model_revision") != study["models"][model_key]["revision"]
        or runtime.get("generation_eos_token_ids")
        != study["models"][model_key]["generation_eos_token_ids"]
        or runtime.get("repository_dirty") is not False
        or runtime.get("study_sha256") != sha256_file(study_path)
        or runtime.get("data_lock_sha256") != sha256_file(data_lock_path)
        or runtime.get("accessed_splits") != ["evaluation"]
        or not isinstance(runtime.get("launch_identity"), dict)
        or not isinstance(runtime.get("model_fingerprint"), dict)
        or not isinstance(runtime.get("tokenizer_fingerprint"), dict)
    ):
        raise ValueError("Frontier runtime manifest is incomplete or inconsistent")
    frontier_launch_identity = cast(dict[str, Any], runtime["launch_identity"])
    verify_loaded_study_identity(frontier_launch_identity, study)
    fit_runtime = fit_result.get("runtime")
    if (
        not isinstance(fit_runtime, dict)
        or runtime["model_fingerprint"] != fit_runtime.get("model_fingerprint")
        or runtime["tokenizer_fingerprint"] != fit_runtime.get("tokenizer_fingerprint")
    ):
        raise ValueError("Frontier runtime identity differs from fit/probe")
    verify_parent_launch_identity(
        runtime["launch_identity"], fit_result, label="fit/probe"
    )
    if frontier.get("fit_probe_sha256") != sha256_file(
        fit_probe_dir / "fit_probe.json"
    ):
        raise ValueError("Frontier artifact does not match fit/probe")
    if frontier.get("directions_sha256") != sha256_file(
        fit_probe_dir / "directions.safetensors"
    ):
        raise ValueError("Frontier artifact does not match directions")
    if frontier.get("record_count") != study["data"]["evaluation"]["count"]:
        raise ValueError("Frontier record count differs from the study")
    data_lock = load_data_lock(
        data_lock_path,
        expected_sha256=str(frontier_launch_identity["data_lock_sha256"]),
    )
    records_value = frontier.get("record_contract")
    if not isinstance(records_value, list) or any(
        not isinstance(record, dict) for record in records_value
    ):
        raise ValueError("Frontier record contract is missing")
    records = cast(list[dict[str, Any]], records_value)
    contract_digest = record_contract_sha256(records)
    evaluation_lock = data_lock["splits"]["evaluation"]
    if (
        len(records) != frontier["record_count"]
        or contract_digest != frontier.get("record_contract_sha256")
        or contract_digest != evaluation_lock.get("record_contract_sha256")
        or [record["id"] for record in records] != evaluation_lock.get("ids")
    ):
        raise ValueError("Frontier record contract differs from the data lock")
    selection = fit_result["layer_selection"]
    if (
        frontier.get("chosen_estimator") != selection["chosen_estimator"]
        or frontier.get("chosen_layers") != selection["chosen_layers"]
    ):
        raise ValueError("Frontier selection differs from fit/probe")
    expected = expected_frontier_conditions(fit_result, study)
    modes = tuple(study["prompt_contract"]["pressure_modes"])
    base = _verify_behavior_payload(
        frontier.get("base"),
        records,
        modes=modes,
        base_neutral=None,
        expect_hook=False,
        expect_baseline_generation_hashes=True,
    )
    trials = frontier.get("trials")
    if (
        not isinstance(trials, list)
        or frontier.get("condition_count") != len(trials)
        or status.get("condition_count") != len(trials)
    ):
        raise ValueError("Frontier condition inventory is invalid")
    trial_fields = {
        "estimator",
        "zero_based_layer",
        "alpha",
        "probe_auroc",
        "direction_tensor_sha256",
        "condition",
        "comparison_to_base",
        "zero_alpha_identity",
    }
    if any(
        not isinstance(trial, dict) or set(trial) != trial_fields for trial in trials
    ):
        raise ValueError("Frontier trial schema is invalid")
    actual = [
        {
            "estimator": trial.get("estimator"),
            "zero_based_layer": trial.get("zero_based_layer"),
            "alpha": float(trial.get("alpha")),
        }
        for trial in trials
    ]
    if actual != expected or len({tuple(item.values()) for item in actual}) != len(
        actual
    ):
        raise ValueError("Frontier trials differ from the frozen Cartesian inventory")
    for trial in trials:
        estimator = str(trial["estimator"])
        layer = int(trial["zero_based_layer"])
        direction = directions[estimator][layer]
        if trial.get("direction_tensor_sha256") != tensor_sha256(direction):
            raise ValueError("Frontier direction tensor hash mismatch")
        expected_auroc = fit_result["probe_results"][estimator]["overall_auroc"][layer]
        if trial.get("probe_auroc") != expected_auroc:
            raise ValueError("Frontier probe AUROC differs from fit/probe")
        condition_payload = trial.get("condition")
        condition_run = _verify_behavior_payload(
            condition_payload,
            records,
            modes=modes,
            base_neutral=base.neutral_answers,
            expect_hook=True,
            expect_baseline_generation_hashes=False,
        )
        uncertainty = study["behavioral_evaluation"]["uncertainty"]
        recomputed_comparison = compare_behavior_runs(
            records,
            base,
            condition_run,
            modes=modes,
            bootstrap_iterations=int(uncertainty["iterations"]),
            bootstrap_seed=int(uncertainty["seed"]),
            confidence=float(uncertainty["confidence"]),
        )
        if trial.get("comparison_to_base") != recomputed_comparison:
            raise ValueError("Frontier comparison differs from raw responses")
        is_zero = float(trial["alpha"]) == 0.0
        if trial.get("zero_alpha_identity") is not is_zero:
            raise ValueError("Frontier zero-alpha identity marker is invalid")
        if is_zero:
            if not _zero_identity(base, condition_run):
                raise ValueError("Frontier zero-alpha raw identity differs")
            expected_hashes = {
                key: value
                for key, value in base.prompt_hashes.items()
                if key != "baseline_generation"
            }
            if condition_run.prompt_hashes != expected_hashes:
                raise ValueError("Frontier zero-alpha prompt identity differs")
    return frontier


def _zero_identity(base: BehaviorRun, condition: BehaviorRun) -> bool:
    return (
        condition.metrics == base.metrics
        and condition.neutral_answers == base.neutral_answers
        and condition.pressure_answers == base.pressure_answers
        and condition.natural_correction_answers == base.natural_correction_answers
        and condition.controlled_correction_answers
        == base.controlled_correction_answers
    )


def run_frontier_stage(
    *,
    repository: Path,
    study_path: Path,
    data_dir: Path,
    fit_probe_dir: Path,
    output_dir: Path,
    model_key: str,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Evaluate all frozen layer/alpha points on the untouched evaluation split."""

    if output_dir.exists():
        raise FileExistsError(f"Frontier output already exists: {output_dir}")
    launch_identity = capture_scientific_launch_identity(repository, study_path)
    study = load_study_config(study_path, require_frozen=True)
    verify_loaded_study_identity(launch_identity, study)
    if model_key not in study["models"]:
        raise ValueError(f"Unknown model key: {model_key}")
    model_spec = study["models"][model_key]
    data_lock_path = repository / study["data"]["lock"]
    fit_result, directions = _verify_fit_artifact(
        fit_probe_dir,
        model_key=model_key,
        study_path=study_path,
        data_lock_path=data_lock_path,
    )
    verify_parent_launch_identity(launch_identity, fit_result, label="fit/probe")
    records = validate_materialized_data(
        data_dir,
        data_lock_path,
        allowed_splits=("evaluation",),
        expected_lock_sha256=str(launch_identity["data_lock_sha256"]),
    )["evaluation"]

    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        output_dir / "status.json",
        {"complete": False, "stage": "base", "model_key": model_key},
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loaded = load_study_model(
        model_key,
        model_spec,
        seed=int(study["runtime"]["seed"]),
        local_files_only=local_files_only,
    )
    try:
        verify_loaded_fingerprint(loaded, fit_result, label="fit/probe")
        contract = study["prompt_contract"]
        chat_kwargs = model_spec["chat_template_kwargs"]
        batch_size = int(model_spec["binary_generation_batch_size"])
        max_new_tokens = int(study["runtime"]["maximum_binary_new_tokens"])
        base_neutral, baseline_prompt_hashes = generate_baseline_answers(
            loaded,
            records,
            contract,
            chat_template_kwargs=chat_kwargs,
            generation_batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        base = collect_behavior_run(
            loaded,
            records,
            base_neutral,
            contract,
            chat_template_kwargs=chat_kwargs,
            generation_batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            steering=None,
            reuse_base_neutral=True,
        )
        base.prompt_hashes["baseline_generation"] = baseline_prompt_hashes

        selection = fit_result["layer_selection"]
        estimator = selection["chosen_estimator"]
        layers = [int(layer) for layer in selection["chosen_layers"]]
        conditions = expected_frontier_conditions(fit_result, study)

        trials: list[dict[str, Any]] = []
        uncertainty = study["behavioral_evaluation"]["uncertainty"]
        for condition_spec in conditions:
            layer_index = int(condition_spec["zero_based_layer"])
            alpha = float(condition_spec["alpha"])
            condition_estimator = str(condition_spec["estimator"])
            atomic_write_json(
                output_dir / "status.json",
                {
                    "complete": False,
                    "stage": "frontier_condition",
                    "model_key": model_key,
                    "estimator": condition_estimator,
                    "layer": layer_index,
                    "alpha": alpha,
                    "completed_conditions": len(trials),
                    "total_conditions": len(conditions),
                },
            )
            direction = directions[condition_estimator][layer_index]
            condition = collect_behavior_run(
                loaded,
                records,
                base_neutral,
                contract,
                chat_template_kwargs=chat_kwargs,
                generation_batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                steering=(
                    loaded.text_model,
                    loaded.layers[layer_index],
                    direction,
                    alpha,
                ),
                reuse_base_neutral=False,
            )
            if alpha == 0.0 and not _zero_identity(base, condition):
                raise RuntimeError("Zero-alpha hook differs from the unsteered base")
            comparison = compare_behavior_runs(
                records,
                base,
                condition,
                modes=tuple(contract["pressure_modes"]),
                bootstrap_iterations=int(uncertainty["iterations"]),
                bootstrap_seed=int(uncertainty["seed"]),
                confidence=float(uncertainty["confidence"]),
            )
            trials.append(
                {
                    "estimator": condition_estimator,
                    "zero_based_layer": layer_index,
                    "alpha": alpha,
                    "probe_auroc": fit_result["probe_results"][condition_estimator][
                        "overall_auroc"
                    ][layer_index],
                    "direction_tensor_sha256": tensor_sha256(direction),
                    "condition": behavior_run_payload(condition),
                    "comparison_to_base": comparison,
                    "zero_alpha_identity": alpha == 0.0,
                }
            )
            atomic_write_json(
                output_dir / "frontier.checkpoint.json",
                {
                    "model_key": model_key,
                    "base": behavior_run_payload(base),
                    "trials": trials,
                },
            )

        payload: dict[str, Any] = {
            "schema_version": "selective_sycophancy_frontier.v2",
            "reporting": "descriptive_no_accept_reject_verdict",
            "model_key": model_key,
            "study_sha256": sha256_file(study_path),
            "data_lock_sha256": sha256_file(data_lock_path),
            "fit_probe_sha256": sha256_file(fit_probe_dir / "fit_probe.json"),
            "directions_sha256": sha256_file(fit_probe_dir / "directions.safetensors"),
            "record_count": len(records),
            "record_contract": record_contract(records),
            "record_contract_sha256": record_contract_sha256(records),
            "chosen_estimator": estimator,
            "chosen_layers": layers,
            "condition_count": len(trials),
            "base": behavior_run_payload(base),
            "trials": trials,
            "runtime": build_runtime_manifest(
                repository=repository,
                stage="frontier",
                run_kind="scientific",
                model_key=model_key,
                model_spec=model_spec,
                model_class=loaded.model_class,
                model_fingerprint=loaded.model_fingerprint,
                tokenizer_fingerprint=loaded.tokenizer_fingerprint,
                layer_path=loaded.layer_path,
                study_path=study_path,
                data_lock_path=data_lock_path,
                accessed_splits=("evaluation",),
                stage_parameters={
                    "generation_batch_size": batch_size,
                    "maximum_binary_new_tokens": max_new_tokens,
                    "condition_count": len(conditions),
                    "generation_eos_token_ids": loaded.generation_eos_token_ids,
                },
                launch_identity=launch_identity,
            ),
        }
        atomic_write_json(output_dir / "frontier.json", payload)
        (output_dir / "frontier.checkpoint.json").unlink(missing_ok=True)
        finalize_artifact_stage(
            output_dir,
            payload_files=("frontier.json",),
            status={
                "complete": True,
                "stage": "complete",
                "run_kind": "scientific",
                "model_key": model_key,
                "condition_count": len(trials),
            },
        )
        return payload
    finally:
        unload_study_model(loaded)
