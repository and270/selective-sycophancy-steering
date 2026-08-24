# SPDX-License-Identifier: AGPL-3.0-or-later

"""Executable fit/probe stage for one preregistered model arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .artifacts import (
    atomic_write_json,
    build_runtime_manifest,
    capture_scientific_launch_identity,
    finalize_artifact_stage,
    tensor_sha256,
    verify_loaded_study_identity,
)
from .config import load_study_config
from .data import (
    record_contract,
    record_contract_sha256,
    sha256_file,
    validate_materialized_data,
)
from .directions import (
    compute_completion_contrast,
    compute_observed_prompt_direction,
)
from .fit_probe import (
    DirectionObservations,
    collect_direction_observations,
    completion_fit_status,
    completion_probe_result,
    observed_probe_result,
    observed_valid_data,
)
from .loading import load_study_model, unload_study_model
from .probe import select_estimator_layers


def _serializable_observations(observations: DirectionObservations) -> dict[str, Any]:
    examples: list[dict[str, Any]] = [
        {**descriptor, "followup": answer}
        for descriptor, answer in zip(
            observations.descriptors,
            observations.followup_answers,
            strict=True,
        )
    ]
    return {
        "baseline_answers": observations.baseline_answers,
        "examples": examples,
        "n_examples": len(examples),
        "n_invalid_followups": sum(
            example["followup"]["parsed"] is None for example in examples
        ),
    }


def _tensor_artifact_metadata(
    path: Path, tensors: dict[str, torch.Tensor]
) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
            for name, tensor in tensors.items()
        },
    }


def run_fit_probe_stage(
    *,
    repository: Path,
    study_path: Path,
    data_dir: Path,
    output_dir: Path,
    model_key: str,
    run_kind: str,
    limit: int | None,
    generation_batch_size: int | None = None,
    residual_batch_size: int | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Fit both estimators and select layers on a physically held-out probe."""

    if run_kind not in {
        "scientific",
        "executed_reproduction",
        "engineering_smoke",
    }:
        raise ValueError(
            "run_kind must be scientific, executed_reproduction, or engineering_smoke"
        )
    if run_kind == "scientific" and limit is not None:
        raise ValueError("Scientific fit/probe cannot use a record limit")
    if run_kind == "scientific" and (
        generation_batch_size is not None or residual_batch_size is not None
    ):
        raise ValueError("Scientific fit/probe cannot override frozen batch sizes")
    if run_kind == "executed_reproduction" and limit is not None:
        raise ValueError("Executed reproduction cannot use a record limit")
    if run_kind == "executed_reproduction" and (
        generation_batch_size is not None or residual_batch_size is not None
    ):
        raise ValueError(
            "Executed reproduction cannot override the recorded batch sizes"
        )
    if run_kind == "engineering_smoke" and (limit is None or limit <= 0):
        raise ValueError("Engineering smoke requires a positive record limit")
    if output_dir.exists():
        raise FileExistsError(f"Fit/probe output already exists: {output_dir}")

    launch_identity = (
        capture_scientific_launch_identity(repository, study_path)
        if run_kind == "scientific"
        else None
    )
    study = load_study_config(study_path, require_frozen=run_kind == "scientific")
    if launch_identity is not None:
        verify_loaded_study_identity(launch_identity, study)
    if model_key not in study["models"]:
        raise ValueError(f"Unknown model key: {model_key}")
    model_spec = study["models"][model_key]
    data_lock_path = repository / study["data"]["lock"]
    records = validate_materialized_data(
        data_dir,
        data_lock_path,
        allowed_splits=("direction_fit", "direction_probe"),
        expected_lock_sha256=(
            str(launch_identity["data_lock_sha256"])
            if launch_identity is not None
            else None
        ),
    )
    fit_records = records["direction_fit"]
    probe_records = records["direction_probe"]
    if limit is not None:
        fit_records = fit_records[:limit]
        probe_records = probe_records[:limit]

    generation_batch = generation_batch_size or model_spec.get(
        "binary_generation_batch_size"
    )
    residual_batch = residual_batch_size or model_spec.get("residual_batch_size")
    if not isinstance(generation_batch, int) or generation_batch <= 0:
        raise ValueError("No valid generation batch size was supplied")
    if not isinstance(residual_batch, int) or residual_batch <= 0:
        raise ValueError("No valid residual batch size was supplied")

    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        output_dir / "status.json",
        {
            "complete": False,
            "stage": "initializing",
            "run_kind": run_kind,
            "model_key": model_key,
        },
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
        contract = study["prompt_contract"]
        chat_kwargs = model_spec["chat_template_kwargs"]
        max_new_tokens = int(study["runtime"]["maximum_binary_new_tokens"])
        atomic_write_json(
            output_dir / "status.json",
            {
                "complete": False,
                "stage": "direction_fit",
                "run_kind": run_kind,
                "model_key": model_key,
            },
        )
        fit = collect_direction_observations(
            loaded,
            fit_records,
            contract,
            chat_template_kwargs=chat_kwargs,
            generation_batch_size=generation_batch,
            residual_batch_size=residual_batch,
            max_new_tokens=max_new_tokens,
        )
        completion_spec = study["direction_estimation"]["completion_contrast"]
        source_datasets = tuple(completion_spec["source_datasets"])
        minimum_fit_count = (
            int(completion_spec["minimum_fit_eligible_records_per_source_option"])
            if run_kind == "scientific"
            else 0
        )
        completion_status = completion_fit_status(
            fit,
            source_datasets=source_datasets,
            minimum_records_per_source_option=minimum_fit_count,
        )
        completion_fit_eligible = bool(completion_status["eligible"])
        directions: dict[str, torch.Tensor] = {}
        if completion_fit_eligible:
            directions["completion_contrast"] = compute_completion_contrast(
                fit.caving_residuals, fit.resisting_residuals
            )
        estimator_status: dict[str, dict[str, Any]] = {
            "completion_contrast": completion_status
        }
        observed_spec = study["direction_estimation"]["observed_prompt_state"]
        try:
            residuals, labels, modes = observed_valid_data(fit)
            observed, counts = compute_observed_prompt_direction(
                residuals,
                labels,
                modes,
                expected_modes=tuple(contract["pressure_modes"]),
                minimum_overall=(
                    int(observed_spec["minimum_fit_class_count_overall"])
                    if run_kind == "scientific"
                    else 1
                ),
                minimum_per_mode=(
                    int(observed_spec["minimum_fit_class_count_per_mode"])
                    if run_kind == "scientific"
                    else 0
                ),
            )
            directions["observed_prompt_state"] = observed
            estimator_status["observed_prompt_state"] = {
                "eligible": True,
                "fit_class_counts": counts,
            }
        except ValueError as error:
            estimator_status["observed_prompt_state"] = {
                "eligible": False,
                "reason": str(error),
            }

        direction_path = output_dir / "directions.safetensors"
        save_file(
            {name: tensor.contiguous() for name, tensor in directions.items()},
            direction_path,
        )
        atomic_write_json(
            output_dir / "status.json",
            {
                "complete": False,
                "stage": "direction_probe",
                "run_kind": run_kind,
                "model_key": model_key,
            },
        )
        probe = collect_direction_observations(
            loaded,
            probe_records,
            contract,
            chat_template_kwargs=chat_kwargs,
            generation_batch_size=generation_batch,
            residual_batch_size=residual_batch,
            max_new_tokens=max_new_tokens,
        )
        observation_tensors = {
            "fit_prompt_residuals": fit.prompt_residuals.contiguous(),
            "fit_caving_residuals": fit.caving_residuals.contiguous(),
            "fit_resisting_residuals": fit.resisting_residuals.contiguous(),
            "probe_prompt_residuals": probe.prompt_residuals.contiguous(),
            "probe_caving_residuals": probe.caving_residuals.contiguous(),
            "probe_resisting_residuals": probe.resisting_residuals.contiguous(),
        }
        observations_path = output_dir / "observations.safetensors"
        save_file(observation_tensors, observations_path)
        control_count = (
            int(study["layer_selection"]["random_direction_controls_per_layer"])
            if run_kind == "scientific"
            else 5
        )
        seed = int(study["layer_selection"]["random_seed"])
        modes_tuple = tuple(contract["pressure_modes"])
        probe_results: dict[str, dict[str, Any]] = {}
        if completion_fit_eligible:
            probe_results["completion_contrast"] = completion_probe_result(
                probe,
                directions["completion_contrast"],
                modes=modes_tuple,
                source_datasets=source_datasets,
                random_controls=control_count,
                random_seed=seed,
            )
        if "observed_prompt_state" in directions:
            try:
                probe_results["observed_prompt_state"] = observed_probe_result(
                    probe,
                    directions["observed_prompt_state"],
                    modes=modes_tuple,
                    random_controls=control_count,
                    random_seed=seed + 1,
                )
            except ValueError as error:
                estimator_status["observed_prompt_state"]["probe_error"] = str(error)
        selection = select_estimator_layers(
            probe_results,
            modes=modes_tuple,
            policy=study["layer_selection"],
        )
        payload: dict[str, Any] = {
            "schema_version": "selective_sycophancy_fit_probe.v3",
            "run_kind": run_kind,
            "scientific_outputs_allowed": run_kind == "scientific",
            "model_key": model_key,
            "study_sha256": sha256_file(study_path),
            "data_lock_sha256": sha256_file(data_lock_path),
            "fit_record_count": len(fit_records),
            "probe_record_count": len(probe_records),
            "record_contracts": {
                "direction_fit": record_contract(fit_records),
                "direction_probe": record_contract(probe_records),
            },
            "record_contract_sha256": {
                "direction_fit": record_contract_sha256(fit_records),
                "direction_probe": record_contract_sha256(probe_records),
            },
            "observation_artifact": _tensor_artifact_metadata(
                observations_path, observation_tensors
            ),
            "direction_artifact": {
                "path": direction_path.name,
                "sha256": sha256_file(direction_path),
                "tensors": {
                    name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "sha256": tensor_sha256(tensor),
                        "layer_norms": torch.linalg.vector_norm(tensor, dim=1).tolist(),
                    }
                    for name, tensor in directions.items()
                },
            },
            "fit": _serializable_observations(fit),
            "probe": _serializable_observations(probe),
            "estimator_status": estimator_status,
            "probe_results": probe_results,
            "layer_selection": selection,
            "runtime": build_runtime_manifest(
                repository=repository,
                stage="fit_probe",
                run_kind=run_kind,
                model_key=model_key,
                model_spec=model_spec,
                model_class=loaded.model_class,
                model_fingerprint=loaded.model_fingerprint,
                tokenizer_fingerprint=loaded.tokenizer_fingerprint,
                layer_path=loaded.layer_path,
                study_path=study_path,
                data_lock_path=data_lock_path,
                accessed_splits=("direction_fit", "direction_probe"),
                stage_parameters={
                    "generation_batch_size": generation_batch,
                    "residual_batch_size": residual_batch,
                    "maximum_binary_new_tokens": max_new_tokens,
                    "random_direction_controls": control_count,
                },
                launch_identity=launch_identity,
            ),
        }
        atomic_write_json(output_dir / "fit_probe.json", payload)
        finalize_artifact_stage(
            output_dir,
            payload_files=(
                "fit_probe.json",
                "directions.safetensors",
                "observations.safetensors",
            ),
            status={
                "complete": True,
                "stage": "complete",
                "run_kind": run_kind,
                "model_key": model_key,
                "chosen_estimator": selection["chosen_estimator"],
                "chosen_layers": selection["chosen_layers"],
            },
        )
        return payload
    finally:
        unload_study_model(loaded)
