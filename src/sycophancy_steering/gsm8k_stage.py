# SPDX-License-Identifier: AGPL-3.0-or-later

"""Paired low-cost evaluation on a frozen 256-item GSM8K sample."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as parquet
import torch

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
from .config import load_study_config
from .data import sha256_file
from .frontier_stage import (
    _verify_fit_artifact,
    verify_frontier_artifact,
    verify_loaded_fingerprint,
)
from .gsm8k import (
    GSM8KHarnessContract,
    load_pinned_harness_contract,
    score_response,
    select_sample,
    wilson_interval,
)
from .hooks import SteeringAudit, steer_transformer_layer
from .inference import render_chat_texts
from .loading import load_study_model, unload_study_model
from .metrics import paired_cluster_bootstrap_mean_delta


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sample_identity(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        ("\n".join(str(row["sample_sha256"]) for row in rows) + "\n").encode()
    ).hexdigest()


def _generate_responses(
    loaded: Any,
    prompts: list[str],
    *,
    chat_template_kwargs: dict[str, Any],
    maximum_new_tokens: int,
    stop_strings: list[str],
    steering: tuple[torch.nn.Module, torch.nn.Module, torch.Tensor, float] | None,
) -> tuple[list[dict[str, str]], dict[str, int] | None]:
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    rendered = render_chat_texts(
        loaded.tokenizer,
        chats,
        chat_template_kwargs=chat_template_kwargs,
    )
    context = (
        steer_transformer_layer(
            steering[0], steering[1], steering[2], alpha=steering[3]
        )
        if steering is not None
        else nullcontext(None)
    )
    responses: list[dict[str, str]] = []
    with context as audit:
        for text in rendered:
            inputs = loaded.tokenizer(
                text,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(loaded.device)
            input_length = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                output = loaded.model.generate(
                    **inputs,
                    max_new_tokens=maximum_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=loaded.tokenizer.pad_token_id,
                    eos_token_id=loaded.generation_eos_token_ids,
                    stop_strings=stop_strings,
                    tokenizer=loaded.tokenizer,
                )
            if not isinstance(output, torch.Tensor) or output.shape[0] != 1:
                raise RuntimeError("GSM8K generation returned an invalid shape")
            response = loaded.tokenizer.decode(
                output[0, input_length:], skip_special_tokens=True
            )
            responses.append(
                {
                    "text": str(response),
                    "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    audit_payload = asdict(audit) if isinstance(audit, SteeringAudit) else None
    if steering is not None and (
        audit_payload is None
        or audit_payload["calls"] <= 0
        or audit_payload["modified_batch_rows"] <= 0
    ):
        raise RuntimeError("GSM8K steering hook was installed but did not execute")
    return responses, audit_payload


def _score_condition(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str]],
    *,
    harness: GSM8KHarnessContract,
) -> dict[str, Any]:
    if len(rows) != len(responses):
        raise ValueError("GSM8K response count mismatch")
    examples: list[dict[str, Any]] = []
    strict_correct = 0
    flexible_correct = 0
    for doc_id, (row, response) in enumerate(zip(rows, responses, strict=True)):
        score = score_response(response["text"], str(row["answer"]), harness=harness)
        strict_correct += int(score["strict_correct"])
        flexible_correct += int(score["flexible_correct"])
        examples.append(
            {
                "doc_id": doc_id,
                "source_index": int(row["source_index"]),
                "sample_sha256": row["sample_sha256"],
                "question_sha256": hashlib.sha256(
                    str(row["question"]).encode()
                ).hexdigest(),
                "prompt_sha256": response["prompt_sha256"],
                "response": response["text"],
                **score,
            }
        )
    total = len(rows)
    return {
        "record_count": total,
        "strict_correct_count": strict_correct,
        "strict_sampled_accuracy": strict_correct / total,
        "strict_wilson_95_ci": wilson_interval(
            correct=strict_correct, total=total, confidence=0.95
        ),
        "flexible_correct_count": flexible_correct,
        "flexible_sampled_accuracy": flexible_correct / total,
        "flexible_wilson_95_ci": wilson_interval(
            correct=flexible_correct, total=total, confidence=0.95
        ),
        "examples": examples,
    }


def _validate_scored_condition(
    scores: dict[str, Any], *, expected_records: int
) -> None:
    try:
        examples = scores["examples"]
        if (
            expected_records <= 0
            or scores.get("record_count") != expected_records
            or not isinstance(examples, list)
            or len(examples) != expected_records
            or any(not isinstance(example, dict) for example in examples)
        ):
            raise ValueError
        typed_examples = cast(list[dict[str, Any]], examples)
        strict_correct = 0
        flexible_correct = 0
        for index, example in enumerate(typed_examples):
            source_index = example.get("source_index")
            reference = example.get("reference")
            strict_prediction = example.get("strict_prediction")
            flexible_prediction = example.get("flexible_prediction")
            expected_strict = strict_prediction == reference
            expected_flexible = flexible_prediction == reference
            if (
                example.get("doc_id") != index
                or not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or source_index < 0
                or any(
                    not _valid_sha256(example.get(field))
                    for field in (
                        "sample_sha256",
                        "question_sha256",
                        "prompt_sha256",
                    )
                )
                or not isinstance(example.get("response"), str)
                or not isinstance(reference, str)
                or not (strict_prediction is None or isinstance(strict_prediction, str))
                or not (
                    flexible_prediction is None or isinstance(flexible_prediction, str)
                )
                or example.get("strict_correct") is not expected_strict
                or example.get("flexible_correct") is not expected_flexible
            ):
                raise ValueError
            strict_correct += int(expected_strict)
            flexible_correct += int(expected_flexible)
        expected = {
            "record_count": expected_records,
            "strict_correct_count": strict_correct,
            "strict_sampled_accuracy": strict_correct / expected_records,
            "strict_wilson_95_ci": wilson_interval(
                correct=strict_correct, total=expected_records, confidence=0.95
            ),
            "flexible_correct_count": flexible_correct,
            "flexible_sampled_accuracy": flexible_correct / expected_records,
            "flexible_wilson_95_ci": wilson_interval(
                correct=flexible_correct, total=expected_records, confidence=0.95
            ),
            "examples": examples,
        }
        if scores != expected:
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GSM8K scores differ from persisted examples") from error


def verify_gsm8k_artifact(
    directory: Path,
    *,
    model_key: str,
    study_path: Path,
    data_lock_path: Path,
    fit_probe_dir: Path,
    frontier_dir: Path,
    fit_result: dict[str, Any],
    frontier: dict[str, Any],
) -> dict[str, Any]:
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("stage") != "complete"
        or status.get("run_kind") != "scientific"
        or status.get("model_key") != model_key
    ):
        raise ValueError("GSM8K status is not complete")
    verify_artifact_manifest(
        directory,
        status=status,
        expected_files=("sampled_gsm8k.json",),
    )
    payload = json.loads((directory / "sampled_gsm8k.json").read_text(encoding="utf-8"))
    study = load_study_config(study_path, require_frozen=True)
    spec = study["sampled_gsm8k"]
    runtime = payload.get("runtime")
    sample_records = int(spec["sample_records"])
    if (
        payload.get("schema_version") != "selective_sycophancy_sampled_gsm8k.v2"
        or payload.get("reporting") != "sampled_accuracy_not_full_gsm8k"
        or payload.get("model_key") != model_key
        or payload.get("study_sha256") != sha256_file(study_path)
        or payload.get("fit_probe_sha256")
        != sha256_file(fit_probe_dir / "fit_probe.json")
        or payload.get("frontier_sha256") != sha256_file(frontier_dir / "frontier.json")
        or payload.get("dataset_parquet_sha256") != spec["parquet_sha256"]
        or payload.get("population_records") != int(spec["population_records"])
        or payload.get("sample_records") != sample_records
        or not isinstance(runtime, dict)
        or runtime.get("stage") != "sampled_gsm8k"
        or runtime.get("run_kind") != "scientific"
        or runtime.get("model_key") != model_key
        or runtime.get("repository_dirty") is not False
        or runtime.get("study_sha256") != sha256_file(study_path)
        or runtime.get("data_lock_sha256") != sha256_file(data_lock_path)
        or runtime.get("accessed_splits") != []
        or runtime.get("generation_eos_token_ids")
        != study["models"][model_key]["generation_eos_token_ids"]
    ):
        raise ValueError("GSM8K artifact contract is invalid")
    verify_parent_launch_identity(
        runtime["launch_identity"], fit_result, label="fit/probe"
    )
    verify_parent_launch_identity(
        runtime["launch_identity"], frontier, label="frontier"
    )
    if runtime.get("model_fingerprint") != fit_result.get("runtime", {}).get(
        "model_fingerprint"
    ) or runtime.get("tokenizer_fingerprint") != fit_result.get("runtime", {}).get(
        "tokenizer_fingerprint"
    ):
        raise ValueError("GSM8K model identity differs from fit/probe")
    harness = load_pinned_harness_contract(spec)
    harness_contract = payload.get("harness_contract")
    if (
        not isinstance(harness_contract, dict)
        or harness_contract.get("lm_eval_version") != harness.version
        or harness_contract.get("task_name") != harness.task_name
        or harness_contract.get("task_yaml_path") != harness.task_yaml_path
        or harness_contract.get("task_yaml_sha256") != harness.task_yaml_sha256
        or harness_contract.get("exact_match_options") != harness.exact_match_options
    ):
        raise ValueError("GSM8K harness contract is invalid")
    base = payload.get("base")
    if not isinstance(base, dict):
        raise ValueError("GSM8K base scores are missing")
    _validate_scored_condition(base, expected_records=sample_records)
    expected_sample_identity = hashlib.sha256(
        (
            "\n".join(str(item["sample_sha256"]) for item in base["examples"]) + "\n"
        ).encode()
    ).hexdigest()
    if (
        payload.get("ordered_sample_identity_sha256") != expected_sample_identity
        or expected_sample_identity != spec["ordered_sample_identity_sha256"]
    ):
        raise ValueError("GSM8K sample identity differs from examples or study")
    selection = fit_result["layer_selection"]
    chosen_estimator = selection["chosen_estimator"]
    chosen_layers = selection["chosen_layers"]
    condition = payload.get("condition")
    comparison = payload.get("comparison")
    if chosen_estimator is None or not chosen_layers:
        if condition is not None or comparison is not None:
            raise ValueError("GSM8K unexpectedly contains a steering condition")
        return payload
    primary_matches = [
        trial
        for trial in frontier["trials"]
        if trial["estimator"] == chosen_estimator
        and int(trial["zero_based_layer"]) == int(chosen_layers[0])
        and float(trial["alpha"]) == -2.0
    ]
    if len(primary_matches) != 1 or not isinstance(condition, dict):
        raise ValueError("GSM8K primary condition is missing")
    scores = condition.get("scores")
    if (
        condition.get("estimator") != chosen_estimator
        or condition.get("zero_based_layer") != int(chosen_layers[0])
        or float(condition.get("alpha")) != -2.0
        or condition.get("direction_tensor_sha256")
        != primary_matches[0]["direction_tensor_sha256"]
        or not isinstance(scores, dict)
    ):
        raise ValueError("GSM8K primary condition differs from frontier")
    _validate_scored_condition(scores, expected_records=sample_records)
    if [item["sample_sha256"] for item in scores["examples"]] != [
        item["sample_sha256"] for item in base["examples"]
    ]:
        raise ValueError("GSM8K condition sample differs from base")
    audit = condition.get("hook_audit")
    if (
        not isinstance(audit, dict)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in audit.values()
        )
        or int(audit.get("calls", 0)) <= 0
        or int(audit.get("modified_batch_rows", 0)) <= 0
    ):
        raise ValueError("GSM8K hook audit is invalid")
    ids = [str(index) for index in range(sample_records)]
    bootstrap: dict[str, Any] = {}
    for offset, metric in enumerate(("strict_correct", "flexible_correct")):
        base_values = {
            item_id: float(base["examples"][int(item_id)][metric]) for item_id in ids
        }
        condition_values = {
            item_id: float(scores["examples"][int(item_id)][metric]) for item_id in ids
        }
        bootstrap[metric] = paired_cluster_bootstrap_mean_delta(
            base_values,
            condition_values,
            iterations=10_000,
            seed=20260805 + offset,
            confidence=0.95,
        )
    expected_comparison = {
        "strict_sampled_accuracy_condition_minus_base": (
            scores["strict_sampled_accuracy"] - base["strict_sampled_accuracy"]
        ),
        "flexible_sampled_accuracy_condition_minus_base": (
            scores["flexible_sampled_accuracy"] - base["flexible_sampled_accuracy"]
        ),
        "paired_sampled_accuracy_intervals": bootstrap,
    }
    if comparison != expected_comparison:
        raise ValueError("GSM8K comparison differs from persisted examples")
    return payload


def run_gsm8k_stage(
    *,
    repository: Path,
    study_path: Path,
    fit_probe_dir: Path,
    frontier_dir: Path,
    gsm8k_path: Path,
    output_dir: Path,
    model_key: str,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Run base and the strongest fixed steering dose on sampled GSM8K."""

    if output_dir.exists():
        raise FileExistsError(f"GSM8K output already exists: {output_dir}")
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
    frontier = verify_frontier_artifact(
        frontier_dir,
        model_key=model_key,
        study_path=study_path,
        data_lock_path=data_lock_path,
        fit_probe_dir=fit_probe_dir,
        fit_result=fit_result,
        directions=directions,
    )
    verify_parent_launch_identity(launch_identity, fit_result, label="fit/probe")
    verify_parent_launch_identity(launch_identity, frontier, label="frontier")
    spec = study["sampled_gsm8k"]
    harness = load_pinned_harness_contract(spec)
    if sha256_file(gsm8k_path) != spec["parquet_sha256"]:
        raise ValueError("GSM8K parquet hash mismatch")
    population = parquet.read_table(gsm8k_path).to_pylist()
    if len(population) != int(spec["population_records"]):
        raise ValueError("GSM8K population row count mismatch")
    rows = select_sample(population, count=int(spec["sample_records"]))
    if _sample_identity(rows) != spec["ordered_sample_identity_sha256"]:
        raise ValueError("GSM8K ordered sample identity mismatch")

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
        verify_loaded_fingerprint(loaded, frontier, label="frontier")
        prompts = [
            str(spec["prompt_template"]).format(question=str(row["question"]))
            for row in rows
        ]
        decoding = spec["decoding"]
        base_responses, _ = _generate_responses(
            loaded,
            prompts,
            chat_template_kwargs=model_spec["chat_template_kwargs"],
            maximum_new_tokens=int(decoding["maximum_new_tokens"]),
            stop_strings=list(spec["stop_strings"]),
            steering=None,
        )
        base = _score_condition(rows, base_responses, harness=harness)
        atomic_write_json(
            output_dir / "gsm8k.checkpoint.json",
            {"model_key": model_key, "base": base, "condition": None},
        )

        condition: dict[str, Any] | None = None
        comparison: dict[str, Any] | None = None
        estimator = fit_result["layer_selection"]["chosen_estimator"]
        selected_layers = fit_result["layer_selection"]["chosen_layers"]
        if estimator is not None and selected_layers:
            layer = int(selected_layers[0])
            alpha = -2.0
            matched = [
                trial
                for trial in frontier["trials"]
                if trial["estimator"] == estimator
                and int(trial["zero_based_layer"]) == layer
                and float(trial["alpha"]) == alpha
            ]
            if len(matched) != 1:
                raise ValueError("Frontier lacks the preregistered GSM8K condition")
            direction = directions[str(estimator)][layer]
            if tensor_sha256(direction) != matched[0]["direction_tensor_sha256"]:
                raise ValueError("GSM8K direction differs from frontier")
            atomic_write_json(
                output_dir / "status.json",
                {
                    "complete": False,
                    "stage": "steered",
                    "model_key": model_key,
                    "estimator": estimator,
                    "layer": layer,
                    "alpha": alpha,
                },
            )
            responses, audit = _generate_responses(
                loaded,
                prompts,
                chat_template_kwargs=model_spec["chat_template_kwargs"],
                maximum_new_tokens=int(decoding["maximum_new_tokens"]),
                stop_strings=list(spec["stop_strings"]),
                steering=(
                    loaded.text_model,
                    loaded.layers[layer],
                    direction,
                    alpha,
                ),
            )
            condition_scores = _score_condition(rows, responses, harness=harness)
            condition = {
                "estimator": estimator,
                "zero_based_layer": layer,
                "alpha": alpha,
                "direction_tensor_sha256": tensor_sha256(direction),
                "scores": condition_scores,
                "hook_audit": audit,
            }
            ids = [str(index) for index in range(len(rows))]
            bootstrap: dict[str, Any] = {}
            for offset, metric in enumerate(("strict_correct", "flexible_correct")):
                base_values = {
                    item_id: float(base["examples"][int(item_id)][metric])
                    for item_id in ids
                }
                condition_values = {
                    item_id: float(condition_scores["examples"][int(item_id)][metric])
                    for item_id in ids
                }
                bootstrap[metric] = paired_cluster_bootstrap_mean_delta(
                    base_values,
                    condition_values,
                    iterations=10000,
                    seed=20260805 + offset,
                    confidence=0.95,
                )
            comparison = {
                "strict_sampled_accuracy_condition_minus_base": (
                    condition_scores["strict_sampled_accuracy"]
                    - base["strict_sampled_accuracy"]
                ),
                "flexible_sampled_accuracy_condition_minus_base": (
                    condition_scores["flexible_sampled_accuracy"]
                    - base["flexible_sampled_accuracy"]
                ),
                "paired_sampled_accuracy_intervals": bootstrap,
            }
            atomic_write_json(
                output_dir / "gsm8k.checkpoint.json",
                {
                    "model_key": model_key,
                    "base": base,
                    "condition": condition,
                    "comparison": comparison,
                },
            )

        payload: dict[str, Any] = {
            "schema_version": "selective_sycophancy_sampled_gsm8k.v2",
            "reporting": "sampled_accuracy_not_full_gsm8k",
            "model_key": model_key,
            "study_sha256": sha256_file(study_path),
            "fit_probe_sha256": sha256_file(fit_probe_dir / "fit_probe.json"),
            "frontier_sha256": sha256_file(frontier_dir / "frontier.json"),
            "dataset_parquet_sha256": sha256_file(gsm8k_path),
            "population_records": len(population),
            "sample_records": len(rows),
            "ordered_sample_identity_sha256": _sample_identity(rows),
            "harness_contract": {
                "lm_eval_version": harness.version,
                "task_name": harness.task_name,
                "task_yaml_path": harness.task_yaml_path,
                "task_yaml_sha256": harness.task_yaml_sha256,
                "exact_match_options": harness.exact_match_options,
            },
            "base": base,
            "condition": condition,
            "comparison": comparison,
            "runtime": build_runtime_manifest(
                repository=repository,
                stage="sampled_gsm8k",
                run_kind="scientific",
                model_key=model_key,
                model_spec=model_spec,
                model_class=loaded.model_class,
                model_fingerprint=loaded.model_fingerprint,
                tokenizer_fingerprint=loaded.tokenizer_fingerprint,
                layer_path=loaded.layer_path,
                study_path=study_path,
                data_lock_path=data_lock_path,
                accessed_splits=(),
                stage_parameters={
                    "sample_records": len(rows),
                    "maximum_new_tokens": int(spec["decoding"]["maximum_new_tokens"]),
                    "batch_size": int(spec["decoding"]["batch_size"]),
                    "generation_eos_token_ids": loaded.generation_eos_token_ids,
                    "lm_eval_version": harness.version,
                    "lm_eval_task_yaml_sha256": harness.task_yaml_sha256,
                    "steering_alpha": -2.0,
                },
                launch_identity=launch_identity,
            ),
        }
        atomic_write_json(output_dir / "sampled_gsm8k.json", payload)
        (output_dir / "gsm8k.checkpoint.json").unlink(missing_ok=True)
        finalize_artifact_stage(
            output_dir,
            payload_files=("sampled_gsm8k.json",),
            status={
                "complete": True,
                "stage": "complete",
                "run_kind": "scientific",
                "model_key": model_key,
                "has_steered_condition": condition is not None,
            },
        )
        return payload
    finally:
        unload_study_model(loaded)
