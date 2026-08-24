# SPDX-License-Identifier: AGPL-3.0-or-later

"""Neutral WikiText trajectory KL evaluation for every frontier point."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
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
from .hooks import SteeringAudit, steer_trajectory_positions
from .kl import distribution_metrics_from_logits, select_neutral_contexts
from .loading import load_study_model, unload_study_model


def _mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int]:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(values), size=len(values))
        samples[index] = float(np.mean(values[selected]))
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(samples, tail)),
        "upper": float(np.quantile(samples, 1.0 - tail)),
        "confidence": confidence,
        "iterations": iterations,
        "seed": seed,
        "n_contexts": len(values),
    }


def _macro_summary(
    context_means: list[float],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    values = np.asarray(context_means, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "mean_bootstrap_95_ci": _mean_interval(
            values,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
        ),
    }


def _micro_summary(token_values: list[float]) -> dict[str, float | int]:
    values = np.asarray(token_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Token-micro values must be a finite non-empty vector")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "n_tokens": len(values),
    }


def exploratory_kl_trial_summary(trial: dict[str, Any]) -> dict[str, Any]:
    """Return the compact console summary for an exploratory KL trial."""

    try:
        condition = trial["condition"]
        return {
            "alpha": trial["alpha"],
            "forward_kl_nats": condition["token_micro"]["forward_kl_nats"]["mean"],
            "forward_kl_95_ci": condition["prompt_macro"]["forward_kl_nats"][
                "mean_bootstrap_95_ci"
            ],
            "js_nats": condition["token_micro"]["jensen_shannon_nats"]["mean"],
            "top1_agreement": condition["token_micro"]["top1_agreement"]["mean"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("Exploratory KL trial summary schema is invalid") from error


def _semantic_numbers_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return actual == expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _semantic_numbers_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _semantic_numbers_equal(actual[key], value)
                for key, value in expected.items()
            )
        )
    return actual == expected


def _validate_kl_trial_statistics(
    trial: dict[str, Any],
    *,
    contexts: list[dict[str, Any]],
    zero_alpha_tolerance: float,
) -> None:
    try:
        per_context = trial["per_context"]
        if not isinstance(per_context, list) or len(per_context) != len(contexts):
            raise ValueError
        all_kl: list[float] = []
        all_js: list[float] = []
        all_agreement: list[float] = []
        context_kl: list[float] = []
        context_js: list[float] = []
        context_agreement: list[float] = []
        for expected_context, item in zip(contexts, per_context, strict=True):
            if not isinstance(item, dict) or (
                item.get("row_index") != expected_context["row_index"]
                or item.get("context_sha256") != expected_context["sha256"]
            ):
                raise ValueError
            kl_values = item.get("forward_kl_nats_by_token")
            js_values = item.get("jensen_shannon_nats_by_token")
            agreement_values = item.get("top1_agreement_by_token")
            if (
                not isinstance(kl_values, list)
                or not isinstance(js_values, list)
                or not isinstance(agreement_values, list)
                or not kl_values
                or len(kl_values) != len(js_values)
                or len(kl_values) != len(agreement_values)
                or item.get("token_count") != len(kl_values)
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < -1e-12
                    for value in [*kl_values, *js_values]
                )
                or any(not isinstance(value, bool) for value in agreement_values)
            ):
                raise ValueError
            numeric_kl = [float(cast(int | float, value)) for value in kl_values]
            numeric_js = [float(cast(int | float, value)) for value in js_values]
            numeric_agreement = [float(cast(bool, value)) for value in agreement_values]
            means = {
                "forward_kl_nats_mean": float(np.mean(numeric_kl)),
                "jensen_shannon_nats_mean": float(np.mean(numeric_js)),
                "top1_agreement_mean": float(np.mean(numeric_agreement)),
            }
            if any(
                not _semantic_numbers_equal(item.get(key), value)
                for key, value in means.items()
            ):
                raise ValueError
            all_kl.extend(numeric_kl)
            all_js.extend(numeric_js)
            all_agreement.extend(numeric_agreement)
            context_kl.append(means["forward_kl_nats_mean"])
            context_js.append(means["jensen_shannon_nats_mean"])
            context_agreement.append(means["top1_agreement_mean"])
        expected_prompt_macro = {
            "forward_kl_nats": _macro_summary(
                context_kl, iterations=10_000, seed=20260805, confidence=0.95
            ),
            "jensen_shannon_nats": _macro_summary(
                context_js, iterations=10_000, seed=20260806, confidence=0.95
            ),
            "top1_agreement": _macro_summary(
                context_agreement, iterations=10_000, seed=20260807, confidence=0.95
            ),
        }
        expected_token_micro = {
            "forward_kl_nats": _micro_summary(all_kl),
            "jensen_shannon_nats": _micro_summary(all_js),
            "top1_agreement": _micro_summary(all_agreement),
        }
        expected_hook_audit = {
            "calls": len(contexts),
            "prefill_calls": len(contexts),
            "decode_calls": 0,
            "modified_batch_rows": len(contexts),
            "modified_token_positions": len(all_kl),
        }
        maximum_difference = trial.get("maximum_absolute_logit_difference")
        alpha = trial.get("alpha")
        if (
            trial.get("token_count") != len(all_kl)
            or trial.get("hook_audit") != expected_hook_audit
            or not _semantic_numbers_equal(
                trial.get("prompt_macro"), expected_prompt_macro
            )
            or not _semantic_numbers_equal(
                trial.get("token_micro"), expected_token_micro
            )
            or not isinstance(maximum_difference, (int, float))
            or isinstance(maximum_difference, bool)
            or not math.isfinite(float(maximum_difference))
            or float(maximum_difference) < 0.0
            or not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or (
                float(cast(int | float, alpha)) == 0.0
                and float(maximum_difference) > zero_alpha_tolerance
            )
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("KL trial semantic verification failed") from error


def _tokenize_context(
    loaded: Any, text: str, *, maximum_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = loaded.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_tokens,
        return_token_type_ids=False,
    ).to(loaded.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or attention_mask.shape != input_ids.shape
        or int(attention_mask.sum().item()) != input_ids.shape[1]
    ):
        raise ValueError("KL raw context tokenization must be one unpadded sequence")
    return input_ids, attention_mask


def _token_ids_sha256(token_ids: torch.Tensor) -> str:
    payload = (
        token_ids.detach().to(dtype=torch.int64, device="cpu").contiguous().numpy()
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _generate_base_trajectories(
    loaded: Any,
    contexts: list[dict[str, Any]],
    *,
    context_maximum_tokens: int,
    continuation_maximum_tokens: int,
) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    eos_ids = set(loaded.generation_eos_token_ids)
    for context in contexts:
        input_ids, attention_mask = _tokenize_context(
            loaded, context["text"], maximum_tokens=context_maximum_tokens
        )
        prompt_length = int(input_ids.shape[1])
        with torch.inference_mode():
            generated = loaded.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=continuation_maximum_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.generation_eos_token_ids,
            )
        continuation = generated[0, prompt_length:].detach().to(device="cpu")
        if continuation.ndim != 1 or continuation.numel() == 0:
            raise RuntimeError("KL base generation produced no continuation tokens")
        token_ids = [int(value) for value in continuation.tolist()]
        trajectories.append(
            {
                "row_index": int(context["row_index"]),
                "context_sha256": str(context["sha256"]),
                "prompt_token_count": prompt_length,
                "prompt_token_ids_sha256": _token_ids_sha256(input_ids),
                "continuation_token_ids": token_ids,
                "continuation_token_count": len(token_ids),
                "terminated_on_frozen_eos": token_ids[-1] in eos_ids,
                "continuation_text": loaded.tokenizer.decode(
                    token_ids, skip_special_tokens=False
                ),
            }
        )
    return trajectories


def _trajectory_logits(
    loaded: Any,
    text: str,
    continuation_token_ids: list[int],
    *,
    context_maximum_tokens: int,
    steering: tuple[torch.nn.Module, torch.Tensor, float] | None,
) -> tuple[torch.Tensor, dict[str, int] | None]:
    prompt_ids, _ = _tokenize_context(
        loaded, text, maximum_tokens=context_maximum_tokens
    )
    continuation = torch.tensor(
        continuation_token_ids, dtype=torch.long, device=loaded.device
    )
    if continuation.ndim != 1 or continuation.numel() == 0:
        raise ValueError("KL continuation must contain at least one token")
    replay_prefix = continuation[:-1].unsqueeze(0)
    full_input_ids = torch.cat([prompt_ids, replay_prefix], dim=1)
    attention_mask = torch.ones_like(full_input_ids)
    prompt_length = int(prompt_ids.shape[1])
    prediction_positions = torch.arange(
        prompt_length - 1,
        prompt_length - 1 + continuation.numel(),
        device=loaded.device,
    )
    positions_mask = torch.zeros_like(full_input_ids, dtype=torch.bool)
    positions_mask[0, prediction_positions] = True
    hook_context = (
        steer_trajectory_positions(
            steering[0],
            steering[1],
            alpha=steering[2],
            positions_mask=positions_mask,
        )
        if steering is not None
        else nullcontext(None)
    )
    with hook_context as audit, torch.inference_mode():
        outputs = loaded.model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    logits = outputs.logits
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Model did not return KL trajectory logits")
    selected = (
        logits[0, prediction_positions, :]
        .detach()
        .to(dtype=torch.float32, device="cpu")
    )
    if selected.shape[0] != continuation.numel() or not torch.isfinite(selected).all():
        raise RuntimeError("KL trajectory logits are invalid")
    audit_payload = asdict(audit) if isinstance(audit, SteeringAudit) else None
    if steering is not None and (
        audit_payload is None
        or audit_payload["calls"] != 1
        or audit_payload["modified_token_positions"] != continuation.numel()
    ):
        raise RuntimeError("KL trajectory steering audit differs from replay contract")
    return selected, audit_payload


def _add_audit(total: dict[str, int], audit: dict[str, int] | None) -> None:
    if audit is None:
        return
    for key, value in audit.items():
        total[key] = total.get(key, 0) + int(value)


def _evaluate_trial(
    loaded: Any,
    contexts: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    *,
    context_maximum_tokens: int,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    zero_alpha_tolerance: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    if len(contexts) != len(trajectories):
        raise ValueError("KL contexts and trajectories are not aligned")
    per_context: list[dict[str, Any]] = []
    all_kl: list[float] = []
    all_js: list[float] = []
    all_agreement: list[float] = []
    context_kl: list[float] = []
    context_js: list[float] = []
    context_agreement: list[float] = []
    maximum_logit_difference = 0.0
    audit_total: dict[str, int] = {}
    for context, trajectory in zip(contexts, trajectories, strict=True):
        if (
            int(context["row_index"]) != trajectory["row_index"]
            or str(context["sha256"]) != trajectory["context_sha256"]
        ):
            raise ValueError("KL context identity differs from frozen trajectory")
        base_logits, _ = _trajectory_logits(
            loaded,
            context["text"],
            trajectory["continuation_token_ids"],
            context_maximum_tokens=context_maximum_tokens,
            steering=None,
        )
        condition_logits, audit = _trajectory_logits(
            loaded,
            context["text"],
            trajectory["continuation_token_ids"],
            context_maximum_tokens=context_maximum_tokens,
            steering=(loaded.layers[layer], direction, alpha),
        )
        _add_audit(audit_total, audit)
        difference = float(torch.max(torch.abs(condition_logits - base_logits)).item())
        maximum_logit_difference = max(maximum_logit_difference, difference)
        metrics = distribution_metrics_from_logits(base_logits, condition_logits)
        kl_values = [float(value) for value in metrics["forward_kl"].tolist()]
        js_values = [float(value) for value in metrics["jensen_shannon"].tolist()]
        agreement_values = [
            float(value) for value in metrics["top1_agreement"].tolist()
        ]
        all_kl.extend(kl_values)
        all_js.extend(js_values)
        all_agreement.extend(agreement_values)
        context_kl.append(float(np.mean(kl_values)))
        context_js.append(float(np.mean(js_values)))
        context_agreement.append(float(np.mean(agreement_values)))
        per_context.append(
            {
                "row_index": trajectory["row_index"],
                "context_sha256": trajectory["context_sha256"],
                "token_count": len(kl_values),
                "forward_kl_nats_by_token": kl_values,
                "jensen_shannon_nats_by_token": js_values,
                "top1_agreement_by_token": [bool(value) for value in agreement_values],
                "forward_kl_nats_mean": context_kl[-1],
                "jensen_shannon_nats_mean": context_js[-1],
                "top1_agreement_mean": context_agreement[-1],
            }
        )
    if alpha == 0.0 and maximum_logit_difference > zero_alpha_tolerance:
        raise RuntimeError("Zero-alpha KL trajectory logits differ from base")
    return {
        "maximum_absolute_logit_difference": maximum_logit_difference,
        "token_count": len(all_kl),
        "prompt_macro": {
            "forward_kl_nats": _macro_summary(
                context_kl,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                confidence=confidence,
            ),
            "jensen_shannon_nats": _macro_summary(
                context_js,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + 1,
                confidence=confidence,
            ),
            "top1_agreement": _macro_summary(
                context_agreement,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + 2,
                confidence=confidence,
            ),
        },
        "token_micro": {
            "forward_kl_nats": _micro_summary(all_kl),
            "jensen_shannon_nats": _micro_summary(all_js),
            "top1_agreement": _micro_summary(all_agreement),
        },
        "per_context": per_context,
        "hook_audit": audit_total,
    }


def verify_kl_artifact(
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
        raise ValueError("KL status is not complete")
    verify_artifact_manifest(
        directory,
        status=status,
        expected_files=("neutral_trajectory_kl.json",),
    )
    payload = json.loads(
        (directory / "neutral_trajectory_kl.json").read_text(encoding="utf-8")
    )
    study = load_study_config(study_path, require_frozen=True)
    spec = study["neutral_kl"]
    contexts = payload.get("contexts")
    trajectories = payload.get("trajectories")
    trials = payload.get("trials")
    parent_trials = frontier.get("trials")
    runtime = payload.get("runtime")
    if (
        payload.get("schema_version") != "selective_sycophancy_neutral_trajectory_kl.v2"
        or payload.get("model_key") != model_key
        or payload.get("study_sha256") != sha256_file(study_path)
        or payload.get("fit_probe_sha256")
        != sha256_file(fit_probe_dir / "fit_probe.json")
        or payload.get("frontier_sha256") != sha256_file(frontier_dir / "frontier.json")
        or payload.get("directions_sha256")
        != sha256_file(fit_probe_dir / "directions.safetensors")
        or payload.get("wikitext_parquet_sha256") != spec["parquet_sha256"]
        or not isinstance(contexts, list)
        or any(not isinstance(context, dict) for context in contexts)
        or payload.get("context_count") != len(contexts)
        or len(contexts) != int(spec["contexts"])
        or not isinstance(trajectories, list)
        or any(not isinstance(trajectory, dict) for trajectory in trajectories)
        or len(trajectories) != len(contexts)
        or not isinstance(trials, list)
        or any(not isinstance(trial, dict) for trial in trials)
        or not isinstance(parent_trials, list)
        or any(not isinstance(trial, dict) for trial in parent_trials)
        or payload.get("condition_count") != len(trials)
        or len(trials) != len(parent_trials)
        or not isinstance(runtime, dict)
        or runtime.get("stage") != "neutral_trajectory_kl"
        or runtime.get("run_kind") != "scientific"
        or runtime.get("model_key") != model_key
        or runtime.get("repository_dirty") is not False
        or runtime.get("study_sha256") != sha256_file(study_path)
        or runtime.get("data_lock_sha256") != sha256_file(data_lock_path)
        or runtime.get("accessed_splits") != []
        or runtime.get("generation_eos_token_ids")
        != study["models"][model_key]["generation_eos_token_ids"]
    ):
        raise ValueError("KL artifact contract is invalid")
    typed_contexts = cast(list[dict[str, Any]], contexts)
    typed_trajectories = cast(list[dict[str, Any]], trajectories)
    typed_trials = cast(list[dict[str, Any]], trials)
    typed_parent_trials = cast(list[dict[str, Any]], parent_trials)
    ordered_context_sha256 = hashlib.sha256(
        (
            "\n".join(str(context["sha256"]) for context in typed_contexts) + "\n"
        ).encode()
    ).hexdigest()
    if ordered_context_sha256 != spec["ordered_context_sha256"]:
        raise ValueError("KL context identity differs from the frozen study")
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
        raise ValueError("KL model identity differs from fit/probe")
    contract = payload.get("trajectory_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("chat_template") is not False
        or contract.get("truncation_side") != "right"
        or contract.get("context_maximum_tokens") != int(spec["context_maximum_tokens"])
        or contract.get("continuation_maximum_tokens")
        != int(spec["continuation_maximum_tokens"])
        or contract.get("generation_eos_token_ids")
        != study["models"][model_key]["generation_eos_token_ids"]
        or contract.get("fixed_base_trajectory") is not True
        or contract.get("eos_prediction_included") is not True
    ):
        raise ValueError("KL trajectory contract is invalid")
    for context, trajectory in zip(typed_contexts, typed_trajectories, strict=True):
        row_index = context.get("row_index")
        context_sha256 = context.get("sha256")
        token_ids = trajectory.get("continuation_token_ids")
        if (
            set(context) != {"row_index", "sha256"}
            or not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or not isinstance(context_sha256, str)
            or len(context_sha256) != 64
            or trajectory.get("row_index") != row_index
            or trajectory.get("context_sha256") != context_sha256
            or not isinstance(token_ids, list)
            or not token_ids
            or len(token_ids) > int(spec["continuation_maximum_tokens"])
            or trajectory.get("continuation_token_count") != len(token_ids)
            or trajectory.get("terminated_on_frozen_eos")
            is not (
                token_ids[-1] in study["models"][model_key]["generation_eos_token_ids"]
            )
            or not isinstance(trajectory.get("continuation_text"), str)
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
        ):
            raise ValueError("KL trajectory primitives are invalid")
    for trial, parent_trial in zip(typed_trials, typed_parent_trials, strict=True):
        trial_alpha = trial.get("alpha")
        parent_alpha = parent_trial.get("alpha")
        if (
            trial.get("estimator") != parent_trial.get("estimator")
            or trial.get("zero_based_layer") != parent_trial.get("zero_based_layer")
            or not isinstance(trial_alpha, (int, float))
            or isinstance(trial_alpha, bool)
            or not isinstance(parent_alpha, (int, float))
            or isinstance(parent_alpha, bool)
            or float(trial_alpha) != float(parent_alpha)
            or trial.get("direction_tensor_sha256")
            != parent_trial.get("direction_tensor_sha256")
        ):
            raise ValueError("KL trial inventory differs from frontier")
        _validate_kl_trial_statistics(
            trial,
            contexts=typed_contexts,
            zero_alpha_tolerance=float(
                spec["zero_alpha_max_absolute_logit_difference"]
            ),
        )
        expected_token_counts = [
            len(cast(list[Any], trajectory["continuation_token_ids"]))
            for trajectory in typed_trajectories
        ]
        per_context = trial.get("per_context")
        if (
            not isinstance(per_context, list)
            or any(not isinstance(item, dict) for item in per_context)
            or [cast(dict[str, Any], item).get("token_count") for item in per_context]
            != expected_token_counts
        ):
            raise ValueError("KL trial token counts differ from trajectories")
    return payload


def run_kl_stage(
    *,
    repository: Path,
    study_path: Path,
    fit_probe_dir: Path,
    frontier_dir: Path,
    wikitext_path: Path,
    output_dir: Path,
    model_key: str,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Evaluate trajectory distribution shift for every behavioral condition."""

    if output_dir.exists():
        raise FileExistsError(f"KL output already exists: {output_dir}")
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
    kl_spec = study["neutral_kl"]
    if sha256_file(wikitext_path) != kl_spec["parquet_sha256"]:
        raise ValueError("WikiText parquet hash mismatch")
    table = parquet.read_table(wikitext_path, columns=["text"])
    rows = [str(value) for value in table.column("text").to_pylist()]
    contexts = select_neutral_contexts(
        rows,
        count=int(kl_spec["contexts"]),
        minimum_characters=256,
    )
    ordered_context_hash = hashlib.sha256(
        ("\n".join(item["sha256"] for item in contexts) + "\n").encode()
    ).hexdigest()
    if ordered_context_hash != kl_spec["ordered_context_sha256"]:
        raise ValueError("Neutral context identity hash mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        output_dir / "status.json",
        {"complete": False, "stage": "base_trajectories", "model_key": model_key},
    )
    if torch.cuda.is_available():
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
        loaded.tokenizer.truncation_side = "right"
        context_maximum_tokens = int(kl_spec["context_maximum_tokens"])
        continuation_maximum_tokens = int(kl_spec["continuation_maximum_tokens"])
        trajectories = _generate_base_trajectories(
            loaded,
            contexts,
            context_maximum_tokens=context_maximum_tokens,
            continuation_maximum_tokens=continuation_maximum_tokens,
        )
        atomic_write_json(
            output_dir / "trajectories.checkpoint.json",
            {"model_key": model_key, "trajectories": trajectories},
        )
        trials: list[dict[str, Any]] = []
        bootstrap_iterations = 10000
        bootstrap_seed = 20260805
        confidence = 0.95
        for trial_index, frontier_trial in enumerate(frontier["trials"]):
            estimator = str(frontier_trial["estimator"])
            layer = int(frontier_trial["zero_based_layer"])
            alpha = float(frontier_trial["alpha"])
            direction = directions[estimator][layer]
            if tensor_sha256(direction) != frontier_trial["direction_tensor_sha256"]:
                raise ValueError("Frontier trial direction hash mismatch")
            atomic_write_json(
                output_dir / "status.json",
                {
                    "complete": False,
                    "stage": "condition",
                    "model_key": model_key,
                    "condition_index": trial_index,
                    "layer": layer,
                    "alpha": alpha,
                },
            )
            evaluation = _evaluate_trial(
                loaded,
                contexts,
                trajectories,
                context_maximum_tokens=context_maximum_tokens,
                layer=layer,
                direction=direction,
                alpha=alpha,
                zero_alpha_tolerance=float(
                    kl_spec["zero_alpha_max_absolute_logit_difference"]
                ),
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
                confidence=confidence,
            )
            trials.append(
                {
                    "estimator": estimator,
                    "zero_based_layer": layer,
                    "alpha": alpha,
                    "direction_tensor_sha256": tensor_sha256(direction),
                    **evaluation,
                }
            )
            atomic_write_json(
                output_dir / "kl.checkpoint.json",
                {
                    "model_key": model_key,
                    "trajectories": trajectories,
                    "trials": trials,
                },
            )

        payload: dict[str, Any] = {
            "schema_version": "selective_sycophancy_neutral_trajectory_kl.v2",
            "model_key": model_key,
            "study_sha256": sha256_file(study_path),
            "fit_probe_sha256": sha256_file(fit_probe_dir / "fit_probe.json"),
            "frontier_sha256": sha256_file(frontier_dir / "frontier.json"),
            "directions_sha256": sha256_file(fit_probe_dir / "directions.safetensors"),
            "wikitext_parquet_sha256": sha256_file(wikitext_path),
            "context_count": len(contexts),
            "contexts": [
                {"row_index": item["row_index"], "sha256": item["sha256"]}
                for item in contexts
            ],
            "trajectory_contract": {
                "chat_template": False,
                "truncation_side": "right",
                "context_maximum_tokens": context_maximum_tokens,
                "continuation_maximum_tokens": continuation_maximum_tokens,
                "generation_eos_token_ids": loaded.generation_eos_token_ids,
                "fixed_base_trajectory": True,
                "eos_prediction_included": True,
                "float_precision": "float64 softmax and accumulation",
            },
            "trajectories": trajectories,
            "condition_count": len(trials),
            "trials": trials,
            "runtime": build_runtime_manifest(
                repository=repository,
                stage="neutral_trajectory_kl",
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
                    "context_count": len(contexts),
                    "context_maximum_tokens": context_maximum_tokens,
                    "continuation_maximum_tokens": continuation_maximum_tokens,
                    "condition_count": len(trials),
                    "generation_eos_token_ids": loaded.generation_eos_token_ids,
                },
                launch_identity=launch_identity,
            ),
        }
        atomic_write_json(output_dir / "neutral_trajectory_kl.json", payload)
        (output_dir / "kl.checkpoint.json").unlink(missing_ok=True)
        (output_dir / "trajectories.checkpoint.json").unlink(missing_ok=True)
        finalize_artifact_stage(
            output_dir,
            payload_files=("neutral_trajectory_kl.json",),
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
