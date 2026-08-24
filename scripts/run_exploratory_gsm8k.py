# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproduce paired GSM8K results for every steering alpha.

Outputs record the paired capability endpoint and layer-selection scope in
structured metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import torch
from safetensors.torch import load_file

from sycophancy_steering.artifacts import atomic_write_json, tensor_sha256
from sycophancy_steering.config import load_study_config
from sycophancy_steering.data import sha256_file
from sycophancy_steering.exploratory_identity import (
    exploratory_source_identity,
    load_verified_exploratory_fit,
)
from sycophancy_steering.gsm8k import load_pinned_harness_contract, select_sample
from sycophancy_steering.gsm8k_stage import (
    _generate_responses,
    _score_condition,
    _validate_scored_condition,
)
from sycophancy_steering.loading import load_study_model, unload_study_model
from sycophancy_steering.resume import load_expanded_gsm8k_resume

REPOSITORY = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsm8k-path", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def paired_changes(
    base: dict[str, Any], condition: dict[str, Any], *, metric: str
) -> dict[str, int]:
    result = {
        "improved": 0,
        "regressed": 0,
        "unchanged_correct": 0,
        "unchanged_incorrect": 0,
    }
    for before, after in zip(base["examples"], condition["examples"], strict=True):
        before_correct = bool(before[metric])
        after_correct = bool(after[metric])
        if not before_correct and after_correct:
            result["improved"] += 1
        elif before_correct and not after_correct:
            result["regressed"] += 1
        elif before_correct:
            result["unchanged_correct"] += 1
        else:
            result["unchanged_incorrect"] += 1
    return result


def paired_inference(
    base: dict[str, Any],
    condition: dict[str, Any],
    *,
    metric: str,
    iterations: int = 10000,
) -> dict[str, object]:
    differences = np.asarray(
        [
            int(bool(after[metric])) - int(bool(before[metric]))
            for before, after in zip(
                base["examples"], condition["examples"], strict=True
            )
        ],
        dtype=np.float64,
    )
    improved = int(np.count_nonzero(differences == 1.0))
    regressed = int(np.count_nonzero(differences == -1.0))
    discordant = improved + regressed
    if discordant:
        lower_tail = sum(
            math.comb(discordant, value)
            for value in range(min(improved, regressed) + 1)
        ) / (2**discordant)
        exact_two_sided_p = min(1.0, 2.0 * lower_tail)
    else:
        exact_two_sided_p = 1.0
    rng = np.random.default_rng(20260805)
    indices = rng.integers(
        0, len(differences), size=(iterations, len(differences)), endpoint=False
    )
    bootstrap = differences[indices].mean(axis=1)
    return {
        "condition_minus_base_accuracy": float(differences.mean()),
        "paired_bootstrap_95_ci": {
            "lower": float(np.quantile(bootstrap, 0.025)),
            "upper": float(np.quantile(bootstrap, 0.975)),
            "iterations": iterations,
            "seed": 20260805,
        },
        "improved": improved,
        "regressed": regressed,
        "discordant": discordant,
        "exact_two_sided_sign_p": exact_two_sided_p,
    }


def validate_resumed_results(
    base: dict[str, Any],
    trials: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    harness: Any,
) -> None:
    """Rescore every stored response and recompute paired checkpoint summaries."""

    conditions = [base, *(trial["condition"] for trial in trials)]
    for condition in conditions:
        _validate_scored_condition(condition, expected_records=len(rows))
        responses = [str(example["response"]) for example in condition["examples"]]
        rescored = _score_condition(rows, responses, harness=harness)
        if rescored != condition:
            raise ValueError("GSM8K checkpoint responses do not reproduce scores")
    for trial in trials:
        condition = trial["condition"]
        expected_changes = {
            metric: paired_changes(base, condition, metric=f"{metric}_correct")
            for metric in ("strict", "flexible")
        }
        expected_inference = {
            metric: paired_inference(base, condition, metric=f"{metric}_correct")
            for metric in ("strict", "flexible")
        }
        if trial.get("paired_changes") != expected_changes:
            raise ValueError("GSM8K checkpoint paired changes differ")
        if trial.get("paired_inference") != expected_inference:
            raise ValueError("GSM8K checkpoint paired inference differs")
        audit = trial.get("hook_audit")
        if not isinstance(audit, dict):
            raise ValueError("GSM8K checkpoint hook audit is missing")
        required = (
            "calls",
            "prefill_calls",
            "decode_calls",
            "modified_batch_rows",
            "modified_token_positions",
        )
        if any(
            not isinstance(audit.get(field), int) or int(audit[field]) < 0
            for field in required
        ):
            raise ValueError("GSM8K checkpoint hook audit is invalid")
        if (
            audit["calls"] <= 0
            or audit["prefill_calls"] != len(rows)
            or audit["calls"] != audit["prefill_calls"] + audit["decode_calls"]
            or audit["modified_batch_rows"] < audit["calls"]
            or audit["modified_token_positions"] < audit["calls"]
        ):
            raise ValueError("GSM8K checkpoint hook audit is inconsistent")


def main() -> None:
    args = arguments()
    study_path = args.study.resolve()
    study = load_study_config(study_path, require_frozen=False)
    model_spec = study["models"][args.model_key]
    fit, direction_path = load_verified_exploratory_fit(
        repository=REPOSITORY,
        study_path=study_path,
        study=study,
        fit_dir=args.fit_dir,
        model_key=args.model_key,
    )
    spec = study["sampled_gsm8k"]
    if sha256_file(args.gsm8k_path) != spec["parquet_sha256"]:
        raise ValueError("GSM8K parquet hash mismatch")
    population = parquet.read_table(args.gsm8k_path).to_pylist()
    rows = select_sample(population, count=args.count)
    harness = load_pinned_harness_contract(spec)
    prompts = [
        str(spec["prompt_template"]).format(question=str(row["question"]))
        for row in rows
    ]
    estimator = fit["layer_selection"]["chosen_estimator"]
    layers = fit["layer_selection"]["chosen_layers"]
    if estimator != "observed_prompt_state" or not layers:
        raise ValueError("Expanded GSM8K requires a selected observed estimator")
    layer = int(layers[0])
    direction = load_file(direction_path, device="cpu")[estimator][layer]
    alphas = [
        float(alpha)
        for alpha in study["activation_operator"]["alpha_grid"]
        if float(alpha) != 0.0
    ]
    checkpoint = args.output.with_name(args.output.stem + ".checkpoint.json")
    identity = {
        "schema_version": "expanded_exploratory_gsm8k.v1",
        "evidence_scope": {
            "endpoint_results": "paired_capability_estimates",
            "interpretation": "frozen_256_item_sample",
            "layer_selection": "five_seeded_random_controls_per_layer",
        },
        "model_key": args.model_key,
        **exploratory_source_identity(fit),
        "record_count": args.count,
        "selection": "first N rows of frozen deterministic GSM8K sample ordering",
        "sample_sha256": hashlib.sha256(
            ("\n".join(str(row["sample_sha256"]) for row in rows) + "\n").encode()
        ).hexdigest(),
        "estimator": estimator,
        "zero_based_layer": layer,
        "alphas": alphas,
        "direction_tensor_sha256": tensor_sha256(direction),
    }
    resume_state = load_expanded_gsm8k_resume(
        output_path=args.output,
        checkpoint_path=checkpoint,
        resume=args.resume,
        expected_identity=identity,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loaded = load_study_model(
        args.model_key,
        model_spec,
        seed=int(study["runtime"]["seed"]),
        local_files_only=True,
    )
    try:
        decoding = spec["decoding"]
        generation = {
            "chat_template_kwargs": model_spec["chat_template_kwargs"],
            "maximum_new_tokens": int(decoding["maximum_new_tokens"]),
            "stop_strings": list(spec["stop_strings"]),
        }
        if resume_state is None:
            base_responses, _ = _generate_responses(
                loaded, prompts, steering=None, **generation
            )
            base = _score_condition(rows, base_responses, harness=harness)
            trials: list[dict[str, Any]] = []
            remaining_alphas = alphas
            payload = {**identity, "base": base, "trials": trials}
            atomic_write_json(checkpoint, payload)
        else:
            base, trials, remaining_alphas = resume_state
            validate_resumed_results(base, trials, rows=rows, harness=harness)
            payload = {**identity, "base": base, "trials": trials}
        for alpha in remaining_alphas:
            responses, audit = _generate_responses(
                loaded,
                prompts,
                steering=(
                    loaded.text_model,
                    loaded.layers[layer],
                    direction,
                    alpha,
                ),
                **generation,
            )
            condition = _score_condition(rows, responses, harness=harness)
            trial = {
                "alpha": alpha,
                "condition": condition,
                "paired_changes": {
                    metric: paired_changes(base, condition, metric=f"{metric}_correct")
                    for metric in ("strict", "flexible")
                },
                "paired_inference": {
                    metric: paired_inference(
                        base, condition, metric=f"{metric}_correct"
                    )
                    for metric in ("strict", "flexible")
                },
                "hook_audit": audit,
            }
            trials.append(trial)
            atomic_write_json(checkpoint, payload)
        atomic_write_json(args.output, payload)
        checkpoint.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "output": args.output.as_posix(),
                    "model_key": args.model_key,
                    "record_count": args.count,
                    "layer": layer,
                    "base": {
                        "strict": base["strict_correct_count"],
                        "flexible": base["flexible_correct_count"],
                    },
                    "trials": [
                        {
                            "alpha": trial["alpha"],
                            "strict": trial["condition"]["strict_correct_count"],
                            "flexible": trial["condition"]["flexible_correct_count"],
                            "paired_inference": trial["paired_inference"],
                        }
                        for trial in trials
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        unload_study_model(loaded)


if __name__ == "__main__":
    main()
