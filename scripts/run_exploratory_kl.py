# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproduce neutral-trajectory KL for every steering alpha.

Outputs record the fixed-trajectory distribution endpoint and layer-selection
scope in structured metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
from sycophancy_steering.kl import select_neutral_contexts
from sycophancy_steering.kl_stage import (
    _evaluate_trial,
    _generate_base_trajectories,
    exploratory_kl_trial_summary,
)
from sycophancy_steering.loading import load_study_model, unload_study_model

REPOSITORY = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wikitext-path", type=Path, required=True)
    return parser.parse_args()


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
    estimator = fit["layer_selection"]["chosen_estimator"]
    layers = fit["layer_selection"]["chosen_layers"]
    if estimator != "observed_prompt_state" or not layers:
        raise ValueError("Expanded KL requires a selected observed estimator")
    layer = int(layers[0])
    direction = load_file(direction_path, device="cpu")[estimator][layer]
    alphas = [
        float(alpha)
        for alpha in study["activation_operator"]["alpha_grid"]
        if float(alpha) != 0.0
    ]
    kl_spec = study["neutral_kl"]
    if sha256_file(args.wikitext_path) != kl_spec["parquet_sha256"]:
        raise ValueError("Wikitext parquet hash mismatch")
    table = parquet.read_table(args.wikitext_path, columns=["text"])
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
        raise ValueError("Wikitext context identity mismatch")
    checkpoint = args.output.with_name(args.output.stem + ".checkpoint.json")
    if args.output.exists() or checkpoint.exists():
        raise FileExistsError("Expanded KL output/checkpoint already exists")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loaded = load_study_model(
        args.model_key,
        model_spec,
        seed=int(study["runtime"]["seed"]),
        local_files_only=True,
    )
    try:
        context_maximum_tokens = int(kl_spec["context_maximum_tokens"])
        trajectories = _generate_base_trajectories(
            loaded,
            contexts,
            context_maximum_tokens=context_maximum_tokens,
            continuation_maximum_tokens=int(kl_spec["continuation_maximum_tokens"]),
        )
        common = {
            "loaded": loaded,
            "contexts": contexts,
            "trajectories": trajectories,
            "context_maximum_tokens": context_maximum_tokens,
            "layer": layer,
            "direction": direction,
            "zero_alpha_tolerance": 1.0e-6,
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260805,
            "confidence": 0.95,
        }
        zero = _evaluate_trial(alpha=0.0, **common)
        if zero["maximum_absolute_logit_difference"] != 0.0:
            raise RuntimeError("Alpha-zero trajectory identity failed")
        trials: list[dict[str, object]] = []
        payload = {
            "schema_version": "expanded_exploratory_trajectory_kl.v1",
            "evidence_scope": {
                "endpoint_results": "fixed_trajectory_distribution_estimates",
                "interpretation": "frozen_64_context_neutral_trajectory_sample",
                "layer_selection": "five_seeded_random_controls_per_layer",
            },
            "model_key": args.model_key,
            **exploratory_source_identity(fit),
            "estimator": estimator,
            "zero_based_layer": layer,
            "alphas": alphas,
            "direction_tensor_sha256": tensor_sha256(direction),
            "ordered_context_sha256": ordered_context_hash,
            "context_count": len(contexts),
            "contexts": contexts,
            "trajectory_token_counts": [
                item["continuation_token_count"] for item in trajectories
            ],
            "zero": zero,
            "trials": trials,
        }
        atomic_write_json(checkpoint, payload)
        for alpha in alphas:
            trials.append(
                {
                    "alpha": alpha,
                    "condition": _evaluate_trial(alpha=alpha, **common),
                }
            )
            atomic_write_json(checkpoint, payload)
        atomic_write_json(args.output, payload)
        checkpoint.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "output": args.output.as_posix(),
                    "model_key": args.model_key,
                    "context_count": len(contexts),
                    "layer": layer,
                    "zero_max_logit_difference": zero[
                        "maximum_absolute_logit_difference"
                    ],
                    "trials": [exploratory_kl_trial_summary(trial) for trial in trials],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        unload_study_model(loaded)


if __name__ == "__main__":
    main()
