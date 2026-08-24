# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproduce the completed behavior frontier.

This runner reproduces the executed five-control panel and records its endpoint
and layer-selection scope in structured metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from sycophancy_steering.artifacts import atomic_write_json, tensor_sha256
from sycophancy_steering.behavior import collect_behavior_run, generate_baseline_answers
from sycophancy_steering.comparison import compare_behavior_runs
from sycophancy_steering.config import load_study_config
from sycophancy_steering.data import validate_materialized_data
from sycophancy_steering.exploratory_identity import (
    exploratory_source_identity,
    load_verified_exploratory_fit,
)
from sycophancy_steering.frontier_stage import behavior_run_payload
from sycophancy_steering.loading import load_study_model, unload_study_model

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPOSITORY / "data" / "materialized" / "multimodel_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    repository = REPOSITORY
    study_path = args.study.resolve()
    study = load_study_config(study_path, require_frozen=False)
    model_spec = study["models"][args.model_key]
    fit, direction_path = load_verified_exploratory_fit(
        repository=repository,
        study_path=study_path,
        study=study,
        fit_dir=args.fit_dir,
        model_key=args.model_key,
    )
    expected_record_count = int(study["data"]["evaluation"]["count"])
    records = validate_materialized_data(
        args.data_dir.resolve(),
        repository / study["data"]["lock"],
        allowed_splits=("evaluation",),
    )["evaluation"]
    if len(records) != expected_record_count:
        raise ValueError("Complete exploratory evaluation split is unavailable")
    estimator = fit["layer_selection"]["chosen_estimator"]
    layers = fit["layer_selection"]["chosen_layers"]
    if estimator != "observed_prompt_state" or not layers:
        raise ValueError("Expanded fit did not select an observed-state direction")
    layer = int(layers[0])
    direction = load_file(direction_path, device="cpu")[estimator][layer]
    alphas = tuple(
        float(alpha)
        for alpha in study["activation_operator"]["alpha_grid"]
        if float(alpha) != 0.0
    )
    checkpoint = args.output.with_name(args.output.stem + ".checkpoint.json")
    if args.output.exists() or checkpoint.exists():
        raise FileExistsError("Expanded behavior output/checkpoint already exists")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loaded = load_study_model(
        args.model_key,
        model_spec,
        seed=int(study["runtime"]["seed"]),
        local_files_only=True,
    )
    try:
        contract = study["prompt_contract"]
        chat_kwargs = model_spec["chat_template_kwargs"]
        batch_size = int(model_spec["binary_generation_batch_size"])
        maximum_new_tokens = int(study["runtime"]["maximum_binary_new_tokens"])
        base_neutral, baseline_hashes = generate_baseline_answers(
            loaded,
            records,
            contract,
            chat_template_kwargs=chat_kwargs,
            generation_batch_size=batch_size,
            max_new_tokens=maximum_new_tokens,
        )
        base = collect_behavior_run(
            loaded,
            records,
            base_neutral,
            contract,
            chat_template_kwargs=chat_kwargs,
            generation_batch_size=batch_size,
            max_new_tokens=maximum_new_tokens,
            steering=None,
            reuse_base_neutral=True,
        )
        base.prompt_hashes["baseline_generation"] = baseline_hashes
        trials: list[dict[str, object]] = []
        payload: dict[str, object] = {
            "schema_version": "expanded_exploratory_behavior.v1",
            "evidence_scope": {
                "endpoint_results": "direct_intervention_estimates",
                "interpretation": "executed_checkpoint_layer_coefficient_grid",
                "layer_selection": "five_seeded_random_controls_per_layer",
            },
            "model_key": args.model_key,
            **exploratory_source_identity(fit),
            "record_count": len(records),
            "record_selection": "complete frozen evaluation split",
            "fit_probe_record_counts": {
                "fit": fit["fit_record_count"],
                "probe": fit["probe_record_count"],
            },
            "estimator": estimator,
            "zero_based_layer": layer,
            "probe_auroc": fit["probe_results"][estimator]["overall_auroc"][layer],
            "direction_tensor_sha256": tensor_sha256(direction),
            "alphas": list(alphas),
            "bootstrap_iterations": args.bootstrap_iterations,
            "base": behavior_run_payload(base),
            "trials": trials,
        }
        atomic_write_json(checkpoint, payload)
        for alpha in alphas:
            condition = collect_behavior_run(
                loaded,
                records,
                base_neutral,
                contract,
                chat_template_kwargs=chat_kwargs,
                generation_batch_size=batch_size,
                max_new_tokens=maximum_new_tokens,
                steering=(
                    loaded.text_model,
                    loaded.layers[layer],
                    direction,
                    alpha,
                ),
                reuse_base_neutral=False,
            )
            comparison = compare_behavior_runs(
                records,
                base,
                condition,
                modes=tuple(contract["pressure_modes"]),
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=20260805,
                confidence=0.95,
            )
            trials.append(
                {
                    "alpha": alpha,
                    "condition": behavior_run_payload(condition),
                    "comparison_to_base": comparison,
                }
            )
            atomic_write_json(checkpoint, payload)
        atomic_write_json(args.output, payload)
        checkpoint.unlink(missing_ok=True)
        fields = (
            "neutral_accuracy",
            "neutral_invalid_rate",
            "pressure_error",
            "pressure_invalid_rate",
            "natural_correct_suggestion_update_rate",
            "controlled_correction_acceptance_rate",
            "controlled_correction_invalid_rate",
        )
        print(
            json.dumps(
                {
                    "output": args.output.as_posix(),
                    "model_key": args.model_key,
                    "record_count": len(records),
                    "layer": layer,
                    "probe_auroc": payload["probe_auroc"],
                    "base": {field: base.metrics[field] for field in fields},
                    "trials": [
                        {
                            "alpha": trial["alpha"],
                            "metrics": {
                                field: trial["condition"]["metrics"][field]
                                for field in fields
                            },
                            "deltas": trial["comparison_to_base"][
                                "deltas_condition_minus_base"
                            ],
                            "intervals": trial["comparison_to_base"]["intervals"],
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
