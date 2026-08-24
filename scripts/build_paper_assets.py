# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate manuscript result tables from completed scientific artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sycophancy_steering.artifacts import verify_launch_identity_digest
from sycophancy_steering.config import load_study_config
from sycophancy_steering.frontier_stage import (
    _verify_fit_artifact,
    verify_frontier_artifact,
)
from sycophancy_steering.gsm8k_stage import verify_gsm8k_artifact
from sycophancy_steering.kl_stage import verify_kl_artifact

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = REPOSITORY / "configs" / "studies" / "multimodel_v1.json"

MODELS = {
    "qwen35_4b": "Qwen3.5-4B",
    "gemma4_e2b_it": "Gemma 4 E2B-it",
    "gemma4_e4b_it": "Gemma 4 E4B-it (NF4)",
}


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f}\\%"


def _delta_pp(condition: float, base: float) -> str:
    return f"{100.0 * (float(condition) - float(base)):+.2f}"


def _primary_trial(frontier: dict[str, Any]) -> dict[str, Any]:
    layers = frontier.get("chosen_layers")
    estimator = frontier.get("chosen_estimator")
    if not isinstance(layers, list) or not layers or not isinstance(estimator, str):
        raise ValueError("No preregistered steering layer is available")
    matches = [
        trial
        for trial in frontier["trials"]
        if trial["estimator"] == estimator
        and int(trial["zero_based_layer"]) == int(layers[0])
        and float(trial["alpha"]) == -2.0
    ]
    if len(matches) != 1:
        raise ValueError("Primary alpha=-2 frontier condition is missing or duplicated")
    return matches[0]


def _matching_trial(payload: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    matches = [
        trial
        for trial in payload["trials"]
        if trial["estimator"] == primary["estimator"]
        and int(trial["zero_based_layer"]) == int(primary["zero_based_layer"])
        and float(trial["alpha"]) == float(primary["alpha"])
    ]
    if len(matches) != 1:
        raise ValueError("Matching KL trial is missing or duplicated")
    return matches[0]


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime")
    launch = runtime.get("launch_identity") if isinstance(runtime, dict) else None
    if (
        not isinstance(launch, dict)
        or not isinstance(launch.get("identity_sha256"), str)
        or not launch["identity_sha256"]
    ):
        raise ValueError("Artifact has no scientific launch identity")
    verify_launch_identity_digest(launch)
    return launch


def build(
    results_root: Path,
    *,
    repository: Path = REPOSITORY,
    study_path: Path = DEFAULT_STUDY,
) -> str:
    for key in MODELS:
        fit_probe_dir = results_root / key / "fit_probe"
        if not fit_probe_dir.is_dir():
            raise ValueError(f"Missing fit/probe artifact directory: {fit_probe_dir}")
    study = load_study_config(study_path, require_frozen=True)
    data_lock_path = repository / str(study["data"]["lock"])
    behavior_rows: list[str] = []
    auxiliary_rows: list[str] = []
    identities: list[dict[str, Any]] = []
    for key, label in MODELS.items():
        root = results_root / key
        fit_probe_dir = root / "fit_probe"
        frontier_dir = root / "frontier"
        fit_result, directions = _verify_fit_artifact(
            fit_probe_dir,
            model_key=key,
            study_path=study_path,
            data_lock_path=data_lock_path,
        )
        frontier = verify_frontier_artifact(
            frontier_dir,
            model_key=key,
            study_path=study_path,
            data_lock_path=data_lock_path,
            fit_probe_dir=fit_probe_dir,
            fit_result=fit_result,
            directions=directions,
        )
        kl = verify_kl_artifact(
            root / "neutral_kl",
            model_key=key,
            study_path=study_path,
            data_lock_path=data_lock_path,
            fit_probe_dir=fit_probe_dir,
            frontier_dir=frontier_dir,
            fit_result=fit_result,
            frontier=frontier,
        )
        gsm = verify_gsm8k_artifact(
            root / "sampled_gsm8k",
            model_key=key,
            study_path=study_path,
            data_lock_path=data_lock_path,
            fit_probe_dir=fit_probe_dir,
            frontier_dir=frontier_dir,
            fit_result=fit_result,
            frontier=frontier,
        )
        identities.extend((_identity(frontier), _identity(kl), _identity(gsm)))
        primary = _primary_trial(frontier)
        base_metrics = frontier["base"]["metrics"]
        condition_metrics = primary["condition"]["metrics"]
        natural_key = "natural_correct_suggestion_update_rate"
        controlled_key = "controlled_correction_acceptance_rate"
        natural_transition = (
            f"{_percent(base_metrics[natural_key])} $\\rightarrow$ "
            f"{_percent(condition_metrics[natural_key])}"
        )
        controlled_transition = (
            f"{_percent(base_metrics[controlled_key])} $\\rightarrow$ "
            f"{_percent(condition_metrics[controlled_key])}"
        )
        behavior_rows.append(
            " & ".join(
                [
                    label,
                    str(primary["zero_based_layer"]),
                    (
                        f"{base_metrics['pressure_error_count']}/"
                        f"{base_metrics['pressure_denominator']} $\\rightarrow$ "
                        f"{condition_metrics['pressure_error_count']}/"
                        f"{condition_metrics['pressure_denominator']}"
                    ),
                    _delta_pp(
                        condition_metrics["pressure_error"],
                        base_metrics["pressure_error"],
                    ),
                    natural_transition,
                    controlled_transition,
                ]
            )
            + " \\\\"
        )
        kl_trial = _matching_trial(kl, primary)
        gsm_condition = gsm.get("condition")
        if not isinstance(gsm_condition, dict) or float(gsm_condition["alpha"]) != -2.0:
            raise ValueError("Primary sampled GSM8K condition is missing")
        auxiliary_rows.append(
            " & ".join(
                [
                    label,
                    f"{kl_trial['token_micro']['forward_kl_nats']['mean']:.6g}",
                    f"{kl_trial['prompt_macro']['top1_agreement']['mean']:.4f}",
                    (
                        f"{gsm['base']['flexible_correct_count']}/"
                        f"{gsm['base']['record_count']} $\\rightarrow$ "
                        f"{gsm_condition['scores']['flexible_correct_count']}/"
                        f"{gsm_condition['scores']['record_count']}"
                    ),
                ]
            )
            + " \\\\"
        )
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(
            "Paper inputs do not share one frozen scientific code identity"
        )
    return (
        "\n".join(
            [
                "% Generated mechanically; do not edit.",
                (
                    "The cross-model artifacts share one frozen scientific code "
                    "identity. Table~\\ref{tab:cross-behavior} reports the "
                    "prespecified top-probe-layer, $\\alpha=-2$ point; the complete "
                    "frontier remains in the released artifacts."
                ),
                "",
                "\\begin{table*}[t]",
                "\\centering",
                "\\small",
                (
                    "\\caption{Cross-model behavioral measurements at the "
                    "prespecified primary dose. $\\Delta$ is condition minus "
                    "base in percentage points.}"
                ),
                "\\label{tab:cross-behavior}",
                "\\begin{tabular}{lrrrll}",
                "\\toprule",
                (
                    "Model & Layer & Pressure errors & $\\Delta$ pp & Natural "
                    "correction & Controlled correction \\\\"
                ),
                "\\midrule",
                *behavior_rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table*}",
                "",
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                (
                    "\\caption{Distribution and sampled capability measurements "
                    "at the same primary dose. KL is token-micro forward KL in "
                    "nats; agreement is prompt-macro next-token top-1 agreement.}"
                ),
                "\\label{tab:cross-aux}",
                "\\begin{tabular}{lrrl}",
                "\\toprule",
                "Model & KL & Top-1 agreement & GSM8K flexible correct \\\\",
                "\\midrule",
                *auxiliary_rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/generated/cross_model_results.tex"),
    )
    args = parser.parse_args()
    content = build(
        args.results_root.resolve(),
        repository=args.repository.resolve(),
        study_path=args.study.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
