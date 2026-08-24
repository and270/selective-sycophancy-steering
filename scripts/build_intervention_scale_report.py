# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive residual-relative intervention scales from completed fit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "results" / "INTERVENTION_SCALE_COMPARISON.json"
EXECUTED_ALPHA_MAGNITUDES = (0.5, 1.0, 2.0)
SIGNIFICANT_DIGITS = 12


@dataclass(frozen=True)
class ModelRun:
    """Identity and location of one completed model run."""

    model_key: str
    model_label: str
    relative_run_path: Path


DEFAULT_RUNS = (
    ModelRun(
        "qwen35_2b",
        "Qwen3.5-2B",
        Path("results/runs/expanded_qwen35_2b_20260805"),
    ),
    ModelRun(
        "qwen35_4b",
        "Qwen3.5-4B",
        Path("results/runs/expanded_qwen35_4b_20260805"),
    ),
    ModelRun(
        "gemma4_e2b_it",
        "Gemma 4 E2B",
        Path("results/runs/expanded_gemma4_e2b_20260805"),
    ),
    ModelRun(
        "gemma4_e4b_it",
        "Gemma 4 E4B",
        Path("results/runs/expanded_gemma4_e4b_20260805"),
    ),
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Scale analysis produced a non-finite value")
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _relative_path(path: Path, repository: Path) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Artifact is outside the repository: {path}") from error


def _standard_median(values: torch.Tensor) -> float:
    ordered = torch.sort(values).values
    count = int(ordered.numel())
    if count == 0:
        raise ValueError("Probe residual tensor has no records")
    midpoint = count // 2
    if count % 2:
        return float(ordered[midpoint].item())
    return float(((ordered[midpoint - 1] + ordered[midpoint]) / 2).item())


def _load_probe_layer(
    observations_path: Path,
    *,
    observation_metadata: dict[str, Any],
    primary_layer: int,
) -> tuple[torch.Tensor, dict[str, Any], tuple[int, int, int]]:
    tensor_metadata_by_name = _require_mapping(
        observation_metadata.get("tensors"),
        field="observation_artifact.tensors",
    )
    tensor_metadata = _require_mapping(
        tensor_metadata_by_name.get("probe_prompt_residuals"),
        field="observation_artifact.tensors.probe_prompt_residuals",
    )
    recorded_shape = tensor_metadata.get("shape")
    if (
        not isinstance(recorded_shape, list)
        or len(recorded_shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in recorded_shape
        )
    ):
        raise ValueError("probe_prompt_residuals shape metadata is invalid")
    if primary_layer < 0 or primary_layer >= recorded_shape[1]:
        raise ValueError("Primary layer is outside probe tensor metadata")
    with safe_open(observations_path, framework="pt", device="cpu") as handle:
        tensor_names = set(handle.keys())
        if "probe_prompt_residuals" not in tensor_names:
            raise ValueError("observations.safetensors lacks probe_prompt_residuals")
        tensor_slice = handle.get_slice("probe_prompt_residuals")
        full_shape = tuple(tensor_slice.get_shape())
        if len(full_shape) != 3:
            raise ValueError(
                "probe_prompt_residuals must have shape [records, layers, width]"
            )
        if list(full_shape) != recorded_shape:
            raise ValueError("probe_prompt_residuals shape differs from fit metadata")
        selected_layer = tensor_slice[:, primary_layer, :]
    if selected_layer.shape != (full_shape[0], full_shape[2]):
        raise ValueError("probe_prompt_residuals shape differs from fit metadata")
    if tensor_metadata.get("dtype") != str(selected_layer.dtype):
        raise ValueError("probe_prompt_residuals dtype differs from fit metadata")
    if not torch.isfinite(selected_layer).all():
        raise ValueError("probe_prompt_residuals contains non-finite values")
    return selected_layer, tensor_metadata, full_shape


def analyze_model_run(
    repository: Path,
    run: ModelRun,
    *,
    alpha_magnitudes: Sequence[float] = EXECUTED_ALPHA_MAGNITUDES,
) -> dict[str, Any]:
    """Compute one model's residual-relative intervention scales."""

    fit_probe_dir = (repository / run.relative_run_path / "fit_probe").resolve()
    fit_probe_path = fit_probe_dir / "fit_probe.json"
    fit = _require_mapping(
        json.loads(fit_probe_path.read_text(encoding="utf-8")), field="fit_probe"
    )
    if fit.get("model_key") != run.model_key:
        raise ValueError(
            f"Model key mismatch for {run.model_label}: {fit.get('model_key')!r}"
        )

    selection = _require_mapping(fit.get("layer_selection"), field="layer_selection")
    estimator = selection.get("chosen_estimator")
    layers = selection.get("chosen_layers")
    if not isinstance(estimator, str) or not estimator:
        raise ValueError(f"No chosen estimator for {run.model_label}")
    if (
        not isinstance(layers, list)
        or not layers
        or isinstance(layers[0], bool)
        or not isinstance(layers[0], int)
    ):
        raise ValueError(f"No primary chosen layer for {run.model_label}")
    primary_layer = layers[0]

    observation_metadata = _require_mapping(
        fit.get("observation_artifact"), field="observation_artifact"
    )
    observation_relative = observation_metadata.get("path")
    if (
        not isinstance(observation_relative, str)
        or Path(observation_relative).is_absolute()
        or len(Path(observation_relative).parts) != 1
    ):
        raise ValueError("Observation artifact path must name a direct child file")
    observations_path = fit_probe_dir / observation_relative
    expected_observations_sha256 = observation_metadata.get("sha256")
    if not isinstance(expected_observations_sha256, str):
        raise ValueError("Observation artifact SHA-256 is missing")
    actual_observations_sha256 = sha256_file(observations_path)
    if actual_observations_sha256 != expected_observations_sha256:
        raise ValueError(
            f"Observations SHA-256 mismatch for {run.model_label}: "
            f"expected {expected_observations_sha256}, "
            f"got {actual_observations_sha256}"
        )

    selected_probe, probe_metadata, probe_shape = _load_probe_layer(
        observations_path,
        observation_metadata=observation_metadata,
        primary_layer=primary_layer,
    )
    on_disk_dtype = str(selected_probe.dtype)
    selected_probe = selected_probe.to(dtype=torch.float64)
    per_record_norms = torch.linalg.vector_norm(selected_probe, ord=2, dim=1)
    mean_residual_norm = float(per_record_norms.mean().item())
    median_residual_norm = _standard_median(per_record_norms)
    if mean_residual_norm <= 0:
        raise ValueError(
            f"Mean probe residual norm is not positive for {run.model_label}"
        )

    direction_artifact = _require_mapping(
        fit.get("direction_artifact"), field="direction_artifact"
    )
    direction_tensors = _require_mapping(
        direction_artifact.get("tensors"), field="direction_artifact.tensors"
    )
    direction_metadata = _require_mapping(
        direction_tensors.get(estimator),
        field=f"direction_artifact.tensors.{estimator}",
    )
    layer_norms = direction_metadata.get("layer_norms")
    if not isinstance(layer_norms, list) or primary_layer >= len(layer_norms):
        raise ValueError(f"Direction norm metadata is incomplete for {run.model_label}")
    direction_norm = float(layer_norms[primary_layer])
    if not math.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError(f"Direction norm is invalid for {run.model_label}")

    alpha_values = tuple(float(value) for value in alpha_magnitudes)
    if (
        not alpha_values
        or any(not math.isfinite(value) or value <= 0 for value in alpha_values)
        or tuple(sorted(set(alpha_values))) != alpha_values
    ):
        raise ValueError("Alpha magnitudes must be positive, unique, and increasing")
    relative_updates = []
    for alpha_magnitude in alpha_values:
        fraction = alpha_magnitude * direction_norm / mean_residual_norm
        relative_updates.append(
            {
                "alpha_magnitude": alpha_magnitude,
                "fraction_of_mean_probe_residual_norm": _rounded(fraction),
                "percent_of_mean_probe_residual_norm": _rounded(100 * fraction),
            }
        )

    return {
        "direction_l2_norm": direction_norm,
        "model_key": run.model_key,
        "model_label": run.model_label,
        "primary_estimator": estimator,
        "primary_zero_based_layer": primary_layer,
        "probe_prompt_residuals": {
            "computation_dtype": "float64",
            "first_axis_count": probe_shape[0],
            "hidden_size": probe_shape[2],
            "l2_mean": _rounded(mean_residual_norm),
            "l2_median": _rounded(median_residual_norm),
            "on_disk_dtype": on_disk_dtype,
        },
        "relative_update_by_alpha_magnitude": relative_updates,
        "source": {
            "fit_probe_path": _relative_path(fit_probe_path, repository),
            "fit_probe_sha256": sha256_file(fit_probe_path),
            "observations_path": _relative_path(observations_path, repository),
            "observations_sha256": actual_observations_sha256,
            "probe_prompt_residuals_tensor_sha256": probe_metadata.get("sha256"),
        },
    }


def build_report(
    repository: Path = REPOSITORY,
    *,
    runs: Sequence[ModelRun] = DEFAULT_RUNS,
    alpha_magnitudes: Sequence[float] = EXECUTED_ALPHA_MAGNITUDES,
) -> dict[str, Any]:
    """Build the deterministic comparison payload."""

    return {
        "definitions": {
            "alpha_magnitudes": [float(value) for value in alpha_magnitudes],
            "direction_norm": (
                "||d_l||_2 from fit_probe.json metadata for the chosen estimator "
                "and primary layer"
            ),
            "mean_probe_residual_norm": "mean_i ||H_probe[i, l, :]||_2",
            "median_probe_residual_norm": "median_i ||H_probe[i, l, :]||_2",
            "numeric_rounding": f"{SIGNIFICANT_DIGITS} significant decimal digits",
            "relative_update_fraction": (
                "rho(alpha) = |alpha| * ||d_l||_2 / mean_i ||H_probe[i, l, :]||_2"
            ),
            "residual_rows": (
                "first-axis rows of probe_prompt_residuals at the primary "
                "zero-based layer"
            ),
        },
        "models": [
            analyze_model_run(
                repository,
                run,
                alpha_magnitudes=alpha_magnitudes,
            )
            for run in runs
        ],
        "schema_version": "intervention_scale_comparison.v1",
    }


def rendered_report(report: dict[str, Any]) -> str:
    return (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered_report(report), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path (default: results/INTERVENTION_SCALE_COMPARISON.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero instead of writing when the tracked report differs",
    )
    args = parser.parse_args()

    report = build_report()
    output = args.output if args.output.is_absolute() else REPOSITORY / args.output
    content = rendered_report(report)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Intervention scale report is stale: {output}")
    else:
        write_report(output, report)
        print(output.relative_to(REPOSITORY))


if __name__ == "__main__":
    main()
