# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / ("build_intervention_scale_report.py")
)
SPEC = importlib.util.spec_from_file_location("build_intervention_scale_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _synthetic_run(repository: Path, *, observations_sha256: str | None = None):
    relative_run = Path("results/runs/synthetic_model")
    fit_dir = repository / relative_run / "fit_probe"
    fit_dir.mkdir(parents=True)
    observations = torch.tensor(
        [
            [[99.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
            [[99.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
            [[99.0, 0.0, 0.0], [8.0, 15.0, 0.0]],
            [[99.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    observations_path = fit_dir / "observations.safetensors"
    save_file({"probe_prompt_residuals": observations}, observations_path)
    actual_sha256 = REPORT.sha256_file(observations_path)
    fit = {
        "direction_artifact": {
            "tensors": {
                "completion_contrast": {"layer_norms": [200.0, 300.0]},
                "observed_prompt_state": {"layer_norms": [10.0, 15.0]},
            }
        },
        "layer_selection": {
            "chosen_estimator": "observed_prompt_state",
            "chosen_layers": [1],
        },
        "model_key": "synthetic_model",
        "observation_artifact": {
            "path": "observations.safetensors",
            "sha256": observations_sha256 or actual_sha256,
            "tensors": {
                "probe_prompt_residuals": {
                    "dtype": str(observations.dtype),
                    "sha256": "synthetic-probe-tensor-hash",
                    "shape": list(observations.shape),
                }
            },
        },
    }
    fit_path = fit_dir / "fit_probe.json"
    fit_path.write_text(
        json.dumps(fit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return REPORT.ModelRun(
        "synthetic_model",
        "Synthetic Model",
        relative_run,
    )


def test_build_report_uses_primary_layer_and_float64_standard_median(
    tmp_path: Path,
) -> None:
    run = _synthetic_run(tmp_path)

    report = REPORT.build_report(tmp_path, runs=(run,))

    model = report["models"][0]
    assert model["primary_estimator"] == "observed_prompt_state"
    assert model["primary_zero_based_layer"] == 1
    assert model["direction_l2_norm"] == 15.0
    assert model["probe_prompt_residuals"] == {
        "computation_dtype": "float64",
        "first_axis_count": 4,
        "hidden_size": 3,
        "l2_mean": 7.5,
        "l2_median": 5.5,
        "on_disk_dtype": "torch.float32",
    }
    assert model["relative_update_by_alpha_magnitude"] == [
        {
            "alpha_magnitude": 0.5,
            "fraction_of_mean_probe_residual_norm": 1.0,
            "percent_of_mean_probe_residual_norm": 100.0,
        },
        {
            "alpha_magnitude": 1.0,
            "fraction_of_mean_probe_residual_norm": 2.0,
            "percent_of_mean_probe_residual_norm": 200.0,
        },
        {
            "alpha_magnitude": 2.0,
            "fraction_of_mean_probe_residual_norm": 4.0,
            "percent_of_mean_probe_residual_norm": 400.0,
        },
    ]
    assert model["source"]["fit_probe_path"] == (
        "results/runs/synthetic_model/fit_probe/fit_probe.json"
    )
    assert model["source"]["observations_path"] == (
        "results/runs/synthetic_model/fit_probe/observations.safetensors"
    )


def test_report_rendering_is_deterministic(tmp_path: Path) -> None:
    run = _synthetic_run(tmp_path)

    first = REPORT.rendered_report(REPORT.build_report(tmp_path, runs=(run,)))
    second = REPORT.rendered_report(REPORT.build_report(tmp_path, runs=(run,)))

    assert first == second
    assert '"schema_version": "intervention_scale_comparison.v1"' in first


def test_observation_file_hash_mismatch_fails_before_analysis(tmp_path: Path) -> None:
    run = _synthetic_run(tmp_path, observations_sha256="0" * 64)

    with pytest.raises(ValueError, match="Observations SHA-256 mismatch"):
        REPORT.build_report(tmp_path, runs=(run,))
