# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import sycophancy_steering.artifacts as artifacts
from sycophancy_steering.artifacts import finalize_artifact_stage

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/build_paper_assets.py"
SPEC = importlib.util.spec_from_file_location("build_paper_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _install_semantic_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "load_study_config",
        lambda _path, *, require_frozen: {"data": {"lock": "data-lock.json"}},
    )

    def load_json(directory: Path, filename: str) -> dict[str, object]:
        return json.loads((directory / filename).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        MODULE,
        "_verify_fit_artifact",
        lambda directory, **_kwargs: (
            {"runtime": {"launch_identity": {"identity_sha256": "identity"}}},
            {},
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "verify_frontier_artifact",
        lambda directory, **_kwargs: load_json(directory, "frontier.json"),
    )
    monkeypatch.setattr(
        MODULE,
        "verify_kl_artifact",
        lambda directory, **_kwargs: load_json(directory, "neutral_trajectory_kl.json"),
    )
    monkeypatch.setattr(
        MODULE,
        "verify_gsm8k_artifact",
        lambda directory, **_kwargs: load_json(directory, "sampled_gsm8k.json"),
    )


def _make_model(root: Path, key: str, *, identity: str = "identity") -> None:
    (root / key / "fit_probe").mkdir(parents=True, exist_ok=True)
    launch_identity = {"repository_commit": identity}
    launch_identity["identity_sha256"] = artifacts._launch_identity_sha256(
        launch_identity
    )
    runtime = {"launch_identity": launch_identity}
    metrics = {
        "pressure_error_count": 3,
        "pressure_denominator": 10,
        "pressure_error": 0.3,
        "natural_correct_suggestion_update_rate": 0.8,
        "controlled_correction_acceptance_rate": 0.9,
    }
    condition_metrics = {
        "pressure_error_count": 2,
        "pressure_denominator": 10,
        "pressure_error": 0.2,
        "natural_correct_suggestion_update_rate": 0.7,
        "controlled_correction_acceptance_rate": 0.85,
    }
    frontier = {
        "model_key": key,
        "chosen_estimator": "observed_prompt_state",
        "chosen_layers": [4],
        "base": {"metrics": metrics},
        "trials": [
            {
                "estimator": "observed_prompt_state",
                "zero_based_layer": 4,
                "alpha": -2.0,
                "condition": {"metrics": condition_metrics},
            }
        ],
        "runtime": runtime,
    }
    kl = {
        "model_key": key,
        "trials": [
            {
                "estimator": "observed_prompt_state",
                "zero_based_layer": 4,
                "alpha": -2.0,
                "token_micro": {"forward_kl_nats": {"mean": 0.001}},
                "prompt_macro": {"top1_agreement": {"mean": 0.99}},
            }
        ],
        "runtime": runtime,
    }
    gsm = {
        "model_key": key,
        "base": {"flexible_correct_count": 100, "record_count": 256},
        "condition": {
            "alpha": -2.0,
            "scores": {"flexible_correct_count": 98, "record_count": 256},
        },
        "runtime": runtime,
    }
    for name, payload, filename in (
        ("frontier", frontier, "frontier.json"),
        ("neutral_kl", kl, "neutral_trajectory_kl.json"),
        ("sampled_gsm8k", gsm, "sampled_gsm8k.json"),
    ):
        directory = root / key / name
        _write_json(directory / filename, payload)
        finalize_artifact_stage(
            directory,
            payload_files=(filename,),
            status={
                "complete": True,
                "stage": "complete",
                "run_kind": "scientific",
                "model_key": key,
            },
        )


def test_build_paper_assets_requires_complete_shared_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_semantic_stubs(monkeypatch)
    for key in MODULE.MODELS:
        _make_model(tmp_path, key)

    latex = MODULE.build(tmp_path)

    assert "Qwen3.5-4B" in latex
    assert "3/10 $\\rightarrow$ 2/10" in latex
    assert "0.001" in latex


def test_build_paper_assets_requires_fit_probe_parent(tmp_path: Path) -> None:
    for key in MODULE.MODELS:
        _make_model(tmp_path, key)
    (tmp_path / "qwen35_4b" / "fit_probe").rmdir()

    with pytest.raises(ValueError, match=r"fit|Fit"):
        MODULE.build(tmp_path)


def test_build_paper_assets_rejects_same_digest_different_identity_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_semantic_stubs(monkeypatch)
    for key in MODULE.MODELS:
        _make_model(tmp_path, key)
    key = "gemma4_e4b_it"
    for directory, filename in (
        ("frontier", "frontier.json"),
        ("neutral_kl", "neutral_trajectory_kl.json"),
        ("sampled_gsm8k", "sampled_gsm8k.json"),
    ):
        path = tmp_path / key / directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["runtime"]["launch_identity"]["repository_commit"] = "different"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        MODULE.build(tmp_path)


def test_build_paper_assets_rejects_mixed_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_semantic_stubs(monkeypatch)
    for index, key in enumerate(MODULE.MODELS):
        _make_model(tmp_path, key, identity=f"identity-{index}")

    with pytest.raises(ValueError, match="identity"):
        MODULE.build(tmp_path)
