# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sycophancy_steering.artifacts as artifacts
import sycophancy_steering.fit_probe_stage as fit_probe_stage
from sycophancy_steering.fit_probe import DirectionObservations


def test_scientific_fit_rejects_batch_overrides_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot override frozen batch sizes"):
        fit_probe_stage.run_fit_probe_stage(
            repository=tmp_path,
            study_path=tmp_path / "missing-study.json",
            data_dir=tmp_path / "missing-data",
            output_dir=tmp_path / "output",
            model_key="qwen35_4b",
            run_kind="scientific",
            limit=None,
            generation_batch_size=1,
            residual_batch_size=2,
        )


def test_executed_reproduction_rejects_limits_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot use a record limit"):
        fit_probe_stage.run_fit_probe_stage(
            repository=tmp_path,
            study_path=tmp_path / "missing-study.json",
            data_dir=tmp_path / "missing-data",
            output_dir=tmp_path / "output",
            model_key="qwen35_4b",
            run_kind="executed_reproduction",
            limit=1,
        )


def test_executed_reproduction_rejects_batch_overrides_before_io(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="recorded batch sizes"):
        fit_probe_stage.run_fit_probe_stage(
            repository=tmp_path,
            study_path=tmp_path / "missing-study.json",
            data_dir=tmp_path / "missing-data",
            output_dir=tmp_path / "output",
            model_key="qwen35_4b",
            run_kind="executed_reproduction",
            limit=None,
            generation_batch_size=1,
        )


def test_scientific_fit_rejects_loaded_study_toc_tou_before_data_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tagged = {"runtime": {"seed": 0}, "data": {"lock": "data-lock.json"}}
    altered = {"runtime": {"seed": 123456}, "data": {"lock": "data-lock.json"}}
    launch = {"study_payload_sha256": artifacts._canonical_json_sha256(tagged)}
    monkeypatch.setattr(
        fit_probe_stage,
        "capture_scientific_launch_identity",
        lambda _repository, _study_path: launch,
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "load_study_config",
        lambda _study_path, *, require_frozen: altered,
    )

    with pytest.raises(RuntimeError, match="study payload"):
        fit_probe_stage.run_fit_probe_stage(
            repository=tmp_path,
            study_path=tmp_path / "study.json",
            data_dir=tmp_path / "missing-data",
            output_dir=tmp_path / "output",
            model_key="qwen35_4b",
            run_kind="scientific",
            limit=None,
        )


def test_ineligible_completion_is_omitted_without_aborting_observed_estimator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = {
        "models": {
            "m": {
                "binary_generation_batch_size": 1,
                "residual_batch_size": 1,
                "chat_template_kwargs": {},
                "generation_eos_token_ids": [1],
            }
        },
        "data": {"lock": "lock.json", "materialized_dir": "data"},
        "runtime": {"seed": 0, "maximum_binary_new_tokens": 1},
        "prompt_contract": {"pressure_modes": ["doubt"]},
        "direction_estimation": {
            "completion_contrast": {
                "source_datasets": ["trivia_qa", "truthful_qa"],
                "minimum_fit_eligible_records_per_source_option": 10,
            },
            "observed_prompt_state": {
                "minimum_fit_class_count_overall": 1,
                "minimum_fit_class_count_per_mode": 1,
            },
        },
        "layer_selection": {
            "random_direction_controls_per_layer": 1,
            "random_seed": 0,
        },
    }
    study_path = tmp_path / "study.json"
    study_path.write_text("{}\n", encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    launch = {
        "study_payload_sha256": artifacts._canonical_json_sha256(study),
        "data_lock_sha256": artifacts.sha256_file(lock_path),
    }
    observation = DirectionObservations(
        baseline_answers={"r": {"text": "A", "parsed": "A", "prompt_sha256": "a" * 64}},
        descriptors=[
            {
                "record_id": "r",
                "source_dataset": "trivia_qa",
                "correct_option": "A",
                "mode": "doubt",
            }
        ],
        followup_answers=[{"text": "A", "parsed": "A", "prompt_sha256": "b" * 64}],
        prompt_residuals=torch.ones((1, 1, 2)),
        caving_residuals=torch.zeros((1, 1, 2)),
        resisting_residuals=torch.zeros((1, 1, 2)),
    )
    loaded = SimpleNamespace(
        model_class="Model",
        model_fingerprint={},
        tokenizer_fingerprint={},
        layer_path="layers",
        peak_memory_bytes=0,
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "capture_scientific_launch_identity",
        lambda *_args: launch,
    )
    monkeypatch.setattr(
        fit_probe_stage, "load_study_config", lambda *_args, **_kwargs: study
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "validate_materialized_data",
        lambda *_args, **_kwargs: {"direction_fit": [{}], "direction_probe": [{}]},
    )
    monkeypatch.setattr(
        fit_probe_stage, "load_study_model", lambda *_args, **_kwargs: loaded
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "collect_direction_observations",
        lambda *_args, **_kwargs: observation,
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "completion_fit_status",
        lambda *_args, **_kwargs: {
            "eligible": False,
            "fit_source_option_record_counts": {"trivia_qa|B": 0},
            "minimum_records_per_source_option": 10,
        },
    )

    def forbidden_completion(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("ineligible completion direction was computed")

    monkeypatch.setattr(
        fit_probe_stage, "compute_completion_contrast", forbidden_completion
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "observed_valid_data",
        lambda *_args: (torch.ones((1, 1, 2)), [0], ["doubt"]),
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "compute_observed_prompt_direction",
        lambda *_args, **_kwargs: (torch.ones((1, 2)), {"overall": {"0": 1}}),
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "observed_probe_result",
        lambda *_args, **_kwargs: {"overall_auroc": [1.0]},
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "select_estimator_layers",
        lambda *_args, **_kwargs: {
            "chosen_estimator": "observed_prompt_state",
            "chosen_layers": [0],
        },
    )
    monkeypatch.setattr(fit_probe_stage, "record_contract", lambda _records: [])
    monkeypatch.setattr(
        fit_probe_stage, "record_contract_sha256", lambda _records: "c" * 64
    )
    monkeypatch.setattr(
        fit_probe_stage,
        "save_file",
        lambda _tensors, path: Path(path).write_bytes(b"safetensors-stub"),
    )
    monkeypatch.setattr(fit_probe_stage, "build_runtime_manifest", lambda **_kwargs: {})
    monkeypatch.setattr(fit_probe_stage, "unload_study_model", lambda _loaded: None)

    payload = fit_probe_stage.run_fit_probe_stage(
        repository=tmp_path,
        study_path=study_path,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        model_key="m",
        run_kind="scientific",
        limit=None,
    )

    assert payload["estimator_status"]["completion_contrast"]["eligible"] is False
    assert "completion_contrast" not in payload["direction_artifact"]["tensors"]
    assert "completion_contrast" not in payload["probe_results"]
    assert "observed_prompt_state" in payload["direction_artifact"]["tensors"]
