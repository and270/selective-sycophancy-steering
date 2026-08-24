# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from sycophancy_steering.data import record_contract_sha256
from sycophancy_steering.directions import (
    compute_completion_contrast,
    compute_observed_prompt_direction,
)
from sycophancy_steering.fit_probe import (
    DirectionObservations,
    completion_fit_status,
    completion_probe_result,
    observed_probe_result,
    observed_valid_data,
)
from sycophancy_steering.fit_probe_stage import _serializable_observations
from sycophancy_steering.frontier_stage import (
    _verify_behavior_payload,
    _verify_fit_artifact,
    _verify_fit_semantics,
    expected_frontier_conditions,
    verify_frontier_artifact,
)
from sycophancy_steering.metrics import compute_behavior_metrics
from sycophancy_steering.probe import select_estimator_layers


def test_expected_frontier_inventory_is_complete_and_ordered() -> None:
    fit = {
        "layer_selection": {
            "chosen_estimator": "observed_prompt_state",
            "chosen_layers": [3, 7],
        }
    }
    study = {"activation_operator": {"alpha_grid": [-2.0, -1.0, -0.5, 0.0]}}

    assert expected_frontier_conditions(fit, study) == [
        {
            "estimator": "observed_prompt_state",
            "zero_based_layer": 3,
            "alpha": 0.0,
        },
        {
            "estimator": "observed_prompt_state",
            "zero_based_layer": 3,
            "alpha": -2.0,
        },
        {
            "estimator": "observed_prompt_state",
            "zero_based_layer": 3,
            "alpha": -1.0,
        },
        {
            "estimator": "observed_prompt_state",
            "zero_based_layer": 3,
            "alpha": -0.5,
        },
        {
            "estimator": "observed_prompt_state",
            "zero_based_layer": 7,
            "alpha": -2.0,
        },
        {
            "estimator": "observed_prompt_state",
            "zero_based_layer": 7,
            "alpha": -1.0,
        },
        {
            "estimator": "observed_prompt_state",
            "zero_based_layer": 7,
            "alpha": -0.5,
        },
    ]


def _synthetic_observations(records: list[dict[str, object]]) -> DirectionObservations:
    digest = "1" * 64
    baseline = {
        str(record["id"]): {
            "text": str(record["correct_option"]),
            "parsed": str(record["correct_option"]),
            "prompt_sha256": digest,
        }
        for record in records
    }
    descriptors = [
        {
            "record_id": str(record["id"]),
            "source_dataset": str(record["source_dataset"]),
            "correct_option": str(record["correct_option"]),
            "mode": "doubt",
            "caving_completion": str(record["wrong_option"]),
            "resisting_completion": str(record["correct_option"]),
            "prompt_sha256": digest,
            "caving_text_sha256": digest,
            "resisting_text_sha256": digest,
        }
        for record in records
    ]
    followups = [
        {
            "text": str(records[0]["wrong_option"]),
            "parsed": str(records[0]["wrong_option"]),
        },
        {
            "text": str(records[1]["correct_option"]),
            "parsed": str(records[1]["correct_option"]),
        },
    ]
    prompt = torch.tensor(
        [
            [[2.0, 1.0], [1.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    caving = torch.tensor(
        [
            [[3.0, 1.0], [2.0, 1.0]],
            [[2.0, 1.0], [3.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    resisting = torch.zeros_like(caving)
    return DirectionObservations(
        baseline_answers=baseline,
        descriptors=descriptors,
        followup_answers=followups,
        prompt_residuals=prompt,
        caving_residuals=caving,
        resisting_residuals=resisting,
    )


def test_fit_semantics_recompute_directions_and_probe_results() -> None:
    fit_records: list[dict[str, object]] = [
        {
            "id": "fit-1",
            "source_dataset": "trivia_qa",
            "source_parent_split": "direction",
            "correct_option": "A",
            "wrong_option": "B",
            "pressure_variant": {"doubt": 0},
        },
        {
            "id": "fit-2",
            "source_dataset": "truthful_qa",
            "source_parent_split": "direction",
            "correct_option": "B",
            "wrong_option": "A",
            "pressure_variant": {"doubt": 0},
        },
    ]
    probe_records = [
        {**record, "id": str(record["id"]).replace("fit", "probe")}
        for record in fit_records
    ]
    fit = _synthetic_observations(fit_records)
    probe = _synthetic_observations(probe_records)
    completion = compute_completion_contrast(
        fit.caving_residuals, fit.resisting_residuals
    )
    residuals, labels, item_modes = observed_valid_data(fit)
    observed, counts = compute_observed_prompt_direction(
        residuals,
        labels,
        item_modes,
        expected_modes=("doubt",),
        minimum_overall=1,
        minimum_per_mode=1,
    )
    directions = {
        "completion_contrast": completion,
        "observed_prompt_state": observed,
    }
    policy = {
        "random_direction_controls_per_layer": 2,
        "random_seed": 7,
        "candidate_count": 1,
        "observed_gate": {
            "minimum_overall_auroc": 0.5,
            "minimum_evaluable_mode_auroc": 0.5,
            "maximum_unevaluable_modes": 0,
            "must_exceed_random_quantile": False,
            "must_exceed_random_max_quantile": False,
        },
        "completion_gate": {
            "source_datasets": ["trivia_qa", "truthful_qa"],
            "minimum_probe_records_per_source_option": 0,
            "minimum_overall_auroc": 0.5,
            "minimum_each_mode_auroc": 0.5,
            "minimum_each_correct_option_auroc": 0.5,
            "minimum_each_source_auroc": 0.5,
            "minimum_each_source_option_auroc": 0.5,
            "must_exceed_random_quantile": False,
            "must_exceed_random_max_quantile": False,
        },
    }
    probe_results = {
        "completion_contrast": completion_probe_result(
            probe,
            completion,
            modes=("doubt",),
            source_datasets=("trivia_qa", "truthful_qa"),
            random_controls=2,
            random_seed=7,
        ),
        "observed_prompt_state": observed_probe_result(
            probe,
            observed,
            modes=("doubt",),
            random_controls=2,
            random_seed=8,
        ),
    }
    fit_hash = record_contract_sha256(fit_records)  # type: ignore[arg-type]
    probe_hash = record_contract_sha256(probe_records)  # type: ignore[arg-type]
    result = {
        "record_contracts": {
            "direction_fit": fit_records,
            "direction_probe": probe_records,
        },
        "record_contract_sha256": {
            "direction_fit": fit_hash,
            "direction_probe": probe_hash,
        },
        "fit": _serializable_observations(fit),
        "probe": _serializable_observations(probe),
        "estimator_status": {
            "completion_contrast": {
                "eligible": True,
                "fit_source_option_record_counts": {
                    "trivia_qa|A": 1,
                    "trivia_qa|B": 0,
                    "truthful_qa|A": 0,
                    "truthful_qa|B": 1,
                },
            },
            "observed_prompt_state": {
                "eligible": True,
                "fit_class_counts": counts,
            },
        },
        "probe_results": probe_results,
        "layer_selection": select_estimator_layers(
            probe_results, modes=("doubt",), policy=policy
        ),
    }
    tensors = {
        "fit_prompt_residuals": fit.prompt_residuals,
        "fit_caving_residuals": fit.caving_residuals,
        "fit_resisting_residuals": fit.resisting_residuals,
        "probe_prompt_residuals": probe.prompt_residuals,
        "probe_caving_residuals": probe.caving_residuals,
        "probe_resisting_residuals": probe.resisting_residuals,
    }
    study = {
        "prompt_contract": {"pressure_modes": ["doubt"]},
        "direction_estimation": {
            "observed_prompt_state": {
                "minimum_fit_class_count_overall": 1,
                "minimum_fit_class_count_per_mode": 1,
            },
            "completion_contrast": {
                "source_datasets": ["trivia_qa", "truthful_qa"],
                "minimum_fit_eligible_records_per_source_option": 0,
            },
        },
        "layer_selection": policy,
    }
    data_lock = {
        "splits": {
            "direction_fit": {
                "ids": [record["id"] for record in fit_records],
                "record_contract_sha256": fit_hash,
            },
            "direction_probe": {
                "ids": [record["id"] for record in probe_records],
                "record_contract_sha256": probe_hash,
            },
        }
    }

    _verify_fit_semantics(
        result,  # type: ignore[arg-type]
        directions,
        tensors,
        study=study,
        data_lock=data_lock,
        expected_shape=(2, 2),
    )

    ineligible_study = copy.deepcopy(study)
    ineligible_study["direction_estimation"]["completion_contrast"][  # type: ignore[index]
        "minimum_fit_eligible_records_per_source_option"
    ] = 2
    ineligible_probe_results = {
        "observed_prompt_state": probe_results["observed_prompt_state"]
    }
    ineligible_result = copy.deepcopy(result)
    ineligible_result["estimator_status"]["completion_contrast"] = (
        completion_fit_status(
            fit,
            source_datasets=("trivia_qa", "truthful_qa"),
            minimum_records_per_source_option=2,
        )
    )
    ineligible_result["probe_results"] = ineligible_probe_results
    ineligible_result["layer_selection"] = select_estimator_layers(
        ineligible_probe_results, modes=("doubt",), policy=policy
    )
    _verify_fit_semantics(
        ineligible_result,  # type: ignore[arg-type]
        {"observed_prompt_state": observed},
        tensors,
        study=ineligible_study,
        data_lock=data_lock,
        expected_shape=(2, 2),
    )

    forged = copy.deepcopy(result)
    forged["probe_results"]["observed_prompt_state"]["overall_auroc"][0] = 0.0
    with pytest.raises(ValueError, match="Probe results differ"):
        _verify_fit_semantics(
            forged,  # type: ignore[arg-type]
            directions,
            tensors,
            study=study,
            data_lock=data_lock,
            expected_shape=(2, 2),
        )


def test_behavior_verifier_recomputes_metrics_and_rejects_empty_payloads() -> None:
    modes = ("doubt",)
    records = [
        {"id": "r1", "correct_option": "A", "source_dataset": "trivia_qa"},
        {"id": "r2", "correct_option": "B", "source_dataset": "truthful_qa"},
    ]
    neutral = {
        "r1": {"text": "A", "parsed": "A"},
        "r2": {"text": "A", "parsed": "A"},
    }
    pressure = {"r1": {"doubt": {"text": "A", "parsed": "A"}}}
    natural = {"r2": {"text": "B", "parsed": "B"}}
    controlled = {
        "r1": {"text": "A", "parsed": "A"},
        "r2": {"text": "B", "parsed": "B"},
    }
    digest = "0" * 64
    payload = {
        "metrics": compute_behavior_metrics(
            records,
            {"r1": "A", "r2": "A"},
            {"r1": "A", "r2": "A"},
            {"r1": {"doubt": "A"}},
            {"r2": "B"},
            {"r1": "A", "r2": "B"},
            modes=modes,
        ),
        "neutral_answers": neutral,
        "pressure_answers": pressure,
        "natural_correction_answers": natural,
        "controlled_correction_answers": controlled,
        "prompt_hashes": {
            "baseline_generation": {"r1": digest, "r2": digest},
            "neutral": {"r1": digest, "r2": digest},
            "pressure": {"r1": {"doubt": digest}},
            "natural_correction": {"r2": digest},
            "controlled_correction": {"r1": digest, "r2": digest},
        },
        "hook_audit": None,
    }

    verified = _verify_behavior_payload(
        payload,
        records,
        modes=modes,
        base_neutral=None,
        expect_hook=False,
        expect_baseline_generation_hashes=True,
    )
    assert verified.metrics == payload["metrics"]

    forged_metrics = copy.deepcopy(payload)
    forged_metrics["metrics"]["neutral_correct_count"] = 999
    with pytest.raises(ValueError, match="metrics differ"):
        _verify_behavior_payload(
            forged_metrics,
            records,
            modes=modes,
            base_neutral=None,
            expect_hook=False,
            expect_baseline_generation_hashes=True,
        )

    empty = copy.deepcopy(payload)
    empty["neutral_answers"] = {}
    with pytest.raises(ValueError, match="response inventory"):
        _verify_behavior_payload(
            empty,
            records,
            modes=modes,
            base_neutral=None,
            expect_hook=False,
            expect_baseline_generation_hashes=True,
        )


def test_fit_verifier_rejects_incomplete_status_before_other_inputs(
    tmp_path: Path,
) -> None:
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    (fit_dir / "status.json").write_text(
        json.dumps({"complete": False, "stage": "fit"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="status"):
        _verify_fit_artifact(
            fit_dir,
            model_key="qwen35_4b",
            study_path=tmp_path / "missing-study.json",
            data_lock_path=tmp_path / "missing-lock.json",
        )


def test_frontier_verifier_rejects_incomplete_status_before_other_inputs(
    tmp_path: Path,
) -> None:
    frontier_dir = tmp_path / "frontier"
    frontier_dir.mkdir()
    (frontier_dir / "status.json").write_text(
        json.dumps({"complete": False, "stage": "condition"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="status"):
        verify_frontier_artifact(
            frontier_dir,
            model_key="qwen35_4b",
            study_path=tmp_path / "missing-study.json",
            data_lock_path=tmp_path / "missing-lock.json",
            fit_probe_dir=tmp_path / "missing-fit",
            fit_result={},
            directions={},
        )
