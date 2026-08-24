# SPDX-License-Identifier: AGPL-3.0-or-later

"""Held-out layer gates and preregistered estimator fallback selection."""

from __future__ import annotations

import math
from typing import Any


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _validate_result(
    estimator: str,
    result: dict[str, Any],
    modes: tuple[str, ...],
) -> int:
    overall = result.get("overall_auroc")
    random = result.get("random_control_q95")
    random_max = result.get("random_control_max_over_layers_q95")
    by_mode = result.get("by_mode_auroc")
    if not isinstance(overall, list) or not overall:
        raise ValueError(f"{estimator} has no per-layer overall AUROC")
    layer_count = len(overall)
    if not isinstance(random, list) or len(random) != layer_count:
        raise ValueError(f"{estimator} random-control layer count differs")
    if not isinstance(random_max, (int, float)) or not math.isfinite(random_max):
        raise ValueError(f"{estimator} random max-statistic threshold is invalid")
    if not isinstance(by_mode, dict) or set(by_mode) != set(modes):
        raise ValueError(f"{estimator} per-mode inventory differs")
    if any(
        not isinstance(by_mode[mode], list) or len(by_mode[mode]) != layer_count
        for mode in modes
    ):
        raise ValueError(f"{estimator} per-mode layer count differs")
    return layer_count


def _completion_subgroups(
    result: dict[str, Any], layer_count: int, gate: dict[str, Any]
) -> tuple[
    dict[str, list[float | None]],
    dict[str, list[float | None]],
    dict[str, list[float | None]],
    dict[str, int],
]:
    sources = gate.get("source_datasets")
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(source, str) for source in sources)
        or len(sources) != len(set(sources))
    ):
        raise ValueError("Completion gate source inventory is invalid")
    expected = (
        {"A", "B"},
        set(sources),
        {f"{source}|{option}" for source in sources for option in ("A", "B")},
    )
    maps: list[dict[str, list[float | None]]] = []
    for field, keys in zip(
        (
            "by_correct_option_auroc",
            "by_source_auroc",
            "by_source_option_auroc",
        ),
        expected,
        strict=True,
    ):
        value = result.get(field)
        if (
            not isinstance(value, dict)
            or set(value) != keys
            or any(
                not isinstance(curve, list) or len(curve) != layer_count
                for curve in value.values()
            )
        ):
            raise ValueError(f"Completion {field} inventory differs")
        maps.append(value)
    counts = result.get("source_option_record_counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != expected[2]
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in counts.values()
        )
    ):
        raise ValueError("Completion source-option count inventory differs")
    return maps[0], maps[1], maps[2], counts


def select_estimator_layers(
    probe_results: dict[str, dict[str, Any]],
    *,
    modes: tuple[str, ...],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Gate every layer, rank candidates, then apply primary/fallback order."""

    if not modes or len(modes) != len(set(modes)):
        raise ValueError("Pressure modes must be unique and non-empty")
    unknown = set(probe_results) - {
        "observed_prompt_state",
        "completion_contrast",
    }
    if unknown:
        raise ValueError(f"Unknown direction estimators: {sorted(unknown)}")
    candidate_count = policy.get("candidate_count")
    if not isinstance(candidate_count, int) or candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    by_estimator: dict[str, Any] = {}
    for estimator, result in probe_results.items():
        layer_count = _validate_result(estimator, result, modes)
        overall = result["overall_auroc"]
        by_mode = result["by_mode_auroc"]
        random = result["random_control_q95"]
        random_max = result["random_control_max_over_layers_q95"]
        gate = policy[
            "observed_gate"
            if estimator == "observed_prompt_state"
            else "completion_gate"
        ]
        completion_diagnostics = (
            _completion_subgroups(result, layer_count, gate)
            if estimator == "completion_contrast"
            else None
        )
        checks: list[dict[str, Any]] = []
        passing: list[int] = []
        for layer in range(layer_count):
            mode_values = {mode: by_mode[mode][layer] for mode in modes}
            subgroup_values: dict[str, dict[str, float | None]] = {}
            subgroup_checks: dict[str, bool] = {}
            if estimator == "observed_prompt_state":
                evaluable = [
                    value for value in mode_values.values() if value is not None
                ]
                unevaluable = len(mode_values) - len(evaluable)
                mode_pass = (
                    unevaluable <= gate["maximum_unevaluable_modes"]
                    and bool(evaluable)
                    and all(
                        _at_least(value, gate["minimum_evaluable_mode_auroc"])
                        for value in evaluable
                    )
                )
                overall_pass = _at_least(overall[layer], gate["minimum_overall_auroc"])
            else:
                if completion_diagnostics is None:
                    raise RuntimeError("Completion diagnostics were not initialized")
                option_curves, source_curves, source_option_curves, counts = (
                    completion_diagnostics
                )
                mode_pass = all(
                    value is not None
                    and _at_least(value, gate["minimum_each_mode_auroc"])
                    for value in mode_values.values()
                )
                overall_pass = _at_least(overall[layer], gate["minimum_overall_auroc"])
                subgroup_values = {
                    "correct_option": {
                        key: curve[layer] for key, curve in option_curves.items()
                    },
                    "source": {
                        key: curve[layer] for key, curve in source_curves.items()
                    },
                    "source_option": {
                        key: curve[layer] for key, curve in source_option_curves.items()
                    },
                }
                thresholds = {
                    "correct_option": gate["minimum_each_correct_option_auroc"],
                    "source": gate["minimum_each_source_auroc"],
                    "source_option": gate["minimum_each_source_option_auroc"],
                }
                subgroup_checks = {
                    name: all(
                        value is not None and _at_least(value, thresholds[name])
                        for value in values.values()
                    )
                    for name, values in subgroup_values.items()
                }
                subgroup_checks["source_option_counts"] = all(
                    count >= gate["minimum_probe_records_per_source_option"]
                    for count in counts.values()
                )
            random_per_layer_pass = (
                overall[layer] > random[layer]
                if gate.get("must_exceed_random_quantile") is True
                else True
            )
            random_familywise_pass = (
                overall[layer] > random_max
                if gate.get("must_exceed_random_max_quantile") is True
                else True
            )
            layer_checks = {
                "overall": overall_pass,
                "modes": mode_pass,
                "random_per_layer": random_per_layer_pass,
                "random_familywise": random_familywise_pass,
                **subgroup_checks,
            }
            passed = all(layer_checks.values())
            check_payload = {
                "layer": layer,
                "mode_aurocs": mode_values,
                "checks": layer_checks,
                "passed": passed,
            }
            if subgroup_values:
                check_payload["subgroup_aurocs"] = subgroup_values
            checks.append(check_payload)
            if passed:
                passing.append(layer)
        candidates = sorted(passing, key=lambda index: (-overall[index], index))[
            :candidate_count
        ]
        by_estimator[estimator] = {
            "layer_gate_checks": checks,
            "passing_layers": passing,
            "candidate_layers": candidates,
        }

    chosen_estimator: str | None = None
    chosen_layers: list[int] = []
    for estimator in ("observed_prompt_state", "completion_contrast"):
        candidates = by_estimator.get(estimator, {}).get("candidate_layers", [])
        if candidates:
            chosen_estimator = estimator
            chosen_layers = list(candidates)
            break
    return {
        "by_estimator": by_estimator,
        "chosen_estimator": chosen_estimator,
        "chosen_layers": chosen_layers,
    }
