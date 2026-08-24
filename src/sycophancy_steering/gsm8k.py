# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sampled GSM8K scoring compatible with the pinned harness filters."""

from __future__ import annotations

import hashlib
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, TypedDict, cast


class GSM8KScore(TypedDict):
    reference: str
    strict_prediction: str | None
    flexible_prediction: str | None
    strict_correct: bool
    flexible_correct: bool


_REFERENCE = re.compile(r"#### (\-?[0-9\.\,]+)")
_STRICT_RESPONSE = re.compile(r"The answer is (\-?[0-9\.\,]+).")
_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


@dataclass(frozen=True)
class GSM8KHarnessContract:
    """Hash-validated vendoring of lm-eval 0.4.12 GSM8K semantics."""

    version: str
    task_name: str
    task_yaml_path: str
    task_yaml_sha256: str
    strict_regex: str
    strict_group_select: int
    flexible_regex: str
    flexible_group_select: int
    exact_match_options: dict[str, Any]

    def _extract(self, response: str, *, strict: bool) -> str:
        pattern = self.strict_regex if strict else self.flexible_regex
        group_select = (
            self.strict_group_select if strict else self.flexible_group_select
        )
        matches = re.findall(pattern, response if isinstance(response, str) else "")
        if not matches:
            return "[invalid]"
        selected: Any = matches[group_select]
        if isinstance(selected, tuple):
            nonempty = [item for item in selected if item]
            selected = nonempty[0] if nonempty else "[invalid]"
        return str(selected).strip()

    def _correct(self, prediction: str, reference: str) -> bool:
        normalized_prediction = prediction
        normalized_reference = reference
        for pattern in self.exact_match_options["regexes_to_ignore"]:
            normalized_prediction = re.sub(pattern, "", normalized_prediction)
            normalized_reference = re.sub(pattern, "", normalized_reference)
        if self.exact_match_options["ignore_case"]:
            normalized_prediction = normalized_prediction.lower()
            normalized_reference = normalized_reference.lower()
        if self.exact_match_options["ignore_punctuation"]:
            table = str.maketrans("", "", string.punctuation)
            normalized_prediction = normalized_prediction.translate(table)
            normalized_reference = normalized_reference.translate(table)
        return normalized_prediction == normalized_reference

    def score(self, response: str, answer: str) -> tuple[str, str, bool, bool]:
        strict = self._extract(response, strict=True)
        flexible = self._extract(response, strict=False)
        return (
            strict,
            flexible,
            self._correct(strict, answer),
            self._correct(flexible, answer),
        )


def load_pinned_harness_contract(spec: dict[str, Any]) -> GSM8KHarnessContract:
    """Load and validate lm-eval 0.4.12's exact GSM8K task contract."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for the GSM8K contract") from error

    reference_version = spec.get("lm_eval_version")
    if reference_version != "0.4.12":
        raise ValueError("Frozen lm-eval reference version differs from 0.4.12")
    task_path = (
        Path(__file__).resolve().parent
        / "contracts"
        / "gsm8k-cot-zeroshot-lm-eval-0.4.12.yaml"
    )
    task_bytes = task_path.read_bytes()
    task_sha256 = hashlib.sha256(task_bytes).hexdigest()
    if task_sha256 != spec.get("lm_eval_task_yaml_sha256"):
        raise ValueError("Installed lm-eval GSM8K task YAML hash mismatch")
    task = yaml.safe_load(task_bytes)
    if not isinstance(task, dict):
        raise ValueError("Pinned lm-eval GSM8K task YAML is invalid")

    filters = task.get("filter_list")
    if not isinstance(filters, list) or len(filters) != 2:
        raise ValueError("Pinned lm-eval GSM8K filter inventory differs")
    filters_by_name = {
        item.get("name"): item.get("filter")
        for item in filters
        if isinstance(item, dict)
    }
    strict_pipeline = filters_by_name.get("strict-match")
    flexible_pipeline = filters_by_name.get("flexible-extract")
    strict_spec = spec.get("strict_filter")
    flexible_spec = spec.get("flexible_filter")
    metric_options = spec.get("exact_match_normalization")
    if not all(
        isinstance(value, dict)
        for value in (strict_spec, flexible_spec, metric_options)
    ):
        raise ValueError("Frozen GSM8K filter/metric contract is invalid")
    strict_spec = cast(dict[str, Any], strict_spec)
    flexible_spec = cast(dict[str, Any], flexible_spec)
    metric_options = cast(dict[str, Any], metric_options)
    metric_list = task.get("metric_list")
    generation = task.get("generation_kwargs")
    if (
        task.get("task") != "gsm8k_cot_zeroshot"
        or task.get("dataset_path") != spec.get("dataset")
        or task.get("dataset_name") != spec.get("configuration")
        or task.get("test_split") != spec.get("split")
        or task.get("doc_to_text") != spec.get("lm_eval_doc_to_text")
        or str(spec.get("lm_eval_doc_to_text", "")).replace(
            "{{question}}", "{question}"
        )
        != spec.get("prompt_template")
        or not isinstance(generation, dict)
        or generation.get("do_sample") is not False
        or generation.get("until") != spec.get("stop_strings")
        or not isinstance(strict_pipeline, list)
        or strict_pipeline
        != [
            {"function": "regex", "regex_pattern": strict_spec.get("regex")},
            {"function": "take_first"},
        ]
        or not isinstance(flexible_pipeline, list)
        or flexible_pipeline
        != [
            {
                "function": "regex",
                "group_select": flexible_spec.get("group_select"),
                "regex_pattern": flexible_spec.get("regex"),
            },
            {"function": "take_first"},
        ]
        or metric_list
        != [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "higher_is_better": True,
                "ignore_case": metric_options.get("ignore_case"),
                "ignore_punctuation": metric_options.get("ignore_punctuation"),
                "regexes_to_ignore": metric_options.get("regexes_to_ignore"),
            }
        ]
        or strict_spec.get("group_select") != 0
    ):
        raise ValueError("Frozen study and lm-eval GSM8K task contract differ")

    options = {
        "ignore_case": bool(metric_options["ignore_case"]),
        "ignore_punctuation": bool(metric_options["ignore_punctuation"]),
        "regexes_to_ignore": list(metric_options["regexes_to_ignore"]),
    }
    return GSM8KHarnessContract(
        version=str(reference_version),
        task_name=str(task["task"]),
        task_yaml_path=(
            "sycophancy_steering/contracts/gsm8k-cot-zeroshot-lm-eval-0.4.12.yaml"
        ),
        task_yaml_sha256=task_sha256,
        strict_regex=str(strict_spec["regex"]),
        strict_group_select=int(strict_spec["group_select"]),
        flexible_regex=str(flexible_spec["regex"]),
        flexible_group_select=int(flexible_spec["group_select"]),
        exact_match_options=options,
    )


def _normalize_number(value: str) -> str:
    normalized = value.replace(",", "").replace("$", "")
    return normalized[:-1] if normalized.endswith(".") else normalized


def reference_answer(answer: str) -> str:
    """Extract the canonical GSM8K answer following the final hash marker."""

    matches = _REFERENCE.findall(answer)
    if not matches:
        raise ValueError("GSM8K reference answer has no #### numeric marker")
    return _normalize_number(matches[-1])


def strict_extract(response: str) -> str | None:
    matches = _STRICT_RESPONSE.findall(response)
    return _normalize_number(matches[0]) if matches else None


def flexible_extract(response: str) -> str | None:
    matches = _FLEXIBLE.findall(response)
    if not matches:
        return None
    first, second = matches[-1]
    return _normalize_number(first or second)


def score_response(
    response: str,
    answer: str,
    *,
    harness: GSM8KHarnessContract | None = None,
) -> GSM8KScore:
    reference = reference_answer(answer)
    if harness is None:
        strict = strict_extract(response)
        flexible = flexible_extract(response)
        strict_correct = strict == reference
        flexible_correct = flexible == reference
    else:
        strict, flexible, strict_correct, flexible_correct = harness.score(
            response, answer
        )
    return {
        "reference": reference,
        "strict_prediction": strict,
        "flexible_prediction": flexible,
        "strict_correct": strict_correct,
        "flexible_correct": flexible_correct,
    }


def select_sample(rows: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    """Hash-order a simple sample without replacement from GSM8K rows."""

    if count <= 0 or count > len(rows):
        raise ValueError("Sample count must be positive and no larger than population")
    candidates: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        question = str(row["question"])
        answer = str(row["answer"])
        digest = hashlib.sha256(
            f"gsm8k-sample-v1\0{source_index}\0{question}\0{answer}".encode()
        ).hexdigest()
        candidates.append(
            {
                **row,
                "source_index": source_index,
                "sample_sha256": digest,
            }
        )
    candidates.sort(key=lambda item: (item["sample_sha256"], item["source_index"]))
    return candidates[:count]


def wilson_interval(
    *, correct: int, total: int, confidence: float
) -> dict[str, float | int]:
    """Two-sided Wilson score interval for a sampled binary accuracy."""

    if total <= 0 or not 0 <= correct <= total or not 0 < confidence < 1:
        raise ValueError("Wilson count or confidence is invalid")
    probability = correct / total
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (probability + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "correct": correct,
        "total": total,
        "accuracy": probability,
        "confidence": confidence,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }
