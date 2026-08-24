# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived in part from Heretic's activation-steering fit/probe runner.

"""Direction-observation collection and held-out probe computation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .directions import (
    binary_auroc,
    projection_scores,
    random_direction_auroc_thresholds,
)
from .inference import (
    extract_last_token_residuals,
    generate_binary_answers,
    render_chat_texts,
)
from .prompts import build_baseline_chat, build_followup_chat


@dataclass
class DirectionObservations:
    baseline_answers: dict[str, dict[str, str | None]]
    descriptors: list[dict[str, str]]
    followup_answers: list[dict[str, str | None]]
    prompt_residuals: Tensor
    caving_residuals: Tensor
    resisting_residuals: Tensor


def answers_by_id(
    records: list[dict[str, Any]],
    answers: list[dict[str, str | None]],
) -> dict[str, dict[str, str | None]]:
    if len(records) != len(answers):
        raise ValueError("Record/answer count mismatch")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Record ids must be unique")
    return dict(zip(ids, answers, strict=True))


def build_direction_chats(
    records: list[dict[str, Any]],
    baseline_answers: dict[str, dict[str, str | None]],
    contract: dict[str, Any],
) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    """Expand base-eligible records into matched pressure examples."""

    if set(baseline_answers) != {str(record["id"]) for record in records}:
        raise ValueError("Baseline answers do not align with direction records")
    modes = tuple(contract["pressure_modes"])
    descriptors: list[dict[str, str]] = []
    chats: list[list[dict[str, str]]] = []
    for record in records:
        record_id = str(record["id"])
        answer = baseline_answers[record_id]
        if answer["parsed"] != record["correct_option"]:
            continue
        baseline_text = answer["text"]
        if not isinstance(baseline_text, str) or not baseline_text:
            raise ValueError("Eligible baseline answer has no response text")
        for mode in modes:
            descriptors.append(
                {
                    "record_id": record_id,
                    "source_dataset": str(record["source_dataset"]),
                    "correct_option": str(record["correct_option"]),
                    "mode": mode,
                    "caving_completion": str(record["wrong_option"]),
                    "resisting_completion": str(record["correct_option"]),
                }
            )
            chats.append(
                build_followup_chat(
                    record,
                    baseline_response=baseline_text,
                    mode=mode,
                    contract=contract,
                )
            )
    return descriptors, chats


def collect_direction_observations(
    loaded: Any,
    records: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    chat_template_kwargs: dict[str, Any],
    generation_batch_size: int,
    residual_batch_size: int,
    max_new_tokens: int,
) -> DirectionObservations:
    """Generate labels and capture prompt/forced-completion residuals."""

    baseline_chats = [build_baseline_chat(record, contract) for record in records]
    baseline_texts = render_chat_texts(
        loaded.tokenizer,
        baseline_chats,
        chat_template_kwargs=chat_template_kwargs,
    )
    baseline_list = generate_binary_answers(
        loaded.model,
        loaded.tokenizer,
        baseline_chats,
        chat_template_kwargs=chat_template_kwargs,
        batch_size=generation_batch_size,
        device=loaded.device,
        max_new_tokens=max_new_tokens,
        eos_token_ids=loaded.generation_eos_token_ids,
    )
    baseline_answers = answers_by_id(records, baseline_list)
    for record, text in zip(records, baseline_texts, strict=True):
        baseline_answers[str(record["id"])]["prompt_sha256"] = hashlib.sha256(
            text.encode()
        ).hexdigest()
    descriptors, followup_chats = build_direction_chats(
        records, baseline_answers, contract
    )
    if not descriptors:
        raise ValueError("Direction split has no base-eligible examples")
    followup_answers = generate_binary_answers(
        loaded.model,
        loaded.tokenizer,
        followup_chats,
        chat_template_kwargs=chat_template_kwargs,
        batch_size=generation_batch_size,
        device=loaded.device,
        max_new_tokens=max_new_tokens,
        eos_token_ids=loaded.generation_eos_token_ids,
    )
    prompt_texts = render_chat_texts(
        loaded.tokenizer,
        followup_chats,
        chat_template_kwargs=chat_template_kwargs,
    )
    caving_texts = render_chat_texts(
        loaded.tokenizer,
        followup_chats,
        chat_template_kwargs=chat_template_kwargs,
        completions=[item["caving_completion"] for item in descriptors],
    )
    resisting_texts = render_chat_texts(
        loaded.tokenizer,
        followup_chats,
        chat_template_kwargs=chat_template_kwargs,
        completions=[item["resisting_completion"] for item in descriptors],
    )
    for descriptor, prompt, caving, resisting in zip(
        descriptors,
        prompt_texts,
        caving_texts,
        resisting_texts,
        strict=True,
    ):
        descriptor["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        descriptor["caving_text_sha256"] = hashlib.sha256(caving.encode()).hexdigest()
        descriptor["resisting_text_sha256"] = hashlib.sha256(
            resisting.encode()
        ).hexdigest()
    extraction = {
        "model": loaded.model,
        "tokenizer": loaded.tokenizer,
        "layers": loaded.layers,
        "batch_size": residual_batch_size,
        "device": loaded.device,
    }
    prompt_residuals = extract_last_token_residuals(texts=prompt_texts, **extraction)
    caving_residuals = extract_last_token_residuals(texts=caving_texts, **extraction)
    resisting_residuals = extract_last_token_residuals(
        texts=resisting_texts, **extraction
    )
    return DirectionObservations(
        baseline_answers=baseline_answers,
        descriptors=descriptors,
        followup_answers=followup_answers,
        prompt_residuals=prompt_residuals,
        caving_residuals=caving_residuals,
        resisting_residuals=resisting_residuals,
    )


def observed_valid_data(
    observations: DirectionObservations,
) -> tuple[Tensor, Tensor, list[str]]:
    valid_indices: list[int] = []
    labels: list[bool] = []
    modes: list[str] = []
    if len(observations.descriptors) != len(observations.followup_answers):
        raise ValueError("Observed descriptors and answers do not align")
    for index, (descriptor, answer) in enumerate(
        zip(observations.descriptors, observations.followup_answers, strict=True)
    ):
        parsed = answer["parsed"]
        if parsed is None:
            continue
        valid_indices.append(index)
        labels.append(parsed == descriptor["caving_completion"])
        modes.append(descriptor["mode"])
    if not valid_indices:
        raise ValueError("Observed-state data contains no valid followup answers")
    return (
        observations.prompt_residuals[valid_indices],
        torch.tensor(labels, dtype=torch.bool),
        modes,
    )


def source_option_record_counts(
    observations: DirectionObservations,
    *,
    source_datasets: tuple[str, ...],
) -> dict[str, int]:
    if not source_datasets or len(source_datasets) != len(set(source_datasets)):
        raise ValueError("Source strata must be unique and non-empty")
    record_strata: dict[str, tuple[str, str]] = {}
    for descriptor in observations.descriptors:
        record_id = descriptor.get("record_id")
        source = descriptor.get("source_dataset")
        correct_option = descriptor.get("correct_option")
        if (
            not isinstance(record_id, str)
            or source not in source_datasets
            or correct_option not in {"A", "B"}
        ):
            raise ValueError("Direction descriptor subgroup fields are invalid")
        stratum = (source, correct_option)
        if record_id in record_strata and record_strata[record_id] != stratum:
            raise ValueError("Direction record has inconsistent subgroup fields")
        record_strata[record_id] = stratum
    return {
        f"{source}|{option}": sum(
            stratum == (source, option) for stratum in record_strata.values()
        )
        for source in source_datasets
        for option in ("A", "B")
    }


def completion_fit_status(
    observations: DirectionObservations,
    *,
    source_datasets: tuple[str, ...],
    minimum_records_per_source_option: int,
) -> dict[str, Any]:
    if (
        not isinstance(minimum_records_per_source_option, int)
        or isinstance(minimum_records_per_source_option, bool)
        or minimum_records_per_source_option < 0
    ):
        raise ValueError("Completion fit subgroup threshold is invalid")
    counts = source_option_record_counts(observations, source_datasets=source_datasets)
    eligible = all(
        count >= minimum_records_per_source_option for count in counts.values()
    )
    status: dict[str, Any] = {
        "eligible": eligible,
        "fit_source_option_record_counts": counts,
    }
    if not eligible:
        status["reason"] = "Insufficient eligible fit records in a source-option cell"
    return status


def _layer_aurocs(scores: Tensor, labels: Tensor) -> list[float]:
    return [binary_auroc(scores[:, layer], labels) for layer in range(scores.shape[1])]


def observed_probe_result(
    observations: DirectionObservations,
    direction: Tensor,
    *,
    modes: tuple[str, ...],
    random_controls: int,
    random_seed: int,
) -> dict[str, Any]:
    residuals, labels, item_modes = observed_valid_data(observations)
    scores = projection_scores(residuals, direction)
    by_mode: dict[str, list[float | None]] = {}
    for mode in modes:
        mask = torch.tensor([item_mode == mode for item_mode in item_modes])
        mode_labels = labels[mask]
        if mode_labels.numel() == 0 or mode_labels.all() or (~mode_labels).all():
            by_mode[mode] = [None] * scores.shape[1]
        else:
            mode_aurocs: list[float | None] = [
                float(value) for value in _layer_aurocs(scores[mask], mode_labels)
            ]
            by_mode[mode] = mode_aurocs
    random_q95, random_max_q95 = random_direction_auroc_thresholds(
        residuals,
        labels,
        direction,
        controls=random_controls,
        seed=random_seed,
        quantile=0.95,
    )
    return {
        "overall_auroc": _layer_aurocs(scores, labels),
        "by_mode_auroc": by_mode,
        "random_control_q95": random_q95,
        "random_control_max_over_layers_q95": random_max_q95,
        "n_valid": len(labels),
        "n_caved": int(labels.sum().item()),
        "n_resisted": int((~labels).sum().item()),
    }


def _completion_group_aurocs(
    scores: Tensor,
    labels: Tensor,
    values: list[str],
    expected_values: tuple[str, ...],
) -> dict[str, list[float | None]]:
    output: dict[str, list[float | None]] = {}
    for value in expected_values:
        mask = torch.tensor([item == value for item in values], dtype=torch.bool)
        if not mask.any():
            output[value] = [None] * scores.shape[1]
        else:
            output[value] = [
                float(auroc) for auroc in _layer_aurocs(scores[mask], labels[mask])
            ]
    return output


def completion_probe_result(
    observations: DirectionObservations,
    direction: Tensor,
    *,
    modes: tuple[str, ...],
    source_datasets: tuple[str, ...],
    random_controls: int,
    random_seed: int,
) -> dict[str, Any]:
    positive_scores = projection_scores(observations.caving_residuals, direction)
    negative_scores = projection_scores(observations.resisting_residuals, direction)
    scores = torch.cat([positive_scores, negative_scores], dim=0)
    labels = torch.cat(
        [
            torch.ones(len(positive_scores), dtype=torch.bool),
            torch.zeros(len(negative_scores), dtype=torch.bool),
        ]
    )
    source_option_counts = source_option_record_counts(
        observations, source_datasets=source_datasets
    )
    item_modes = [item["mode"] for item in observations.descriptors]
    item_sources = [item["source_dataset"] for item in observations.descriptors]
    item_options = [item["correct_option"] for item in observations.descriptors]
    item_source_options = [
        f"{source}|{option}"
        for source, option in zip(item_sources, item_options, strict=True)
    ]
    by_mode = _completion_group_aurocs(
        scores, labels, [*item_modes, *item_modes], modes
    )
    by_option = _completion_group_aurocs(
        scores, labels, [*item_options, *item_options], ("A", "B")
    )
    by_source = _completion_group_aurocs(
        scores, labels, [*item_sources, *item_sources], source_datasets
    )
    source_option_groups = tuple(
        f"{source}|{option}" for source in source_datasets for option in ("A", "B")
    )
    by_source_option = _completion_group_aurocs(
        scores,
        labels,
        [*item_source_options, *item_source_options],
        source_option_groups,
    )
    residuals = torch.cat(
        [observations.caving_residuals, observations.resisting_residuals], dim=0
    )
    random_q95, random_max_q95 = random_direction_auroc_thresholds(
        residuals,
        labels,
        direction,
        controls=random_controls,
        seed=random_seed,
        quantile=0.95,
    )
    return {
        "overall_auroc": _layer_aurocs(scores, labels),
        "by_mode_auroc": by_mode,
        "by_correct_option_auroc": by_option,
        "by_source_auroc": by_source,
        "by_source_option_auroc": by_source_option,
        "source_option_record_counts": source_option_counts,
        "random_control_q95": random_q95,
        "random_control_max_over_layers_q95": random_max_q95,
        "n_pairs": len(observations.descriptors),
    }
