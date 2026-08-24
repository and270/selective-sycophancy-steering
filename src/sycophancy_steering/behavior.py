# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived in part from Heretic's activation-steering behavior runner.

"""Frozen-denominator behavioral collection for steering frontier points."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor, nn

from .fit_probe import answers_by_id, build_direction_chats
from .hooks import SteeringAudit, steer_transformer_layer
from .inference import generate_binary_answers, render_chat_texts
from .metrics import compute_behavior_metrics
from .prompts import (
    build_baseline_chat,
    build_controlled_correction_chat,
    build_natural_correction_chat,
)


@dataclass
class BehaviorRun:
    metrics: dict[str, Any]
    neutral_answers: dict[str, dict[str, str | None]]
    pressure_answers: dict[str, dict[str, dict[str, str | None]]]
    natural_correction_answers: dict[str, dict[str, str | None]]
    controlled_correction_answers: dict[str, dict[str, str | None]]
    prompt_hashes: dict[str, Any]
    hook_audit: dict[str, int] | None


def _hash_chats(
    tokenizer: Any,
    chats: list[list[dict[str, str]]],
    chat_template_kwargs: dict[str, Any],
) -> list[str]:
    texts = render_chat_texts(
        tokenizer, chats, chat_template_kwargs=chat_template_kwargs
    )
    return [hashlib.sha256(text.encode()).hexdigest() for text in texts]


def generate_baseline_answers(
    loaded: Any,
    records: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    chat_template_kwargs: dict[str, Any],
    generation_batch_size: int,
    max_new_tokens: int,
) -> tuple[dict[str, dict[str, str | None]], dict[str, str]]:
    chats = [build_baseline_chat(record, contract) for record in records]
    hashes = _hash_chats(loaded.tokenizer, chats, chat_template_kwargs)
    answers = generate_binary_answers(
        loaded.model,
        loaded.tokenizer,
        chats,
        chat_template_kwargs=chat_template_kwargs,
        batch_size=generation_batch_size,
        device=loaded.device,
        max_new_tokens=max_new_tokens,
        eos_token_ids=loaded.generation_eos_token_ids,
    )
    return answers_by_id(records, answers), {
        str(record["id"]): digest
        for record, digest in zip(records, hashes, strict=True)
    }


def collect_behavior_run(
    loaded: Any,
    records: list[dict[str, Any]],
    base_neutral: dict[str, dict[str, str | None]],
    contract: dict[str, Any],
    *,
    chat_template_kwargs: dict[str, Any],
    generation_batch_size: int,
    max_new_tokens: int,
    steering: tuple[nn.Module, nn.Module, Tensor, float] | None,
    reuse_base_neutral: bool,
) -> BehaviorRun:
    """Collect neutral, pressure, and both correction-control conditions."""

    record_by_id = {str(record["id"]): record for record in records}
    if len(record_by_id) != len(records) or set(base_neutral) != set(record_by_id):
        raise ValueError("Base-neutral identities do not match behavior records")
    context = (
        steer_transformer_layer(
            steering[0], steering[1], steering[2], alpha=steering[3]
        )
        if steering is not None
        else nullcontext(None)
    )
    prompt_hashes: dict[str, Any] = {}
    with context as audit:
        if reuse_base_neutral:
            neutral_answers = base_neutral
            baseline_chats = [
                build_baseline_chat(record, contract) for record in records
            ]
            prompt_hashes["neutral"] = dict(
                zip(
                    record_by_id,
                    _hash_chats(loaded.tokenizer, baseline_chats, chat_template_kwargs),
                    strict=True,
                )
            )
        else:
            neutral_answers, prompt_hashes["neutral"] = generate_baseline_answers(
                loaded,
                records,
                contract,
                chat_template_kwargs=chat_template_kwargs,
                generation_batch_size=generation_batch_size,
                max_new_tokens=max_new_tokens,
            )

        descriptors, pressure_chats = build_direction_chats(
            records, base_neutral, contract
        )
        pressure_hashes = _hash_chats(
            loaded.tokenizer, pressure_chats, chat_template_kwargs
        )
        pressure_list = generate_binary_answers(
            loaded.model,
            loaded.tokenizer,
            pressure_chats,
            chat_template_kwargs=chat_template_kwargs,
            batch_size=generation_batch_size,
            device=loaded.device,
            max_new_tokens=max_new_tokens,
            eos_token_ids=loaded.generation_eos_token_ids,
        )
        pressure_answers: dict[str, dict[str, dict[str, str | None]]] = {}
        pressure_hash_map: dict[str, dict[str, str]] = {}
        for descriptor, answer, digest in zip(
            descriptors, pressure_list, pressure_hashes, strict=True
        ):
            record_id = descriptor["record_id"]
            mode = descriptor["mode"]
            pressure_answers.setdefault(record_id, {})[mode] = answer
            pressure_hash_map.setdefault(record_id, {})[mode] = digest
        prompt_hashes["pressure"] = pressure_hash_map

        natural_records = [
            record
            for record in records
            if base_neutral[str(record["id"])]["parsed"] != record["correct_option"]
        ]
        natural_chats = []
        for record in natural_records:
            text = base_neutral[str(record["id"])]["text"]
            if not isinstance(text, str):
                raise ValueError("Base-ineligible baseline response is not text")
            natural_chats.append(
                build_natural_correction_chat(
                    record, baseline_response=text, contract=contract
                )
            )
        prompt_hashes["natural_correction"] = dict(
            zip(
                (str(record["id"]) for record in natural_records),
                _hash_chats(loaded.tokenizer, natural_chats, chat_template_kwargs),
                strict=True,
            )
        )
        natural_list = generate_binary_answers(
            loaded.model,
            loaded.tokenizer,
            natural_chats,
            chat_template_kwargs=chat_template_kwargs,
            batch_size=generation_batch_size,
            device=loaded.device,
            max_new_tokens=max_new_tokens,
            eos_token_ids=loaded.generation_eos_token_ids,
        )
        natural_answers = answers_by_id(natural_records, natural_list)

        controlled_chats = [
            build_controlled_correction_chat(record, contract=contract)
            for record in records
        ]
        prompt_hashes["controlled_correction"] = dict(
            zip(
                record_by_id,
                _hash_chats(loaded.tokenizer, controlled_chats, chat_template_kwargs),
                strict=True,
            )
        )
        controlled_list = generate_binary_answers(
            loaded.model,
            loaded.tokenizer,
            controlled_chats,
            chat_template_kwargs=chat_template_kwargs,
            batch_size=generation_batch_size,
            device=loaded.device,
            max_new_tokens=max_new_tokens,
            eos_token_ids=loaded.generation_eos_token_ids,
        )
        controlled_answers = answers_by_id(records, controlled_list)

    audit_payload = asdict(audit) if isinstance(audit, SteeringAudit) else None
    if steering is not None and (
        audit_payload is None
        or audit_payload["calls"] <= 0
        or audit_payload["modified_batch_rows"] <= 0
    ):
        raise RuntimeError("Steering hook was installed but did not execute")

    parsed_base = {
        record_id: answer["parsed"] for record_id, answer in base_neutral.items()
    }
    parsed_neutral = {
        record_id: answer["parsed"] for record_id, answer in neutral_answers.items()
    }
    parsed_pressure = {
        record_id: {mode: answer["parsed"] for mode, answer in answers.items()}
        for record_id, answers in pressure_answers.items()
    }
    parsed_natural = {
        record_id: answer["parsed"] for record_id, answer in natural_answers.items()
    }
    parsed_controlled = {
        record_id: answer["parsed"] for record_id, answer in controlled_answers.items()
    }
    metrics = compute_behavior_metrics(
        records,
        parsed_base,
        parsed_neutral,
        parsed_pressure,
        parsed_natural,
        parsed_controlled,
        modes=tuple(contract["pressure_modes"]),
    )
    return BehaviorRun(
        metrics=metrics,
        neutral_answers=neutral_answers,
        pressure_answers=pressure_answers,
        natural_correction_answers=natural_answers,
        controlled_correction_answers=controlled_answers,
        prompt_hashes=prompt_hashes,
        hook_audit=audit_payload,
    )
