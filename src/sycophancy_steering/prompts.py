# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived in part from Heretic's factual-sycophancy prompt utilities.

"""Exact factual-sycophancy chat construction and response parsing."""

from __future__ import annotations

from typing import Any


def _options(record: dict[str, Any]) -> dict[str, str]:
    options = record.get("options")
    if not isinstance(options, dict) or set(options) != {"A", "B"}:
        raise ValueError("Binary prompt record must contain exactly A/B options")
    return {"A": str(options["A"]).strip(), "B": str(options["B"]).strip()}


def format_binary_prompt(record: dict[str, Any], contract: dict[str, Any]) -> str:
    """Render an exact two-option factual prompt."""

    options = _options(record)
    template = contract.get("baseline_template")
    if not isinstance(template, str) or not template:
        raise ValueError("Prompt contract has no baseline_template")
    return template.format(
        question=str(record["question"]).strip(),
        option_a=options["A"],
        option_b=options["B"],
    )


def _variant_index(record: dict[str, Any], mode: str) -> int:
    variants = record.get("pressure_variant")
    if not isinstance(variants, dict) or mode not in variants:
        raise ValueError(f"Record is missing its frozen {mode} variant")
    index = variants[mode]
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(f"Invalid {mode} pressure variant index")
    return index


def render_pressure_message(
    record: dict[str, Any],
    mode: str,
    contract: dict[str, Any],
) -> str:
    """Render one preregistered incorrect-pressure message."""

    modes = contract.get("pressure_modes")
    if not isinstance(modes, list) or mode not in modes:
        raise ValueError(f"Unknown pressure mode: {mode}")
    templates = contract.get("pressure_templates", {}).get(mode)
    if not isinstance(templates, list) or not templates:
        raise ValueError(f"Prompt contract has no templates for {mode}")
    index = _variant_index(record, mode)
    if not 0 <= index < len(templates):
        raise ValueError(f"Invalid {mode} pressure variant index")
    return str(templates[index]).format(
        wrong_option=str(record["wrong_option"]),
        correct_option=str(record["correct_option"]),
    )


def render_correct_suggestion_message(
    record: dict[str, Any], contract: dict[str, Any]
) -> str:
    """Render a correct suggestion using the matched wrong-suggestion variant."""

    templates = contract.get("pressure_templates", {}).get("correct_suggest")
    if not isinstance(templates, list) or not templates:
        raise ValueError("Prompt contract has no correct_suggest templates")
    index = _variant_index(record, "wrong_suggest")
    if not 0 <= index < len(templates):
        raise ValueError("Invalid correct-suggestion variant index")
    return str(templates[index]).format(correct_option=str(record["correct_option"]))


def build_baseline_chat(
    record: dict[str, Any], contract: dict[str, Any]
) -> list[dict[str, str]]:
    return [{"role": "user", "content": format_binary_prompt(record, contract)}]


def build_followup_chat(
    record: dict[str, Any],
    *,
    baseline_response: str,
    mode: str,
    contract: dict[str, Any],
) -> list[dict[str, str]]:
    """Build pressure rooted in the model's actual unsteered baseline response."""

    if not isinstance(baseline_response, str) or not baseline_response:
        raise ValueError("baseline_response must be non-empty text")
    return [
        *build_baseline_chat(record, contract),
        {"role": "assistant", "content": baseline_response},
        {
            "role": "user",
            "content": render_pressure_message(record, mode, contract),
        },
    ]


def build_natural_correction_chat(
    record: dict[str, Any],
    *,
    baseline_response: str,
    contract: dict[str, Any],
) -> list[dict[str, str]]:
    """Ask the model to update from its naturally incorrect baseline response."""

    if not isinstance(baseline_response, str):
        raise ValueError("baseline_response must be text")
    return [
        *build_baseline_chat(record, contract),
        {"role": "assistant", "content": baseline_response},
        {
            "role": "user",
            "content": render_correct_suggestion_message(record, contract),
        },
    ]


def build_controlled_correction_chat(
    record: dict[str, Any], *, contract: dict[str, Any]
) -> list[dict[str, str]]:
    """Force a wrong prior answer, then supply the correct suggestion."""

    wrong = str(record.get("wrong_option", ""))
    if wrong not in {"A", "B"}:
        raise ValueError("Record wrong_option must be A or B")
    return [
        *build_baseline_chat(record, contract),
        {"role": "assistant", "content": wrong},
        {
            "role": "user",
            "content": render_correct_suggestion_message(record, contract),
        },
    ]


def parse_binary_letter(text: str) -> str | None:
    """Accept only one whitespace-wrapped canonical A/B response."""

    normalized = text.strip()
    return normalized if normalized in {"A", "B"} else None
