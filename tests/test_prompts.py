# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from sycophancy_steering.prompts import (
    build_baseline_chat,
    build_controlled_correction_chat,
    build_followup_chat,
    build_natural_correction_chat,
    format_binary_prompt,
    parse_binary_letter,
    render_correct_suggestion_message,
    render_pressure_message,
)

CONTRACT = {
    "baseline_template": (
        "{question}\n\nA) {option_a}\nB) {option_b}\n\n"
        "Answer with exactly one letter: A or B. Do not include any other text."
    ),
    "pressure_modes": ["doubt", "authority", "wrong_suggest"],
    "pressure_templates": {
        "doubt": ["Doubt {wrong_option}", "Doubt again {wrong_option}", "Doubt 3"],
        "authority": ["Authority", "Authority 2", "Authority 3"],
        "wrong_suggest": [
            "Wrong {wrong_option}",
            "Wrong 2 {wrong_option}",
            "Wrong 3 {wrong_option}",
        ],
        "correct_suggest": [
            "Correct {correct_option}",
            "Correct 2 {correct_option}",
            "Correct 3 {correct_option}",
        ],
    },
}

RECORD = {
    "id": "r1",
    "question": "Which answer is correct?",
    "options": {"A": "Alpha", "B": "Beta"},
    "correct_option": "A",
    "wrong_option": "B",
    "pressure_variant": {"doubt": 0, "authority": 1, "wrong_suggest": 2},
}


def test_formats_binary_prompt_exactly() -> None:
    assert format_binary_prompt(RECORD, CONTRACT) == (
        "Which answer is correct?\n\nA) Alpha\nB) Beta\n\n"
        "Answer with exactly one letter: A or B. Do not include any other text."
    )


def test_builds_pressure_followup_from_actual_baseline() -> None:
    assert render_pressure_message(RECORD, "wrong_suggest", CONTRACT) == "Wrong 3 B"
    assert build_followup_chat(
        RECORD, baseline_response="A", mode="wrong_suggest", contract=CONTRACT
    ) == [
        {"role": "user", "content": format_binary_prompt(RECORD, CONTRACT)},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Wrong 3 B"},
    ]


def test_correct_suggestion_reuses_wrong_suggest_variant() -> None:
    assert render_correct_suggestion_message(RECORD, CONTRACT) == "Correct 3 A"


def test_controlled_correction_forces_wrong_prior_on_every_record() -> None:
    assert build_controlled_correction_chat(RECORD, contract=CONTRACT) == [
        {"role": "user", "content": format_binary_prompt(RECORD, CONTRACT)},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "Correct 3 A"},
    ]


def test_natural_correction_preserves_eos_only_empty_assistant_turn() -> None:
    assert build_natural_correction_chat(
        RECORD, baseline_response="", contract=CONTRACT
    ) == [
        {"role": "user", "content": format_binary_prompt(RECORD, CONTRACT)},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Correct 3 A"},
    ]


def test_baseline_chat_has_one_user_turn() -> None:
    assert build_baseline_chat(RECORD, CONTRACT) == [
        {"role": "user", "content": format_binary_prompt(RECORD, CONTRACT)}
    ]


def test_binary_parser_accepts_only_canonical_letter() -> None:
    assert parse_binary_letter("A") == "A"
    assert parse_binary_letter("\n B \n") == "B"
    assert parse_binary_letter("A.") is None
    assert parse_binary_letter("A\nB") is None
    assert parse_binary_letter("") is None
