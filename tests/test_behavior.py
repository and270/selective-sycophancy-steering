# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import sycophancy_steering.behavior as behavior

CONTRACT = {
    "baseline_template": "{question}\nA) {option_a}\nB) {option_b}",
    "pressure_modes": ["doubt", "authority", "wrong_suggest"],
    "pressure_templates": {
        "doubt": ["D", "D", "D"],
        "authority": ["X", "X", "X"],
        "wrong_suggest": ["W {wrong_option}", "W {wrong_option}", "W {wrong_option}"],
        "correct_suggest": [
            "C {correct_option}",
            "C {correct_option}",
            "C {correct_option}",
        ],
    },
}
RECORDS = [
    {
        "id": "r1",
        "source_dataset": "trivia_qa",
        "question": "Q1",
        "options": {"A": "yes", "B": "no"},
        "correct_option": "A",
        "wrong_option": "B",
        "pressure_variant": {"doubt": 0, "authority": 0, "wrong_suggest": 0},
    },
    {
        "id": "r2",
        "source_dataset": "truthful_qa",
        "question": "Q2",
        "options": {"A": "yes", "B": "no"},
        "correct_option": "B",
        "wrong_option": "A",
        "pressure_variant": {"doubt": 0, "authority": 0, "wrong_suggest": 0},
    },
    {
        "id": "r3",
        "source_dataset": "trivia_qa",
        "question": "Q3",
        "options": {"A": "yes", "B": "no"},
        "correct_option": "A",
        "wrong_option": "B",
        "pressure_variant": {"doubt": 0, "authority": 0, "wrong_suggest": 0},
    },
]
BASE = {
    "r1": {"text": "A", "parsed": "A"},
    "r2": {"text": "B", "parsed": "B"},
    "r3": {"text": "", "parsed": None},
}


def test_collect_behavior_uses_frozen_denominators_and_both_update_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = [
        # Six pressure answers: r1 then r2, three modes each.
        [
            {"text": "A", "parsed": "A"},
            {"text": "bad", "parsed": None},
            {"text": "B", "parsed": "B"},
            {"text": "B", "parsed": "B"},
            {"text": "B", "parsed": "B"},
            {"text": "B", "parsed": "B"},
        ],
        [{"text": "A", "parsed": "A"}],
        [
            {"text": "A", "parsed": "A"},
            {"text": "bad", "parsed": None},
            {"text": "B", "parsed": "B"},
        ],
    ]

    def fake_generate(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return queued.pop(0)

    class FakeTokenizer:
        def apply_chat_template(self, chats, **_kwargs):  # type: ignore[no-untyped-def]
            return [str(chat) for chat in chats]

    monkeypatch.setattr(behavior, "generate_binary_answers", fake_generate)
    loaded = SimpleNamespace(
        model=object(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        generation_eos_token_ids=[9],
    )

    run = behavior.collect_behavior_run(
        loaded,
        RECORDS,
        BASE,
        CONTRACT,
        chat_template_kwargs={},
        generation_batch_size=2,
        max_new_tokens=1,
        steering=None,
        reuse_base_neutral=True,
    )

    assert queued == []
    assert run.metrics["eligible_ids"] == ["r1", "r2"]
    assert run.metrics["pressure_error_count"] == 2
    assert run.metrics["natural_correct_suggestion_update_rate"] == 1.0
    assert run.metrics["controlled_correction_acceptance_rate"] == 1 / 3
    assert set(run.controlled_correction_answers) == {"r1", "r2", "r3"}


def test_installed_but_unexecuted_hook_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        behavior,
        "generate_binary_answers",
        lambda *_args, **_kwargs: [],
    )
    loaded = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        device="cpu",
        generation_eos_token_ids=[9],
    )

    with pytest.raises((RuntimeError, ValueError)):
        behavior.collect_behavior_run(
            loaded,
            [],
            {},
            CONTRACT,
            chat_template_kwargs={},
            generation_batch_size=1,
            max_new_tokens=1,
            steering=(nn.Identity(), nn.Identity(), torch.ones(2), -1.0),
            reuse_base_neutral=True,
        )
