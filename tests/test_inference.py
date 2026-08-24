# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import torch
from torch import nn

from sycophancy_steering.inference import (
    extract_last_token_residuals,
    generate_binary_answers,
    render_chat_texts,
)


class Batch(dict):
    def to(self, device: str):  # type: ignore[no-untyped-def]
        return Batch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, chats, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs.get("enable_thinking") is False
        return [f"PROMPT:{chat[-1]['content']}" for chat in chats]

    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["padding"] is True
        assert kwargs["return_token_type_ids"] is False
        if texts == ["first", "second"]:
            return Batch(
                {
                    "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                }
            )
        return Batch(
            {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.ones(2, 2, dtype=torch.long),
            }
        )

    def batch_decode(self, tokens, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["skip_special_tokens"] is True
        return ["A" if int(row[0]) == 7 else "invalid" for row in tokens]


class GenerateModel(nn.Module):
    def generate(self, input_ids, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is True
        assert kwargs["eos_token_id"] == [99]
        assert kwargs["pad_token_id"] == 0
        assert kwargs["max_new_tokens"] == 4
        suffix = torch.tensor([[7], [8]], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class AddLayer(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.amount


class ResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AddLayer(1.0), AddLayer(2.0)])

    def forward(self, input_ids, attention_mask, **kwargs):  # type: ignore[no-untyped-def]
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden)
        return {"last_hidden_state": hidden}


def test_render_chat_texts_uses_frozen_template_kwargs_and_completions() -> None:
    tokenizer = FakeTokenizer()
    chats = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]

    rendered = render_chat_texts(
        tokenizer,
        chats,
        chat_template_kwargs={"enable_thinking": False},
        completions=["A", "B"],
    )

    assert rendered == ["PROMPT:oneA", "PROMPT:twoB"]


def test_generate_binary_answers_preserves_raw_and_strict_parsed_text() -> None:
    answers = generate_binary_answers(
        GenerateModel(),
        FakeTokenizer(),
        [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
        chat_template_kwargs={"enable_thinking": False},
        batch_size=2,
        device="cpu",
        max_new_tokens=4,
        eos_token_ids=[99],
    )

    assert answers == [
        {"text": "A", "parsed": "A"},
        {"text": "invalid", "parsed": None},
    ]


def test_extract_residuals_uses_last_nonpadding_position_per_row() -> None:
    model = ResidualModel()

    residuals = extract_last_token_residuals(
        model,
        FakeTokenizer(),
        model.layers,
        ["first", "second"],
        batch_size=2,
        device="cpu",
    )

    assert tuple(residuals.shape) == (2, 2, 1)
    torch.testing.assert_close(
        residuals,
        torch.tensor([[[3.0], [5.0]], [[6.0], [8.0]]]),
    )
