# SPDX-License-Identifier: AGPL-3.0-or-later

"""Neutral-context distribution-shift metrics for runtime steering."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor


def distribution_metrics_from_logits(
    base_logits: Tensor, condition_logits: Tensor
) -> dict[str, Tensor]:
    """Return per-context forward KL, Jensen-Shannon, and top-1 agreement."""

    if (
        base_logits.ndim != 2
        or condition_logits.shape != base_logits.shape
        or base_logits.shape[0] == 0
    ):
        raise ValueError("Logits must share non-empty (context, vocabulary) shape")
    if (
        not torch.isfinite(base_logits).all()
        or not torch.isfinite(condition_logits).all()
    ):
        raise ValueError("Distribution logits must be finite")
    base = base_logits.to(dtype=torch.float64)
    condition = condition_logits.to(dtype=torch.float64)
    log_p = torch.log_softmax(base, dim=-1)
    log_q = torch.log_softmax(condition, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    forward_kl = (p * (log_p - log_q)).sum(dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    jensen_shannon = 0.5 * (
        (p * (log_p - log_m)).sum(dim=-1) + (q * (log_q - log_m)).sum(dim=-1)
    )
    tolerance = 1.0e-14
    if torch.any(forward_kl < -tolerance) or torch.any(jensen_shannon < -tolerance):
        raise FloatingPointError("KL/JS became materially negative in float64")
    forward_kl = torch.where(forward_kl.abs() <= tolerance, 0.0, forward_kl)
    jensen_shannon = torch.where(jensen_shannon.abs() <= tolerance, 0.0, jensen_shannon)
    agreement = (base.argmax(dim=-1) == condition.argmax(dim=-1)).to(torch.float64)
    return {
        "forward_kl": forward_kl,
        "jensen_shannon": jensen_shannon,
        "top1_agreement": agreement,
    }


def _canonical_text(text: str) -> str:
    return " ".join(text.split())


def select_neutral_contexts(
    rows: list[str],
    *,
    count: int,
    minimum_characters: int,
) -> list[dict[str, Any]]:
    """Select a deterministic hash-ordered subset of non-heading WikiText rows."""

    if count <= 0 or minimum_characters <= 0:
        raise ValueError("Context count and minimum length must be positive")
    eligible: list[dict[str, Any]] = []
    for row_index, raw in enumerate(rows):
        text = _canonical_text(str(raw))
        if not text or (text.startswith("=") and text.endswith("=")):
            continue
        if len(text) < minimum_characters:
            continue
        digest = hashlib.sha256(
            f"neutral-kl-v1\0{row_index}\0{text}".encode()
        ).hexdigest()
        eligible.append({"row_index": row_index, "text": text, "sha256": digest})
    eligible.sort(key=lambda item: (item["sha256"], item["row_index"]))
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} neutral contexts meet the frozen selection rule"
        )
    return eligible[:count]
