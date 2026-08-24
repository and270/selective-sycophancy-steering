# SPDX-License-Identifier: AGPL-3.0-or-later

"""Engineering-only architecture and memory smoke for one frozen model arm."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from sycophancy_steering.config import load_study_config
from sycophancy_steering.hooks import steer_transformer_layer
from sycophancy_steering.inference import (
    extract_last_token_residuals,
    generate_binary_answers,
    render_chat_texts,
)
from sycophancy_steering.loading import load_study_model, unload_study_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--study", type=Path, default=Path("configs/studies/multimodel_v1.json")
    )
    args = parser.parse_args()
    study = load_study_config(args.study, require_frozen=False)
    if args.model_key not in study["models"]:
        raise ValueError(f"Unknown model key: {args.model_key}")
    spec = study["models"][args.model_key]
    batch_size = 4
    synthetic = (
        "This is an engineering canary, not a study item. "
        + "context " * 128
        + "\nA) Alpha\nB) Beta\nAnswer with exactly one letter: A or B."
    )
    chats = [[{"role": "user", "content": synthetic}] for _ in range(batch_size)]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loaded = load_study_model(
        args.model_key, spec, seed=int(study["runtime"]["seed"]), local_files_only=True
    )
    try:
        answers = generate_binary_answers(
            loaded.model,
            loaded.tokenizer,
            chats,
            chat_template_kwargs=spec["chat_template_kwargs"],
            batch_size=batch_size,
            device=loaded.device,
            max_new_tokens=1,
            eos_token_ids=loaded.generation_eos_token_ids,
        )
        texts = render_chat_texts(
            loaded.tokenizer,
            chats,
            chat_template_kwargs=spec["chat_template_kwargs"],
        )
        residuals = extract_last_token_residuals(
            loaded.model,
            loaded.tokenizer,
            loaded.layers,
            texts,
            batch_size=batch_size,
            device=loaded.device,
        )
        width = int(spec["expected_hidden_size"])
        layer_index = len(loaded.layers) // 2
        direction = torch.ones(width, dtype=torch.float32)
        with steer_transformer_layer(
            loaded.text_model,
            loaded.layers[layer_index],
            direction,
            alpha=0.0,
        ) as audit:
            zero_answers = generate_binary_answers(
                loaded.model,
                loaded.tokenizer,
                chats,
                chat_template_kwargs=spec["chat_template_kwargs"],
                batch_size=batch_size,
                device=loaded.device,
                max_new_tokens=1,
                eos_token_ids=loaded.generation_eos_token_ids,
            )
        if zero_answers != answers:
            raise RuntimeError("Zero-alpha engineering identity check failed")
        decode_chats = [
            [
                {
                    "role": "user",
                    "content": (
                        "Output exactly these four lowercase words, "
                        "separated by spaces: alpha beta gamma delta"
                    ),
                }
            ]
        ]
        with steer_transformer_layer(
            loaded.text_model,
            loaded.layers[layer_index],
            direction,
            alpha=0.0,
        ) as decode_audit:
            decode_answers = generate_binary_answers(
                loaded.model,
                loaded.tokenizer,
                decode_chats,
                chat_template_kwargs=spec["chat_template_kwargs"],
                batch_size=1,
                device=loaded.device,
                max_new_tokens=4,
                eos_token_ids=loaded.generation_eos_token_ids,
            )
        if decode_audit.decode_calls <= 0:
            raise RuntimeError("Cached-decode engineering hook did not execute")
        expected_shape = (batch_size, int(spec["expected_transformer_layers"]), width)
        if tuple(residuals.shape) != expected_shape:
            raise RuntimeError(
                f"Residual shape {tuple(residuals.shape)} differs from {expected_shape}"
            )
        payload = {
            "run_kind": "engineering_smoke_only",
            "model_key": args.model_key,
            "model_id": spec["id"],
            "revision": spec["revision"],
            "model_class": loaded.model_class,
            "model_fingerprint": loaded.model_fingerprint,
            "tokenizer_fingerprint": loaded.tokenizer_fingerprint,
            "layer_path": loaded.layer_path,
            "batch_size": batch_size,
            "synthetic_prompt_only": True,
            "answers": answers,
            "residual_shape": list(residuals.shape),
            "residual_dtype": str(residuals.dtype),
            "zero_alpha_identity": True,
            "hook_audit": asdict(audit),
            "decode_canary_answers": decode_answers,
            "decode_hook_audit": asdict(decode_audit),
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        unload_study_model(loaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
