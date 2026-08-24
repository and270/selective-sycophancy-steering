# SPDX-License-Identifier: AGPL-3.0-or-later

"""Engineering-only smoke for fixed-path neutral trajectory KL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sycophancy_steering.config import load_study_config
from sycophancy_steering.kl_stage import (
    _evaluate_trial,
    _generate_base_trajectories,
)
from sycophancy_steering.loading import load_study_model, unload_study_model

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = REPOSITORY / "configs/studies/multimodel_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    study = load_study_config(STUDY, require_frozen=False)
    spec = study["models"][args.model_key]
    contexts = [
        {
            "row_index": 0,
            "sha256": "synthetic-neutral-context-0",
            "text": (
                "The observatory recorded clear skies during the evening. "
                "Researchers compared the new measurements with the prior log. "
            )
            * 4,
        },
        {
            "row_index": 1,
            "sha256": "synthetic-neutral-context-1",
            "text": (
                "A library catalog lists books by subject and publication year. "
                "The updated index was checked against the archived copy. "
            )
            * 4,
        },
    ]
    loaded = load_study_model(
        args.model_key,
        spec,
        seed=0,
        local_files_only=True,
    )
    try:
        trajectories = _generate_base_trajectories(
            loaded,
            contexts,
            context_maximum_tokens=128,
            continuation_maximum_tokens=3,
        )
        layer = len(loaded.layers) // 2
        direction = torch.ones(spec["expected_hidden_size"], dtype=torch.float32)
        direction /= math.sqrt(direction.numel())
        zero = _evaluate_trial(
            loaded,
            contexts,
            trajectories,
            context_maximum_tokens=128,
            layer=layer,
            direction=direction,
            alpha=0.0,
            zero_alpha_tolerance=1.0e-6,
            bootstrap_iterations=100,
            bootstrap_seed=7,
            confidence=0.95,
        )
        nonzero = _evaluate_trial(
            loaded,
            contexts,
            trajectories,
            context_maximum_tokens=128,
            layer=layer,
            direction=direction,
            alpha=0.01,
            zero_alpha_tolerance=1.0e-6,
            bootstrap_iterations=100,
            bootstrap_seed=7,
            confidence=0.95,
        )
        if zero["maximum_absolute_logit_difference"] != 0.0:
            raise RuntimeError("Zero-alpha trajectory identity failed")
        if nonzero["token_micro"]["forward_kl_nats"]["mean"] <= 0.0:
            raise RuntimeError("Nonzero trajectory KL was not positive")
        print(
            json.dumps(
                {
                    "run_kind": "engineering_smoke_only",
                    "synthetic_contexts_only": True,
                    "model_key": args.model_key,
                    "layer": layer,
                    "generation_eos_token_ids": loaded.generation_eos_token_ids,
                    "trajectory_token_counts": [
                        item["continuation_token_count"] for item in trajectories
                    ],
                    "zero_max_logit_difference": zero[
                        "maximum_absolute_logit_difference"
                    ],
                    "nonzero_token_micro_kl": nonzero["token_micro"]["forward_kl_nats"][
                        "mean"
                    ],
                    "nonzero_hook_audit": nonzero["hook_audit"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        unload_study_model(loaded)


if __name__ == "__main__":
    raise SystemExit(main())
