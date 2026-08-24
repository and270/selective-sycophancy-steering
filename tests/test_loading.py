# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import torch

from sycophancy_steering.loading import (
    _snapshot_content_fingerprint,
    _verify_checkpoint_content,
    build_quantization_config,
)


def test_snapshot_content_fingerprint_binds_every_file(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights-v1")

    first = _snapshot_content_fingerprint(tmp_path)
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights-v2")
    second = _snapshot_content_fingerprint(tmp_path)

    assert first["content_tree_sha256"] != second["content_tree_sha256"]
    assert [entry["path"] for entry in first["files"]] == [
        "config.json",
        "model-00001.safetensors",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in first["files"])


def test_checkpoint_content_must_match_frozen_tree() -> None:
    model_spec = {
        "expected_checkpoint_file_count": 2,
        "expected_checkpoint_content_tree_sha256": "a" * 64,
    }
    fingerprint = {"file_count": 2, "content_tree_sha256": "a" * 64}

    _verify_checkpoint_content(model_spec, fingerprint)

    fingerprint["content_tree_sha256"] = "b" * 64
    try:
        _verify_checkpoint_content(model_spec, fingerprint)
    except RuntimeError as error:
        assert "content tree" in str(error)
    else:
        raise AssertionError("Modified checkpoint content was accepted")


def test_bf16_model_has_no_quantization_config() -> None:
    assert build_quantization_config({"quantization": None}) is None


def test_e4b_policy_builds_exact_nf4_config() -> None:
    config = build_quantization_config(
        {
            "quantization": {
                "method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            }
        }
    )
    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert config.bnb_4bit_use_double_quant is True
