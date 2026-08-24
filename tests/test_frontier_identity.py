# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sycophancy_steering.frontier_stage import verify_loaded_fingerprint


def test_loaded_model_and_tokenizer_fingerprints_must_match_parent() -> None:
    loaded = SimpleNamespace(
        model_fingerprint={"config_sha256": "model"},
        tokenizer_fingerprint={"vocabulary_sha256": "tokenizer"},
    )
    artifact = {
        "runtime": {
            "model_fingerprint": {"config_sha256": "model"},
            "tokenizer_fingerprint": {"vocabulary_sha256": "tokenizer"},
        }
    }

    verify_loaded_fingerprint(loaded, artifact, label="parent")

    artifact["runtime"]["tokenizer_fingerprint"] = {"vocabulary_sha256": "tampered"}
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        verify_loaded_fingerprint(loaded, artifact, label="parent")
