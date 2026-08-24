# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "results" / "EXPLORATORY_ARTIFACT_MANIFEST.json"
SUMMARY = REPOSITORY / "results" / "FOUR_MODEL_EXPLORATORY_FRONTIER.json"
REPORT = SUMMARY.with_suffix(".md")
SCALE_REPORT = REPOSITORY / "results" / "INTERVENTION_SCALE_COMPARISON.json"


def test_exploratory_release_manifest_is_portable_and_bound() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    summary_digest = hashlib.sha256(SUMMARY.read_bytes()).hexdigest()

    assert manifest["schema_version"] == "selective_sycophancy_frontier_release.v2"
    assert manifest["evidence_scope"]["endpoint_results"] == (
        "verified_from_persisted_response_and_metric_primitives"
    )
    assert manifest["evidence_scope"]["layer_selection"] == (
        "held_out_probe_with_five_seeded_random_controls_per_layer"
    )
    assert manifest["summary"]["sha256"] == summary_digest
    assert manifest["summary"]["bytes"] == SUMMARY.stat().st_size
    assert manifest["report"]["bytes"] == REPORT.stat().st_size
    assert (
        manifest["report"]["sha256"] == hashlib.sha256(REPORT.read_bytes()).hexdigest()
    )
    assert manifest["intervention_scale"] == {
        "path": "INTERVENTION_SCALE_COMPARISON.json",
        "bytes": SCALE_REPORT.stat().st_size,
        "sha256": hashlib.sha256(SCALE_REPORT.read_bytes()).hexdigest(),
    }
    assert manifest["redistributed_dataset_content"]["wikitext_2_context_text"] == {
        "embedded": True,
        "license": "CC-BY-SA-3.0",
        "notice": "licenses/WIKITEXT-2-CC-BY-SA-3.0.md",
    }
    assert manifest["redistributed_dataset_content"][
        "gsm8k_reference_and_reconstructed_text"
    ] == {
        "embedded": True,
        "license": "MIT",
        "notice": "licenses/GSM8K-MIT.txt",
    }
    assert set(manifest["notices"]) == {
        "CITATION.cff",
        "DATA_LICENSES.md",
        "LICENSE",
        "MODEL_TERMS.md",
        "NOTICE.md",
        "THIRD_PARTY.md",
        "licenses/GSM8K-MIT.txt",
        "licenses/WIKITEXT-2-CC-BY-SA-3.0.md",
        "paper/references.bib",
    }
    assert set(manifest["models"]) == {
        "Qwen3.5-2B",
        "Qwen3.5-4B",
        "Gemma 4 E2B",
        "Gemma 4 E4B",
    }
    entries = [
        entry
        for model in manifest["models"].values()
        for entry in model["files"].values()
    ]
    assert len(entries) == 20
    for entry in entries:
        path = PurePosixPath(entry["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert int(entry["bytes"]) > 0
        assert len(entry["sha256"]) == 64

    for model in manifest["models"].values():
        provenance = model["provenance"]
        assert provenance["source_launch_class"] == "engineering_smoke"
        assert provenance["source_repository_dirty"] is True
        assert provenance["strict_tagged_pipeline_eligible_at_launch"] is False
        assert isinstance(provenance["repository_commit"], str)
        for kind in ("behavior", "fit", "gsm", "kl"):
            transformed = model["files"][kind]
            assert len(transformed["source_sha256"]) == 64
            assert "evidence scope" in transformed["transformation"]
        assert "<REPOSITORY>" in model["files"]["fit"]["transformation"]

    for entry in manifest["notices"].values():
        path = PurePosixPath(entry["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        source = REPOSITORY.joinpath(*path.parts)
        assert int(entry["bytes"]) == source.stat().st_size
        assert entry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
