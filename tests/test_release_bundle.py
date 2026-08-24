# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.verify_release_bundle import EMBEDDED_MANIFEST, verify_bundle


def _manifest(payload: bytes) -> dict[str, object]:
    entry = {
        "path": "payload.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return {
        "summary": entry,
        "report": {
            "path": "report.md",
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
        "intervention_scale": {
            "path": "scale.json",
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
        "notices": {
            "LICENSE": {
                "path": "LICENSE",
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        },
        "models": {
            "Model": {
                "files": {
                    "behavior": {
                        "path": "model/behavior.json",
                        "bytes": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    }
                }
            }
        },
    }


def _write_bundle(path: Path, manifest: dict[str, object], payload: bytes) -> None:
    members = {
        "payload.json": payload,
        "report.md": b"",
        "scale.json": b"",
        "LICENSE": b"",
        "model/behavior.json": b"",
        EMBEDDED_MANIFEST: (json.dumps(manifest) + "\n").encode(),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)


def test_release_bundle_verifies_inventory_and_hashes(tmp_path: Path) -> None:
    payload = b'{"result": 1}\n'
    manifest = _manifest(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    archive = tmp_path / "release.zip"
    _write_bundle(archive, manifest, payload)

    result = verify_bundle(archive, tracked_manifest=manifest_path)

    assert result["status"] == "verified"
    assert result["member_count"] == 6


def test_release_bundle_rejects_member_tamper(tmp_path: Path) -> None:
    payload = b'{"result": 1}\n'
    manifest = _manifest(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    archive = tmp_path / "release.zip"
    _write_bundle(archive, manifest, b"tampered")

    with pytest.raises(ValueError, match="size mismatch"):
        verify_bundle(archive, tracked_manifest=manifest_path)
