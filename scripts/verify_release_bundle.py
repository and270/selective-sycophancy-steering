# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify every hash-bound member of the published study evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

EMBEDDED_MANIFEST = "EXPLORATORY_ARTIFACT_MANIFEST.json"
DEFAULT_MANIFEST = Path("results/EXPLORATORY_ARTIFACT_MANIFEST.json")


def _entry(entry: Any, *, label: str) -> tuple[str, int, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} entry is not an object")
    path = entry.get("path")
    size = entry.get("bytes")
    digest = entry.get("sha256")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} path is invalid")
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} byte count is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    return path, size, digest


def expected_members(manifest: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """Return the exact archive inventory declared by a release manifest."""

    entries: list[tuple[str, tuple[str, int, str]]] = []
    for key in ("summary", "report", "intervention_scale"):
        entries.append((key, _entry(manifest.get(key), label=key)))

    notices = manifest.get("notices")
    if not isinstance(notices, dict) or not notices:
        raise ValueError("Manifest notices are missing")
    for name, entry in notices.items():
        entries.append((f"notice:{name}", _entry(entry, label=f"notice:{name}")))

    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Manifest models are missing")
    for model_name, model in models.items():
        files = model.get("files") if isinstance(model, dict) else None
        if not isinstance(files, dict) or not files:
            raise ValueError(f"Manifest files are missing for {model_name}")
        for kind, entry in files.items():
            label = f"model:{model_name}:{kind}"
            entries.append((label, _entry(entry, label=label)))

    result: dict[str, tuple[int, str]] = {}
    for label, (path, size, digest) in entries:
        if path in result:
            raise ValueError(f"Duplicate manifest path: {path} ({label})")
        result[path] = (size, digest)
    return result


def _member_sha256(bundle: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with bundle.open(name, "r") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(
    archive: Path,
    *,
    tracked_manifest: Path | None = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Verify archive inventory, member sizes/hashes, and manifest identity."""

    archive = archive.resolve()
    with zipfile.ZipFile(archive, "r") as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member names")
        if EMBEDDED_MANIFEST not in names:
            raise ValueError("Archive has no embedded release manifest")

        manifest = json.loads(bundle.read(EMBEDDED_MANIFEST))
        if not isinstance(manifest, dict):
            raise ValueError("Embedded release manifest is not an object")
        if tracked_manifest is not None:
            tracked = json.loads(tracked_manifest.resolve().read_text(encoding="utf-8"))
            if manifest != tracked:
                raise ValueError("Embedded and tracked release manifests differ")

        expected = expected_members(manifest)
        expected_names = set(expected) | {EMBEDDED_MANIFEST}
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise ValueError(
                f"Archive inventory mismatch: missing={missing}, extra={extra}"
            )

        info_by_name = {info.filename: info for info in infos}
        for name, (expected_size, expected_digest) in expected.items():
            actual_size = info_by_name[name].file_size
            if actual_size != expected_size:
                raise ValueError(
                    f"Archive member size mismatch for {name}: "
                    f"expected {expected_size}, found {actual_size}"
                )
            if _member_sha256(bundle, name) != expected_digest:
                raise ValueError(f"Archive member SHA-256 mismatch: {name}")

    archive_digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            archive_digest.update(chunk)
    return {
        "archive": archive.as_posix(),
        "bytes": archive.stat().st_size,
        "member_count": len(names),
        "sha256": archive_digest.hexdigest(),
        "status": "verified",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Tracked manifest to compare with the embedded copy",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_bundle(args.archive, tracked_manifest=args.manifest),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
