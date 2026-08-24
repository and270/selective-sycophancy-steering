# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify and package the hash-bound exploratory run artifacts.

The complete run tree is too large for ordinary Git history. This helper turns
its release subset into a portable Zip64 bundle whose model artifacts are
bound by the tracked four-model endpoint summary. Dataset notices travel with
the bundle because the KL artifacts embed selected WikiText-2 context text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPOSITORY / "results" / "FOUR_MODEL_EXPLORATORY_FRONTIER.json"
DEFAULT_SCALE_REPORT = REPOSITORY / "results" / "INTERVENTION_SCALE_COMPARISON.json"
DEFAULT_RUNS_ROOT = REPOSITORY / "results" / "runs"
DEFAULT_MANIFEST = REPOSITORY / "results" / "EXPLORATORY_ARTIFACT_MANIFEST.json"
EXPECTED_SUMMARY_SHA256 = (
    "cf3ff0773b0df82418200576a8c7e09825b82306bfa16c4ac09fb180ab07f421"
)
EXPECTED_REPORT_SHA256 = (
    "560534a35dd84a4622ae77a61e881e71a69d5f29c860fe19e8b07b162403231c"
)
EXPECTED_SCALE_REPORT_SHA256 = (
    "652dc2eeba7d26fa457f51227bd930a50a874add4909ec4018e312ab971af518"
)

FILES = {
    "behavior": Path("expanded_behavior.json"),
    "directions": Path("fit_probe/directions.safetensors"),
    "fit": Path("fit_probe/fit_probe.json"),
    "gsm": Path("expanded_gsm8k.json"),
    "kl": Path("expanded_kl.json"),
}
NOTICE_FILES = {
    "CITATION.cff": REPOSITORY / "CITATION.cff",
    "DATA_LICENSES.md": REPOSITORY / "DATA_LICENSES.md",
    "LICENSE": REPOSITORY / "LICENSE",
    "MODEL_TERMS.md": REPOSITORY / "MODEL_TERMS.md",
    "NOTICE.md": REPOSITORY / "NOTICE.md",
    "THIRD_PARTY.md": REPOSITORY / "THIRD_PARTY.md",
    "licenses/GSM8K-MIT.txt": REPOSITORY / "licenses" / "GSM8K-MIT.txt",
    "licenses/WIKITEXT-2-CC-BY-SA-3.0.md": (
        REPOSITORY / "licenses" / "WIKITEXT-2-CC-BY-SA-3.0.md"
    ),
    "paper/references.bib": REPOSITORY / "paper" / "references.bib",
}
ARCHIVE_TIMESTAMP = (2026, 8, 8, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sanitize_repository_paths(value: Any) -> Any:
    """Replace this checkout's absolute prefix in JSON string values."""

    if isinstance(value, dict):
        return {key: _sanitize_repository_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_repository_paths(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        repository = REPOSITORY.as_posix()
        index = normalized.casefold().find(repository.casefold())
        if index >= 0:
            return (
                normalized[:index]
                + "<REPOSITORY>"
                + normalized[index + len(repository) :]
            )
    return value


def _contains_repository_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_repository_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_repository_path(item) for item in value)
    if isinstance(value, str):
        return REPOSITORY.as_posix().casefold() in value.replace("\\", "/").casefold()
    return False


def packaged_file_bytes(
    kind: str,
    source: Path,
    *,
    evidence_scope: dict[str, Any],
) -> bytes | None:
    """Return portable JSON bytes, or ``None`` for a byte-identical file."""

    if kind == "directions":
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    sanitized = _sanitize_repository_paths(payload)
    if not isinstance(sanitized, dict):
        raise ValueError(f"Expected a JSON object in portable artifact: {source}")
    if _contains_repository_path(sanitized):
        raise ValueError(f"Absolute repository path remains in artifact: {source}")
    if kind == "fit":
        if sanitized.pop("scientific_outputs_allowed", None) is not False:
            raise ValueError(f"Unexpected source launch gate in fit artifact: {source}")
    else:
        if sanitized.pop("scientific_evidence", None) is not False:
            raise ValueError(f"Unexpected legacy endpoint gate in artifact: {source}")
    sanitized["evidence_scope"] = evidence_scope
    content = json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    encoded = content.encode("utf-8")
    return encoded


def packaged_file_transformation(kind: str) -> str | None:
    if kind == "directions":
        return None
    if kind == "fit":
        return (
            "Absolute checkout prefixes in JSON string values replaced with "
            "<REPOSITORY>; legacy source launch-gate field replaced by the "
            "structured release evidence scope; all other parsed values preserved."
        )
    return (
        "Legacy endpoint gate replaced by the structured release evidence scope; "
        "all other parsed values preserved."
    )


def load_summary(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != EXPECTED_SUMMARY_SHA256:
        raise ValueError(
            "Exploratory summary SHA-256 mismatch: "
            f"expected {EXPECTED_SUMMARY_SHA256}, found {digest}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence_scope = payload.get("evidence_scope")
    if not isinstance(evidence_scope, dict):
        raise ValueError("Evidence scope is missing or invalid")
    if evidence_scope.get("endpoint_results") != (
        "verified_from_persisted_response_and_metric_primitives"
    ):
        raise ValueError("The release must identify its verified endpoint evidence")
    if evidence_scope.get("layer_selection") != (
        "held_out_probe_with_five_seeded_random_controls_per_layer"
    ):
        raise ValueError("The release must identify its executed layer screen")
    return payload


def build_manifest(
    summary: dict[str, Any],
    runs_root: Path,
    *,
    summary_path: Path,
    verify: bool,
    scale_report_path: Path = DEFAULT_SCALE_REPORT,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label, model in summary["models"].items():
        run_id = Path(str(model["run_directory"])).name
        run_root = runs_root / run_id
        entries: dict[str, Any] = {}
        for kind, relative in FILES.items():
            path = run_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing exploratory artifact: {path}")
            expected = str(model["artifact_sha256"][kind])
            if verify:
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(
                        f"SHA-256 mismatch for {label} {kind}: "
                        f"expected {expected}, found {actual}"
                    )
            packaged = packaged_file_bytes(
                kind,
                path,
                evidence_scope=summary["evidence_scope"],
            )
            entries[kind] = {
                "path": (Path(run_id) / relative).as_posix(),
                "bytes": len(packaged) if packaged is not None else path.stat().st_size,
                "sha256": (
                    sha256_bytes(packaged) if packaged is not None else expected
                ),
            }
            if packaged is not None:
                entries[kind]["source_sha256"] = expected
                entries[kind]["transformation"] = packaged_file_transformation(kind)
        fit_payload = json.loads((run_root / FILES["fit"]).read_text(encoding="utf-8"))
        runtime = fit_payload.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError(f"Missing fit/probe runtime provenance for {label}")
        provenance = {
            "repository_commit": runtime.get("repository_commit"),
            "source_launch_class": fit_payload.get("run_kind"),
            "source_repository_dirty": runtime.get("repository_dirty"),
            "strict_tagged_pipeline_eligible_at_launch": fit_payload.get(
                "scientific_outputs_allowed"
            ),
            "study_sha256": fit_payload.get("study_sha256"),
            "uv_lock_sha256": runtime.get("uv_lock_sha256"),
        }
        if provenance["source_launch_class"] != "engineering_smoke":
            raise ValueError(f"Unexpected exploratory run kind for {label}")
        if provenance["source_repository_dirty"] is not True:
            raise ValueError(f"Expected dirty exploratory provenance for {label}")
        if provenance["strict_tagged_pipeline_eligible_at_launch"] is not False:
            raise ValueError(f"Invalid exploratory evidence gate for {label}")
        models[label] = {
            "model_key": model["model_key"],
            "run_id": run_id,
            "primary_zero_based_layer": model["primary_zero_based_layer"],
            "provenance": provenance,
            "files": entries,
        }
    notices = {
        arcname: {
            "bytes": source.stat().st_size,
            "path": arcname,
            "sha256": sha256_file(source),
        }
        for arcname, source in NOTICE_FILES.items()
    }
    scale_report_digest = sha256_file(scale_report_path)
    if scale_report_digest != EXPECTED_SCALE_REPORT_SHA256:
        raise ValueError("Intervention-scale report SHA-256 mismatch")
    return {
        "schema_version": "selective_sycophancy_frontier_release.v2",
        "evidence_scope": summary["evidence_scope"],
        "summary": {
            "path": "FOUR_MODEL_EXPLORATORY_FRONTIER.json",
            "bytes": summary_path.stat().st_size,
            "sha256": EXPECTED_SUMMARY_SHA256,
        },
        "report": {
            "path": "FOUR_MODEL_EXPLORATORY_FRONTIER.md",
            "bytes": summary_path.with_suffix(".md").stat().st_size,
            "sha256": EXPECTED_REPORT_SHA256,
        },
        "intervention_scale": {
            "path": "INTERVENTION_SCALE_COMPARISON.json",
            "bytes": scale_report_path.stat().st_size,
            "sha256": EXPECTED_SCALE_REPORT_SHA256,
        },
        "inventory_scope": (
            "Every archive member except the embedded manifest itself has a "
            "path, byte count, and SHA-256 entry here."
        ),
        "provenance_scope": (
            "Endpoint primitives are hash-bound and recomputable. Original "
            "launch metadata are preserved; incomplete clean-commit identifiers "
            "limit exact direction refitting from a known source revision."
        ),
        "redistributed_dataset_content": {
            "gsm8k_reference_and_reconstructed_text": {
                "embedded": True,
                "license": "MIT",
                "notice": "licenses/GSM8K-MIT.txt",
            },
            "wikitext_2_context_text": {
                "embedded": True,
                "license": "CC-BY-SA-3.0",
                "notice": "licenses/WIKITEXT-2-CC-BY-SA-3.0.md",
            },
        },
        "notices": notices,
        "models": models,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    content = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def create_archive(
    archive: Path,
    *,
    summary_path: Path,
    runs_root: Path,
    manifest: dict[str, Any],
    scale_report_path: Path = DEFAULT_SCALE_REPORT,
) -> None:
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest_content = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report_path = summary_path.with_suffix(".md")
    if sha256_file(report_path) != EXPECTED_REPORT_SHA256:
        raise ValueError("Exploratory Markdown report SHA-256 mismatch")
    with zipfile.ZipFile(
        archive,
        mode="x",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as bundle:
        _archive_file(
            bundle,
            summary_path,
            arcname="FOUR_MODEL_EXPLORATORY_FRONTIER.json",
        )
        _archive_file(
            bundle,
            report_path,
            arcname="FOUR_MODEL_EXPLORATORY_FRONTIER.md",
        )
        _archive_file(
            bundle,
            scale_report_path,
            arcname="INTERVENTION_SCALE_COMPARISON.json",
        )
        _archive_bytes(
            bundle,
            manifest_content.encode("utf-8"),
            arcname="EXPLORATORY_ARTIFACT_MANIFEST.json",
        )
        for arcname, source in NOTICE_FILES.items():
            _archive_file(bundle, source, arcname=arcname)
        for model in manifest["models"].values():
            for kind, entry in model["files"].items():
                relative = Path(entry["path"])
                source = runs_root / relative
                packaged = packaged_file_bytes(
                    kind,
                    source,
                    evidence_scope=manifest["evidence_scope"],
                )
                if packaged is None:
                    _archive_file(bundle, source, arcname=relative.as_posix())
                else:
                    if sha256_bytes(packaged) != entry["sha256"]:
                        raise ValueError(f"Packaged file hash drift: {relative}")
                    _archive_bytes(bundle, packaged, arcname=relative.as_posix())


def _zip_info(arcname: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(arcname, date_time=ARCHIVE_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def _archive_bytes(bundle: zipfile.ZipFile, content: bytes, *, arcname: str) -> None:
    bundle.writestr(_zip_info(arcname), content, compress_type=zipfile.ZIP_STORED)


def _archive_file(bundle: zipfile.ZipFile, source: Path, *, arcname: str) -> None:
    info = _zip_info(arcname)
    info.file_size = source.stat().st_size
    with (
        source.open("rb") as source_handle,
        bundle.open(
            info,
            mode="w",
            force_zip64=True,
        ) as archive_handle,
    ):
        shutil.copyfileobj(source_handle, archive_handle, length=8 * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--scale-report", type=Path, default=DEFAULT_SCALE_REPORT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help=(
            "Only for regenerating the tracked size manifest from trusted local files."
        ),
    )
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    scale_report_path = args.scale_report.resolve()
    runs_root = args.runs_root.resolve()
    summary = load_summary(summary_path)
    manifest = build_manifest(
        summary,
        runs_root,
        summary_path=summary_path,
        verify=not args.skip_hash_verification,
        scale_report_path=scale_report_path,
    )
    write_manifest(args.manifest.resolve(), manifest)
    if args.archive is not None:
        if args.skip_hash_verification:
            raise ValueError("Archive creation requires full hash verification")
        create_archive(
            args.archive.resolve(),
            summary_path=summary_path,
            runs_root=runs_root,
            manifest=manifest,
            scale_report_path=scale_report_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
