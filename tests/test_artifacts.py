# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch

import sycophancy_steering.artifacts as artifacts
from sycophancy_steering.artifacts import (
    atomic_write_json,
    build_runtime_manifest,
    capture_scientific_launch_identity,
    finalize_artifact_stage,
    require_clean_repository,
    tensor_sha256,
    verify_artifact_manifest,
    verify_parent_launch_identity,
)


def test_atomic_json_is_sorted_lf_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"

    atomic_write_json(path, {"z": 1, "a": [2, 3]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [2, 3], "z": 1}
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()


def test_stage_manifest_binds_completed_payload_files(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "result.json", {"value": 1})
    (tmp_path / "tensor.bin").write_bytes(b"primitive")
    finalize_artifact_stage(
        tmp_path,
        payload_files=("result.json", "tensor.bin"),
        status={"complete": True, "stage": "complete"},
    )
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    verify_artifact_manifest(
        tmp_path,
        status=status,
        expected_files=("result.json", "tensor.bin"),
    )
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content manifest"):
        verify_artifact_manifest(
            tmp_path,
            status=status,
            expected_files=("result.json", "tensor.bin"),
        )


def test_stage_manifest_rejects_unmanifested_extra_files(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "result.json", {"value": 1})
    finalize_artifact_stage(
        tmp_path,
        payload_files=("result.json",),
        status={"complete": True, "stage": "complete"},
    )
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    atomic_write_json(tmp_path / "unmanifested-result.json", {"value": 2})

    with pytest.raises(ValueError, match="content manifest"):
        verify_artifact_manifest(
            tmp_path,
            status=status,
            expected_files=("result.json",),
        )


def test_atomic_json_rejects_nan_without_replacing_existing(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    atomic_write_json(path, {"complete": True})

    with pytest.raises(ValueError):
        atomic_write_json(path, {"bad": float("nan")})

    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}


def test_scientific_repository_gate_requires_clean_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(artifacts, "_git_state", lambda _path: ("abc123", False))
    assert require_clean_repository(tmp_path) == "abc123"

    monkeypatch.setattr(artifacts, "_git_state", lambda _path: ("abc123", True))
    with pytest.raises(RuntimeError, match="clean"):
        require_clean_repository(tmp_path)


def test_parent_launch_identity_must_match() -> None:
    current = {"repository_commit": "same", "tracked_content_sha256": "tree"}
    current["identity_sha256"] = artifacts._launch_identity_sha256(current)
    parent = {"runtime": {"launch_identity": copy.deepcopy(current)}}
    verify_parent_launch_identity(current, parent, label="parent")

    parent["runtime"]["launch_identity"]["repository_commit"] = "different"
    with pytest.raises(ValueError, match="identity"):
        verify_parent_launch_identity(current, parent, label="parent")

    parent["runtime"]["launch_identity"] = copy.deepcopy(current)
    parent["runtime"]["launch_identity"]["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"digest|identity"):
        verify_parent_launch_identity(current, parent, label="parent")


def test_launch_identity_requires_frozen_tag_and_rejects_midrun_dirty_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".venv").mkdir()
    (repository / ".gitignore").write_text(
        ".venv/\nignored-study.json\n", encoding="utf-8"
    )
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repository / "data-lock.json").write_text("{}\n", encoding="utf-8")
    study_path = repository / "study.json"
    study_path.write_text(
        json.dumps(
            {
                "runtime": {"required_git_tag": "freeze-v1"},
                "data": {"lock": "data-lock.json"},
            }
        ),
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "-b", "main"],
        ["git", "config", "user.name", "test"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", "freeze"],
        ["git", "tag", "freeze-v1"],
    ):
        subprocess.run(
            command,
            cwd=repository,
            capture_output=True,
            check=True,
        )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", str(repository / ".venv"))
    monkeypatch.setattr(artifacts.sys, "prefix", str(repository / ".venv"))
    monkeypatch.setattr(
        artifacts,
        "_package_origins",
        lambda _repository: {"test": "inside-project-venv"},
    )

    identity = capture_scientific_launch_identity(repository, study_path)
    assert identity["required_git_tag"] == "freeze-v1"
    assert identity["tracked_file_count"] == 4
    assert Path(identity["executables"]["git"]["path"]).is_absolute()
    assert len(identity["executables"]["git"]["sha256"]) == 64

    external_study = tmp_path / "external-study.json"
    external_study.write_bytes(study_path.read_bytes())
    with pytest.raises(RuntimeError, match="tracked tagged repository input"):
        capture_scientific_launch_identity(repository, external_study)

    ignored_study = repository / "ignored-study.json"
    ignored_study.write_bytes(study_path.read_bytes())
    with pytest.raises(RuntimeError, match="tracked tagged repository input"):
        capture_scientific_launch_identity(repository, ignored_study)

    external_lock = tmp_path / "external-data-lock.json"
    external_lock.write_text("{}\n", encoding="utf-8")
    study_path.write_text(
        json.dumps(
            {
                "runtime": {"required_git_tag": "freeze-v2"},
                "data": {"lock": "../external-data-lock.json"},
            }
        ),
        encoding="utf-8",
    )
    for command in (
        ["git", "add", "study.json"],
        ["git", "commit", "-m", "external lock contract"],
        ["git", "tag", "freeze-v2"],
    ):
        subprocess.run(
            command,
            cwd=repository,
            capture_output=True,
            check=True,
        )
    with pytest.raises(RuntimeError, match="tracked tagged repository input"):
        capture_scientific_launch_identity(repository, study_path)

    (repository / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean"):
        capture_scientific_launch_identity(repository, study_path)


def test_launch_identity_rejects_same_named_branch_without_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".venv").mkdir()
    (repository / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repository / "data-lock.json").write_text("{}\n", encoding="utf-8")
    study_path = repository / "study.json"
    study_path.write_text(
        json.dumps(
            {
                "runtime": {"required_git_tag": "freeze-v1"},
                "data": {"lock": "data-lock.json"},
            }
        ),
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "-b", "main"],
        ["git", "config", "user.name", "test"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", "freeze"],
        ["git", "branch", "freeze-v1"],
    ):
        subprocess.run(command, cwd=repository, capture_output=True, check=True)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", str(repository / ".venv"))
    monkeypatch.setattr(artifacts.sys, "prefix", str(repository / ".venv"))
    monkeypatch.setattr(
        artifacts,
        "_package_origins",
        lambda _repository: {"test": "inside-project-venv"},
    )

    with pytest.raises(
        (RuntimeError, subprocess.CalledProcessError), match=r"tag|reference"
    ):
        capture_scientific_launch_identity(repository, study_path)


def test_loaded_study_payload_must_match_launch_identity() -> None:
    tagged = {"runtime": {"seed": 0}, "data": {"lock": "data-lock.json"}}
    launch = {"study_payload_sha256": artifacts._canonical_json_sha256(tagged)}
    artifacts.verify_loaded_study_identity(launch, tagged)

    altered = {"runtime": {"seed": 123456}, "data": {"lock": "data-lock.json"}}
    with pytest.raises(RuntimeError, match="study payload"):
        artifacts.verify_loaded_study_identity(launch, altered)


def test_runtime_manifest_records_fit_eos_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(artifacts, "_git_state", lambda _path: ("abc123", False))
    monkeypatch.setattr(
        artifacts,
        "_nvidia_smi_identity",
        lambda: {"path": None, "sha256": None, "driver_version": None},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    study_path = tmp_path / "study.json"
    lock_path = tmp_path / "data-lock.json"
    for path in (study_path, lock_path):
        path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    manifest = build_runtime_manifest(
        repository=tmp_path,
        stage="fit_probe",
        run_kind="engineering_smoke_only",
        model_key="model",
        model_spec={
            "id": "example/model",
            "revision": "a" * 40,
            "generation_eos_token_ids": [1, 50, 106],
            "dtype": "bfloat16",
        },
        model_class="Model",
        model_fingerprint={},
        tokenizer_fingerprint={},
        layer_path="model.layers",
        study_path=study_path,
        data_lock_path=lock_path,
        accessed_splits=("direction_fit", "direction_probe"),
    )

    assert manifest["generation_eos_token_ids"] == [1, 50, 106]


def test_tensor_hash_is_dtype_and_device_canonical() -> None:
    first = tensor_sha256(torch.tensor([1.0, 2.0], dtype=torch.float32))
    second = tensor_sha256(torch.tensor([1.0, 2.0], dtype=torch.float64))

    assert first == second
    assert len(first) == 64
