# SPDX-License-Identifier: AGPL-3.0-or-later

"""Atomic artifacts, canonical hashes, and launch/runtime provenance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .data import sha256_file


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _launch_identity_sha256(identity: dict[str, Any]) -> str:
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    return _canonical_json_sha256(body)


def verify_launch_identity_digest(identity: dict[str, Any]) -> None:
    digest = identity.get("identity_sha256")
    if not isinstance(digest, str) or digest != _launch_identity_sha256(identity):
        raise ValueError("Scientific launch identity digest is invalid")


def verify_loaded_study_identity(
    launch_identity: dict[str, Any], study_payload: dict[str, Any]
) -> None:
    expected = launch_identity.get("study_payload_sha256")
    if (
        not isinstance(expected, str)
        or _canonical_json_sha256(study_payload) != expected
    ):
        raise RuntimeError(
            "Loaded scientific study payload differs from launch identity"
        )


def atomic_write_json(path: Path, payload: object) -> None:
    """Serialize strict JSON and atomically replace the destination."""

    serialized = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_sha256(tensor: Tensor) -> str:
    """Hash canonical contiguous CPU float32 bytes."""

    canonical = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


_ARTIFACT_MANIFEST = "artifact_manifest.json"


def _validated_payload_names(payload_files: tuple[str, ...]) -> tuple[str, ...]:
    if (
        not payload_files
        or len(payload_files) != len(set(payload_files))
        or any(
            not name
            or Path(name).name != name
            or name in {"status.json", _ARTIFACT_MANIFEST}
            for name in payload_files
        )
    ):
        raise ValueError("Artifact content manifest file inventory is invalid")
    return tuple(sorted(payload_files))


def finalize_artifact_stage(
    directory: Path,
    *,
    payload_files: tuple[str, ...],
    status: dict[str, Any],
) -> dict[str, Any]:
    """Write a reproducible payload tree and bind it from final status."""

    names = _validated_payload_names(payload_files)
    files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Artifact payload file is missing: {name}")
        files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": "selective_sycophancy_artifact_manifest.v1",
        "files": files,
    }
    manifest_path = directory / _ARTIFACT_MANIFEST
    atomic_write_json(manifest_path, manifest)
    final_status = dict(status)
    final_status.update(
        {
            "artifact_manifest_path": _ARTIFACT_MANIFEST,
            "artifact_manifest_sha256": sha256_file(manifest_path),
        }
    )
    atomic_write_json(directory / "status.json", final_status)
    return final_status


def verify_artifact_manifest(
    directory: Path,
    *,
    status: dict[str, Any],
    expected_files: tuple[str, ...],
) -> dict[str, Any]:
    """Verify the final status-to-manifest-to-payload hash chain."""

    try:
        names = _validated_payload_names(expected_files)
        if status.get("artifact_manifest_path") != _ARTIFACT_MANIFEST:
            raise ValueError
        manifest_path = directory / _ARTIFACT_MANIFEST
        if not manifest_path.is_file() or status.get(
            "artifact_manifest_sha256"
        ) != sha256_file(manifest_path):
            raise ValueError
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_directory_names = set(names) | {"status.json", _ARTIFACT_MANIFEST}
        entries = list(directory.iterdir())
        if {entry.name for entry in entries} != expected_directory_names or any(
            not entry.is_file() or entry.is_symlink() for entry in entries
        ):
            raise ValueError
        files = manifest.get("files")
        if (
            manifest.get("schema_version")
            != "selective_sycophancy_artifact_manifest.v1"
            or not isinstance(files, dict)
            or tuple(sorted(files)) != names
        ):
            raise ValueError
        for name in names:
            path = directory / name
            metadata = files[name]
            if (
                not path.is_file()
                or not isinstance(metadata, dict)
                or metadata.get("sha256") != sha256_file(path)
                or metadata.get("size_bytes") != path.stat().st_size
            ):
                raise ValueError
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Artifact content manifest verification failed") from error
    return manifest


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _resolved_executable(name: str, *, repository: Path | None = None) -> Path:
    resolved_name = shutil.which(name)
    if resolved_name is None:
        raise RuntimeError(f"Required executable is unavailable: {name}")
    executable = Path(resolved_name).resolve()
    if not executable.is_file():
        raise RuntimeError(f"Resolved executable is not a file: {name}")
    if repository is not None:
        try:
            executable.relative_to(repository.resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError(f"Refusing repository-local executable: {name}")
    return executable


def _git_executable(repository: Path) -> str:
    return str(_resolved_executable("git", repository=repository))


def _git_state(repository: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(  # nosec B603
            [_git_executable(repository), "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # nosec B603
                [_git_executable(repository), "status", "--porcelain"],
                cwd=repository,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def require_clean_repository(repository: Path) -> str:
    """Return the commit or fail before a scientific run from a dirty tree."""

    commit, dirty = _git_state(repository)
    if commit == "unavailable" or dirty:
        raise RuntimeError(
            "Scientific execution requires an available clean Git repository"
        )
    return commit


def _git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(  # nosec B603
        [_git_executable(repository), *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(  # nosec B603
        [_git_executable(repository), *arguments],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout


def _tagged_tracked_input(
    repository: Path,
    path: Path,
    *,
    tag_commit: str,
    label: str,
) -> dict[str, str]:
    root = repository.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError(
            f"{label} must be a tracked tagged repository input"
        ) from error
    if not resolved.is_file():
        raise RuntimeError(f"{label} must be a tracked tagged repository input")
    try:
        tracked = _git_text(repository, "ls-files", "--error-unmatch", "--", relative)
        tagged_blob_oid = _git_text(repository, "rev-parse", f"{tag_commit}:{relative}")
        working_blob_oid = _git_text(
            repository,
            "hash-object",
            "--path",
            relative,
            resolved.as_posix(),
        )
        tagged_bytes = _git_bytes(repository, "show", f"{tag_commit}:{relative}")
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{label} must be a tracked tagged repository input"
        ) from error
    current_bytes = resolved.read_bytes()
    if tracked != relative or working_blob_oid != tagged_blob_oid:
        raise RuntimeError(f"{label} differs from its frozen tagged repository blob")
    return {
        "path": relative,
        "git_blob_oid": tagged_blob_oid,
        "tagged_blob_sha256": hashlib.sha256(tagged_bytes).hexdigest(),
        "sha256": hashlib.sha256(current_bytes).hexdigest(),
    }


def _tracked_content_identity(repository: Path) -> tuple[str, int]:
    completed = subprocess.run(  # nosec B603
        [_git_executable(repository), "ls-files", "-z"],
        cwd=repository,
        capture_output=True,
        check=True,
    )
    relatives = [value for value in completed.stdout.split(b"\0") if value]
    digest = hashlib.sha256()
    for relative_bytes in relatives:
        relative = relative_bytes.decode("utf-8")
        path = (repository / relative).resolve()
        if not path.is_file() or repository.resolve() not in path.parents:
            raise RuntimeError(f"Tracked scientific source is unavailable: {relative}")
        content_hash = hashlib.sha256(path.read_bytes()).digest()
        digest.update(relative_bytes)
        digest.update(b"\0")
        digest.update(content_hash)
    return digest.hexdigest(), len(relatives)


def _package_origins(repository: Path) -> dict[str, str]:
    expected_prefix = (repository / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError("Scientific Python prefix is not the project .venv")
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env is None or Path(virtual_env).resolve() != expected_prefix:
        raise RuntimeError("Scientific VIRTUAL_ENV does not identify the project .venv")
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("Scientific execution forbids PYTHONPATH contamination")
    origins: dict[str, str] = {}
    for module_name in (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "numpy",
        "pyarrow",
        "bitsandbytes",
        "yaml",
    ):
        spec = importlib.util.find_spec(module_name)
        origin = None if spec is None else spec.origin
        if origin is None:
            raise RuntimeError(
                f"Required scientific package is unavailable: {module_name}"
            )
        resolved = Path(origin).resolve()
        if expected_prefix not in resolved.parents:
            raise RuntimeError(
                f"Scientific package is outside project .venv: {module_name}={resolved}"
            )
        origins[module_name] = resolved.as_posix()
    return origins


def capture_scientific_launch_identity(
    repository: Path, study_path: Path
) -> dict[str, Any]:
    """Snapshot immutable code/config/dependency identity before inference."""

    study_bytes = study_path.read_bytes()
    study_payload = json.loads(study_bytes.decode("utf-8"))
    required_tag = study_payload.get("runtime", {}).get("required_git_tag")
    if not isinstance(required_tag, str) or not required_tag:
        raise RuntimeError("Frozen study has no required Git tag")
    if (
        required_tag.startswith("-")
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", required_tag) is None
    ):
        raise RuntimeError("Frozen Git tag has invalid syntax")
    commit = require_clean_repository(repository)
    tag_ref = f"refs/tags/{required_tag}"
    try:
        tag_commit = _git_text(
            repository, "rev-parse", "--verify", f"{tag_ref}^{{commit}}"
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Frozen Git tag reference is unavailable") from error
    if tag_commit != commit:
        raise RuntimeError("Scientific HEAD does not match the frozen Git tag")
    study_input = _tagged_tracked_input(
        repository,
        study_path,
        tag_commit=tag_commit,
        label="Scientific study",
    )
    data_lock_path = repository / str(study_payload["data"]["lock"])
    data_lock_input = _tagged_tracked_input(
        repository,
        data_lock_path,
        tag_commit=tag_commit,
        label="Scientific data lock",
    )
    uv_lock_input = _tagged_tracked_input(
        repository,
        repository / "uv.lock",
        tag_commit=tag_commit,
        label="Scientific dependency lock",
    )
    tracked_sha256, tracked_count = _tracked_content_identity(repository)
    git_executable = _resolved_executable("git", repository=repository)
    identity: dict[str, Any] = {
        "executables": {
            "git": {
                "path": git_executable.as_posix(),
                "sha256": sha256_file(git_executable),
            }
        },
        "required_git_tag": required_tag,
        "repository_commit": commit,
        "repository_tree": _git_text(repository, "rev-parse", "HEAD^{tree}"),
        "tracked_content_sha256": tracked_sha256,
        "tracked_file_count": tracked_count,
        "tracked_scientific_inputs": {
            "study": study_input,
            "data_lock": data_lock_input,
            "uv_lock": uv_lock_input,
        },
        "study_sha256": study_input["sha256"],
        "study_payload_sha256": _canonical_json_sha256(study_payload),
        "data_lock_sha256": data_lock_input["sha256"],
        "uv_lock_sha256": uv_lock_input["sha256"],
        "python_prefix": Path(sys.prefix).resolve().as_posix(),
        "package_origins": _package_origins(repository),
    }
    identity["identity_sha256"] = _launch_identity_sha256(identity)
    return identity


def verify_scientific_launch_identity(
    expected: dict[str, Any], repository: Path, study_path: Path
) -> None:
    actual = capture_scientific_launch_identity(repository, study_path)
    if actual != expected:
        raise RuntimeError("Scientific launch identity changed during execution")


def verify_parent_launch_identity(
    current: dict[str, Any], parent: dict[str, Any], *, label: str
) -> None:
    runtime = parent.get("runtime")
    launch = runtime.get("launch_identity") if isinstance(runtime, dict) else None
    if not isinstance(launch, dict):
        raise ValueError(f"{label} artifact used a different scientific code identity")
    verify_launch_identity_digest(current)
    verify_launch_identity_digest(launch)
    if launch != current:
        raise ValueError(f"{label} artifact used a different scientific code identity")


def _nvidia_smi_identity() -> dict[str, str]:
    try:
        executable = _resolved_executable("nvidia-smi")
        driver = (
            subprocess.run(  # nosec B603
                [
                    str(executable),
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        return {
            "path": executable.as_posix(),
            "sha256": sha256_file(executable),
            "driver_version": driver,
        }
    except (OSError, RuntimeError, subprocess.CalledProcessError, IndexError):
        return {
            "path": "unavailable",
            "sha256": "unavailable",
            "driver_version": "unavailable",
        }


def build_runtime_manifest(
    *,
    repository: Path,
    stage: str,
    run_kind: str,
    model_key: str,
    model_spec: dict[str, Any],
    model_class: str,
    model_fingerprint: dict[str, Any],
    tokenizer_fingerprint: dict[str, Any],
    layer_path: str,
    study_path: Path,
    data_lock_path: Path,
    accessed_splits: tuple[str, ...],
    stage_parameters: dict[str, Any] | None = None,
    launch_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture launch identities and live software/hardware state."""

    commit, dirty = _git_state(repository)
    if run_kind == "scientific":
        if launch_identity is None:
            raise RuntimeError("Scientific runtime manifest requires a launch identity")
        verify_scientific_launch_identity(launch_identity, repository, study_path)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    nvidia_smi = _nvidia_smi_identity()
    return {
        "created_at_unix": time.time(),
        "stage": stage,
        "run_kind": run_kind,
        "command": sys.argv,
        "repository_commit": commit,
        "repository_dirty": dirty,
        "launch_identity": launch_identity,
        "study_path": study_path.as_posix(),
        "study_sha256": sha256_file(study_path),
        "data_lock_path": data_lock_path.as_posix(),
        "data_lock_sha256": sha256_file(data_lock_path),
        "uv_lock_sha256": sha256_file(repository / "uv.lock"),
        "accessed_splits": list(accessed_splits),
        "stage_parameters": stage_parameters or {},
        "model_key": model_key,
        "model_id": model_spec["id"],
        "model_revision": model_spec["revision"],
        "generation_eos_token_ids": list(model_spec["generation_eos_token_ids"]),
        "model_class": model_class,
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "layer_path": layer_path,
        "dtype": model_spec["dtype"],
        "quantization": model_spec.get("quantization"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "flash_linear_attention": _package_version("flash-linear-attention"),
        "causal_conv1d": _package_version("causal-conv1d"),
        "triton": _package_version("triton"),
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": nvidia_smi["driver_version"],
        "nvidia_smi": nvidia_smi,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "gpu": gpu,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "peak_gpu_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else None
        ),
    }
