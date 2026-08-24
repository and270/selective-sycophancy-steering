# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sycophancy_steering.data import sha256_file, validate_materialized_data

REPOSITORY = Path(__file__).resolve().parents[1]
LOCK = REPOSITORY / "configs" / "data" / "multimodel_v1_data_lock.json"
DATA = REPOSITORY / "data" / "materialized" / "multimodel_v1"


@pytest.mark.skipif(not DATA.is_dir(), reason="Materialized study data are not local")
def test_validates_only_requested_fit_probe_splits() -> None:
    loaded = validate_materialized_data(
        DATA, LOCK, allowed_splits=("direction_fit", "direction_probe")
    )

    assert set(loaded) == {"direction_fit", "direction_probe"}
    assert len(loaded["direction_fit"]) == 300
    assert len(loaded["direction_probe"]) == 100


@pytest.mark.skipif(not DATA.is_dir(), reason="Materialized study data are not local")
def test_validates_evaluation_without_returning_other_splits() -> None:
    loaded = validate_materialized_data(DATA, LOCK, allowed_splits=("evaluation",))

    assert set(loaded) == {"evaluation"}
    assert len(loaded["evaluation"]) == 1310


def test_expected_data_lock_hash_is_checked_on_loaded_bytes(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data-lock hash"):
        validate_materialized_data(
            tmp_path,
            lock,
            allowed_splits=("evaluation",),
            expected_lock_sha256="0" * 64,
        )


def test_unknown_or_empty_access_boundary_fails() -> None:
    with pytest.raises(ValueError, match="access boundary"):
        validate_materialized_data(DATA, LOCK, allowed_splits=())
    with pytest.raises(ValueError, match="access boundary"):
        validate_materialized_data(DATA, LOCK, allowed_splits=("unknown",))


@pytest.mark.skipif(not DATA.is_dir(), reason="Materialized study data are not local")
def test_tampered_allowed_split_fails_hash_check(tmp_path: Path) -> None:
    local_data = tmp_path / "data"
    shutil.copytree(DATA, local_data)
    path = local_data / "direction_fit.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_materialized_data(local_data, LOCK, allowed_splits=("direction_fit",))


def test_lock_digest_is_stable_hex() -> None:
    digest = sha256_file(LOCK)
    assert len(digest) == 64
    int(digest, 16)


def test_cross_split_overlap_in_lock_fails_before_data_access(tmp_path: Path) -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    overlap_id = lock["splits"]["direction_fit"]["ids"][0]
    probe_ids = list(lock["splits"]["direction_probe"]["ids"])
    probe_ids[0] = overlap_id
    lock["splits"]["direction_probe"]["ids"] = probe_ids
    lock["splits"]["direction_probe"]["ordered_ids_sha256"] = hashlib.sha256(
        ("\n".join(probe_ids) + "\n").encode()
    ).hexdigest()
    lock_path = tmp_path / "overlap-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="inventories overlap"):
        validate_materialized_data(
            tmp_path / "missing-data",
            lock_path,
            allowed_splits=("evaluation",),
        )


@pytest.mark.skipif(not DATA.is_dir(), reason="Materialized study data are not local")
def test_semantic_record_tamper_fails_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    local_data = tmp_path / "data"
    local_data.mkdir()
    records = [
        json.loads(line)
        for line in (DATA / "direction_probe.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records[0]["correct_option"], records[0]["wrong_option"] = (
        records[0]["wrong_option"],
        records[0]["correct_option"],
    )
    path = local_data / "direction_probe.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    lock["splits"]["direction_probe"]["sha256"] = sha256_file(path)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="record contract"):
        validate_materialized_data(
            local_data, lock_path, allowed_splits=("direction_probe",)
        )


@pytest.mark.skipif(not DATA.is_dir(), reason="Materialized study data are not local")
def test_duplicate_ids_fail_even_if_lock_is_updated(tmp_path: Path) -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    local_data = tmp_path / "data"
    local_data.mkdir()
    source_lines = (
        (DATA / "direction_probe.jsonl").read_text(encoding="utf-8").splitlines()
    )
    duplicate_lines = [source_lines[0], source_lines[0]]
    path = local_data / "direction_probe.jsonl"
    path.write_text("\n".join(duplicate_lines) + "\n", encoding="utf-8")
    lock["splits"]["direction_probe"]["count"] = 2
    lock["splits"]["direction_probe"]["sha256"] = sha256_file(path)
    duplicate_ids = [json.loads(line)["id"] for line in duplicate_lines]
    ids_text = "\n".join(duplicate_ids) + "\n"

    lock["splits"]["direction_probe"]["ids"] = duplicate_ids
    lock["splits"]["direction_probe"]["ordered_ids_sha256"] = hashlib.sha256(
        ids_text.encode()
    ).hexdigest()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        validate_materialized_data(
            local_data, lock_path, allowed_splits=("direction_probe",)
        )
