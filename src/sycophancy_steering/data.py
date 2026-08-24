# SPDX-License-Identifier: AGPL-3.0-or-later

"""Hash-bound, stage-scoped access to materialized study records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

DATA_LOCK_SCHEMA = "selective_sycophancy_data_lock.v1"
CANONICAL_SPLITS = {"direction_fit", "direction_probe", "evaluation"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(records: list[dict[str, Any]]) -> str:
    return ordered_id_values_sha256([str(record["id"]) for record in records])


def ordered_id_values_sha256(ids: list[str]) -> str:
    payload = "\n".join(ids) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


_RECORD_CONTRACT_FIELDS = (
    "id",
    "source_dataset",
    "source_parent_split",
    "correct_option",
    "wrong_option",
    "pressure_variant",
)


def record_contract(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the ordered, non-text fields that bind behavioral semantics."""

    contract: list[dict[str, Any]] = []
    for record in records:
        missing = set(_RECORD_CONTRACT_FIELDS) - set(record)
        if missing:
            raise ValueError(f"Record contract is missing fields: {sorted(missing)}")
        contract.append({field: record[field] for field in _RECORD_CONTRACT_FIELDS})
    return contract


def record_contract_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        record_contract(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_data_lock(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("Scientific data-lock hash differs from launch identity")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Record {line_number} at {path} is not an object")
        records.append(value)
    return records


def _validate_records(records: list[dict[str, Any]], *, split: str) -> None:
    required = {
        "id",
        "question",
        "correct_option",
        "wrong_option",
        "options",
        "pressure_variant",
    }
    ids: list[str] = []
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"{split} record is missing fields: {sorted(missing)}")
        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{split} contains an invalid record id")
        ids.append(record_id)
        if record["correct_option"] not in {"A", "B"}:
            raise ValueError(f"{split}/{record_id} has an invalid correct option")
        if record["wrong_option"] not in {"A", "B"}:
            raise ValueError(f"{split}/{record_id} has an invalid wrong option")
        if record["wrong_option"] == record["correct_option"]:
            raise ValueError(f"{split}/{record_id} has identical A/B labels")
        if not isinstance(record["options"], dict) or set(record["options"]) != {
            "A",
            "B",
        }:
            raise ValueError(f"{split}/{record_id} has invalid options")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{split} contains duplicate record ids")


def _validate_lock_id_inventories(
    split_lock: dict[str, Any],
) -> dict[str, list[str]]:
    inventories: dict[str, list[str]] = {}
    for split in sorted(CANONICAL_SPLITS):
        expected = split_lock[split]
        if not isinstance(expected, dict):
            raise ValueError(f"Data lock split entry is invalid: {split}")
        ids = expected.get("ids")
        if (
            not isinstance(ids, list)
            or any(not isinstance(value, str) or not value for value in ids)
            or len(ids) != expected.get("count")
        ):
            raise ValueError(f"Data lock ID inventory is invalid: {split}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Data lock ID inventory contains duplicate ids: {split}")
        if ordered_id_values_sha256(ids) != expected.get("ordered_ids_sha256"):
            raise ValueError(f"Data lock ID inventory hash mismatch: {split}")
        contract_hash = expected.get("record_contract_sha256")
        if (
            not isinstance(contract_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", contract_hash) is None
        ):
            raise ValueError(f"Data lock record contract hash is invalid: {split}")
        inventories[split] = ids
    split_names = sorted(inventories)
    for left_index, left in enumerate(split_names):
        left_ids = set(inventories[left])
        for right in split_names[left_index + 1 :]:
            overlap = left_ids & set(inventories[right])
            if overlap:
                raise ValueError(f"Data lock split inventories overlap: {left}/{right}")
    return inventories


def validate_materialized_data(
    data_dir: Path,
    lock_path: Path,
    *,
    allowed_splits: tuple[str, ...],
    expected_lock_sha256: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Validate and return only the splits permitted for the current stage."""

    requested = set(allowed_splits)
    if (
        not allowed_splits
        or len(requested) != len(allowed_splits)
        or not requested <= CANONICAL_SPLITS
    ):
        raise ValueError(
            "Requested data access boundary is empty, duplicate, or unknown"
        )
    lock = load_data_lock(lock_path, expected_sha256=expected_lock_sha256)
    if lock.get("schema_version") != DATA_LOCK_SCHEMA:
        raise ValueError("Unsupported data-lock schema")
    split_lock = lock.get("splits")
    if not isinstance(split_lock, dict) or set(split_lock) != CANONICAL_SPLITS:
        raise ValueError("Data lock has a noncanonical split inventory")
    locked_ids = _validate_lock_id_inventories(split_lock)

    canonical_root = data_dir.resolve()
    loaded: dict[str, list[dict[str, Any]]] = {}
    for split in allowed_splits:
        expected = split_lock[split]
        path = (data_dir / f"{split}.jsonl").resolve()
        if path.parent != canonical_root:
            raise ValueError(f"Materialized path escapes data directory: {split}")
        if not path.is_file():
            raise ValueError(f"Materialized split is missing: {split}")
        actual_hash = sha256_file(path)
        if actual_hash != expected.get("sha256"):
            raise ValueError(f"Materialized split hash mismatch: {split}")
        records = _load_records(path)
        if len(records) != expected.get("count"):
            raise ValueError(f"Materialized split count mismatch: {split}")
        _validate_records(records, split=split)
        actual_ids = [str(record["id"]) for record in records]
        if actual_ids != locked_ids[split]:
            raise ValueError(f"Materialized ordered ID inventory mismatch: {split}")
        if ordered_ids_sha256(records) != expected.get("ordered_ids_sha256"):
            raise ValueError(f"Materialized ordered ID hash mismatch: {split}")
        if record_contract_sha256(records) != expected.get("record_contract_sha256"):
            raise ValueError(f"Materialized record contract hash mismatch: {split}")
        loaded[split] = records
    return loaded
