"""Fingerprint legacy SQLite telemetry in bounded batches."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from src.ingestion.manifest import is_label

CORE_COLUMNS = ("row_id", "timestamp", "timestamp_ns", "entity_id")
CONTENT_DOMAIN = b"COMET legacy canonical telemetry content v1\x00"


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def dataset_provenance(dataset: dict, batches: Iterable | None = None) -> dict:
    """Fingerprint canonical telemetry rows read from legacy tables.

    Chunk boundaries do not enter the digest. Labels and incidental SQLite
    columns are absent, and length prefixes separate schema/record encodings.
    Immutable imports retain their existing source and import fingerprints.
    """
    if dataset.get("storage_version") == 2:
        return {"kind": "immutable_import", "dataset_fingerprint": dataset["import_fingerprint"],
                "import_fingerprint": dataset["import_fingerprint"],
                "source_sha256": dataset["source_sha256"]}
    if batches is None:
        raise ValueError("Legacy content provenance requires canonical telemetry batches")
    columns = [{key: column.get(key) for key in ("name", "source_name", "role", "dtype", "unit")}
               for column in dataset["columns"]
               if column.get("role") not in {"label", "annotation", "excluded"}
               and not is_label(column.get("source_name", column["name"]))
               and not is_label(column["name"])]
    schema = {"core_columns": CORE_COLUMNS, "columns": columns}
    schema_bytes = _encoded(schema)
    digest = hashlib.sha256(CONTENT_DOMAIN)
    digest.update(len(schema_bytes).to_bytes(8, "big"))
    digest.update(schema_bytes)
    names = [*CORE_COLUMNS, *(column["name"] for column in columns)]
    row_count = 0
    for batch in batches:
        missing = set(names) - set(batch.schema.names)
        if missing:
            raise ValueError(f"Legacy canonical columns are missing: {sorted(missing)}")
        data = batch.select(names).to_pydict()
        for index in range(batch.num_rows):
            row_count += 1
            if data["row_id"][index] != row_count:
                raise ValueError("Legacy row IDs must be contiguous from 1. Reimport the source")
            record = _encoded([data[name][index] for name in names])
            digest.update(len(record).to_bytes(8, "big"))
            digest.update(record)
    if row_count != int(dataset["row_count"]):
        raise ValueError("Legacy row_count differs from canonical telemetry. Reimport the source")
    content_hash = digest.hexdigest()
    schema_hash = hashlib.sha256(schema_bytes).hexdigest()
    manifest_hash = hashlib.sha256(_encoded(dataset.get("manifest") or {})).hexdigest()
    identity = {"kind": "legacy_content", "content_sha256": content_hash,
                "schema_sha256": schema_hash, "manifest_sha256": manifest_hash}
    return {**identity, "dataset_fingerprint": hashlib.sha256(_encoded(identity)).hexdigest(),
            "row_count": row_count, "import_fingerprint": None, "source_sha256": None,
            "content_policy": "canonical safe telemetry; original source bytes unavailable"}
