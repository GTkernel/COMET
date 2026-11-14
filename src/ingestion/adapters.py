"""Read structured files in bounded batches before any model preprocessing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

FORMATS = {".csv": "csv", ".tsv": "tsv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".parquet": "parquet"}


def source_format(source: str | Path, manifest: dict | None = None) -> str:
    fmt = (manifest or {}).get("format", "auto")
    if fmt == "auto":
        fmt = FORMATS.get(Path(source).suffix.lower())
    if fmt not in set(FORMATS.values()):
        raise ValueError("Unsupported source format. Use CSV/TSV, JSON array/JSONL, or Parquet")
    return fmt


def _object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"Duplicate JSON key {key!r}")
        record[key] = value
    return record


def _json_records(source: Path):
    # Parsing events exposes duplicate keys that decoded dictionaries would hide.
    import ijson
    with source.open("rb") as stream:
        events = iter(ijson.parse(stream))
        first = next(events, None)
        if first != ("", "start_array", None):
            raise ValueError("JSON source must contain a top-level array of flat objects")
        record = None
        key = None
        for prefix, event, value in events:
            if event == "start_map" and prefix == "item":
                record = {}
            elif event == "map_key" and prefix == "item":
                if value in record:
                    raise ValueError(f"Duplicate JSON key {value!r}")
                key = value
                record[key] = None
            elif event in {"null", "boolean", "number", "string"} and record is not None and prefix == f"item.{key}":
                record[key] = value
            elif event == "end_map" and prefix == "item":
                yield record
                record = None
            elif event == "end_array" and prefix == "":
                continue
            else:
                raise ValueError("JSON records must be flat objects with scalar values")


def rows(source: str | Path, manifest: dict | None = None) -> Iterator[dict]:
    path = Path(source)
    manifest = manifest or {}
    fmt = source_format(path, manifest)
    if fmt in {"csv", "tsv"}:
        with path.open("r", encoding=manifest.get("encoding", "utf-8-sig"), newline="") as stream:
            reader = csv.reader(stream, delimiter="\t" if fmt == "tsv" else ",", strict=True)
            headers = next(reader, None)
            if not headers or any(not h.strip() for h in headers) or len(set(headers)) != len(headers):
                raise ValueError("CSV/TSV headers must be nonblank and unique")
            for number, values in enumerate(reader, 2):
                if len(values) != len(headers):
                    raise ValueError(f"Record {number} has {len(values)} fields. Expected {len(headers)}")
                yield dict(zip(headers, values, strict=True))
    elif fmt == "parquet":
        parquet = pq.ParquetFile(path)
        headers = parquet.schema_arrow.names
        if len(set(headers)) != len(headers):
            raise ValueError("Parquet contains duplicate column names")
        for batch in parquet.iter_batches(batch_size=10000):
            for record in batch.to_pylist():
                yield record
    else:
        if fmt == "json":
            records = _json_records(path)
            for record in records:
                yield record
        else:
            with path.open("r", encoding=manifest.get("encoding", "utf-8-sig")) as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line, object_pairs_hook=_object)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL record on line {line_number}") from exc


def batches(source: str | Path, manifest: dict | None = None, *, batch_size: int = 10000) -> Iterator[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    expected = None
    batch = []
    for number, record in enumerate(rows(source, manifest), 1):
        if not isinstance(record, dict) or not record or any(not isinstance(k, str) or not k.strip() for k in record):
            raise ValueError(f"Record {number} must be an object with nonblank string keys")
        if any(isinstance(v, (dict, list, tuple)) for v in record.values()):
            raise ValueError(f"Record {number} contains nested data. Provide a flat telemetry mapping")
        if expected is None:
            expected = set(record)
        elif set(record) != expected:
            raise ValueError(f"Record {number} has inconsistent columns")
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
