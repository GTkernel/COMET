"""Import telemetry in batches and publish completed datasets transactionally."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterator

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.db import get_connection, get_dataset, init_db, utc_now
from .adapters import batches, source_format
from .manifest import column_names, is_label, is_numeric_literal, load_manifest, resolve_manifest

ADAPTER_VERSION = "2"
CORE = ["row_id", "timestamp", "timestamp_ns", "entity_id"]
PARQUET_PART_ROWS = 65_536
PARQUET_PART_BYTES = 32 * 1024 * 1024


def _root(storage_dir=None) -> Path:
    if storage_dir is not None:
        return Path(storage_dir).resolve()
    from src.common import paths
    return Path(getattr(paths, "STORAGE_ROOT", paths.V2_ROOT)).resolve()


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _discard_generated(path: Path) -> None:
    """Remove only a caller-owned staging output, never a linked directory."""
    parent = path.parent.resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(parent):
        raise ValueError("Generated output escapes its staging directory")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _parquet_parts(path: Path | str) -> list[Path]:
    """Accept historical single files and new ordered immutable part directories."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        parts = sorted(path.glob("part-*.parquet"))
        if parts:
            return parts
    raise ValueError(f"Parquet telemetry has no completed parts: {path}")


def _parquet_batches(path, *, batch_size, columns=None):
    # Dataset scanners may retain every parsed footer. Open one bounded part at
    # a time and close it even when a consumer stops before the final batch.
    for part in _parquet_parts(path):
        with contextlib.closing(pq.ParquetFile(part)) as reader:
            yield from reader.iter_batches(batch_size=batch_size, columns=columns, use_threads=False)


def _parquet_sql(path: Path) -> str:
    return _literal(str(path / "part-*.parquet" if path.is_dir() else path))


class _ParquetPartWriter:
    """Bound buffers and footer metadata by writing one row group per part.

    Parts stay in staging until the directory is atomically replaced. Reject
    records larger than the configured byte limit. That limit counts Arrow's
    uncompressed values. Compression, encoding buffers and the incoming batch
    need additional memory.
    """

    def __init__(self, path: Path, schema: pa.Schema, *, max_rows=None, max_bytes=None):
        self.path = Path(path)
        self.schema = schema
        self.max_rows = PARQUET_PART_ROWS if max_rows is None else max_rows
        self.max_bytes = PARQUET_PART_BYTES if max_bytes is None else max_bytes
        if self.max_rows < 1 or self.max_bytes < 1:
            raise ValueError("Parquet part limits must be positive")
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.batches = []
        self.buffer_rows = self.buffer_bytes = self.rows_written = self.part_count = 0

    def __enter__(self):
        _discard_generated(self.temporary)
        self.temporary.mkdir(parents=True)
        return self

    def _flush(self):
        table = pa.Table.from_batches(self.batches, schema=self.schema)
        part = self.temporary / f"part-{self.part_count:08d}.parquet"
        pending = part.with_suffix(".tmp")
        pq.write_table(table, pending, compression="zstd", row_group_size=max(1, table.num_rows))
        pending.replace(part)
        self.part_count += 1
        self.rows_written += table.num_rows
        self.batches.clear()
        self.buffer_rows = self.buffer_bytes = 0

    def write_batch(self, batch: pa.RecordBatch):
        if batch.schema != self.schema:
            raise ValueError("Parquet output batches have inconsistent schemas")
        offset = 0
        while offset < batch.num_rows:
            available = self.max_bytes - self.buffer_bytes
            count = min(batch.num_rows - offset, self.max_rows - self.buffer_rows)
            candidate = batch.slice(offset, count)
            if candidate.nbytes > available:
                low, high = 0, count
                while low < high:
                    middle = (low + high + 1) // 2
                    if batch.slice(offset, middle).nbytes <= available:
                        low = middle
                    else:
                        high = middle - 1
                count = low
                candidate = batch.slice(offset, count)
            if not count:
                if self.buffer_rows:
                    self._flush()
                    continue
                raise ValueError("One telemetry record exceeds the Parquet part byte limit")
            self.batches.append(candidate)
            self.buffer_rows += count
            self.buffer_bytes += candidate.nbytes
            offset += count
            if self.buffer_rows == self.max_rows or self.buffer_bytes == self.max_bytes:
                self._flush()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.batches.clear()
            _discard_generated(self.temporary)
            return False
        try:
            if self.buffer_rows or not self.part_count:
                self._flush()
            _discard_generated(self.path)
            self.temporary.replace(self.path)
        except BaseException:
            self.batches.clear()
            _discard_generated(self.temporary)
            raise
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _import_lock(path: Path):
    """OS lock releases on process death, allowing interrupted imports to resume."""
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This source is already being imported") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("This source is already being imported") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def inspect_source(source, manifest=None) -> dict:
    """Inspect a bounded sample to suggest a mapping for full ingestion."""
    source = Path(source).resolve()
    supplied = load_manifest(manifest)
    from .esa import recognize_esa_source, recognize_processed_esa_source
    if source.is_dir() or source.suffix.lower() == ".zip":
        if recognize_esa_source(source):
            esa = supplied.get("esa", {})
            issues = [] if esa.get("trusted_pickle") else ["Set esa.trusted_pickle=true only for a trusted ESA archive"]
            if esa.get("version") is None:
                issues.append("Set esa.version to 1 or 2")
            return {"source": str(source), "format": "esa-raw", "manifest": supplied,
                    "suggested_manifest": supplied, "requires_manifest": bool(issues), "issues": issues,
                    "sample_rows": 0, "columns": []}
        if not recognize_processed_esa_source(source):
            raise ValueError("Directory is not a recognized ESA mission layout")
    if recognize_processed_esa_source(source):
        suggested = {**supplied, "esa": {**supplied.get("esa", {}),
                                         "profile": supplied.get("esa", {}).get("profile", "esa-reference")}}
        issues = []
        if source.is_dir() or source.name.endswith(".test.csv"):
            train_directory = source if source.is_dir() else source.parent
            if len(list(train_directory.glob("*_months.train.csv"))) > 1 and not suggested["esa"].get("processed_train"):
                issues.append("Choose esa.processed_train from the available mission training files")
        if source.is_file() and not list(source.parent.glob("*_months.test.csv")) and not suggested["esa"].get("allow_single"):
            issues.append("A lone processed CSV requires esa.allow_single and explicit split boundaries")
        if suggested["esa"].get("version") is None:
            issues.append("Set esa.version to 1 or 2")
        return {"source": str(source), "format": "esa-processed", "manifest": suggested,
                "suggested_manifest": suggested, "requires_manifest": bool(issues), "issues": issues,
                "sample_rows": 0, "columns": [],
                "validation": "ESA pairing, channel schema, and annotations are validated during ingest"}
    sample = next(batches(source, supplied, batch_size=256), [])
    if not sample:
        raise ValueError("Source contains no telemetry records")
    resolved, columns, issues = resolve_manifest(list(sample[0]), sample, supplied)
    return {"source": str(source), "format": source_format(source, supplied),
            "manifest": resolved, "suggested_manifest": resolved, "columns": columns,
            "requires_manifest": bool(issues), "issues": issues, "sample_rows": len(sample),
            "validation": "Sample inspected. Ingest validates every record"}


def _timestamp(value, manifest: dict) -> tuple[str, int]:
    if value is None or value == "" or isinstance(value, bool):
        raise ValueError("Timestamp cannot be missing")
    try:
        unit = manifest.get("epoch_unit")
        if unit:
            # Keep integer nanoseconds because float conversion loses precision.
            numeric = int(value) if str(value).lstrip("-+").isdigit() else float(value)
            stamp = pd.to_datetime(numeric, unit=unit, utc=True)
        else:
            if is_numeric_literal(value):
                raise ValueError("Numeric timestamps require an explicit epoch_unit")
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(manifest.get("timezone", "UTC"), ambiguous="raise", nonexistent="raise")
            stamp = stamp.tz_convert("UTC")
        if pd.isna(stamp):
            raise ValueError("Timestamp cannot be NaT")
        return stamp.isoformat(), int(stamp.value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid timestamp {value!r}: {exc}") from exc


def _numeric(value, name):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Column {name!r} contains a nonnumeric value {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Column {name!r} contains a nonfinite value")
    return number


def _normalized(records, manifest, columns, start):
    normalized = []
    for index, record in enumerate(records, start + 1):
        for name, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Record {index}, column {name!r}: nonfinite value")
        timestamp, nanos = ((None, index) if manifest.get("ordering") == "row" else
                            _timestamp(record[manifest["timestamp"]], manifest))
        entities = [record[name] for name in manifest["entity_columns"]]
        if any(value is None or value == "" for value in entities):
            raise ValueError(f"Record {index} has a missing entity identity")
        row = {"_seq": index, "timestamp": timestamp, "timestamp_ns": nanos,
               "entity_id": json.dumps([str(x) for x in entities], separators=(",", ":")) if entities else ""}
        for column in columns:
            value = record[column["source_name"]]
            if column["source_name"] == manifest.get("gap_column"):
                gap_value = _numeric(value, column["source_name"])
                if gap_value not in {0.0, 1.0}:
                    raise ValueError("gap_column values must be 0 or 1")
                value = str(int(gap_value))
            elif column["role"] == "numeric":
                value = _numeric(value, column["source_name"])
            elif column["role"] == "categorical" and value not in (None, "") and column["source_name"] in manifest.get("_inferred_categorical", ()):
                from .manifest import is_numeric_literal
                if is_numeric_literal(value):
                    raise ValueError(f"Column {column['source_name']!r} mixes numeric and categorical values. Specify categorical_features explicitly")
                value = str(value)
            elif value is not None:
                value = str(value)
            row[column["name"]] = value
        if manifest["layout"] == "long":
            channel = record[manifest["channel_column"]]
            if channel is None or channel == "" or is_label(str(channel)):
                raise ValueError(f"Record {index} has an invalid or annotation channel identity")
            row["_channel"] = str(channel)
            row["__comet_value"] = _numeric(record[manifest["value_column"]], manifest["value_column"])
        normalized.append(row)
    return normalized


def _schema(columns, long=False):
    fields = [pa.field("_seq", pa.int64()), pa.field("timestamp", pa.string()),
              pa.field("timestamp_ns", pa.int64()), pa.field("entity_id", pa.string())]
    fields += [pa.field(c["name"], pa.float64() if c["role"] == "numeric" else pa.string()) for c in columns]
    if long:
        fields += [pa.field("_channel", pa.string()), pa.field("__comet_value", pa.float64())]
    return pa.schema(fields)


def _can_stage_prepared_esa(source, manifest, columns):
    """Only adapter-produced, fully mapped ESA tables use the Arrow fast path."""
    esa = manifest.get("esa", {})
    return (
        source_format(source, manifest) == "parquet"
        and manifest.get("_source_sorted_unique") is True
        and isinstance(esa, dict) and bool(esa.get("adapter_version"))
        and manifest["layout"] == "wide"
        and not manifest.get("entity_columns")
        and not manifest.get("epoch_unit")
        and manifest.get("timezone", "UTC") == "UTC"
        and not manifest.get("label_columns")
        and not manifest.get("_inferred_categorical")
        and all(column["role"] in {"numeric", "categorical", "context"}
                and column["source_name"] in manifest.get(
                    {"numeric": "numeric_features", "categorical": "categorical_features",
                     "context": "context_columns"}[column["role"]], [])
                for column in columns)
    )


def _validate_prepared_schema(schema, manifest, columns):
    expected = {manifest["timestamp"], *(column["source_name"] for column in columns)}
    if len(schema.names) != len(set(schema.names)) or set(schema.names) != expected:
        raise ValueError("Prepared ESA telemetry has inconsistent or duplicate columns")
    timestamp_type = schema.field(manifest["timestamp"]).type
    if not (pa.types.is_string(timestamp_type) or pa.types.is_large_string(timestamp_type)
            or pa.types.is_timestamp(timestamp_type) or pa.types.is_date(timestamp_type)):
        raise ValueError("Invalid timestamp schema in prepared ESA telemetry")
    for column in columns:
        dtype = schema.field(column["source_name"]).type
        if column["role"] == "numeric" or column["source_name"] == manifest.get("gap_column"):
            valid = (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)
                     or pa.types.is_boolean(dtype) or pa.types.is_null(dtype))
            if column["source_name"] == manifest.get("gap_column"):
                valid = valid or pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
        else:
            valid = pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_null(dtype)
        if not valid:
            raise ValueError(f"Prepared ESA column {column['source_name']!r} has incompatible schema for its declared role")


def _prepared_esa_batches(source, manifest, columns, batch_size):
    with contextlib.closing(pq.ParquetFile(source)) as reader:
        _validate_prepared_schema(reader.schema_arrow, manifest, columns)
        for batch in reader.iter_batches(batch_size=batch_size):
            _validate_prepared_schema(batch.schema, manifest, columns)
            yield batch


def _normalized_prepared_esa(batch, manifest, columns, start, schema):
    """Validate columns with Arrow kernels and format timestamps per row."""
    timestamp = batch.column(batch.schema.get_field_index(manifest["timestamp"]))
    if timestamp.null_count:
        raise ValueError("Invalid timestamp: prepared ESA timestamps cannot be missing")
    try:
        timestamp_values = timestamp.to_pylist()
        if any(is_numeric_literal(value) for value in timestamp_values):
            raise ValueError("Numeric timestamps require an explicit epoch_unit")
        parsed = pd.DatetimeIndex(pd.to_datetime(timestamp_values, format="mixed", utc=True,
                                               errors="raise")).as_unit("ns")
        if parsed.isna().any():
            raise ValueError("Timestamp cannot be NaT")
        canonical = pa.array([stamp.isoformat() for stamp in parsed], type=pa.string())
    except (ValueError, TypeError, OverflowError, pa.ArrowInvalid) as exc:
        raise ValueError(f"Invalid timestamp in prepared ESA telemetry: {exc}") from exc
    arrays = [pa.array(np.arange(start + 1, start + batch.num_rows + 1, dtype=np.int64)), canonical,
              pa.array(parsed.asi8), pa.array(np.full(batch.num_rows, "", dtype=object), type=pa.string())]
    for column in columns:
        value = batch.column(batch.schema.get_field_index(column["source_name"]))
        if column["role"] == "numeric" or column["source_name"] == manifest.get("gap_column"):
            try:
                number = pc.cast(value, pa.float64())
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise ValueError(f"Column {column['source_name']!r} contains a nonnumeric value") from exc
            if pc.any(pc.invert(pc.is_finite(number))).as_py():
                raise ValueError(f"Column {column['source_name']!r} contains a nonfinite value")
            if column["source_name"] == manifest.get("gap_column"):
                if number.null_count or not pc.all(pc.is_in(number, value_set=pa.array([0.0, 1.0]))).as_py():
                    raise ValueError("gap_column values must be 0 or 1")
                value = pc.cast(pc.cast(number, pa.int8()), pa.string())
            else:
                value = number
        else:
            value = pc.cast(value, pa.string())
        arrays.append(value)
    return pa.Table.from_arrays(arrays, schema=schema)


def _stage_source(source, manifest, columns, stage, batch_size):
    progress_path = stage / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    committed = progress.get("rows_staged", 0)
    part_count = progress.get("parts", 0)
    raw = stage / "raw"
    raw.mkdir(exist_ok=True)
    schema = _schema(columns, manifest["layout"] == "long")
    seen = 0
    prepared = _can_stage_prepared_esa(source, manifest, columns)
    source_batches = (_prepared_esa_batches(source, manifest, columns, batch_size) if prepared
                      else batches(source, manifest, batch_size=batch_size))
    with contextlib.closing(source_batches):
        for records in source_batches:
            end = seen + (records.num_rows if prepared else len(records))
            if end <= committed:
                seen = end
                continue
            skip = max(0, committed - seen)
            offset = max(seen, committed)
            if prepared:
                table = _normalized_prepared_esa(records.slice(skip), manifest, columns, offset, schema)
            else:
                table = pa.Table.from_pylist(_normalized(records[skip:], manifest, columns, offset), schema=schema)
            path = raw / f"part-{part_count:08d}.parquet"
            temporary = path.with_suffix(".tmp")
            pq.write_table(table, temporary)
            temporary.replace(path)
            seen = end
            part_count += 1
            _atomic_json(progress_path, {"status": "staging", "rows_staged": seen, "parts": part_count})
    if seen == 0:
        raise ValueError("Source contains no rows")
    if seen < committed:
        raise ValueError("Source changed during import. Staged rows exceed source length")
    # An interrupted write may leave an uncheckpointed part. Exclude it from the sort.
    return [str(raw / f"part-{index:08d}.parquet") for index in range(part_count)], seen


def _write_query(conn, query, path: Path, batch_size):
    reader = conn.sql(query).to_arrow_reader(min(batch_size, 2048))
    with contextlib.closing(reader), _ParquetPartWriter(path, reader.schema) as writer:
        for batch in reader:
            writer.write_batch(batch)
    return writer.rows_written


def _sorted_dedup(conn, keys, payload, policy, path: Path, batch_size):
    """Sort on disk and deduplicate in bounded Arrow batches."""
    query = "SELECT * FROM raw ORDER BY " + ",".join(map(_quote, [*keys, "_seq"]))
    with contextlib.closing(conn.sql(query).to_arrow_reader(min(batch_size, 1024))) as reader:
        _dedup_batches(reader, reader.schema, keys, payload, policy, path)


def _presorted_dedup(parts, keys, payload, policy, path: Path, batch_size):
    """Validate and deduplicate an adapter-produced, already sorted source.

    Adapters that already produce chronological Parquet can skip the external
    sort. Check the order here before publishing, including when reusing cached
    adapter output.
    """
    if not parts:
        raise ValueError("Prepared telemetry contains no parts")
    with contextlib.closing(pq.ParquetFile(parts[0])) as first_part:
        schema = first_part.schema_arrow

    def prepared_batches():
        # Thousands of staged parts must not hold thousands of file handles or
        # decoded footers. Closing this generator also closes the current reader
        # if duplicate validation raises while consuming its batch.
        for part in parts:
            with contextlib.closing(pq.ParquetFile(part)) as parquet_file:
                if parquet_file.schema_arrow != schema:
                    raise ValueError("Prepared telemetry parts have inconsistent schemas")
                yield from parquet_file.iter_batches(batch_size=min(batch_size, 2048))

    with contextlib.closing(prepared_batches()) as source_batches:
        _dedup_batches(source_batches, schema, keys, payload, policy, path)


def _dedup_batches(source_batches, schema, keys, payload, policy, path):
    """Validate ordered rows and carry the final duplicate group into the next batch."""
    pending = None
    pending_payload = None
    pending_key = None
    previous_input_key = None
    with _ParquetPartWriter(path, schema) as writer:
        for batch in source_batches:
            if batch.num_rows == 0:
                continue
            key_columns = [batch.column(batch.schema.get_field_index(key)).to_pylist() for key in keys]
            payload_batch = batch.select(payload)
            group_starts = [0]
            for index in range(batch.num_rows):
                key = tuple(column[index] for column in key_columns)
                if previous_input_key is not None and key < previous_input_key:
                    raise ValueError(
                        "Prepared telemetry declared chronological order but is unsorted: "
                        f"{dict(zip(keys, key))} follows {dict(zip(keys, previous_input_key))}"
                    )
                if index and key != tuple(column[index - 1] for column in key_columns):
                    group_starts.append(index)
                previous_input_key = key
            group_starts.append(batch.num_rows)
            emitted = []
            selected = []
            for group_index in range(len(group_starts) - 1):
                begin, end = group_starts[group_index:group_index + 2]
                key = tuple(column[begin] for column in key_columns)
                first_payload = None
                if policy == "error" and (end - begin > 1 or (pending is not None and key == pending_key)):
                    first_payload = payload_batch.slice(begin, 1)
                if policy == "error" and end - begin > 1:
                    for duplicate in range(begin + 1, end):
                        if not first_payload.equals(payload_batch.slice(duplicate, 1)):
                            raise ValueError(
                                f"Conflicting duplicate observations for {dict(zip(keys, key))}; "
                                "set duplicate_policy explicitly"
                            )
                candidate = begin if policy != "last" else end - 1
                if pending is not None and key == pending_key:
                    if policy == "error" and not pending_payload.equals(first_payload):
                        raise ValueError(
                            f"Conflicting duplicate observations for {dict(zip(keys, key))}; "
                            "set duplicate_policy explicitly"
                        )
                    if policy == "last":
                        pending = batch.slice(candidate, 1)
                        pending_payload = payload_batch.slice(candidate, 1)
                else:
                    if pending is not None:
                        emitted.append(pending)
                        pending = pending_payload = pending_key = None
                    if group_index == len(group_starts) - 2:
                        pending = batch.slice(candidate, 1)
                        pending_payload = payload_batch.slice(candidate, 1)
                        pending_key = key
                    else:
                        selected.append(candidate)
                if pending is not None and group_index < len(group_starts) - 2:
                    emitted.append(pending)
                    pending = pending_payload = pending_key = None
            if selected:
                emitted.append(batch.take(pa.array(selected, type=pa.int32())))
            if emitted:
                writer.write_batch(pa.concat_batches(emitted))
        if pending is not None:
            writer.write_batch(pending)


def _write_wide_telemetry(dedup_path: Path, columns, path: Path, batch_size):
    """Assign stable row IDs while scanning the already sorted Parquet stream."""
    names = ["timestamp", "timestamp_ns", "entity_id"] + [c["name"] for c in columns]
    next_id = 1
    with contextlib.closing(pq.ParquetFile(_parquet_parts(dedup_path)[0])) as first:
        schema = pa.schema([pa.field("row_id", pa.int64())] + [first.schema_arrow.field(name) for name in names])
    with _ParquetPartWriter(path, schema) as writer, contextlib.closing(
        _parquet_batches(dedup_path, batch_size=min(batch_size, 2048), columns=names)
    ) as source_batches:
        for batch in source_batches:
            arrays = [pa.array(range(next_id, next_id + batch.num_rows), type=pa.int64())]
            arrays.extend(batch.column(batch.schema.get_field_index(name)) for name in names)
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
            next_id += batch.num_rows
    return next_id - 1


def _canonicalize(parts, manifest, columns, stage, final_dir, batch_size, scratch):
    artifact = stage / "artifact"
    artifact.mkdir(exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(stage / "sort.duckdb")) as conn:
        conn.execute(f"SET memory_limit={_literal(os.environ.get('COMET_DUCKDB_MEMORY_LIMIT', '512MB'))}")
        conn.execute("SET threads=2")
        conn.execute("SET preserve_insertion_order=false")
        conn.execute(f"SET temp_directory={_literal(str(scratch))}")
        keys = ["entity_id", "timestamp_ns"] + (["_channel"] if manifest["layout"] == "long" else [])
        payload = [name for name in _schema(columns, manifest["layout"] == "long").names if name != "_seq"]
        dedup_path = stage / "dedup.parquet"
        if manifest.get("_source_sorted_unique"):
            if manifest["layout"] != "wide":
                raise ValueError("The adapter sorted-source fast path supports wide telemetry only")
            _presorted_dedup(parts, keys, payload, manifest["duplicate_policy"], dedup_path, batch_size)
        else:
            conn.execute("CREATE OR REPLACE VIEW raw AS SELECT * FROM read_parquet([" + ",".join(map(_literal, parts)) + "])")
            _sorted_dedup(conn, keys, payload, manifest["duplicate_policy"], dedup_path, batch_size)
        features = [column for column in columns if column["role"] != "label"]
        labels = [c for c in columns if c["role"] == "label"]
        if manifest["layout"] == "long" or labels:
            conn.execute("CREATE OR REPLACE VIEW dedup AS SELECT * FROM read_parquet(" + _parquet_sql(dedup_path) + ")")
        if manifest["layout"] == "long":
            if any(c["role"] != "context" for c in features):
                raise ValueError("Long-layout extra columns must be explicitly mapped as context_columns")
            channels = [row[0] for row in conn.execute("SELECT DISTINCT _channel FROM dedup ORDER BY _channel LIMIT 4097").fetchall()]
            if len(channels) > 4096:
                raise ValueError("Long layout exceeds 4096 channels. Select a smaller channel schema")
            if set(manifest.get("units", {})).difference(channels):
                raise ValueError("Long-layout units must identify numeric channel names present in the source")
            context_sql = ",".join(_quote(c["name"]) for c in features)
            if features:
                conflict = conn.execute(f"SELECT entity_id,timestamp_ns FROM (SELECT DISTINCT entity_id,timestamp_ns,{context_sql} FROM dedup) GROUP BY entity_id,timestamp_ns HAVING count(*)>1 LIMIT 1").fetchone()
                if conflict:
                    raise ValueError("Long-layout context conflicts between channels at the same timestamp")
            mapping = column_names(channels + [c["source_name"] for c in features])
            long_features = [{"name": mapping[channel], "source_name": channel, "role": "numeric", "dtype": "float64",
                              **({"unit": manifest["units"][channel]} if channel in manifest.get("units", {}) else {})} for channel in channels]
            if set(c["name"] for c in features).intersection(c["name"] for c in long_features):
                raise ValueError("Channel and context identities collide. Rename context columns")
            pivot = [f"max(__comet_value) FILTER (WHERE _channel={_literal(channel)}) AS {_quote(mapping[channel])}" for channel in channels]
            pivot += [f"first({_quote(c['name'])}) AS {_quote(c['name'])}" for c in features]
            conn.execute("CREATE OR REPLACE VIEW telemetry AS SELECT entity_id,timestamp_ns,first(timestamp) AS timestamp," + ",".join(pivot) + " FROM dedup GROUP BY entity_id,timestamp_ns")
            features = long_features + features
        telemetry_path = artifact / "telemetry.parquet"
        if manifest["layout"] == "long":
            feature_sql = "".join("," + _quote(c["name"]) for c in features)
            query = "SELECT row_number() OVER (ORDER BY entity_id,timestamp_ns) AS row_id,timestamp,timestamp_ns,entity_id" + feature_sql + " FROM telemetry ORDER BY entity_id,timestamp_ns"
            row_count = _write_query(conn, query, telemetry_path, batch_size)
        else:
            row_count = _write_wide_telemetry(dedup_path, features, telemetry_path, batch_size)
        point_path = None
        if labels:
            conn.execute("CREATE OR REPLACE VIEW published AS SELECT * FROM read_parquet(" + _parquet_sql(telemetry_path) + ")")
            channel_expr = "d._channel" if manifest["layout"] == "long" else "NULL::VARCHAR"
            queries = [f"SELECT p.row_id,p.timestamp,p.timestamp_ns,p.entity_id,{_literal(c['source_name'])} AS label_name,d.{_quote(c['name'])} AS label_value,{channel_expr} AS channel FROM published p JOIN dedup d USING(entity_id,timestamp_ns) WHERE d.{_quote(c['name'])} IS NOT NULL" for c in labels]
            point_path = artifact / "point-annotations.parquet"
            _write_query(conn, "SELECT DISTINCT * FROM (" + " UNION ALL ".join(queries) + ") ORDER BY row_id,label_name,channel", point_path, batch_size)
        external = manifest.get("annotations_path")
        annotation_path = None
        if external:
            source_annotations = Path(external).resolve()
            if not source_annotations.is_file():
                raise ValueError("annotations_path does not exist")
            destination = artifact / ("annotations" + source_annotations.suffix)
            shutil.copyfile(source_annotations, destination)
            annotation_path = str(final_dir / destination.name)
        elif point_path:
            annotation_path = str(final_dir / point_path.name)
        return {"storage_version": 2, "row_count": row_count,
                "telemetry_format": "parquet-parts", "parquet_part_rows": PARQUET_PART_ROWS,
                "parquet_part_bytes": PARQUET_PART_BYTES,
                "telemetry_path": str(final_dir / telemetry_path.name), "artifact_dir": str(final_dir),
                "annotations_path": annotation_path, "point_annotations_path": str(final_dir / point_path.name) if point_path else None,
                "annotation_format": manifest.get("annotation_format", "points" if point_path else None),
                "manifest": manifest, "columns": features}


def _existing(fingerprint, db_path):
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT dataset_id FROM datasets WHERE status='ingested' AND json_extract(metadata_json,'$.import_fingerprint')=? ORDER BY dataset_id LIMIT 1", (fingerprint,)).fetchone()
    if row:
        result = load_dataset(row[0], db_path=db_path)
        if Path(result["telemetry_path"]).exists():
            result["duplicate"] = True
            return result
    return None


def _publish_catalog(source, metadata, db_path):
    now = utc_now()
    stat = source.stat()
    with get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT dataset_id FROM datasets WHERE status='ingested' AND json_extract(metadata_json,'$.import_fingerprint')=? LIMIT 1", (metadata["import_fingerprint"],)).fetchone()
        if row:
            return int(row[0])
        cursor = conn.execute("INSERT INTO datasets(name,status,row_count,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?)", (source.stem, "ingested", metadata["row_count"], json.dumps(metadata), now, now))
        dataset_id = int(cursor.lastrowid)
        conn.execute("INSERT INTO dataset_files(dataset_id,file_name,file_path,file_size,modified_time,content_hash,row_count,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (dataset_id, source.name, str(source), stat.st_size, str(stat.st_mtime_ns), metadata["source_sha256"], metadata["row_count"], "ingested", now))
        columns = list(metadata["columns"])
        if metadata["manifest"].get("ordering") != "row":
            columns.insert(0, {"name": "timestamp", "source_name": metadata["manifest"]["timestamp"], "role": "timestamp", "dtype": "timestamp"})
        conn.executemany("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,inferred_type,role,reasoning,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", [(dataset_id, c["source_name"], c["name"], c["dtype"], c["role"], "Validated import contract; no training eligibility decisions applied", now, now) for c in columns])
        conn.commit()
    return dataset_id


def ingest_source(source, manifest=None, *, db_path=None, storage_dir=None, batch_size=10000) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    source = Path(source).resolve()
    input_source = source
    supplied = load_manifest(manifest)
    root = _root(storage_dir)
    from .esa import (prepare_esa_source, prepare_processed_esa_source,
                      recognize_esa_source, recognize_processed_esa_source)
    if (source.is_dir() or source.suffix.lower() == ".zip") and recognize_esa_source(source):
        source, supplied = prepare_esa_source(source, supplied, storage_dir=root, batch_size=batch_size)
        source = Path(source).resolve()
    elif recognize_processed_esa_source(source):
        source, supplied = prepare_processed_esa_source(source, supplied, storage_dir=root, batch_size=batch_size)
        source = Path(source).resolve()
    elif source.is_dir() or source.suffix.lower() == ".zip":
        raise ValueError("Directory is not a recognized ESA mission")
    inspection = inspect_source(source, supplied)
    if inspection["issues"]:
        raise ValueError("Manifest required: " + "; ".join(inspection["issues"]))
    resolved, columns = inspection["manifest"], inspection["columns"]
    source_hash = _sha256(source)
    fingerprint_data = {"source_sha256": source_hash, "manifest": resolved, "adapter_version": ADAPTER_VERSION}
    if resolved.get("annotations_path"):
        fingerprint_data["annotations_sha256"] = _sha256(Path(resolved["annotations_path"]))
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    init_db(db_path)
    existing = _existing(fingerprint, db_path)
    if existing:
        return existing
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    final_dir = root / "datasets" / fingerprint
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = staging / fingerprint
    stage.mkdir(exist_ok=True)
    with _import_lock(staging / (fingerprint + ".lock")):
        existing = _existing(fingerprint, db_path)
        if existing:
            return existing
        if final_dir.exists():
            metadata = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
        else:
            try:
                parts, source_rows = _stage_source(source, resolved, columns, stage, batch_size)
                if _sha256(source) != source_hash:
                    raise ValueError("Source changed during import. Retry with a stable source")
                scratch_root = Path(os.environ.get("COMET_SCRATCH_DIR", str(root / "scratch")))
                metadata = _canonicalize(parts, resolved, columns, stage, final_dir, batch_size, scratch_root / fingerprint)
                metadata.update({"import_fingerprint": fingerprint, "source_sha256": source_hash,
                                 "source_path": str(source), "source_rows": source_rows,
                                 "adapter_version": ADAPTER_VERSION, "created_at": utc_now()})
                if input_source != source:
                    metadata["input_source_path"] = str(input_source)
                if "annotations_sha256" in fingerprint_data:
                    metadata["annotations_sha256"] = fingerprint_data["annotations_sha256"]
                _atomic_json(stage / "artifact" / "metadata.json", metadata)
                (stage / "artifact").replace(final_dir)
            except Exception as exc:
                progress_path = stage / "progress.json"
                progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
                progress.update({"status": "failed", "error": str(exc)})
                _atomic_json(progress_path, progress)
                raise
        dataset_id = _publish_catalog(source, metadata, db_path)
        _atomic_json(stage / "progress.json", {"status": "complete", "dataset_id": dataset_id, "rows_staged": metadata["source_rows"]})
    result = load_dataset(dataset_id, db_path=db_path)
    result["duplicate"] = False
    return result


def set_feature_roles(dataset_id: int, columns: list[dict], *, db_path=None,
                      connection=None, validate_only=False) -> dict | None:
    """Persist user-selected roles without changing telemetry or historical models."""
    dataset = load_dataset(dataset_id, db_path=db_path)
    aliases = {"numeric_telemetry_feature": "numeric", "categorical_feature": "categorical",
               "context_event_note": "context", "identifier": "context", "label": "excluded", "annotation": "excluded"}
    decisions = {c["sanitized_name"]: aliases.get(c["role"], c["role"]) for c in columns}
    if len(decisions) != len(columns):
        raise ValueError("Feature config contains duplicate column decisions")
    allowed = {"numeric", "categorical", "context", "excluded", "timestamp"}
    if any(role not in allowed for role in decisions.values()):
        raise ValueError("Feature config contains an unsupported role")
    timestamp = dataset["manifest"].get("timestamp")
    if dataset["storage_version"] == 2:
        timestamp = "timestamp"
    if timestamp and any((name == timestamp) != (role == "timestamp") for name, role in decisions.items()):
        raise ValueError("Feature configs cannot change timestamp identity. Re-ingest with a manifest")
    if sum(role == "timestamp" for role in decisions.values()) > 1:
        raise ValueError("Feature config requires a unique timestamp")
    with contextlib.nullcontext(connection) if connection is not None else get_connection(db_path) as conn:
        if dataset["storage_version"] == 2:
            metadata = json.loads(conn.execute("SELECT metadata_json FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()[0])
            known = {c["name"] for c in metadata["columns"]} | {"timestamp"}
        else:
            known = {row["name"] for row in conn.execute(f"PRAGMA table_info({_quote(f'raw_dataset_{int(dataset_id)}')})")
                     if row["name"] not in {"row_id", "dataset_id", "source_file_id"}}
        if set(decisions) - known:
            raise ValueError(f"Unknown feature config columns: {sorted(set(decisions) - known)}")
        if any(is_label(name) and role != "excluded" for name, role in decisions.items()):
            raise ValueError("Annotations cannot be selected as telemetry features or context")
        gap = dataset["manifest"].get("gap_column")
        if gap:
            gap_name = next((c["name"] for c in dataset["columns"] if c["source_name"] == gap), gap)
            if decisions.get(gap_name) != "context":
                raise ValueError("The declared gap column must remain context")
        if dataset["storage_version"] == 2:
            for column in metadata["columns"]:
                column["role"] = decisions.get(column["name"], "excluded")
            metadata["manifest"]["feature_roles"] = decisions
            if validate_only:
                return {**dataset, **metadata}
            conn.execute("UPDATE datasets SET metadata_json=?,updated_at=? WHERE dataset_id=?",
                         (json.dumps(metadata), utc_now(), dataset_id))
        else:
            if validate_only:
                return dataset
            for name in known:
                role = decisions.get(name, "excluded")
                cursor = conn.execute("UPDATE dataset_columns SET role=?,updated_at=? WHERE dataset_id=? AND sanitized_name=?",
                                      (role, utc_now(), dataset_id, name))
                if not cursor.rowcount:
                    conn.execute("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,role,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                                 (dataset_id, name, name, role, utc_now(), utc_now()))


def load_dataset(dataset_id, *, db_path=None) -> dict:
    row = get_dataset(int(dataset_id), db_path=db_path)
    if row is None:
        raise ValueError(f"Dataset {dataset_id} does not exist")
    metadata = json.loads(row.get("metadata_json") or "{}")
    result = {**row, **metadata}
    if result.get("storage_version") == 2:
        return result
    with get_connection(db_path) as conn:
        columns = conn.execute("SELECT column_name,sanitized_name,role,inferred_type FROM dataset_columns WHERE dataset_id=? ORDER BY dataset_column_id", (int(dataset_id),)).fetchall()
    aliases = {"numeric_telemetry_feature": "numeric", "categorical_feature": "categorical", "context_event_note": "context", "identifier": "context"}
    safe = [{"name": c["sanitized_name"] or c["column_name"], "source_name": c["column_name"], "role": aliases.get(c["role"], c["role"]), "dtype": c["inferred_type"]} for c in columns if not is_label(c["column_name"]) and not is_label(c["sanitized_name"] or "") and c["role"] not in {"label", "annotation", "excluded"}]
    timestamps = [c for c in safe if c["role"] == "timestamp"]
    if not timestamps:
        timestamps = [c for c in safe if c["name"] in {"timestamp", "datetime", "time", "date_time"}]
    timestamp_name = timestamps[0]["name"] if len(timestamps) == 1 else None
    if len(timestamps) > 1:
        raise ValueError("Legacy dataset has ambiguous timestamps. Choose one in its feature config or reimport with a manifest")
    result.update({"storage_version": 1, "legacy": True,
                   "manifest": {"version": 1, "timestamp": timestamp_name, "timezone": "UTC", "entity_columns": [], "splits": {"fractions": [.7, .15, .15]},
                                **({"ordering": "row"} if timestamp_name is None else {})},
                   "columns": [c for c in safe if c["name"] != timestamp_name]})
    return result


def iter_batches(dataset_id, *, db_path=None, batch_size=10000, _dataset=None) -> Iterator[pa.RecordBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metadata = _dataset if _dataset is not None else load_dataset(dataset_id, db_path=db_path)
    if metadata.get("storage_version") == 2:
        # This allowlist is also a defense against an annotation accidentally added later.
        names = CORE + [c["name"] for c in metadata["columns"] if c["role"] not in {"label", "annotation", "excluded"} and not is_label(c["source_name"])]
        numeric = {c["name"]: c for c in metadata["columns"] if c["role"] == "numeric"}
        for batch in _parquet_batches(metadata["telemetry_path"], columns=names, batch_size=batch_size):
            for name, column in numeric.items():
                index = batch.schema.get_field_index(name)
                if not pa.types.is_floating(batch.schema.field(index).type):
                    values = [_numeric(value, column["source_name"]) for value in batch.column(index).to_pylist()]
                    batch = batch.set_column(index, name, pa.array(values, type=pa.float64()))
            yield batch
    else:
        with get_connection(db_path) as conn:
            table = f"raw_dataset_{int(dataset_id)}"
            available = [row[1] for row in conn.execute(f"PRAGMA table_info({_quote(table)})")]
            names = [name for name in available if not is_label(name)]
            excluded = {c["sanitized_name"] for c in conn.execute("SELECT sanitized_name FROM dataset_columns WHERE dataset_id=? AND role IN ('label','annotation','excluded')", (int(dataset_id),))}
            names = [name for name in names if name not in excluded]
            if not names:
                raise ValueError(f"Legacy raw table {table} is absent")
            timestamp_name = metadata["manifest"].get("timestamp")
            if timestamp_name and timestamp_name not in names:
                raise ValueError("Legacy dataset requires a unique timestamp column. Reimport with an explicit manifest")
            cursor = conn.execute(f"SELECT {','.join(map(_quote,names))} FROM {_quote(table)} ORDER BY row_id")
            previous = None
            while records := cursor.fetchmany(batch_size):
                canonical = []
                for record in records:
                    row = dict(record)
                    stamp, nanos = (_timestamp(row[timestamp_name], metadata["manifest"]) if timestamp_name else
                                    (None, int(row["row_id"])))
                    if previous is not None and nanos <= previous:
                        raise ValueError("Legacy timestamps are unsorted or duplicated. Please reimport to sort while preserving historical row references")
                    previous = nanos
                    row.update(timestamp=stamp, timestamp_ns=nanos, entity_id="")
                    for column in metadata["columns"]:
                        if column["role"] == "numeric":
                            row[column["name"]] = _numeric(row[column["name"]], column["source_name"])
                    canonical.append(row)
                yield pa.RecordBatch.from_pylist(canonical)


def read_window(dataset_id, start_row_id, end_row_id, *, db_path=None) -> list[dict]:
    if start_row_id < 1 or end_row_id < start_row_id:
        raise ValueError("Expected 1 <= start_row_id <= end_row_id")
    metadata = load_dataset(dataset_id, db_path=db_path)
    if metadata.get("storage_version") == 2:
        names = CORE + [c["name"] for c in metadata["columns"] if c["role"] not in {"label", "annotation", "excluded"} and not is_label(c["source_name"])]
        rows = []
        for part in _parquet_parts(metadata["telemetry_path"]):
            with contextlib.closing(pq.ParquetFile(part)) as reader:
                row_id_index = reader.schema_arrow.get_field_index("row_id")
                groups = []
                for group in range(reader.num_row_groups):
                    stats = reader.metadata.row_group(group).column(row_id_index).statistics
                    if stats is not None and stats.has_min_max:
                        if stats.min > end_row_id:
                            break
                        if stats.max < start_row_id:
                            continue
                    groups.append(group)
                for batch in reader.iter_batches(columns=names, row_groups=groups, batch_size=2048, use_threads=False):
                    ids = batch.column("row_id")
                    selected = batch.filter(pc.and_(pc.greater_equal(ids, start_row_id), pc.less_equal(ids, end_row_id)))
                    rows.extend(selected.to_pylist())
                    if batch.num_rows and ids[-1].as_py() >= end_row_id:
                        return rows
        return rows
    # Historical windows must retain their original raw values and row references,
    # even if the source would require reimport before modern chronological fitting.
    with get_connection(db_path) as conn:
        table = f"raw_dataset_{int(dataset_id)}"
        available = [row[1] for row in conn.execute(f"PRAGMA table_info({_quote(table)})")]
        allowed = {"row_id", "dataset_id", "source_file_id", metadata["manifest"].get("timestamp"), *[c["name"] for c in metadata["columns"]]}
        names = [name for name in available if name in allowed and not is_label(name)]
        if not names:
            raise ValueError(f"Legacy raw table {table} is absent")
        rows = [dict(row) for row in conn.execute(f"SELECT {','.join(map(_quote, names))} FROM {_quote(table)} WHERE row_id BETWEEN ? AND ? ORDER BY row_id", (start_row_id, end_row_id))]
        timestamp_name = metadata["manifest"].get("timestamp")
        if timestamp_name:
            for row in rows:
                row["timestamp"] = row.get(timestamp_name)
        return rows


def profile_dataset(dataset_id, *, db_path=None, batch_size=10000, cardinality_limit=1024) -> dict:
    """Describe the dataset in batches. Training decides feature eligibility later."""
    metadata = load_dataset(dataset_id, db_path=db_path)
    columns = [c for c in metadata["columns"] if c["role"] not in {"excluded", "label", "annotation"}]
    stats = {c["name"]: {"count": 0, "missing": 0, "minimum": None, "maximum": None,
             "unique": set(), "unique_count_approximate": False} for c in columns}
    for batch in iter_batches(dataset_id, db_path=db_path, batch_size=batch_size):
        for column in columns:
            name = column["name"]
            values = batch.column(batch.schema.get_field_index(name)).to_pylist()
            state = stats[name]
            state["count"] += len(values)
            for value in values:
                if value is None or value == "":
                    state["missing"] += 1
                    continue
                if len(state["unique"]) < cardinality_limit:
                    state["unique"].add(value)
                elif value not in state["unique"]:
                    state["unique_count_approximate"] = True
                if column["role"] == "numeric":
                    state["minimum"] = value if state["minimum"] is None else min(state["minimum"], value)
                    state["maximum"] = value if state["maximum"] is None else max(state["maximum"], value)
    for state in stats.values():
        state["unique_count_lower_bound"] = len(state.pop("unique"))
        state["missing_ratio"] = state["missing"] / state["count"] if state["count"] else None
    return {"dataset_id": int(dataset_id), "row_count": metadata["row_count"], "columns": stats,
            "purpose": "descriptive only; no model feature eligibility decisions"}
