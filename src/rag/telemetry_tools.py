"""Read bounded telemetry windows for packets and RAG, excluding annotations."""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from src.db import get_anomaly, init_db

MAX_WINDOW_ROWS = 10000


def get_raw_window(dataset_id: int, start_row_id: int, end_row_id: int, *,
                   db_path: str | Path | None = None) -> list[dict[str, Any]] | dict[str, str]:
    from src.ingestion.store import read_window

    if any(type(value) is not int or value <= 0 for value in (dataset_id, start_row_id, end_row_id)):
        return _error("invalid_row_range", "Dataset and row IDs must be positive integers.")
    if not 0 <= end_row_id - start_row_id < MAX_WINDOW_ROWS:
        return _error("invalid_row_range", f"Select an ordered range of at most {MAX_WINDOW_ROWS} rows.")
    try:
        return read_window(dataset_id, start_row_id, end_row_id, db_path=db_path)
    except ValueError as exc:
        return _error("invalid_dataset", str(exc))


def get_context_window(dataset_id: int, start_row_id: int, end_row_id: int, *,
                       db_path: str | Path | None = None) -> list[dict[str, Any]] | dict[str, str]:
    from src.ingestion.store import load_dataset

    rows = get_raw_window(dataset_id, start_row_id, end_row_id, db_path=db_path)
    if isinstance(rows, dict):
        if rows.get("error") != "invalid_dataset":
            return rows
        prepared = _prepared_window(dataset_id, start_row_id, end_row_id, context=True, db_path=db_path)
        return prepared if prepared is not None else rows
    dataset = load_dataset(dataset_id, db_path=db_path)
    names = {"row_id", "timestamp", "timestamp_ns", "entity_id"}
    names.update(column["name"] for column in dataset["columns"] if column["role"] in {"context", "categorical"})
    return [{name: value for name, value in row.items() if name in names} for row in rows]


def get_column_summary(dataset_id: int, column_name: str, start_row_id: int, end_row_id: int, *,
                       db_path: str | Path | None = None) -> dict[str, Any]:
    from src.ingestion.store import load_dataset

    prepared_only = False
    rows = get_raw_window(dataset_id, start_row_id, end_row_id, db_path=db_path)
    if isinstance(rows, dict):
        if rows.get("error") != "invalid_dataset":
            return rows
        prepared = _prepared_window(dataset_id, start_row_id, end_row_id, context=False, db_path=db_path)
        if prepared is None or not prepared or column_name not in prepared[0]:
            return rows
        rows = prepared
        prepared_only = True
    dataset = load_dataset(dataset_id, db_path=db_path)
    column = next((item for item in dataset["columns"] if item["name"] == column_name and item["role"] == "numeric"), None)
    if column is None and prepared_only and dataset["storage_version"] == 1 and rows and column_name in rows[0]:
        # A prepared-only historical catalog may have no raw column metadata.
        from src.ingestion.store import _quote
        from src.db import get_connection
        with get_connection(db_path) as conn:
            numeric = {r["name"] for r in conn.execute(f"PRAGMA table_info({_quote(f'model_ready_dataset_{dataset_id}')})")
                       if r["type"].upper() in {"REAL", "INTEGER", "NUMERIC", "FLOAT", "DOUBLE"}}
        if column_name in numeric and column_name not in {"row_id", "dataset_id"}:
            column = {"value_space": "legacy_prepared", "unit": None}
    if column is None:
        # Prepared one-hot features are queryable alongside numeric values in original units.
        from src.ingestion.manifest import is_label
        from src.db import get_connection, table_exists
        categorical = [c for c in dataset["columns"] if c["role"] == "categorical"]
        if dataset["storage_version"] != 1 or is_label(column_name) or not any(
                column_name.startswith(c["name"] + "__") for c in categorical):
            return _error("unknown_column", "Select a numeric telemetry column. Annotations are excluded.")
        from src.ingestion.store import _quote
        table = f"model_ready_dataset_{dataset_id}"
        with get_connection(db_path) as conn:
            if not table_exists(conn, table) or column_name not in {r["name"] for r in conn.execute(f"PRAGMA table_info({_quote(table)})")}:
                return _error("unknown_column", "Prepared feature does not exist.")
            rows = [dict(r) for r in conn.execute(f"SELECT {_quote(column_name)} FROM {_quote(table)} WHERE row_id BETWEEN ? AND ? ORDER BY row_id", (start_row_id, end_row_id))]
        column = {"value_space": "one_hot"}
    values = [_to_float(row.get(column_name)) for row in rows]
    present = [value for value in values if value is not None]
    first, last = (present[0], present[-1]) if present else (None, None)
    return {"dataset_id": dataset_id, "column_name": column_name, "unit": column.get("unit"),
            "value_space": column.get("value_space", "original"),
            "start_row_id": start_row_id, "end_row_id": end_row_id,
            "count": len(values), "missing_count": len(values) - len(present),
            "min": min(present) if present else None, "max": max(present) if present else None,
            "mean": sum(present) / len(present) if present else None, "first": first, "last": last,
            "delta": last - first if present else None}


def _prepared_window(dataset_id, start_row_id, end_row_id, *, context, db_path):
    """Read bounded historical context/features when their raw source is absent."""
    from src.db import get_connection, table_exists
    from src.ingestion.store import load_dataset, _quote
    from src.ingestion.manifest import is_label
    dataset = load_dataset(dataset_id, db_path=db_path)
    if dataset["storage_version"] != 1:
        return None
    table = f"{'context' if context else 'model'}_ready_dataset_{dataset_id}"
    with get_connection(db_path) as conn:
        if not table_exists(conn, table):
            return None
        blocked = {r["name"] for r in conn.execute(
            "SELECT COALESCE(sanitized_name,column_name) AS name FROM dataset_columns WHERE dataset_id=? AND role IN ('label','annotation','excluded')", (dataset_id,))}
        names = [r["name"] for r in conn.execute(f"PRAGMA table_info({_quote(table)})")
                 if not is_label(r["name"]) and not any(
                     r["name"] == name or r["name"].startswith(name + "__") for name in blocked if name)]
        if not names:
            return None
        return [dict(r) for r in conn.execute(f"SELECT {','.join(map(_quote, names))} FROM {_quote(table)} WHERE dataset_id=? AND row_id BETWEEN ? AND ? ORDER BY row_id",
                                               (dataset_id, start_row_id, end_row_id))]


def get_rows_around_peak(dataset_id: int, anomaly_id: str, *, radius: int = 2,
                         db_path: str | Path | None = None) -> list[dict[str, Any]] | dict[str, str]:
    init_db(db_path)
    if type(radius) is not int or not 0 <= radius < MAX_WINDOW_ROWS // 2:
        return _error("invalid_radius", "Radius must be a nonnegative integer below 5000.")
    anomaly = get_anomaly(anomaly_id, db_path=db_path)
    if anomaly is None or int(anomaly["dataset_id"]) != dataset_id:
        return _error("anomaly_not_found", "No anomaly matches this target dataset.")
    try:
        packet = json.loads(anomaly.get("packet_json") or "{}")
    except (TypeError, ValueError):
        packet = {}
    peak = next((packet.get(key) for key in ("peak_row_id", "peak_row")
                 if isinstance(packet, dict) and type(packet.get(key)) is int and packet[key] > 0), None)
    row_id = peak or (int(anomaly["end_row_id"]) if anomaly.get("score_timestamp") else
                      (int(anomaly["start_row_id"]) + int(anomaly["end_row_id"])) // 2)
    return get_context_window(dataset_id, max(1, row_id - radius), row_id + radius, db_path=db_path)


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _error(code: str, message: str) -> dict[str, str]:
    return {"error": code, "message": message}
