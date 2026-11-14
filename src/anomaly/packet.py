from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.db import get_anomaly, get_connection, get_model, init_db, table_exists
from src.ingestion.manifest import is_label


_LEGACY_REDACTION_NOTE = (
    "Legacy checkpoint used annotation-like feature identities. Names and contribution "
    "details were redacted. Retrain with a version 2 manifest."
)


def build_anomaly_packet(
    anomaly_id: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    init_db(db_path)
    anomaly = get_anomaly(anomaly_id, db_path=db_path)
    if anomaly is None:
        raise ValueError(f"Anomaly {anomaly_id} does not exist.")

    dataset_id = int(anomaly["dataset_id"])
    model_id = int(anomaly["model_id"])
    model = get_model(model_id, db_path=db_path)
    if model is None:
        raise ValueError(f"Model {model_id} does not exist.")

    model_metadata = _json_dict(model["metadata_json"])
    if model_metadata.get("pipeline_version") == 2:
        return _canonical_packet(dict(anomaly), model_metadata, db_path=db_path)
    model_config = model_metadata.get("config") or {}
    features_used = model_metadata.get("feature_columns") or []
    raw_table = f"raw_dataset_{dataset_id}"
    start_row_id = int(anomaly["start_row_id"])
    end_row_id = int(anomaly["end_row_id"])
    top_contributing_features = _json_list(anomaly["contributing_features_json"])
    blocked_names = _legacy_excluded_names(dataset_id, db_path=db_path)
    features_used, top_contributing_features, redacted_feature_names = (
        _redact_legacy_annotation_features(features_used, top_contributing_features, blocked_names)
    )

    from src.rag.telemetry_tools import MAX_WINDOW_ROWS, get_raw_window, get_context_window
    if start_row_id <= 0 or not 0 <= end_row_id - start_row_id < MAX_WINDOW_ROWS:
        raise ValueError(f"Select an ordered diagnostic range of at most {MAX_WINDOW_ROWS} rows.")
    model_rows = get_raw_window(dataset_id, start_row_id, end_row_id, db_path=db_path)
    context_rows = get_context_window(dataset_id, start_row_id, end_row_id, db_path=db_path)
    raw_exists = _table_exists(raw_table, db_path=db_path)
    prepared_values = isinstance(model_rows, dict)
    if prepared_values:
        if raw_exists:
            raise ValueError("Raw telemetry window is unavailable.")
        table = model_metadata.get("model_ready_table") or f"model_ready_dataset_{dataset_id}"
        model_rows = _prepared_rows(table, dataset_id, start_row_id, end_row_id,
                                    blocked_names, db_path=db_path, required=True)
    if isinstance(context_rows, dict):
        context_rows = _prepared_rows(f"context_ready_dataset_{dataset_id}", dataset_id,
                                      start_row_id, end_row_id, blocked_names, db_path=db_path)
    context_rows = [{name: value for name, value in row.items() if not _excluded_feature(name, blocked_names)}
                    for row in context_rows]

    top_feature_names = [
        str(item["feature"])
        for item in top_contributing_features
        if isinstance(item, dict) and item.get("feature")
    ]
    numeric_summary = _numeric_summary(model_rows, top_feature_names)
    for item in numeric_summary.values():
        item.update(unit=None, value_space="prepared_model_values" if prepared_values else "original_values",
                    original_units_available=not prepared_values)
    categorical_changes = _categorical_changes(context_rows)
    stored_packet = _json_dict(anomaly["packet_json"])
    input_start = stored_packet.get("input_window_start_timestamp") or (
        model_rows[0].get("timestamp") if model_rows else None
    ) or anomaly["start_timestamp"]
    input_end = stored_packet.get("input_window_end_timestamp") or (
        model_rows[-1].get("timestamp") if model_rows else None
    ) or anomaly["end_timestamp"]

    return {
        "anomaly_id": anomaly_id,
        "run_id": anomaly["run_id"],
        "dataset_id": dataset_id,
        "model_id": model_id,
        "legacy": True,
        "score_timestamp": anomaly["score_timestamp"] or anomaly["end_timestamp"],
        "time_window": {
            "window_index": int(anomaly["window_index"]),
            "start_row_id": end_row_id,
            "end_row_id": end_row_id,
            "start_timestamp": anomaly["score_timestamp"] or anomaly["end_timestamp"],
            "end_timestamp": anomaly["score_timestamp"] or anomaly["end_timestamp"],
        },
        "input_window": {
            "start_row_id": start_row_id,
            "end_row_id": end_row_id,
            "start_timestamp": input_start,
            "end_timestamp": input_end,
        },
        "anomaly_score": float(anomaly["anomaly_score"]),
        "threshold": float(anomaly["threshold"]),
        "model_context": {
            "features_used": features_used,
            "window_size": model_config.get("window_size"),
            "diagnostic_value_space": "prepared_model_values" if prepared_values else "original_values",
            "original_units_available": not prepared_values,
            "diagnostic_note": (
                "Only legacy prepared values are available. These may be standardized or encoded. "
                "Original physical units cannot be recovered."
            ) if prepared_values else None,
        },
        "numeric_summary": numeric_summary,
        "categorical_changes": categorical_changes,
        "top_contributing_features": top_contributing_features,
        "redacted_annotation_feature_count": len(redacted_feature_names),
        "redaction_note": _LEGACY_REDACTION_NOTE if redacted_feature_names else None,
        "raw_data_reference": {
            "table": raw_table,
            "available": raw_exists,
            "query": {
                "start_row_id": start_row_id,
                "end_row_id": end_row_id,
            },
        },
    }


def _canonical_packet(anomaly: dict, model_metadata: dict, *, db_path=None) -> dict:
    from src.db import get_detection_run
    from src.ingestion.store import load_dataset
    from src.rag.telemetry_tools import get_raw_window, get_context_window

    dataset_id, model_id = int(anomaly["dataset_id"]), int(anomaly["model_id"])
    run = get_detection_run(anomaly["run_id"], dataset_id=dataset_id, model_id=model_id, db_path=db_path)
    if run is None:
        raise ValueError("Anomaly does not belong to a completed detection run.")
    start, end = int(anomaly["start_row_id"]), int(anomaly["end_row_id"])
    rows = get_raw_window(dataset_id, start, end, db_path=db_path)
    context = get_context_window(dataset_id, start, end, db_path=db_path)
    if isinstance(rows, dict) or isinstance(context, dict):
        raise ValueError("Target telemetry window is unavailable or exceeds the diagnostic row limit.")
    dataset = load_dataset(dataset_id, db_path=db_path)
    contributions = _json_list(anomaly["contributing_features_json"])
    numeric = {item["name"]: item for item in dataset["columns"] if item["role"] == "numeric"}
    source_to_internal = {item.get("source_name", key): key for key, item in numeric.items()}
    requested = [str(item["feature"]) for item in contributions if isinstance(item, dict) and item.get("feature")]
    features = [source_to_internal.get(name, name) for name in requested
                if name in source_to_internal or name in numeric]
    summary = _numeric_summary(rows, features)
    for name, item in summary.items():
        item["unit"] = numeric[name].get("unit")
        item["source_name"] = numeric[name].get("source_name", name)
    run_metadata = _json_dict(run["metadata_json"])
    return {
        "anomaly_id": anomaly["anomaly_id"], "run_id": run["run_id"],
        "dataset_id": dataset_id, "model_id": model_id,
        "score_timestamp": anomaly["score_timestamp"] or anomaly["end_timestamp"],
        "time_window": {"start_row_id": end, "end_row_id": end,
                        "start_timestamp": anomaly["score_timestamp"] or anomaly["end_timestamp"],
                        "end_timestamp": anomaly["score_timestamp"] or anomaly["end_timestamp"]},
        "input_window": {"start_row_id": start, "end_row_id": end,
                         "start_timestamp": rows[0]["timestamp"] if rows else None,
                         "end_timestamp": rows[-1]["timestamp"] if rows else None},
        "anomaly_score": float(anomaly["anomaly_score"]), "threshold": float(anomaly["threshold"]),
        "model_context": {"features_used": model_metadata.get("feature_columns", []),
                          "window_size": (model_metadata.get("config") or {}).get("window_size"),
                          "transform_fingerprint": run_metadata.get("transform_fingerprint")},
        "documented_mappings": dataset.get("manifest", {}).get("documented_mappings", {}),
        "numeric_summary": summary, "categorical_changes": _categorical_changes(context),
        "top_contributing_features": contributions,
        "contribution_note": "Model feature contributions are explanatory scores, not channel detections.",
        "raw_data_reference": {"dataset_id": dataset_id, "run_id": run["run_id"], "available": True,
                               "query": {"start_row_id": start, "end_row_id": end}},
    }


def _numeric_summary(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    if not rows:
        return summary

    available = set(rows[0].keys())
    for feature in feature_names:
        if is_label(feature):
            continue
        if feature not in available:
            continue
        values = [_to_float(row.get(feature)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            continue
        start = values[0]
        end = values[-1]
        delta = end - start
        summary[feature] = {
            "start": start,
            "end": end,
            "min": min(values),
            "max": max(values),
            "delta": delta,
        }
    return summary


def _categorical_changes(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    changes: dict[str, list[Any]] = {}
    if not rows:
        return changes

    excluded = {"row_id", "dataset_id", "timestamp", "timestamp_ns", "entity_id", "source_file_id"}
    for column in rows[0].keys():
        if column in excluded or is_label(column):
            continue
        values = [row.get(column) for row in rows]
        compact_values = []
        seen = set()
        for value in values:
            marker = json.dumps(value, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            compact_values.append(value)
        if len(compact_values) > 1:
            changes[column] = compact_values
    return changes


def _redact_legacy_annotation_features(
    feature_columns: Any,
    contributions: list[Any],
    blocked_names: set[str] | None = None,
) -> tuple[list[Any], list[Any], set[str]]:
    redacted: set[str] = set()
    safe_features = []
    for feature in feature_columns if isinstance(feature_columns, list) else []:
        if isinstance(feature, str) and _excluded_feature(feature, blocked_names or set()):
            redacted.add(feature)
            continue
        safe_features.append(feature)

    safe_contributions = []
    for contribution in contributions:
        feature = contribution.get("feature") if isinstance(contribution, dict) else None
        if isinstance(feature, str) and _excluded_feature(feature, blocked_names or set()):
            redacted.add(feature)
            continue
        safe_contributions.append(contribution)
    return safe_features, safe_contributions, redacted


def _excluded_feature(name: str, blocked_names: set[str]) -> bool:
    return is_label(name) or any(name == blocked or name.startswith(blocked + "__")
                                 for blocked in blocked_names)


def _legacy_excluded_names(dataset_id: int, *, db_path=None) -> set[str]:
    with get_connection(db_path) as conn:
        columns = conn.execute("SELECT column_name,sanitized_name,role FROM dataset_columns WHERE dataset_id=?",
                               (dataset_id,)).fetchall()
    return {name for column in columns for name in (column["column_name"], column["sanitized_name"])
            if name and (column["role"] in {"label", "annotation", "excluded"} or is_label(name))}


def _prepared_rows(table: str, dataset_id: int, start: int, end: int, blocked_names: set[str], *,
                   db_path=None, required: bool = False) -> list[dict]:
    """Read a bounded legacy window whose original units may be unavailable."""
    quote = lambda name: '"' + name.replace('"', '""') + '"'
    with get_connection(db_path) as conn:
        if not table_exists(conn, table):
            if required:
                raise ValueError(f"Required legacy prepared table {table} does not exist.")
            return []
        names = [row["name"] for row in conn.execute(f"PRAGMA table_info({quote(table)})")
                 if not _excluded_feature(row["name"], blocked_names)]
        if not {"row_id", "dataset_id"} <= set(names):
            raise ValueError("Legacy prepared telemetry requires row_id and dataset_id columns.")
        selected = ",".join(quote(name) for name in names)
        rows = conn.execute(f"SELECT {selected} FROM {quote(table)} WHERE dataset_id=? "
                            "AND row_id BETWEEN ? AND ? ORDER BY row_id", (dataset_id, start, end)).fetchall()
    return [{key: value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
             for key, value in dict(row).items()} for row in rows]


def _table_exists(table_name: str, *, db_path: str | Path | None) -> bool:
    with get_connection(db_path) as conn:
        return table_exists(conn, table_name)


def _json_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, dict) else {}


def _json_list(payload: str | None) -> list[Any]:
    if not payload:
        return []
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, list) else []


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None
