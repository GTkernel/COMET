from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.db import get_connection, get_dataset, get_model, init_db, insert_anomaly, table_exists
from src.modeling.lstm_autoencoder import LSTMAutoencoder, create_sliding_windows


def detect_anomalies(
    dataset_id: int,
    model_id: int,
    *,
    db_path: str | Path | None = None,
    partition: str = "all",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Score prepared telemetry windows with a trained model and store detected anomalies."""
    init_db(db_path)

    # Checks
    dataset = get_dataset(dataset_id, db_path=db_path)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    model_row = get_model(model_id, db_path=db_path)
    if model_row is None:
        raise ValueError(f"Model {model_id} does not exist.")
    if _json_dict(model_row["metadata_json"]).get("pipeline_version") == 2:
        from src.modeling.pipeline import detect
        return detect(dataset_id, model_id, db_path=db_path, partition=partition, batch_size=batch_size)
    if int(model_row["dataset_id"]) != dataset_id:
        raise ValueError("Legacy checkpoints support their original dataset only. Retrain to use another dataset.")
    if partition != "all":
        raise ValueError("Legacy checkpoints do not record reliable partitions. Retrain for partition-aware scoring.")

    return _detect_legacy_run(dataset_id, model_id, model_row, db_path=db_path,
                              batch_size=batch_size)


def _detect_legacy_run(dataset_id: int, model_id: int, model_row: dict, *, db_path, batch_size) -> dict:
    """Record a separate immutable run for a legacy checkpoint and its original dataset."""
    import math
    import uuid
    import pyarrow as pa
    from src.ingestion.store import _ParquetPartWriter
    from src.common.paths import STORAGE_ROOT
    from src.db import utc_now
    from src.modeling.window_dataset import WindowDataset
    from src.modeling.scoring import score_batches
    import numpy as np

    model_path = Path(model_row["model_path"])
    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
    metadata = _json_dict(model_row["metadata_json"])
    config = checkpoint.get("config") or metadata.get("config") or {}
    features = checkpoint.get("feature_columns") or metadata.get("feature_columns")
    if not features:
        raise ValueError("Legacy checkpoint has no recorded feature columns")
    table = checkpoint.get("model_ready_table") or f"model_ready_dataset_{dataset_id}"
    if table != f"model_ready_dataset_{dataset_id}":
        raise ValueError("Legacy checkpoint table does not match its original dataset")
    rows, values = _load_model_ready_rows(dataset_id, table, features, db_path=db_path)
    window_size, stride = int(config.get("window_size", 100)), int(config.get("stride", 1))
    if window_size < 1 or stride < 1:
        raise ValueError("Legacy window_size and stride must be positive")
    threshold = float(model_row["threshold"])
    if not math.isfinite(threshold):
        raise ValueError("Legacy threshold must be finite")
    starts = np.arange(0, max(0, len(rows) - window_size + 1), stride, dtype=np.int64)
    model = LSTMAutoencoder(len(features), int(config.get("hidden_size", 64)), int(config.get("num_layers", 1)))
    model.load_state_dict(checkpoint["model_state_dict"])
    run_id = uuid.uuid4().hex
    directory = STORAGE_ROOT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    scores_path = directory / "scores.parquet"
    run = {"run_id": run_id, "dataset_id": dataset_id, "model_id": model_id,
           "pipeline_version": 1, "legacy": True, "target_dataset_id": dataset_id,
           "partition": "all", "window_size": window_size, "stride": stride,
           "batch_size": int(batch_size or config.get("batch_size", 128)),
           "threshold": threshold, "scores_path": str(scores_path), "scores_format": "parquet-parts",
           "coverage_policy": "window_end; original legacy preprocessing unavailable"}
    with get_connection(db_path) as conn:
        conn.execute("INSERT INTO detection_runs(run_id,dataset_id,model_id,status,artifact_dir,metadata_json,created_at) VALUES(?,?,?,?,?,?,?)",
                     (run_id, dataset_id, model_id, "running", str(directory), json.dumps(run), utc_now()))
    anomaly_stream = None
    try:
        score_map = np.lib.format.open_memmap(directory / "scores.npy", mode="w+", dtype="float64", shape=(len(rows),))
        score_map[:] = np.nan
        windows = WindowDataset(values, starts, window_size)
        anomaly_log = directory / "anomalies.jsonl"
        anomaly_stream = anomaly_log.open("w", encoding="utf-8")
        anomaly_count = 0
        for offset, scores, feature_errors in score_batches(model, windows, run["batch_size"]):
            for local, score in enumerate(scores):
                index = offset + local
                start = int(starts[index])
                end = start + window_size - 1
                score_map[end] = float(score)
                if float(score) <= threshold:
                    continue
                timestamp = rows[end].get("timestamp")
                top, method = _top_contributing_features(
                    feature_columns=features,
                    feature_errors=torch.as_tensor(feature_errors[local]),
                    window=values[start:end + 1])
                anomaly_id = f"legacy_{run_id}_{index}"
                packet = {"anomaly_id": anomaly_id, "run_id": run_id, "legacy": True,
                          "dataset_id": dataset_id, "model_id": model_id,
                          "window_index": index, "start_row_id": int(rows[start]["row_id"]),
                          "end_row_id": int(rows[end]["row_id"]),
                          "input_window_start_timestamp": rows[start].get("timestamp"),
                          "score_timestamp": timestamp, "anomaly_score": float(score),
                          "threshold": threshold, "top_contributing_features": top,
                          "contribution_method": method}
                anomaly_stream.write(json.dumps((anomaly_id, dataset_id, model_id, index,
                                                 packet["start_row_id"], packet["end_row_id"], timestamp,
                                                 timestamp, float(score), threshold, json.dumps(top),
                                                 json.dumps(packet), utc_now(), run_id, timestamp)) + "\n")
                anomaly_count += 1
        anomaly_stream.close()
        anomaly_stream = None
        score_map.flush()
        schema = pa.schema([("row_id", pa.int64()), ("timestamp", pa.string()),
                            ("timestamp_ns", pa.int64()), ("entity_id", pa.string()),
                            ("partition", pa.string()), ("score", pa.float64()),
                            ("is_anomaly", pa.bool_()), ("coverage", pa.string())])
        with _ParquetPartWriter(scores_path, schema) as writer:
            for first in range(0, len(rows), 10000):
                output = {name: [] for name in schema.names}
                for i in range(first, min(first + 10000, len(rows))):
                    value = float(score_map[i])
                    timestamp = rows[i].get("timestamp")
                    try:
                        import pandas as pd
                        timestamp_ns = pd.Timestamp(timestamp).value if timestamp else None
                    except (ValueError, TypeError, OverflowError):
                        timestamp_ns = None
                    record = (int(rows[i]["row_id"]), timestamp, timestamp_ns, "", "legacy",
                              value if math.isfinite(value) else None,
                              bool(math.isfinite(value) and value > threshold),
                              "scored" if math.isfinite(value) else "warmup_or_stride")
                    for key, item in zip(schema.names, record):
                        output[key].append(item)
                writer.write_batch(pa.RecordBatch.from_pydict(output, schema=schema))
        run.update(windows_scored=len(starts), anomalies_detected=anomaly_count,
                   row_count=len(rows), coverage={"scored": len(starts),
                                                     "warmup_or_stride": len(rows)-len(starts)})
        (directory / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
        with get_connection(db_path) as conn:
            conn.execute("BEGIN")
            with anomaly_log.open("r", encoding="utf-8") as stream:
                from itertools import islice
                while lines := list(islice(stream, 1000)):
                    conn.executemany("""INSERT INTO anomalies(anomaly_id,dataset_id,model_id,window_index,start_row_id,end_row_id,
                        start_timestamp,end_timestamp,anomaly_score,threshold,contributing_features_json,packet_json,created_at,run_id,score_timestamp)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", [json.loads(line) for line in lines])
            conn.execute("UPDATE detection_runs SET status='completed',metadata_json=?,completed_at=? WHERE run_id=?",
                         (json.dumps(run), utc_now(), run_id))
        return run
    except Exception:
        if anomaly_stream is not None:
            anomaly_stream.close()
        with get_connection(db_path) as conn:
            conn.execute("UPDATE detection_runs SET status='failed',completed_at=? WHERE run_id=?",
                         (utc_now(), run_id))
        raise


def _load_model_ready_rows(
    dataset_id: int,
    table_name: str,
    feature_columns: list[str],
    *,
    db_path: str | Path | None,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Load ordered model-ready rows and tensor values for the model feature columns."""
    with get_connection(db_path) as conn:
        if not table_exists(conn, table_name):
            raise ValueError(f"Required table {table_name} is missing. Run prepare first.")
        existing_columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
        }
        missing_features = [column for column in feature_columns if column not in existing_columns]
        if missing_features:
            raise ValueError(f"Model features are missing from {table_name}: {missing_features}")

        select_columns = ["row_id", *feature_columns]
        if "timestamp" in existing_columns:
            select_columns.insert(1, "timestamp")
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT {", ".join(_quote_identifier(column) for column in select_columns)}
                FROM {_quote_identifier(table_name)}
                WHERE dataset_id = ?
                ORDER BY row_id
                """,
                (dataset_id,),
            ).fetchall()
        ]

    if not rows:
        raise ValueError(f"No rows found in {table_name}.")
    values = torch.tensor(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=torch.float32,
    )
    return rows, values


def _top_contributing_features(
    *,
    feature_columns: list[str],
    feature_errors: torch.Tensor,
    window: torch.Tensor,
    limit: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """Rank features by reconstruction error, falling back to feature change when needed."""
    scores = [float(value) for value in feature_errors.tolist()]
    method = "per_feature_reconstruction_error"
    if not any(score > 0 for score in scores):
        scores = [float(abs(window[-1, index] - window[0, index])) for index in range(window.size(1))]
        method = "absolute_feature_change"

    ranked = sorted(
        zip(feature_columns, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature, "score": score, "rank": rank}
        for rank, (feature, score) in enumerate(ranked[:limit], start=1)
    ], method


def _json_dict(payload: str | None) -> dict[str, Any]:
    """Parse a JSON object payload and return an empty dict for missing payloads."""
    if not payload:
        return {}
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, dict) else {}


def _quote_identifier(identifier: str) -> str:
    """Return a SQLite-quoted identifier safe for generated SQL."""
    return '"' + identifier.replace('"', '""') + '"'
