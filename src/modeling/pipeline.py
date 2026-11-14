"""Fit telemetry models on training rows and record immutable scoring runs."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np

from src.db import get_connection, get_model, init_db, insert_model_metadata, utc_now
from src.modeling.provenance import dataset_provenance
from src.modeling.transforms import (PARTITIONS, build_row_index, fingerprint, fit_transform,
                                     resolve_features, search_index, transform_to_memmap)
from src.modeling.window_dataset import WindowDataset, build_window_index

DEFAULT_CONFIG = {"window_size": 60, "stride": 1, "hidden_size": 32, "num_layers": 1,
                  "batch_size": 128, "epochs": 3, "learning_rate": 0.001,
                  "threshold_percentile": 95, "seed": 0, "max_categories": 64,
                  "numeric_transform": "standard", "feature_eligibility": "require_all",
                  "window_centering": False}


def _root(storage_dir=None) -> Path:
    from src.common.paths import STORAGE_ROOT
    return Path(storage_dir) if storage_dir is not None else STORAGE_ROOT


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path=None, *, defaults=None) -> dict:
    config = {**DEFAULT_CONFIG, **(defaults or {})}
    if path is None and defaults is None:
        from src.modeling.train_model import resolve_training_config_path
        path = resolve_training_config_path()
    if path is not None:
        import yaml
        parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError("Training configuration must be a mapping")
        config.update({key: value for key, value in parsed.items() if value is not None})
    for key in ("window_size", "stride", "hidden_size", "num_layers", "batch_size", "epochs", "max_categories"):
        if isinstance(config[key], bool) or int(config[key]) != float(config[key]) or int(config[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
        config[key] = int(config[key])
    for key in ("learning_rate", "threshold_percentile"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
        config[key] = float(config[key])
    if config["threshold_percentile"] > 100:
        raise ValueError("threshold_percentile cannot exceed 100")
    config["seed"] = int(config["seed"])
    if config["numeric_transform"] not in ("standard", "asinh"):
        raise ValueError("numeric_transform must be standard or asinh")
    if config["feature_eligibility"] not in ("require_all", "observed_training"):
        raise ValueError("feature_eligibility must be require_all or observed_training")
    if not isinstance(config["window_centering"], bool):
        raise ValueError("window_centering must be a boolean")
    if config["seed"] < 0:
        raise ValueError("seed must be nonnegative")
    # Legacy configs split windows. Reusing that setting would
    # recreate leakage. Dataset manifests are the only split source in v2.
    if "train_test_split" in config:
        import warnings
        warnings.warn("Replace train_test_split with manifest splits. New models partition rows before fitting. Set splits.fractions in the manifest.", UserWarning, stacklevel=2)
        config.pop("train_test_split")
    return config


def prepare(dataset_id: int, *, db_path=None, storage_dir=None, _dataset=None) -> dict:
    """Index partitions and gaps before fitting features or normalization."""
    from src.ingestion.store import iter_batches, load_dataset
    init_db(db_path)
    dataset = _dataset if _dataset is not None else load_dataset(dataset_id, db_path=db_path)
    preparation_id = uuid.uuid4().hex
    directory = _root(storage_dir) / "preparations" / preparation_id
    directory.mkdir(parents=True, exist_ok=False)
    with get_connection(db_path) as connection:
        connection.execute("INSERT INTO preparations(preparation_id,dataset_id,artifact_dir,metadata_json,status,created_at) VALUES(?,?,?,?,?,?)",
                           (preparation_id, dataset_id, str(directory), "{}", "running", utc_now()))
    try:
        index, summary = build_row_index(iter_batches(dataset_id, db_path=db_path, _dataset=dataset), int(dataset["row_count"]),
                                         directory / "rows.npy", dataset.get("manifest") or {}, columns=dataset["columns"])
        del index
        summary.update(preparation_id=preparation_id, dataset_id=dataset_id, artifact_dir=str(directory),
                       pipeline_version=2, rows_path=str(directory / "rows.npy"),
                       structural_only=True, manifest_fingerprint=fingerprint(dataset.get("manifest") or {}))
        _write_json(directory / "preparation.json", summary)
        with get_connection(db_path) as connection:
            connection.execute("UPDATE preparations SET status='completed',metadata_json=? WHERE preparation_id=?",
                               (json.dumps(summary), preparation_id))
        return summary
    except Exception:
        with get_connection(db_path) as connection:
            connection.execute("UPDATE preparations SET status='failed' WHERE preparation_id=?", (preparation_id,))
        raise


def _exclude_training_annotations(dataset: dict, rows: np.ndarray) -> int:
    """Apply an optional annotation filter to training rows only."""
    import pyarrow.dataset as pads
    enabled = (dataset.get("manifest", {}).get("preprocessing") or {}).get("exclude_training_anomalies", False)
    if not enabled:
        return 0
    path = dataset.get("point_annotations_path") or dataset.get("annotations_path")
    if not path:
        raise ValueError("exclude_training_anomalies requires annotations")
    annotations = pads.dataset(path, format="parquet")
    schema_names = set(annotations.schema.names)
    processed_points = {"timestamp", "channel", "label_value", "source_partition"}
    if processed_points <= schema_names and "row_id" not in schema_names:
        from src.modeling.transforms import _timestamp_ns
        selected = {column.get("source_name", column["name"]) for column in dataset["columns"]
                    if column["role"] in {"numeric", "categorical"}}
        entity_spans = []
        begin = 0
        while begin < len(rows):
            stop = search_index(rows, "entity", rows[begin]["entity"], start=begin, side="right")
            entity_spans.append((begin, stop))
            begin = stop
        count = 0
        for batch in annotations.to_batches(columns=sorted(processed_points), batch_size=10000):
            for point in batch.to_pylist():
                if point["source_partition"] != "train" or point["channel"] not in selected:
                    continue
                label = point["label_value"]
                if label is None:
                    continue
                try:
                    numeric_label = float(label)
                    if not math.isfinite(numeric_label):
                        raise ValueError("Nonfinite anomaly label")
                except (TypeError, ValueError) as error:
                    raise ValueError("Training exclusion requires numeric anomaly labels") from error
                if numeric_label <= 0:
                    continue
                timestamp = _timestamp_ns(point["timestamp"])
                matches = []
                for entity_start, entity_stop in entity_spans:
                    position = search_index(rows, "timestamp_ns", timestamp, start=entity_start,
                                            end=entity_stop, side="left")
                    if position < entity_stop and rows[position]["timestamp_ns"] == timestamp:
                        matches.append(position)
                if len(matches) != 1:
                    raise ValueError("Processed ESA annotation timestamp does not uniquely match canonical telemetry")
                position = matches[0]
                if rows[position]["partition"] == 0 and not rows[position]["excluded"]:
                    rows["excluded"][position] = True
                    count += 1
        rows.flush()
        return count
    if not {"row_id", "label_value"} <= schema_names:
        required = {"Channel", "StartTime", "EndTime", "Category"}
        if not required <= schema_names:
            raise ValueError("Training exclusion requires point labels or ESA event intervals")
        from src.modeling.transforms import _timestamp_ns
        selected = {column.get("source_name", column["name"]) for column in dataset["columns"]
                    if column["role"] in {"numeric", "categorical"}}
        count = 0
        # Read small annotation batches. Segmenting telemetry by entity keeps
        # interval lookup logarithmic in row count, without loading timestamps.
        for batch in annotations.to_batches(columns=sorted(required), batch_size=1000):
            for event in batch.to_pylist():
                if event["Category"] != "Anomaly" or event["Channel"] not in selected:
                    continue
                start, end = _timestamp_ns(event["StartTime"]), _timestamp_ns(event["EndTime"])
                if end < start:
                    raise ValueError("ESA annotation ends before it starts")
                begin = 0
                while begin < len(rows):
                    entity = rows[begin]["entity"]
                    stop = search_index(rows, "entity", entity, start=begin, side="right")
                    left = search_index(rows, "timestamp_ns", start, start=begin, end=stop, side="left")
                    right = search_index(rows, "timestamp_ns", end, start=begin, end=stop, side="right")
                    for chunk_start in range(left, right, 10000):
                        chunk = rows[chunk_start:min(right, chunk_start + 10000)]
                        mask = (chunk["partition"] == 0) & ~chunk["excluded"]
                        count += int(mask.sum())
                        chunk["excluded"][mask] = True
                    begin = stop
        rows.flush()
        return count
    count = 0
    for batch in annotations.to_batches(columns=["row_id", "label_value"], batch_size=10000):
        data = batch.to_pydict()
        for row_id, label in zip(data["row_id"], data["label_value"]):
            if label is None:
                continue
            try:
                numeric_label = float(label)
                if not math.isfinite(numeric_label):
                    raise ValueError("Nonfinite anomaly label")
                anomalous = numeric_label > 0
            except (TypeError, ValueError) as error:
                raise ValueError("Training exclusion requires numeric anomaly labels") from error
            position = int(row_id) - 1
            if position < 0 or position >= len(rows) or rows[position]["row_id"] != row_id:
                raise ValueError("Annotation row_id does not match canonical telemetry")
            if anomalous and rows[position]["partition"] == 0 and not rows[position]["excluded"]:
                rows["excluded"][position] = True
                count += 1
    rows.flush()
    return count


def train(dataset_id: int, config_path=None, *, db_path=None, storage_dir=None, models_dir=None) -> dict:
    """Fit preprocessing on training rows and calibrate on validation windows."""
    import torch
    from torch.utils.data import DataLoader
    from src.ingestion.store import iter_batches, load_dataset
    from src.modeling.lstm_autoencoder import LSTMAutoencoder, reconstruction_loss
    from src.modeling.scoring import calibrate
    init_db(db_path)
    config = _load_config(config_path)
    if config_path is None:
        from src.modeling.train_model import resolve_training_config_path
        config_path = resolve_training_config_path()
    dataset = load_dataset(dataset_id, db_path=db_path)
    training_provenance = dataset_provenance(
        dataset, iter_batches(dataset_id, db_path=db_path) if dataset.get("storage_version") != 2 else None)
    preparation = prepare(dataset_id, db_path=db_path, storage_dir=storage_dir)
    directory = (Path(models_dir) if models_dir is not None else _root(storage_dir) / "models") / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    rows = np.load(preparation["rows_path"], mmap_mode="r+")
    # Keep immutable structural preparations untouched by training-only masks.
    training_rows = np.lib.format.open_memmap(directory / "rows.npy", mode="w+", dtype=rows.dtype, shape=rows.shape)
    for begin in range(0, len(rows), 10000):
        training_rows[begin:begin + 10000] = rows[begin:begin + 10000]
    del rows
    rows = training_rows
    excluded = _exclude_training_annotations(dataset, rows)
    forward_fill = (dataset.get("manifest", {}).get("preprocessing") or {}).get("forward_fill", True)
    state = fit_transform(iter_batches(dataset_id, db_path=db_path), rows, dataset["columns"],
                          max_categories=config["max_categories"], gap_ns=preparation["max_gap_ns"] if forward_fill else 0,
                          numeric_transform=config["numeric_transform"], feature_eligibility=config["feature_eligibility"])
    state["training_dataset_id"] = dataset_id
    state["training_manifest_fingerprint"] = preparation["manifest_fingerprint"]
    state["fingerprint"] = fingerprint({key: value for key, value in state.items() if key != "fingerprint"})
    _write_json(directory / "transform.json", state)
    values = transform_to_memmap(iter_batches(dataset_id, db_path=db_path), rows, state, directory / "values.npy", dataset["columns"])
    if training_provenance["kind"] == "legacy_content" and dataset_provenance(
            dataset, iter_batches(dataset_id, db_path=db_path)) != training_provenance:
        raise ValueError("Legacy telemetry changed during training. Retry with a stable source or reimport it")
    train_starts = build_window_index(rows, directory / "train_windows.npy", window_size=config["window_size"],
                                     stride=config["stride"], partition="train")
    validation_starts = build_window_index(rows, directory / "validation_windows.npy", window_size=config["window_size"],
                                          stride=1, partition="validation")
    if not len(train_starts) or not len(validation_starts):
        raise ValueError("Training and validation each need complete windows. Reduce window_size or revise manifest splits")
    offsets = [f["offset"] for f in state["features"] if f["role"] == "numeric"] if config["window_centering"] else None
    training_windows = WindowDataset(values, train_starts, config["window_size"], numeric_offsets=offsets)
    validation_windows = WindowDataset(values, validation_starts, config["window_size"], numeric_offsets=offsets)
    torch.manual_seed(config["seed"])
    model = LSTMAutoencoder(len(state["output_names"]), config["hidden_size"], config["num_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    epoch_losses = []
    for _ in range(config["epochs"]):
        model.train()
        total_loss, sample_count = 0.0, 0
        # Sequential order avoids RandomSampler's O(number_of_windows) permutation.
        for batch in DataLoader(training_windows, batch_size=config["batch_size"], shuffle=False, num_workers=0):
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a nonfinite loss. Inspect input scaling")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            sample_count += len(batch)
        epoch_losses.append(total_loss / sample_count)
    threshold, calibration = calibrate(model, validation_windows, directory / "calibration.npy",
                                       percentile=config["threshold_percentile"], batch_size=config["batch_size"])
    checkpoint = {"pipeline_version": 2, "model_state_dict": model.state_dict(), "dataset_id": dataset_id,
                  "config": config, "num_features": len(state["output_names"]), "feature_columns": state["output_names"],
                  "transform_fingerprint": state["fingerprint"], "threshold": threshold,
                  "training_data_provenance": training_provenance}
    torch.save(checkpoint, directory / "model.pt")
    summary = {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1],
               "window_centering": config["window_centering"],
               "feature_eligibility": state["feature_eligibility"], "excluded_features": state["excluded_features"],
               "training_window_count": len(training_windows), "validation_window_count": len(validation_windows),
               "excluded_training_rows": excluded, "calibration": calibration, "partitions": preparation,
               "threshold_method": "validation_reconstruction_error_percentile", "seed": config["seed"]}
    metadata = {"pipeline_version": 2, "artifact_dir": str(directory), "transform_path": str(directory / "transform.json"),
                "transform_fingerprint": state["fingerprint"], "model_fingerprint": _hash_file(directory / "model.pt"),
                "feature_columns": state["output_names"], "feature_contract": state["features"], "config": config,
                "training_rows_path": str(directory / "rows.npy"), "preparation_id": preparation["preparation_id"],
                "training_data_provenance": training_provenance}
    _write_json(directory / "training.json", {"metadata": metadata, "summary": summary})
    model_id = insert_model_metadata(dataset_id, directory / "model.pt", config_path=config_path, threshold=threshold,
                                     threshold_method=summary["threshold_method"], threshold_percentile=config["threshold_percentile"],
                                     training_summary=summary, metadata=metadata, db_path=db_path)
    return {"model_id": model_id, "dataset_id": dataset_id, "model_path": str(directory / "model.pt"),
            "config_path": str(config_path),
            "excluded_features": state["excluded_features"],
            "threshold": threshold, "training_window_count": len(training_windows), "feature_count": len(state["output_names"]),
            "validation_window_count": len(validation_windows), "partitions": preparation["partition_bounds"],
            "transform_fingerprint": state["fingerprint"]}


def get_run(run_id: str, *, db_path=None) -> dict:
    init_db(db_path)
    with get_connection(db_path) as connection:
        row = connection.execute("SELECT * FROM detection_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"Detection run does not exist: {run_id}")
    run = dict(row)
    if run["status"] != "completed":
        raise ValueError(f"Detection run is not completed: {run_id}")
    return {**json.loads(run["metadata_json"]), **run}


def _contributions(state: dict, errors: np.ndarray, endpoint: np.ndarray, *, limit: int = 5) -> list[dict]:
    contributions = []
    for feature in state["features"]:
        begin, width = feature["offset"], feature["width"]
        entry = {"feature": feature["source_name"], "score": float(errors[begin:begin + width].sum()),
                 "unit": feature.get("unit")}
        if feature["role"] == "numeric":
            value = float(endpoint[begin])
            if state.get("numeric_transform", "standard") == "asinh":
                value = math.sinh(value)
            entry["value"] = None if endpoint[begin + 1] else value * feature["scale"] + feature["mean"]
            entry["missing"] = bool(endpoint[begin + 1])
        contributions.append(entry)
    return [dict(entry, rank=rank) for rank, entry in enumerate(sorted(contributions, key=lambda entry: entry["score"], reverse=True)[:limit], 1)]


def detect(dataset_id: int, model_id: int, *, db_path=None, storage_dir=None,
           partition: str = "all", batch_size: int | None = None) -> dict:
    """Score a compatible target, with bounded memory and complete row coverage."""
    import torch
    import pyarrow as pa
    from src.ingestion.store import _ParquetPartWriter, iter_batches, load_dataset
    from src.modeling.lstm_autoencoder import LSTMAutoencoder
    from src.modeling.scoring import score_batches
    init_db(db_path)
    dataset = load_dataset(dataset_id, db_path=db_path)
    model_row = get_model(model_id, db_path=db_path)
    if model_row is None:
        raise ValueError(f"Model does not exist: {model_id}")
    metadata = json.loads(model_row["metadata_json"])
    if metadata.get("pipeline_version") != 2:
        raise ValueError("Legacy models only support their original dataset. Retrain for reusable preprocessing")
    state = json.loads(Path(metadata["transform_path"]).read_text(encoding="utf-8"))
    if fingerprint({key: value for key, value in state.items() if key != "fingerprint"}) != state["fingerprint"]:
        raise ValueError("Model preprocessing fingerprint mismatch")
    resolve_features(state, dataset["columns"])
    if _hash_file(Path(model_row["model_path"])) != metadata["model_fingerprint"]:
        raise ValueError("Model checkpoint fingerprint mismatch")
    checkpoint = torch.load(model_row["model_path"], weights_only=True, map_location="cpu")
    if checkpoint["transform_fingerprint"] != state["fingerprint"]:
        raise ValueError("Checkpoint preprocessing fingerprint mismatch")
    config = checkpoint["config"]
    batch_size = int(batch_size) if batch_size is not None else int(config["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_provenance = dataset_provenance(
        dataset, iter_batches(dataset_id, db_path=db_path) if dataset.get("storage_version") != 2 else None)
    if int(model_row["dataset_id"]) == dataset_id and target_provenance["kind"] == "legacy_content":
        fitted_provenance = checkpoint.get("training_data_provenance") or {}
        if fitted_provenance.get("kind") != "legacy_content" or not fitted_provenance.get("dataset_fingerprint"):
            raise ValueError("Legacy-derived checkpoint lacks fitted content provenance. Retrain before scoring this dataset")
        if fitted_provenance["dataset_fingerprint"] != target_provenance["dataset_fingerprint"]:
            raise ValueError("Legacy telemetry changed since training. Reimport or retrain before scoring this dataset")
    run_id = uuid.uuid4().hex
    directory = _root(storage_dir) / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    run = {"run_id": run_id, "dataset_id": dataset_id, "model_id": model_id, "pipeline_version": 2,
           "ordering": dataset.get("manifest", {}).get("ordering", "timestamp"),
           "model_fingerprint": metadata["model_fingerprint"], "transform_fingerprint": state["fingerprint"],
           "target_manifest_fingerprint": fingerprint(dataset.get("manifest") or {}),
           "target_import_fingerprint": target_provenance["import_fingerprint"],
           "target_source_sha256": target_provenance["source_sha256"],
           "target_data_provenance": target_provenance, "partition": partition,
           "batch_size": batch_size, "threshold": float(model_row["threshold"]),
           "window_centering": bool(config.get("window_centering", False)),
           "scores_path": str(directory / "scores.parquet"), "scores_format": "parquet-parts",
           "artifact_dir": str(directory),
           "score_timestamp": "window_end", "scoring_stride": 1}
    with get_connection(db_path) as connection:
        connection.execute("INSERT INTO detection_runs(run_id,dataset_id,model_id,status,artifact_dir,metadata_json,created_at) VALUES(?,?,?,?,?,?,?)",
                           (run_id, dataset_id, model_id, "running", str(directory), json.dumps(run), utc_now()))
    try:
        target_manifest = dataset.get("manifest") or {}
        explicit_target_splits = target_manifest.get("_splits_explicit", any(
            key in target_manifest.get("splits", {}) for key in ("train_end", "validation_end", "test_start")
        ))
        rows, _ = build_row_index(iter_batches(dataset_id, db_path=db_path), int(dataset["row_count"]),
                                   directory / "rows.npy", target_manifest,
                                   inference=(int(model_row["dataset_id"]) != dataset_id and partition == "all"
                                              and not explicit_target_splits),
                                   columns=dataset["columns"])
        # Same-dataset splits and training exclusions are frozen at fit time.
        if int(model_row["dataset_id"]) == dataset_id:
            fitted_rows = np.load(metadata["training_rows_path"], mmap_mode="r")
            if len(fitted_rows) != len(rows):
                raise ValueError("Training dataset row count changed")
            for begin in range(0, len(rows), 10000):
                rows[begin:begin + 10000] = fitted_rows[begin:begin + 10000]
        values = transform_to_memmap(iter_batches(dataset_id, db_path=db_path), rows, state, directory / "values.npy", dataset["columns"])
        if target_provenance["kind"] == "legacy_content" and dataset_provenance(
                dataset, iter_batches(dataset_id, db_path=db_path)) != target_provenance:
            raise ValueError("Legacy telemetry changed during scoring preparation. Retry with a stable source")
        starts = build_window_index(rows, directory / "windows.npy", window_size=config["window_size"], stride=1, partition=partition)
        offsets = [f["offset"] for f in state["features"] if f["role"] == "numeric"] if config.get("window_centering", False) else None
        windows = WindowDataset(values, starts, config["window_size"], numeric_offsets=offsets)
        model = LSTMAutoencoder(len(state["output_names"]), config["hidden_size"], config["num_layers"])
        model.load_state_dict(checkpoint["model_state_dict"])
        score_map = np.lib.format.open_memmap(directory / "scores.npy", mode="w+", dtype="float64", shape=(len(rows),))
        # Use NaN as the internal marker and null in exported Parquet.
        score_map[:] = np.nan
        anomaly_count = 0
        anomaly_log = directory / "anomalies.jsonl"
        # Finish all catalog reads before publishing alerts. A large rollback
        # journal can acquire an exclusive lock while SQLite spills its cache.
        with anomaly_log.open("w", encoding="utf-8") as stream:
            for score_offset, scores, feature_errors in score_batches(model, windows, batch_size):
                for local, score in enumerate(scores):
                    window_index = score_offset + local
                    start = int(starts[window_index])
                    end = start + config["window_size"] - 1
                    score_map[end] = float(score)
                    if float(score) <= run["threshold"]:
                        continue
                    timestamp = None if dataset["manifest"].get("ordering") == "row" else _iso_timestamp(int(rows[end]["timestamp_ns"]))
                    input_start = None if dataset["manifest"].get("ordering") == "row" else _iso_timestamp(int(rows[start]["timestamp_ns"]))
                    top = _contributions(state, feature_errors[local], values[end])
                    anomaly_id = f"anomaly_{run_id}_{window_index}"
                    packet = {"anomaly_id": anomaly_id, "run_id": run_id, "dataset_id": dataset_id, "model_id": model_id,
                              "window_index": window_index, "start_row_id": int(rows[start]["row_id"]), "end_row_id": int(rows[end]["row_id"]),
                              "start_timestamp": timestamp, "end_timestamp": timestamp, "score_timestamp": timestamp,
                              "input_window_start_timestamp": input_start, "input_window_end_timestamp": timestamp,
                              "anomaly_score": float(score), "threshold": run["threshold"], "top_contributing_features": top,
                              "contribution_method": "sum_of_encoded_feature_reconstruction_errors",
                              "transform_fingerprint": state["fingerprint"]}
                    packet["window_centering"] = run["window_centering"]
                    record = (anomaly_id, dataset_id, model_id, window_index, packet["start_row_id"], packet["end_row_id"],
                              timestamp, timestamp, float(score), run["threshold"], json.dumps(top), json.dumps(packet), utc_now(), run_id, timestamp)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    anomaly_count += 1
        score_map.flush()
        schema = pa.schema([("row_id", pa.int64()), ("timestamp", pa.string()), ("timestamp_ns", pa.int64()),
                            ("entity_id", pa.string()), ("partition", pa.string()), ("score", pa.float64()),
                            ("is_anomaly", pa.bool_()), ("coverage", pa.string())])
        offset, coverage_counts = 0, {}
        partition_names = list(PARTITIONS)
        with _ParquetPartWriter(Path(run["scores_path"]), schema) as writer:
            for batch in iter_batches(dataset_id, db_path=db_path):
                data = batch.to_pydict()
                output = {name: [] for name in schema.names}
                for local in range(batch.num_rows):
                    index = offset + local
                    score = float(score_map[index])
                    part = partition_names[int(rows[index]["partition"])]
                    coverage = "scored" if math.isfinite(score) else ("gap" if rows[index]["gap"] else (
                        "excluded" if rows[index]["excluded"] else (
                        "outside_partition" if partition != "all" and part != partition else "warmup_or_stride")))
                    record = (int(data["row_id"][local]), data["timestamp"][local], int(data["timestamp_ns"][local]),
                              str(data["entity_id"][local]), part, score if math.isfinite(score) else None,
                              bool(math.isfinite(score) and score > run["threshold"]), coverage)
                    for name, value in zip(schema.names, record):
                        output[name].append(value)
                    coverage_counts[coverage] = coverage_counts.get(coverage, 0) + 1
                writer.write_batch(pa.RecordBatch.from_pydict(output, schema=schema))
                offset += batch.num_rows
        if target_provenance["kind"] == "legacy_content" and dataset_provenance(
                dataset, iter_batches(dataset_id, db_path=db_path)) != target_provenance:
            raise ValueError("Legacy telemetry changed during scoring. Retry with a stable source")
        run.update(windows_scored=len(windows), anomalies_detected=anomaly_count, row_count=len(rows), coverage=coverage_counts)
        _write_json(directory / "run.json", run)
        with get_connection(db_path) as connection:
            # Publish alerts and completion in one transaction. Opening another
            # catalog connection before commit can conflict with SQLite cache spill.
            connection.execute("BEGIN")
            from itertools import islice
            with anomaly_log.open("r", encoding="utf-8") as stream:
                while lines := list(islice(stream, 1000)):
                    connection.executemany("""INSERT INTO anomalies(anomaly_id,dataset_id,model_id,window_index,start_row_id,end_row_id,
                        start_timestamp,end_timestamp,anomaly_score,threshold,contributing_features_json,packet_json,created_at,run_id,score_timestamp)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", [json.loads(line) for line in lines])
            connection.execute("UPDATE detection_runs SET status='completed',metadata_json=?,completed_at=? WHERE run_id=?",
                               (json.dumps(run), utc_now(), run_id))
        return run
    except Exception:
        with get_connection(db_path) as connection:
            connection.execute("UPDATE detection_runs SET status='failed',completed_at=? WHERE run_id=?", (utc_now(), run_id))
        raise


def _iso_timestamp(timestamp_ns: int) -> str:
    import pandas as pd
    return pd.Timestamp(timestamp_ns, tz="UTC").isoformat()


def diagnostics(run_id: str, *, db_path=None, output_dir=None) -> dict:
    """Summarize a completed run using its recorded scores and threshold."""
    from src.ingestion.store import _parquet_batches
    from src.common.score_artifact import validate_score_counts
    from collections import Counter
    run = get_run(run_id, db_path=db_path)
    count, total, minimum, maximum = 0, 0.0, float("inf"), -float("inf")
    source_rows = scored_rows = 0
    coverage_counts = Counter()
    for batch in _parquet_batches(run["scores_path"], columns=["score", "coverage"], batch_size=10000):
        source_rows += batch.num_rows
        coverage = batch.column("coverage").to_pylist()
        coverage_counts.update(coverage)
        scored_rows += coverage.count("scored")
        array = batch.column(0).drop_null().to_numpy()
        if len(array) != coverage.count("scored") or not np.isfinite(array).all():
            raise ValueError("Inconsistent score artifact: scored rows require finite reconstruction scores")
        if len(array):
            count += len(array)
            total += float(array.sum(dtype=np.float64))
            minimum, maximum = min(minimum, float(array.min())), max(maximum, float(array.max()))
    validate_score_counts(run, source_rows, scored_rows, coverage_counts)
    result = {"run_id": run_id, "dataset_id": run["dataset_id"], "model_id": run["model_id"],
              "score_count": count, "score_mean": total / count if count else None,
              "score_min": minimum if count else None, "score_max": maximum if count else None,
              "threshold": run["threshold"], "anomalies_detected": run["anomalies_detected"],
              "coverage": run["coverage"], "score_timestamp": "window_end", "scores_path": run["scores_path"]}
    from src.modeling.run_diagnostics import enrich_diagnostics
    result = enrich_diagnostics(result, run, db_path=db_path, output_dir=output_dir)
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"diagnostics_{run_id}.json"
        result["summary_path"] = str(path)
        _write_json(path, result)
        result["output_path"] = str(path)
    return result
