import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.db import get_model
from src.ingestion.store import ingest_source, load_dataset, read_window
from src.modeling.monitoring import monitor


def test_updates_use_only_recent_past_and_retry_completed_periods(tmp_path):
    torch.set_num_threads(1)
    db = tmp_path / "catalog.db"
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"window_size": 4, "hidden_size": 4, "batch_size": 16, "epochs": 1}))
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    results = []
    for changed in (False, True):
        source = tmp_path / f"source-{changed}.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "signal", "is_anomaly_signal"])
            for hour in range(24 * 10):
                value = hour % 24 + (10000 if changed and hour >= 24 * 7 else 0)
                writer.writerow([(origin + timedelta(hours=hour)).isoformat(), value, int(changed)])
        dataset = ingest_source(source, db_path=db, storage_dir=tmp_path)
        result = monitor(dataset["dataset_id"], "2024-01-07", "2024-01-09", history_days=4,
                         validation_days=1, update_days=1, config_path=config, db_path=db, storage_dir=tmp_path)
        assert result["rows"] == 48 and result["not_esa_benchmark"]
        assert len(result["periods"]) == 2
        for period in result["periods"]:
            model = get_model(period["model_id"], db_path=db)
            metadata = json.loads(model["metadata_json"])
            rows = np.load(metadata["training_rows_path"], mmap_mode="r")
            cutoff = datetime.fromisoformat(period["cutoff"]).timestamp() * 1e9
            assert rows["timestamp_ns"].max() < cutoff
            history = load_dataset(period["training_dataset_id"], db_path=db)
            assert all(c["role"] != "label" for c in history["columns"])
            raw = read_window(period["target_dataset_id"], 1, 24, db_path=db)
            assert len(raw) == 24 and all("is_anomaly_signal" not in row for row in raw)
        with Path(result["predictions_path"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 48
        assert len({row["timestamp"] for row in rows}) == 48
        assert all(row["partition"] == "test" for row in rows)
        results.append((result, rows))
        repeated = monitor(dataset["dataset_id"], "2024-01-07", "2024-01-09", history_days=4,
                           validation_days=1, update_days=1, config_path=config, db_path=db, storage_dir=tmp_path)
        assert [p["run_id"] for p in repeated["periods"]] == [p["run_id"] for p in result["periods"]]
        period_file = Path(result["periods"][0]["export"]["path"])
        original_export = period_file.read_bytes()
        combined = Path(result["predictions_path"]).read_bytes()
        period_file.write_bytes(original_export.splitlines(keepends=True)[0])
        with pytest.raises(ValueError, match="export changed"):
            monitor(dataset["dataset_id"], "2024-01-07", "2024-01-09", history_days=4,
                    validation_days=1, update_days=1, config_path=config, db_path=db, storage_dir=tmp_path)
        assert Path(result["predictions_path"]).read_bytes() == combined
        period_file.write_bytes(original_export)
    keys = ("timestamp", "score", "is_anomaly", "coverage", "row_id", "entity_id", "partition")
    assert [{k: row[k] for k in keys} for row in results[0][1][:24]] == [
        {k: row[k] for k in keys} for row in results[1][1][:24]]


def test_monitoring_rejects_invalid_intervals_before_training(tmp_path):
    with pytest.raises(ValueError, match="history_days"):
        monitor(1, "2024-01-01", "2024-01-02", history_days=1, validation_days=2,
                db_path=tmp_path / "db", storage_dir=tmp_path)


def test_snapshot_rejects_legacy_mutation_before_publication(tmp_path, monkeypatch):
    import pyarrow as pa
    from src.db import get_connection, init_db
    from src.modeling import monitoring

    db = tmp_path / "catalog.db"
    init_db(db)
    rows = [{"row_id": i + 1, "timestamp": str(i), "timestamp_ns": i * 10**9,
             "entity_id": "", "signal": float(i)} for i in range(3)]
    dataset = {"dataset_id": 99, "storage_version": 1, "row_count": 3, "manifest": {},
               "columns": [{"name": "signal", "source_name": "signal", "role": "numeric", "dtype": "float64"}]}

    def batches(*args, **kwargs):
        yield pa.RecordBatch.from_pylist([dict(row) for row in rows])

    original_range = monitoring._range_batches

    def mutated(*args, **kwargs):
        yield from original_range(*args, **kwargs)
        rows[0]["signal"] = 999.0

    monkeypatch.setattr(monitoring, "iter_batches", batches)
    monkeypatch.setattr(monitoring, "_range_batches", mutated)
    with pytest.raises(ValueError, match="Legacy telemetry changed"):
        monitoring._snapshot([(dataset, 0, 3 * 10**9)], {}, db_path=db, storage_dir=tmp_path, entity_id=None)
    with get_connection(db) as connection:
        assert connection.execute("SELECT count(*) FROM datasets").fetchone()[0] == 0


def test_monitoring_requires_one_entity_and_cli_dispatches_options(tmp_path, monkeypatch):
    from src.cli.__main__ import _parser, _dispatch
    from src.modeling import monitoring

    source = tmp_path / "devices.csv"
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "device", "signal"])
        for hour in range(24 * 8):
            for entity in ("a", "b"):
                writer.writerow([(origin + timedelta(hours=hour)).isoformat(), entity, hour % 12])
    db = tmp_path / "db"
    dataset = ingest_source(source, {"entity_columns": ["device"]}, db_path=db, storage_dir=tmp_path)
    with pytest.raises(ValueError, match="one entity"):
        monitor(dataset["dataset_id"], "2024-01-07", "2024-01-08", history_days=4,
                validation_days=1, db_path=db, storage_dir=tmp_path)
    captured = {}

    def selected(*args, **kwargs):
        captured.update(kwargs)
        return "dispatched"

    monkeypatch.setattr(monitoring, "monitor", selected)
    args = _parser().parse_args(["monitor", "--dataset-id", str(dataset["dataset_id"]),
                                "--start", "2024-01-07", "--end", "2024-01-08", "--entity-id", "a"])
    assert _dispatch(args) == "dispatched"
    assert captured["entity_id"] == "a" and captured["history_days"] == 90


def test_monitoring_rejects_legacy_change_since_job_identity(tmp_path, monkeypatch):
    from src.modeling import monitoring, provenance

    dataset = {"dataset_id": 1, "storage_version": 1, "manifest": {}}
    calls = []

    def changing_identity(*args, **kwargs):
        calls.append(1)
        return {"dataset_fingerprint": "before" if len(calls) <= 2 else "after"}

    monkeypatch.setattr(monitoring, "load_dataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(monitoring, "iter_batches", lambda *a, **kw: iter(()))
    monkeypatch.setattr(provenance, "dataset_provenance", changing_identity)
    with pytest.raises(ValueError, match="Legacy monitoring source changed"):
        monitor(1, "2024-01-07", "2024-01-08", db_path=tmp_path / "db", storage_dir=tmp_path)


def test_separate_batch_allows_reordered_features_extras_and_absent_context(tmp_path):
    torch.set_num_threads(1)
    db = tmp_path / "db"
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    history_file, target_file = tmp_path / "history.csv", tmp_path / "target.csv"
    with history_file.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "signal", "operator"])
        for hour in range(24 * 6):
            writer.writerow([(origin + timedelta(hours=hour)).isoformat(), hour % 12, "note"])
    with target_file.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["extra", "signal", "timestamp"])
        for hour in range(24 * 6, 24 * 8):
            writer.writerow([999, hour % 12, (origin + timedelta(hours=hour)).isoformat()])
    history = ingest_source(history_file, {"context_columns": ["operator"]}, db_path=db, storage_dir=tmp_path)
    target = ingest_source(target_file, db_path=db, storage_dir=tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"window_size": 4, "stride": 4, "hidden_size": 4, "epochs": 1}))
    result = monitor(target["dataset_id"], "2024-01-07", "2024-01-09", history_dataset_id=history["dataset_id"],
                     history_days=4, validation_days=1, update_days=1, config_path=config,
                     db_path=db, storage_dir=tmp_path)
    assert result["rows"] == 48
    for period in result["periods"]:
        model = get_model(period["model_id"], db_path=db)
        metadata = json.loads(model["metadata_json"])
        assert [f["source_name"] for f in metadata["feature_contract"]] == ["signal"]
