"""Legacy content snapshots and spill-safe atomic detection publication."""
import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from src.db import get_connection, get_model, init_db, insert_dataset, utc_now
from src.ingestion.store import ingest_source, iter_batches, load_dataset
from src.modeling import pipeline
from src.modeling.provenance import dataset_provenance


def _config(root):
    path = root / "config.json"
    path.write_text(json.dumps({"window_size": 4, "hidden_size": 4, "epochs": 1,
                                "batch_size": 7, "seed": 0}), encoding="utf-8")
    return path


def _legacy(root, db, *, count=80):
    init_db(db)
    dataset_id = insert_dataset("legacy", status="ingested", row_count=count, db_path=db)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with get_connection(db) as connection:
        connection.execute(f"CREATE TABLE raw_dataset_{dataset_id}(row_id INTEGER PRIMARY KEY,timestamp TEXT,x TEXT,is_anomaly TEXT,secret TEXT)")
        connection.executemany(f"INSERT INTO raw_dataset_{dataset_id} VALUES(?,?,?,?,?)",
                               [(index + 1, (start + timedelta(seconds=index)).isoformat(), str(index % 9), "0", "hidden")
                                for index in range(count)])
        roles = [("timestamp", "timestamp", "timestamp"), ("x", "numeric", "numeric_telemetry_feature"),
                 ("is_anomaly", "numeric", "numeric_telemetry_feature"), ("secret", "text", "annotation")]
        connection.executemany("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,inferred_type,role,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                               [(dataset_id, name, name, dtype, role, utc_now(), utc_now()) for name, dtype, role in roles])
    return dataset_id


@pytest.fixture
def trained_legacy(tmp_path):
    torch.set_num_threads(1)
    db = tmp_path / "catalog.db"
    dataset_id = _legacy(tmp_path, db)
    model = pipeline.train(dataset_id, _config(tmp_path), db_path=db, storage_dir=tmp_path)
    return tmp_path, db, dataset_id, model


def test_legacy_content_provenance_is_batched_and_excludes_labels(trained_legacy):
    root, db, dataset_id, model = trained_legacy
    dataset = load_dataset(dataset_id, db_path=db)
    small = dataset_provenance(dataset, iter_batches(dataset_id, db_path=db, batch_size=1))
    large = dataset_provenance(dataset, iter_batches(dataset_id, db_path=db, batch_size=31))
    assert small == large
    assert small["kind"] == "legacy_content"
    assert small["source_sha256"] is small["import_fingerprint"] is None
    with get_connection(db) as connection:
        connection.execute(f"UPDATE raw_dataset_{dataset_id} SET is_anomaly='1',secret='changed annotation'")
    assert dataset_provenance(dataset, iter_batches(dataset_id, db_path=db)) == small
    run = pipeline.detect(dataset_id, model["model_id"], db_path=db, storage_dir=root, partition="test")
    assert run["target_data_provenance"] == small
    assert run["target_source_sha256"] is run["target_import_fingerprint"] is None
    assert pq.read_table(run["scores_path"]).num_rows == 80
    checkpoint = torch.load(model["model_path"], weights_only=True)
    assert checkpoint["training_data_provenance"] == small


@pytest.mark.parametrize("mutation", ["timestamps", "values"])
def test_changed_same_id_legacy_content_rejected_before_run_creation(trained_legacy, mutation):
    root, db, dataset_id, model = trained_legacy
    with get_connection(db) as connection:
        before = connection.execute("SELECT count(*) FROM detection_runs").fetchone()[0]
        if mutation == "timestamps":
            connection.execute(f"UPDATE raw_dataset_{dataset_id} SET timestamp=replace(timestamp,'2024-01-01','2024-01-02')")
        else:
            connection.execute(f"UPDATE raw_dataset_{dataset_id} SET x='1234' WHERE row_id=1")
    with pytest.raises(ValueError, match="changed since training"):
        pipeline.detect(dataset_id, model["model_id"], db_path=db, storage_dir=root)
    with get_connection(db) as connection:
        assert connection.execute("SELECT count(*) FROM detection_runs").fetchone()[0] == before
        assert connection.execute("SELECT count(*) FROM anomalies").fetchone()[0] == 0


def test_pre_fix_legacy_derived_checkpoint_requires_retraining(trained_legacy):
    root, db, dataset_id, model = trained_legacy
    checkpoint = torch.load(model["model_path"], weights_only=True)
    checkpoint.pop("training_data_provenance")
    torch.save(checkpoint, model["model_path"])
    row = get_model(model["model_id"], db_path=db)
    metadata = json.loads(row["metadata_json"])
    metadata["model_fingerprint"] = pipeline._hash_file(Path(model["model_path"]))
    with get_connection(db) as connection:
        connection.execute("UPDATE models SET metadata_json=? WHERE model_id=?", (json.dumps(metadata), model["model_id"]))
    with pytest.raises(ValueError, match="lacks fitted content provenance. Retrain"):
        pipeline.detect(dataset_id, model["model_id"], db_path=db, storage_dir=root)


def test_independent_legacy_target_records_its_actual_content(trained_legacy):
    root, db, training_id, model = trained_legacy
    target_id = _legacy(root, db)
    with get_connection(db) as connection:
        connection.execute(f"UPDATE raw_dataset_{target_id} SET x='1000' WHERE row_id>40")
    run = pipeline.detect(target_id, model["model_id"], db_path=db, storage_dir=root)
    trained_provenance = json.loads(get_model(model["model_id"], db_path=db)["metadata_json"])["training_data_provenance"]
    assert run["dataset_id"] == target_id != training_id
    assert run["target_data_provenance"]["kind"] == "legacy_content"
    assert run["target_data_provenance"]["content_sha256"] != trained_provenance["content_sha256"]
    assert run["windows_scored"] == 77
    with get_connection(db) as connection:
        assert all(row[0] == target_id for row in connection.execute("SELECT dataset_id FROM anomalies WHERE run_id=?", (run["run_id"],)))


def test_future_legacy_content_changes_provenance_without_changing_fitted_transform(trained_legacy):
    root, db, dataset_id, first = trained_legacy
    first_metadata = json.loads(get_model(first["model_id"], db_path=db)["metadata_json"])
    with get_connection(db) as connection:
        connection.execute(f"UPDATE raw_dataset_{dataset_id} SET x='1000' WHERE row_id>56")
    second = pipeline.train(dataset_id, root / "config.json", db_path=db, storage_dir=root)
    second_metadata = json.loads(get_model(second["model_id"], db_path=db)["metadata_json"])
    assert first["transform_fingerprint"] == second["transform_fingerprint"]
    assert first_metadata["training_data_provenance"] != second_metadata["training_data_provenance"]


@pytest.fixture
def spill_target(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    db = tmp_path / "catalog.db"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    datasets = []
    for name, count in (("train", 80), ("target", 2048)):
        source = tmp_path / f"{name}.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "x"])
            writer.writerows([[(start + timedelta(seconds=index)).isoformat(), index % 9] for index in range(count)])
        datasets.append(ingest_source(source, db_path=db, storage_dir=tmp_path))
    model = pipeline.train(datasets[0]["dataset_id"], _config(tmp_path), db_path=db, storage_dir=tmp_path)
    with get_connection(db) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        connection.execute("UPDATE models SET threshold=-1 WHERE model_id=?", (model["model_id"],))

    original_connection = pipeline.get_connection
    def tiny_cache(*args, **kwargs):
        connection = original_connection(*args, **kwargs)
        connection.execute("PRAGMA cache_size=1")
        connection.execute("PRAGMA cache_spill=ON")
        return connection
    monkeypatch.setattr(pipeline, "get_connection", tiny_cache)
    from src.modeling import scoring
    def every_window_alerts(_model, windows, batch_size):
        for offset in range(0, len(windows), batch_size):
            count = min(batch_size, len(windows) - offset)
            yield offset, np.ones(count), np.ones((count, windows.values.shape[1]))
    monkeypatch.setattr(scoring, "score_batches", every_window_alerts)
    return tmp_path, db, datasets[1]["dataset_id"], model["model_id"]


def test_spilled_alert_publication_has_no_nested_catalog_reads(spill_target):
    root, db, target_id, model_id = spill_target
    run = pipeline.detect(target_id, model_id, db_path=db, storage_dir=root, batch_size=19)
    assert run["anomalies_detected"] == run["windows_scored"] == 2045
    assert pq.read_table(run["scores_path"]).num_rows == 2048
    with get_connection(db) as connection:
        assert connection.execute("SELECT status FROM detection_runs WHERE run_id=?", (run["run_id"],)).fetchone()[0] == "completed"
        assert connection.execute("SELECT count(*) FROM anomalies WHERE run_id=?", (run["run_id"],)).fetchone()[0] == 2045


def test_spilled_publication_failure_rolls_back_all_alerts(spill_target):
    root, db, target_id, model_id = spill_target
    with get_connection(db) as connection:
        connection.execute("""CREATE TRIGGER reject_late_alert BEFORE INSERT ON anomalies
            WHEN NEW.window_index>=1500 BEGIN SELECT RAISE(ABORT,'simulated publication failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="publication failure"):
        pipeline.detect(target_id, model_id, db_path=db, storage_dir=root, batch_size=19)
    with get_connection(db) as connection:
        assert connection.execute("SELECT status FROM detection_runs").fetchone()[0] == "failed"
        assert connection.execute("SELECT count(*) FROM anomalies").fetchone()[0] == 0
