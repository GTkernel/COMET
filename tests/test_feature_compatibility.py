"""Original feature decisions remain authoritative in the bounded pipeline."""
import json
from pathlib import Path

import pytest
import yaml

from src.db import get_connection, init_db
from src.cleaning.prepare_dataset import prepare_dataset
from src.ingestion.store import ingest_source, iter_batches, load_dataset
from src.modeling.pipeline import _load_config, train
from src.rag.telemetry_tools import get_context_window, get_column_summary, get_rows_around_peak


def test_default_config_is_loaded_and_explicit_config_overrides(tmp_path, monkeypatch):
    from src.modeling import train_model
    default = tmp_path / "default.yaml"
    default.write_text("window_size: 7\nhidden_size: 5\n", encoding="utf-8")
    monkeypatch.setattr(train_model, "DEFAULT_TRAINING_CONFIG_PATH", default)
    assert _load_config()["window_size"] == 7
    assert train_model.load_training_config(default)["hidden_size"] == 5
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("window_size: 3\n", encoding="utf-8")
    assert _load_config(explicit)["window_size"] == 3
    assert _load_config(defaults={"window_size": 2})["window_size"] == 2
    assert train_model.load_training_config(tmp_path / "absent.yaml")["window_size"] == train_model.DEFAULT_TRAINING_CONFIG["window_size"]


@pytest.mark.parametrize("canonical", [False, True])
def test_prepared_roles_reach_training_context_and_summaries(tmp_path, canonical):
    pytest.importorskip("torch").set_num_threads(1)
    db = tmp_path / "catalog.db"
    init_db(db)
    if canonical:
        source = tmp_path / "telemetry.csv"
        source.write_text("event_time,signal,mode,drop_me\n" + "".join(
            f"2020-01-01T00:00:{i:02d}Z,{i},{i % 2},{i * 9}\n" for i in range(40)), encoding="utf-8")
        imported = ingest_source(source, {"timestamp": "event_time", "numeric_columns": ["signal", "mode", "drop_me"]},
                          db_path=db, storage_dir=tmp_path)
        dataset_id = imported["dataset_id"]
        timestamp = "timestamp"
    else:
        dataset_id = 1
        timestamp = "event_time"
        with get_connection(db) as conn:
            conn.execute("INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) VALUES(1,'legacy','ingested',40,'{}','2020','2020')")
            conn.execute("CREATE TABLE raw_dataset_1(row_id INTEGER PRIMARY KEY,dataset_id INTEGER,event_time TEXT,signal TEXT,mode TEXT,drop_me TEXT)")
            conn.executemany("INSERT INTO raw_dataset_1 VALUES(?,?,?,?,?,?)", [
                (i + 1, 1, f"2020-01-01T00:00:{i:02d}Z", str(i), str(i % 2), str(i * 9)) for i in range(40)])
            conn.executemany("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,role,inferred_type,created_at,updated_at) VALUES(1,?,?,?,?, '2020','2020')", [
                ("event_time", "event_time", "timestamp", "datetime"),
                ("signal", "signal", "numeric_telemetry_feature", "numeric"),
                ("mode", "mode", "context", "text"),
                ("drop_me", "drop_me", "numeric_telemetry_feature", "numeric")])
    config = tmp_path / "features.yaml"
    config.write_text(yaml.safe_dump({"dataset_id": dataset_id, "columns": [
        {"sanitized_name": name, "role": role} for name, role in [
            (timestamp, "timestamp"), ("signal", "context"),
            ("mode", "numeric_telemetry_feature"), ("drop_me", "excluded")]]}), encoding="utf-8")
    prepare_dataset(dataset_id, config, db_path=db, logs_dir=tmp_path / "logs")
    metadata = load_dataset(dataset_id, db_path=db)
    if not canonical:
        assert "ordering" not in metadata["manifest"]  # Preserve existing timestamped legacy provenance.
    assert {c["name"] for c in metadata["columns"] if c["role"] == "numeric"} == {"mode"}
    assert "drop_me" not in next(iter_batches(dataset_id, db_path=db)).schema.names
    context = get_context_window(dataset_id, 1, 2, db_path=db)
    assert "signal" in context[0] and "mode" not in context[0]
    assert context[0]["timestamp"].startswith("2020-01-01")
    assert get_column_summary(dataset_id, "mode", 1, 2, db_path=db)["max"] == 1
    training = tmp_path / "train.yaml"
    training.write_text("window_size: 2\nepochs: 1\nhidden_size: 2\nbatch_size: 8\n", encoding="utf-8")
    result = train(dataset_id, training, db_path=db, storage_dir=tmp_path)
    transform = json.loads((Path(result["model_path"]).parent / "transform.json").read_text())
    assert [f["source_name"] for f in transform["features"]] == ["mode"]


def test_profile_canonical_generates_editable_feature_config(tmp_path):
    from src.profiling.dataset_profile import profile_dataset
    source = tmp_path / "data.csv"
    source.write_text("timestamp,signal\n2020-01-01,1\n2020-01-02,2\n", encoding="utf-8")
    db = tmp_path / "catalog.db"
    imported = ingest_source(source, db_path=db, storage_dir=tmp_path)
    result = profile_dataset(imported["dataset_id"], db_path=db, generated_config_dir=tmp_path / "configs")
    assert yaml.safe_load(Path(result["config_path"]).read_text())["columns"]


def test_peak_lookup_preserves_stored_diagnostic_peak(monkeypatch, tmp_path):
    from src.rag import telemetry_tools
    monkeypatch.setattr(telemetry_tools, "get_anomaly", lambda *a, **k: {
        "dataset_id": 1, "start_row_id": 1, "end_row_id": 9, "score_timestamp": "2020-01-01",
        "packet_json": '{"peak_row_id": 3}'})
    monkeypatch.setattr(telemetry_tools, "get_context_window", lambda d, a, b, **kw: [a, b])
    assert get_rows_around_peak(1, "alert", radius=1, db_path=tmp_path / "db") == [2, 4]


def test_prepared_only_context_and_summary_remain_available(tmp_path):
    db = tmp_path / "prepared.db"
    init_db(db)
    with get_connection(db) as conn:
        conn.execute("INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) VALUES(1,'legacy','prepared',1,'{}','2020','2020')")
        conn.execute("CREATE TABLE model_ready_dataset_1(row_id INTEGER,dataset_id INTEGER,timestamp TEXT,signal REAL,ground_truth REAL)")
        conn.execute("INSERT INTO model_ready_dataset_1 VALUES(1,1,'2020-01-01',0.5,1)")
        conn.execute("ALTER TABLE model_ready_dataset_1 ADD COLUMN outcome__bad INTEGER DEFAULT 1")
        conn.execute("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,role,created_at,updated_at) VALUES(1,'outcome','outcome','label','2020','2020')")
        conn.execute("CREATE TABLE context_ready_dataset_1(row_id INTEGER,dataset_id INTEGER,timestamp TEXT,note TEXT,fault_label TEXT)")
        conn.execute("INSERT INTO context_ready_dataset_1 VALUES(1,1,'2020-01-01','check','fault')")
    assert get_context_window(1, 1, 1, db_path=db) == [{"row_id": 1, "dataset_id": 1, "timestamp": "2020-01-01", "note": "check"}]
    summary = get_column_summary(1, "signal", 1, 1, db_path=db)
    assert summary["mean"] == 0.5 and summary["value_space"] == "legacy_prepared"
    assert "error" in get_column_summary(1, "ground_truth", 1, 1, db_path=db)
    assert "error" in get_column_summary(1, "outcome__bad", 1, 1, db_path=db)


def test_legacy_one_hot_summary_remains_available(tmp_path):
    db = tmp_path / "onehot.db"
    init_db(db)
    with get_connection(db) as conn:
        conn.execute("INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) VALUES(1,'legacy','prepared',1,'{}','2020','2020')")
        conn.execute("CREATE TABLE raw_dataset_1(row_id INTEGER,dataset_id INTEGER,timestamp TEXT,mode TEXT)")
        conn.execute("INSERT INTO raw_dataset_1 VALUES(1,1,'2020-01-01','safe hold')")
        conn.execute("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,role,created_at,updated_at) VALUES(1,'mode','mode','categorical','2020','2020')")
        conn.execute("CREATE TABLE model_ready_dataset_1(row_id INTEGER,dataset_id INTEGER,mode__safe_hold INTEGER)")
        conn.execute("INSERT INTO model_ready_dataset_1 VALUES(1,1,1)")
    assert get_column_summary(1, "mode__safe_hold", 1, 1, db_path=db)["mean"] == 1


def test_invalid_feature_config_does_not_replace_prepared_tables(tmp_path):
    db = tmp_path / "invalid.db"
    init_db(db)
    with get_connection(db) as conn:
        conn.execute("INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) VALUES(1,'legacy','prepared',1,'{}','2020','2020')")
        conn.execute("CREATE TABLE raw_dataset_1(row_id INTEGER,dataset_id INTEGER,timestamp TEXT,signal TEXT)")
        conn.execute("INSERT INTO raw_dataset_1 VALUES(1,1,'2020-01-01','3')")
        conn.execute("CREATE TABLE model_ready_dataset_1(marker TEXT)")
        conn.execute("INSERT INTO model_ready_dataset_1 VALUES('preserve')")
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump({"columns": [{"sanitized_name": "unknown", "role": "numeric_telemetry_feature"}]}))
    with pytest.raises(ValueError, match="Unknown"):
        prepare_dataset(1, config, db_path=db, logs_dir=tmp_path / "logs")
    with get_connection(db) as conn:
        assert conn.execute("SELECT marker FROM model_ready_dataset_1").fetchone()[0] == "preserve"


def test_canonical_invalid_numeric_role_is_retryable_without_metadata_changes(tmp_path, monkeypatch):
    from src.modeling.transforms import fingerprint
    from src.common import paths
    monkeypatch.setattr(paths, "STORAGE_ROOT", tmp_path)
    db = tmp_path / "catalog.db"
    source = tmp_path / "data.csv"
    source.write_text("timestamp,note,signal\n2020-01-01,hello,1\n2020-01-02,world,2\n")
    dataset = ingest_source(source, {"context_columns": ["note"]}, db_path=db, storage_dir=tmp_path)
    before = load_dataset(dataset["dataset_id"], db_path=db)
    config = tmp_path / "features.yaml"
    columns = [{"sanitized_name": name, "role": role} for name, role in [
        ("timestamp", "timestamp"), ("note", "numeric_telemetry_feature"), ("signal", "excluded")]]
    config.write_text(yaml.safe_dump({"columns": columns}))
    with pytest.raises(ValueError, match="numeric"):
        prepare_dataset(dataset["dataset_id"], config, db_path=db, logs_dir=tmp_path / "logs")
    assert load_dataset(dataset["dataset_id"], db_path=db)["columns"] == before["columns"]
    columns[1]["role"] = "context"
    columns[2]["role"] = "numeric_telemetry_feature"
    config.write_text(yaml.safe_dump({"columns": columns}))
    prepared = prepare_dataset(dataset["dataset_id"], config, db_path=db, logs_dir=tmp_path / "logs")
    assert prepared["manifest_fingerprint"] == fingerprint(load_dataset(dataset["dataset_id"], db_path=db)["manifest"])


def test_second_legacy_table_failure_rolls_back_first_table(tmp_path, monkeypatch):
    import importlib
    preparation = importlib.import_module("src.cleaning.prepare_dataset")
    db = tmp_path / "rollback.db"
    init_db(db)
    with get_connection(db) as conn:
        conn.execute("INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) VALUES(1,'legacy','prepared',1,'{}','2020','2020')")
        conn.execute("CREATE TABLE raw_dataset_1(row_id INTEGER,dataset_id INTEGER,timestamp TEXT,signal TEXT)")
        conn.execute("INSERT INTO raw_dataset_1 VALUES(1,1,'2020-01-01','3')")
        conn.execute("CREATE TABLE model_ready_dataset_1(marker TEXT)")
        conn.execute("INSERT INTO model_ready_dataset_1 VALUES('preserve')")
    config = tmp_path / "features.yaml"
    config.write_text(yaml.safe_dump({"columns": [
        {"sanitized_name": "timestamp", "role": "timestamp"},
        {"sanitized_name": "signal", "role": "numeric_telemetry_feature"}]}))
    def fail(**kwargs):
        raise RuntimeError("simulated context write failure")
    monkeypatch.setattr(preparation, "_write_context_ready_table", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        prepare_dataset(1, config, db_path=db, logs_dir=tmp_path / "logs")
    with get_connection(db) as conn:
        assert conn.execute("SELECT marker FROM model_ready_dataset_1").fetchone()[0] == "preserve"
        assert conn.execute("SELECT count(*) FROM dataset_columns").fetchone()[0] == 0


def test_timestamp_free_csv_uses_explicit_row_order_without_fake_dates(tmp_path, monkeypatch):
    from src.common import paths
    from src.ingestion.csv_ingestion import ingest_csvs
    from src.modeling.pipeline import detect
    from src.ingestion.store import _parquet_batches
    from src.profiling.dataset_profile import profile_dataset
    pytest.importorskip("torch").set_num_threads(1)
    monkeypatch.setattr(paths, "STORAGE_ROOT", tmp_path)
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "ordered.csv").write_text("signal\n" + "".join(f"{i}\n" for i in range(40)))
    db = tmp_path / "catalog.db"
    assert ingest_csvs(source_dir, db_path=db, logs_dir=tmp_path / "logs")["ingested"] == 1
    dataset = load_dataset(1, db_path=db)
    assert dataset["manifest"]["ordering"] == "row"
    batch = next(iter_batches(1, db_path=db))
    assert batch.column("timestamp").null_count == 40
    assert batch.column("timestamp_ns").to_pylist() == list(range(1, 41))
    profile = profile_dataset(1, db_path=db, generated_config_dir=tmp_path / "configs")
    prepare_dataset(1, profile["config_path"], db_path=db, logs_dir=tmp_path / "logs")
    config = tmp_path / "train.yaml"
    config.write_text("window_size: 2\nepochs: 1\nhidden_size: 2\n")
    from src.modeling.train_model import train_model
    result = train_model(1, config, db_path=db, models_dir=tmp_path / "custom-model-directory")
    assert Path(result["model_path"]).is_relative_to(tmp_path / "custom-model-directory")
    run = detect(1, result["model_id"], partition="all", db_path=db, storage_dir=tmp_path)
    assert run["windows_scored"] == 37  # Windows still respect the three partition boundaries.
    for batch in _parquet_batches(run["scores_path"], batch_size=10):
        assert batch.column("timestamp").null_count == batch.num_rows
    from src.modeling.monitoring import monitor
    with pytest.raises(ValueError, match="real timestamps"):
        monitor(1, "2020-01-01", "2020-02-01", db_path=db, storage_dir=tmp_path)


def test_config_null_values_and_obsolete_split_are_explicit(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("epochs: null\ntrain_test_split: 0.8\n")
    with pytest.warns(UserWarning, match="manifest splits"):
        assert _load_config(path)["epochs"] == 3
