"""Historical checkpoints remain same-dataset usable without stale detection leakage."""
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from src.anomaly.detect import detect_anomalies
from src.anomaly.packet import build_anomaly_packet
from src.cli.__main__ import list_anomalies
from src.evaluation.export import export_predictions
from src.db import (
    get_connection,
    get_detection_run,
    init_db,
    insert_anomaly,
    insert_model_metadata,
)
from src.modeling.lstm_autoencoder import LSTMAutoencoder


def test_legacy_profile_prepare_preserves_yaml_roles_and_quoted_values(tmp_path):
    import yaml
    from src.cleaning.prepare_dataset import prepare_dataset
    from src.profiling.dataset_profile import profile_dataset

    db = tmp_path / "legacy-yaml.db"
    init_db(db)
    values = [
        (index + 1, 1, f"2020-01-01T00:00:{index:02d}Z", str(index + 1),
         "safe:hold" if index % 2 else 'science, "A"', f"operator: check #{index}")
        for index in range(4)
    ]
    with get_connection(db) as conn:
        conn.execute(
            "INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) "
            "VALUES(1,'legacy','ingested',4,'{}','2020','2020')"
        )
        conn.execute(
            "CREATE TABLE raw_dataset_1(row_id INTEGER PRIMARY KEY,dataset_id INTEGER,"
            "timestamp TEXT,voltage TEXT,mode TEXT,note TEXT)"
        )
        conn.executemany("INSERT INTO raw_dataset_1 VALUES(?,?,?,?,?,?)", values)
    generated = tmp_path / "generated"
    generated.mkdir()
    config_path = generated / "feature_config_dataset_1.yaml"
    config_path.write_text("overrides:\n  mode: categorical_feature\n  note: context\n", encoding="utf-8")
    profile_dataset(1, db_path=db, generated_config_dir=generated)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    columns = {column["sanitized_name"]: column for column in config["columns"]}
    assert columns["mode"]["role"] == "categorical_feature"
    assert columns["mode"]["sample_values"] == [values[0][4], values[1][4]]
    assert columns["note"]["role"] == "context"
    prepared = prepare_dataset(1, config_path, db_path=db, logs_dir=tmp_path / "logs")
    assert prepared["row_count"] == 4
    with get_connection(db) as conn:
        rows = conn.execute("SELECT * FROM context_ready_dataset_1 ORDER BY row_id").fetchall()
        assert [(row["mode"], row["note"]) for row in rows] == [(row[4], row[5]) for row in values]
        features = {row["name"] for row in conn.execute("PRAGMA table_info(model_ready_dataset_1)")}
        assert {"voltage", "mode__safe_hold", "mode__science_a"} <= features
        assert "note" not in features


def test_legacy_reruns_are_isolated_and_report_window_end(tmp_path, monkeypatch):
    from src.common import paths
    from src.ingestion import store
    monkeypatch.setattr(store, "PARQUET_PART_ROWS", 3)
    monkeypatch.setattr(paths, "STORAGE_ROOT", tmp_path)
    # detect's imported constant has the old path, so set it in its module too.
    monkeypatch.setenv("COMET_STORAGE_DIR", str(tmp_path))
    db = tmp_path / "legacy.db"
    init_db(db)
    with get_connection(db) as conn:
        conn.execute("INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) VALUES(1,'legacy','prepared',8,'{}','2020','2020')")
        conn.execute("CREATE TABLE model_ready_dataset_1(row_id INTEGER PRIMARY KEY,dataset_id INTEGER,timestamp TEXT,signal REAL)")
        conn.executemany("INSERT INTO model_ready_dataset_1 VALUES(?,?,?,?)",
                         [(i + 1, 1, f"2020-01-01T00:00:{i:02d}Z", float(i)) for i in range(8)])
    checkpoint = tmp_path / "old.pt"
    model = LSTMAutoencoder(1, hidden_size=4, num_layers=1)
    torch.save({"model_state_dict": model.state_dict(), "feature_columns": ["signal"],
                "config": {"window_size": 3, "stride": 1, "hidden_size": 4, "num_layers": 1}}, checkpoint)
    model_id = insert_model_metadata(1, checkpoint, config_path=None, threshold=-1,
                                     threshold_method="legacy", threshold_percentile=95,
                                     training_summary={}, metadata={}, db_path=db)
    first = detect_anomalies(1, model_id, db_path=db, batch_size=2)
    assert first["windows_scored"] == first["anomalies_detected"] == 6
    assert pq.read_table(first["scores_path"]).num_rows == 8
    parts = sorted(Path(first["scores_path"]).glob("*.parquet"))
    assert len(parts) == 3
    for part in parts:
        with pq.ParquetFile(part) as reader:
            assert reader.metadata.num_row_groups == 1 and reader.metadata.num_rows <= 3
    with get_connection(db) as conn:
        alert = conn.execute("SELECT * FROM anomalies WHERE run_id=? LIMIT 1", (first["run_id"],)).fetchone()
        assert alert["start_timestamp"] == alert["end_timestamp"] == alert["score_timestamp"]
        assert json.loads(alert["packet_json"])["input_window_start_timestamp"] < alert["score_timestamp"]
        conn.execute("UPDATE models SET threshold=1000000000 WHERE model_id=?", (model_id,))
    second = detect_anomalies(1, model_id, db_path=db, batch_size=3)
    assert second["run_id"] != first["run_id"]
    assert second["anomalies_detected"] == 0
    assert get_detection_run(dataset_id=1, model_id=model_id, db_path=db)["run_id"] == second["run_id"]
    assert list_anomalies(dataset_id=1, model_id=model_id, db_path=db) == []
    assert len(list_anomalies(run_id=first["run_id"], db_path=db)) == 6
    from src.modeling.diagnostics import run_model_diagnostics
    selected = run_model_diagnostics(1, model_id, run_id=first["run_id"], db_path=db,
                                     output_dir=tmp_path / "diagnostics")
    assert selected["run_id"] == first["run_id"]
    assert selected["anomalies_detected"] == 6
    assert selected["threshold"] == -1
    part_export = tmp_path / "part-export.csv"
    export_predictions(first["run_id"], part_export, db_path=db)
    historical_file = tmp_path / "historical-scores.parquet"
    pq.write_table(pq.read_table(first["scores_path"]), historical_file)
    with get_connection(db) as conn:
        metadata = json.loads(conn.execute("SELECT metadata_json FROM detection_runs WHERE run_id=?", (first["run_id"],)).fetchone()[0])
        metadata.update(scores_path=str(historical_file), scores_format="parquet")
        conn.execute("UPDATE detection_runs SET metadata_json=? WHERE run_id=?", (json.dumps(metadata), first["run_id"]))
    historical = run_model_diagnostics(1, model_id, run_id=first["run_id"], db_path=db,
                                       output_dir=tmp_path / "historical-diagnostics")
    assert historical["score_count"] == selected["score_count"] == 6
    assert historical["score_mean"] == pytest.approx(selected["score_mean"])
    single_export = tmp_path / "single-export.csv"
    export_predictions(first["run_id"], single_export, db_path=db)
    assert part_export.read_bytes() == single_export.read_bytes()
    with pytest.raises(ValueError, match="completed detection run"):
        run_model_diagnostics(1, model_id, run_id="missing", db_path=db,
                              output_dir=tmp_path / "diagnostics")
    with get_connection(db) as conn:
        assert conn.execute("SELECT count(*) FROM anomalies WHERE run_id=?", (first["run_id"],)).fetchone()[0] == 6


def test_legacy_packet_redacts_annotation_feature_identities(tmp_path, monkeypatch):
    from src.rag import telemetry_tools

    db = tmp_path / "legacy-packet.db"
    init_db(db)
    with get_connection(db) as conn:
        conn.execute(
            "INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) "
            "VALUES(1,'legacy','prepared',2,'{}','2020','2020')"
        )
    model_id = insert_model_metadata(
        1,
        tmp_path / "old.pt",
        config_path=None,
        threshold=1,
        threshold_method="legacy",
        threshold_percentile=95,
        training_summary={},
        metadata={
            "feature_columns": ["signal", "ground_truth", "is_anomaly_signal"],
            "config": {"window_size": 2},
        },
        db_path=db,
    )
    insert_anomaly(
        anomaly_id="legacy-alert",
        dataset_id=1,
        model_id=model_id,
        window_index=0,
        start_row_id=1,
        end_row_id=2,
        start_timestamp="2020-01-01T00:00:00Z",
        end_timestamp="2020-01-01T00:00:01Z",
        anomaly_score=2,
        threshold=1,
        contributing_features=[
            {"feature": "ground_truth", "score": 9, "rank": 1},
            {"feature": "signal", "score": 3, "rank": 2},
            {"feature": "is_anomaly_signal", "score": 1, "rank": 3},
        ],
        packet={},
        db_path=db,
    )
    rows = [
        {"row_id": 1, "timestamp": "2020-01-01T00:00:00Z", "signal": 1.0,
         "ground_truth": 0, "is_anomaly_signal": 0},
        {"row_id": 2, "timestamp": "2020-01-01T00:00:01Z", "signal": 4.0,
         "ground_truth": 1, "is_anomaly_signal": 1},
    ]
    context = [
        {"row_id": 1, "fault_label": "normal", "operating_mode": "idle"},
        {"row_id": 2, "fault_label": "fault", "operating_mode": "active"},
    ]
    monkeypatch.setattr(telemetry_tools, "get_raw_window", lambda *args, **kwargs: rows)
    monkeypatch.setattr(telemetry_tools, "get_context_window", lambda *args, **kwargs: context)

    packet = build_anomaly_packet("legacy-alert", db_path=db)

    assert packet["model_context"]["features_used"] == ["signal"]
    assert packet["time_window"]["start_row_id"] == packet["time_window"]["end_row_id"] == 2
    assert packet["input_window"]["start_row_id"] == 1
    assert packet["input_window"]["start_timestamp"] == rows[0]["timestamp"]
    assert packet["input_window"]["end_timestamp"] == rows[-1]["timestamp"]
    assert packet["top_contributing_features"] == [
        {"feature": "signal", "score": 3, "rank": 2}
    ]
    assert set(packet["numeric_summary"]) == {"signal"}
    assert packet["categorical_changes"] == {"operating_mode": ["idle", "active"]}
    assert packet["redacted_annotation_feature_count"] == 2
    assert packet["redaction_note"] == (
        "Legacy checkpoint used annotation-like feature identities. Names and contribution "
        "details were redacted. Retrain with a version 2 manifest."
    )
    serialized = json.dumps(packet)
    assert "ground_truth" not in serialized
    assert "is_anomaly_signal" not in serialized
    assert "fault_label" not in serialized
