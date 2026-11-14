"""Legacy packets and diagnosis remain available after raw telemetry is archived."""
import json

import pytest

from src.anomaly.packet import build_anomaly_packet
from src.db import get_connection, init_db, insert_anomaly, insert_dataset, insert_model_metadata


def _prepared_only(tmp_path, *, context=True):
    db = tmp_path / "legacy.db"
    init_db(db)
    dataset_id = insert_dataset("legacy", status="prepared", row_count=3, db_path=db)
    with get_connection(db) as conn:
        conn.execute("CREATE TABLE model_ready_dataset_1(row_id INTEGER PRIMARY KEY,dataset_id INTEGER,"
                     "timestamp TEXT,signal REAL,ground_truth REAL,secret REAL)")
        conn.executemany("INSERT INTO model_ready_dataset_1 VALUES(?,?,?,?,?,?)", [
            (1, 1, "2020-01-01T00:00:00Z", -1.0, 0.0, 123.0),
            (2, 1, "2020-01-01T00:00:01Z", 0.0, 1.0, 456.0),
            (3, 1, "2020-01-01T00:00:02Z", 1.0, 0.0, 789.0),
        ])
        conn.execute("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,role,created_at,updated_at) "
                     "VALUES(1,'secret','secret','annotation','2020','2020')")
        if context:
            conn.execute("CREATE TABLE context_ready_dataset_1(row_id INTEGER PRIMARY KEY,dataset_id INTEGER,"
                         "timestamp TEXT,operating_mode TEXT,fault_label TEXT,secret TEXT)")
            conn.executemany("INSERT INTO context_ready_dataset_1 VALUES(?,?,?,?,?,?)", [
                (1, 1, "2020-01-01T00:00:00Z", "idle", "normal", "hidden-A"),
                (2, 1, "2020-01-01T00:00:01Z", "active", "fault", "hidden-B"),
                (3, 1, "2020-01-01T00:00:02Z", "active", "normal", "hidden-C"),
            ])
    model_id = insert_model_metadata(dataset_id, tmp_path / "legacy.pt", config_path=None, threshold=1,
                                     threshold_method="percentile", threshold_percentile=95, training_summary={},
                                     metadata={"feature_columns": ["signal", "ground_truth", "secret"],
                                               "config": {"window_size": 3}}, db_path=db)
    insert_anomaly(anomaly_id="legacy-alert", dataset_id=dataset_id, model_id=model_id, window_index=0,
                   start_row_id=1, end_row_id=3, start_timestamp="2020-01-01T00:00:00Z",
                   end_timestamp="2020-01-01T00:00:02Z", anomaly_score=2, threshold=1,
                   contributing_features=[{"feature": "signal", "score": 2},
                                          {"feature": "ground_truth", "score": 9},
                                          {"feature": "secret", "score": 8}], packet={}, db_path=db)
    return db


@pytest.mark.parametrize("context", [True, False])
def test_prepared_only_packet_marks_values_and_redacts_annotations(tmp_path, context):
    db = _prepared_only(tmp_path, context=context)
    packet = build_anomaly_packet("legacy-alert", db_path=db)
    assert packet["raw_data_reference"]["available"] is False
    assert packet["numeric_summary"]["signal"]["start"] == -1.0
    assert packet["numeric_summary"]["signal"]["end"] == 1.0
    assert packet["numeric_summary"]["signal"]["value_space"] == "prepared_model_values"
    assert packet["numeric_summary"]["signal"]["unit"] is None
    assert packet["model_context"]["original_units_available"] is False
    assert "standardized or encoded" in packet["model_context"]["diagnostic_note"]
    assert packet["model_context"]["features_used"] == ["signal"]
    assert packet["categorical_changes"] == ({"operating_mode": ["idle", "active"]} if context else {})
    text = json.dumps(packet)
    assert all(name not in text for name in ("ground_truth", "fault_label", "secret", "hidden-", "percent_change"))


def test_prepared_only_diagnose_keeps_units_warning_in_prompt(tmp_path, monkeypatch):
    from src.rag import diagnosis
    db = _prepared_only(tmp_path)
    template = tmp_path / "prompt.txt"
    template.write_text("Packet: {anomaly_packet}\nDocuments: {retrieved_chunks}", encoding="utf-8")
    prompts = []
    monkeypatch.setattr(diagnosis, "retrieve_relevant_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(diagnosis, "call_ollama", lambda prompt, **kwargs: (prompts.append(prompt) or None, "offline"))
    result = diagnosis.diagnose_anomaly("legacy-alert", db_path=db, prompt_template_path=template)
    assert result["diagnosis_id"] > 0
    assert prompts == []  # No documents: abstain without invoking the model.
    assert "standardized or encoded" in result["response_text"]
    assert '"original_units_available": false' in result["response_text"]
    assert all(name not in result["response_text"] for name in ("ground_truth", "fault_label", "secret", "hidden-"))


def test_prepared_packet_retains_window_bound(tmp_path):
    db = _prepared_only(tmp_path)
    with get_connection(db) as conn:
        conn.execute("UPDATE anomalies SET end_row_id=10001 WHERE anomaly_id='legacy-alert'")
    with pytest.raises(ValueError, match="at most 10000"):
        build_anomaly_packet("legacy-alert", db_path=db)
