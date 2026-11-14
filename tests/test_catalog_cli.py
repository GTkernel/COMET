"""Catalog compatibility and externally observable CLI failure behavior."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from src.db import _migrate_catalog, get_connection, get_detection_run, init_db


def _legacy_catalog(path):
    # Historical schema, without modern provenance columns/tables.
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE datasets(dataset_id INTEGER PRIMARY KEY,name TEXT,description TEXT,status TEXT,
            row_count INTEGER,metadata_json TEXT,created_at TEXT,updated_at TEXT);
        CREATE TABLE models(model_id INTEGER PRIMARY KEY,dataset_id INTEGER,feature_config_id INTEGER,
            model_path TEXT,config_path TEXT,threshold REAL,threshold_method TEXT,threshold_percentile REAL,
            training_summary_json TEXT,metadata_json TEXT,created_at TEXT);
        CREATE TABLE anomalies(anomaly_id TEXT PRIMARY KEY,dataset_id INTEGER,model_id INTEGER,
            window_index INTEGER,start_row_id INTEGER,end_row_id INTEGER,start_timestamp TEXT,
            end_timestamp TEXT,anomaly_score REAL,threshold REAL,contributing_features_json TEXT,
            packet_json TEXT,created_at TEXT);
        INSERT INTO datasets VALUES(42,'historical','', 'prepared',10,'{}','2020','2020');
        INSERT INTO models VALUES(91,42,NULL,'legacy.pt',NULL,2,'percentile',95,'{}','{}','2020');
        INSERT INTO anomalies VALUES('historical-alert',42,91,3,1,10,'2020-01-01','2020-01-02',3,2,'[]','{}','2020');
        """)


def test_migration_preserves_historical_ids_and_results(tmp_path):
    path = tmp_path / "catalog.db"
    _legacy_catalog(path)
    init_db(path)
    init_db(path)
    with get_connection(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        row = conn.execute("SELECT * FROM anomalies").fetchone()
        assert (row["anomaly_id"], row["dataset_id"], row["model_id"], row["anomaly_score"]) == ("historical-alert", 42, 91, 3)
        assert row["run_id"] is None
        assert conn.execute("SELECT model_path FROM models WHERE model_id=91").fetchone()[0] == "legacy.pt"
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_failed_migration_rolls_back_columns_and_version(tmp_path):
    path = tmp_path / "catalog.db"
    _legacy_catalog(path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE detection_runs(x TEXT)")  # Forces index creation failure.
        conn.commit()
        with pytest.raises(sqlite3.OperationalError):
            _migrate_catalog(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        assert "run_id" not in {row[1] for row in conn.execute("PRAGMA table_info(anomalies)")}
        assert conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0] == 1


def test_newer_catalog_is_not_modified(tmp_path):
    path = tmp_path / "catalog.db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=100")
    with pytest.raises(ValueError, match="newer"):
        init_db(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0] == 0


def test_incomplete_run_is_never_selected(tmp_path):
    path = tmp_path / "catalog.db"
    _legacy_catalog(path)
    init_db(path)
    with get_connection(path) as conn:
        for run_id, status in (("older", "completed"), ("newer", "failed")):
            conn.execute("INSERT INTO detection_runs VALUES(?,?,?,?,?,?,?,?)",
                         (run_id, 42, 91, status, str(tmp_path), "{}", "2026", "2026"))
    assert get_detection_run(dataset_id=42, model_id=91, db_path=path)["run_id"] == "older"
    assert get_detection_run("newer", db_path=path) is None


def test_cli_invalid_source_exits_nonzero(tmp_path):
    env = {**os.environ, "COMET_STORAGE_DIR": str(tmp_path / "runtime")}
    result = subprocess.run([sys.executable, "-m", "src.cli", "ingest", str(tmp_path / "missing.csv")],
                            capture_output=True, text=True, env=env, check=False)
    assert result.returncode != 0
    assert "error:" in result.stderr


def test_raw_window_request_is_bounded():
    from src.rag.telemetry_tools import get_raw_window
    assert get_raw_window(1, 1, 10000000)["error"] == "invalid_row_range"
