from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.paths import DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_db_path(db_path: str | Path | None) -> str:
    if db_path is None:
        path = DB_PATH
    else:
        path = Path(db_path)

    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)


class _CatalogConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(_prepare_db_path(db_path), factory=_CatalogConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    with get_connection(db_path) as conn:
        if conn.execute("PRAGMA user_version").fetchone()[0] > 1:
            raise ValueError("This database was created by a newer COMET version.")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'created',
                row_count INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_files (
                source_file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                modified_time TEXT,
                content_hash TEXT,
                row_count INTEGER,
                status TEXT NOT NULL DEFAULT 'registered',
                created_at TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
            );

            CREATE TABLE IF NOT EXISTS dataset_columns (
                dataset_column_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                column_name TEXT NOT NULL,
                sanitized_name TEXT,
                inferred_type TEXT,
                role TEXT,
                missing_ratio REAL,
                unique_count INTEGER,
                unique_ratio REAL,
                example_values_json TEXT NOT NULL DEFAULT '[]',
                reasoning TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(dataset_id, column_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
            );

            CREATE TABLE IF NOT EXISTS feature_configs (
                feature_config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                config_path TEXT NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}',
                version TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
            );

            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                feature_config_id INTEGER,
                model_path TEXT NOT NULL,
                config_path TEXT,
                threshold REAL,
                threshold_method TEXT,
                threshold_percentile REAL,
                training_summary_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id),
                FOREIGN KEY(feature_config_id) REFERENCES feature_configs(feature_config_id)
            );

            CREATE TABLE IF NOT EXISTS anomalies (
                anomaly_id TEXT PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                window_index INTEGER NOT NULL,
                start_row_id INTEGER NOT NULL,
                end_row_id INTEGER NOT NULL,
                start_timestamp TEXT,
                end_timestamp TEXT,
                anomaly_score REAL NOT NULL,
                threshold REAL NOT NULL,
                contributing_features_json TEXT NOT NULL DEFAULT '[]',
                packet_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id),
                FOREIGN KEY(model_id) REFERENCES models(model_id)
            );

            CREATE TABLE IF NOT EXISTS rag_documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                modified_time TEXT,
                content_hash TEXT,
                document_version TEXT,
                status TEXT NOT NULL DEFAULT 'registered',
                ingested_at TEXT NOT NULL,
                UNIQUE(source_path, document_version)
            );

            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                section_title TEXT,
                page_number INTEGER,
                text_span TEXT,
                chunk_text TEXT NOT NULL,
                embedding_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(document_id, chunk_index),
                FOREIGN KEY(document_id) REFERENCES rag_documents(document_id)
            );

            CREATE TABLE IF NOT EXISTS diagnoses (
                diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                anomaly_id TEXT NOT NULL,
                prompt_template_name TEXT,
                prompt_template_version TEXT,
                llm_model TEXT,
                retrieved_chunks_json TEXT NOT NULL DEFAULT '[]',
                response_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(anomaly_id) REFERENCES anomalies(anomaly_id)
            );
            """
        )
        conn.commit()
        _migrate_catalog(conn)


def _migrate_catalog(conn: sqlite3.Connection) -> None:
    """Add versioned run provenance without rewriting historical data or IDs."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            conn.execute("""CREATE TABLE IF NOT EXISTS preparations (
                preparation_id TEXT PRIMARY KEY, dataset_id INTEGER NOT NULL,
                artifact_dir TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS detection_runs (
                run_id TEXT PRIMARY KEY, dataset_id INTEGER NOT NULL, model_id INTEGER NOT NULL,
                status TEXT NOT NULL, artifact_dir TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id),
                FOREIGN KEY(model_id) REFERENCES models(model_id))""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(anomalies)")}
            if "run_id" not in columns:
                conn.execute("ALTER TABLE anomalies ADD COLUMN run_id TEXT REFERENCES detection_runs(run_id)")
            if "score_timestamp" not in columns:
                conn.execute("ALTER TABLE anomalies ADD COLUMN score_timestamp TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS anomalies_run_idx ON anomalies(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS detection_runs_target_idx ON detection_runs(dataset_id,model_id,status)")
            conn.execute("PRAGMA user_version=1")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def get_detection_run(run_id: str | None = None, *, dataset_id: int | None = None,
                      model_id: int | None = None, db_path=None) -> dict[str, Any] | None:
    """Find a completed run so reports exclude partial results."""
    clauses, params = ["status = 'completed'"], []
    for column, value in (("run_id", run_id), ("dataset_id", dataset_id), ("model_id", model_id)):
        if value is not None:
            clauses.append(column + " = ?")
            params.append(value)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM detection_runs WHERE " + " AND ".join(clauses)
                           + " ORDER BY created_at DESC, rowid DESC LIMIT 1", params).fetchone()
    return dict(row) if row else None


def insert_dataset(
    name: str | None = None,
    *,
    description: str | None = None,
    status: str = "created",
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> int:
    now = utc_now()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO datasets (
                name, description, status, row_count, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                description,
                status,
                row_count,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_dataset(dataset_id: int, db_path: str | Path | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()

    return dict(row) if row is not None else None


def insert_feature_config(
    dataset_id: int,
    config_path: str | Path,
    config: dict[str, Any],
    *,
    version: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO feature_configs(dataset_id, config_path, config_json, version, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (dataset_id, str(config_path), json.dumps(config), version, utc_now()),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_feature_config(
    dataset_id: int,
    config_path: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    query = "SELECT * FROM feature_configs WHERE dataset_id = ?"
    params: list[Any] = [dataset_id]
    if config_path is not None:
        query += " AND config_path = ?"
        params.append(str(config_path))
    query += " ORDER BY feature_config_id DESC LIMIT 1"

    with get_connection(db_path) as conn:
        row = conn.execute(query, params).fetchone()

    return dict(row) if row is not None else None


def insert_model_metadata(
    dataset_id: int,
    model_path: str | Path,
    *,
    config_path: str | Path | None,
    threshold: float,
    threshold_method: str,
    threshold_percentile: float,
    training_summary: dict[str, Any],
    metadata: dict[str, Any],
    feature_config_id: int | None = None,
    db_path: str | Path | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO models (
                dataset_id, feature_config_id, model_path, config_path, threshold,
                threshold_method, threshold_percentile, training_summary_json,
                metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                feature_config_id,
                str(model_path),
                str(config_path) if config_path is not None else None,
                threshold,
                threshold_method,
                threshold_percentile,
                json.dumps(training_summary),
                json.dumps(metadata),
                utc_now(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_model(model_id: int, db_path: str | Path | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def insert_anomaly(
    *,
    anomaly_id: str,
    dataset_id: int,
    model_id: int,
    window_index: int,
    start_row_id: int,
    end_row_id: int,
    start_timestamp: str | None,
    end_timestamp: str | None,
    anomaly_score: float,
    threshold: float,
    contributing_features: list[dict[str, Any]],
    packet: dict[str, Any],
    db_path: str | Path | None = None,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO anomalies (
                anomaly_id, dataset_id, model_id, window_index, start_row_id,
                end_row_id, start_timestamp, end_timestamp, anomaly_score,
                threshold, contributing_features_json, packet_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anomaly_id,
                dataset_id,
                model_id,
                window_index,
                start_row_id,
                end_row_id,
                start_timestamp,
                end_timestamp,
                anomaly_score,
                threshold,
                json.dumps(contributing_features),
                json.dumps(packet),
                utc_now(),
            ),
        )
        conn.commit()


def get_anomaly(anomaly_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM anomalies WHERE anomaly_id = ?",
            (anomaly_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def insert_diagnosis(
    *,
    anomaly_id: str,
    prompt_template_name: str,
    prompt_template_version: str,
    llm_model: str,
    retrieved_chunks: list[dict[str, Any]],
    response_text: str,
    metadata: dict[str, Any],
    db_path: str | Path | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO diagnoses (
                anomaly_id, prompt_template_name, prompt_template_version,
                llm_model, retrieved_chunks_json, response_text, metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anomaly_id,
                prompt_template_name,
                prompt_template_version,
                llm_model,
                json.dumps(retrieved_chunks),
                response_text,
                json.dumps(metadata),
                utc_now(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def insert_dataset_file(
    dataset_id: int,
    file_path: str | Path,
    *,
    file_size: int | None = None,
    modified_time: str | None = None,
    content_hash: str | None = None,
    row_count: int | None = None,
    status: str = "registered",
    db_path: str | Path | None = None,
) -> int:
    path = Path(file_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO dataset_files (
                dataset_id, file_name, file_path, file_size, modified_time,
                content_hash, row_count, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                path.name,
                str(path),
                file_size,
                modified_time,
                content_hash,
                row_count,
                status,
                utc_now(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def update_dataset_row_count(
    dataset_id: int,
    row_count: int,
    *,
    status: str = "ingested",
    db_path: str | Path | None = None,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE datasets
            SET row_count = ?, status = ?, updated_at = ?
            WHERE dataset_id = ?
            """,
            (row_count, status, utc_now(), dataset_id),
        )
        conn.commit()


def get_duplicate_dataset_file(
    *,
    file_name: str,
    file_path: str,
    file_size: int,
    modified_time: str,
    content_hash: str,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM dataset_files
            WHERE file_name = ?
              AND file_path = ?
              AND file_size = ?
              AND modified_time = ?
              AND content_hash = ?
            """,
            (file_name, file_path, file_size, modified_time, content_hash),
        ).fetchone()

    return dict(row) if row is not None else None
