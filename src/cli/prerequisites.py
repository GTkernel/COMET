from __future__ import annotations

from pathlib import Path
from src.cleaning.prepare_dataset import resolve_feature_config_path
from src.common.paths import RAG_DOCS_DIR, RAG_VECTOR_DIR
from src.db import get_anomaly, get_connection, get_dataset, get_model, init_db, table_exists
from src.rag.document_ingestion import _VectorStore


def require_raw_dataset(dataset_id: int, *, db_path: str | Path | None = None) -> None:
    init_db(db_path)
    table_name = f"raw_dataset_{dataset_id}"
    with get_connection(db_path) as conn:
        dataset = get_dataset(dataset_id, db_path=db_path)
        if dataset is None or not table_exists(conn, table_name):
            raise ValueError(
                f"profile requires {table_name}, but it was not found. "
                "Run: python -m src.cli ingest"
            )


def require_profile_output(
    dataset_id: int,
    feature_config: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    table_name = f"raw_dataset_{dataset_id}"
    config_path = resolve_feature_config_path(dataset_id, feature_config)
    with get_connection(db_path) as conn:
        if get_dataset(dataset_id, db_path=db_path) is None or not table_exists(conn, table_name):
            raise ValueError(
                f"prepare requires {table_name}, but it was not found. "
                "Run: python -m src.cli ingest"
            )

        profiled_column = conn.execute(
            """
            SELECT 1
            FROM dataset_columns
            WHERE dataset_id = ?
              AND role IS NOT NULL
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if profiled_column is None:
            raise ValueError(
                "prepare requires profile output in dataset_columns. "
                f"Run: python -m src.cli profile --dataset-id {dataset_id}"
            )

    if not config_path.exists():
        if feature_config is None:
            raise ValueError(f"Feature config for dataset {dataset_id} not found. Run profiling first.")
        raise ValueError(f"prepare requires feature config {config_path}, but it was not found.")


def require_model_ready_dataset(dataset_id: int, *, db_path: str | Path | None = None) -> None:
    init_db(db_path)
    table_name = f"model_ready_dataset_{dataset_id}"
    with get_connection(db_path) as conn:
        if get_dataset(dataset_id, db_path=db_path) is None or not table_exists(conn, table_name):
            raise ValueError(
                f"train requires {table_name}, but it was not found. "
                f"Run: python -m src.cli prepare --dataset-id {dataset_id}"
            )


def require_trained_model(
    dataset_id: int,
    model_id: int,
    *,
    db_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    model_row = get_model(model_id, db_path=db_path)
    if model_row is None:
        raise ValueError(
            "trained model metadata is required, but no model record was found for "
            f"model_id={model_id}. Run: python -m src.cli train --dataset-id {dataset_id}"
        )
    if int(model_row["dataset_id"]) != dataset_id:
        raise ValueError(
            f"detect requires a model trained for dataset {dataset_id}, but model {model_id} "
            f"belongs to dataset {model_row['dataset_id']}."
        )

    model_path = Path(model_row["model_path"])
    if not model_path.exists():
        raise ValueError(
            f"trained model file {model_path} is required, but it was not found. "
            f"Run: python -m src.cli train --dataset-id {dataset_id}"
        )

    require_model_ready_dataset(dataset_id, db_path=db_path)


def require_documents_directory(documents_dir: str | Path = RAG_DOCS_DIR) -> None:
    docs_path = Path(documents_dir)
    if not docs_path.exists() or not docs_path.is_dir():
        raise ValueError(
            f"ingest-docs requires mission document directory {docs_path}, but it was not found. "
            "Create the directory and add .txt, .md, or .pdf files, then run: python -m src.cli ingest-docs"
        )


def require_diagnosis_inputs(
    anomaly_id: str,
    vector_store_dir: str | Path = RAG_VECTOR_DIR,
    *,
    db_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    if get_anomaly(anomaly_id, db_path=db_path) is None:
        raise ValueError(
            "diagnose requires an anomaly record, but none was found for "
            f"anomaly_id={anomaly_id}. Run: python -m src.cli detect --dataset-id <id> --model-id <id>"
        )

    vector_path = Path(vector_store_dir)
    if not vector_path.exists() or not _VectorStore(vector_path).has_documents():
        raise ValueError(
            "diagnose requires ingested mission documents in a local vector store. "
            "Run: python -m src.cli ingest-docs"
        )
