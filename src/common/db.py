from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from src.db import get_connection, init_db,  utc_now
from src.common.paths import DB_PATH


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator:
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def initialize_schema() -> None:
    init_db()


def upsert_feature_config(dataset_id: int, config_path: str, config: dict[str, Any]) -> None:
    payload = json.dumps(config)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO feature_configs(dataset_id, config_path, config_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (dataset_id, config_path, payload, utc_now()),
        )
        conn.commit()
