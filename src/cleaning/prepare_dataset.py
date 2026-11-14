from __future__ import annotations

import json
from contextlib import nullcontext
import logging
import math
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
from src.common.paths import GENERATED_CONFIG_DIR, LOGS_DIR
from src.db import get_connection, get_dataset, get_feature_config, init_db, table_exists


SYSTEM_COLUMNS = {"row_id", "dataset_id", "source_file_id"}


def prepare_dataset(
    dataset_id: int,
    feature_config: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    generated_config_dir: str | Path = GENERATED_CONFIG_DIR,
    logs_dir: str | Path = LOGS_DIR,
) -> dict[str, Any]:
    """Create model-ready and context-ready tables using a dataset feature config.

    If feature_config is omitted, the generated config path is resolved as
    config/generated/feature_config_dataset_<dataset_id>.yaml.
    """
    init_db(db_path)
    logger = _build_logger(Path(logs_dir))
    dataset = get_dataset(dataset_id, db_path=db_path)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    if json.loads(dataset["metadata_json"] or "{}").get("storage_version") == 2:
        from src.ingestion.store import set_feature_roles
        path = resolve_feature_config_path(dataset_id, feature_config, generated_config_dir=generated_config_dir)
        columns = None
        proposed = None
        if feature_config is not None or path.exists():
            config = _load_feature_config(dataset_id, path, db_path=db_path)
            columns = _config_columns(config)
            proposed = set_feature_roles(dataset_id, columns, db_path=db_path, validate_only=True)
        from src.modeling.pipeline import prepare
        result = prepare(dataset_id, db_path=db_path, _dataset=proposed)
        if columns is not None:
            set_feature_roles(dataset_id, columns, db_path=db_path)
        return {**result, "config_path": str(path) if columns is not None else None}

    config_path = resolve_feature_config_path(dataset_id, feature_config, generated_config_dir=generated_config_dir)
    if feature_config is None and not config_path.exists():
        raise ValueError(f"Feature config for dataset {dataset_id} not found. Run profiling first.")

    config = _load_feature_config(dataset_id, config_path, db_path=db_path)
    _warn_if_config_dataset_mismatch(dataset_id, config, config_path, logger)
    columns = _config_columns(config)
    from src.ingestion.store import set_feature_roles
    set_feature_roles(dataset_id, columns, db_path=db_path, validate_only=True)
    raw_table = str(config.get("raw_table") or f"raw_dataset_{dataset_id}")
    model_table = f"model_ready_dataset_{dataset_id}"
    context_table = f"context_ready_dataset_{dataset_id}"

    with get_connection(db_path) as conn:
        if not table_exists(conn, raw_table):
            raise ValueError(f"Raw table {raw_table} does not exist. Run ingestion first.")
        raw_rows = [
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM {_quote_identifier(raw_table)} ORDER BY row_id"
            ).fetchall()
        ]
        if not raw_rows:
            raise ValueError(f"Raw table {raw_table} has no rows to prepare.")

    timestamp_column = _first_column_with_role(columns, "timestamp")
    numeric_columns = [column for column in columns if column["role"] == "numeric_telemetry_feature"]
    categorical_columns = [column for column in columns if column["role"] == "categorical_feature"]
    context_columns = [
        column
        for column in columns
        if column["role"] in {"context", "context_event_note", "identifier", "categorical_feature"}
    ]

    timestamps = _standardized_timestamps(raw_rows, timestamp_column)
    numeric_features = {
        column["sanitized_name"]: _normalize_numeric_series(
            [_normalize_missing(row.get(column["sanitized_name"])) for row in raw_rows]
        )
        for column in numeric_columns
    }
    categorical_features = {
        column["sanitized_name"]: _one_hot_encode(
            [_normalize_missing(row.get(column["sanitized_name"])) for row in raw_rows],
            column["sanitized_name"],
        )
        for column in categorical_columns
    }

    logger.info("Preparing dataset_id=%s from %s", dataset_id, raw_table)
    for column in numeric_columns:
        logger.info("Numeric feature normalized: %s", column["sanitized_name"])
    for column in categorical_columns:
        encoded_names = list(categorical_features[column["sanitized_name"]].keys())
        logger.info("Categorical feature one-hot encoded: %s -> %s", column["sanitized_name"], encoded_names)
    for column in columns:
        if column["role"] == "excluded":
            logger.info("Excluded column dropped: %s", column["sanitized_name"])

    with get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _write_model_ready_table(
            dataset_id=dataset_id, table_name=model_table, raw_rows=raw_rows,
            timestamps=timestamps, numeric_features=numeric_features,
            categorical_features=categorical_features, db_path=db_path, connection=conn)
        _write_context_ready_table(
            dataset_id=dataset_id, table_name=context_table, raw_rows=raw_rows,
            timestamps=timestamps, context_columns=context_columns, db_path=db_path, connection=conn)
        set_feature_roles(dataset_id, columns, db_path=db_path, connection=conn)

    logger.info(
        "Prepared dataset_id=%s model_table=%s context_table=%s rows=%s",
        dataset_id,
        model_table,
        context_table,
        len(raw_rows),
    )
    return {
        "config_path": str(config_path),
        "dataset_id": dataset_id,
        "model_table": model_table,
        "context_table": context_table,
        "row_count": len(raw_rows),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
    }


def resolve_feature_config_path(
    dataset_id: int,
    feature_config: str | Path | None = None,
    *,
    generated_config_dir: str | Path = GENERATED_CONFIG_DIR,
) -> Path:
    """Return the explicit feature config path or the default generated path for a dataset."""
    if feature_config is not None:
        return Path(feature_config)
    return Path(generated_config_dir) / f"feature_config_dataset_{dataset_id}.yaml"


def _warn_if_config_dataset_mismatch(
    dataset_id: int,
    config: dict[str, Any],
    config_path: Path,
    logger: logging.Logger,
) -> None:
    """Warn when an explicitly supplied config appears to belong to another dataset."""
    config_dataset_id = config.get("dataset_id")
    if config_dataset_id is None:
        return
    try:
        parsed_dataset_id = int(config_dataset_id)
    except (TypeError, ValueError):
        return
    if parsed_dataset_id == dataset_id:
        return

    message = (
        f"Feature config {config_path} is for dataset {parsed_dataset_id}, "
        f"but prepare was requested for dataset {dataset_id}."
    )
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _load_feature_config(
    dataset_id: int,
    feature_config_path: Path,
    *,
    db_path: str | Path | None,
) -> dict[str, Any]:
    if feature_config_path.exists():
        text = feature_config_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text) or {}
        if parsed.get("columns"):
            return parsed

    stored = get_feature_config(dataset_id, feature_config_path, db_path=db_path)
    if stored is None:
        stored = get_feature_config(dataset_id, db_path=db_path)
    if stored is not None:
        return json.loads(stored["config_json"])

    raise ValueError(f"Feature config not found or unreadable: {feature_config_path}")


def _config_columns(config: dict[str, Any]) -> list[dict[str, Any]]:
    columns = []
    for column in config.get("columns") or []:
        if not isinstance(column, dict):
            continue
        sanitized_name = column.get("sanitized_name")
        role = column.get("role")
        if sanitized_name and role:
            columns.append({**column, "sanitized_name": str(sanitized_name), "role": str(role)})
    if not columns:
        raise ValueError("Feature config does not contain usable column decisions.")
    return columns


def _first_column_with_role(columns: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    return next((column for column in columns if column["role"] == role), None)


def _standardized_timestamps(
    raw_rows: list[dict[str, Any]],
    timestamp_column: dict[str, Any] | None,
) -> list[str] | None:
    if timestamp_column is None:
        return None
    column_name = timestamp_column["sanitized_name"]
    return [_standardize_timestamp(row.get(column_name)) for row in raw_rows]


def _standardize_timestamp(value: Any) -> str | None:
    text = _normalize_missing(value)
    if text is None:
        return None
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
    ]
    for date_format in formats:
        try:
            return datetime.strptime(text, date_format).isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return text


def _normalize_numeric_series(values: list[str | None]) -> list[float]:
    numeric_values: list[float | None] = []
    for value in values:
        if value is None:
            numeric_values.append(None)
            continue
        try:
            numeric_values.append(float(value))
        except ValueError:
            numeric_values.append(None)

    filled = _fill_numeric_missing(numeric_values)
    mean = sum(filled) / len(filled)
    variance = sum((value - mean) ** 2 for value in filled) / len(filled)
    stddev = math.sqrt(variance)
    if stddev == 0:
        return [0.0 for _ in filled]
    return [(value - mean) / stddev for value in filled]


def _fill_numeric_missing(values: list[float | None]) -> list[float]:
    if all(value is None for value in values):
        return [0.0 for _ in values]

    filled = values[:]
    known = [index for index, value in enumerate(filled) if value is not None]

    for left, right in zip(known, known[1:]):
        left_value = filled[left]
        right_value = filled[right]
        if left_value is None or right_value is None or right == left + 1:
            continue
        step = (right_value - left_value) / (right - left)
        for index in range(left + 1, right):
            filled[index] = left_value + step * (index - left)

    first_known = known[0]
    for index in range(0, first_known):
        filled[index] = filled[first_known]

    last_known = known[-1]
    for index in range(last_known + 1, len(filled)):
        filled[index] = filled[last_known]

    mean = sum(value for value in filled if value is not None) / len(known)
    return [float(value if value is not None else mean) for value in filled]


def _one_hot_encode(values: list[str | None], column_name: str) -> dict[str, list[int]]:
    categories = sorted({value for value in values if value is not None})
    if any(value is None for value in values):
        categories.append("__missing__")

    encoded: dict[str, list[int]] = {}
    for category in categories:
        output_name = f"{column_name}__{_sanitize_feature_token(category)}"
        encoded[output_name] = [
            1 if (value if value is not None else "__missing__") == category else 0
            for value in values
        ]
    return encoded


def _write_model_ready_table(
    *,
    dataset_id: int,
    table_name: str,
    raw_rows: list[dict[str, Any]],
    timestamps: list[str] | None,
    numeric_features: dict[str, list[float]],
    categorical_features: dict[str, dict[str, list[int]]],
    db_path: str | Path | None,
    connection=None,
) -> None:
    feature_columns: list[tuple[str, str]] = []
    if timestamps is not None:
        feature_columns.append(("timestamp", "TEXT"))
    for column_name in numeric_features:
        feature_columns.append((column_name, "REAL"))
    for encoded_columns in categorical_features.values():
        for column_name in encoded_columns:
            feature_columns.append((column_name, "INTEGER"))

    column_sql = ", ".join(
        [_column_definition("row_id", "INTEGER PRIMARY KEY"), _column_definition("dataset_id", "INTEGER NOT NULL")]
        + [_column_definition(name, sql_type) for name, sql_type in feature_columns]
    )
    insert_columns = ["row_id", "dataset_id", *[name for name, _ in feature_columns]]
    placeholders = ", ".join("?" for _ in insert_columns)
    insert_sql = (
        f"INSERT INTO {_quote_identifier(table_name)} "
        f"({', '.join(_quote_identifier(column) for column in insert_columns)}) "
        f"VALUES ({placeholders})"
    )
    output_rows = []
    for row_index, raw_row in enumerate(raw_rows):
        output = [raw_row["row_id"], dataset_id]
        if timestamps is not None:
            output.append(timestamps[row_index])
        for column_name in numeric_features:
            output.append(numeric_features[column_name][row_index])
        for encoded_columns in categorical_features.values():
            for encoded_values in encoded_columns.values():
                output.append(encoded_values[row_index])
        output_rows.append(output)

    with nullcontext(connection) if connection is not None else get_connection(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")
        conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({column_sql})")
        conn.executemany(insert_sql, output_rows)


def _write_context_ready_table(
    *,
    dataset_id: int,
    table_name: str,
    raw_rows: list[dict[str, Any]],
    timestamps: list[str] | None,
    context_columns: list[dict[str, Any]],
    db_path: str | Path | None,
    connection=None,
) -> None:
    context_column_names = [
        column["sanitized_name"]
        for column in context_columns
        if column["sanitized_name"] not in SYSTEM_COLUMNS
    ]
    output_columns = ["row_id", "dataset_id"]
    if timestamps is not None:
        output_columns.append("timestamp")
    output_columns.extend(context_column_names)

    column_defs = [
        _column_definition("row_id", "INTEGER PRIMARY KEY"),
        _column_definition("dataset_id", "INTEGER NOT NULL"),
    ]
    if timestamps is not None:
        column_defs.append(_column_definition("timestamp", "TEXT"))
    column_defs.extend(_column_definition(column, "TEXT") for column in context_column_names)

    insert_sql = (
        f"INSERT INTO {_quote_identifier(table_name)} "
        f"({', '.join(_quote_identifier(column) for column in output_columns)}) "
        f"VALUES ({', '.join('?' for _ in output_columns)})"
    )
    output_rows = []
    for row_index, raw_row in enumerate(raw_rows):
        output = [raw_row["row_id"], dataset_id]
        if timestamps is not None:
            output.append(timestamps[row_index])
        output.extend(raw_row.get(column) for column in context_column_names)
        output_rows.append(output)

    with nullcontext(connection) if connection is not None else get_connection(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")
        conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({', '.join(column_defs)})")
        conn.executemany(insert_sql, output_rows)


def _build_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("satellite_operator.cleaning")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = (logs_dir / "cleaning.log").resolve()
    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    return logger


def _normalize_missing(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "null", "none", "nan"}:
        return None
    return text


def _sanitize_feature_token(value: str) -> str:
    token = re.sub(r"\W+", "_", str(value).strip().lower()).strip("_")
    if not token:
        token = "value"
    if token[0].isdigit():
        token = f"value_{token}"
    return token


def _column_definition(name: str, sql_type: str) -> str:
    return f"{_quote_identifier(name)} {sql_type}"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
