from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.common.paths import GENERATED_CONFIG_DIR
from src.db import get_connection, get_dataset, init_db, insert_feature_config, table_exists, utc_now


LOW_CARDINALITY_UNIQUE_MAX = 20
LOW_CARDINALITY_RATIO_MAX = 0.05
MODEL_MISSING_MAX_RATIO = 0.40
CONTEXT_MISSING_MAX_RATIO = 0.80
IDENTIFIER_UNIQUE_RATIO_MIN = 0.95
SAMPLE_VALUE_LIMIT = 5
IDENTIFIER_NAME_PATTERN = re.compile(
    r"(^|_|\s)(id|uuid|guid|index|row|file|filename|path|hash|key)(_|$|\s)"
)
WEAK_SEQUENCE_NAME_PATTERN = re.compile(
    r"(^|_|\s)(seq|sequence|counter|count|number|num|order|ordinal)(_|$|\s)"
)
CONTINUOUS_TELEMETRY_NAME_PATTERN = re.compile(
    r"(^|_|\s)(rpm|voltage|current|temperature|pressure|angle|error|rate|power|torque|speed)(_|$|\s)"
)


@dataclass(frozen=True)
class ColumnProfile:
    column_name: str
    sanitized_name: str
    inferred_type: str
    role: str
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    sample_values: list[str]
    reasoning: str


def profile_dataset(
    dataset_id: int,
    *,
    db_path: str | Path | None = None,
    generated_config_dir: str | Path = GENERATED_CONFIG_DIR,
) -> dict[str, Any]:
    """Profile a raw ingested dataset and generate column metadata plus a feature config."""
    init_db(db_path)
    dataset = get_dataset(dataset_id, db_path=db_path)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    if json.loads(dataset["metadata_json"] or "{}").get("storage_version") == 2:
        from src.ingestion.store import profile_dataset as profile_canonical, load_dataset
        result = profile_canonical(dataset_id, db_path=db_path)
        canonical = load_dataset(dataset_id, db_path=db_path)
        path = Path(generated_config_dir) / f"feature_config_dataset_{dataset_id}.yaml"
        overrides = _load_user_overrides(path)
        aliases = {"numeric": "numeric_telemetry_feature", "categorical": "categorical_feature"}
        columns = [] if canonical["manifest"].get("ordering") == "row" else [
            {"column_name": "timestamp", "sanitized_name": "timestamp", "role": "timestamp"}]
        columns.extend({"column_name": c["source_name"], "sanitized_name": c["name"],
                        "role": overrides.get(c["name"], aliases.get(c["role"], c["role"]))}
                       for c in canonical["columns"])
        config = {"dataset_id": dataset_id, "version": "2", "columns": columns}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        insert_feature_config(dataset_id, path, config, version="2", db_path=db_path)
        return {**result, "column_count": len(columns), "config_path": str(path), "feature_config_path": str(path)}

    raw_table = f"raw_dataset_{dataset_id}"
    with get_connection(db_path) as conn:
        if not table_exists(conn, raw_table):
            raise ValueError(f"Raw table {raw_table} does not exist. Run ingestion first.")

        rows = [dict(row) for row in conn.execute(f"SELECT * FROM {_quote_identifier(raw_table)}").fetchall()]
        if not rows:
            raise ValueError(f"Raw table {raw_table} has no rows to profile.")

        original_names = _load_original_column_names(conn, dataset_id)

    config_path = Path(generated_config_dir) / f"feature_config_dataset_{dataset_id}.yaml"
    overrides = _load_user_overrides(config_path)
    data_columns = [
        column
        for column in rows[0].keys()
        if column not in {"row_id", "dataset_id", "source_file_id"}
    ]

    profiles = [
        _profile_column(
            sanitized_name=column,
            original_name=original_names.get(column, column),
            values=[row[column] for row in rows],
            total_rows=len(rows),
            override_role=overrides.get(column) or overrides.get(original_names.get(column, column)),
        )
        for column in data_columns
    ]

    _write_column_metadata(dataset_id, profiles, db_path=db_path)

    feature_config = _build_feature_config(dataset_id, raw_table, profiles)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(feature_config, file, sort_keys=False)

    insert_feature_config(
        dataset_id,
        config_path,
        feature_config,
        version=feature_config["version"],
        db_path=db_path,
    )

    return {
        "dataset_id": dataset_id,
        "raw_table": raw_table,
        "row_count": len(rows),
        "column_count": len(profiles),
        "config_path": str(config_path),
    }


def _profile_column(
    *,
    sanitized_name: str,
    original_name: str,
    values: list[Any],
    total_rows: int,
    override_role: str | None,
) -> ColumnProfile:
    """Compute statistics, infer type, and assign a deterministic role for one column."""
    normalized = [_normalize_value(value) for value in values]
    present = [value for value in normalized if value is not None]
    missing_count = total_rows - len(present)
    missing_ratio = missing_count / total_rows if total_rows else 0.0
    unique_values = set(present)
    unique_count = len(unique_values)
    unique_ratio = unique_count / total_rows if total_rows else 0.0
    sample_values = _sample_values(present)
    inferred_type = _infer_type(original_name, present, unique_count, unique_ratio)
    is_timestamp = inferred_type == "timestamp"
    is_identifier = _is_identifier_like(
        original_name,
        sanitized_name,
        inferred_type,
        present,
        unique_ratio,
        total_rows,
    )
    is_low_cardinality = unique_count <= LOW_CARDINALITY_UNIQUE_MAX or unique_ratio <= LOW_CARDINALITY_RATIO_MAX

    role, reasoning = _assign_role(
        inferred_type=inferred_type,
        override_role=override_role,
        is_timestamp=is_timestamp,
        is_identifier=is_identifier,
        is_low_cardinality=is_low_cardinality,
        missing_ratio=missing_ratio,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
    )

    return ColumnProfile(
        column_name=original_name,
        sanitized_name=sanitized_name,
        inferred_type=inferred_type,
        role=role,
        missing_ratio=round(missing_ratio, 6),
        unique_count=unique_count,
        unique_ratio=round(unique_ratio, 6),
        sample_values=sample_values,
        reasoning=reasoning,
    )


def _assign_role(
    *,
    inferred_type: str,
    override_role: str | None,
    is_timestamp: bool,
    is_identifier: bool,
    is_low_cardinality: bool,
    missing_ratio: float,
    unique_count: int,
    unique_ratio: float,
) -> tuple[str, str]:
    """Assign the downstream column role using the design document's priority order."""
    if override_role:
        return override_role, f"User override applied: role set to {override_role}."
    if is_timestamp:
        return "timestamp", "Detected as timestamp before identifier or feature assignment."
    if is_identifier:
        return "identifier", "Detected as identifier-like by column name or near-unique values."
    if missing_ratio > MODEL_MISSING_MAX_RATIO:
        if inferred_type == "text" and unique_count > 0:
            return "context_event_note", "Sparse text column has meaningful non-null values retained for context."
        if missing_ratio <= CONTEXT_MISSING_MAX_RATIO:
            return "context", "Missing ratio exceeds LSTM threshold but remains usable as context."
        return "excluded", "Missing ratio exceeds context retention threshold."
    if inferred_type == "numeric":
        return "numeric_telemetry_feature", "Numeric column with acceptable missing ratio."
    if inferred_type == "categorical" and is_low_cardinality:
        return (
            "categorical_feature",
            f"Low-cardinality categorical column ({unique_count} unique, ratio {unique_ratio:.3f}).",
        )
    if missing_ratio <= CONTEXT_MISSING_MAX_RATIO and inferred_type in {"text", "categorical"}:
        return "context", "Free-text or high-cardinality field retained for diagnostic context."
    return "excluded", "Column does not meet modeling or context retention rules."


def _infer_type(
    column_name: str,
    present: list[str],
    unique_count: int,
    unique_ratio: float,
) -> str:
    """Infer a broad data type from column name hints, parse success, and cardinality."""
    if not present:
        return "text"
    if _looks_timestamp_column(column_name) and _timestamp_ratio(present) >= 0.60:
        return "timestamp"
    if _numeric_ratio(present) >= 0.95:
        return "numeric"
    if _timestamp_ratio(present) >= 0.95:
        return "timestamp"
    if _looks_like_free_text(present):
        return "text"
    if unique_count <= LOW_CARDINALITY_UNIQUE_MAX or unique_ratio <= LOW_CARDINALITY_RATIO_MAX:
        return "categorical"
    return "text"


def _looks_timestamp_column(column_name: str) -> bool:
    """Return whether a column name strongly suggests timestamp-like data."""
    return bool(re.search(r"(^|_|\s)(time|timestamp|date|datetime)(_|$|\s)", column_name.lower()))


def _is_identifier_like(
    original_name: str,
    sanitized_name: str,
    inferred_type: str,
    present: list[str],
    unique_ratio: float,
    total_rows: int,
) -> bool:
    """Detect identifier columns by clear names or numeric sequence-like behavior."""
    combined = f"{original_name} {sanitized_name}".lower()
    has_identifier_name = bool(IDENTIFIER_NAME_PATTERN.search(combined))
    if has_identifier_name:
        return True

    near_unique = total_rows >= 2 and unique_ratio >= IDENTIFIER_UNIQUE_RATIO_MIN
    if not near_unique:
        return False

    if inferred_type == "numeric":
        if CONTINUOUS_TELEMETRY_NAME_PATTERN.search(combined):
            return False
        return _looks_like_numeric_sequence_identifier(present, combined)

    return True


def _looks_like_numeric_sequence_identifier(values: list[str], combined_name: str) -> bool:
    """Return whether numeric values behave like an integer row counter or sequence."""
    numeric_values = _parse_numeric_values(values)
    if len(numeric_values) < 2:
        return False

    integer_like_ratio = sum(1 for value in numeric_values if _is_integer_like(value)) / len(numeric_values)
    if integer_like_ratio < 0.95:
        return False

    diffs = [right - left for left, right in zip(numeric_values, numeric_values[1:])]
    if not diffs:
        return False

    monotonic_ratio = sum(1 for diff in diffs if diff > 0) / len(diffs)
    monotonic_ratio = max(monotonic_ratio, sum(1 for diff in diffs if diff < 0) / len(diffs))
    if monotonic_ratio < 0.95:
        return False

    rounded_diffs = [round(diff, 6) for diff in diffs if diff != 0]
    if not rounded_diffs:
        return False

    most_common_diff_count = max(rounded_diffs.count(diff) for diff in set(rounded_diffs))
    constant_diff_ratio = most_common_diff_count / len(rounded_diffs)
    mostly_step_one = sum(1 for diff in rounded_diffs if abs(abs(diff) - 1.0) < 1e-6) / len(rounded_diffs) >= 0.95
    weak_sequence_name = bool(WEAK_SEQUENCE_NAME_PATTERN.search(combined_name))
    return constant_diff_ratio >= 0.95 and (mostly_step_one or weak_sequence_name)


def _parse_numeric_values(values: list[str]) -> list[float]:
    """Parse strings as floats, skipping any value that cannot be parsed."""
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except ValueError:
            continue
    return parsed


def _is_integer_like(value: float) -> bool:
    """Return whether a numeric value is close enough to an integer."""
    return abs(value - round(value)) < 1e-9


def _looks_like_free_text(values: list[str]) -> bool:
    """Return whether sampled values look like prose or long human-readable text."""
    if not values:
        return False
    sample = values[: min(len(values), 100)]
    text_like_count = sum(
        1
        for value in sample
        if bool(re.search(r"\s", value.strip())) or len(value.strip()) > 40
    )
    return text_like_count / len(sample) >= 0.50


def _numeric_ratio(values: list[str]) -> float:
    """Return the fraction of non-missing values that can be parsed as floats."""
    if not values:
        return 0.0
    numeric_count = 0
    for value in values:
        try:
            float(value)
        except ValueError:
            continue
        numeric_count += 1
    return numeric_count / len(values)


def _timestamp_ratio(values: list[str]) -> float:
    """Return the fraction of sampled values that can be parsed as timestamps."""
    if not values:
        return 0.0
    sample = values[: min(len(values), 100)]
    parsed_count = sum(1 for value in sample if _is_timestamp_value(value))
    return parsed_count / len(sample)


def _is_timestamp_value(value: str) -> bool:
    """Return whether a single string can be parsed as a supported timestamp format."""
    candidate = value.strip()
    if not candidate:
        return False
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
            datetime.strptime(candidate, date_format)
            return True
        except ValueError:
            pass
    try:
        datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _normalize_value(value: Any) -> str | None:
    """Normalize raw SQLite text values and convert common missing tokens to None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "null", "none", "nan"}:
        return None
    return text


def _sample_values(values: list[str]) -> list[str]:
    """Return the first distinct present values for human-readable config examples."""
    samples: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        samples.append(value)
        if len(samples) == SAMPLE_VALUE_LIMIT:
            break
    return samples


def _load_original_column_names(conn, dataset_id: int) -> dict[str, str]:
    """Load the original CSV header for each sanitized column name in a dataset."""
    rows = conn.execute(
        """
        SELECT column_name, sanitized_name
        FROM dataset_columns
        WHERE dataset_id = ?
        """,
        (dataset_id,),
    ).fetchall()
    return {
        row["sanitized_name"]: row["column_name"]
        for row in rows
        if row["sanitized_name"] is not None
    }


def _write_column_metadata(
    dataset_id: int,
    profiles: list[ColumnProfile],
    *,
    db_path: str | Path | None,
) -> None:
    """Upsert profiling results into the dataset_columns metadata table."""
    now = utc_now()
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO dataset_columns (
                dataset_id, column_name, sanitized_name, inferred_type, role,
                missing_ratio, unique_count, unique_ratio, example_values_json,
                reasoning, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, column_name) DO UPDATE SET
                sanitized_name = excluded.sanitized_name,
                inferred_type = excluded.inferred_type,
                role = excluded.role,
                missing_ratio = excluded.missing_ratio,
                unique_count = excluded.unique_count,
                unique_ratio = excluded.unique_ratio,
                example_values_json = excluded.example_values_json,
                reasoning = excluded.reasoning,
                updated_at = excluded.updated_at
            """,
            (
                (
                    dataset_id,
                    profile.column_name,
                    profile.sanitized_name,
                    profile.inferred_type,
                    profile.role,
                    profile.missing_ratio,
                    profile.unique_count,
                    profile.unique_ratio,
                    json.dumps(profile.sample_values),
                    profile.reasoning,
                    now,
                    now,
                )
                for profile in profiles
            ),
        )
        conn.commit()


def _build_feature_config(
    dataset_id: int,
    raw_table: str,
    profiles: list[ColumnProfile],
) -> dict[str, Any]:
    """Build the YAML-serializable feature configuration from column profiles."""
    return {
        "dataset_id": dataset_id,
        "raw_table": raw_table,
        "version": "1",
        "generated_at": utc_now(),
        "thresholds": {
            "low_cardinality_unique_count": LOW_CARDINALITY_UNIQUE_MAX,
            "low_cardinality_unique_ratio": LOW_CARDINALITY_RATIO_MAX,
            "model_missing_max_ratio": MODEL_MISSING_MAX_RATIO,
            "context_missing_max_ratio": CONTEXT_MISSING_MAX_RATIO,
            "identifier_unique_ratio": IDENTIFIER_UNIQUE_RATIO_MIN,
        },
        "columns": [asdict(profile) for profile in profiles],
    }


def _load_user_overrides(config_path: Path) -> dict[str, str]:
    """Read optional user role overrides from a previously generated feature config."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file) or {}
    overrides: dict[str, str] = {}

    for column_name, role in (parsed.get("overrides") or {}).items():
        if role:
            overrides[str(column_name)] = str(role)

    for column in parsed.get("columns") or []:
        if not isinstance(column, dict):
            continue
        override_role = column.get("override_role") or column.get("user_role")
        if override_role:
            for key in ("sanitized_name", "column_name"):
                if column.get(key):
                    overrides[str(column[key])] = str(override_role)

    return overrides


def _quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier while escaping embedded double quotes."""
    return '"' + identifier.replace('"', '""') + '"'
