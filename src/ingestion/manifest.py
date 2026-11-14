"""Versioned, explicit contracts for structured telemetry imports."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROLE_KEYS = {
    "numeric_features": "numeric", "categorical_features": "categorical",
    "context_columns": "context", "label_columns": "label",
}
RESERVED = {"row_id", "timestamp", "timestamp_ns", "entity_id", "_seq", "_channel"}


def is_numeric_literal(value: Any) -> bool:
    """Identify numeric timestamps without treating datetime objects as numbers."""
    if isinstance(value, bool):
        return True
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def is_label(name: str) -> bool:
    normalized = re.sub(r"\W+", "_", name.strip().lower())
    if normalized in {
        "label", "labels", "target", "anomaly", "anomaly_type",
        "ground_truth", "groundtruth", "is_outlier", "outlier",
        "fault_label", "fault_code", "class_label", "annotation",
    }:
        return True
    return normalized.startswith(("is_anomaly", "is_outlier_", "label_", "target_", "anomaly_", "fault_label_")) or normalized.endswith(("_label", "_target", "_ground_truth"))


def column_names(headers: list[str]) -> dict[str, str]:
    """Build stable identifiers when sanitized names collide or columns move."""
    bases = {}
    for header in headers:
        base = re.sub(r"[^a-z0-9_]+", "_", header.strip().lower()).strip("_") or "column"
        if base[0].isdigit() or base in RESERVED:
            base = "source_" + base
        bases[header] = base
    used = set(RESERVED)
    result = {}
    for header in sorted(headers):
        base = bases[header]
        candidate = base
        if list(bases.values()).count(base) > 1 or candidate in used:
            candidate = base + "_" + hashlib.sha256(header.encode()).hexdigest()[:10]
        while candidate in used:
            candidate += "_"
        used.add(candidate)
        result[header] = candidate
    return result


def load_manifest(manifest: dict | str | Path | None) -> dict[str, Any]:
    if manifest is None:
        data = {}
    elif isinstance(manifest, dict):
        data = json.loads(json.dumps(manifest, allow_nan=False))
    else:
        path = Path(manifest)
        with path.open(encoding="utf-8") as stream:
            if path.suffix.lower() in {".yaml", ".yml"}:
                import yaml
                data = yaml.safe_load(stream)
            else:
                data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("Manifest must be an object")
        # Boundaries supplied through a reusable mapping file are deliberate,
        # including a reviewed mapping originally suggested by inspection.
        if "splits" in data:
            data["_splits_explicit"] = True
    if data.get("version", 1) != 1:
        raise ValueError("Only manifest version 1 is supported")
    data.setdefault("version", 1)
    data.setdefault("timezone", "UTC")
    data.setdefault("layout", "wide")
    data.setdefault("duplicate_policy", "error")
    if data.get("ordering", "timestamp") not in {"timestamp", "row"}:
        raise ValueError("ordering must be timestamp or row")
    if data.get("ordering") == "row":
        if data.get("timestamp") is not None or data.get("layout") != "wide":
            raise ValueError("Row ordering requires a wide source without a timestamp mapping")
        if any(data.get(key) is not None for key in ("epoch_unit", "cadence_seconds", "max_gap_seconds", "annotations_path")):
            raise ValueError("Row ordering cannot use calendar-time units, gap durations or interval annotations")
    if data["layout"] not in {"wide", "long"}:
        raise ValueError("layout must be wide or long")
    if data["duplicate_policy"] not in {"error", "first", "last"}:
        raise ValueError("duplicate_policy must be error, first, or last")
    if data.get("epoch_unit") not in {None, "s", "ms", "us", "ns"}:
        raise ValueError("epoch_unit must be s, ms, us, or ns")
    for key in ("cadence_seconds", "max_gap_seconds"):
        if key in data and (not isinstance(data[key], (int, float)) or not math.isfinite(data[key]) or data[key] <= 0):
            raise ValueError(f"{key} must be a positive finite number")
    for key in (*ROLE_KEYS, "entity_columns"):
        if key in data and (not isinstance(data[key], list) or any(not isinstance(x, str) for x in data[key]) or len(set(data[key])) != len(data[key])):
            raise ValueError(f"{key} must be a list of distinct column names")
    units = data.get("units", {})
    if not isinstance(units, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in units.items()):
        raise ValueError("units must map source column or channel names to unit strings")
    # Raw adapters must choose their own prescribed split before generic defaults.
    if "splits" not in data:
        return data
    splits = data["splits"]
    if not isinstance(splits, dict):
        raise ValueError("splits must be an object")
    if data.get("ordering") == "row" and set(splits) != {"fractions"}:
        raise ValueError("Row ordering requires fractional row partitions")
    if "fractions" in splits:
        values = splits["fractions"]
        if "train_end" in splits or "validation_end" in splits or "test_start" in splits:
            raise ValueError("Choose fractions or explicit split boundaries")
        if not isinstance(values, list) or len(values) != 3 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values) or not math.isclose(sum(values), 1.0):
            raise ValueError("splits.fractions requires three positive fractions summing to 1")
    elif "test_start" in splits:
        fraction = splits.get("train_fraction", .85)
        if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("splits.train_fraction must be between 0 and 1")
    elif not {"train_end", "validation_end"}.issubset(splits):
        raise ValueError("Explicit splits require train_end and validation_end")
    data["splits"] = splits
    return data


def resolve_manifest(headers: list[str], sample: list[dict], manifest: dict | None = None) -> tuple[dict, list[dict], list[str]]:
    resolved = load_manifest(manifest)
    resolved.setdefault("_splits_explicit", "splits" in resolved)
    resolved.setdefault("splits", {"fractions": [.7, .15, .15]})
    issues: list[str] = []
    if len(set(headers)) != len(headers) or any(not h.strip() for h in headers):
        raise ValueError("Column names must be nonblank and unique")
    if any(h.startswith("__comet_") for h in headers):
        raise ValueError("Column names beginning __comet_ are reserved")
    timestamp = resolved.get("timestamp")
    if resolved.get("ordering") == "row":
        resolved["timestamp"] = None
    if timestamp is None and resolved.get("ordering") != "row":
        candidates = [h for h in headers if h.strip().lower() in {"timestamp", "datetime", "time", "date_time"}]
        if len(candidates) == 1:
            timestamp = candidates[0]
            resolved["timestamp"] = timestamp
        else:
            issues.append("Specify exactly one timestamp column in the manifest")
    elif timestamp is not None and timestamp not in headers:
        raise ValueError(f"Timestamp column {timestamp!r} is absent")
    assigned: dict[str, str] = {}
    for key, role in ROLE_KEYS.items():
        for name in resolved.get(key, []):
            if name not in headers:
                raise ValueError(f"Mapped column {name!r} is absent")
            if name in assigned:
                raise ValueError(f"Column {name!r} has multiple roles")
            if is_label(name) and role != "label":
                raise ValueError(f"Annotation column {name!r} cannot be a model input or context")
            assigned[name] = role
    entities = resolved.setdefault("entity_columns", [])
    structural = {timestamp, *entities}
    if resolved["layout"] == "long":
        for key, fallback in (("channel_column", "channel"), ("value_column", "value")):
            value = resolved.setdefault(key, fallback)
            if value not in headers:
                issues.append(f"Specify an existing {key}")
            structural.add(value)
        if resolved.get("numeric_features") or resolved.get("categorical_features"):
            raise ValueError("Long layout derives numeric features from channel values. Leave numeric_features/categorical_features unmapped")
    structural_names = [timestamp, *entities]
    if resolved["layout"] == "long":
        structural_names += [resolved["channel_column"], resolved["value_column"]]
    if len(set(structural_names)) != len(structural_names):
        raise ValueError("Timestamp, entity, channel, and value mappings must identify distinct columns")
    for name in entities:
        if name not in headers:
            raise ValueError(f"Entity column {name!r} is absent")
        if is_label(name):
            raise ValueError("Annotations cannot identify an entity")
    if timestamp is not None and is_label(timestamp):
        raise ValueError("Annotations cannot provide timestamps")
    if structural.intersection(assigned):
        raise ValueError("Timestamp, entity, channel, and value columns cannot also have feature roles")
    inferred_categorical = set(resolved.get("_inferred_categorical", []))
    for name in headers:
        if name in structural or name in assigned:
            continue
        if is_label(name):
            assigned[name] = "label"
        else:
            values = [row.get(name) for row in sample if row.get(name) not in (None, "")]
            if not values:
                issues.append(f"Column {name!r} is empty in the inspection sample. Specify its role")
                continue
            # Source order can differ from time order. Use the initial type
            # for validation and require an explicit categorical role if
            # later records mix numeric and categorical values.
            numeric = is_numeric_literal(values[0])
            assigned[name] = "numeric" if numeric else "categorical"
            if not numeric:
                inferred_categorical.add(name)
    gap_column = resolved.get("gap_column")
    if gap_column is not None and (not isinstance(gap_column, str) or assigned.get(gap_column) != "context"):
        raise ValueError("gap_column must name an existing column mapped as context_columns")
    if resolved["layout"] == "wide":
        for name in resolved.get("units", {}):
            if assigned.get(name) != "numeric":
                raise ValueError(f"Unit mapping {name!r} must identify a numeric feature")
    if timestamp and not resolved.get("epoch_unit"):
        for row in sample:
            value = row.get(timestamp)
            if value in (None, ""):
                continue
            if is_numeric_literal(value):
                issues.append("Numeric timestamps require an explicit epoch_unit")
                break
    for key, role in ROLE_KEYS.items():
        resolved[key] = [name for name in headers if assigned.get(name) == role]
    resolved["_inferred_categorical"] = sorted(inferred_categorical)
    names = column_names(list(assigned))
    columns = [{"name": names[name], "source_name": name, "role": role,
                "dtype": "float64" if role == "numeric" else "string",
                **({"unit": resolved["units"][name]} if name in resolved.get("units", {}) else {})}
               for name, role in assigned.items()]
    return resolved, columns, issues
