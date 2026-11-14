"""Train-only feature contracts and causal, bounded-batch transformations.

Structural preparation also runs on ingestion-only installations, so this module
uses NumPy without importing torch. Fitted statistics come from training rows.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PARTITIONS = {"train": 0, "validation": 1, "test": 2}
ROW_DTYPE = np.dtype([
    ("row_id", "<i8"), ("timestamp_ns", "<i8"), ("entity", "<i8"),
    ("partition", "u1"), ("segment", "<i8"), ("excluded", "?"), ("gap", "?"),
])


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def search_index(rows: np.ndarray, field: str, value: int, *, start: int = 0,
                 end: int | None = None, side: str = "left") -> int:
    """Binary search a strided mmap field without NumPy's contiguous input copy."""
    low, high = start, len(rows) if end is None else end
    while low < high:
        middle = (low + high) // 2
        item = rows[middle][field]
        if item < value or (side == "right" and item == value):
            low = middle + 1
        else:
            high = middle
    return low


def _timestamp_ns(value: Any) -> int:
    import pandas as pd
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.value)


def max_gap_ns(manifest: dict) -> int:
    seconds = manifest.get("max_gap_seconds")
    if seconds is None and manifest.get("cadence_seconds"):
        seconds = float(manifest["cadence_seconds"]) * 3
    if seconds is None:
        return 0  # Undeclared cadence: no forward fill, no invented gap boundary.
    if not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise ValueError("max_gap_seconds must be finite and positive")
    return int(float(seconds) * 1_000_000_000)


def build_row_index(batches: Iterable, row_count: int, path: Path, manifest: dict,
                    *, inference: bool = False, columns: list[dict] | None = None) -> tuple[np.memmap, dict]:
    """Assign per-entity chronological partitions before learning any features."""
    if row_count < 1:
        raise ValueError("Cannot prepare an empty dataset")
    rows = np.lib.format.open_memmap(path, mode="w+", dtype=ROW_DTYPE, shape=(row_count,))
    gap_column = manifest.get("gap_column")
    if gap_column and columns is not None:
        mapped = [column for column in columns if column.get("source_name", column["name"]) == gap_column]
        if len(mapped) != 1 or mapped[0].get("role") != "context":
            raise ValueError("gap_column must identify a context column")
        gap_column = mapped[0]["name"]
    splits = manifest.get("splits") or {"fractions": [0.7, 0.15, 0.15]}
    fractions = splits.get("fractions", [0.7, 0.15, 0.15])
    explicit = "train_end" in splits or "validation_end" in splits
    test_boundary = _timestamp_ns(splits["test_start"]) if "test_start" in splits else None
    training_fraction = float(splits.get("train_fraction", 0.85))
    if test_boundary is not None and not 0 < training_fraction < 1:
        raise ValueError("splits.train_fraction must be between zero and one")
    if explicit and not {"train_end", "validation_end"} <= splits.keys():
        raise ValueError("Explicit splits require train_end and validation_end")
    if not explicit and (len(fractions) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in fractions)
                         or not math.isclose(sum(float(v) for v in fractions), 1)):
        raise ValueError("splits.fractions must contain three positive fractions summing to 1")
    train_end = _timestamp_ns(splits["train_end"]) if explicit else None
    validation_end = _timestamp_ns(splits["validation_end"]) if explicit else None
    if explicit and train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    offset, entity_start, entity_code, previous_entity, previous_ts = 0, 0, -1, None, None
    entity_count = 0

    def partition_entity(start: int, end: int) -> None:
        if inference:
            rows["partition"][start:end] = 2
        elif test_boundary is not None:
            eligible = search_index(rows, "timestamp_ns", test_boundary, start=start, end=end, side="right") - start
            train_stop = start + int(Decimal(str(training_fraction)) * eligible)
            valid_stop = start + eligible
            rows["partition"][start:train_stop] = 0
            rows["partition"][train_stop:valid_stop] = 1
            rows["partition"][valid_stop:end] = 2
        elif explicit:
            # Batch the mask allocations even for a single multi-year entity.
            for begin in range(start, end, 10000):
                stop = min(begin + 10000, end)
                stamps = rows["timestamp_ns"][begin:stop]
                rows["partition"][begin:stop] = np.where(stamps <= train_end, 0,
                                                          np.where(stamps <= validation_end, 1, 2))
        else:
            size = end - start
            train_fraction = Decimal(str(fractions[0]))
            validation_fraction = Decimal(str(fractions[1]))
            train_stop = start + int(size * train_fraction)
            valid_stop = start + int(size * (train_fraction + validation_fraction))
            rows["partition"][start:train_stop] = 0
            rows["partition"][train_stop:valid_stop] = 1
            rows["partition"][valid_stop:end] = 2

    for batch in batches:
        data = batch.to_pydict()
        if gap_column and gap_column not in data:
            raise ValueError("Required structural gap_column is missing")
        for index in range(batch.num_rows):
            entity = str(data["entity_id"][index])
            timestamp = int(data["timestamp_ns"][index])
            if entity != previous_entity:
                if previous_entity is not None:
                    partition_entity(entity_start, offset)
                    if entity < previous_entity:
                        raise ValueError("Telemetry must be sorted by entity and timestamp")
                entity_start, entity_code, previous_ts = offset, entity_code + 1, None
                entity_count += 1
            if previous_ts is not None and timestamp <= previous_ts:
                raise ValueError("Telemetry timestamps must increase strictly within each entity")
            if offset >= row_count:
                raise ValueError("Dataset row_count differs from canonical telemetry")
            unavailable = False
            if gap_column:
                value = str(data[gap_column][index]).lower()
                if value not in {"0", "1", "false", "true"}:
                    raise ValueError("Structural gap_column values must be 0/1 or boolean")
                unavailable = value in {"1", "true"}
            rows[offset] = (int(data["row_id"][index]), timestamp, entity_code, 0, 0, False, unavailable)
            previous_entity, previous_ts = entity, timestamp
            offset += 1
    if offset != row_count:
        raise ValueError("Dataset row_count differs from canonical telemetry")
    partition_entity(entity_start, offset)
    counts = {key: 0 for key in PARTITIONS}
    bounds = {key: {"first_timestamp_ns": None, "last_timestamp_ns": None} for key in PARTITIONS}
    gap = max_gap_ns(manifest)
    segment, previous = -1, None
    for begin in range(0, row_count, 10000):
        chunk = rows[begin:begin + 10000]
        for index, row in enumerate(chunk):
            if previous is None or row["entity"] != previous[0] or row["partition"] != previous[1] or row["gap"] != previous[3] or (
                    gap and row["timestamp_ns"] - previous[2] > gap):
                segment += 1
            chunk["segment"][index] = segment
            previous = (row["entity"], row["partition"], row["timestamp_ns"], row["gap"])
        for key, code in PARTITIONS.items():
            stamps = chunk["timestamp_ns"][chunk["partition"] == code]
            counts[key] += len(stamps)
            if len(stamps):
                first, last = bounds[key]["first_timestamp_ns"], bounds[key]["last_timestamp_ns"]
                bounds[key] = {"first_timestamp_ns": min(int(stamps.min()), first) if first is not None else int(stamps.min()),
                               "last_timestamp_ns": max(int(stamps.max()), last) if last is not None else int(stamps.max())}
    rows.flush()
    return rows, {"row_count": row_count, "partition_counts": counts, "partition_bounds": bounds,
                  "ordering": manifest.get("ordering", "timestamp"),
                  "entity_count": entity_count, "segment_count": segment + 1, "splits": splits,
                  "max_gap_ns": gap}


def feature_candidates(columns: list[dict]) -> list[dict]:
    result = []
    for column in columns:
        role = column.get("role")
        if role in {"numeric", "numeric_feature", "feature_numeric", "categorical", "categorical_feature"}:
            result.append({"name": column["name"], "source_name": column.get("source_name", column["name"]),
                           "role": "numeric" if "numeric" in role else "categorical", "dtype": column.get("dtype"),
                           "unit": column.get("unit")})
    if not result:
        raise ValueError("Dataset has no numeric or categorical features. Supply an explicit manifest")
    identities = [column["source_name"] for column in result]
    if len(identities) != len(set(identities)):
        raise ValueError("Feature source names must be unique")
    return result


def fit_transform(batches: Iterable, rows: np.ndarray, columns: list[dict], *, max_categories: int = 64,
                  gap_ns: int = 0, numeric_transform: str = "standard",
                  feature_eligibility: str = "require_all") -> dict:
    """Estimate observed-value statistics from training rows only, in bounded space."""
    if max_categories < 1:
        raise ValueError("max_categories must be positive")
    if numeric_transform not in ("standard", "asinh"):
        raise ValueError("numeric_transform must be standard or asinh")
    if feature_eligibility not in ("require_all", "observed_training"):
        raise ValueError("feature_eligibility must be require_all or observed_training")
    features = feature_candidates(columns)
    accumulators = [{"count": 0, "mean": 0.0, "m2": 0.0, "vocabulary": set(), "truncated": False}
                    for _ in features]
    offset = 0
    for batch in batches:
        data = batch.to_pydict()
        local = rows[offset:offset + batch.num_rows]
        selected = np.flatnonzero((local["partition"] == 0) & ~local["excluded"] & ~local["gap"])
        for feature, acc in zip(features, accumulators):
            values = data[feature["name"]]
            for index in selected:
                value = values[index]
                if value is None:
                    continue
                if feature["role"] == "numeric":
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"Nonfinite value in feature {feature['source_name']}")
                    acc["count"] += 1
                    delta = number - acc["mean"]
                    acc["mean"] += delta / acc["count"]
                    acc["m2"] += delta * (number - acc["mean"])
                else:
                    token = str(value)
                    acc["count"] += 1
                    if token in acc["vocabulary"] or len(acc["vocabulary"]) < max_categories:
                        acc["vocabulary"].add(token)
                    else:
                        acc["truncated"] = True
        offset += batch.num_rows
    excluded = [{"source_name": feature["source_name"], "reason": "no_usable_training_values"}
                for feature, acc in zip(features, accumulators) if not acc["count"]]
    if feature_eligibility == "observed_training":
        eligible = [(feature, acc) for feature, acc in zip(features, accumulators) if acc["count"]]
        if not eligible:
            raise ValueError("No features have usable training values")
        features, accumulators = map(list, zip(*eligible))
    output_names = []
    for feature_index, (feature, acc) in enumerate(zip(features, accumulators)):
        if not acc["count"]:
            raise ValueError(f"Feature {feature['source_name']} has no usable training values")
        feature["offset"] = len(output_names)
        if feature["role"] == "numeric":
            scale = math.sqrt(max(0.0, acc["m2"] / acc["count"])) or 1.0
            if not math.isfinite(scale) or not math.isfinite(acc["mean"]):
                raise ValueError(f"Numeric statistics overflow for {feature['source_name']}")
            feature.update(mean=acc["mean"], scale=scale, count=acc["count"], width=2)
            output_names.extend([f"f{feature_index}:value", f"f{feature_index}:missing"])
        else:
            vocabulary = sorted(acc["vocabulary"])
            feature.update(vocabulary=vocabulary, vocabulary_truncated=acc["truncated"], width=len(vocabulary) + 3)
            output_names.extend(f"f{feature_index}:category:{category}" for category in range(len(vocabulary) + 2))
            output_names.append(f"f{feature_index}:missing")
    state = {"version": 2, "features": features, "output_names": output_names,
             "numeric_transform": numeric_transform,
             "feature_eligibility": feature_eligibility, "excluded_features": excluded,
             "max_gap_ns": int(gap_ns), "missing_policy": "causal_fill_then_training_mean_with_indicator"}
    state["fingerprint"] = fingerprint(state)
    return state


def resolve_features(state: dict, target_columns: list[dict]) -> dict[str, str]:
    """Match features by source identity across column reordering and sanitization."""
    target = {column.get("source_name", column["name"]): column for column in target_columns}
    if len(target) != len(target_columns):
        raise ValueError("Target feature source names must be unique")
    mapping = {}
    for feature in state["features"]:
        source = feature["source_name"]
        column = target.get(source)
        if column is None:
            raise ValueError(f"Required model feature is missing: {source}")
        role = column.get("role", "")
        compatible = (feature["role"] == "numeric" and role in {"numeric", "numeric_feature", "feature_numeric"}) or (
            feature["role"] == "categorical" and role in {"categorical", "categorical_feature"})
        if not compatible:
            raise ValueError(f"Incompatible role/type for required model feature: {source}")
        trained_unit, target_unit = feature.get("unit"), column.get("unit")
        if trained_unit != target_unit:
            raise ValueError(f"Incompatible declared units for required model feature {source}: {trained_unit!r} vs {target_unit!r}")
        mapping[feature["name"]] = column["name"]
    return mapping


def transform_to_memmap(batches: Iterable, rows: np.ndarray, state: dict, path: Path,
                        target_columns: list[dict]) -> np.memmap:
    mapping = resolve_features(state, target_columns)
    numeric_transform = state.get("numeric_transform", "standard")
    if numeric_transform not in ("standard", "asinh"):
        raise ValueError("numeric_transform must be standard or asinh")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype="float32",
                                      shape=(len(rows), len(state["output_names"])))
    category_maps = [{token: index + 2 for index, token in enumerate(feature.get("vocabulary", []))}
                     for feature in state["features"]]
    last_values = [None] * len(state["features"])
    last_times = [0] * len(state["features"])
    previous_segment, offset = None, 0
    numeric_only = not state["max_gap_ns"] and all(feature["role"] == "numeric" for feature in state["features"])
    for batch in batches:
        data = batch.to_pydict()
        out = np.zeros((batch.num_rows, len(state["output_names"])), dtype=np.float32)
        if numeric_only:
            for feature in state["features"]:
                values = data[mapping[feature["name"]]]
                missing = np.fromiter((value is None for value in values), dtype=bool, count=len(values))
                numbers = np.asarray(values, dtype=np.float64)
                numbers[missing] = feature["mean"]
                if not np.isfinite(numbers).all():
                    raise ValueError(f"Nonfinite value in feature {feature['source_name']}")
                with np.errstate(over="ignore", invalid="ignore"):
                    normalized = (numbers - feature["mean"]) / feature["scale"]
                if not np.isfinite(normalized).all() or (np.abs(normalized) > np.finfo(np.float32).max).any():
                    raise ValueError(f"Transformed value overflows float32 in {feature['source_name']}")
                if numeric_transform == "asinh":
                    np.arcsinh(normalized, out=normalized)
                out[:, feature["offset"]] = normalized
                out[:, feature["offset"] + 1] = missing
        else:
            for index in range(batch.num_rows):
                row = rows[offset + index]
                if row["segment"] != previous_segment or row["excluded"] or row["gap"]:
                    last_values = [None] * len(last_values)
                previous_segment = row["segment"]
                for feature_index, feature in enumerate(state["features"]):
                    value = data[mapping[feature["name"]]][index]
                    missing = value is None
                    if feature["role"] == "numeric" and not missing:
                        value = float(value)
                        if not math.isfinite(value):
                            raise ValueError(f"Nonfinite value in feature {feature['source_name']}")
                    if missing and state["max_gap_ns"] and last_values[feature_index] is not None:
                        if row["timestamp_ns"] - last_times[feature_index] <= state["max_gap_ns"]:
                            value = last_values[feature_index]
                    elif not missing and not row["excluded"] and not row["gap"]:
                        last_values[feature_index], last_times[feature_index] = value, row["timestamp_ns"]
                    begin = feature["offset"]
                    if feature["role"] == "numeric":
                        normalized = ((value if value is not None else feature["mean"]) - feature["mean"]) / feature["scale"]
                        if not math.isfinite(normalized) or abs(normalized) > np.finfo(np.float32).max:
                            raise ValueError(f"Transformed value overflows float32 in {feature['source_name']}")
                        if numeric_transform == "asinh":
                            normalized = math.asinh(normalized)
                        out[index, begin:begin + 2] = normalized, float(missing)
                    else:
                        category = 1 if value is None else category_maps[feature_index].get(str(value), 0)
                        out[index, begin + category] = 1
                        out[index, begin + feature["width"] - 1] = float(missing)
        matrix[offset:offset + batch.num_rows] = out
        offset += batch.num_rows
    matrix.flush()
    return matrix
