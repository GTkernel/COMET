"""Decode one trusted ESA channel per process.

Pickle loading can execute code. The caller must confirm trust before running
this module. Decoding needs enough memory for one complete pandas channel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _telecommands(frame: pd.DataFrame, step: int, reference: bool) -> pd.DataFrame:
    """Add zero transitions without looking ahead in the causal profile."""
    if frame.empty:
        return frame
    times = frame.timestamp_ns.to_numpy()
    extra = []
    for i, time in enumerate(times):
        if reference and (i == 0 or time - times[i - 1] > step):
            extra.append((int(time - step), 0.0))
        if i == len(times) - 1 or times[i + 1] - time > step:
            extra.append((int(time + step), 0.0))
    zeros = pd.DataFrame(extra, columns=["timestamp_ns", "value"])
    return pd.concat([frame, zeros], ignore_index=True).sort_values("timestamp_ns")


def decode_channel(source: Path, destination: Path, config: dict) -> dict:
    name = config["channel"]
    with zipfile.ZipFile(source) as archive:
        if any(item.compress_type == 9 for item in archive.infolist()):
            try:
                import zipfile_inflate64  # noqa: F401 - patches pandas' ZIP reader too
            except ImportError as exc:
                raise RuntimeError("Install COMET's esa extra to decode Deflate64 channel archives") from exc
    raw = pd.read_pickle(source)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=name)
    if not isinstance(raw, pd.DataFrame) or list(raw.columns) != [name]:
        raise ValueError(f"{name}: expected a single named column in the trusted ESA pickle")
    index = pd.to_datetime(raw.index, utc=True, errors="raise")
    if index.isna().any():
        raise ValueError(f"{name}: null timestamp")
    frame = pd.DataFrame({"timestamp_ns": index.as_unit("ns").asi8, "value": raw[name].to_numpy()})
    if config.get("telecommand"):
        # Mission 2 telecommands can store numeric observations as objects.
        # Convert before deduplication and zero insertion so Arrow gets one
        # numeric type, including for values originally stored as strings.
        try:
            values = pd.to_numeric(frame["value"], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: telecommand observations must be numeric") from exc
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name}: telecommand observations must be finite")
        frame["value"] = values
    frame = frame.sort_values("timestamp_ns", kind="stable").drop_duplicates()
    if frame.timestamp_ns.duplicated().any():
        raise ValueError(f"{name}: conflicting values at a duplicate timestamp")
    frame = frame.reset_index(drop=True)
    reference = config["profile"] == "esa-reference"
    numeric = pd.api.types.is_numeric_dtype(frame.value)
    if numeric:
        values = pd.to_numeric(frame.value).astype(float)
        if np.isinf(values).any():
            raise ValueError(f"{name}: infinite values")
        if config.get("derivative"):
            # The reference intentionally reproduces upstream forward differencing.
            values = values.shift(-1, fill_value=values.iloc[-1] if len(values) else 0) - values if reference else values.diff()
        frame["value"] = values
    elif reference:
        frame["value"] = pd.factorize(frame.value, sort=False)[0].astype(float)
        numeric = True
    else:
        frame["value"] = frame.value.map(lambda x: None if pd.isna(x) else str(x))
    step = int(config["cadence_seconds"] * 1_000_000_000)
    if config.get("telecommand"):
        frame = _telecommands(frame, step, reference)
    if reference:
        frame["label"] = 0
        for item in config.get("annotations", []):
            mask = frame.timestamp_ns.between(item["start_ns"], item["end_ns"])
            frame.loc[mask, "label"] = item["label"]
    boundary = int(config["test_boundary_ns"])
    stats = {"channel": name, "numeric": bool(numeric), "telecommand": bool(config.get("telecommand")), "partitions": {}}
    fields = [("timestamp_ns", pa.int64()), ("value", pa.float64() if numeric else pa.string()), ("partition", pa.int8())]
    if reference:
        fields.append(("is_override", pa.bool_()))
    schema = pa.schema(fields)
    tmp = destination.with_suffix(".partial")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        # Reference splitting happens before resampling, as in the pinned scripts.
        parts = [(0, frame[frame.timestamp_ns <= boundary]), (1, frame[frame.timestamp_ns > boundary])] if reference else [(0, frame)]
        for partition, part in parts:
            part = part.copy()
            if part.empty:
                stats["partitions"][str(partition)] = None
                continue
            low = int(part.timestamp_ns.iloc[0] // step * step)
            high = int(-(-int(part.timestamp_ns.iloc[-1]) // step) * step)
            first_value = part.value.iloc[0]
            stats["partitions"][str(partition)] = {"start_ns": low, "end_ns": high, "first_value": None if pd.isna(first_value) else first_value}
            if reference:
                # Restore annotated samples at the bin's right edge if resampling
                # would discard them. Communication gaps stay excluded.
                part["bin"] = (part.timestamp_ns // step) * step
                last = part.groupby("bin", sort=False).tail(1).set_index("bin")
                annotated = part[part.label.isin([1, 2])].groupby("bin", sort=False).tail(1)
                restored = annotated[annotated.bin.map(last.label).eq(0)].copy()
                restored["timestamp_ns"] = restored.bin + step
                first = pd.DataFrame({"timestamp_ns": [low], "value": [first_value]})
                # Keep point overrides separate from the held source signal.
                # Restoring one grid sample must not extend it into later bins.
                part["is_override"] = False
                first["is_override"] = False
                restored["is_override"] = True
                part = pd.concat([part, first, restored], ignore_index=True)
                part = part.sort_values("timestamp_ns", kind="stable").drop_duplicates(["timestamp_ns", "is_override"], keep="last")
            part["partition"] = partition
            for offset in range(0, len(part), config["batch_size"]):
                batch = part.iloc[offset: offset + config["batch_size"]]
                writer.write_table(pa.Table.from_pandas(batch[schema.names], schema=schema, preserve_index=False))
    tmp.replace(destination)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stats = decode_channel(args.source, args.destination, config)
    args.destination.with_suffix(".json").write_text(json.dumps(stats), encoding="utf-8")


if __name__ == "__main__":
    main()
