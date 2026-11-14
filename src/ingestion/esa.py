"""Versioned, resumable adapters for trusted ESA Mission 1/2 archives.

The causal profile calculates telemetry independently of annotations. The
``esa-reference`` profile follows upstream offline resampling and records that
processing choice in the dataset provenance.
"""
from __future__ import annotations

import copy
import csv
import hashlib
from itertools import islice
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ESA_REVISION = "aeebcd9ecd3e7266d6d6a035a8081b3da83dfe33"
ADAPTER_VERSION = "1.0.3"
ESA_RECORDS = {"1": "https://zenodo.org/records/12528696", "2": "https://zenodo.org/records/15237121"}
TEST_BOUNDARIES = {1: "2007-01-01T00:00:00Z", 2: "2001-10-01T00:00:00Z"}
PROCESSED_TRAIN_BOUNDARIES = {
    1: {"3_months": "2000-04-01", "10_months": "2000-11-01", "21_months": "2001-10-01", "42_months": "2003-07-01", "84_months": "2007-01-01"},
    2: {"1_months": "2000-02-01", "5_months": "2000-06-01", "10_months": "2000-11-01", "21_months": "2001-10-01"},
}
_PROCESSED_DIRECTORY = re.compile(r"^ESA-Mission([12])-semi-supervised$", re.I)
_PROCESSED_FILE = re.compile(r"^([0-9]+_months)\.(train|test)\.csv$", re.I)


def _esa_version(esa: dict) -> str:
    if esa.get("version") is None:
        raise ValueError("esa.version is required. Set it to 1 or 2")
    version = str(esa["version"]).removeprefix("v")
    if version not in ESA_RECORDS:
        raise ValueError("Supported ESA dataset versions are 1 and 2")
    return version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _preparation_work(storage: Path, kind: str, config: dict, checksums: dict, manifest: dict) -> Path:
    """Keep preparation caches stable across operational batch-size changes.

    An alias lets the new key reuse caches from the earlier format, including
    interrupted work. Existing paths, files and provenance stay in place.
    """
    from .store import _import_lock

    parent = storage / kind
    parent.mkdir(parents=True, exist_ok=True)
    semantic_config = {key: value for key, value in config.items() if key != "batch_size"}
    payload = {"config": semantic_config, "checksums": checksums, "manifest": manifest}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    legacy_payload = {**payload, "config": config}
    legacy_identity = hashlib.sha256(json.dumps(legacy_payload, sort_keys=True).encode()).hexdigest()
    pointer = parent / f"{identity}.cache.json"
    with _import_lock(parent / f"{identity}.cache.lock"):
        if pointer.exists():
            selected = json.loads(pointer.read_text(encoding="utf-8"))["work_identity"]
            if not isinstance(selected, str) or not re.fullmatch(r"[0-9a-f]{64}", selected):
                raise ValueError("Invalid ESA preparation cache identity")
        else:
            selected = legacy_identity if (parent / legacy_identity).is_dir() else identity
            _save_json(pointer, {"work_identity": selected})
        work = parent / selected
        if work.resolve().parent != parent.resolve():
            raise ValueError("ESA preparation cache escapes its storage directory")
        work.mkdir(exist_ok=True)
    return work


def _mission_dir(path: Path) -> Path | None:
    candidates = [path, *sorted(path.glob("*"))] if path.is_dir() else []
    for candidate in candidates:
        if (candidate / "channels").is_dir() and (candidate / "labels.csv").is_file() and (candidate / "anomaly_types.csv").is_file():
            return candidate
    return None


def recognize_esa_source(source: str | Path) -> bool:
    source = Path(source)
    if _mission_dir(source) is not None:
        return True
    if source.is_file() and source.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                return any(re.search(r"(^|/)channels/channel_\d+\.zip$", name) for name in names) and any(name.endswith("/labels.csv") or name == "labels.csv" for name in names)
        except zipfile.BadZipFile:
            return False
    return False


def recognize_processed_esa_source(source: str | Path) -> bool:
    """Recognize the CSV layout emitted by the pinned ESA prep scripts."""
    path = Path(source)
    if not _PROCESSED_DIRECTORY.fullmatch(path.parent.name if path.is_file() else path.name):
        return False
    if path.is_file():
        return _PROCESSED_FILE.fullmatch(path.name) is not None
    return path.is_dir() and any(_PROCESSED_FILE.fullmatch(item.name) for item in path.glob("*.csv"))


def _processed_selection(source: Path, esa: dict) -> tuple[int, list[tuple[str, Path]], str | None]:
    directory = source.parent if source.is_file() else source
    matched = _PROCESSED_DIRECTORY.fullmatch(directory.name)
    if matched is None or not recognize_processed_esa_source(source):
        raise ValueError("Expected an ESA-Mission1/2-semi-supervised processed CSV or directory")
    mission = int(matched.group(1))
    if esa.get("mission") is not None and int(esa["mission"]) != mission:
        raise ValueError("esa.mission conflicts with the processed ESA directory")
    allowed = PROCESSED_TRAIN_BOUNDARIES[mission]
    train_files = {match.group(1): path for path in directory.glob("*.train.csv")
                   if (match := _PROCESSED_FILE.fullmatch(path.name)) and match.group(2).lower() == "train" and match.group(1) in allowed}
    prescribed_test = ("84_months" if mission == 1 else "21_months") + ".test.csv"
    test_file = directory / prescribed_test
    choice = esa.get("processed_train")
    if choice is not None:
        choice = str(choice).removesuffix(".train.csv")
        if choice not in train_files:
            raise ValueError(f"esa.processed_train must select one of {sorted(train_files)}")
    elif source.is_file() and source.name.endswith(".train.csv"):
        choice = _PROCESSED_FILE.fullmatch(source.name).group(1)
        if choice not in train_files:
            raise ValueError("Unrecognized ESA training period")
    elif len(train_files) == 1:
        choice = next(iter(train_files))
    elif len(train_files) > 1:
        raise ValueError(f"Several ESA training periods are present. Set esa.processed_train to one of {sorted(train_files)}")
    files = []
    if choice is not None:
        files.append(("train", train_files[choice]))
    if test_file.is_file():
        files.append(("test", test_file))
    elif source.is_file() and source.name.endswith(".test.csv"):
        files.append(("test", source))
    if not files:
        raise ValueError("No recognized ESA processed train/test CSVs were found")
    if len(files) == 1 and (esa.get("allow_single") is not True or not esa.get("single_partition") == files[0][0]):
        raise ValueError("A lone processed ESA CSV needs esa.allow_single=true and esa.single_partition=train or test")
    return mission, files, choice


def _processed_headers(files: list[tuple[str, Path]]) -> tuple[list[str], list[str], list[str]]:
    headers = []
    for _, path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)) or not header or "timestamp" not in header or any(not name for name in header):
            raise ValueError(f"{path.name}: processed ESA headers must be distinct and include timestamp")
        if headers and set(header) != set(headers):
            raise ValueError("Processed ESA train/test files have different feature or annotation columns")
        headers = headers or header
    features = [name for name in headers if re.fullmatch(r"(?:channel|telecommand)_\d+", name)]
    labels = [name for name in headers if name.startswith("is_anomaly_")]
    if not features or not labels or set(headers) != {"timestamp", *features, *labels}:
        raise ValueError("Processed ESA CSV must contain telemetry and is_anomaly_* annotation columns")
    if set(labels) != {"is_anomaly_" + name for name in features}:
        raise ValueError("Processed ESA annotations must correspond to telemetry columns")
    return headers, features, labels


def _processed_values(values: list[str], name: str, *, labels: bool = False) -> np.ndarray:
    present = np.fromiter((value.strip() != "" for value in values), dtype=bool, count=len(values))
    try:
        numbers = np.asarray(pd.to_numeric([value if item else None for value, item in zip(values, present)], errors="raise"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Processed ESA column {name!r} contains nonnumeric observations") from exc
    if not np.isfinite(numbers[present]).all():
        raise ValueError(f"Processed ESA column {name!r} contains nonfinite observations")
    if labels and (not present.all() or not np.isin(numbers, [0, 1, 2, 3]).all()):
        raise ValueError(f"Processed ESA annotation {name!r} must contain labels 0, 1, 2, or 3")
    return numbers


def _write_processed_pair(files: list[tuple[str, Path]], output: Path, points: Path, features: list[str], labels: list[str], mission: int, train_name: str | None, batch_size: int) -> int:
    schema = pa.schema([("timestamp", pa.string()), *[(name, pa.float64()) for name in features]])
    point_schema = pa.schema([("timestamp", pa.string()), ("channel", pa.string()), ("label_name", pa.string()), ("label_value", pa.int8()), ("source_partition", pa.string())])
    previous_ns = None
    count = 0
    with pq.ParquetWriter(output, schema, compression="zstd") as telemetry_writer, pq.ParquetWriter(points, point_schema, compression="zstd") as annotation_writer:
        for partition, path in files:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                while rows := list(islice(reader, batch_size)):
                    if any(None in row or any(value is None for value in row.values()) for row in rows):
                        raise ValueError(f"{path.name}: a row has more or fewer values than the CSV header")
                    timestamps = [row["timestamp"] for row in rows]
                    try:
                        parsed = pd.to_datetime(timestamps, format="%Y-%m-%d %H:%M:%S", utc=True, errors="raise")
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{path.name}: invalid processed ESA timestamp") from exc
                    if parsed.isna().any():
                        raise ValueError(f"{path.name}: processed ESA timestamp cannot be empty")
                    nanos = parsed.as_unit("ns").asi8
                    if (previous_ns is not None and int(nanos[0]) <= previous_ns) or np.any(nanos[1:] <= nanos[:-1]):
                        raise ValueError("Processed ESA timestamps must be unique and chronological across train/test")
                    if partition == "train" and train_name and int(nanos[-1]) > pd.Timestamp(PROCESSED_TRAIN_BOUNDARIES[mission][train_name], tz="UTC").value:
                        raise ValueError("Processed ESA training rows exceed the declared training boundary")
                    if partition == "test" and int(nanos[0]) <= pd.Timestamp(TEST_BOUNDARIES[mission]).value:
                        raise ValueError("Processed ESA test rows precede or equal the prescribed test boundary")
                    previous_ns = int(nanos[-1])
                    data = {"timestamp": timestamps}
                    for name in features:
                        values = _processed_values([row[name] for row in rows], name)
                        data[name] = pa.array(values, type=pa.float64(), mask=np.isnan(values))
                    telemetry_writer.write_table(pa.Table.from_pydict(data, schema=schema))
                    point_rows = []
                    for name in labels:
                        values = _processed_values([row[name] for row in rows], name, labels=True)
                        for index in np.flatnonzero(values):
                            point_rows.append({"timestamp": timestamps[int(index)], "channel": name.removeprefix("is_anomaly_"), "label_name": name,
                                               "label_value": int(values[index]), "source_partition": partition})
                    if point_rows:
                        annotation_writer.write_table(pa.Table.from_pylist(point_rows, schema=point_schema))
                    count += len(rows)
    if count == 0:
        raise ValueError("Processed ESA source contains no rows")
    return count


def prepare_processed_esa_source(source: str | Path, manifest: dict, *, storage_dir: str | Path, batch_size: int = 10000) -> tuple[Path, dict]:
    """Combine the pinned processed CSV pair in batches, with labels isolated."""
    source, storage = Path(source).resolve(), Path(storage_dir).resolve()
    if not 0 < batch_size <= 100000:
        raise ValueError("batch_size must be between 1 and 100000")
    result = copy.deepcopy(manifest)
    esa = result.setdefault("esa", {})
    if esa.get("profile", "esa-reference") != "esa-reference":
        raise ValueError("Upstream processed ESA CSVs require the esa-reference profile because they already contain offline preprocessing")
    version = _esa_version(esa)
    mission, files, train_name = _processed_selection(source, esa)
    _, features, labels = _processed_headers(files)
    if len(files) == 1 and "splits" not in result:
        raise ValueError("A lone processed ESA CSV also needs explicit manifest splits")
    checksums = {str(path.resolve()): _sha256(path) for _, path in files}
    folder = source.parent if source.is_file() else source
    raw_metadata = folder.parents[2] / f"ESA-Mission{mission}" if len(folder.parents) > 2 else None
    if raw_metadata and all((raw_metadata / name).is_file() for name in ("labels.csv", "anomaly_types.csv")):
        for name in ("labels.csv", "anomaly_types.csv"):
            checksums[str(raw_metadata / name)] = _sha256(raw_metadata / name)
    else:
        raw_metadata = None
    config = {"adapter_version": ADAPTER_VERSION, "mission": mission, "version": version, "profile": "esa-reference",
              "processed_train": train_name, "files": [kind for kind, _ in files], "batch_size": batch_size}
    work = _preparation_work(storage, "esa-processed", config, checksums, result)
    from .store import _import_lock
    with _import_lock(work / "prepare.lock"):
        marker = work / "complete.json"
        output = work / "telemetry.parquet"
        points = work / "point-annotations.parquet"
        if marker.is_file() and output.is_file() and points.is_file():
            cached = json.loads(marker.read_text(encoding="utf-8"))["manifest"]
            if Path(cached["annotations_path"]).is_file():
                cached["_source_sorted_unique"] = True
                return output, cached
        temporary_output = work / "telemetry.partial.parquet"
        temporary_points = work / "point-annotations.partial.parquet"
        rows = _write_processed_pair(files, temporary_output, temporary_points, features, labels, mission, train_name, batch_size)
        temporary_output.replace(output)
        temporary_points.replace(points)
        interval_path = None
        if raw_metadata:
            interval_path = work / "interval-annotations.parquet"
            _annotations(raw_metadata).to_parquet(interval_path, index=False)
        cadence = 30 if mission == 1 else 18
        result.update({"version": 1, "format": "parquet", "layout": "wide", "timestamp": "timestamp", "timezone": "UTC",
                       "_source_sorted_unique": True,
                       "cadence_seconds": cadence, "max_gap_seconds": result.get("max_gap_seconds", cadence * 10),
                       "numeric_features": features, "categorical_features": [], "label_columns": [],
                       "annotations_path": str(interval_path or points), "annotation_format": "esa-intervals" if interval_path else "esa-points"})
        result.setdefault("preprocessing", {})["forward_fill"] = False
        if len(files) == 2:
            result.setdefault("splits", {"test_start": TEST_BOUNDARIES[mission], "train_fraction": .85})
        esa.update({"mission": mission, "version": version, "profile": "esa-reference", "dataset_url": ESA_RECORDS[version],
                    "upstream_revision": ESA_REVISION, "adapter_version": ADAPTER_VERSION, "source_checksums": checksums,
                    "source": str(source), "test_boundary": TEST_BOUNDARIES[mission], "processed_train": train_name,
                    "processed_files": [{"partition": kind, "path": str(path)} for kind, path in files], "processed_rows": rows,
                    "processed_point_annotations_path": str(points), "offline_preprocessing": True})
        _save_json(marker, {"manifest": result})
        return output, result


def _extract(source: Path, storage: Path) -> tuple[Path, dict]:
    from .store import _import_lock
    digest = _sha256(source)
    lock_dir = storage / "esa-extracted"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with _import_lock(lock_dir / f"{digest}.lock"):
        return _extract_locked(source, storage, digest)


def _extract_locked(source: Path, storage: Path, digest: str) -> tuple[Path, dict]:
    destination = storage / "esa-extracted" / digest
    marker = destination / "complete.json"
    if not marker.exists():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            if any(info.compress_type == 9 for info in archive.infolist()):
                try:
                    import zipfile_inflate64  # noqa: F401 - enables official ESA Deflate64 archives
                except ImportError as exc:
                    raise RuntimeError("This ESA ZIP uses Deflate64. Install COMET's esa extra (zipfile-inflate64)") from exc
            for info in archive.infolist():
                relative = PurePosixPath(info.filename)
                if relative.is_absolute() or ".." in relative.parts or "\\" in info.filename or ":" in info.filename:
                    raise ValueError("Unsafe ESA archive member path")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("ESA archive symlinks are not supported")
                target = destination.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == info.file_size:
                    continue
                temporary = target.with_name(target.name + ".partial")
                with archive.open(info) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                temporary.replace(target)
        _save_json(marker, {"sha256": digest})
    root = _mission_dir(destination)
    if root is None:
        raise ValueError("Archive does not contain a recognized ESA mission layout")
    return root, {"archive": {"name": source.name, "sha256": digest}}


def _annotations(root: Path) -> pd.DataFrame:
    labels = pd.read_csv(root / "labels.csv", dtype={"ID": str, "Channel": str})
    required = {"ID", "Channel", "StartTime", "EndTime"}
    if not required.issubset(labels.columns):
        raise ValueError(f"ESA labels are missing columns: {sorted(required - set(labels.columns))}")
    types = pd.read_csv(root / "anomaly_types.csv", dtype={"ID": str})
    if "Category" not in types or types.ID.duplicated().any():
        raise ValueError("ESA anomaly_types must have unique IDs and Category")
    labels = labels.merge(types, on="ID", how="left", validate="many_to_one")
    if labels.Category.isna().any():
        raise ValueError("ESA labels reference an unknown anomaly ID")
    for col in ["StartTime", "EndTime"]:
        labels[col] = pd.to_datetime(labels[col], utc=True, errors="raise")
    if labels[["StartTime", "EndTime"]].isna().any().any() or (labels.EndTime < labels.StartTime).any():
        raise ValueError("ESA annotation intervals are invalid")
    return labels


class _ChannelCursor:
    """Read a decoded, partition-ordered channel once with bounded lookahead."""

    def __init__(self, path: Path, *, numeric: bool, reference: bool, batch_size: int):
        columns = ["timestamp_ns", "value", "partition"] + (["is_override"] if reference else [])
        scanner = ds.dataset(path, format="parquet").scanner(
            columns=columns, batch_size=batch_size, use_threads=False,
            batch_readahead=1, fragment_readahead=1,
        )
        self._reader = scanner.to_reader()
        self.numeric = numeric
        self.reference = reference
        self._times = np.empty(0, dtype=np.int64)
        self._partitions = np.empty(0, dtype=np.int8)
        self._values = np.empty(0, dtype=float if numeric else object)
        self._overrides = np.empty(0, dtype=bool)
        self._offset = 0
        self._done = False
        self._partition = None
        self.latest_ns = None
        self.latest_value = None
        self.reference_first = None
        self.reference_last = None

    def select_partition(self, partition: int) -> None:
        if self._partition != partition:
            self._partition = partition
            self.latest_ns = None
            self.latest_value = None
            self.reference_first = None
            self.reference_last = None

    def _load(self) -> bool:
        while self._offset >= len(self._times) and not self._done:
            try:
                batch = next(self._reader)
            except StopIteration:
                self._done = True
                return False
            self._times = batch.column("timestamp_ns").to_numpy(zero_copy_only=False)
            self._partitions = batch.column("partition").to_numpy(zero_copy_only=False)
            values = batch.column("value")
            self._values = values.to_numpy(zero_copy_only=False) if self.numeric else np.asarray(values.to_pylist(), dtype=object)
            self._overrides = batch.column("is_override").to_numpy(zero_copy_only=False) if self.reference else np.zeros(len(batch), dtype=bool)
            self._offset = 0
        return self._offset < len(self._times)

    def chunks_through(self, partition: int, end_ns: int):
        while self._load():
            current_partition = int(self._partitions[self._offset])
            if current_partition < partition:
                self._offset = int(np.searchsorted(self._partitions, partition, side="left"))
                continue
            if current_partition > partition:
                return
            partition_end = int(np.searchsorted(self._partitions, partition, side="right"))
            start = self._offset
            stop = start + int(np.searchsorted(self._times[start:partition_end], end_ns, side="right"))
            if stop == start:
                return
            self._offset = stop
            yield self._times[start:stop], self._values[start:stop], self._overrides[start:stop]
            if stop < partition_end:
                return

    def close(self) -> None:
        self._reader.close()


def _align_channel(cursor: _ChannelCursor, times: np.ndarray, partition: int, bounds: dict, config: dict) -> tuple[pa.Array, np.ndarray]:
    """Place observations on their first eligible grid sample and hold earlier values."""
    cursor.select_partition(partition)
    count = len(times)
    missing = np.nan if cursor.numeric else None
    updates = np.full(count, missing, dtype=float if cursor.numeric else object)
    observed = np.zeros(count, dtype=np.int64)
    present = np.zeros(count, dtype=bool)
    overrides = np.full(count, missing, dtype=float if cursor.numeric else object)
    override_present = np.zeros(count, dtype=bool)
    if cursor.latest_ns is not None:
        updates[0], observed[0], present[0] = cursor.latest_value, cursor.latest_ns, True
    for source_times, source_values, source_overrides in cursor.chunks_through(partition, int(times[-1])):
        normal = ~source_overrides
        if normal.any():
            normal_times, normal_values = source_times[normal], source_values[normal]
            positions = np.searchsorted(times, normal_times, side="left")
            # Several observations can reach one grid point. The last wins,
            # including when its value is null.
            keep = np.r_[positions[1:] != positions[:-1], True]
            positions = positions[keep]
            updates[positions] = normal_values[keep]
            observed[positions] = normal_times[keep]
            present[positions] = True
            cursor.latest_ns = int(normal_times[-1])
            cursor.latest_value = normal_values[-1]
        if source_overrides.any():
            override_times = source_times[source_overrides]
            override_values = source_values[source_overrides]
            positions = np.searchsorted(times, override_times, side="left")
            exact = times[positions] == override_times
            positions, override_values = positions[exact], override_values[exact]
            if len(positions):
                keep = np.r_[positions[1:] != positions[:-1], True]
                overrides[positions[keep]] = override_values[keep]
                override_present[positions[keep]] = True
    last = np.maximum.accumulate(np.where(present, np.arange(count), -1))
    known = last >= 0
    values = np.full(count, missing, dtype=updates.dtype)
    values[known] = updates[last[known]]
    if cursor.reference:
        # Point restorations never become the held source or predecessor.
        values[override_present] = overrides[override_present]
        # Upstream fills each channel's local grid before the wide frame.
        # Beyond that grid, carry its final output, which may include a
        # restored annotation sample.
        values[times > bounds["end_ns"]] = missing
        # Upstream fills the OUTPUT grid after restoring annotated samples.
        # Keep this state separate from the raw predecessor: an override fills
        # subsequent null grid values, but a usable held observation replaces it.
        if cursor.reference_last is not None and pd.isna(values[0]):
            values[0] = cursor.reference_last
        usable = ~pd.isna(values)
        if usable.any():
            positions = np.flatnonzero(usable)
            if cursor.reference_first is None:
                cursor.reference_first = values[positions[0]]
            previous = np.maximum.accumulate(np.where(usable, np.arange(count), -1))
            known = previous >= 0
            values[known] = values[previous[known]]
            cursor.reference_last = values[positions[-1]]
    elif config.get("max_gap_seconds") is not None:
        stale = np.zeros(count, dtype=bool)
        stale[known] = times[known] - observed[last[known]] > config["max_gap_seconds"] * 1_000_000_000
        values[stale] = missing
    valid = ~pd.isna(values)
    return pa.array(values, type=pa.float64() if cursor.numeric else pa.string(), from_pandas=True), valid


def _backfill_reference(source: Path, destination: Path, schema: pa.Schema, first_values: dict, config: dict) -> int:
    """Fill only leading grid nulls from each partition's first usable output."""
    temporary = destination.with_suffix(".filled.partial")
    rows = 0
    with pq.ParquetFile(source) as reader, pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for batch in reader.iter_batches(batch_size=100_000):
            partitions = batch.column("_comet_esa_partition")
            times = batch.column("_comet_esa_timestamp_ns")
            arrays = []
            for field in schema:
                values = batch.column(field.name)
                for partition, channels in first_values.items():
                    first = channels.get(field.name)
                    if first is not None:
                        leading = pc.and_(pc.equal(partitions, partition), pc.is_null(values))
                        values = pc.if_else(leading, pa.scalar(first, type=field.type), values)
                arrays.append(values)
            table = pa.Table.from_arrays(arrays, schema=schema)
            # Range options select OUTPUT only. Fill the complete partitions
            # first so prefix overrides and future leading backfill agree with
            # slicing the upstream full-grid result.
            step = config["cadence_seconds"] * 1_000_000_000
            keep = pc.or_(pc.equal(partitions, 0), pc.greater(times, config["test_boundary_ns"] // step * step))
            if config.get("start_ns") is not None:
                keep = pc.and_(keep, pc.greater_equal(times, config["start_ns"]))
            if config.get("end_ns") is not None:
                keep = pc.and_(keep, pc.less_equal(times, config["end_ns"]))
            table = table.filter(keep)
            if table.num_rows:
                writer.write_table(table)
                rows += table.num_rows
    temporary.replace(destination)
    source.unlink()
    return rows


def _align(decoded: list[tuple[Path, dict]], destination: Path, config: dict) -> None:
    step = config["cadence_seconds"] * 1_000_000_000
    reference = config["profile"] == "esa-reference"
    fields = [("timestamp", pa.string()), *[(stat["channel"], pa.float64() if stat["numeric"] else pa.string()) for _, stat in decoded]]
    if not reference:
        fields.append(("comet_gap", pa.string()))
    schema = pa.schema(fields)
    staging_schema = schema.append(pa.field("_comet_esa_partition", pa.int8())).append(pa.field("_comet_esa_timestamp_ns", pa.int64())) if reference else schema
    alignment_batch_size = 100_000
    source_batch_size = min(config.get("batch_size", 10000), 10000)
    cursors = []
    completed_rows = 0
    first_values = {}
    progress_path = destination.parent / "alignment-progress.json"
    has_telemetry = any(not stat.get("telecommand", not stat["channel"].startswith("channel_")) for _, stat in decoded)
    try:
        for path, stat in decoded:
            cursors.append(_ChannelCursor(path, numeric=stat["numeric"], reference=reference, batch_size=source_batch_size))
        staging = destination.with_suffix(".partial")
        with pq.ParquetWriter(staging, staging_schema, compression="zstd") as writer:
            for partition in ([0, 1] if reference else [0]):
                available = [stat["partitions"].get(str(partition)) for _, stat in decoded]
                available = [item for item in available if item is not None]
                if not available:
                    continue
                low = min(item["start_ns"] for item in available)
                high = max(item["end_ns"] for item in available)
                if not reference and config.get("start_ns") is not None:
                    low = max(low, -(-config["start_ns"] // step) * step)
                if not reference and config.get("end_ns") is not None:
                    high = min(high, (config["end_ns"] // step) * step)
                for begin in range(low, high + 1, step * alignment_batch_size):
                    times = np.arange(begin, min(high + 1, begin + step * alignment_batch_size), step, dtype=np.int64)
                    arrays = [pa.array(pd.to_datetime(times, utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))]
                    telemetry_available = np.zeros(len(times), dtype=bool)
                    for cursor, (_, stat) in zip(cursors, decoded):
                        bounds = stat["partitions"].get(str(partition))
                        if bounds is None:
                            arrays.append(pa.array([0.0 if reference else None] * len(times), type=pa.float64() if stat["numeric"] else pa.string()))
                            continue
                        values, valid = _align_channel(cursor, times, partition, bounds, config)
                        if not reference and not stat.get("telecommand", not stat["channel"].startswith("channel_")):
                            telemetry_available |= valid
                        arrays.append(values)
                    if not reference:
                        arrays.append(pa.array(np.where(telemetry_available | (not has_telemetry), "0", "1")))
                    else:
                        arrays.append(pa.array(np.full(len(times), partition, dtype=np.int8)))
                        arrays.append(pa.array(times))
                        first_values[partition] = {stat["channel"]: cursor.reference_first for cursor, (_, stat) in zip(cursors, decoded)}
                    writer.write_table(pa.Table.from_arrays(arrays, schema=staging_schema))
                    completed_rows += len(times)
                    _save_json(progress_path, {"status": "aligning", "completed_rows": completed_rows,
                                             "partition": partition, "through_ns": int(times[-1]), "channels": len(decoded)})
        if reference:
            _save_json(progress_path, {"status": "backfilling", "completed_rows": completed_rows, "channels": len(decoded)})
            completed_rows = _backfill_reference(staging, destination, schema, first_values, config)
        else:
            staging.replace(destination)
        _save_json(progress_path, {"status": "complete", "completed_rows": completed_rows, "channels": len(decoded)})
    finally:
        for cursor in cursors:
            cursor.close()


def prepare_esa_source(source: str | Path, manifest: dict, *, storage_dir: str | Path, batch_size: int = 10000) -> tuple[Path, dict]:
    """Decode explicitly trusted raw mission data and return a generic source.

    Completed channel conversions survive interruption. Checksums and all
    transformation options participate in cache identities. Optional ``esa.start``
    and ``esa.end`` select bounded *output* ranges without changing source decoding.
    """
    source, storage = Path(source).resolve(), Path(storage_dir).resolve()
    result = copy.deepcopy(manifest)
    esa = result.setdefault("esa", {})
    if esa.get("trusted_pickle") is not True:
        raise ValueError("Raw ESA pickle data can execute code. Set esa.trusted_pickle=true only for a trusted official source")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    profile = esa.get("profile", "causal")
    if profile not in {"causal", "esa-reference"}:
        raise ValueError("ESA profile must be causal or esa-reference")
    version = _esa_version(esa)
    archive_info = {}
    if source.is_file():
        if not recognize_esa_source(source):
            raise ValueError("Not a recognized ESA mission archive")
        root, archive_info = _extract(source, storage)
    else:
        root = _mission_dir(source)
    if root is None:
        raise ValueError("Expected an ESA mission directory with channels, labels.csv and anomaly_types.csv")
    mission = esa.get("mission")
    if mission is None:
        match = re.search(r"mission[_ -]?([12])", root.name, re.I)
        if match:
            mission = int(match.group(1))
        else:
            raise ValueError("The directory name is ambiguous. Set esa.mission to 1 or 2")
    mission = int(mission)
    if mission not in (1, 2):
        raise ValueError("ESA mission must be 1 or 2")
    cadence = 30 if mission == 1 else 18
    boundary = TEST_BOUNDARIES[mission]
    annotations = _annotations(root)
    channels = [(p, False) for p in sorted((root / "channels").glob("*.zip")) if re.fullmatch(r"channel_\d+", p.stem)]
    if not channels:
        raise ValueError("ESA mission has no recognized channel files")
    tc_path = root / "telecommands.csv"
    if tc_path.exists():
        commands = pd.read_csv(tc_path)
        for name in sorted(commands.loc[commands.Priority >= 3, "Telecommand"].astype(str)):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("Invalid ESA telecommand filename")
            path = root / "telecommands" / f"{name}.zip"
            if not path.is_file():
                raise ValueError(f"Missing ESA telecommand file {name}")
            channels.append((path, True))
    selected = esa.get("channels")
    if selected is not None:
        if not isinstance(selected, list) or not selected or set(selected) - {p.stem for p, _ in channels}:
            raise ValueError("esa.channels must be a nonempty list of existing channel names")
        channels = [(p, tc) for p, tc in channels if p.stem in selected]
        annotations = annotations[annotations.Channel.isin(selected)]
    checksums = {str(p.relative_to(root)).replace(os.sep, "/"): _sha256(p) for p, _ in channels}
    for name in ["labels.csv", "anomaly_types.csv", "telecommands.csv"]:
        if (root / name).exists():
            checksums[name] = _sha256(root / name)
    config = {"adapter_version": ADAPTER_VERSION, "mission": mission, "version": version, "profile": profile, "cadence_seconds": cadence, "test_boundary_ns": pd.Timestamp(boundary).value, "max_gap_seconds": result.get("max_gap_seconds", cadence * 10), "batch_size": batch_size}
    if profile == "esa-reference":
        config["reference_grid_fill_version"] = 3
    for key in ["start", "end"]:
        if esa.get(key) is not None:
            config[f"{key}_ns"] = pd.Timestamp(esa[key]).value
    if config.get("start_ns", -2**63) > config.get("end_ns", 2**63 - 1):
        raise ValueError("esa.start must not exceed esa.end")
    work = _preparation_work(storage, "esa-prepared", config, checksums, result)
    from .store import _import_lock
    with _import_lock(work / "prepare.lock"):
        return _finish_preparation(work, source, result, esa, channels, annotations, config, checksums, archive_info)


def _finish_preparation(work: Path, source: Path, result: dict, esa: dict, channels: list, annotations: pd.DataFrame, config: dict, checksums: dict, archive_info: dict) -> tuple[Path, dict]:
    mission, version, profile = config["mission"], config["version"], config["profile"]
    cadence, boundary = config["cadence_seconds"], TEST_BOUNDARIES[mission]
    output = work / "telemetry.parquet"
    marker = work / "complete.json"
    if marker.exists() and output.exists():
        cached = json.loads(marker.read_text(encoding="utf-8"))["manifest"]
        cached["_source_sorted_unique"] = True
        return output, cached
    decoded = []
    for path, telecommand in channels:
        name = path.stem
        target = work / f"{name}.parquet"
        stat_path = target.with_suffix(".json")
        if not target.exists() or not stat_path.exists():
            index = int(name.split("_")[1]) if not telecommand else -1
            local = {**config, "channel": name, "telecommand": telecommand, "derivative": (4 <= index <= 11 if mission == 1 else 29 <= index <= 46)}
            if profile == "esa-reference":
                local["annotations"] = [{"start_ns": row.StartTime.value, "end_ns": row.EndTime.value, "label": {"Anomaly": 1, "Rare Event": 2}.get(row.Category, 3)} for row in annotations.itertuples() if row.Channel == name]
            cfg = work / f"{name}.config.json"
            _save_json(cfg, local)
            subprocess.run([sys.executable, str(Path(__file__).with_name("esa_decode.py")), str(path), str(target), str(cfg)], check=True)
        decoded.append((target, json.loads(stat_path.read_text(encoding="utf-8"))))
        _save_json(work / "progress.json", {"completed_channels": [stat["channel"] for _, stat in decoded], "total_channels": len(channels)})
    _align(decoded, output, config)
    annotations_path = work / "annotations.parquet"
    annotations.to_parquet(annotations_path, index=False)
    result.update({"version": 1, "format": "parquet", "layout": "wide", "timestamp": "timestamp", "timezone": "UTC", "_source_sorted_unique": True, "cadence_seconds": cadence, "max_gap_seconds": config["max_gap_seconds"], "numeric_features": [stat["channel"] for _, stat in decoded if stat["numeric"]], "categorical_features": [stat["channel"] for _, stat in decoded if not stat["numeric"]], "label_columns": [], "annotations_path": str(annotations_path), "annotation_format": "esa-intervals"})
    if profile == "causal":
        result["context_columns"] = ["comet_gap"]
        result["gap_column"] = "comet_gap"
    # Alignment has already performed bounded causal holding. Filling those
    # expired gaps again would double the configured observation-age limit.
    result.setdefault("preprocessing", {})["forward_fill"] = False
    result.setdefault("splits", {"test_start": boundary, "train_fraction": 0.85})
    departures = ["Empty channel partitions are retained without indexing an empty frame", "Unknown annotation categories map to gap and never reuse a preceding label", "Conflicting duplicate observations are rejected"]
    if profile == "esa-reference":
        departures.append("A duplicated floored test-boundary row is omitted from the combined source to retain a unique chronological train/test boundary")
    esa.update({"mission": mission, "version": version, "profile": profile, "dataset_url": ESA_RECORDS[version], "upstream_revision": ESA_REVISION, "adapter_version": ADAPTER_VERSION, "source_checksums": checksums, "test_boundary": boundary, "configuration": config, "source": str(source), **archive_info, "departures": departures})
    _save_json(marker, {"manifest": result})
    return output, result
