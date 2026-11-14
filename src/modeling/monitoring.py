"""Walk-forward updates on recorded telemetry, using strictly earlier history."""
from __future__ import annotations

import contextlib
import csv
import json
import math
import hashlib
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.ingestion.manifest import is_label
from src.ingestion.store import (CORE, _atomic_json, _existing, _import_lock,
                                 _parquet_parts, _ParquetPartWriter, _publish_catalog,
                                 _root, _sha256, iter_batches, load_dataset)
from src.modeling.pipeline import DEFAULT_CONFIG, _iso_timestamp, _load_config, detect, train
from src.modeling.transforms import _timestamp_ns, fingerprint


def _range_batches(dataset, start, end, *, db_path, entity_id):
    names = CORE + [c["name"] for c in dataset["columns"]
                    if c["role"] not in {"label", "annotation"} and not is_label(c["source_name"])]
    if dataset.get("storage_version") == 2:
        def source():
            for part in _parquet_parts(dataset["telemetry_path"]):
                with contextlib.closing(pq.ParquetFile(part)) as reader:
                    time_column = reader.schema_arrow.get_field_index("timestamp_ns")
                    groups = []
                    for group in range(reader.num_row_groups):
                        stats = reader.metadata.row_group(group).column(time_column).statistics
                        if stats is not None and stats.has_min_max and (stats.max < start or stats.min >= end):
                            continue
                        groups.append(group)
                    yield from reader.iter_batches(columns=names, row_groups=groups,
                                                   batch_size=10000, use_threads=False)
    else:
        def source():
            yield from iter_batches(dataset["dataset_id"], db_path=db_path)
    with contextlib.closing(source()) as batches:
        for batch in batches:
            times = batch.column("timestamp_ns")
            mask = pc.and_(pc.greater_equal(times, start), pc.less(times, end))
            if entity_id is not None:
                mask = pc.and_(mask, pc.equal(batch.column("entity_id"), entity_id))
            selected = batch.filter(mask)
            if selected.num_rows:
                yield selected


def _snapshot(sources, manifest, *, db_path, storage_dir, entity_id):
    """Publish an immutable, label-free canonical snapshot with parent identities."""
    from src.modeling.provenance import dataset_provenance

    parents = [{"dataset_id": d["dataset_id"], "start_ns": start, "end_ns": end,
                "provenance": dataset_provenance(d, iter_batches(d["dataset_id"], db_path=db_path)
                                                if d.get("storage_version") != 2 else None)}
               for d, start, end in sources]
    identity = fingerprint({"monitor_snapshot_version": 1, "parents": parents,
                            "manifest": manifest, "entity_id": entity_id})
    existing = _existing(identity, db_path)
    if existing:
        return existing
    root = _root(storage_dir) / "monitoring" / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / identity
    with _import_lock(root / (identity + ".lock")):
        existing = _existing(identity, db_path)
        if existing:
            return existing
        directory.mkdir(exist_ok=True)
        columns = [c for c in sources[0][0]["columns"]
                   if c["role"] not in {"label", "annotation"} and not is_label(c["source_name"])]
        rows, writer = 0, None
        previous_time, selected_entity = None, entity_id
        with contextlib.ExitStack() as resources:
            for dataset, start, end in sources:
                available = {c["source_name"]: c for c in dataset["columns"]}
                for column in columns:
                    found = available.get(column["source_name"])
                    if (found is None and column["role"] == "context"
                            and column["source_name"] != manifest.get("gap_column")):
                        continue
                    if found is None or (found["role"], found.get("unit")) != (column["role"], column.get("unit")):
                        raise ValueError("Monitoring history and target have incompatible required columns")
                batches = resources.enter_context(contextlib.closing(
                    _range_batches(dataset, start, end, db_path=db_path, entity_id=entity_id)))
                for batch in batches:
                    arrays = [batch.column(name) for name in CORE]
                    for column in columns:
                        found = available.get(column["source_name"])
                        arrays.append(batch.column(found["name"]) if found else pa.nulls(
                            batch.num_rows, type=schema.field(column["name"]).type if writer else pa.string()))
                    batch = pa.RecordBatch.from_arrays(arrays,
                                                       names=CORE + [c["name"] for c in columns])
                    entities = batch.column("entity_id").to_pylist()
                    times = batch.column("timestamp_ns").to_numpy()
                    selected_entity = entities[0] if selected_entity is None else selected_entity
                    if any(value != selected_entity for value in entities):
                        raise ValueError("Monitor one entity at a time. Supply --entity-id")
                    if (previous_time is not None and times[0] <= previous_time) or (times[1:] <= times[:-1]).any():
                        raise ValueError("Monitoring snapshots require strictly increasing timestamps")
                    previous_time = int(times[-1])
                    if writer is None:
                        schema = batch.schema
                        writer = resources.enter_context(_ParquetPartWriter(directory / "telemetry.parquet", schema))
                    arrays = [pa.array(range(rows + 1, rows + batch.num_rows + 1), type=pa.int64()),
                              *[batch.column(name) for name in schema.names[1:]]]
                    writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
                    rows += batch.num_rows
            if writer is None:
                raise ValueError("No telemetry in the requested monitoring interval")
        for (dataset, _, _), parent in zip(sources, parents):
            if dataset.get("storage_version") != 2 and dataset_provenance(
                    dataset, iter_batches(dataset["dataset_id"], db_path=db_path)) != parent["provenance"]:
                raise ValueError("Legacy telemetry changed during snapshot creation. Retry with a stable source")
        source = directory / "snapshot.json"
        _atomic_json(source, {"parents": parents, "manifest": manifest, "entity_id": selected_entity,
                              "parts": [{"name": p.name, "sha256": _sha256(p)}
                                        for p in _parquet_parts(directory / "telemetry.parquet")]})
        metadata = {"storage_version": 2, "row_count": rows, "source_rows": rows,
                    "telemetry_path": str(directory / "telemetry.parquet"), "columns": columns,
                    "manifest": manifest, "import_fingerprint": identity, "source_sha256": _sha256(source),
                    "source_path": str(source), "monitoring_parents": parents,
                    "monitoring_entity_id": selected_entity}
        _atomic_json(directory / "metadata.json", metadata)
        dataset_id = _publish_catalog(source, metadata, db_path)
    return load_dataset(dataset_id, db_path=db_path)


def monitor(dataset_id, start, end, *, history_dataset_id=None, history_days=90,
            validation_days=14, update_days=7, config_path=None, entity_id=None,
            db_path=None, storage_dir=None, output_path=None):
    """Retrain on recent past rows, then score the next disjoint recorded batch."""
    from src.db import init_db
    from src.evaluation.export import export_predictions
    from src.modeling.provenance import dataset_provenance
    from src.modeling.pipeline import get_run

    init_db(db_path)
    if output_path and Path(output_path).suffix.lower() != ".csv":
        raise ValueError("Monitoring output must be a .csv file")
    durations = [float(value) for value in (history_days, validation_days, update_days)]
    if any(not math.isfinite(value) or value <= 0 for value in durations) or durations[0] <= durations[1]:
        raise ValueError("Monitoring durations must be positive, with history_days greater than validation_days")
    converted = [value * 86400 * 1_000_000_000 for value in durations]
    if any(not math.isfinite(value) or value > 2**63 - 1 for value in converted):
        raise ValueError("Monitoring durations exceed the supported timestamp range")
    history_ns, validation_ns, update_ns = map(int, converted)
    if min(history_ns, validation_ns, update_ns) < 1:
        raise ValueError("Monitoring durations must be at least one nanosecond")
    begin, stop = _timestamp_ns(start), _timestamp_ns(end)
    if begin - history_ns <= -(2**63) or stop == -(2**63):
        raise ValueError("Monitoring timestamps and history must fit the supported timestamp range")
    if stop <= begin:
        raise ValueError("Monitoring end must follow start")
    target = load_dataset(dataset_id, db_path=db_path)
    history = load_dataset(history_dataset_id or dataset_id, db_path=db_path)
    for dataset in (history, target):
        if dataset.get("manifest", {}).get("ordering") == "row":
            raise ValueError("Calendar-based monitoring requires real timestamps. This dataset preserves row order only")
        if (dataset.get("manifest", {}).get("esa") or {}).get("profile") == "esa-reference":
            raise ValueError("Operational monitoring requires causal data. ESA-reference preprocessing uses future observations")
    config = _load_config(config_path, defaults={"stride": DEFAULT_CONFIG["window_size"],
                                                "feature_eligibility": "observed_training", "window_centering": True})
    settings = {"monitoring_version": 1, "dataset_id": dataset_id, "history_dataset_id": history["dataset_id"],
                "start_ns": begin, "end_ns": stop, "history_days": durations[0],
                "validation_days": durations[1], "update_days": durations[2],
                "entity_id": entity_id, "config": config,
                "parents": [dataset_provenance(d, iter_batches(d["dataset_id"], db_path=db_path)
                                                if d.get("storage_version") != 2 else None)
                            for d in (history, target)]}
    directory = _root(storage_dir) / "monitoring" / fingerprint(settings)
    directory.mkdir(parents=True, exist_ok=True)
    with _import_lock(directory / "monitor.lock"):
        def validate_parents():
            for dataset, expected in zip((history, target), settings["parents"]):
                if dataset.get("storage_version") != 2 and dataset_provenance(
                        dataset, iter_batches(dataset["dataset_id"], db_path=db_path)) != expected:
                    raise ValueError("Legacy monitoring source changed. Retry with a stable source")

        config_file = directory / "config.json"
        _atomic_json(config_file, config)
        progress = directory / "progress.json"
        periods = json.loads(progress.read_text())["periods"] if progress.exists() else []
        completed = {period["cutoff"] for period in periods}
        for cutoff in range(begin, stop, update_ns):
            validate_parents()
            if _iso_timestamp(cutoff) in completed:
                previous = next(p for p in periods if p["cutoff"] == _iso_timestamp(cutoff))
                saved_run = get_run(previous["run_id"], db_path=db_path)
                if (saved_run["dataset_id"], saved_run["model_id"]) != (
                        previous["target_dataset_id"], previous["model_id"]):
                    raise ValueError("A completed monitoring period has inconsistent run provenance")
                previous.setdefault("threshold", saved_run["threshold"])
                if previous["threshold"] != saved_run["threshold"]:
                    raise ValueError("A completed monitoring period has an inconsistent threshold")
                if _sha256(Path(previous["export"]["path"])) != previous.get("export_sha256"):
                    raise ValueError("A completed monitoring export changed. Restore it before resuming")
                continue
            future_end = min(cutoff + update_ns, stop)
            manifest = json.loads(json.dumps(history.get("manifest") or {}))
            manifest.pop("annotations_path", None)
            manifest.pop("point_annotations_path", None)
            manifest["label_columns"] = []
            manifest["preprocessing"] = {**manifest.get("preprocessing", {}), "exclude_training_anomalies": False}
            manifest["splits"] = {"train_end": _iso_timestamp(cutoff - validation_ns - 1),
                                  "validation_end": _iso_timestamp(cutoff - 1)}
            manifest["_splits_explicit"] = True
            sources = [(history, cutoff - history_ns, min(cutoff, begin))]
            if cutoff > begin:
                sources.append((target, max(begin, cutoff - history_ns), cutoff))
            sources = [(d, left, right) for d, left, right in sources if left < right]
            training = _snapshot(sources, manifest, db_path=db_path, storage_dir=storage_dir, entity_id=entity_id)
            validate_parents()
            trained = train(training["dataset_id"], config_file, db_path=db_path, storage_dir=storage_dir)
            target_manifest = json.loads(json.dumps(target.get("manifest") or {}))
            target_manifest.pop("annotations_path", None)
            target_manifest.pop("point_annotations_path", None)
            target_manifest["label_columns"] = []
            target_manifest["splits"] = {"fractions": [.7, .15, .15]}
            target_manifest["_splits_explicit"] = False
            target_manifest["preprocessing"] = {**target_manifest.get("preprocessing", {}),
                                                 "exclude_training_anomalies": False}
            future = _snapshot([(target, cutoff, future_end)], target_manifest,
                               db_path=db_path, storage_dir=storage_dir,
                               entity_id=training["monitoring_entity_id"])
            validate_parents()
            detected = detect(future["dataset_id"], trained["model_id"], db_path=db_path,
                              storage_dir=storage_dir, partition="all")
            prediction_file = directory / f"{cutoff}-predictions.csv"
            exported = export_predictions(detected, prediction_file)
            period = {"cutoff": _iso_timestamp(cutoff), "end": _iso_timestamp(future_end),
                      "history_start": _iso_timestamp(cutoff - history_ns),
                      "training_dataset_id": training["dataset_id"], "target_dataset_id": future["dataset_id"],
                      "model_id": trained["model_id"], "run_id": detected["run_id"], "export": exported}
            period["excluded_features"] = trained["excluded_features"]
            period["training_partitions"] = trained["partitions"]
            period["threshold"] = trained["threshold"]
            period["export_sha256"] = _sha256(prediction_file)
            periods.append(period)
            completed.add(period["cutoff"])
            _atomic_json(directory / "progress.json", {"settings": settings, "periods": periods})
            print(f"Monitoring period {period['cutoff']} completed: model {trained['model_id']}, run {detected['run_id']}",
                  file=sys.stderr, flush=True)
        validate_parents()
        destination = Path(output_path) if output_path else directory / "predictions.csv"
        if any(destination.resolve() == Path(p["export"]["path"]).resolve() for p in periods):
            raise ValueError("Combined predictions must not replace an immutable period export")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".csv.partial")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as output:
                writer, expected_fields = None, None
                for period in periods:
                    digest = hashlib.sha256()
                    with Path(period["export"]["path"]).open("rb") as source:
                        def lines():
                            for line in source:
                                digest.update(line)
                                yield line.decode("utf-8")
                        reader = csv.DictReader(lines())
                        if writer is None:
                            expected_fields = reader.fieldnames
                            writer = csv.DictWriter(output, fieldnames=[*expected_fields, "dataset_id", "model_id", "run_id", "threshold"])
                            writer.writeheader()
                        elif reader.fieldnames != expected_fields:
                            raise ValueError("Monitoring period exports have inconsistent headers")
                        count = alerts = 0
                        for row in reader:
                            writer.writerow({**row, "dataset_id": period["target_dataset_id"],
                                             "model_id": period["model_id"], "run_id": period["run_id"],
                                             "threshold": period["threshold"]})
                            count += 1
                            alerts += row["is_anomaly"] == "1"
                        if digest.hexdigest() != period["export_sha256"]:
                            raise ValueError("A monitoring export changed while combining predictions")
                        if count != period["export"]["rows"] or alerts != period["export"]["alert_rows"]:
                            raise ValueError("Monitoring period export counts do not match its completed run")
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        result = {"mode": "operational_walk_forward", "settings": settings, "periods": periods,
                  "artifact_dir": str(directory), "predictions_path": str(destination.resolve()),
                  "rows": sum(p["export"]["rows"] for p in periods),
                  "alert_rows": sum(p["export"]["alert_rows"] for p in periods),
                  "not_esa_benchmark": True}
        _atomic_json(directory / "monitoring.json", result)
        _atomic_json(destination.with_suffix(".csv.json"), result)
        return result
