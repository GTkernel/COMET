"""Build diagnostic reports from recorded scores and batched window replay."""
from __future__ import annotations

import json
import math
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from src.db import get_connection, get_model
from src.ingestion.store import _parquet_batches, _quote
from src.modeling.window_dataset import WindowDataset

PLOT_SAMPLE_LIMIT = 10000


def _legacy_windows(run, checkpoint, features, directory, *, db_path):
    """Verify legacy inputs against replay because historical runs did not freeze them."""
    table = f"model_ready_dataset_{run['dataset_id']}"
    count = int(run["row_count"])
    values = np.lib.format.open_memmap(directory / "legacy-values.npy", mode="w+",
                                      dtype="float32", shape=(count, len(features)))
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            f"SELECT {','.join(_quote(name) for name in features)} FROM {_quote(table)} "
            "WHERE dataset_id=? ORDER BY row_id", (run["dataset_id"],))
        offset = 0
        while batch := cursor.fetchmany(10000):
            if offset + len(batch) > count:
                raise ValueError("Legacy diagnostic inputs changed since detection")
            values[offset:offset + len(batch)] = [[float(value) for value in row] for row in batch]
            offset += len(batch)
    if offset != count:
        raise ValueError("Legacy diagnostic inputs changed since detection")
    config = checkpoint.get("config") or {}
    size = int(run.get("window_size", config.get("window_size", 100)))
    stride = int(run.get("stride", config.get("stride", 1)))
    starts = np.lib.format.open_memmap(directory / "legacy-starts.npy", mode="w+", dtype="int64",
                                      shape=(int(run["windows_scored"]),))
    for begin in range(0, len(starts), 10000):
        starts[begin:begin + 10000] = np.arange(begin, min(begin + 10000, len(starts))) * stride
    windows = WindowDataset(values, starts, size)
    windows._diagnostic_maps = (values, starts)
    return windows


def _windows(run, model_row, checkpoint, directory, *, db_path):
    from src.modeling.pipeline import _hash_file
    from src.modeling.transforms import fingerprint

    features = checkpoint.get("feature_columns") or json.loads(model_row["metadata_json"]).get("feature_columns")
    if not features:
        raise ValueError("Model has no recorded diagnostic feature names")
    if run.get("pipeline_version") != 2:
        return _legacy_windows(run, checkpoint, features, directory, db_path=db_path), features
    metadata = json.loads(model_row["metadata_json"])
    if _hash_file(Path(model_row["model_path"])) != run["model_fingerprint"]:
        raise ValueError("Diagnostic checkpoint fingerprint mismatch")
    state = json.loads(Path(metadata["transform_path"]).read_text(encoding="utf-8"))
    actual = fingerprint({key: value for key, value in state.items() if key != "fingerprint"})
    if actual != run["transform_fingerprint"] or checkpoint["transform_fingerprint"] != actual:
        raise ValueError("Diagnostic preprocessing fingerprint mismatch")
    features = list(features)
    for feature in state["features"]:
        for index in range(feature["offset"], feature["offset"] + feature["width"]):
            features[index] = f"{feature['source_name']} [{features[index]}]"
    root = Path(run["artifact_dir"])
    values = np.load(root / "values.npy", mmap_mode="r")
    starts = np.load(root / "windows.npy", mmap_mode="r")
    config = checkpoint["config"]
    offsets = [feature["offset"] for feature in state["features"] if feature["role"] == "numeric"] if run.get("window_centering", False) else None
    windows = WindowDataset(values, starts[:int(run["windows_scored"])], int(config["window_size"]), numeric_offsets=offsets)
    windows._diagnostic_maps = (values, starts)
    return windows, features


def enrich_diagnostics(result: dict, run: dict, *, db_path=None, output_dir=None) -> dict:
    """Exact disk-backed statistics, full feature averages and bounded plot data."""
    import torch
    from src.anomaly.packet import _excluded_feature, _legacy_excluded_names
    from src.modeling.diagnostics import _write_plots
    from src.modeling.lstm_autoencoder import LSTMAutoencoder
    from src.modeling.pipeline import _hash_file
    from src.modeling.scoring import score_batches
    from src.common.paths import SCRATCH_DIR

    count = int(result["score_count"])
    model_row = get_model(run["model_id"], db_path=db_path)
    if model_row is None:
        raise ValueError("Diagnostic model no longer exists")
    checkpoint_hash = _hash_file(Path(model_row["model_path"]))
    checkpoint = torch.load(model_row["model_path"], weights_only=True, map_location="cpu")
    config = checkpoint.get("config") or json.loads(model_row["metadata_json"]).get("config") or {}
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="comet-diagnostics-", dir=SCRATCH_DIR) as temporary, ExitStack() as maps:
        directory = Path(temporary)
        # Use the chronological map for replay checks. Sort the compact map
        # in place for an exact median without an input-sized RAM allocation.
        chronological = np.lib.format.open_memmap(directory / "chronological.npy", mode="w+", dtype="float64",
                                                  shape=(int(run["row_count"]),))
        maps.callback(chronological._mmap.close)
        scores = np.lib.format.open_memmap(directory / "scores.npy", mode="w+", dtype="float64", shape=(count,))
        maps.callback(scores._mmap.close)
        row_offset = score_offset = 0
        mean = m2 = 0.0
        for batch in _parquet_batches(run["scores_path"], columns=["score"], batch_size=10000):
            data = batch.column(0).to_pylist()
            values = np.asarray([np.nan if value is None else value for value in data], dtype="float64")
            chronological[row_offset:row_offset + len(values)] = values
            row_offset += len(values)
            usable = values[np.isfinite(values)]
            if len(usable):
                scores[score_offset:score_offset + len(usable)] = usable
                batch_mean = float(usable.mean())
                delta = batch_mean - mean
                total = score_offset + len(usable)
                m2 += float(np.square(usable - batch_mean).sum()) + delta * delta * score_offset * len(usable) / total
                mean += delta * len(usable) / total
                score_offset = total
        if score_offset != count or row_offset != len(chronological):
            raise ValueError("Diagnostic score artifact changed while reading")
        if count:
            lower, upper = (count - 1) // 2, count // 2
            scores.partition((lower, upper))
            median = float(scores[lower] / 2 + scores[upper] / 2)
        else:
            median = None
        stats = {"min": result["score_min"], "max": result["score_max"], "mean": result["score_mean"],
                 "median": median, "standard_deviation": math.sqrt(max(0.0, m2 / count)) if count else None}
        windows, features = _windows(run, model_row, checkpoint, directory, db_path=db_path)
        for array in windows._diagnostic_maps:
            maps.callback(array._mmap.close)
        if len(windows) != count:
            raise ValueError("Diagnostic window count differs from recorded scores")
        model = LSTMAutoencoder(len(features), int(config.get("hidden_size", 64)), int(config.get("num_layers", 1)))
        model.load_state_dict(checkpoint["model_state_dict"])
        blocked_names = _legacy_excluded_names(run["dataset_id"], db_path=db_path) if run.get("pipeline_version") != 2 else set()
        safe_indices = [index for index, name in enumerate(features) if not _excluded_feature(name, blocked_names)]
        safe_features = [features[index] for index in safe_indices]
        feature_totals = np.zeros(len(safe_features), dtype="float64")
        sampled_errors, sampled_features = [], []
        sample_stride = max(1, math.ceil(count / PLOT_SAMPLE_LIMIT))
        anomalous_count = 0
        for offset, replay, per_feature in score_batches(model, windows, int(run.get("batch_size", config.get("batch_size", 128)))):
            endpoints = np.asarray(windows.starts[offset:offset + len(replay)]) + windows.window_size - 1
            recorded = chronological[endpoints]
            if not np.allclose(replay, recorded, rtol=1e-5, atol=1e-7, equal_nan=False):
                raise ValueError("Diagnostic inputs or checkpoint changed since detection. Recorded scores do not match replay")
            above = recorded > float(run["threshold"])
            feature_totals += per_feature[above][:, safe_indices].sum(axis=0, dtype="float64")
            anomalous_count += int(above.sum())
            for local in range(len(replay)):
                if (offset + local) % sample_stride == 0:
                    sampled_errors.append(float(recorded[local]))
                    sampled_features.append(per_feature[local, safe_indices].copy())
        if anomalous_count != int(run["anomalies_detected"]):
            raise ValueError("Diagnostic anomaly count differs from the selected run")
        if _hash_file(Path(model_row["model_path"])) != checkpoint_hash:
            raise ValueError("Diagnostic checkpoint changed while reading")
        averages = feature_totals / anomalous_count if anomalous_count else feature_totals
        top = [{"feature": safe_features[index], "average_reconstruction_error": float(averages[index]), "rank": rank}
               for rank, index in enumerate(sorted(range(len(averages)), key=lambda index: averages[index], reverse=True)[:10], 1)] if anomalous_count else []
        result.update(reconstruction_error_statistics=stats, score_median=median,
                      score_standard_deviation=stats["standard_deviation"], windows_evaluated=count,
                      windows_above_threshold=anomalous_count,
                      percentage_windows_above_threshold=100 * anomalous_count / count if count else None,
                      top_contributing_features=top, model_file_path=model_row["model_path"],
                      threshold_percentile=model_row.get("threshold_percentile"),
                      window_size=windows.window_size, stride=run.get("scoring_stride", run.get("stride", 1)),
                      diagnostic_replay_verified=True,
                      redacted_annotation_feature_count=len(features) - len(safe_features),
                      plot_sampling={"method": "every_nth_scored_window", "stride": sample_stride,
                                     "sampled_windows": len(sampled_errors), "total_windows": count,
                                     "approximate_plots": sample_stride > 1,
                                     "note": "Statistics and feature rankings use every scored window. Plots use the recorded sample."})
        if output_dir is not None:
            plot_dir = Path(output_dir) / f"run_{run['run_id']}"
            plots = _write_plots(dataset_id=run["dataset_id"], model_id=run["model_id"], errors=sampled_errors,
                                 threshold=float(run["threshold"]), threshold_percentile=model_row.get("threshold_percentile"),
                                 feature_columns=safe_features,
                                 per_feature_errors=torch.as_tensor(np.asarray(sampled_features).reshape(len(sampled_errors), len(safe_features))),
                                 anomalous_indices=[index for index, score in enumerate(sampled_errors) if score > run["threshold"]],
                                 top_features=top, output_dir=plot_dir,
                                 sampling_note=(f"Plot sample: {len(sampled_errors)} / {count} scored windows, stride {sample_stride}. "
                                                "feature rankings use all anomalous windows") if sample_stride > 1 else None)
            result["plots"] = {name: str(path) for name, path in plots.items()}
    return result
