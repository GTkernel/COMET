from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.anomaly.detect import _json_dict, _load_model_ready_rows
from src.common.logging import get_app_logger
from src.common.paths import DIAGNOSTICS_DIR
from src.db import get_dataset, get_model, init_db
from src.modeling.lstm_autoencoder import LSTMAutoencoder, create_sliding_windows


def run_model_diagnostics(
    dataset_id: int,
    model_id: int,
    *,
    db_path: str | Path | None = None,
    output_dir: str | Path = DIAGNOSTICS_DIR,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Score model-ready windows and save diagnostic statistics and plots."""
    init_db(db_path)
    logger = get_app_logger()
    dataset = get_dataset(dataset_id, db_path=db_path)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    model_row = get_model(model_id, db_path=db_path)
    if model_row is None:
        raise ValueError(f"Model metadata for model {model_id} not found.")
    from src.db import get_detection_run
    from src.modeling.pipeline import diagnostics
    run = get_detection_run(run_id, dataset_id=dataset_id, model_id=model_id, db_path=db_path)
    if run is not None:
        return diagnostics(run["run_id"], db_path=db_path, output_dir=output_dir)
    if run_id is not None or _json_dict(model_row["metadata_json"]).get("pipeline_version") == 2:
        raise ValueError("Model diagnostics require a completed detection run for this target dataset and model.")
    if int(model_row["dataset_id"]) != dataset_id:
        raise ValueError(f"Model {model_id} was trained for dataset {model_row['dataset_id']}, not {dataset_id}.")

    model_path = Path(model_row["model_path"])
    if not model_path.exists():
        raise ValueError(f"Model file does not exist: {model_path}")

    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
    metadata = _json_dict(model_row["metadata_json"])
    model_config = checkpoint.get("config") or metadata.get("config") or {}
    feature_columns = checkpoint.get("feature_columns") or metadata.get("feature_columns")
    if not feature_columns:
        raise ValueError(f"Feature names cannot be resolved for model {model_id}.")

    threshold = float(model_row["threshold"])
    threshold_percentile = _optional_float(model_row.get("threshold_percentile") or checkpoint.get("threshold_percentile"))
    model_ready_table = checkpoint.get("model_ready_table") or f"model_ready_dataset_{dataset_id}"
    _, values = _load_model_ready_rows(dataset_id, model_ready_table, feature_columns, db_path=db_path)
    window_size = int(model_config.get("window_size", 100))
    stride = int(model_config.get("stride", 1))
    windows = create_sliding_windows(values, window_size=window_size, stride=stride)
    if windows.size(0) == 0:
        raise ValueError(
            "Reconstruction errors cannot be computed because no windows were created. "
            f"Need at least window_size={window_size} rows."
        )

    model = LSTMAutoencoder(
        num_features=len(feature_columns),
        hidden_size=int(model_config.get("hidden_size", 64)),
        num_layers=int(model_config.get("num_layers", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        reconstructed = model(windows)
        per_window_errors = torch.mean((reconstructed - windows) ** 2, dim=(1, 2))
        per_feature_errors = torch.mean((reconstructed - windows) ** 2, dim=1)

    errors = [float(value) for value in per_window_errors.tolist()]
    if not errors:
        raise ValueError("Reconstruction errors cannot be computed.")

    anomalous_indices = [index for index, error in enumerate(errors) if error > threshold]
    top_features = _top_contributing_features(
        feature_columns=feature_columns,
        per_feature_errors=per_feature_errors,
        anomalous_indices=anomalous_indices,
    )
    stats = _error_statistics(errors)
    anomaly_count = len(anomalous_indices)
    anomaly_percentage = (anomaly_count / len(errors)) * 100
    plot_paths = _write_plots(
        dataset_id=dataset_id,
        model_id=model_id,
        errors=errors,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        feature_columns=feature_columns,
        per_feature_errors=per_feature_errors,
        anomalous_indices=anomalous_indices,
        top_features=top_features,
        output_dir=Path(output_dir),
    )

    summary = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "model_file_path": str(model_path),
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "model_ready_table": model_ready_table,
        "window_size": window_size,
        "stride": stride,
        "reconstruction_error_statistics": stats,
        "windows_evaluated": len(errors),
        "windows_above_threshold": anomaly_count,
        "percentage_windows_above_threshold": anomaly_percentage,
        "top_contributing_features": top_features,
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }
    summary_path = Path(output_dir) / f"model_{model_id}_dataset_{dataset_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    _log_summary(logger, summary)
    return summary


def _error_statistics(errors: list[float]) -> dict[str, float]:
    return {
        "min": min(errors),
        "max": max(errors),
        "mean": statistics.fmean(errors),
        "median": statistics.median(errors),
        "standard_deviation": statistics.pstdev(errors),
    }


def _top_contributing_features(
    *,
    feature_columns: list[str],
    per_feature_errors: torch.Tensor,
    anomalous_indices: list[int],
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not anomalous_indices:
        return []

    selected = per_feature_errors[anomalous_indices]
    average_errors = torch.mean(selected, dim=0)
    ranked = sorted(
        zip(feature_columns, [float(value) for value in average_errors.tolist()], strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature, "average_reconstruction_error": score, "rank": rank}
        for rank, (feature, score) in enumerate(ranked[:limit], start=1)
    ]


def _write_plots(
    *,
    dataset_id: int,
    model_id: int,
    errors: list[float],
    threshold: float,
    threshold_percentile: float | None,
    feature_columns: list[str],
    per_feature_errors: torch.Tensor,
    anomalous_indices: list[int],
    top_features: list[dict[str, Any]],
    output_dir: Path,
    sampling_note: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"model_{model_id}_dataset_{dataset_id}"
    plot_paths = {
        "error_histogram": prefix.with_name(prefix.name + "_error_histogram.png"),
        "error_timeseries": prefix.with_name(prefix.name + "_error_timeseries.png"),
        "top_features": prefix.with_name(prefix.name + "_top_features.png"),
        "feature_heatmap": prefix.with_name(prefix.name + "_feature_heatmap.png"),
        "error_quantiles": prefix.with_name(prefix.name + "_error_quantiles.png"),
    }
    _plot_error_histogram(errors, threshold, plot_paths["error_histogram"], sampling_note=sampling_note)
    _plot_error_timeseries(errors, threshold, plot_paths["error_timeseries"], sampling_note=sampling_note)
    _plot_top_features(top_features, plot_paths["top_features"], sampling_note=sampling_note)
    _plot_feature_heatmap(
        feature_columns=feature_columns,
        per_feature_errors=per_feature_errors,
        anomalous_indices=anomalous_indices,
        top_features=top_features,
        path=plot_paths["feature_heatmap"],
        sampling_note=sampling_note,
    )
    _plot_error_quantiles(errors, threshold, threshold_percentile, plot_paths["error_quantiles"], sampling_note=sampling_note)
    return plot_paths


def _save_plot(fig, path: Path, sampling_note: str | None) -> None:
    if sampling_note:
        fig.suptitle(sampling_note, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Description": sampling_note or "All scored windows"})
    plt.close(fig)


def _plot_error_histogram(errors: list[float], threshold: float, path: Path, *, sampling_note: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=min(40, max(5, len(errors))), color="#4c78a8", edgecolor="white")
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=2, label=f"threshold={threshold:.6g}")
    upper_limit = _robust_upper_limit(errors, percentile=99.0, minimum=threshold * 1.25)
    if upper_limit is not None:
        ax.set_xlim(left=0, right=upper_limit)
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Count of windows")
    ax.set_title("Reconstruction Error Distribution (x-axis capped at p99)")
    ax.legend()
    _save_plot(fig, path, sampling_note)


def _plot_error_timeseries(errors: list[float], threshold: float, path: Path, *, sampling_note: str | None = None) -> None:
    indices = list(range(len(errors)))
    anomalous_indices = [index for index, error in enumerate(errors) if error > threshold]
    anomalous_errors = [errors[index] for index in anomalous_indices]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(indices, errors, color="#4c78a8", linewidth=1.5, label="reconstruction error")
    ax.scatter(anomalous_indices, anomalous_errors, color="#d62728", s=24, label="above threshold", zorder=3)
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=2, label=f"threshold={threshold:.6g}")
    upper_limit = _robust_upper_limit(errors, percentile=99.0, minimum=threshold * 1.25)
    if upper_limit is not None:
        ax.set_ylim(bottom=0, top=upper_limit)
    ax.set_xlabel("Sample position (see sampling caption)" if sampling_note else "Window index")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction Error Over Window Index (y-axis capped at p99)")
    ax.legend()
    _save_plot(fig, path, sampling_note)


def _plot_top_features(top_features: list[dict[str, Any]], path: Path, *, sampling_note: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if not top_features:
        ax.text(0.5, 0.5, "No windows above threshold", ha="center", va="center")
        ax.set_axis_off()
    else:
        features = [item["feature"] for item in reversed(top_features)]
        scores = [item["average_reconstruction_error"] for item in reversed(top_features)]
        ax.barh(features, scores, color="#59a14f")
        ax.set_xlabel("Average per-feature reconstruction error")
        ax.set_ylabel("Feature")
        ax.set_title("Top Contributing Features Across Anomalous Windows")
    _save_plot(fig, path, sampling_note)


def _plot_feature_heatmap(
    *,
    feature_columns: list[str],
    per_feature_errors: torch.Tensor,
    anomalous_indices: list[int],
    top_features: list[dict[str, Any]],
    path: Path,
    sampling_note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if not anomalous_indices or not top_features:
        ax.text(0.5, 0.5, "No anomalous windows to display", ha="center", va="center")
        ax.set_axis_off()
    else:
        top_feature_names = [item["feature"] for item in top_features]
        feature_indices = [feature_columns.index(feature) for feature in top_feature_names]
        heatmap = per_feature_errors[anomalous_indices][:, feature_indices].detach().cpu().numpy()
        vmax = _heatmap_upper_limit(heatmap)
        image = ax.imshow(heatmap, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(top_feature_names)))
        ax.set_xticklabels(top_feature_names, rotation=45, ha="right")
        y_tick_positions = _sample_tick_positions(len(anomalous_indices), max_ticks=20)
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels([str(anomalous_indices[index]) for index in y_tick_positions])
        ax.set_xlabel("Feature")
        ax.set_ylabel("Anomalous sample position (see sampling caption)" if sampling_note else "Anomalous window index")
        ax.set_title("Per-Feature Reconstruction Error Heatmap")
        fig.colorbar(image, ax=ax, label="Per-feature reconstruction error")
    _save_plot(fig, path, sampling_note)


def _sample_tick_positions(length: int, *, max_ticks: int) -> list[int]:
    if length <= 0:
        return []
    if length <= max_ticks:
        return list(range(length))
    step = math.ceil(length / max_ticks)
    positions = list(range(0, length, step))
    if positions[-1] != length - 1:
        positions.append(length - 1)
    return positions


def _heatmap_upper_limit(values: Any) -> float | None:
    flattened = [float(value) for row in values for value in row if math.isfinite(float(value))]
    if not flattened:
        return None
    flattened.sort()
    percentile_index = min(len(flattened) - 1, math.ceil(len(flattened) * 0.95) - 1)
    upper_limit = flattened[percentile_index]
    return upper_limit if upper_limit > 0 else None


def _robust_upper_limit(values: list[float], *, percentile: float, minimum: float) -> float | None:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return None
    index = min(len(finite_values) - 1, math.ceil(len(finite_values) * (percentile / 100.0)) - 1)
    upper_limit = max(finite_values[index], minimum)
    return upper_limit if upper_limit > 0 else None


def _plot_error_quantiles(
    errors: list[float],
    threshold: float,
    threshold_percentile: float | None,
    path: Path,
    *, sampling_note: str | None = None,
) -> None:
    sorted_errors = sorted(errors)
    if len(sorted_errors) == 1:
        percentiles = [100.0]
    else:
        percentiles = [(index / (len(sorted_errors) - 1)) * 100 for index in range(len(sorted_errors))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentiles, sorted_errors, color="#4c78a8", linewidth=1.5)
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=2, label=f"threshold={threshold:.6g}")
    if threshold_percentile is not None:
        ax.axvline(
            threshold_percentile,
            color="#d62728",
            linestyle=":",
            linewidth=2,
            label=f"threshold percentile={threshold_percentile:g}",
        )
    upper_limit = _robust_upper_limit(errors, percentile=99.0, minimum=threshold * 1.25)
    if upper_limit is not None:
        ax.set_ylim(bottom=0, top=upper_limit)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction Error Quantiles (y-axis capped at p99)")
    ax.legend()
    _save_plot(fig, path, sampling_note)


def _log_summary(logger, summary: dict[str, Any]) -> None:
    stats = summary["reconstruction_error_statistics"]
    logger.info(
        "model_diagnostics dataset_id=%s model_id=%s min=%s max=%s mean=%s median=%s std=%s "
        "threshold=%s threshold_percentile=%s windows=%s above_threshold=%s percent_above=%s top_features=%s",
        summary["dataset_id"],
        summary["model_id"],
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["median"],
        stats["standard_deviation"],
        summary["threshold"],
        summary["threshold_percentile"],
        summary["windows_evaluated"],
        summary["windows_above_threshold"],
        summary["percentage_windows_above_threshold"],
        summary["top_contributing_features"],
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
