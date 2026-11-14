from __future__ import annotations

import ast
import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from src.common.paths import EVALUATION_DIR
from src.db import get_connection, get_dataset, get_model, get_detection_run, init_db


@dataclass(frozen=True)
class GroundTruthInterval:
    index: int
    start: datetime
    end: datetime
    root_cause: str
    affected: list[str]
    category: str


@dataclass(frozen=True)
class DetectedAnomaly:
    anomaly_id: str
    start: datetime
    end: datetime
    anomaly_score: float
    threshold: float
    window_index: int


def evaluate_anomalies(
    dataset_id: int,
    model_id: int,
    metadata_path: str | Path,
    *,
    db_path: str | Path | None = None,
    output_dir: str | Path = EVALUATION_DIR,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare detected anomaly windows against labeled anomaly intervals."""
    init_db(db_path)
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        raise ValueError(f"Metadata file does not exist: {metadata_file}")

    dataset = get_dataset(dataset_id, db_path=db_path)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")
    model = get_model(model_id, db_path=db_path)
    if model is None:
        raise ValueError(f"Model {model_id} does not exist.")
    modern = json.loads(model["metadata_json"] or "{}").get("pipeline_version") == 2
    if not modern and int(model["dataset_id"]) != dataset_id:
        raise ValueError(f"Model {model_id} was trained for dataset {model['dataset_id']}, not {dataset_id}.")
    latest_run = get_detection_run(run_id, dataset_id=dataset_id, model_id=model_id, db_path=db_path)
    if modern or run_id:
        if latest_run is None:
            raise ValueError("Evaluation requires a completed run for this target dataset and model.")
        run_id = latest_run["run_id"]
        output_dir = Path(output_dir) / run_id
    elif latest_run is not None:
        run_id = latest_run["run_id"]
        output_dir = Path(output_dir) / run_id

    ground_truth = load_ground_truth_intervals(metadata_file)
    detections = load_detected_anomalies(dataset_id, model_id, run_id=run_id, db_path=db_path)
    matches = _match_intervals(ground_truth, detections)
    summary = _build_summary(dataset_id, model_id, metadata_file, ground_truth, detections, matches)
    summary.update(metric="legacy-overlap", run_id=run_id,
                   benchmark_note="COMET interval-overlap evaluation; not ESA benchmark performance.")
    plot_paths = _write_plots(
        dataset_id=dataset_id,
        model_id=model_id,
        ground_truth=ground_truth,
        detections=detections,
        matches=matches,
        summary=summary,
        output_dir=Path(output_dir),
    )
    summary["plots"] = {name: str(path) for name, path in plot_paths.items()}

    summary_path = Path(output_dir) / f"dataset_{dataset_id}_model_{model_id}_evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_ground_truth_intervals(metadata_path: str | Path) -> list[GroundTruthInterval]:
    intervals: list[GroundTruthInterval] = []
    with Path(metadata_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"start_time", "end_time", "root_cause", "affected", "category"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata file is missing required columns: {sorted(missing)}")
        for index, row in enumerate(reader, start=1):
            start = _parse_timestamp(row["start_time"])
            end = _parse_timestamp(row["end_time"])
            if end < start:
                raise ValueError(f"Metadata interval {index} has end_time before start_time.")
            intervals.append(
                GroundTruthInterval(
                    index=index,
                    start=start,
                    end=end,
                    root_cause=row["root_cause"],
                    affected=_parse_affected(row["affected"]),
                    category=str(row["category"]),
                )
            )
    return intervals


def load_detected_anomalies(
    dataset_id: int,
    model_id: int,
    *,
    db_path: str | Path | None = None,
    run_id: str | None = None,
) -> list[DetectedAnomaly]:
    from src.db import get_detection_run
    run = get_detection_run(run_id, dataset_id=dataset_id, model_id=model_id, db_path=db_path)
    if run_id is not None and run is None:
        raise ValueError("No matching completed detection run.")
    if run is not None:
        run_id = run["run_id"]
    run_filter = "AND run_id = ?" if run_id else "AND run_id IS NULL"
    parameters = (dataset_id, model_id, run_id) if run_id else (dataset_id, model_id)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT anomaly_id, start_timestamp, end_timestamp, score_timestamp, anomaly_score, threshold, window_index
            FROM anomalies
            WHERE dataset_id = ? AND model_id = ?
            {run_filter}
            ORDER BY window_index
            """,
            parameters,
        ).fetchall()

    detections = []
    for row in rows:
        if row["start_timestamp"] is None or row["end_timestamp"] is None:
            continue
        start = _parse_timestamp(row["score_timestamp"] or row["start_timestamp"])
        end = _parse_timestamp(row["end_timestamp"])
        if end < start:
            start, end = end, start
        detections.append(
            DetectedAnomaly(
                anomaly_id=row["anomaly_id"],
                start=start,
                end=end,
                anomaly_score=float(row["anomaly_score"]),
                threshold=float(row["threshold"]),
                window_index=int(row["window_index"]),
            )
        )
    return detections


def _match_intervals(
    ground_truth: list[GroundTruthInterval],
    detections: list[DetectedAnomaly],
) -> dict[str, Any]:
    gt_to_detections: dict[int, list[int]] = {interval.index: [] for interval in ground_truth}
    detection_to_gt: dict[str, list[int]] = {detection.anomaly_id: [] for detection in detections}

    for interval in ground_truth:
        for detection in detections:
            if _overlaps(interval.start, interval.end, detection.start, detection.end):
                gt_to_detections[interval.index].append(detection.window_index)
                detection_to_gt[detection.anomaly_id].append(interval.index)

    matched_gt = {index for index, window_indices in gt_to_detections.items() if window_indices}
    matched_detection_ids = {
        anomaly_id for anomaly_id, gt_indices in detection_to_gt.items() if gt_indices
    }
    missed_gt = {interval.index for interval in ground_truth} - matched_gt
    unmatched_detection_ids = {detection.anomaly_id for detection in detections} - matched_detection_ids

    delays = []
    first_detection_by_gt: dict[int, str] = {}
    detections_by_window = {detection.window_index: detection for detection in detections}
    for interval in ground_truth:
        window_indices = gt_to_detections[interval.index]
        if not window_indices:
            continue
        first_detection = min((detections_by_window[index] for index in window_indices), key=lambda item: item.start)
        first_detection_by_gt[interval.index] = first_detection.anomaly_id
        delay = max(0.0, (first_detection.start - interval.start).total_seconds())
        delays.append({"ground_truth_index": interval.index, "delay_seconds": delay})

    return {
        "gt_to_detections": gt_to_detections,
        "detection_to_gt": detection_to_gt,
        "matched_gt": matched_gt,
        "missed_gt": missed_gt,
        "matched_detection_ids": matched_detection_ids,
        "unmatched_detection_ids": unmatched_detection_ids,
        "first_detection_by_gt": first_detection_by_gt,
        "delays": delays,
    }


def _build_summary(
    dataset_id: int,
    model_id: int,
    metadata_path: Path,
    ground_truth: list[GroundTruthInterval],
    detections: list[DetectedAnomaly],
    matches: dict[str, Any],
) -> dict[str, Any]:
    matched_detections = len(matches["matched_detection_ids"])
    total_detections = len(detections)
    matched_gt = len(matches["matched_gt"])
    total_gt = len(ground_truth)
    precision = matched_detections / total_detections if total_detections else None
    recall = matched_gt / total_gt if total_gt else None
    f1 = None if precision is None or recall is None else (
        (2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    delay_values = [item["delay_seconds"] for item in matches["delays"]]

    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "metadata_path": str(metadata_path),
        "ground_truth_intervals": total_gt,
        "detected_anomalies": total_detections,
        "matched_ground_truth_intervals": matched_gt,
        "missed_ground_truth_intervals": total_gt - matched_gt,
        "matched_detected_anomalies": matched_detections,
        "unmatched_detected_anomalies": total_detections - matched_detections,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "undefined_metrics": [name for name, value in (("precision", precision), ("recall", recall), ("f1", f1)) if value is None],
        "mean_detection_delay_seconds": statistics.fmean(delay_values) if delay_values else None,
        "median_detection_delay_seconds": statistics.median(delay_values) if delay_values else None,
        "matched_ground_truth_indices": sorted(matches["matched_gt"]),
        "missed_ground_truth_indices": sorted(matches["missed_gt"]),
        "unmatched_anomaly_ids": sorted(matches["unmatched_detection_ids"]),
        "detection_delays": matches["delays"],
    }


def _write_plots(
    *,
    dataset_id: int,
    model_id: int,
    ground_truth: list[GroundTruthInterval],
    detections: list[DetectedAnomaly],
    matches: dict[str, Any],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"dataset_{dataset_id}_model_{model_id}"
    paths = {
        "timeline": prefix.with_name(prefix.name + "_timeline.png"),
        "score_by_time": prefix.with_name(prefix.name + "_score_by_time.png"),
        "score_distribution_matched_vs_unmatched": prefix.with_name(
            prefix.name + "_score_distribution_matched_vs_unmatched.png"
        ),
        "confusion_summary": prefix.with_name(prefix.name + "_confusion_summary.png"),
        "detection_delay_histogram": prefix.with_name(prefix.name + "_detection_delay_histogram.png"),
        "detection_delay_by_category": prefix.with_name(prefix.name + "_detection_delay_by_category.png"),
        "ground_truth_by_category": prefix.with_name(prefix.name + "_ground_truth_by_category.png"),
        "category_summary": prefix.with_name(prefix.name + "_category_summary.png"),
        "detections_by_root_cause": prefix.with_name(prefix.name + "_detections_by_root_cause.png"),
        "missed_intervals": prefix.with_name(prefix.name + "_missed_intervals.png"),
    }
    _plot_timeline(ground_truth, detections, matches, paths["timeline"])
    _plot_score_by_time(ground_truth, detections, matches, paths["score_by_time"])
    _plot_score_distribution(detections, matches, paths["score_distribution_matched_vs_unmatched"])
    _plot_confusion_summary(summary, paths["confusion_summary"])
    _plot_detection_delay_histogram(matches, paths["detection_delay_histogram"])
    _plot_detection_delay_by_category(ground_truth, matches, paths["detection_delay_by_category"])
    _plot_ground_truth_by_category(ground_truth, paths["ground_truth_by_category"])
    _plot_category_summary(ground_truth, matches, paths["category_summary"])
    _plot_detections_by_root_cause(ground_truth, detections, matches, paths["detections_by_root_cause"])
    _plot_missed_intervals(ground_truth, matches, paths["missed_intervals"])
    return paths


def _plot_timeline(
    ground_truth: list[GroundTruthInterval],
    detections: list[DetectedAnomaly],
    matches: dict[str, Any],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for interval in ground_truth:
        color = "#59a14f" if interval.index in matches["matched_gt"] else "#e15759"
        ax.plot([interval.start, interval.end], [1, 1], color=color, linewidth=6, solid_capstyle="butt")
    for detection in detections:
        color = "#4e79a7" if detection.anomaly_id in matches["matched_detection_ids"] else "#f28e2b"
        ax.plot([detection.start, detection.end], [0, 0], color=color, linewidth=4, solid_capstyle="butt")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Detected windows", "Ground truth intervals"])
    ax.set_title("Ground Truth and Detected Anomaly Timeline")
    _format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_score_by_time(
    ground_truth: list[GroundTruthInterval],
    detections: list[DetectedAnomaly],
    matches: dict[str, Any],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for interval in ground_truth:
        ax.axvspan(interval.start, interval.end, color="#e15759", alpha=0.12)
    if detections:
        times = [detection.start for detection in detections]
        scores = [detection.anomaly_score for detection in detections]
        colors = ["#59a14f" if detection.anomaly_id in matches["matched_detection_ids"] else "#f28e2b" for detection in detections]
        ax.scatter(times, scores, c=colors, s=32)
        ax.axhline(detections[0].threshold, color="#d62728", linestyle="--", linewidth=1.5, label="threshold")
    ax.set_title("Detected Anomaly Scores Over Time")
    ax.set_xlabel("Detection start time")
    ax.set_ylabel("Anomaly score")
    ax.legend(loc="best")
    _format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_score_distribution(
    detections: list[DetectedAnomaly],
    matches: dict[str, Any],
    path: Path,
) -> None:
    matched = [d.anomaly_score for d in detections if d.anomaly_id in matches["matched_detection_ids"]]
    unmatched = [d.anomaly_score for d in detections if d.anomaly_id in matches["unmatched_detection_ids"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([matched, unmatched], label=["matched", "unmatched"], bins=20, color=["#59a14f", "#f28e2b"])
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion_summary(summary: dict[str, Any], path: Path) -> None:
    labels = ["matched detections", "unmatched detections", "missed intervals"]
    values = [
        summary["matched_detected_anomalies"],
        summary["unmatched_detected_anomalies"],
        summary["missed_ground_truth_intervals"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#59a14f", "#f28e2b", "#e15759"])
    ax.set_title("Evaluation Summary Counts")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_detection_delay_histogram(matches: dict[str, Any], path: Path) -> None:
    delays = [item["delay_seconds"] for item in matches["delays"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    if delays:
        ax.hist(delays, bins=min(20, max(5, len(delays))), color="#4e79a7", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "No matched intervals", ha="center", va="center")
    ax.set_title("Detection Delay Distribution")
    ax.set_xlabel("Delay seconds")
    ax.set_ylabel("Matched intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_detection_delay_by_category(
    ground_truth: list[GroundTruthInterval],
    matches: dict[str, Any],
    path: Path,
) -> None:
    delay_values = _median_detection_delays_by_category(ground_truth, matches)
    categories = list(delay_values)
    medians = [delay_values[category] for category in categories]
    fig, ax = plt.subplots(figsize=(9, 5))
    if categories:
        ax.bar(categories, medians, color="#4e79a7")
    else:
        ax.text(0.5, 0.5, "No matched intervals", ha="center", va="center")
    ax.set_title("Median Detection Delay by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Median delay [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _median_detection_delays_by_category(
    ground_truth: list[GroundTruthInterval],
    matches: dict[str, Any],
) -> dict[str, float]:
    gt_by_index = {interval.index: interval for interval in ground_truth}
    values: dict[str, list[float]] = {}
    for item in matches["delays"]:
        category = gt_by_index[item["ground_truth_index"]].category
        values.setdefault(category, []).append(item["delay_seconds"])
    return {
        category: statistics.median(values[category])
        for category in sorted(values, key=_category_sort_key)
    }


def _plot_ground_truth_by_category(ground_truth: list[GroundTruthInterval], path: Path) -> None:
    counts = _count_by([interval.category for interval in ground_truth])
    _plot_bar(counts, "Distribution of Anomalies by Category", "Category", "Anomaly events [#]", path, rotation=0)


def _plot_category_summary(
    ground_truth: list[GroundTruthInterval],
    matches: dict[str, Any],
    path: Path,
) -> None:
    counts = _count_by([interval.category for interval in ground_truth])
    delay_values = _median_detection_delays_by_category(ground_truth, matches)
    categories = sorted(counts, key=_category_sort_key)

    fig, ax = plt.subplots(figsize=(9, 5))

    if categories:
        valid_delays = [delay for category, delay in delay_values.items() if category in counts]
        max_delay = max(valid_delays) if valid_delays else 1.0
        norm = mcolors.Normalize(vmin=0.0, vmax=max_delay)
        cmap = plt.get_cmap("plasma")
        colors = [
            cmap(norm(delay_values[category])) if category in delay_values else "#d9d9d9"
            for category in categories
        ]
        ax.bar(categories, [counts[category] for category in categories], color=colors)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        colorbar.set_label("Median delay [s]")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")

    ax.set_title("Distribution of Anomalies by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Anomaly events [#]")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_detections_by_root_cause(
    ground_truth: list[GroundTruthInterval],
    detections: list[DetectedAnomaly],
    matches: dict[str, Any],
    path: Path,
) -> None:
    gt_by_index = {interval.index: interval for interval in ground_truth}
    root_causes = []
    for detection in detections:
        gt_indices = matches["detection_to_gt"].get(detection.anomaly_id) or []
        if gt_indices:
            root_causes.append(gt_by_index[gt_indices[0]].root_cause)
        else:
            root_causes.append("unmatched")
    counts = _count_by(root_causes)
    _plot_bar(counts, "Detected Windows by Matched Root Cause", "Root cause", "Detected windows", path)


def _plot_missed_intervals(
    ground_truth: list[GroundTruthInterval],
    matches: dict[str, Any],
    path: Path,
) -> None:
    missed = [interval for interval in ground_truth if interval.index in matches["missed_gt"]][:20]
    fig, ax = plt.subplots(figsize=(12, max(3, 0.35 * max(1, len(missed)))))
    ax.set_axis_off()
    if not missed:
        ax.text(0.5, 0.5, "No missed ground-truth intervals", ha="center", va="center")
    else:
        rows = [
            [interval.index, interval.start.isoformat(sep=" "), interval.end.isoformat(sep=" "), interval.root_cause, interval.category]
            for interval in missed
        ]
        table = ax.table(
            cellText=rows,
            colLabels=["Index", "Start", "End", "Root cause", "Category"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
    ax.set_title("Missed Ground-Truth Intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_bar(
    counts: dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    *,
    rotation: int = 45,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if counts:
        labels = list(counts)
        values = [counts[label] for label in labels]
        ax.bar(labels, values, color="#4e79a7")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _count_by(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda item: _category_sort_key(item[0])))

def _category_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _parse_affected(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _overlaps(left_start: datetime, left_end: datetime, right_start: datetime, right_end: datetime) -> bool:
    return left_start <= right_end and right_start <= left_end


def _format_time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
