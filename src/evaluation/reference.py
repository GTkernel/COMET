"""Isolated, pinned ESA global evaluation without continuous-score coercion."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REFERENCE_REVISION = "aeebcd9ecd3e7266d6d6a035a8081b3da83dfe33"


def run_reference_evaluation(predictions_path: str | Path, annotations_path: str | Path, *, esa_root: str | Path, python_executable: str | Path, output_path: str | Path | None = None) -> dict:
    """Evaluate in a separately supplied Python environment and ESA checkout.

    Requires numpy, pandas, scikit-learn and portion in that environment (plus
    pyarrow for Parquet annotations). No model or ingestion dependencies enter it.
    """
    root = Path(esa_root).resolve()
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    if revision != REFERENCE_REVISION:
        raise ValueError(f"ESA reference evaluation requires checkout {REFERENCE_REVISION}. Got {revision}")
    dirty = subprocess.run(["git", "-C", str(root), "diff", "--ignore-cr-at-eol", "HEAD", "--", "timeeval/metrics"], check=True, text=True, capture_output=True).stdout
    if dirty.strip():
        raise ValueError("Pinned ESA metric files have local modifications")
    completed = subprocess.run([str(python_executable), str(Path(__file__).resolve()), "--worker", str(root), str(Path(predictions_path).resolve()), str(Path(annotations_path).resolve())], check=False, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"ESA reference worker failed: {completed.stderr[-4000:]}")
    result = json.loads(completed.stdout)
    result["reference_revision"] = revision
    result["prediction_source"] = str(Path(predictions_path).resolve())
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _empty_result(reason: str, has_predictions: bool, has_truth: bool) -> dict:
    # Keep defined event scores and record undefined denominators as null.
    precision = 0.0 if has_predictions and has_truth else None
    recall = 0.0 if has_truth else None
    return {"status": "undefined_metrics", "reason": reason, "metrics": {"EW_precision": precision, "EW_recall": recall, "EW_F_0.50": 0.0 if has_truth else None, "alarming_precision": None, "AFF_precision": None, "AFF_recall": 0.0 if has_truth else None, "AFF_F_0.50": None}}


def _read_transitions(path: Path):
    """Validate every row while retaining only equivalent binary change points.

    ESA holds each binary prediction until its next timestamp. Keeping the
    transitions and endpoint timestamps preserves those event intervals, with
    memory use determined by event count.
    """
    import numpy as np
    import pandas as pd
    transitions = []
    count = alerts = 0
    previous_time = previous_alarm = last = entity = None
    for frame in pd.read_csv(path, chunksize=100000):
        if not {"timestamp", "is_anomaly"}.issubset(frame):
            raise ValueError("Predictions require timestamp and is_anomaly columns")
        if frame.empty:
            continue
        times = pd.to_datetime(frame.timestamp, utc=True, errors="raise", format="mixed").dt.tz_localize(None)
        if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing or (previous_time is not None and times.iloc[0] <= previous_time):
            raise ValueError("Prediction timestamps must be valid, unique and sorted")
        if not frame.is_anomaly.isin([0, 1]).all():
            raise ValueError("Reference evaluation requires explicit binary predictions")
        if "entity_id" in frame:
            entities = frame.entity_id.fillna("").unique()
            if len(entities) != 1 or (entity is not None and entity != entities[0]):
                raise ValueError("ESA global evaluation requires a single entity")
            entity = entities[0]
        values = frame.is_anomaly.to_numpy(dtype=np.uint8)
        changes = np.r_[previous_alarm is None or values[0] != previous_alarm, values[1:] != values[:-1]]
        transitions.extend(zip(times[changes], values[changes]))
        previous_time, previous_alarm = times.iloc[-1], values[-1]
        last = (previous_time, previous_alarm)
        count += len(frame)
        alerts += int(values.sum())
    if last is not None and transitions[-1][0] != last[0]:
        transitions.append(last)
    return pd.DataFrame(transitions, columns=["timestamp", "is_anomaly"]), count, alerts


def _worker(root: Path, predictions_path: Path, annotations_path: Path) -> dict:
    import contextlib
    import importlib
    import math
    import types

    import numpy as np
    import pandas as pd

    predictions, row_count, alert_count = _read_transitions(predictions_path)
    if predictions.empty:
        return _empty_result("no prediction rows", False, False)
    times = predictions.timestamp
    labels = pd.read_parquet(annotations_path) if annotations_path.suffix.lower() == ".parquet" else pd.read_csv(annotations_path)
    if not {"ID", "StartTime", "EndTime", "Category"}.issubset(labels):
        raise ValueError("ESA interval annotations require ID, StartTime, EndTime and Category")
    for column in ["StartTime", "EndTime"]:
        labels[column] = pd.to_datetime(labels[column], utc=True, errors="raise", format="mixed").dt.tz_localize(None)
    if labels[["StartTime", "EndTime"]].isna().any().any() or (labels.EndTime < labels.StartTime).any():
        raise ValueError("Invalid ESA annotation intervals")
    full_range = (times.iloc[0], times.iloc[-1])
    labels = labels[(labels.EndTime >= full_range[0]) & (labels.StartTime <= full_range[1])].copy()
    labels["StartTime"] = labels.StartTime.clip(lower=full_range[0])
    labels["EndTime"] = labels.EndTime.clip(upper=full_range[1])
    labels = labels.drop(columns=["Channel"], errors="ignore").drop_duplicates()
    has_truth = bool((labels.Category == "Anomaly").any())
    has_alerts = bool(predictions.is_anomaly.any())
    if not has_alerts or not has_truth:
        result = _empty_result("no alerts" if not has_alerts else "no selected ground-truth events", has_alerts, has_truth)
    elif full_range[0] == full_range[1]:
        result = _empty_result("zero-duration evaluation range", has_alerts, has_truth)
    else:
        # Load metric modules from the verified checkout so relative imports
        # work without TimeEval's runner, Docker or dataset environment.
        for name, directory in [("timeeval", root / "timeeval"), ("timeeval.metrics", root / "timeeval" / "metrics")]:
            module = types.ModuleType(name)
            module.__path__ = [str(directory)]
            sys.modules[name] = module
        metric = importlib.import_module("timeeval.metrics.ESA_ADB_metrics").ESAScores
        pairs = np.empty((len(times), 2), dtype=object)
        pairs[:, 0] = times.to_numpy()
        pairs[:, 0] = [pd.Timestamp(item) for item in pairs[:, 0]]
        pairs[:, 1] = predictions.is_anomaly.to_numpy(dtype=np.uint8)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                measured = metric(betas=0.5, select_labels={"Category": "Anomaly"}, full_range=full_range).score(labels, pairs)
            metrics = {key: float(value) if math.isfinite(float(value)) else None for key, value in measured.items()}
            result = {"status": "complete" if all(value is not None for value in metrics.values()) else "undefined_metrics", "metrics": metrics}
        except ZeroDivisionError:
            result = {"status": "undefined_metrics", "reason": "pinned upstream metric has a zero-duration denominator", "metrics": {}}
    result.update({"metric": "ESA-ADB ESAScores", "scope": "global", "selected_category": "Anomaly", "rows": row_count, "metric_change_points": len(predictions), "alert_rows": alert_count, "ground_truth_events": int(labels.loc[labels.Category == "Anomaly", "ID"].nunique()), "full_range": [str(value) for value in full_range], "interval_policy": "clip to exported evaluation range"})
    return result


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("Use run_reference_evaluation from COMET")
    print(json.dumps(_worker(*(Path(value) for value in sys.argv[2:])), allow_nan=False))
