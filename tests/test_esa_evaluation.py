import csv
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.export import export_predictions
from src.evaluation.reference import _worker


def test_export_warmup_and_partition(tmp_path):
    source = tmp_path / "scores.parquet"
    pd.DataFrame({"row_id": [0, 1, 2], "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03"], "timestamp_ns": [1, 2, 3], "entity_id": ["x"] * 3, "partition": ["train", "test", "test"], "score": [None, None, 1.25], "is_anomaly": [False, True, True], "coverage": ["outside_partition", "warmup_or_stride", "scored"]}).to_parquet(source)
    path = tmp_path / "prediction.csv"
    result = export_predictions({"scores_path": str(source), "partition": "test", "run_id": "run1"}, path)
    assert result["rows"] == 2
    assert result["alert_rows"] == 1
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["is_anomaly"] == "0" and rows[0]["score"] == ""
    assert rows[1]["score"] == "1.25"


@pytest.mark.parametrize("alarms,truth,reason", [([0, 0], True, "no alerts"), ([0, 1], False, "no selected ground-truth events")])
def test_undefined_metrics_are_explicit(tmp_path, alarms, truth, reason):
    predictions = tmp_path / "p.csv"
    labels = tmp_path / "l.csv"
    pd.DataFrame({"timestamp": ["2000-01-01", "2000-01-02"], "is_anomaly": alarms}).to_csv(predictions, index=False)
    pd.DataFrame([["a", "2000-01-01", "2000-01-02", "Anomaly"]] if truth else [], columns=["ID", "StartTime", "EndTime", "Category"]).to_csv(labels, index=False)
    result = _worker(Path("unused"), predictions, labels)
    assert result["status"] == "undefined_metrics" and result["reason"] == reason
    assert result["metrics"]["AFF_precision"] is None


def test_continuous_predictions_rejected(tmp_path):
    predictions = tmp_path / "p.csv"
    pd.DataFrame({"timestamp": ["2000-01-01"], "is_anomaly": [0.8]}).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="binary"):
        _worker(Path("unused"), predictions, tmp_path / "unused.csv")


def test_transition_reduction_preserves_boundaries(tmp_path):
    from src.evaluation.reference import _read_transitions
    path = tmp_path / "p.csv"
    pd.DataFrame({"timestamp": pd.date_range("2000", periods=8, freq="s"), "is_anomaly": [0, 0, 1, 1, 0, 0, 1, 1]}).to_csv(path, index=False)
    transitions, rows, alerts = _read_transitions(path)
    assert rows == 8 and alerts == 4
    assert transitions.timestamp.dt.second.tolist() == [0, 2, 4, 6, 7]


def test_pinned_reference_subprocess(tmp_path):
    import os
    import sys
    from src.evaluation.reference import REFERENCE_REVISION, run_reference_evaluation
    root = os.environ.get("COMET_ESA_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set COMET_ESA_REFERENCE_ROOT for pinned upstream metric validation")
    pytest.importorskip("portion")
    predictions = tmp_path / "p.csv"
    labels = tmp_path / "l.csv"
    pd.DataFrame({"timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"], "is_anomaly": [0, 1, 0, 0]}).to_csv(predictions, index=False)
    pd.DataFrame([["a", "2000-01-02", "2000-01-02 12:00:00", "Anomaly"]], columns=["ID", "StartTime", "EndTime", "Category"]).to_csv(labels, index=False)
    result = run_reference_evaluation(predictions, labels, esa_root=root, python_executable=sys.executable)
    assert result["reference_revision"] == REFERENCE_REVISION
    assert result["metrics"]["EW_recall"] == 1.0
    assert result["metrics"]["EW_precision"] > 0
