import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from src.db import get_connection, get_model
from src.evaluation.export import export_predictions
from src.ingestion.store import ingest_source
from src.modeling.pipeline import detect, diagnostics, get_run, prepare, train
from src.modeling.scoring import calibrate
from src.modeling.window_dataset import WindowDataset


@pytest.fixture
def trained(tmp_path):
    torch.set_num_threads(1)
    db = tmp_path / "catalog.db"
    source = tmp_path / "train.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "signal", "is_anomaly_signal"])
        for index in range(120):
            writer.writerow([f"2024-01-01T00:{index // 60:02d}:{index % 60:02d}Z", np.sin(index / 10), int(index > 100)])
    dataset = ingest_source(source, db_path=db, storage_dir=tmp_path, batch_size=17)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"window_size": 4, "hidden_size": 4, "epochs": 1, "batch_size": 7,
                                  "numeric_transform": "asinh", "window_centering": True}), encoding="utf-8")
    model = train(dataset["dataset_id"], config, db_path=db, storage_dir=tmp_path)
    return tmp_path, db, dataset, model


def test_pipeline_preserves_labels_outside_features_and_full_run_coverage(trained):
    root, db, dataset, model = trained
    model_row = get_model(model["model_id"], db_path=db)
    metadata = json.loads(model_row["metadata_json"])
    state = json.loads(Path(metadata["transform_path"]).read_text())
    assert state["numeric_transform"] == "asinh"
    assert [feature["source_name"] for feature in metadata["feature_contract"]] == ["signal"]
    first = detect(dataset["dataset_id"], model["model_id"], db_path=db, storage_dir=root, partition="test", batch_size=3)
    second = detect(dataset["dataset_id"], model["model_id"], db_path=db, storage_dir=root, partition="test", batch_size=11)
    assert first["run_id"] != second["run_id"]
    a = pq.read_table(first["scores_path"]).to_pydict()
    b = pq.read_table(second["scores_path"]).to_pydict()
    assert len(a["row_id"]) == 120
    np.testing.assert_allclose([np.nan if score is None else score for score in a["score"]],
                               [np.nan if score is None else score for score in b["score"]], rtol=1e-5, atol=1e-7, equal_nan=True)
    assert all(not alert for alert, coverage in zip(a["is_anomaly"], a["coverage"]) if coverage != "scored")
    assert first["coverage"]["scored"] == 15
    result = diagnostics(first["run_id"], db_path=db)
    assert result["score_count"] == 15
    with get_connection(db) as connection:
        for anomaly in connection.execute("SELECT * FROM anomalies WHERE run_id=?", (first["run_id"],)):
            assert anomaly["dataset_id"] == dataset["dataset_id"]
            assert anomaly["start_timestamp"] == anomaly["end_timestamp"] == anomaly["score_timestamp"]
            packet = json.loads(anomaly["packet_json"])
            assert "is_anomaly_signal" not in json.dumps(packet)
            assert packet["input_window_start_timestamp"] < packet["score_timestamp"]
    assert get_run(first["run_id"], db_path=db)["status"] == "completed"


def test_score_rotation_preserves_predictions_and_diagnostics(trained, monkeypatch):
    from src.ingestion import store

    root, db, dataset, model = trained
    baseline = detect(dataset["dataset_id"], model["model_id"], db_path=db, storage_dir=root,
                      partition="test", batch_size=3)
    monkeypatch.setattr(store, "PARQUET_PART_ROWS", 11)
    rotated = detect(dataset["dataset_id"], model["model_id"], db_path=db, storage_dir=root,
                     partition="test", batch_size=3)
    parts = sorted(Path(rotated["scores_path"]).glob("*.parquet"))
    assert len(parts) == 11
    for part in parts:
        with pq.ParquetFile(part) as reader:
            assert reader.metadata.num_row_groups == 1
            assert reader.metadata.num_rows <= 11
    assert pq.read_table(baseline["scores_path"]).equals(pq.read_table(rotated["scores_path"]))
    first = diagnostics(baseline["run_id"], db_path=db)
    second = diagnostics(rotated["run_id"], db_path=db)
    for field in ("score_count", "score_mean", "score_min", "score_max", "coverage", "anomalies_detected"):
        if isinstance(first[field], float):
            assert first[field] == pytest.approx(second[field])
        else:
            assert first[field] == second[field]
    a, b = root / "baseline.csv", root / "rotated.csv"
    export_predictions(baseline["run_id"], a, db_path=db)
    export_predictions(rotated["run_id"], b, db_path=db)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("removed_part", [1, -1])
def test_missing_score_parts_fail_before_reports_publish(trained, monkeypatch, removed_part):
    from src.ingestion import store

    root, db, dataset, model = trained
    monkeypatch.setattr(store, "PARQUET_PART_ROWS", 11)
    run = detect(dataset["dataset_id"], model["model_id"], db_path=db, storage_dir=root, partition="test")
    sorted(Path(run["scores_path"]).glob("*.parquet"))[removed_part].unlink()
    destination = root / "existing.csv"
    destination.write_text("keep existing export", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete score artifact"):
        export_predictions(run["run_id"], destination, db_path=db)
    assert destination.read_text(encoding="utf-8") == "keep existing export"
    assert not destination.with_suffix(".csv.partial").exists()
    output_dir = root / "diagnostics"
    with pytest.raises(ValueError, match="Incomplete score artifact"):
        diagnostics(run["run_id"], db_path=db, output_dir=output_dir)
    assert not output_dir.exists()


def test_cross_dataset_inference_and_early_contract_failure(trained):
    root, db, dataset, model = trained
    source = root / "target.jsonl"
    with source.open("w", encoding="utf-8") as stream:
        for index in range(10):
            stream.write(json.dumps({"extra": 7, "signal": float(index), "timestamp": f"2025-01-01T00:00:{index:02d}Z"}) + "\n")
    target = ingest_source(source, db_path=db, storage_dir=root)
    result = detect(target["dataset_id"], model["model_id"], db_path=db, storage_dir=root)
    assert result["dataset_id"] != dataset["dataset_id"]
    assert result["windows_scored"] == 7
    assert result["target_import_fingerprint"] == target["import_fingerprint"]
    assert result["target_source_sha256"] == target["source_sha256"]
    persisted = get_run(result["run_id"], db_path=db)
    assert persisted["target_import_fingerprint"] == target["import_fingerprint"]
    assert persisted["target_source_sha256"] == target["source_sha256"]
    source = root / "missing.csv"
    source.write_text("timestamp,other\n2025-01-01T00:00:00Z,1\n", encoding="utf-8")
    missing = ingest_source(source, db_path=db, storage_dir=root)
    with get_connection(db) as connection:
        before = connection.execute("SELECT count(*) FROM detection_runs").fetchone()[0]
    with pytest.raises(ValueError, match="missing"):
        detect(missing["dataset_id"], model["model_id"], db_path=db, storage_dir=root)
    with get_connection(db) as connection:
        assert connection.execute("SELECT count(*) FROM detection_runs").fetchone()[0] == before


def test_too_short_target_has_explicit_no_alert_warmup(trained):
    root, db, _, model = trained
    source = root / "short.csv"
    source.write_text("timestamp,signal\n2025-01-01T00:00:00Z,1\n", encoding="utf-8")
    target = ingest_source(source, db_path=db, storage_dir=root)
    run = detect(target["dataset_id"], model["model_id"], db_path=db, storage_dir=root)
    assert run["windows_scored"] == 0
    assert run["anomalies_detected"] == 0
    assert pq.read_table(run["scores_path"]).column("score").null_count == 1
    assert diagnostics(run["run_id"], db_path=db)["score_mean"] is None


def test_independent_target_preserves_explicit_partition_boundaries(trained):
    root, db, _, model = trained
    source = root / "partitioned-target.json"
    source.write_text(json.dumps([
        {"timestamp": f"2025-01-01T00:00:{index:02d}Z", "signal": float(index)}
        for index in range(15)
    ]), encoding="utf-8")
    target = ingest_source(source, {"splits": {"train_end": "2025-01-01T00:00:04Z",
                                               "validation_end": "2025-01-01T00:00:09Z"}},
                           db_path=db, storage_dir=root)
    run = detect(target["dataset_id"], model["model_id"], db_path=db, storage_dir=root)
    scores = pq.read_table(run["scores_path"]).to_pydict()
    assert scores["partition"] == ["train"] * 5 + ["validation"] * 5 + ["test"] * 5
    assert run["windows_scored"] == 6
    assert [index for index, score in enumerate(scores["score"]) if score is not None] == [3, 4, 8, 9, 13, 14]


def test_reviewed_manifest_file_preserves_fraction_partitions(trained):
    root, db, _, model = trained
    source = root / "mapped-target.json"
    source.write_text(json.dumps([
        {"timestamp": f"2025-01-01T00:00:{index:02d}Z", "signal": float(index)}
        for index in range(20)
    ]), encoding="utf-8")
    mapping = root / "reviewed-mapping.json"
    mapping.write_text(json.dumps({"splits": {"fractions": [.5, .25, .25]},
                                    "_splits_explicit": False}), encoding="utf-8")
    target = ingest_source(source, mapping, db_path=db, storage_dir=root)
    run = detect(target["dataset_id"], model["model_id"], db_path=db, storage_dir=root)
    scores = pq.read_table(run["scores_path"]).to_pydict()
    assert scores["partition"] == ["train"] * 10 + ["validation"] * 5 + ["test"] * 5
    assert run["windows_scored"] == 11


def test_calibration_exact_percentile_matches_numpy_without_materialized_windows(tmp_path):
    class Zero(torch.nn.Module):
        def forward(self, value):
            return torch.zeros_like(value)
    values = np.lib.format.open_memmap(tmp_path / "values.npy", mode="w+", dtype="float32", shape=(100, 2))
    values[:] = np.arange(200).reshape(100, 2)
    windows = WindowDataset(values, np.arange(95), 6)
    threshold, summary = calibrate(Zero(), windows, tmp_path / "scores.npy", percentile=93.5, batch_size=7)
    expected = [float(np.square(values[i:i + 6]).mean()) for i in range(95)]
    assert threshold == pytest.approx(np.percentile(expected, 93.5))
    assert summary["count"] == len(windows)


def test_failed_run_rolls_back_its_anomalies(trained, monkeypatch):
    from src.modeling import scoring
    root, db, dataset, model = trained
    original = scoring.score_batches

    def interrupted(*args, **kwargs):
        for result in original(*args, **kwargs):
            yield result
            raise RuntimeError("simulated interrupted scoring")

    monkeypatch.setattr(scoring, "score_batches", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        detect(dataset["dataset_id"], model["model_id"], db_path=db, storage_dir=root, batch_size=2)
    with get_connection(db) as connection:
        failed = connection.execute("SELECT run_id,status FROM detection_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        assert failed["status"] == "failed"
        assert connection.execute("SELECT count(*) FROM anomalies WHERE run_id=?", (failed["run_id"],)).fetchone()[0] == 0
    with pytest.raises(ValueError, match="not completed"):
        get_run(failed["run_id"], db_path=db)


def test_future_target_values_do_not_change_earlier_scores(trained):
    root, db, _, model = trained
    runs = []
    for version in (0, 1):
        source = root / f"future_{version}.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "signal", "is_anomaly_signal"])
            for index in range(30):
                value = float(index) if index < 20 else float(index + version * 1000)
                writer.writerow([f"2025-01-01T00:00:{index:02d}Z", value, version])
        target = ingest_source(source, db_path=db, storage_dir=root)
        runs.append(detect(target["dataset_id"], model["model_id"], db_path=db, storage_dir=root))
    a, b = [pq.read_table(run["scores_path"]).column("score").to_pylist() for run in runs]
    assert a[:20] == b[:20]
    assert a[20:] != b[20:]


def test_opt_in_point_label_exclusion_does_not_stitch_training_windows(tmp_path):
    from src.modeling.pipeline import _exclude_training_annotations
    from src.modeling.transforms import build_row_index
    from src.modeling.window_dataset import build_window_index
    import pyarrow as pa
    rows_data = [{"row_id": index + 1, "timestamp_ns": index, "entity_id": ""} for index in range(100)]
    rows, _ = build_row_index(pa.Table.from_pylist(rows_data).to_batches(), 100, tmp_path / "rows.npy", {})
    annotations = tmp_path / "annotations.parquet"
    pq.write_table(pa.table({"row_id": [10, 11, 90], "label_value": ["1", "2", "1"]}), annotations)
    dataset = {"annotations_path": str(annotations), "manifest": {"preprocessing": {"exclude_training_anomalies": True}}}
    assert _exclude_training_annotations(dataset, rows) == 2
    assert not rows[89]["excluded"]
    starts = build_window_index(rows, tmp_path / "starts.npy", window_size=4, partition="train")
    assert all(not rows[start:start + 4]["excluded"].any() for start in starts)


def test_processed_esa_point_exclusion_maps_timestamps_and_preserves_splits(tmp_path):
    from src.modeling.pipeline import _exclude_training_annotations
    from src.modeling.transforms import build_row_index
    import pyarrow as pa
    rows_data = [{"row_id": index + 1, "timestamp_ns": index * 1_000_000_000, "entity_id": ""} for index in range(100)]
    splits = {"test_start": "1970-01-01T00:01:29Z", "train_fraction": .85}
    rows, _ = build_row_index(pa.Table.from_pylist(rows_data).to_batches(), 100, tmp_path / "rows.npy",
                              {"splits": splits})
    partitions = rows["partition"].copy()
    annotations = tmp_path / "processed_points.parquet"
    pq.write_table(pa.table({
        "timestamp": ["1970-01-01 00:00:09", "1970-01-01 00:00:10", "1970-01-01 00:00:20",
                      "1970-01-01 00:00:30", "1970-01-01 00:01:15", "1970-01-01 00:01:16",
                      "1970-01-01 00:01:29", "1970-01-01 00:01:30"],
        "channel": ["signal", "signal", "unselected", "signal", "signal", "signal", "signal", "signal"],
        "label_value": [1, 0, 2, 1, 3, 1, 2, 1],
        "source_partition": ["train", "train", "train", "test", "train", "train", "train", "test"],
    }), annotations)
    dataset = {"point_annotations_path": str(annotations),
               "columns": [{"source_name": "signal", "name": "signal", "role": "numeric"}],
               "manifest": {"splits": splits, "preprocessing": {"exclude_training_anomalies": True}}}

    assert _exclude_training_annotations(dataset, rows) == 2
    assert np.flatnonzero(rows["excluded"]).tolist() == [9, 75]
    assert np.array_equal(rows["partition"], partitions)


def test_esa_interval_exclusion_preserves_categories_and_test_rows(tmp_path):
    from src.modeling.pipeline import _exclude_training_annotations
    from src.modeling.transforms import build_row_index
    import pyarrow as pa
    rows_data = [{"row_id": index + 1, "timestamp_ns": index * 1_000_000_000, "entity_id": ""} for index in range(100)]
    rows, _ = build_row_index(pa.Table.from_pylist(rows_data).to_batches(), 100, tmp_path / "rows.npy", {})
    annotations = tmp_path / "annotations.parquet"
    pq.write_table(pa.table({"Channel": ["signal", "signal", "unselected"],
                            "Category": ["Anomaly", "Rare Event", "Anomaly"],
                            "StartTime": ["1970-01-01T00:00:09Z", "1970-01-01T00:00:15Z", "1970-01-01T00:00:20Z"],
                            "EndTime": ["1970-01-01T00:01:29Z", "1970-01-01T00:00:15Z", "1970-01-01T00:00:20Z"]}), annotations)
    dataset = {"annotations_path": str(annotations), "columns": [{"source_name": "signal", "name": "signal", "role": "numeric"}],
               "manifest": {"preprocessing": {"exclude_training_anomalies": True}}}
    assert _exclude_training_annotations(dataset, rows) == 61  # inclusive 9..69, only train
    assert not rows[89]["excluded"]
