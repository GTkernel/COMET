import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from src.db import get_model
from src.ingestion.store import ingest_source
from src.modeling.pipeline import detect, diagnostics, train


def _run(tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / "data.csv"
    source.write_text("timestamp,signal\n" + "".join(
        f"2024-01-01T00:00:{index:02d}Z,{index % 5}\n" for index in range(40)), encoding="utf-8")
    db = tmp_path / "catalog.db"
    dataset = ingest_source(source, db_path=db, storage_dir=tmp_path)
    config = tmp_path / "model.yaml"
    config.write_text("window_size: 2\nhidden_size: 3\nepochs: 1\nbatch_size: 4\n", encoding="utf-8")
    model = train(dataset["dataset_id"], config, db_path=db, storage_dir=tmp_path)
    run = detect(dataset["dataset_id"], model["model_id"], db_path=db, storage_dir=tmp_path, partition="test")
    return db, model, run


def test_completed_run_restores_exact_statistics_rankings_and_five_plots(tmp_path, monkeypatch):
    from src.modeling import run_diagnostics
    from matplotlib.figure import Figure

    db, model, run = _run(tmp_path)
    monkeypatch.setattr(run_diagnostics, "PLOT_SAMPLE_LIMIT", 2)
    captions = []
    savefig = Figure.savefig
    def capture_caption(figure, *args, **kwargs):
        captions.append(figure._suptitle.get_text() if figure._suptitle else "")
        return savefig(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture_caption)
    before = Path(run["artifact_dir"], "run.json").read_bytes()
    result = diagnostics(run["run_id"], db_path=db, output_dir=tmp_path / "diagnostics")
    scores = np.asarray(pq.read_table(run["scores_path"], columns=["score"]).column(0).drop_null())
    stats = result["reconstruction_error_statistics"]
    assert stats["median"] == pytest.approx(np.median(scores))
    assert stats["standard_deviation"] == pytest.approx(np.std(scores))
    assert result["windows_evaluated"] == len(scores)
    assert result["diagnostic_replay_verified"]
    assert result["plot_sampling"]["approximate_plots"]
    assert result["plot_sampling"]["sampled_windows"] <= 2
    assert set(result["plots"]) == {"error_histogram", "error_timeseries", "top_features", "feature_heatmap", "error_quantiles"}
    assert all(Path(path).is_file() for path in result["plots"].values())
    assert len(captions) == 5
    assert all("Plot sample: 2 / 5 scored windows, stride 3" in caption for caption in captions)
    assert all("run_" + run["run_id"] in path for path in result["plots"].values())
    assert Path(run["artifact_dir"], "run.json").read_bytes() == before
    assert all("signal" in item["feature"] for item in result["top_contributing_features"])
    model_row = get_model(model["model_id"], db_path=db)
    checkpoint = torch.load(model_row["model_path"], weights_only=True, map_location="cpu")
    from src.modeling.lstm_autoencoder import LSTMAutoencoder
    replay = LSTMAutoencoder(2, 3, 1)
    replay.load_state_dict(checkpoint["model_state_dict"])
    replay.eval()
    values = np.load(Path(run["artifact_dir"]) / "values.npy")
    starts = np.load(Path(run["artifact_dir"]) / "windows.npy")[:len(scores)]
    windows = torch.from_numpy(np.stack([values[start:start + 2] for start in starts]))
    with torch.no_grad():
        feature_errors = (replay(windows) - windows).square().mean(dim=1).numpy()
    expected = feature_errors[scores > run["threshold"]].mean(axis=0)
    if len(result["top_contributing_features"]):
        actual = {item["feature"]: item["average_reconstruction_error"] for item in result["top_contributing_features"]}
        assert actual["signal [f0:value]"] == pytest.approx(expected[0])
        assert actual["signal [f0:missing]"] == pytest.approx(expected[1])


def test_diagnostics_reject_modified_frozen_inputs_and_checkpoint(tmp_path):
    db, model, run = _run(tmp_path)
    values = np.load(Path(run["artifact_dir"]) / "values.npy", mmap_mode="r+")
    values[-2:, 0] += 100
    values.flush()
    del values
    with pytest.raises(ValueError, match="changed since detection"):
        diagnostics(run["run_id"], db_path=db, output_dir=tmp_path / "bad-report")
    assert not (tmp_path / "bad-report").exists()
    metadata = json.loads(get_model(model["model_id"], db_path=db)["metadata_json"])
    checkpoint = Path(metadata["artifact_dir"]) / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        diagnostics(run["run_id"], db_path=db)


def test_empty_completed_run_keeps_null_statistics_and_five_plots(tmp_path):
    db, model, _ = _run(tmp_path)
    tiny = tmp_path / "tiny.csv"
    tiny.write_text("timestamp,signal\n2025-01-01T00:00:00Z,1\n", encoding="utf-8")
    target = ingest_source(tiny, db_path=db, storage_dir=tmp_path)
    run = detect(target["dataset_id"], model["model_id"], db_path=db, storage_dir=tmp_path)
    result = diagnostics(run["run_id"], db_path=db, output_dir=tmp_path / "empty-plots")
    assert result["windows_evaluated"] == 0
    assert all(value is None for value in result["reconstruction_error_statistics"].values())
    assert result["top_contributing_features"] == []
    assert len(result["plots"]) == 5


def test_legacy_diagnostics_redact_catalog_annotation_roles(tmp_path, monkeypatch):
    from src.common import paths
    from src.db import get_connection, init_db, insert_model_metadata
    from src.anomaly.detect import detect_anomalies
    from src.modeling.lstm_autoencoder import LSTMAutoencoder

    monkeypatch.setattr(paths, "STORAGE_ROOT", tmp_path)
    db = tmp_path / "legacy.db"
    init_db(db)
    with get_connection(db) as connection:
        connection.execute("INSERT INTO datasets(dataset_id,name,status,row_count,metadata_json,created_at,updated_at) VALUES(1,'legacy','prepared',6,'{}','2020','2020')")
        connection.execute("CREATE TABLE model_ready_dataset_1(row_id INTEGER,dataset_id INTEGER,timestamp TEXT,signal REAL,outcome__bad REAL,ground_truth REAL)")
        connection.executemany("INSERT INTO model_ready_dataset_1 VALUES(?,?,?,?,?,?)", [
            (index + 1, 1, f"2024-01-01T00:00:{index:02d}Z", index, 1, 1) for index in range(6)])
        connection.execute("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,role,created_at,updated_at) VALUES(1,'outcome','outcome','annotation','2020','2020')")
    features = ["signal", "outcome__bad", "ground_truth"]
    checkpoint = tmp_path / "old.pt"
    config = {"window_size": 2, "hidden_size": 4, "num_layers": 1}
    torch.save({"model_state_dict": LSTMAutoencoder(3, 4, 1).state_dict(), "feature_columns": features,
                "config": config}, checkpoint)
    model_id = insert_model_metadata(1, checkpoint, config_path=None, threshold=-1, threshold_method="legacy",
                                     threshold_percentile=95, training_summary={}, metadata={}, db_path=db)
    run = detect_anomalies(1, model_id, db_path=db)
    result = diagnostics(run["run_id"], db_path=db, output_dir=tmp_path / "legacy-plots")
    assert result["redacted_annotation_feature_count"] == 2
    assert [item["feature"] for item in result["top_contributing_features"]] == ["signal"]
    assert result["diagnostic_replay_verified"]
