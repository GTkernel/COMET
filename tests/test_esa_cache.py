import hashlib
import json

import pandas as pd

from src.ingestion.esa import _preparation_work, prepare_esa_source, prepare_processed_esa_source


def test_raw_cache_survives_batch_size_change(tmp_path, monkeypatch):
    root = tmp_path / "ESA-Mission1"
    (root / "channels").mkdir(parents=True)
    pd.Series([1.0, 2.0], index=pd.date_range("2000", periods=2, freq="30s")).to_frame("channel_1").to_pickle(root / "channels" / "channel_1.zip")
    pd.DataFrame(columns=["ID", "Channel", "StartTime", "EndTime"]).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(columns=["ID", "Category"]).to_csv(root / "anomaly_types.csv", index=False)
    manifest = {"esa": {"trusted_pickle": True, "version": "2"}}
    output, first = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=1)
    import src.ingestion.esa as adapter
    monkeypatch.setattr(adapter.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache decoded again")))
    second_output, second = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=17)
    assert output == second_output
    assert first == second


def test_processed_cache_survives_batch_size_change(tmp_path, monkeypatch):
    root = tmp_path / "ESA-Mission2-semi-supervised"
    root.mkdir()
    for name, timestamp in (("21_months.train.csv", "2001-09-30 23:59:42"), ("21_months.test.csv", "2001-10-01 00:00:18")):
        pd.DataFrame({"timestamp": [timestamp], "channel_1": [1.0], "is_anomaly_channel_1": [0]}).to_csv(root / name, index=False)
    manifest = {"esa": {"version": "2"}}
    output, first = prepare_processed_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=1)
    import src.ingestion.esa as adapter
    monkeypatch.setattr(adapter, "_write_processed_pair", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache rewritten")))
    second_output, second = prepare_processed_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=17)
    assert output == second_output
    assert first == second


def test_legacy_incomplete_preparation_reused_without_moving_files(tmp_path):
    config = {"adapter_version": "1.0.3", "profile": "causal", "batch_size": 10_000}
    checksums, manifest = {"channel_1": "example"}, {"esa": {"version": "2"}}
    legacy = hashlib.sha256(json.dumps({"config": config, "checksums": checksums, "manifest": manifest}, sort_keys=True).encode()).hexdigest()
    original = tmp_path / "esa-prepared" / legacy
    original.mkdir(parents=True)
    checkpoint = original / "progress.json"
    checkpoint.write_text('{"completed_channels":["channel_1"]}', encoding="utf-8")
    first = _preparation_work(tmp_path, "esa-prepared", config, checksums, manifest)
    second = _preparation_work(tmp_path, "esa-prepared", {**config, "batch_size": 37}, checksums, manifest)
    assert first == second == original
    assert checkpoint.exists()
