from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ingestion.esa import (
    prepare_esa_source, prepare_processed_esa_source,
    recognize_esa_source, recognize_processed_esa_source,
)
from src.ingestion.esa_decode import decode_channel


def _decoded_fixture(tmp_path, name, rows, *, numeric, partitions, reference=False):
    path = tmp_path / f"{name}.parquet"
    schema = pa.schema([("timestamp_ns", pa.int64()), ("value", pa.float64() if numeric else pa.string()),
                        ("partition", pa.int8()), *([("is_override", pa.bool_())] if reference else [])])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=2)
    return path, {"channel": name, "numeric": numeric, "telecommand": False, "partitions": partitions}


def test_alignment_carries_nulls_categories_and_gap_expiry_across_100k_rows(tmp_path):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    high = 100004 * step
    numeric = _decoded_fixture(tmp_path, "channel_1", [
        {"timestamp_ns": 0, "value": 1., "partition": 0},
        {"timestamp_ns": 99999 * step - 5_000_000_000, "value": 2., "partition": 0},
        {"timestamp_ns": 100000 * step + 5_000_000_000, "value": 3., "partition": 0},
        {"timestamp_ns": 100002 * step, "value": None, "partition": 0},
        {"timestamp_ns": 100003 * step, "value": 4., "partition": 0},
    ], numeric=True, partitions={"0": {"start_ns": 0, "end_ns": high, "first_value": 1.}})
    categorical = _decoded_fixture(tmp_path, "channel_2", [
        {"timestamp_ns": 5_000_000_000, "value": "idle", "partition": 0},
        {"timestamp_ns": 99999 * step + 5_000_000_000, "value": "active", "partition": 0},
        {"timestamp_ns": 100001 * step, "value": None, "partition": 0},
        {"timestamp_ns": 100003 * step, "value": "done", "partition": 0},
    ], numeric=False, partitions={"0": {"start_ns": 0, "end_ns": high, "first_value": "idle"}})
    output = tmp_path / "aligned.parquet"
    _align([numeric, categorical], output, {"profile": "causal", "cadence_seconds": 30, "max_gap_seconds": 30,
                                           "batch_size": 1})
    frame = pd.read_parquet(output)
    assert len(frame) == 100005
    assert frame.channel_1.iloc[99999] == 2.
    assert pd.isna(frame.channel_1.iloc[100000])
    assert frame.channel_1.iloc[100001] == 3.
    assert pd.isna(frame.channel_1.iloc[100002])
    assert frame.channel_1.iloc[100003:].tolist() == [4., 4.]
    assert pd.isna(frame.channel_2.iloc[0]) and frame.channel_2.iloc[1] == "idle"
    assert pd.isna(frame.channel_2.iloc[2])
    assert frame.channel_2.iloc[100000] == "active"
    assert frame.channel_2.iloc[100001:100003].isna().all()
    assert frame.channel_2.iloc[100003:].tolist() == ["done", "done"]
    assert frame.comet_gap.iloc[100002] == "1"
    import json
    progress = json.loads((tmp_path / "alignment-progress.json").read_text())
    assert progress["status"] == "complete" and progress["completed_rows"] == len(frame)


def test_alignment_reference_overrides_last_one_sample_across_100k_rows(tmp_path):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    rows = [{"timestamp_ns": time, "value": value, "partition": 0, "is_override": override}
            for time, value, override in [
                (0, 1., False), (99999 * step + 5_000_000_000, 3., False),
                (100000 * step, 7., True), (100001 * step, 4., False),
                (100002 * step, None, True), (100004 * step, None, False),
                (100005 * step, 5., False),
            ]]
    decoded = _decoded_fixture(tmp_path, "channel_1", rows, numeric=True, reference=True,
                               partitions={"0": {"start_ns": 0, "end_ns": 100005 * step, "first_value": 1.}, "1": None})
    output = tmp_path / "aligned.parquet"
    _align([decoded], output, {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1,
                              "test_boundary_ns": 100005 * step})
    frame = pd.read_parquet(output)
    assert frame.channel_1.iloc[99999:].tolist() == [1., 7., 4., 4., 4., 4., 5.]


@pytest.mark.parametrize("observations,expected", [
    ([(0, 1.), (30, 4.), (60, None), (90, 5.)], [1., 4., 4., 5.]),
    ([(0, None), (30, 4.), (60, None), (90, 5.)], [4., 4., 4., 5.]),
    # The unsampled value 7 is replaced by null before the next grid point;
    # upstream backfills from the first usable OUTPUT sample, which is 4.
    ([(0, None), (5, 7.), (10, None), (60, 4.)], [4., 4., 4.]),
])
def test_reference_fills_null_output_grid_like_upstream(tmp_path, observations, expected):
    from src.ingestion.esa import _align

    rows = [{"timestamp_ns": seconds * 1_000_000_000, "value": value, "partition": 0, "is_override": False}
            for seconds, value in observations]
    high = observations[-1][0] * 1_000_000_000
    decoded = _decoded_fixture(tmp_path, "channel_1", rows, numeric=True, reference=True,
                               partitions={"0": {"start_ns": 0, "end_ns": high, "first_value": observations[0][1]}, "1": None})
    output = tmp_path / "aligned.parquet"
    _align([decoded], output, {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1,
                              "test_boundary_ns": high})
    result = pd.read_parquet(output)
    assert result.channel_1.tolist() == expected
    assert not any(name.startswith("_comet_esa_") for name in result.columns)


def test_reference_null_output_fill_carries_across_grid_batches(tmp_path):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    decoded = _decoded_fixture(tmp_path, "channel_1", [
        {"timestamp_ns": 0, "value": None, "partition": 0, "is_override": False},
        {"timestamp_ns": 99998 * step, "value": 4., "partition": 0, "is_override": False},
        {"timestamp_ns": 99999 * step, "value": None, "partition": 0, "is_override": False},
        {"timestamp_ns": 100002 * step, "value": 5., "partition": 0, "is_override": False},
    ], numeric=True, reference=True, partitions={"0": {"start_ns": 0, "end_ns": 100002 * step, "first_value": None}, "1": None})
    output = tmp_path / "aligned.parquet"
    _align([decoded], output, {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1,
                              "test_boundary_ns": 100002 * step})
    result = pd.read_parquet(output).channel_1
    assert result.iloc[:100002].eq(4.).all()
    assert result.iloc[-1] == 5.


def test_reference_all_null_partition_does_not_borrow_training_fill(tmp_path):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    decoded = _decoded_fixture(tmp_path, "channel_1", [
        {"timestamp_ns": 0, "value": 1., "partition": 0, "is_override": False},
        {"timestamp_ns": step, "value": 4., "partition": 0, "is_override": False},
        {"timestamp_ns": 2 * step, "value": None, "partition": 1, "is_override": False},
        {"timestamp_ns": 4 * step, "value": None, "partition": 1, "is_override": False},
    ], numeric=True, reference=True, partitions={"0": {"start_ns": 0, "end_ns": step, "first_value": 1.},
                                                "1": {"start_ns": 2 * step, "end_ns": 4 * step, "first_value": None}})
    output = tmp_path / "aligned.parquet"
    _align([decoded], output, {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1,
                              "test_boundary_ns": step})
    result = pd.read_parquet(output).channel_1
    assert result.iloc[:2].tolist() == [1., 4.]
    assert result.iloc[2:].isna().all()


@pytest.mark.parametrize("start,end", [(60, None), (90, None), (None, 30), (30, 90)])
def test_reference_output_range_equals_slice_after_full_grid_fill(tmp_path, start, end):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    decoded = _decoded_fixture(tmp_path, "channel_1", [
        {"timestamp_ns": 0, "value": None, "partition": 0, "is_override": False},
        {"timestamp_ns": 2 * step, "value": 4., "partition": 0, "is_override": True},
        {"timestamp_ns": 3 * step, "value": None, "partition": 0, "is_override": False},
        {"timestamp_ns": 4 * step, "value": 5., "partition": 0, "is_override": False},
    ], numeric=True, reference=True, partitions={"0": {"start_ns": 0, "end_ns": 4 * step, "first_value": None}, "1": None})
    config = {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1, "test_boundary_ns": 4 * step}
    full_output = tmp_path / "full.parquet"
    _align([decoded], full_output, config)
    clipped_output = tmp_path / "clipped.parquet"
    limits = {f"{key}_ns": value * 1_000_000_000 for key, value in [("start", start), ("end", end)] if value is not None}
    _align([decoded], clipped_output, {**config, **limits})
    full = pd.read_parquet(full_output)
    times = pd.to_datetime(full.timestamp).dt.as_unit("ns").astype("int64")
    selected = np.ones(len(full), dtype=bool)
    if start is not None:
        selected &= times >= start * 1_000_000_000
    if end is not None:
        selected &= times <= end * 1_000_000_000
    pd.testing.assert_frame_equal(pd.read_parquet(clipped_output), full[selected].reset_index(drop=True))


def test_reference_output_fill_uses_override_until_usable_raw_grid_returns(tmp_path):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    decoded = _decoded_fixture(tmp_path, "channel_1", [
        {"timestamp_ns": 0, "value": 1., "partition": 0, "is_override": False},
        {"timestamp_ns": step, "value": 4., "partition": 0, "is_override": True},
        {"timestamp_ns": 2 * step, "value": None, "partition": 0, "is_override": False},
        {"timestamp_ns": 3 * step, "value": None, "partition": 0, "is_override": True},
        {"timestamp_ns": 4 * step, "value": 5., "partition": 0, "is_override": False},
    ], numeric=True, reference=True, partitions={"0": {"start_ns": 0, "end_ns": 4 * step, "first_value": 1.}, "1": None})
    output = tmp_path / "aligned.parquet"
    _align([decoded], output, {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1,
                              "test_boundary_ns": 4 * step})
    assert pd.read_parquet(output).channel_1.tolist() == [1., 4., 4., 4., 5.]


@pytest.mark.parametrize("end_slot", [1, 100000])
def test_reference_final_restoration_fills_wider_channel_grid(tmp_path, end_slot):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    end = end_slot * step
    short = _decoded_fixture(tmp_path, "channel_1", [
        {"timestamp_ns": 0, "value": 1., "partition": 0, "is_override": False},
        {"timestamp_ns": end - step + 5_000_000_000, "value": 7., "partition": 0, "is_override": False},
        {"timestamp_ns": end - step + 10_000_000_000, "value": 3., "partition": 0, "is_override": False},
        {"timestamp_ns": end, "value": 7., "partition": 0, "is_override": True},
    ], numeric=True, reference=True, partitions={"0": {"start_ns": 0, "end_ns": end, "first_value": 1.}, "1": None})
    long = _decoded_fixture(tmp_path, "channel_2", [
        {"timestamp_ns": 0, "value": 10., "partition": 0, "is_override": False},
        {"timestamp_ns": end + 2 * step, "value": 20., "partition": 0, "is_override": False},
    ], numeric=True, reference=True, partitions={"0": {"start_ns": 0, "end_ns": end + 2 * step, "first_value": 10.}, "1": None})
    output = tmp_path / "aligned.parquet"
    _align([short, long], output, {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1,
                                 "test_boundary_ns": end + 2 * step})
    frame = pd.read_parquet(output)
    assert frame.channel_1.iloc[-3:].tolist() == [7., 7., 7.]
    assert frame.channel_1.iloc[end_slot - 1] == 1.


def test_alignment_restricted_test_range_discards_training_cursor_state(tmp_path):
    from src.ingestion.esa import _align

    step = 30_000_000_000
    decoded = _decoded_fixture(tmp_path, "channel_1", [
        {"timestamp_ns": 0, "value": 1., "partition": 0, "is_override": False},
        {"timestamp_ns": 2 * step, "value": 2., "partition": 0, "is_override": False},
        {"timestamp_ns": 2 * step + 5_000_000_000, "value": 9., "partition": 1, "is_override": False},
        {"timestamp_ns": 4 * step, "value": None, "partition": 1, "is_override": False},
        {"timestamp_ns": 5 * step, "value": 10., "partition": 1, "is_override": False},
    ], numeric=True, reference=True, partitions={
        "0": {"start_ns": 0, "end_ns": 2 * step, "first_value": 1.},
        "1": {"start_ns": 2 * step, "end_ns": 5 * step, "first_value": 9.},
    })
    output = tmp_path / "aligned.parquet"
    _align([decoded], output, {"profile": "esa-reference", "cadence_seconds": 30, "batch_size": 1,
                              "test_boundary_ns": 2 * step, "start_ns": 3 * step})
    assert pd.read_parquet(output).channel_1.tolist() == [9., 9., 10.]


def mission(tmp_path, channels, labels=None, mission_id=1):
    root = tmp_path / f"ESA-Mission{mission_id}"
    (root / "channels").mkdir(parents=True)
    for channel, values in channels.items():
        values.to_frame(channel).to_pickle(root / "channels" / f"{channel}.zip")
    pd.DataFrame(labels or [], columns=["ID", "Channel", "StartTime", "EndTime"]).to_csv(root / "labels.csv", index=False)
    pd.DataFrame([["a", "Anomaly"], ["g", "Communication Gap"]], columns=["ID", "Category"]).to_csv(root / "anomaly_types.csv", index=False)
    return root


def test_raw_requires_explicit_pickle_trust(tmp_path):
    from src.ingestion.store import inspect_source

    root = mission(tmp_path, {"channel_1": pd.Series([1, 2], index=pd.date_range("2000", periods=2, freq="30s"))})
    assert recognize_esa_source(root)
    with pytest.raises(ValueError, match="trusted_pickle"):
        prepare_esa_source(root, {}, storage_dir=tmp_path / "store")
    inspected = inspect_source(root, {"esa": {"trusted_pickle": True}})
    assert any("esa.version" in issue for issue in inspected["issues"])
    with pytest.raises(ValueError, match="esa.version"):
        prepare_esa_source(root, {"esa": {"trusted_pickle": True}}, storage_dir=tmp_path / "store")


def test_causal_labels_and_future_do_not_change_prefix(tmp_path):
    times = pd.date_range("2000", periods=4, freq="30s")
    root = mission(tmp_path, {"channel_4": pd.Series([1., 3., 8., 100.], index=times)}, [["a", "channel_4", str(times[0]), str(times[1])]])
    manifest = {"esa": {"trusted_pickle": True, "profile": "causal", "version": "2"}}
    output, resolved = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=2)
    first = pd.read_parquet(output)
    pd.DataFrame(columns=["ID", "Channel", "StartTime", "EndTime"]).to_csv(root / "labels.csv", index=False)
    pd.Series([1., 3., 8., 9999.], index=times).to_frame("channel_4").to_pickle(root / "channels" / "channel_4.zip")
    output2, _ = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=3)
    second = pd.read_parquet(output2)
    pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])
    assert first.channel_4.iloc[1] == 2
    assert set(first.columns) == {"timestamp", "channel_4", "comet_gap"}
    assert resolved["annotation_format"] == "esa-intervals"
    assert "reference_grid_fill_version" not in resolved["esa"]["configuration"]
    assert pd.read_parquet(resolved["annotations_path"]).ID.tolist() == ["a"]


def test_reference_restores_lost_annotations_and_unknown_categories(tmp_path):
    times = pd.to_datetime(["2000-01-01 00:00:00", "2000-01-01 00:00:05", "2000-01-01 00:00:10", "2000-01-01 00:00:30"])
    root = mission(tmp_path, {"channel_1": pd.Series([1., 7., 3., 4.], index=times)}, [["a", "channel_1", str(times[1]), str(times[1])]])
    manifest = {"esa": {"trusted_pickle": True, "profile": "esa-reference", "version": "2"}}
    output, _ = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=1)
    result = pd.read_parquet(output)
    assert result.channel_1.tolist() == [1., 7.]
    pd.DataFrame([["g", "channel_1", str(times[1]), str(times[1])]], columns=["ID", "Channel", "StartTime", "EndTime"]).to_csv(root / "labels.csv", index=False)
    output2, _ = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store", batch_size=2)
    assert pd.read_parquet(output2).channel_1.tolist() == [1., 4.]


def test_reference_annotation_override_lasts_one_grid_sample(tmp_path):
    times = pd.Timestamp("2000-01-01") + pd.to_timedelta([0, 5, 10, 100], unit="s")
    root = mission(tmp_path, {"channel_1": pd.Series([1., 7., 3., 4.], index=times)}, [["a", "channel_1", str(times[1]), str(times[1])]])
    output, _ = prepare_esa_source(root, {"esa": {"trusted_pickle": True, "profile": "esa-reference", "version": "2"}}, storage_dir=tmp_path / "store", batch_size=1)
    assert pd.read_parquet(output).channel_1.tolist() == [1., 7., 3., 3., 4.]


def test_cache_preserves_current_split_and_units_manifest(tmp_path):
    root = mission(tmp_path, {"channel_1": pd.Series([1., 2.], index=pd.date_range("2000", periods=2, freq="30s"))})
    first = {"esa": {"trusted_pickle": True, "version": "2"}, "splits": {"fractions": [.7, .15, .15]}, "units": {"channel_1": "V"}}
    second = {"esa": {"trusted_pickle": True, "version": "2"}, "splits": {"fractions": [.6, .2, .2]}, "units": {"channel_1": "mV"}}
    out1, _ = prepare_esa_source(root, first, storage_dir=tmp_path / "store")
    out2, resolved = prepare_esa_source(root, second, storage_dir=tmp_path / "store")
    assert out1 != out2
    assert resolved["units"] == second["units"] and resolved["splits"] == second["splits"]


def test_mission2_categorical_is_not_factorized_causally(tmp_path):
    root = mission(tmp_path, {"channel_1": pd.Series(["idle", "active"], index=pd.date_range("2000", periods=2, freq="18s"))}, mission_id=2)
    output, manifest = prepare_esa_source(root, {"esa": {"trusted_pickle": True, "version": "2"}}, storage_dir=tmp_path / "store")
    assert manifest["cadence_seconds"] == 18
    assert manifest["categorical_features"] == ["channel_1"]
    assert pd.read_parquet(output).channel_1.tolist() == ["idle", "active"]


def test_completed_channel_cache_is_reused(tmp_path, monkeypatch):
    root = mission(tmp_path, {"channel_1": pd.Series([1., 2.], index=pd.date_range("2000", periods=2, freq="30s"))})
    manifest = {"esa": {"trusted_pickle": True, "version": "2"}}
    output, _ = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store")
    (output.parent / "complete.json").unlink()
    import src.ingestion.esa as adapter
    monkeypatch.setattr(adapter.subprocess, "run", lambda *args, **kwargs: pytest.fail("decoded completed channel again"))
    output2, _ = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store")
    assert output == output2


def test_preparation_lock_rejects_concurrent_cache_writers(tmp_path):
    from src.ingestion.store import _import_lock
    root = mission(tmp_path, {"channel_1": pd.Series([1., 2.], index=pd.date_range("2000", periods=2, freq="30s"))})
    manifest = {"esa": {"trusted_pickle": True, "version": "2"}}
    output, _ = prepare_esa_source(root, manifest, storage_dir=tmp_path / "store")
    with _import_lock(output.parent / "prepare.lock"):
        with pytest.raises(RuntimeError, match="already being imported"):
            prepare_esa_source(root, manifest, storage_dir=tmp_path / "store")


def test_decoder_rejects_conflicting_duplicates(tmp_path):
    source = tmp_path / "channel_1.zip"
    pd.DataFrame({"channel_1": [1., 2.]}, index=pd.to_datetime(["2000", "2000"])).to_pickle(source)
    config = {"channel": "channel_1", "profile": "causal", "cadence_seconds": 30, "test_boundary_ns": pd.Timestamp("2007").value, "batch_size": 10}
    with pytest.raises(ValueError, match="conflicting"):
        decode_channel(source, tmp_path / "out.parquet", config)


def test_causal_has_no_backfill_and_respects_gap(tmp_path):
    root = mission(tmp_path, {"channel_1": pd.Series([10., 20.], index=pd.to_datetime(["2000-01-01 00:00:05", "2000-01-01 00:02:00"]))})
    output, _ = prepare_esa_source(root, {"esa": {"trusted_pickle": True, "version": "2"}, "max_gap_seconds": 30}, storage_dir=tmp_path / "store")
    values = pd.read_parquet(output).channel_1.to_numpy()
    assert np.isnan(values[0]) and values[1] == 10 and np.isnan(values[2]) and values[-1] == 20
    frame = pd.read_parquet(output)
    assert frame.comet_gap.tolist() == ["1", "0", "1", "1", "0"]


def test_reference_combined_source_keeps_test_out_of_training(tmp_path):
    times = pd.to_datetime(["2006-12-31 23:59:30", "2007-01-01 00:00:00", "2007-01-01 00:00:05", "2007-01-01 00:00:30"])
    root = mission(tmp_path, {"channel_1": pd.Series([1., 2., 99., 100.], index=times)})
    output, manifest = prepare_esa_source(root, {"esa": {"trusted_pickle": True, "profile": "esa-reference", "version": "2"}}, storage_dir=tmp_path / "store")
    frame = pd.read_parquet(output)
    assert not frame.timestamp.duplicated().any()
    assert frame.channel_1.tolist() == [1., 2., 100.]
    assert manifest["esa"]["configuration"]["reference_grid_fill_version"] == 3
    assert any("boundary row" in item for item in manifest["esa"]["departures"])


def test_raw_archive_rejects_traversal(tmp_path):
    import zipfile
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("Mission1/channels/channel_1.zip", "placeholder")
        stream.writestr("Mission1/labels.csv", "placeholder")
        stream.writestr("../escape", "untrusted")
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_esa_source(archive, {"esa": {"trusted_pickle": True, "version": "2"}}, storage_dir=tmp_path / "store")


def test_reference_matches_zero_order_hold_fixture(tmp_path):
    times = pd.to_datetime(["2000-01-01 00:00:05", "2000-01-01 00:00:20", "2000-01-01 00:00:50", "2000-01-01 00:01:01"])
    raw = pd.Series([2., 6., 10., 30.], index=times)
    root = mission(tmp_path, {"channel_4": raw})
    output, _ = prepare_esa_source(root, {"esa": {"trusted_pickle": True, "profile": "esa-reference", "version": "2"}}, storage_dir=tmp_path / "store", batch_size=1)
    # The pinned Mission1 script forward differences before resampling and
    # initializes the first floored grid point from the first observation.
    expected = pd.Series(np.diff(raw, append=raw.iloc[-1]), index=times)
    grid = pd.date_range(times[0].floor("30s"), times[-1].ceil("30s"), freq="30s")
    resampled = expected.reindex(grid, method="ffill")
    resampled.iloc[0] = expected.iloc[0]
    assert pd.read_parquet(output).channel_4.tolist() == resampled.tolist()


def test_raw_to_catalog_annotation_separation(tmp_path):
    from src.ingestion.store import ingest_source, iter_batches
    root = mission(tmp_path, {"channel_1": pd.Series([1., 2., 3.], index=pd.date_range("2000", periods=3, freq="30s"))}, [["a", "channel_1", "2000-01-01", "2000-01-01 00:00:30"]])
    result = ingest_source(root, {"esa": {"trusted_pickle": True, "version": "2"}}, db_path=tmp_path / "catalog.db", storage_dir=tmp_path / "store", batch_size=2)
    assert result["row_count"] == 3
    assert result["manifest"]["esa"]["mission"] == 1
    assert result["manifest"]["preprocessing"]["forward_fill"] is False
    assert pd.read_parquet(result["annotations_path"]).ID.tolist() == ["a"]
    columns = next(iter_batches(result["dataset_id"], db_path=tmp_path / "catalog.db")).schema.names
    assert not {"ID", "Category", "Channel", "StartTime", "EndTime"}.intersection(columns)


def test_object_telecommands_become_finite_numeric_before_zero_transitions(tmp_path):
    source = tmp_path / "telecommand_1.zip"
    times = pd.to_datetime(["2000-01-01 00:00:00", "2000-01-01 00:00:36"])
    pd.Series(["1", "2.5"], index=times).to_frame("telecommand_1").to_pickle(source)
    config = {"channel": "telecommand_1", "telecommand": True, "profile": "causal", "cadence_seconds": 18,
              "test_boundary_ns": pd.Timestamp("2001-10-01").value, "batch_size": 2}
    output = tmp_path / "telecommand.parquet"
    stats = decode_channel(source, output, config)
    assert stats["numeric"] is True
    assert pd.read_parquet(output).value.tolist() == [1.0, 0.0, 2.5, 0.0]
    pd.Series(["1", "bad"], index=times).to_frame("telecommand_1").to_pickle(source)
    with pytest.raises(ValueError, match="telecommand observations must be numeric"):
        decode_channel(source, output, config)
    pd.Series(["1", "inf"], index=times).to_frame("telecommand_1").to_pickle(source)
    with pytest.raises(ValueError, match="telecommand observations must be finite"):
        decode_channel(source, output, config)


def _processed_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["timestamp", "channel_1", "telecommand_1", "is_anomaly_channel_1", "is_anomaly_telecommand_1"]).to_csv(path, index=False)


def test_processed_esa_pair_separates_labels_and_pins_provenance(tmp_path):
    folder = tmp_path / "data" / "preprocessed" / "multivariate" / "ESA-Mission2-semi-supervised"
    train = folder / "21_months.train.csv"
    test = folder / "21_months.test.csv"
    _processed_csv(train, [["2001-09-30 23:59:24", 1.0, 0, 0, 0], ["2001-10-01 00:00:00", 2.0, 1, 1, 0]])
    _processed_csv(test, [["2001-10-01 00:00:18", 3.0, 0, 0, 3], ["2001-10-01 00:00:36", 4.0, 1, 0, 0]])
    assert recognize_processed_esa_source(folder)
    assert recognize_processed_esa_source(train)
    output, manifest = prepare_processed_esa_source(train, {"esa": {"version": "2"}}, storage_dir=tmp_path / "store", batch_size=1)
    frame = pd.read_parquet(output)
    assert list(frame) == ["timestamp", "channel_1", "telecommand_1"]
    assert frame.channel_1.tolist() == [1.0, 2.0, 3.0, 4.0]
    points = pd.read_parquet(manifest["annotations_path"])
    assert set(points.channel) == {"channel_1", "telecommand_1"}
    assert sorted(points.label_value.tolist()) == [1, 3]
    assert manifest["annotation_format"] == "esa-points"
    assert manifest["splits"] == {"test_start": "2001-10-01T00:00:00Z", "train_fraction": .85}
    assert manifest["esa"]["profile"] == "esa-reference"
    assert manifest["esa"]["upstream_revision"]
    assert set(manifest["esa"]["source_checksums"]) == {str(train), str(test)}
    assert manifest["esa"]["processed_rows"] == 4
    assert manifest["_source_sorted_unique"] is True


def test_processed_esa_pair_import_is_label_free_and_repeatable(tmp_path):
    from src.ingestion.store import ingest_source, inspect_source, iter_batches

    folder = tmp_path / "ESA-Mission2-semi-supervised"
    train = folder / "21_months.train.csv"
    test = folder / "21_months.test.csv"
    _processed_csv(train, [["2001-09-30 23:59:42", 1.0, 0, 1, 0], ["2001-10-01 00:00:00", 2.0, 1, 0, 0]])
    _processed_csv(test, [["2001-10-01 00:00:18", 3.0, 0, 0, 2]])
    supplied = {"esa": {"version": "2"}}
    missing_version = inspect_source(train)
    assert any("esa.version" in issue for issue in missing_version["issues"])
    inspected = inspect_source(train, supplied)
    assert inspected["format"] == "esa-processed" and not inspected["issues"]
    result = ingest_source(train, supplied, db_path=tmp_path / "catalog.db", storage_dir=tmp_path / "runtime", batch_size=1)
    assert result["row_count"] == 3
    assert result["input_source_path"] == str(train)
    assert result["annotation_format"] == "esa-points"
    assert set(pd.read_parquet(result["annotations_path"]).label_value) == {1, 2}
    batch = next(iter_batches(result["dataset_id"], db_path=tmp_path / "catalog.db"))
    assert not any("anomaly" in name for name in batch.schema.names)
    again = ingest_source(train, supplied, db_path=tmp_path / "catalog.db", storage_dir=tmp_path / "runtime", batch_size=1)
    assert again["dataset_id"] == result["dataset_id"] and again["duplicate"] is True


def test_processed_esa_directory_requires_train_choice(tmp_path):
    folder = tmp_path / "ESA-Mission1-semi-supervised"
    one = [["2000-03-31 23:59:30", 1, 0, 0, 0]]
    _processed_csv(folder / "3_months.train.csv", one)
    _processed_csv(folder / "84_months.train.csv", [["2006-12-31 23:59:30", 1, 0, 0, 0]])
    _processed_csv(folder / "84_months.test.csv", [["2007-01-01 00:00:30", 2, 0, 0, 0]])
    with pytest.raises(ValueError, match="processed_train"):
        prepare_processed_esa_source(folder, {"esa": {"version": "2"}}, storage_dir=tmp_path / "store")
    _, manifest = prepare_processed_esa_source(folder, {"esa": {"version": "2", "processed_train": "3_months"}}, storage_dir=tmp_path / "store", batch_size=1)
    assert manifest["esa"]["processed_train"] == "3_months"


def test_processed_esa_lone_csv_needs_explicit_partition_and_splits(tmp_path):
    train = tmp_path / "ESA-Mission1-semi-supervised" / "3_months.train.csv"
    _processed_csv(train, [["2000-03-31 23:59:30", 1, 0, 0, 0]])
    with pytest.raises(ValueError, match="allow_single"):
        prepare_processed_esa_source(train, {"esa": {"version": "2"}}, storage_dir=tmp_path / "store")
    with pytest.raises(ValueError, match="explicit manifest splits"):
        prepare_processed_esa_source(train, {"esa": {"version": "2", "allow_single": True, "single_partition": "train"}}, storage_dir=tmp_path / "store")
    output, manifest = prepare_processed_esa_source(train, {"esa": {"version": "2", "allow_single": True, "single_partition": "train"},
                                                      "splits": {"fractions": [.7, .15, .15]}}, storage_dir=tmp_path / "store")
    assert pd.read_parquet(output).shape[0] == 1
    assert manifest["annotation_format"] == "esa-points"
