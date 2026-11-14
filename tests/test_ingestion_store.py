import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.db import get_connection
from src.ingestion import store
from src.ingestion.manifest import column_names
from src.ingestion.manifest import load_manifest


def write_source(path, records):
    if path.suffix in {".csv", ".tsv"}:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="\t" if path.suffix == ".tsv" else ",")
            writer.writeheader()
            writer.writerows(records)
    elif path.suffix == ".json":
        path.write_text(json.dumps(records), encoding="utf-8")
    elif path.suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    else:
        pq.write_table(pa.Table.from_pylist(records), path)


def ingest(tmp_path, path, manifest=None, batch_size=2):
    return store.ingest_source(path, manifest, storage_dir=tmp_path / "storage", db_path=tmp_path / "catalog.db", batch_size=batch_size)


def read(tmp_path, dataset):
    return [row for batch in store.iter_batches(dataset["dataset_id"], db_path=tmp_path / "catalog.db", batch_size=2) for row in batch.to_pylist()]


@pytest.fixture
def records():
    return [
        {"timestamp": "2024-01-01T00:00:02Z", "voltage": 3.0, "mode": "B", "is_anomaly_voltage": 1},
        {"timestamp": "2024-01-01T00:00:00Z", "voltage": 1.0, "mode": "A", "is_anomaly_voltage": 0},
        {"timestamp": "2024-01-01T00:00:01Z", "voltage": 2.0, "mode": "A", "is_anomaly_voltage": 0},
    ]


def test_formats_and_chunks_equivalent(tmp_path, records):
    results = []
    for suffix in (".csv", ".tsv", ".json", ".jsonl", ".parquet"):
        path = tmp_path / ("input" + suffix)
        write_source(path, records)
        for batch_size in (1, 2, 100):
            dataset = ingest(tmp_path, path, batch_size=batch_size)
            results.append(read(tmp_path, dataset))
            assert dataset["row_count"] == 3
            assert not any("is_anomaly" in key for row in results[-1] for key in row)
            assert pq.read_table(dataset["annotations_path"]).num_rows == 3
    assert all(result == results[0] for result in results)
    assert [r["row_id"] for r in results[0]] == [1, 2, 3]
    assert [r["voltage"] for r in results[0]] == [1, 2, 3]


def test_conflicting_duplicate_and_retry(tmp_path, records):
    path = tmp_path / "duplicates.csv"
    write_source(path, records + [dict(records[0], voltage=20)])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        ingest(tmp_path, path)
    with get_connection(tmp_path / "catalog.db") as conn:
        assert conn.execute("SELECT count(*) FROM datasets").fetchone()[0] == 0
    resolved = ingest(tmp_path, path, {"duplicate_policy": "last"})
    assert read(tmp_path, resolved)[-1]["voltage"] == 20
    duplicate = ingest(tmp_path, path, {"duplicate_policy": "last"})
    assert duplicate["duplicate"]
    assert duplicate["dataset_id"] == resolved["dataset_id"]


def test_identical_duplicates_collapse(tmp_path, records):
    path = tmp_path / "identical.json"
    write_source(path, records + [records[0]])
    dataset = ingest(tmp_path, path)
    assert dataset["row_count"] == 3
    assert dataset["source_rows"] == 4


def test_adapter_presorted_fast_path_validates_order_and_duplicates(tmp_path):
    path = tmp_path / "prepared.parquet"
    ordered = [
        {"timestamp": "2024-01-01T00:00:00Z", "voltage": 1.0},
        {"timestamp": "2024-01-01T00:00:00Z", "voltage": 1.0},
        {"timestamp": "2024-01-01T00:00:01Z", "voltage": 2.0},
    ]
    write_source(path, ordered)
    result = ingest(tmp_path, path, {"_source_sorted_unique": True}, batch_size=1)
    assert result["row_count"] == 2
    assert [row["voltage"] for row in read(tmp_path, result)] == [1.0, 2.0]

    path = tmp_path / "misordered.parquet"
    write_source(path, list(reversed(ordered[::2])))
    with pytest.raises(ValueError, match="declared chronological order but is unsorted"):
        ingest(tmp_path, path, {"_source_sorted_unique": True}, batch_size=1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("policy", ["first", "last"])
def test_presorted_matches_generic_duplicate_policy_across_parts(tmp_path, batch_size, policy):
    path = tmp_path / "prepared.jsonl"
    records = [
        {"timestamp": "2024-01-01T00:00:00Z", "voltage": value, "label": index}
        for index, value in enumerate([1.0, 10.0, 11.0, 12.0, 13.0])
    ] + [
        {"timestamp": "2024-01-01T00:00:01Z", "voltage": 2.0, "label": 0},
        {"timestamp": "2024-01-01T00:00:02Z", "voltage": 3.0, "label": 0},
        {"timestamp": "2024-01-01T00:00:02Z", "voltage": 30.0, "label": 1},
        {"timestamp": "2024-01-01T00:00:03Z", "voltage": 4.0, "label": 0},
    ]
    write_source(path, records)
    generic = ingest(tmp_path, path, {"duplicate_policy": policy}, batch_size=batch_size)
    prepared = ingest(tmp_path, path, {"duplicate_policy": policy, "_source_sorted_unique": True},
                      batch_size=batch_size)
    assert read(tmp_path, prepared) == read(tmp_path, generic)
    assert [row["voltage"] for row in read(tmp_path, prepared)] == (
        [1.0, 2.0, 3.0, 4.0] if policy == "first" else [13.0, 2.0, 30.0, 4.0]
    )
    assert pq.read_table(prepared["annotations_path"]).to_pylist() == pq.read_table(
        generic["annotations_path"]
    ).to_pylist()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("presorted", [False, True])
def test_conflicting_duplicates_rejected_across_parts(tmp_path, batch_size, presorted):
    path = tmp_path / "conflicting.jsonl"
    records = [{"timestamp": "2024-01-01T00:00:00Z", "voltage": 1.0}] * 5
    records.append({"timestamp": "2024-01-01T00:00:00Z", "voltage": 2.0})
    write_source(path, records)
    with pytest.raises(ValueError, match="Conflicting duplicate observations"):
        ingest(tmp_path, path, {"_source_sorted_unique": presorted}, batch_size=batch_size)
    with get_connection(tmp_path / "catalog.db") as conn:
        assert conn.execute("SELECT count(*) FROM datasets").fetchone()[0] == 0


@pytest.mark.parametrize("failure", [None, "duplicate", "schema"])
def test_presorted_opens_one_part_at_a_time_and_closes_on_failure(tmp_path, monkeypatch, failure):
    parts = []
    for index in range(6):
        row = {"entity_id": "", "timestamp_ns": index, "_seq": index + 1, "voltage": 1.0}
        if failure == "duplicate" and index == 3:
            row.update(timestamp_ns=2, voltage=2.0)
        if failure == "schema" and index == 3:
            row["voltage"] = "wrong type"
        part = tmp_path / f"part-{index}.parquet"
        pq.write_table(pa.Table.from_pylist([row]), part)
        parts.append(str(part))

    original_reader = pq.ParquetFile
    active = set()
    maximum_open = 0

    class TrackedReader:
        def __init__(self, *args, **kwargs):
            nonlocal maximum_open
            self.reader = original_reader(*args, **kwargs)
            active.add(self)
            maximum_open = max(maximum_open, len(active))

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def close(self):
            try:
                self.reader.close()
            finally:
                active.remove(self)

    monkeypatch.setattr(store.pq, "ParquetFile", TrackedReader)
    output = tmp_path / "dedup.parquet"
    arguments = (parts, ["entity_id", "timestamp_ns"], ["entity_id", "timestamp_ns", "voltage"],
                 "error", output, 1)
    if failure:
        message = "Conflicting duplicate" if failure == "duplicate" else "inconsistent schemas"
        with pytest.raises(ValueError, match=message):
            store._presorted_dedup(*arguments)
        assert not output.exists()
        assert not output.with_suffix(".parquet.tmp").exists()
    else:
        store._presorted_dedup(*arguments)
        assert pq.read_table(output).column("timestamp_ns").to_pylist() == list(range(6))
    assert maximum_open == 1
    assert not active


def test_streaming_duplicate_policy_across_batches(tmp_path):
    path = tmp_path / "cross-batch.jsonl"
    records = [{"timestamp": "2024-01-01T00:00:00Z", "voltage": 1, "label": 0},
               {"timestamp": "2024-01-01T00:00:01Z", "voltage": 2, "label": 0},
               {"timestamp": "2024-01-01T00:00:00Z", "voltage": 3, "label": 1}]
    write_source(path, records)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        ingest(tmp_path, path, batch_size=1)
    first = ingest(tmp_path, path, {"duplicate_policy": "first"}, batch_size=1)
    last = ingest(tmp_path, path, {"duplicate_policy": "last"}, batch_size=1)
    assert [row["voltage"] for row in read(tmp_path, first)] == [1, 2]
    assert [row["voltage"] for row in read(tmp_path, last)] == [3, 2]
    assert [row["row_id"] for row in read(tmp_path, last)] == [1, 2]
    assert pq.read_table(first["annotations_path"]).column("label_value").to_pylist()[0] == "0"
    assert pq.read_table(last["annotations_path"]).column("label_value").to_pylist()[0] == "1"


def test_ten_thousand_wide_rows_under_small_duckdb_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("COMET_DUCKDB_MEMORY_LIMIT", "64MB")
    path = tmp_path / "wide.parquet"
    rows = [{"timestamp": 1704067200 + index, **{f"sensor_{feature}": float(index + feature)
                                                   for feature in range(32)}}
            for index in reversed(range(10000))]
    rows.append(dict(rows[-1]))
    write_source(path, rows)
    result = ingest(tmp_path, path, {"epoch_unit": "s"}, batch_size=37)
    assert result["source_rows"] == 10001
    assert result["row_count"] == 10000
    assert [row["sensor_0"] for row in store.read_window(result["dataset_id"], 1, 2,
                                                          db_path=tmp_path / "catalog.db")] == [0.0, 1.0]
    timestamps = pq.read_table(result["telemetry_path"], columns=["timestamp_ns"]).column(0).to_pylist()
    assert timestamps == sorted(timestamps)
    staged_dedup = tmp_path / "storage" / "staging" / result["import_fingerprint"] / "dedup.parquet"
    assert staged_dedup.is_dir()
    assert len(list(staged_dedup.glob("part-*.parquet"))) < 500


def prepared_esa_manifest():
    return {"timestamp": "timestamp", "numeric_features": ["channel.one", "channel one"],
            "categorical_features": ["mode"], "context_columns": ["comet_gap"],
            "gap_column": "comet_gap", "label_columns": [], "entity_columns": [],
            "timezone": "UTC", "_source_sorted_unique": True,
            "esa": {"version": 2, "adapter_version": "test"}}


def prepared_esa_rows():
    return [
        {"timestamp": "2024-01-01T01:00:00.123456789+01:00", "channel.one": 1.0,
         "channel one": None, "mode": "001", "comet_gap": 0.0},
        {"timestamp": "2024-01-01T00:00:00.123456790Z", "channel.one": None,
         "channel one": 2.0, "mode": "True", "comet_gap": 1.0},
        {"timestamp": "2024-01-01 00:00:01.000000123", "channel.one": 3.0,
         "channel one": 3.0, "mode": "", "comet_gap": 0.0},
        {"timestamp": "2024-01-01T00:00:02Z", "channel.one": 4.0,
         "channel one": 4.0, "mode": None, "comet_gap": 0.0},
    ]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 100])
def test_prepared_esa_arrow_staging_matches_generic_rows(tmp_path, monkeypatch, batch_size):
    path = tmp_path / "esa-prepared.parquet"
    write_source(path, prepared_esa_rows())
    manifest = prepared_esa_manifest()
    generic = ingest(tmp_path, path, {key: value for key, value in manifest.items() if key != "esa"},
                     batch_size=batch_size)

    def reject_python_normalization(*args, **kwargs):
        raise AssertionError("Prepared ESA staging must not normalize row dictionaries")

    monkeypatch.setattr(store, "_normalized", reject_python_normalization)
    prepared = ingest(tmp_path, path, manifest, batch_size=batch_size)
    assert read(tmp_path, prepared) == read(tmp_path, generic)
    rows = read(tmp_path, prepared)
    assert rows[0]["timestamp_ns"] == 1704067200123456789
    assert rows[1]["timestamp_ns"] == rows[0]["timestamp_ns"] + 1
    assert rows[2]["timestamp"] == "2024-01-01T00:00:01.000000123+00:00"
    assert [row["mode"] for row in rows] == ["001", "True", "", None]
    assert [row["comet_gap"] for row in rows] == ["0", "1", "0", "0"]


@pytest.mark.parametrize("field,value,message", [
    ("channel.one", float("inf"), "nonfinite"),
    ("channel.one", float("nan"), "nonfinite"),
    ("timestamp", "invalid", "Invalid timestamp"),
    ("timestamp", "NaT", "Invalid timestamp"),
    ("timestamp", None, "timestamps cannot be missing"),
    ("timestamp", "1704067200", "Numeric timestamps require"),
    ("comet_gap", 2.0, "gap_column values"),
    ("comet_gap", None, "gap_column values"),
])
def test_prepared_esa_arrow_staging_validates_later_batches(tmp_path, field, value, message):
    path = tmp_path / "bad-esa-prepared.parquet"
    rows = prepared_esa_rows()
    rows[-1][field] = value
    write_source(path, rows)
    with pytest.raises(ValueError, match=message):
        ingest(tmp_path, path, prepared_esa_manifest(), batch_size=2)
    with get_connection(tmp_path / "catalog.db") as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'").fetchone():
            assert conn.execute("SELECT count(*) FROM datasets").fetchone()[0] == 0


@pytest.mark.parametrize("numeric_timestamp", ["２０２４", "٢٠٢٤"])
@pytest.mark.parametrize("prepared", [False, True])
def test_unicode_numeric_timestamps_rejected_beyond_inspection_sample(tmp_path, numeric_timestamp, prepared):
    path = tmp_path / "unicode-numeric-timestamp.parquet"
    template = prepared_esa_rows()[0]
    rows = [{**template, "timestamp": f"2024-01-01T00:{index // 60:02}:{index % 60:02}Z"}
            for index in range(257)]
    rows[-1]["timestamp"] = numeric_timestamp
    write_source(path, rows)
    manifest = prepared_esa_manifest()
    if not prepared:
        manifest.pop("esa")
    with pytest.raises(ValueError, match="Numeric timestamps require an explicit epoch_unit"):
        ingest(tmp_path, path, manifest, batch_size=17)
    with get_connection(tmp_path / "catalog.db") as conn:
        assert conn.execute("SELECT count(*) FROM datasets").fetchone()[0] == 0


@pytest.mark.parametrize("field,value", [("channel.one", "1.0"), ("mode", 1.0)])
def test_prepared_esa_arrow_staging_rejects_incompatible_schema(tmp_path, field, value):
    path = tmp_path / "bad-schema-esa-prepared.parquet"
    rows = prepared_esa_rows()
    for row in rows:
        row[field] = value
    write_source(path, rows)
    with pytest.raises(ValueError, match="incompatible schema for its declared role"):
        ingest(tmp_path, path, prepared_esa_manifest(), batch_size=2)


def test_prepared_esa_arrow_staging_resumes_across_changed_batch_size(tmp_path, monkeypatch):
    path = tmp_path / "resume-esa-prepared.parquet"
    write_source(path, prepared_esa_rows())
    original_checkpoint = store._atomic_json
    stopped = False

    def stop_after_checkpoint(path, data):
        nonlocal stopped
        original_checkpoint(path, data)
        if path.name == "progress.json" and data.get("rows_staged") == 2 and not stopped:
            stopped = True
            raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr(store, "_atomic_json", stop_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        ingest(tmp_path, path, prepared_esa_manifest(), batch_size=2)
    monkeypatch.setattr(store, "_atomic_json", original_checkpoint)
    prepared = ingest(tmp_path, path, prepared_esa_manifest(), batch_size=3)
    manifest = prepared_esa_manifest()
    generic = ingest(tmp_path, path, {key: value for key, value in manifest.items() if key != "esa"}, batch_size=2)
    assert read(tmp_path, prepared) == read(tmp_path, generic)
    assert prepared["source_rows"] == 4


def test_shuffled_mixed_category_requires_explicit_role(tmp_path):
    path = tmp_path / "shuffled.jsonl"
    write_source(path, [{"timestamp": "2024-01-02T00:00:00Z", "mode": "nominal"},
                        {"timestamp": "2024-01-01T00:00:00Z", "mode": "2"}])
    assert store.inspect_source(path)["columns"][0]["role"] == "categorical"
    with pytest.raises(ValueError, match="mixes numeric and categorical"):
        ingest(tmp_path, path, batch_size=1)
    explicit = ingest(tmp_path, path, {"categorical_features": ["mode"]}, batch_size=1)
    assert [row["mode"] for row in read(tmp_path, explicit)] == ["2", "nominal"]


def test_processed_esa_inspection_flags_training_pair_ambiguity(tmp_path):
    directory = tmp_path / "ESA-Mission1-semi-supervised"
    directory.mkdir()
    for name in ("3_months.train.csv", "6_months.train.csv", "84_months.test.csv"):
        (directory / name).write_text("timestamp,x\n2024-01-01,1\n", encoding="utf-8")
    for source in (directory, directory / "84_months.test.csv"):
        inspected = store.inspect_source(source, {"esa": {"version": "2"}})
        assert inspected["format"] == "esa-processed"
        assert inspected["requires_manifest"]
        assert "processed_train" in inspected["issues"][0]
    missing_version = store.inspect_source(directory / "3_months.train.csv")
    assert any("esa.version" in issue for issue in missing_version["issues"])
    chosen = store.inspect_source(directory / "3_months.train.csv", {"esa": {"version": "2"}})
    assert not chosen["requires_manifest"]


@pytest.mark.parametrize("bad", [float("inf"), "NaN", "text"])
def test_every_numeric_record_validated(tmp_path, records, bad):
    path = tmp_path / "invalid.csv"
    write_source(path, records + [dict(records[0], voltage=bad)])
    with pytest.raises(ValueError, match="numeric|nonfinite"):
        ingest(tmp_path, path, {"numeric_features": ["voltage"]})


def test_duplicate_headers_and_json_keys_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,x,x\n2024-01-01,1,2\n")
    with pytest.raises(ValueError, match="unique"):
        store.inspect_source(path)
    for suffix, text in ((".json", '[{"timestamp":"2024-01-01","x":1,"x":2}]'), (".jsonl", '{"timestamp":"2024-01-01","x":1,"x":2}')):
        path = tmp_path / ("bad" + suffix)
        path.write_text(text)
        with pytest.raises(ValueError, match="Duplicate JSON key"):
            store.inspect_source(path)


def test_collisions_are_unique_and_stable():
    names = ["a b", "a-b", "a_b", "a_b_2", "ROW ID", "timestamp", "1a", "___"]
    mapped = column_names(names)
    assert mapped == column_names(list(reversed(names)))
    assert len(set(mapped.values())) == len(names)
    assert not set(mapped.values()).intersection(store.CORE)


def test_label_cannot_be_feature_context_or_entity(tmp_path, records):
    path = tmp_path / "label.csv"
    write_source(path, records)
    for key in ("numeric_features", "categorical_features", "context_columns", "entity_columns"):
        with pytest.raises(ValueError, match="Annotation|annotation"):
            ingest(tmp_path, path, {key: ["is_anomaly_voltage"]})


def test_long_layout_entities_context_and_sparse_channels(tmp_path):
    records = [{"time": "2024-01-01T00:00:00Z", "craft": "B", "channel": "a", "value": 4, "note": "hi", "label": 0},
               {"time": "2024-01-01T00:00:00Z", "craft": "A", "channel": "b", "value": 2, "note": "ok", "label": 1},
               {"time": "2024-01-01T00:00:00Z", "craft": "A", "channel": "a", "value": 1, "note": "ok", "label": 0}]
    path = tmp_path / "long.jsonl"
    write_source(path, records)
    result = ingest(tmp_path, path, {"layout": "long", "timestamp": "time", "entity_columns": ["craft"], "context_columns": ["note"]})
    rows = read(tmp_path, result)
    assert len(rows) == 2
    assert rows[0]["a"] == 1 and rows[0]["b"] == 2
    assert rows[1]["a"] == 4 and rows[1]["b"] is None
    assert rows[0]["note"] == "ok"
    assert set(pq.read_table(result["annotations_path"]).column("channel").to_pylist()) == {"a", "b"}


def test_interruption_resumes_committed_parts(tmp_path, records, monkeypatch):
    path = tmp_path / "resume.json"
    write_source(path, records)
    original = store._atomic_json
    stopped = False
    def stop_after_checkpoint(path, data):
        nonlocal stopped
        original(path, data)
        if not stopped and data.get("status") == "staging":
            stopped = True
            raise RuntimeError("simulated process interruption")
    monkeypatch.setattr(store, "_atomic_json", stop_after_checkpoint)
    with pytest.raises(RuntimeError, match="interruption"):
        ingest(tmp_path, path, batch_size=1)
    monkeypatch.setattr(store, "_atomic_json", original)
    result = ingest(tmp_path, path, batch_size=2)
    assert result["row_count"] == 3
    assert [r["voltage"] for r in read(tmp_path, result)] == [1, 2, 3]


def test_numeric_timestamp_needs_unit_and_preserves_nanoseconds(tmp_path):
    path = tmp_path / "epoch.json"
    records = [{"timestamp": 1704067200000000001, "x": 1}, {"timestamp": 1704067200000000002, "x": 2}]
    write_source(path, records)
    assert store.inspect_source(path)["requires_manifest"]
    result = ingest(tmp_path, path, {"epoch_unit": "ns"})
    assert [r["timestamp_ns"] for r in read(tmp_path, result)] == [r["timestamp"] for r in records]


def test_invalid_timestamp_and_inconsistent_json_schema(tmp_path):
    path = tmp_path / "bad.json"
    write_source(path, [{"timestamp": "invalid", "x": 1}])
    with pytest.raises(ValueError, match="timestamp"):
        ingest(tmp_path, path)
    write_source(path, [{"timestamp": "2024-01-01", "x": 1}, {"timestamp": "2024-01-02", "y": 2}])
    with pytest.raises(ValueError, match="inconsistent columns"):
        ingest(tmp_path, path)


def test_bounded_profile_and_window_isolation(tmp_path, records):
    path = tmp_path / "profile.csv"
    write_source(path, records)
    dataset = ingest(tmp_path, path)
    profile = store.profile_dataset(dataset["dataset_id"], db_path=tmp_path / "catalog.db", cardinality_limit=2)
    assert profile["columns"]["voltage"]["unique_count_approximate"]
    assert profile["columns"]["voltage"]["unique_count_lower_bound"] == 2
    window = store.read_window(dataset["dataset_id"], 2, 2, db_path=tmp_path / "catalog.db")
    assert len(window) == 1 and window[0]["voltage"] == 2
    assert "is_anomaly_voltage" not in window[0]


def test_auto_schema_does_not_change_when_future_type_changes(tmp_path):
    path = tmp_path / "schema.json"
    records = [{"timestamp": "2024-01-01", "x": 1}, {"timestamp": "2024-01-02", "x": 2}]
    write_source(path, records)
    before = store.inspect_source(path)["columns"]
    records[-1]["x"] = "unexpected category"
    write_source(path, records)
    after = store.inspect_source(path)["columns"]
    assert after == before
    assert after[0]["role"] == "numeric"
    with pytest.raises(ValueError, match="nonnumeric"):
        ingest(tmp_path, path)


def test_each_timestamp_requires_epoch_unit_even_after_inspection(tmp_path):
    path = tmp_path / "timestamps.jsonl"
    records = [{"timestamp": f"2024-01-{i % 28 + 1:02d}T00:00:{i // 28:02d}Z", "x": 1} for i in range(256)]
    records.append({"timestamp": 1704067200000000001, "x": 2})
    write_source(path, records)
    assert not store.inspect_source(path)["requires_manifest"]
    with pytest.raises(ValueError, match="epoch_unit"):
        ingest(tmp_path, path)


def test_units_preserved_for_wide_and_long(tmp_path):
    path = tmp_path / "units.json"
    write_source(path, [{"timestamp": "2024-01-01", "voltage": 1}])
    result = ingest(tmp_path, path, {"units": {"voltage": "V"}})
    assert result["columns"][0]["unit"] == "V"
    write_source(path, [{"timestamp": "2024-01-01", "channel": "voltage", "value": 1}])
    result = ingest(tmp_path, path, {"layout": "long", "units": {"voltage": "V"}})
    assert result["columns"][0]["unit"] == "V"
    # The persisted source manifest must be reusable for the same layout.
    assert ingest(tmp_path, path, result["manifest"])["dataset_id"] == result["dataset_id"]
    with pytest.raises(ValueError, match="units"):
        load_manifest({"units": {"voltage": 2}})


def test_default_split_is_deferred_until_generic_resolution(tmp_path, records):
    assert "splits" not in load_manifest({"esa": {"mission": 1}})
    path = tmp_path / "generic.json"
    write_source(path, records)
    assert store.inspect_source(path)["manifest"]["splits"] == {"fractions": [.7, .15, .15]}
    prescribed = {"test_start": "2024-01-01T00:00:02Z", "train_fraction": .85}
    assert store.inspect_source(path, {"splits": prescribed})["manifest"]["splits"] == prescribed
    with pytest.raises(ValueError, match="Choose fractions"):
        load_manifest({"splits": {"fractions": [.7, .15, .15], "test_start": "2024-01-01"}})


def test_gap_marker_is_context_and_binary(tmp_path):
    path = tmp_path / "gap.json"
    write_source(path, [{"timestamp": "2024-01-01", "x": 1, "esa_gap": False}, {"timestamp": "2024-01-02", "x": 2, "esa_gap": True}])
    manifest = {"gap_column": "esa_gap", "context_columns": ["esa_gap"]}
    result = ingest(tmp_path, path, manifest)
    assert [row["esa_gap"] for row in read(tmp_path, result)] == ["0", "1"]
    assert next(c for c in result["columns"] if c["name"] == "esa_gap")["role"] == "context"
    with pytest.raises(ValueError, match="gap_column"):
        ingest(tmp_path, path, {"gap_column": "esa_gap"})
    write_source(path, [{"timestamp": "2024-01-01", "x": 1, "esa_gap": 2}])
    with pytest.raises(ValueError, match="gap_column"):
        ingest(tmp_path, path, manifest)


def test_legacy_reader_normalizes_training_but_preserves_historical_windows(tmp_path):
    from src.db import init_db, insert_dataset, utc_now
    db_path = tmp_path / "catalog.db"
    init_db(db_path)
    dataset_id = insert_dataset("legacy", status="ingested", row_count=2, db_path=db_path)
    with get_connection(db_path) as conn:
        conn.execute(f"CREATE TABLE raw_dataset_{dataset_id}(row_id INTEGER PRIMARY KEY,timestamp TEXT,x TEXT,is_anomaly TEXT,secret TEXT)")
        conn.executemany(f"INSERT INTO raw_dataset_{dataset_id} VALUES(?,?,?,?,?)", [(1, "2024-01-01", "1", "0", "private"), (2, "2024-01-02", "2", "1", "private")])
        conn.executemany("INSERT INTO dataset_columns(dataset_id,column_name,sanitized_name,inferred_type,role,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", [(dataset_id, name, name, dtype, role, utc_now(), utc_now()) for name, dtype, role in [("timestamp", "timestamp", "timestamp"), ("x", "numeric", "numeric_telemetry_feature"), ("is_anomaly", "numeric", "numeric_telemetry_feature"), ("secret", "text", "annotation")]])
    metadata = store.load_dataset(dataset_id, db_path=db_path)
    assert metadata["columns"] == [{"name": "x", "source_name": "x", "role": "numeric", "dtype": "numeric"}]
    modern = [row for batch in store.iter_batches(dataset_id, db_path=db_path, batch_size=1) for row in batch.to_pylist()]
    assert modern[0]["x"] == 1 and "timestamp_ns" in modern[0]
    assert all("is_anomaly" not in row and "secret" not in row for row in modern)
    with get_connection(db_path) as conn:
        conn.execute(f"UPDATE raw_dataset_{dataset_id} SET timestamp='2023-01-01' WHERE row_id=2")
    with pytest.raises(ValueError, match="reimport"):
        list(store.iter_batches(dataset_id, db_path=db_path))
    historical = store.read_window(dataset_id, 2, 2, db_path=db_path)
    assert historical == [{"row_id": 2, "timestamp": "2023-01-01", "x": "2"}]


def test_reserved_annotation_prefix_and_structural_mapping_collision(tmp_path):
    path = tmp_path / "annotation.json"
    write_source(path, [{"timestamp": "2024-01-01", "x": 1, "is_anomalySensor": 1}])
    result = ingest(tmp_path, path)
    assert [c["source_name"] for c in result["columns"]] == ["x"]
    with pytest.raises(ValueError, match="distinct columns"):
        ingest(tmp_path, path, {"entity_columns": ["timestamp"]})
    with pytest.raises(ValueError, match="numeric feature"):
        ingest(tmp_path, path, {"units": {"is_anomalySensor": "truth"}})


def test_common_ground_truth_columns_never_enter_telemetry_or_context(tmp_path):
    path = tmp_path / "ground_truth.json"
    write_source(path, [
        {"timestamp": "2024-01-01T00:00:00Z", "signal": 1,
         "anomaly_type": "drift", "fault_code": "F01", "is_outlier": 1, "ground_truth": 1},
        {"timestamp": "2024-01-01T00:00:01Z", "signal": 2,
         "anomaly_type": "none", "fault_code": "none", "is_outlier": 0, "ground_truth": 0},
    ])
    result = ingest(tmp_path, path)
    assert [item["source_name"] for item in result["columns"]] == ["signal"]
    assert set(read(tmp_path, result)[0]) == {"row_id", "timestamp", "timestamp_ns", "entity_id", "signal"}
    assert pq.read_table(result["annotations_path"]).num_rows == 8
    with pytest.raises(ValueError, match="Annotation column"):
        ingest(tmp_path, path, {"context_columns": ["fault_code"]})


@pytest.mark.parametrize("batch_size", [1, 13, 60])
def test_parquet_parts_bound_rows_bytes_and_footer_metadata(tmp_path, batch_size):
    table = pa.table({"row_id": range(60), "text": ["x" * (20 if index % 2 else 100) for index in range(60)]})
    destination = tmp_path / "parts.parquet"
    with store._ParquetPartWriter(destination, table.schema, max_rows=7, max_bytes=500) as writer:
        for batch in table.to_batches(max_chunksize=batch_size):
            writer.write_batch(batch)
    assert writer.rows_written == 60
    parts = store._parquet_parts(destination)
    assert len(parts) > 1
    for part in parts:
        with pq.ParquetFile(part) as reader:
            assert reader.metadata.num_rows <= 7
            assert reader.num_row_groups == 1
            assert reader.read().nbytes <= 500
    assert pq.read_table(destination).equals(table)
    assert not destination.with_suffix(".parquet.tmp").exists()


def test_parquet_part_writer_rejects_oversized_record_without_publishing(tmp_path):
    table = pa.table({"payload": ["x" * 1000]})
    destination = tmp_path / "too-large.parquet"
    with pytest.raises(ValueError, match="One telemetry record exceeds"):
        with store._ParquetPartWriter(destination, table.schema, max_bytes=100) as writer:
            writer.write_batch(table.to_batches()[0])
    assert not destination.exists()
    assert not destination.with_suffix(".parquet.tmp").exists()


@pytest.mark.parametrize("batch_size", [1, 4, 100])
def test_ingestion_parts_preserve_order_ids_annotations_and_windows(tmp_path, monkeypatch, batch_size):
    monkeypatch.setattr(store, "PARQUET_PART_ROWS", 3)
    path = tmp_path / "parts.jsonl"
    rows = [{"timestamp": 1704067200 + index, "x": float(index), "label": int(index == 5)}
            for index in reversed(range(10))]
    write_source(path, rows + [dict(rows[0])])
    dataset = ingest(tmp_path, path, {"epoch_unit": "s"}, batch_size=batch_size)
    assert dataset["telemetry_format"] == "parquet-parts"
    assert Path(dataset["telemetry_path"]).is_dir()
    assert dataset["row_count"] == 10
    assert dataset["source_rows"] == 11
    parts = store._parquet_parts(dataset["telemetry_path"])
    assert len(parts) == 4
    result = read(tmp_path, dataset)
    assert [row["row_id"] for row in result] == list(range(1, 11))
    assert [row["x"] for row in result] == list(range(10))
    assert [row["label_value"] for row in pq.read_table(dataset["annotations_path"]).to_pylist()] == ["0"] * 5 + ["1"] + ["0"] * 4
    assert store.read_window(dataset["dataset_id"], 3, 8, db_path=tmp_path / "catalog.db") == result[2:8]


def test_interrupted_part_writing_retries_without_stale_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PARQUET_PART_ROWS", 3)
    path = tmp_path / "interrupted-parts.jsonl"
    write_source(path, [{"timestamp": 1704067200 + index, "x": index} for index in range(10)])
    original = store._ParquetPartWriter._flush
    stopped = False

    def stop_after_part(writer):
        nonlocal stopped
        original(writer)
        if not stopped:
            stopped = True
            raise KeyboardInterrupt("simulated part interruption")

    monkeypatch.setattr(store._ParquetPartWriter, "_flush", stop_after_part)
    with pytest.raises(KeyboardInterrupt, match="part interruption"):
        ingest(tmp_path, path, {"epoch_unit": "s"}, batch_size=2)
    assert not list((tmp_path / "storage" / "datasets").iterdir())
    assert not list((tmp_path / "storage" / "staging").glob("*/dedup.parquet.tmp"))
    monkeypatch.setattr(store._ParquetPartWriter, "_flush", original)
    result = ingest(tmp_path, path, {"epoch_unit": "s"}, batch_size=7)
    assert result["row_count"] == 10
    assert [row["x"] for row in read(tmp_path, result)] == list(range(10))
    assert ingest(tmp_path, path, {"epoch_unit": "s"})["dataset_id"] == result["dataset_id"]


def test_existing_v2_single_file_reader_remains_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PARQUET_PART_ROWS", 2)
    path = tmp_path / "old-reader.jsonl"
    write_source(path, [{"timestamp": 1704067200 + index, "x": index} for index in range(5)])
    dataset = ingest(tmp_path, path, {"epoch_unit": "s"})
    expected = read(tmp_path, dataset)
    historical = tmp_path / "historical-telemetry.parquet"
    pq.write_table(pq.read_table(dataset["telemetry_path"]), historical, row_group_size=2)
    with get_connection(tmp_path / "catalog.db") as conn:
        row = conn.execute("SELECT metadata_json FROM datasets WHERE dataset_id=?", (dataset["dataset_id"],)).fetchone()
        metadata = json.loads(row[0])
        metadata["telemetry_path"] = str(historical)
        metadata.pop("telemetry_format")
        conn.execute("UPDATE datasets SET metadata_json=? WHERE dataset_id=?", (json.dumps(metadata), dataset["dataset_id"]))
    assert read(tmp_path, dataset) == expected
    assert store.read_window(dataset["dataset_id"], 2, 4, db_path=tmp_path / "catalog.db") == expected[1:4]


def test_part_reader_closes_each_footer_and_on_early_consumer_stop(tmp_path, monkeypatch):
    destination = tmp_path / "sequential.parquet"
    table = pa.table({"row_id": range(10)})
    with store._ParquetPartWriter(destination, table.schema, max_rows=3) as writer:
        writer.write_batch(table.to_batches()[0])
    original = pq.ParquetFile
    active = set()
    maximum = 0

    class TrackedReader:
        def __init__(self, *args, **kwargs):
            nonlocal maximum
            self.reader = original(*args, **kwargs)
            active.add(self)
            maximum = max(maximum, len(active))

        def iter_batches(self, *args, **kwargs):
            return self.reader.iter_batches(*args, **kwargs)

        def close(self):
            self.reader.close()
            active.remove(self)

    monkeypatch.setattr(store.pq, "ParquetFile", TrackedReader)
    for stop_early in (True, False):
        batches = store._parquet_batches(destination, batch_size=2)
        assert next(batches).column("row_id").to_pylist() == [0, 1]
        if not stop_early:
            list(batches)
        batches.close()
        assert not active
    assert maximum == 1
