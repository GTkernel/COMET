from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from src.modeling.transforms import (build_row_index, fit_transform, resolve_features,
                                     transform_to_memmap)
from src.modeling.window_dataset import WindowDataset, build_window_index


COLUMNS = [{"name": "signal", "source_name": "signal", "role": "numeric", "dtype": "float64"},
           {"name": "mode", "source_name": "mode", "role": "categorical", "dtype": "string"},
           {"name": "operator", "source_name": "operator", "role": "context", "dtype": "string"}]


def batches(values, *, chunk_size=13):
    return pa.Table.from_pylist(values).to_batches(max_chunksize=chunk_size)


def records(size=100):
    return [{"row_id": i + 1, "timestamp_ns": i * 1_000_000_000,
             "timestamp": f"{i}", "entity_id": "", "signal": float(i % 10),
             "mode": "_missing" if i % 2 else "_unknown", "operator": "context"} for i in range(size)]


def test_window_reference_removes_level_offsets_without_mutating_values():
    values = np.array([[10, 0, 1, 0], [11, 0, 0, 1], [20, 1, 1, 0]], dtype="float32")
    original = values.copy()
    first = WindowDataset(values, np.array([0]), 3, numeric_offsets=[0])[0].numpy()
    shifted = values.copy()
    shifted[:, 0] += 100
    second = WindowDataset(shifted, np.array([0]), 3, numeric_offsets=[0])[0].numpy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], [0, 1, 10])
    np.testing.assert_array_equal(first[:, 1:], original[:, 1:])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("numeric_transform", ["standard", "asinh"])
def test_future_values_labels_and_categories_do_not_change_fit(tmp_path, numeric_transform):
    first = records()
    second = [dict(row) for row in first]
    for row in second[70:]:
        row["signal"], row["mode"] = 100000.0, "future_only"
    index1, _ = build_row_index(batches(first), len(first), tmp_path / "rows1.npy", {})
    index2, _ = build_row_index(batches(second), len(second), tmp_path / "rows2.npy", {})
    fit1 = fit_transform(batches(first), index1, COLUMNS, numeric_transform=numeric_transform)
    fit2 = fit_transform(batches(second, chunk_size=3), index2, COLUMNS, numeric_transform=numeric_transform)
    assert fit1 == fit2
    assert [feature["source_name"] for feature in fit1["features"]] == ["signal", "mode"]
    assert "future_only" not in fit1["features"][1]["vocabulary"]
    a = transform_to_memmap(batches(first), index1, fit1, tmp_path / "a.npy", COLUMNS)
    b = transform_to_memmap(batches(second), index2, fit2, tmp_path / "b.npy", COLUMNS)
    np.testing.assert_array_equal(a[:70], b[:70])
    assert b[70, fit2["features"][1]["offset"]] == 1  # unknown category slot


def test_asinh_preserves_extremes_missingness_and_original_units(tmp_path):
    from src.modeling.pipeline import _contributions

    values = records()
    for i, row in enumerate(values[:70]):
        row["signal"] = 2.0 + (i % 2) * 0.00001
    values[85]["signal"], values[86]["signal"] = 1.0, 3.0
    values[87]["signal"] = None
    index, _ = build_row_index(batches(values), 100, tmp_path / "rows.npy", {})
    state = fit_transform(batches(values), index, COLUMNS, numeric_transform="asinh")
    matrix = transform_to_memmap(batches(values), index, state, tmp_path / "values.npy", COLUMNS)
    assert -15 < matrix[85, 0] < -10
    assert 10 < matrix[86, 0] < 15
    assert matrix[87, 0] == 0 and matrix[87, 1] == 1
    errors = np.zeros(matrix.shape[1])
    errors[0] = 1
    for position in (85, 86, 87):
        contribution = _contributions(state, errors, matrix[position])[0]
        if position == 87:
            assert contribution["value"] is None and contribution["missing"]
        else:
            assert contribution["value"] == pytest.approx(values[position]["signal"], rel=1e-5)
    legacy = dict(state)
    legacy.pop("numeric_transform")
    uncompressed = transform_to_memmap(batches(values), index, legacy, tmp_path / "legacy.npy", COLUMNS)
    np.testing.assert_allclose(matrix[:, 0], np.arcsinh(uncompressed[:, 0]), rtol=1e-6)
    np.testing.assert_array_equal(matrix[:, 1:], uncompressed[:, 1:])


def test_causal_fill_respects_gap_partition_and_never_uses_future(tmp_path):
    values = records(20)
    for i in (2, 3, 4, 14):
        values[i]["signal"] = None
    index, _ = build_row_index(batches(values), len(values), tmp_path / "rows.npy", {"max_gap_seconds": 2})
    state = fit_transform(batches(values), index, COLUMNS, gap_ns=2_000_000_000)
    matrix = transform_to_memmap(batches(values), index, state, tmp_path / "values.npy", COLUMNS)
    assert matrix[2, 0] == matrix[1, 0]  # last observed value within age limit
    assert matrix[3, 0] == matrix[1, 0]
    assert matrix[4, 0] == 0  # observation expired: training mean
    assert matrix[14, 0] == 0  # first validation row never fills from training
    assert np.all(matrix[[2, 3, 4, 14], 1] == 1)


@pytest.mark.parametrize("encoding", ["standard", "asinh"])
def test_numeric_batches_match_scalar_path_and_reject_nonfinite(tmp_path, encoding):
    values = records()
    for i in (2, 14, 70, 87):
        values[i]["signal"] = None
    values[90]["signal"] = 1000000.0
    columns = COLUMNS[:1]
    rows, _ = build_row_index(batches(values), len(values), tmp_path / "rows.npy", {})
    state = fit_transform(batches(values), rows, columns, numeric_transform=encoding)
    fast = transform_to_memmap(batches(values), rows, state, tmp_path / "fast.npy", columns)
    # One nanosecond enables the scalar path but expires before the next row.
    scalar = transform_to_memmap(batches(values, chunk_size=3), rows, dict(state, max_gap_ns=1),
                                 tmp_path / "scalar.npy", columns)
    np.testing.assert_allclose(fast, scalar, rtol=1e-6, atol=1e-7)
    for invalid in (float("nan"), float("inf")):
        values[90]["signal"] = invalid
        with pytest.raises(ValueError, match="Nonfinite"):
            transform_to_memmap(batches(values), rows, state, tmp_path / "invalid.npy", columns)


def test_windows_never_cross_entities_gaps_partitions_or_exclusions(tmp_path):
    values = records(100)
    for i, value in enumerate(values):
        value["entity_id"] = "a" if i < 50 else "b"
        value["timestamp_ns"] = (i % 50 + (100 if i % 50 >= 10 else 0)) * 1_000_000_000
    index, summary = build_row_index(batches(values), 100, tmp_path / "rows.npy", {"max_gap_seconds": 3})
    assert summary["partition_counts"] == {"train": 70, "validation": 14, "test": 16}
    index["excluded"][5] = True
    starts = build_window_index(index, tmp_path / "starts.npy", window_size=4)
    for start in starts:
        window = index[start:start + 4]
        assert len(set(window["segment"])) == 1
        assert not window["excluded"].any()
        assert len(set(window["entity"])) == 1
        assert len(set(window["partition"])) == 1
        assert max(np.diff(window["timestamp_ns"])) <= 3_000_000_000


def test_contract_maps_reordered_columns_and_rejects_missing_or_wrong_roles(tmp_path):
    values = records()
    index, _ = build_row_index(batches(values), 100, tmp_path / "rows.npy", {})
    state = fit_transform(batches(values), index, COLUMNS)
    target = [dict(COLUMNS[1], name="new_mode"), dict(COLUMNS[0], name="new_signal"),
              {"name": "extra", "source_name": "extra", "role": "numeric"}]
    assert resolve_features(state, target) == {"signal": "new_signal", "mode": "new_mode"}
    with pytest.raises(ValueError, match="missing"):
        resolve_features(state, target[1:])
    target[1]["role"] = "label"
    with pytest.raises(ValueError, match="Incompatible"):
        resolve_features(state, target)


def test_all_missing_training_feature_rejected_despite_future_values(tmp_path):
    values = records()
    for row in values[:70]:
        row["signal"] = None
    index, _ = build_row_index(batches(values), 100, tmp_path / "rows.npy", {})
    with pytest.raises(ValueError, match="no usable training"):
        fit_transform(batches(values), index, COLUMNS)


def test_operational_eligibility_ignores_validation_and_future_observations(tmp_path):
    values = records()
    for row in values[:70]:
        row["signal"] = None
    index, _ = build_row_index(batches(values), 100, tmp_path / "rows.npy", {})
    state = fit_transform(batches(values), index, COLUMNS, feature_eligibility="observed_training")
    assert [f["source_name"] for f in state["features"]] == ["mode"]
    assert state["excluded_features"] == [{"source_name": "signal", "reason": "no_usable_training_values"}]
    with pytest.raises(ValueError, match="No features"):
        fit_transform(batches(values), index, COLUMNS[:1], feature_eligibility="observed_training")


def test_category_sentinels_never_collide_with_literal_values(tmp_path):
    values = records()
    values[10]["mode"] = None
    index, _ = build_row_index(batches(values), 100, tmp_path / "rows.npy", {})
    state = fit_transform(batches(values), index, COLUMNS)
    matrix = transform_to_memmap(batches(values), index, state, tmp_path / "values.npy", COLUMNS)
    category_offset = state["features"][1]["offset"]
    assert matrix[10, category_offset + 1] == 1
    assert matrix[0, category_offset + 1] == 0
    assert len(state["output_names"]) == len(set(state["output_names"]))


def test_categorical_fill_retains_raw_missingness_indicator(tmp_path):
    values = records()
    values[1]["mode"] = None
    index, _ = build_row_index(batches(values), 100, tmp_path / "rows.npy", {})
    state = fit_transform(batches(values), index, COLUMNS, gap_ns=5_000_000_000)
    matrix = transform_to_memmap(batches(values), index, state, tmp_path / "values.npy", COLUMNS)
    feature = state["features"][1]
    begin, end = feature["offset"], feature["offset"] + feature["width"]
    np.testing.assert_array_equal(matrix[0, begin:end - 1], matrix[1, begin:end - 1])
    assert matrix[0, end - 1] == 0
    assert matrix[1, end - 1] == 1


def test_prescribed_esa_test_boundary_is_retained(tmp_path):
    values = records(100)
    index, summary = build_row_index(batches(values), 100, tmp_path / "rows.npy",
                                    {"splits": {"test_start": "1970-01-01T00:01:19Z", "train_fraction": 0.85}})
    assert summary["partition_counts"] == {"train": 68, "validation": 12, "test": 20}
    assert index[79]["partition"] == 1
    assert index[80]["partition"] == 2


@pytest.mark.parametrize("fractions,expected", [
    ([.7, .15, .15], {"train": 70, "validation": 15, "test": 15}),
    ([.29, .28, .43], {"train": 29, "validation": 28, "test": 43}),
])
def test_fraction_boundaries_use_declared_decimal_values(tmp_path, fractions, expected):
    values = records(100)
    _, summary = build_row_index(batches(values), 100, tmp_path / "rows.npy",
                                 {"splits": {"fractions": fractions}})
    assert summary["partition_counts"] == expected


def test_declared_unit_conflict_rejected(tmp_path):
    values = records()
    columns = [dict(COLUMNS[0], unit="V"), *COLUMNS[1:]]
    index, _ = build_row_index(batches(values), 100, tmp_path / "rows.npy", {})
    state = fit_transform(batches(values), index, columns)
    target_columns = [dict(columns[0], unit="mV"), *columns[1:]]
    with pytest.raises(ValueError, match="declared units"):
        resolve_features(state, target_columns)
    with pytest.raises(ValueError, match="declared units"):
        resolve_features(state, [dict(COLUMNS[0]), *columns[1:]])


def test_regular_grid_structural_gaps_break_windows_and_filling(tmp_path):
    values = records(30)
    for index, row in enumerate(values):
        row["mapped_gap"] = "1" if 5 <= index <= 7 else "0"
    values[8]["signal"] = None
    columns = [*COLUMNS, {"source_name": "ESA gap", "name": "mapped_gap", "role": "context"}]
    index, _ = build_row_index(batches(values), 30, tmp_path / "rows.npy",
                               {"gap_column": "ESA gap", "max_gap_seconds": 20}, columns=columns)
    state = fit_transform(batches(values), index, columns, gap_ns=20_000_000_000)
    matrix = transform_to_memmap(batches(values), index, state, tmp_path / "values.npy", columns)
    assert matrix[8, 0] == 0  # no carry across flagged outage
    starts = build_window_index(index, tmp_path / "starts.npy", window_size=4)
    for start in starts:
        assert not index[start:start + 4]["gap"].any()
        assert len(set(index[start:start + 4]["segment"])) == 1
    assert not set(range(2, 8)).intersection(starts)
    assert all(feature["source_name"] != "ESA gap" for feature in state["features"])
