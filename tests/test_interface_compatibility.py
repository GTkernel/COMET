"""Original Python imports and CLI output remain usable alongside new commands."""
import importlib
import json

import pytest

from src.cli.__main__ import _parser, _print_result


@pytest.mark.parametrize("module,names", [
    ("src.cli.prerequisites", ["require_raw_dataset", "require_profile_output", "require_model_ready_dataset",
                               "require_trained_model", "require_documents_directory", "require_diagnosis_inputs"]),
    ("src.common.db", ["connect", "initialize_schema", "upsert_feature_config"]),
    ("src.config.settings", ["load_app_config"]),
    ("src.telemetry", ["TelemetryRecord", "TelemetrySource"]),
    ("src.telemetry.csv_adapter", ["csv_row_to_telemetry_record", "csv_rows_to_telemetry_records"]),
    ("src.db", ["insert_dataset_file", "update_dataset_row_count", "get_duplicate_dataset_file"]),
    ("src.ingestion.csv_ingestion", ["CsvReadResult", "CsvFileMetadata", "sanitize_column_names"]),
])
def test_original_imports(module, names):
    imported = importlib.import_module(module)
    assert all(hasattr(imported, name) for name in names)


def test_original_csv_record_conversion_and_validation():
    from src.telemetry.csv_adapter import csv_row_to_telemetry_record
    record = csv_row_to_telemetry_record(headers=["Time", "Voltage"], sanitized_headers=["time", "voltage"],
                                          row=["2020-01-01", "12"], source_file="mission.csv", row_number=3)
    assert record.timestamp == "2020-01-01"
    assert record.telemetry == {"time": "2020-01-01", "voltage": "12"}
    assert record.to_dict()["source"]["metadata"]["row_number"] == 3
    with pytest.raises(ValueError, match="matching lengths"):
        csv_row_to_telemetry_record(headers=["x"], sanitized_headers=["x"], row=[],
                                    source_file="mission.csv", row_number=1)


def test_original_header_names_remain_unique_for_suffix_collisions():
    from src.ingestion.csv_ingestion import sanitize_column_names
    assert sanitize_column_names(["Voltage", "Voltage", "row_id", "1 watt"]) == [
        "voltage", "voltage_2", "csv_row_id", "column_1_watt"]
    names = sanitize_column_names(["a", "a", "a_2", "a_2_"])
    assert len(set(names)) == 4


def test_original_dataset_file_helpers(tmp_path):
    from src.db import (get_dataset, get_duplicate_dataset_file, init_db, insert_dataset,
                        insert_dataset_file, update_dataset_row_count)
    db = tmp_path / "catalog.db"
    init_db(db)
    dataset_id = insert_dataset("legacy", db_path=db)
    source = tmp_path / "raw.csv"
    source_id = insert_dataset_file(dataset_id, source, file_size=3, modified_time="123",
                                    content_hash="sha", row_count=2, db_path=db)
    update_dataset_row_count(dataset_id, 2, db_path=db)
    assert get_dataset(dataset_id, db_path=db)["row_count"] == 2
    duplicate = get_duplicate_dataset_file(file_name="raw.csv", file_path=str(source), file_size=3,
                                            modified_time="123", content_hash="sha", db_path=db)
    assert duplicate["source_file_id"] == source_id


def test_cli_original_summary_and_explicit_json(capsys):
    result = {"ingested": 2, "skipped_duplicates": 1, "failed": 0}
    args = _parser().parse_args(["ingest"])
    _print_result(args, result)
    assert capsys.readouterr().out == "CSV ingestion complete: 2 ingested, 1 duplicates skipped, 0 failed.\n"
    for argv in (["--json", "ingest"], ["ingest", "--json"]):
        _print_result(_parser().parse_args(argv), result)
        assert json.loads(capsys.readouterr().out) == result


def test_cli_original_tsv(capsys):
    rows = [{"anomaly_id": "a", "dataset_id": 1, "model_id": 2, "window_index": 3,
             "anomaly_score": 1.25, "start_timestamp": None, "end_timestamp": "2020"}]
    _print_result(_parser().parse_args(["list-anomalies"]), rows)
    assert capsys.readouterr().out == (
        "anomaly_id\tdataset_id\tmodel_id\twindow_index\tscore\tstart_timestamp\tend_timestamp\n"
        "a\t1\t2\t3\t1.25\t\t2020\n")


def test_cli_undefined_evaluation_is_printable(capsys):
    args = _parser().parse_args(["evaluate", "--dataset-id", "1", "--model-id", "2", "--metadata", "labels.csv"])
    _print_result(args, {"precision": None, "recall": None, "f1": None})
    assert "precision=undefined recall=undefined f1=undefined" in capsys.readouterr().out


def test_app_config_defaults_are_isolated(tmp_path):
    from src.config.settings import load_app_config
    path = tmp_path / "absent.json"
    first = load_app_config(path)
    first["rag"]["chunk_size"] = 1
    assert load_app_config(path)["rag"]["chunk_size"] == 500
