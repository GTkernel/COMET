from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.telemetry.types import TelemetryRecord, TelemetrySource


TIMESTAMP_COLUMNS = {"timestamp", "time", "datetime", "date_time"}
SPACECRAFT_ID_COLUMNS = {"spacecraft_id", "satellite_id", "sat_id", "vehicle_id"}


def csv_row_to_telemetry_record(
    *,
    headers: Sequence[str],
    sanitized_headers: Sequence[str],
    row: Sequence[str],
    source_file: str | Path,
    row_number: int,
    dataset_id: int | None = None,
    source_file_id: int | None = None,
    received_at: str | None = None,
) -> TelemetryRecord:
    """Map one CSV row into the canonical telemetry record shape."""
    if len(headers) != len(sanitized_headers) or len(row) != len(sanitized_headers):
        raise ValueError("CSV row, headers, and sanitized headers must have matching lengths.")

    telemetry = dict(zip(sanitized_headers, row, strict=True))
    source_path = Path(source_file)
    timestamp = _first_matching_value(telemetry, TIMESTAMP_COLUMNS)
    spacecraft_id = _first_matching_value(telemetry, SPACECRAFT_ID_COLUMNS)

    return TelemetryRecord(
        timestamp=timestamp,
        sequence_id=row_number,
        spacecraft_id=spacecraft_id,
        source=TelemetrySource(
            type="csv",
            name=source_path.name,
            received_at=received_at,
            metadata={
                "source_path": str(source_file),
                "row_number": row_number,
                "dataset_id": dataset_id,
                "source_file_id": source_file_id,
                "original_headers": list(headers),
            },
        ),
        telemetry=telemetry,
        metadata={
            "dataset_id": dataset_id,
            "source_file_id": source_file_id,
        },
    )


def csv_rows_to_telemetry_records(
    *,
    headers: Sequence[str],
    sanitized_headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    source_file: str | Path,
    dataset_id: int | None = None,
    source_file_id: int | None = None,
    received_at: str | None = None,
) -> list[TelemetryRecord]:
    return [
        csv_row_to_telemetry_record(
            headers=headers,
            sanitized_headers=sanitized_headers,
            row=row,
            source_file=source_file,
            row_number=index,
            dataset_id=dataset_id,
            source_file_id=source_file_id,
            received_at=received_at,
        )
        for index, row in enumerate(rows, start=1)
    ]


def _first_matching_value(values: dict[str, Any], names: set[str]) -> str | None:
    for name in names:
        value = values.get(name)
        if value not in {None, ""}:
            return str(value)
    return None
