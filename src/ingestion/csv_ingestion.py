from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

from src.common.paths import DATA_DIR, LOGS_DIR


def ingest_csvs(
    data_dir: str | Path = DATA_DIR,
    *,
    db_path: str | Path | None = None,
    logs_dir: str | Path = LOGS_DIR,
) -> dict[str, int]:
    """Import each CSV through the resumable adapter and report failed files."""
    from src.ingestion.store import ingest_source, inspect_source

    logger = _build_logger(Path(logs_dir))
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    summary = {"ingested": 0, "skipped_duplicates": 0, "failed": 0}
    for csv_path in sorted(data_path.glob("*.csv")):
        try:
            inspected = inspect_source(csv_path)
            manifest = None
            timestamp_candidates = [c for c in inspected["columns"] if c["source_name"].strip().lower() in {"timestamp", "datetime", "time", "date_time"}]
            if not timestamp_candidates and inspected["issues"] == ["Specify exactly one timestamp column in the manifest"]:
                manifest = {"ordering": "row"}
                logger.warning("CSV %s has no unique timestamp. Preserving row order with calendar time unavailable", csv_path.name)
            result = ingest_source(csv_path, manifest, db_path=db_path)
            key = "skipped_duplicates" if result.get("duplicate") else "ingested"
            summary[key] += 1
        except Exception as exc:
            logger.error("Failed to ingest CSV %s: %s", csv_path.name, exc)
            summary["failed"] += 1
    return summary



def _build_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("satellite_operator.ingestion")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = (logs_dir / "ingestion.log").resolve()
    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    return logger


@dataclass(frozen=True)
class CsvFileMetadata:
    file_name: str
    file_path: str
    file_size: int
    modified_time: str
    content_hash: str


@dataclass(frozen=True)
class CsvReadResult:
    headers: list[str]
    sanitized_headers: list[str]
    rows: list[list[str]]


def sanitize_column_names(headers: Iterable[str]) -> list[str]:
    sanitized: list[str] = []
    seen: dict[str, int] = {}
    used: set[str] = set()
    reserved = {"row_id", "dataset_id", "source_file_id"}

    for index, header in enumerate(headers, start=1):
        name = re.sub(r"\W+", "_", header.strip().lower()).strip("_")
        if not name:
            name = f"column_{index}"
        if name[0].isdigit():
            name = f"column_{name}"
        if name in reserved:
            name = f"csv_{name}"

        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"
        while name in used:
            name += "_"
        used.add(name)
        sanitized.append(name)

    return sanitized
