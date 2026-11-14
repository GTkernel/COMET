"""Export immutable runs as complete, timestamp-aligned global predictions."""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path


def export_predictions(run: dict | str, output_path: str | Path, *, db_path=None) -> dict:
    """Write CSV for the run's selected partition, retaining warm-up rows.

    This is global detection only. Feature contributions are not channel alarms.
    Multiple entities require separate evaluation and are rejected explicitly.
    """
    from src.ingestion.store import _parquet_batches
    from src.common.score_artifact import validate_score_counts
    if not isinstance(run, dict):
        from src.db import get_detection_run
        run = get_detection_run(run, db_path=db_path)
        if run is None:
            raise ValueError("Detection run does not exist")
    metadata = run.get("metadata_json") or run.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    run = {**metadata, **run}
    source = Path(run["scores_path"])
    if run.get("status", "completed") not in {"completed", "complete"}:
        raise ValueError("Only completed detection runs can be exported")
    destination = Path(output_path)
    if destination.suffix.lower() != ".csv":
        raise ValueError("Prediction export currently requires a .csv output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() == source.resolve():
        raise ValueError("Export must not replace its source score artifact")
    temporary = destination.with_suffix(".csv.partial")
    entity = None
    previous = None
    rows = alerts = warmup = 0
    source_rows = scored_rows = 0
    coverage_counts = Counter()
    partition = run.get("partition", "all")
    fields = ["timestamp", "score", "is_anomaly", "coverage", "row_id", "entity_id", "partition"]
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for batch in _parquet_batches(source, batch_size=10000):
                for row in batch.to_pylist():
                    source_rows += 1
                    coverage = row.get("coverage", "scored")
                    coverage_counts[coverage] += 1
                    covered = coverage == "scored"
                    score = row.get("score")
                    if covered and (score is None or not math.isfinite(score)):
                        raise ValueError("Scored rows must contain finite reconstruction scores")
                    scored_rows += covered
                    if partition not in (None, "all") and row.get("partition") != partition:
                        continue
                    row_entity = row.get("entity_id", "")
                    if entity is None:
                        entity = row_entity
                    if entity != row_entity:
                        raise ValueError("ESA global export requires one entity. Evaluate each entity separately")
                    time = row["timestamp_ns"]
                    if previous is not None and time <= previous:
                        raise ValueError("Score timestamps must be unique and strictly increasing")
                    previous = time
                    alert = int(covered and bool(row["is_anomaly"]))
                    writer.writerow({"timestamp": row["timestamp"], "score": score if covered else "", "is_anomaly": alert, "coverage": coverage, "row_id": row["row_id"], "entity_id": row_entity, "partition": row.get("partition", "")})
                    rows += 1
                    alerts += alert
                    warmup += not covered
        validate_score_counts(run, source_rows, scored_rows, coverage_counts)
        if not rows:
            raise ValueError("The selected detection partition has no prediction rows")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    summary = {"path": str(destination.resolve()), "run_id": run.get("run_id", run.get("id")), "dataset_id": run.get("dataset_id"), "model_id": run.get("model_id"), "partition": partition, "rows": rows, "alert_rows": alerts, "unscored_rows": warmup, "scope": "global", "warmup_policy": "explicit no-alert", "timestamp_policy": "window-end"}
    destination.with_suffix(".csv.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
