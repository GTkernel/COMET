"""Validate a full ESA import and run bounded model checks.

Run from the checkout: python scripts/validate_esa.py --archive PATH --mission 1
    --output-root PATH --trust-official-pickle
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf-8")


def disk_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def source_fingerprint() -> dict:
    """Identify the tested local Python sources without requiring a commit."""
    digest = hashlib.sha256()
    paths = sorted([*REPO.joinpath("src").rglob("*.py"), Path(__file__).resolve()])
    for path in paths:
        digest.update(path.relative_to(REPO).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"sha256": digest.hexdigest(), "python_files": len(paths)}


def imported_provenance(root: Path) -> dict:
    imported = json.loads((root / "import.json").read_text(encoding="utf-8"))
    esa = imported["manifest"]["esa"]
    checksum_bytes = json.dumps(esa["source_checksums"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    telemetry = Path(imported["telemetry_path"])
    parts = list(telemetry.glob("*.parquet")) if telemetry.is_dir() else [telemetry]
    return {"dataset_id": imported["dataset_id"], "row_count": imported["row_count"],
            "feature_count": sum(column["role"] in {"numeric", "categorical"} for column in imported["columns"]),
            "import_fingerprint": imported["import_fingerprint"],
            "duplicate": bool(imported.get("duplicate", False)),
            "upstream_revision": esa["upstream_revision"], "esa_adapter_version": esa["adapter_version"],
            "preprocessing": esa["configuration"], "source_file_count": len(esa["source_checksums"]),
            "source_checksums_sha256": hashlib.sha256(checksum_bytes).hexdigest(),
            "telemetry_parts": len(parts), "telemetry_bytes": sum(path.stat().st_size for path in parts)}


def worker(args):
    from src.ingestion.store import ingest_source, iter_batches, load_dataset
    root = Path(args.output_root).resolve()
    db = root / "db" / "satellite_operator.db"
    result_path = root / f"{args.worker}.json"
    if args.worker == "import":
        boundary = "2007-01-01T00:00:00Z" if args.mission == 1 else "2001-10-01T00:00:00Z"
        manifest = {"version": 1, "esa": {"mission": args.mission, "version": "2", "profile": "causal", "trusted_pickle": True},
                    "splits": {"test_start": boundary, "train_fraction": .85}}
        imported = ingest_source(args.archive, manifest, db_path=db, storage_dir=root, batch_size=10000)
        write_json(result_path, imported)
        return

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from src.modeling.pipeline import train, detect, diagnostics
    from src.evaluation.export import export_predictions

    torch.set_num_threads(1)
    imported = json.loads((root / "import.json").read_text(encoding="utf-8"))
    dataset = load_dataset(imported["dataset_id"], db_path=db)
    boundary = pd.Timestamp("2007-01-01T00:00:00Z" if args.mission == 1 else "2001-10-01T00:00:00Z").value
    # Keep the last 4096 pre-boundary rows and first 2048 test rows in bounded RAM.
    # This subset checks model execution. Accuracy needs a separate evaluation.
    before, after = deque(maxlen=4096), []
    candidates = [c for c in dataset["columns"] if c["role"] == "numeric"]
    for batch in iter_batches(dataset["dataset_id"], db_path=db, batch_size=4096):
        for row in batch.to_pylist():
            if row["timestamp_ns"] < boundary:
                before.append(row)
            elif len(after) < 2048:
                after.append(row)
        if len(after) == 2048:
            break
    before = list(before)
    if len(before) < 128 or len(after) < 128:
        raise ValueError("Official boundary does not have enough observations for the bounded check")
    features = [c for c in candidates if any(row.get(c["name"]) is not None for row in before[:int(len(before)*.7)])][:8]
    if not features:
        raise ValueError("No numeric features have usable training values in this bounded check")
    paths = []
    manifest = {"version": 1, "timestamp": "timestamp", "numeric_features": [c["source_name"] for c in features],
                "cadence_seconds": dataset["manifest"]["cadence_seconds"],
                "max_gap_seconds": dataset["manifest"]["max_gap_seconds"],
                "preprocessing": {"forward_fill": False}, "splits": {"fractions": [.7,.15,.15]},
                "provenance": {"parent_dataset_id": dataset["dataset_id"], "purpose": "bounded smoke test; not benchmark"}}
    gap_source = dataset["manifest"].get("gap_column")
    gap_mapping = next((column for column in dataset["columns"] if column["source_name"] == gap_source), None)
    if gap_source:
        if gap_mapping is None or gap_mapping["role"] != "context":
            raise ValueError("Parent dataset is missing its structural gap contract")
        manifest.update(context_columns=[gap_source], gap_column=gap_source)
    for name, rows in (("bounded-training", before), ("bounded-target", after)):
        path = root / f"{name}.parquet"
        columns = {"timestamp": [row["timestamp"] for row in rows]}
        columns.update({c["source_name"]: pa.array([row.get(c["name"]) for row in rows], type=pa.float64()) for c in features})
        if gap_mapping:
            columns[gap_source] = pa.array([row[gap_mapping["name"]] for row in rows], type=pa.string())
        pq.write_table(pa.table(columns), path)
        current_manifest = dict(manifest)
        if name == "bounded-target":
            current_manifest.pop("splits", None)
        paths.append(ingest_source(path, current_manifest, db_path=db, storage_dir=root))
    config = root / "bounded-config.json"
    write_json(config, {"window_size": 16, "hidden_size": 8, "epochs": 1, "batch_size": 64, "seed": 0})
    model = train(paths[0]["dataset_id"], config, db_path=db, storage_dir=root)
    test = detect(paths[0]["dataset_id"], model["model_id"], db_path=db, storage_dir=root, partition="test")
    target = detect(paths[1]["dataset_id"], model["model_id"], db_path=db, storage_dir=root, batch_size=64)
    other = detect(paths[1]["dataset_id"], model["model_id"], db_path=db, storage_dir=root, batch_size=17)
    scores = [pq.read_table(run["scores_path"], columns=["score"]).column(0).to_numpy() for run in (target, other)]
    np.testing.assert_allclose(scores[0], scores[1], rtol=1e-5, atol=1e-6, equal_nan=True)
    exported = export_predictions(target["run_id"], root / "bounded-predictions.csv", db_path=db)
    write_json(result_path, {"parent_dataset_id": dataset["dataset_id"], "features": features,
                            "training_rows": len(before), "target_rows": len(after), "model": model,
                            "held_out_test": diagnostics(test["run_id"], db_path=db),
                            "independent_target": diagnostics(target["run_id"], db_path=db),
                            "inference_batch_equivalence": True, "export": exported,
                            "claim": "Checks full ingestion and bounded model execution. Benchmark reproduction requires separate evaluation."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--mission", type=int, choices=(1,2), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trust-official-pickle", action="store_true")
    parser.add_argument("--worker", choices=("import", "bounded"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.trust_official_pickle:
        parser.error("Raw decoding requires --trust-official-pickle after verifying the official archive")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["COMET_STORAGE_DIR"] = str(root)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if args.worker:
        worker(args)
        return
    import psutil

    archive = Path(args.archive).resolve()
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    report = {"archive": str(archive), "archive_bytes": archive.stat().st_size, "sha256": digest.hexdigest(),
              "mission": args.mission, "dataset_version": "2", "python": sys.version,
              "platform": platform.platform(), "stages": {},
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "local_source": source_fingerprint(),
              "initial_disk_bytes": disk_bytes(root),
              "initial_prepared_cache_markers": len(list(root.glob("esa-prepared/*/complete.json"))),
              "measurement": "Peak process-tree RSS sampled every 0.25 seconds. Disk includes retained caches and earlier attempts.",
              "claim": "Checks complete ingestion and bounded model execution. Benchmark reproduction and accuracy need separate evaluation.",
              "versions": {name: importlib.metadata.version(name) for name in ("numpy","pandas","pyarrow","duckdb","torch","psutil")}}
    for stage in ("import", "bounded"):
        command = [sys.executable, str(Path(__file__).resolve()), "--archive", str(archive),
                   "--mission", str(args.mission), "--output-root", str(root), "--trust-official-pickle", "--worker", stage]
        started, peak = time.monotonic(), 0
        with (root / f"{stage}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            measured = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    rss = sum(p.memory_info().rss for p in [measured, *measured.children(recursive=True)] if p.is_running())
                    peak = max(peak, rss)
                except psutil.Error:
                    pass
                time.sleep(.25)
        report["stages"][stage] = {"seconds": time.monotonic()-started, "peak_tree_rss_bytes_sampled": peak,
                                   "exit_code": process.returncode, "log": str(root / f"{stage}.log")}
        report["disk_bytes"] = disk_bytes(root)
        report["stages"][stage]["retained_disk_bytes"] = report["disk_bytes"]
        if stage == "import" and process.returncode == 0:
            report["imported_dataset"] = imported_provenance(root)
        write_json(root / "validation-report.json", report)
        print(json.dumps(report["stages"][stage]), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
