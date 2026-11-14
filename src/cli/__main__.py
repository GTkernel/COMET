"""Run COMET commands and load optional dependencies as each command needs them."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comet")
    parser.add_argument("--storage-dir", help="Runtime root (or COMET_STORAGE_DIR).")
    parser.add_argument("--scratch-dir", help="Temporary directory (or COMET_SCRATCH_DIR).")
    parser.add_argument("--db", help="SQLite catalog path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for legacy commands.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "ingest", "run"):
        command = commands.add_parser(name, help=f"{name.capitalize()} structured telemetry.")
        command.add_argument("input", **({"nargs": "?"} if name == "ingest" else {}))
        command.add_argument("--manifest", help="JSON/YAML manifest describing the source columns and import settings.")
        command.add_argument("--batch-size", type=int, default=10000)
        if name == "inspect":
            command.add_argument("--output-manifest", help="Save the suggested mapping.")
        if name == "run":
            command.add_argument("--config", help="Model training YAML.")
    for name in ("profile", "prepare", "train", "detect", "model-diagnostics", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--dataset-id", type=int, required=True)
        if name == "prepare":
            command.add_argument("--feature-config", help="Editable feature-role YAML for SQLite or Parquet datasets.")
        if name == "train":
            command.add_argument("--config")
        if name in ("detect", "model-diagnostics", "evaluate"):
            command.add_argument("--model-id", type=int, required=True)
        if name == "detect":
            command.add_argument("--partition", choices=("all", "train", "validation", "test"), default="all")
            command.add_argument("--batch-size", type=int)
        if name in ("model-diagnostics", "evaluate"):
            command.add_argument("--run-id", help="Select a completed run. Defaults to the latest for this dataset and model.")
            command.add_argument("--output-dir")
        if name == "evaluate":
            command.add_argument("--metric", choices=("legacy-overlap", "esa-reference"), default="legacy-overlap")
            command.add_argument("--metadata", required=True, help="Ground-truth interval CSV for legacy-overlap, CSV/Parquet for ESA reference.")
            command.add_argument("--esa-checkout")
            command.add_argument("--reference-python")
    command = commands.add_parser("monitor", help="Update from recent history and score later recorded observations.")
    command.add_argument("--dataset-id", type=int, required=True)
    command.add_argument("--history-dataset-id", type=int)
    command.add_argument("--start", required=True, help="Start scoring at this UTC ISO timestamp.")
    command.add_argument("--end", required=True, help="Stop before this UTC ISO timestamp.")
    command.add_argument("--history-days", type=float, default=90)
    command.add_argument("--validation-days", type=float, default=14)
    command.add_argument("--update-days", type=float, default=7)
    command.add_argument("--entity-id")
    command.add_argument("--config")
    command.add_argument("--output", help="Combined operational prediction CSV.")
    command = commands.add_parser("export-predictions")
    command.add_argument("--run-id", required=True)
    command.add_argument("--output", required=True)
    command = commands.add_parser("list-anomalies")
    command.add_argument("--dataset-id", type=int)
    command.add_argument("--model-id", type=int)
    command.add_argument("--run-id")
    command.add_argument("--limit", type=int, default=20)
    for name in ("anomaly-packet", "diagnose"):
        command = commands.add_parser(name)
        command.add_argument("--anomaly-id", required=True)
        if name == "diagnose":
            command.add_argument("--llm-model", default=None, help="Override the accepted local diagnosis model profile.")
    command = commands.add_parser("ingest-docs")
    command.add_argument("--reindex-semantic", action="store_true", help="Build a separate versioned semantic Chroma index using local embeddinggemma:300m.")
    command.add_argument("--chunk-size", type=int, default=500)
    command.add_argument("--chunk-overlap", type=int, default=50)
    for command in commands.choices.values():
        command.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                             help="Print machine-readable JSON.")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    # Set locations before importing modules that establish default paths.
    if args.storage_dir:
        os.environ["COMET_STORAGE_DIR"] = str(Path(args.storage_dir).resolve())
    if args.scratch_dir:
        os.environ["COMET_SCRATCH_DIR"] = str(Path(args.scratch_dir).resolve())
    from src.common.logging import get_app_logger

    logger = get_app_logger()
    try:
        result = _dispatch(args)
        _print_result(args, result)
        if isinstance(result, dict) and result.get("failed", 0):
            parser.exit(1, "error: one or more imports failed. Successful imports remain available.\n")
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        logger.exception("command=%s status=failed", args.command)
        parser.exit(1, f"error: {exc}\n")


def _print_result(args, result) -> None:
    """Print the original command summaries unless JSON output was requested."""
    command = args.command
    if args.json or command in {"inspect", "run", "monitor", "export-predictions", "anomaly-packet"} or (
        command == "ingest" and args.input is not None
    ) or (command == "evaluate" and args.metric != "legacy-overlap"):
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str, allow_nan=False))
        return
    if command == "list-anomalies":
        if not result:
            print("No anomalies found. Run detect first.")
        else:
            print("anomaly_id\tdataset_id\tmodel_id\twindow_index\tscore\tstart_timestamp\tend_timestamp")
            for row in result:
                print(f"{row['anomaly_id']}\t{row['dataset_id']}\t{row['model_id']}\t{row['window_index']}\t"
                      f"{row['anomaly_score']:.6g}\t{row['start_timestamp'] or ''}\t{row['end_timestamp'] or ''}")
        return
    if command == "ingest":
        print(f"CSV ingestion complete: {result['ingested']} ingested, "
              f"{result['skipped_duplicates']} duplicates skipped, {result['failed']} failed.")
        return
    if command == "model-diagnostics":
        stats = result.get("reconstruction_error_statistics", {})
        features = ", ".join(f"{item['feature']}={item['average_reconstruction_error']:.6g}"
                             for item in result.get("top_contributing_features", [])[:5]) or "none"
        values = {"dataset_id": result["dataset_id"], "model_id": result["model_id"],
                  "windows": result.get("windows_evaluated", result.get("score_count")),
                  "above_threshold": result.get("windows_above_threshold", result.get("anomalies_detected")),
                  "percent_above": result.get("percentage_windows_above_threshold"),
                  "threshold": result["threshold"], "threshold_percentile": result.get("threshold_percentile"),
                  "min": stats.get("min", result.get("score_min")), "max": stats.get("max", result.get("score_max")),
                  "mean": stats.get("mean", result.get("score_mean")), "median": stats.get("median"),
                  "std": stats.get("standard_deviation"), "top_features": features,
                  "summary": result.get("summary_path", result.get("output_path"))}
        label = "Model diagnostics"
    else:
        fields = {
            "profile": ("Dataset profiling", [("dataset_id", "dataset_id"), ("columns", "column_count"), ("config", "config_path")]),
            "prepare": ("Dataset preparation", [("dataset_id", "dataset_id"), ("model_table", "model_table"), ("context_table", "context_table"), ("rows", "row_count"), ("config", "config_path")]),
            "train": ("Model training", [("model_id", "model_id"), ("dataset_id", "dataset_id"), ("threshold", "threshold"), ("path", "model_path"), ("config", "config_path")]),
            "detect": ("Anomaly detection", [("dataset_id", "dataset_id"), ("model_id", "model_id"), ("windows", "windows_scored"), ("anomalies", "anomalies_detected")]),
            "evaluate": ("Evaluation", [("dataset_id", "dataset_id"), ("model_id", "model_id"), ("ground_truth", "ground_truth_intervals"), ("detections", "detected_anomalies"), ("precision", "precision"), ("recall", "recall"), ("f1", "f1"), ("mean_delay", "mean_detection_delay_seconds"), ("median_delay", "median_detection_delay_seconds"), ("summary", "summary_path")]),
            "ingest-docs": ("Document ingestion", [("found", "documents_found"), ("ingested", "documents_ingested"), ("skipped", "documents_skipped"), ("chunks", "chunks_ingested"), ("vector_store", "vector_store")]),
            "diagnose": ("Diagnosis", [("diagnosis_id", "diagnosis_id"), ("anomaly_id", "anomaly_id"), ("llm_status", "llm_status")]),
        }
        label, mapping = fields[command]
        values = {key: result.get(source) for key, source in mapping}
        if command == "profile" and values["columns"] is None:
            values["columns"] = len(result.get("columns", {}))
        if command == "detect" and values["windows"] is None:
            values["windows"] = result.get("score_count")
        if command == "prepare" and values["config"] is None:
            from src.cleaning.prepare_dataset import resolve_feature_config_path
            values["config"] = resolve_feature_config_path(args.dataset_id, args.feature_config)
        if command == "train" and values["config"] is None:
            from src.modeling.train_model import resolve_training_config_path
            values["config"] = resolve_training_config_path(args.config)
        if command == "diagnose":
            values = {"diagnosis_id": result["diagnosis_id"], "anomaly_id": result["anomaly_id"],
                      "chunks": len(result["retrieved_chunks"]), "llm_status": result["llm_status"]}
    def display(key, value):
        if value is None:
            return "undefined" if key in {"precision", "recall", "f1"} else "None"
        if command == "train" and key == "threshold":
            return f"{value:.6f}"
        if command == "evaluate" and key in {"precision", "recall", "f1"}:
            return f"{value:.4f}"
        if command == "model-diagnostics":
            if key == "percent_above":
                return f"{value:.2f}%"
            if key in {"threshold", "min", "max", "mean", "median", "std"}:
                return f"{value:.6g}"
        return str(value)
    print(f"{label} complete: " + " ".join(f"{key}={display(key, value)}" for key, value in values.items()))


def _selected_run(args) -> dict[str, Any] | None:
    from src.db import get_detection_run

    run = get_detection_run(args.run_id, dataset_id=args.dataset_id,
                            model_id=args.model_id, db_path=args.db)
    if args.run_id and run is None:
        raise ValueError("No completed run matches this run ID, target dataset and model.")
    return run


def _dispatch(args) -> Any:
    from src.db import init_db

    init_db(args.db)
    command = args.command
    if command == "inspect":
        from src.ingestion.store import inspect_source

        result = inspect_source(args.input, args.manifest)
        if args.output_manifest:
            manifest = result.get("manifest", result.get("suggested_manifest"))
            if manifest is None:
                raise ValueError("Inspection could not suggest a manifest. Review the reported schema assumptions.")
            reusable = {key: value for key, value in manifest.items() if not key.startswith("_")}
            Path(args.output_manifest).write_text(json.dumps(reusable, indent=2), encoding="utf-8")
        return result
    if command in ("ingest", "run"):
        from src.ingestion.store import ingest_source

        if args.input is None:
            from src.ingestion.csv_ingestion import ingest_csvs
            return ingest_csvs(db_path=args.db)
        imported = ingest_source(args.input, args.manifest, db_path=args.db,
                                 storage_dir=args.storage_dir, batch_size=args.batch_size)
        if command == "ingest":
            return imported
        from src.modeling.pipeline import train, detect, diagnostics

        trained = train(imported["dataset_id"], args.config, db_path=args.db, storage_dir=args.storage_dir)
        detected = detect(imported["dataset_id"], trained["model_id"], db_path=args.db,
                          storage_dir=args.storage_dir, partition="test")
        return {"ingestion": imported, "training": trained, "detection": detected,
                "diagnostics": diagnostics(detected["run_id"], db_path=args.db)}
    if command == "monitor":
        from src.modeling.monitoring import monitor
        return monitor(args.dataset_id, args.start, args.end, history_dataset_id=args.history_dataset_id,
                       history_days=args.history_days, validation_days=args.validation_days,
                       update_days=args.update_days, config_path=args.config, entity_id=args.entity_id,
                       db_path=args.db, storage_dir=args.storage_dir, output_path=args.output)
    if command == "profile":
        from src.profiling.dataset_profile import profile_dataset
        return profile_dataset(args.dataset_id, db_path=args.db)
    if command == "prepare":
        from src.cleaning.prepare_dataset import prepare_dataset
        return prepare_dataset(args.dataset_id, args.feature_config, db_path=args.db)
    if command == "train":
        from src.modeling.train_model import train_model
        return train_model(args.dataset_id, args.config, db_path=args.db)
    if command == "detect":
        from src.anomaly.detect import detect_anomalies
        return detect_anomalies(args.dataset_id, args.model_id, db_path=args.db,
                                partition=args.partition, batch_size=args.batch_size)
    if command == "list-anomalies":
        return list_anomalies(dataset_id=args.dataset_id, model_id=args.model_id,
                              run_id=args.run_id, limit=args.limit, db_path=args.db)
    if command == "model-diagnostics":
        run = _selected_run(args)
        if run:
            from src.modeling.pipeline import diagnostics
            from src.common.paths import DIAGNOSTICS_DIR
            return diagnostics(run["run_id"], db_path=args.db, output_dir=args.output_dir or DIAGNOSTICS_DIR)
        from src.modeling.diagnostics import run_model_diagnostics
        return run_model_diagnostics(args.dataset_id, args.model_id, db_path=args.db)
    if command == "export-predictions":
        from src.evaluation.export import export_predictions
        return export_predictions(args.run_id, Path(args.output), db_path=args.db)
    if command == "evaluate":
        run = _selected_run(args)
        if args.metric == "legacy-overlap":
            from src.evaluation.anomaly_evaluation import evaluate_anomalies
            kwargs = {"output_dir": args.output_dir} if args.output_dir else {}
            return evaluate_anomalies(args.dataset_id, args.model_id, args.metadata,
                                      run_id=run["run_id"] if run else None, db_path=args.db, **kwargs)
        if not run or not args.esa_checkout or not args.reference_python:
            raise ValueError("ESA reference evaluation requires a completed run, --esa-checkout and --reference-python.")
        from src.common.paths import EVALUATION_DIR
        from src.evaluation.export import export_predictions
        from src.evaluation.reference import run_reference_evaluation

        output = Path(args.output_dir or EVALUATION_DIR) / run["run_id"]
        output.mkdir(parents=True, exist_ok=True)
        predictions = output / "predictions.csv"
        export_predictions(run["run_id"], predictions, db_path=args.db)
        return run_reference_evaluation(predictions, Path(args.metadata), esa_root=Path(args.esa_checkout),
                                        python_executable=Path(args.reference_python), output_path=output / "esa.json")
    if command == "anomaly-packet":
        from src.anomaly.packet import build_anomaly_packet
        return build_anomaly_packet(args.anomaly_id, db_path=args.db)
    if command == "ingest-docs":
        from src.rag.document_ingestion import ingest_documents
        return ingest_documents(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, db_path=args.db,
                                semantic=args.reindex_semantic)
    if command == "diagnose":
        from src.rag.diagnosis import diagnose_anomaly
        return diagnose_anomaly(args.anomaly_id, llm_model=args.llm_model, db_path=args.db)
    raise ValueError(f"Unknown command: {command}")


def list_anomalies(*, dataset_id=None, model_id=None, run_id=None, limit=20, db_path=None):
    from src.db import get_connection, get_detection_run, init_db

    if limit <= 0:
        raise ValueError("limit must be positive.")
    init_db(db_path)
    if run_id is None and dataset_id is not None and model_id is not None:
        latest = get_detection_run(dataset_id=dataset_id, model_id=model_id, db_path=db_path)
        if latest is not None:
            run_id = latest["run_id"]
    if run_id and not get_detection_run(run_id, dataset_id=dataset_id, model_id=model_id, db_path=db_path):
        raise ValueError("No matching completed detection run.")
    clauses, params = [], []
    for key, value in (("dataset_id", dataset_id), ("model_id", model_id), ("run_id", run_id)):
        if value is not None:
            clauses.append(f"{key} = ?")
            params.append(value)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with get_connection(db_path) as conn:
        return [dict(row) for row in conn.execute(
            "SELECT anomaly_id,dataset_id,model_id,run_id,window_index,anomaly_score,"
            f"start_timestamp,end_timestamp,score_timestamp FROM anomalies {where} "
            "ORDER BY anomaly_score DESC LIMIT ?", [*params, limit])]


if __name__ == "__main__":
    main()
