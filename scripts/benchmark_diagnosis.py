"""Compare local diagnosis models using separate development and acceptance cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CORPUS = REPO / "tests" / "fixtures" / "diagnosis_cases.json"
MODELS = ("gemma:2b", "qwen3.5:2b", "qwen3.5:4b", "gemma4:e2b")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_cases(path=CORPUS) -> tuple[dict, list[dict]]:
    corpus = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = corpus["cases"]
    if len(cases) != 80 or len({case["id"] for case in cases}) != 80:
        raise ValueError("Corpus requires exactly 80 uniquely identified cases")
    families = {}
    for split in ("development", "acceptance"):
        selected = [case for case in cases if case["split"] == split]
        if len(selected) != 40:
            raise ValueError("Each corpus split requires exactly 40 cases")
        families[split] = {case["family"] for case in selected}
    if families["development"] & families["acceptance"]:
        raise ValueError("Related case families cross the locked split")
    return corpus, cases


def percentile(values: list[float], fraction=.95) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def selection_score(bundle: dict, selection: dict | None, case: dict, status: str) -> dict:
    from src.rag.evidence import validate_selection
    failed = selection is None or status != "ollama_success"
    if selection is not None:
        try:
            validate_selection(bundle, selection)
        except ValueError:
            failed = True
    expected_facts = {fact["id"] for fact in bundle["facts"]}
    relevant = set(case["expected"]["relevant_chunk_ids"])
    expected_excerpts = {item["id"] for item in bundle["excerpts"] if item["chunk_id"] in relevant}
    expected = {"fact_ids": expected_facts, "excerpt_ids": expected_excerpts,
                "missing_information": set(bundle["missing_information"])}
    chosen = {key: set(selection.get(key, [])) if not failed else set() for key in expected}
    tp = sum(len(chosen[key] & expected[key]) for key in expected)
    fp = sum(len(chosen[key] - expected[key]) for key in expected)
    fn = sum(len(expected[key] - chosen[key]) for key in expected)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"selection_f1": f1, "valid_selection": not failed,
            "fallback": failed, "answerable": case["expected"]["answerable_selection"],
            "correct_abstention": failed or "root_cause_evidence" in chosen["missing_information"],
            "expected_excerpt_ids": sorted(expected_excerpts), "selected_excerpt_ids": sorted(chosen["excerpt_ids"]),
            "status": status}


def aggregate(rows: list[dict]) -> dict:
    actual = [row for row in rows if not row.get("failure_control")]
    answerable = [row for row in actual if row["score"]["answerable"]]
    abstain = [row for row in actual if row.get("insufficient_root_cause_evidence")]
    p95 = percentile([row["seconds"] for row in actual])
    fallback_rate = sum(row["score"]["fallback"] for row in answerable) / len(answerable) if answerable else None
    abstention = sum(row["score"]["correct_abstention"] for row in abstain) / len(abstain) if abstain else None
    unsafe = sum(row.get("unsupported_generated_claims", 0) for row in actual)
    passed = bool(actual) and p95 is not None and p95 <= 5 and fallback_rate is not None and fallback_rate <= .05 and abstention is not None and abstention >= .95 and unsafe == 0
    return {"cases_and_repeats": len(rows), "mean_selection_f1": statistics.fmean(row["score"]["selection_f1"] for row in answerable) if answerable else None,
            "warm_end_to_end_p95_seconds": p95, "answerable_fallback_rate": fallback_rate,
            "correct_abstention_rate": abstention, "unsupported_generated_claims": unsafe,
            "maximum_loaded_vram_bytes": max((row.get("loaded_vram_bytes", 0) for row in rows), default=0),
            "passes_model_gates": passed,
            "note": "Retrieval recall is evaluated separately. Root-cause accuracy is unmeasured."}


def choose_model(results: dict, retrieval_recall: float | None) -> str | None:
    if retrieval_recall is None or retrieval_recall < .9:
        return None
    eligible = [(name, result) for name, result in results.items() if result["passes_model_gates"]]
    if not eligible:
        return None
    return min(eligible, key=lambda pair: (-pair[1]["mean_selection_f1"], pair[1]["warm_end_to_end_p95_seconds"], pair[1]["maximum_loaded_vram_bytes"], pair[0]))[0]


def ollama(route: str, data=None, timeout=30):
    request = urllib.request.Request("http://127.0.0.1:11434/api/" + route,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def loaded_vram() -> int:
    return sum(int(model.get("size_vram", 0)) for model in ollama("ps").get("models", []))


def original_baseline(case: dict, model: str) -> dict:
    """Replay the frozen historical prompt for comparison with the corrected pipeline."""
    prompt = ("You are a satellite operator helping diagnose telemetry anomalies. Use only the provided anomaly packet, telemetry summaries, and retrieved mission documentation. "
              "If evidence is insufficient, say what additional information is needed. Do not claim specific category, event class, severity or root cause unless explicitly present in the packet or documents. "
              "If you infer a likely subsystem from feature names, label it as an inference.\n"
              "Anomaly Packet:\n" + json.dumps(case["packet"]) + "\nRetrieved Mission Documentation:\n" + json.dumps(case["documents"]) +
              "\nControlled Telemetry Tools Available: get_raw_window(dataset_id,start_row_id,end_row_id); get_context_window(dataset_id,start_row_id,end_row_id); get_column_summary(dataset_id,column_name,start_row_id,end_row_id); get_rows_around_peak(dataset_id,anomaly_id)."
              "\nReturn summary, most likely subsystem, telemetry evidence, documentation references, recommended actions and additional data needed.")
    started = time.monotonic()
    response = ollama("generate", {"model": model, "prompt": prompt, "stream": False}, timeout=30)
    text = response.get("response", "")
    return {"case_id": case["id"], "seconds": time.monotonic() - started, "response": text,
            "done": response.get("done"), "done_reason": response.get("done_reason"),
            "eval_count": response.get("eval_count"), "prompt_eval_count": response.get("prompt_eval_count"),
            "review_status": "Free-form claims require manual review. Substring matches and LLM judging are excluded."}


def evaluate(args):
    tested_source = source_fingerprint()
    from src.rag.diagnosis import diagnose_packet
    from src.rag.evidence import build_evidence, render_report
    _, cases = load_cases(args.corpus)
    split = "development" if args.stage == "development" else "acceptance"
    cases = [case for case in cases if case["split"] == split]
    profile_path = args.output / "development.json"
    if args.stage == "acceptance":
        profile = json.loads(profile_path.read_text())
        if profile["corpus_sha256"] != fingerprint(args.corpus):
            raise ValueError("Acceptance corpus differs from the locked development corpus")
        if not profile.get("selected_model"):
            save(args.output / "acceptance.json", {"status": "no_development_candidate_passed", "selected_model": None})
            return
        models = [profile["selected_model"]]
        repeats = 3
    else:
        models, repeats = args.models, 1
    retrieval = json.loads((args.output / "retrieval.json").read_text())
    if retrieval["corpus_sha256"] != fingerprint(args.corpus):
        raise ValueError("Retrieval evaluation corpus changed")
    rows, summaries, cold = {}, {}, {}
    for model in models:
        try:
            metadata = ollama("show", {"model": model})
            identity = next(item for item in ollama("tags")["models"] if item["name"] == model or item.get("model") == model)
            if args.stage == "acceptance" and identity["digest"] != profile["model_digests"][model]:
                raise ValueError("Selected model digest changed before locked acceptance")
            ollama("chat", {"model": model, "messages": [], "keep_alive": 0})
            started = time.monotonic()
            ollama("chat", {"model": model, "messages": [], "keep_alive": "10m"}, timeout=120)
            cold[model] = {"seconds": time.monotonic() - started, "digest": identity["digest"], "capabilities": metadata.get("capabilities", [])}
        except Exception as error:
            summaries[model] = {"passes_model_gates": False, "error": str(error)}
            continue
        measured = []
        for case in cases:
            entry = retrieval["cases"][case["id"]]
            chunks = entry["chunks"]
            expected = {**case["expected"], "relevant_chunk_ids": [chunk["chunk_id"] for chunk in chunks if chunk.get("benchmark_relevant")]}
            scored_case = {**case, "expected": expected}
            for repeat in range(repeats):
                started = time.monotonic()
                from src.rag.semantic import retrieve_semantic
                identifiers = [item["feature"] for item in case["packet"]["top_contributing_features"]]
                query = "Telemetry anomaly " + " ".join(identifiers) + " missing observations operating mode documentation"
                live = retrieve_semantic(query, args.output / "retrieval-cases" / case["id"] / "index",
                    identifiers=identifiers, min_similarity=retrieval["min_similarity"], deadline=started + 5)
                chunks = live["chunks"]
                expected_files = {d["source_file"] for d in case["documents"] if d["chunk_id"] in case["expected"]["relevant_chunk_ids"]}
                scored_case = {**case, "expected": {**case["expected"], "relevant_chunk_ids": [c["chunk_id"] for c in chunks if c["source_file"] in expected_files]}}
                result = diagnose_packet(case["packet"], chunks=chunks, llm_model=model,
                    timeout_seconds=0 if case["expected"]["expected_service_failure"] else max(0, 5 - (time.monotonic()-started)), retrieval_status=live["status"])
                bundle = build_evidence(case["packet"], chunks)
                selection = result.get("structured_diagnosis")
                if selection and "selection" in selection:
                    selection = selection["selection"]
                score = selection_score(bundle, selection, scored_case, result.get("llm_status", result.get("status", "unknown")))
                canonical = render_report(bundle, selection) if selection is not None else ""
                allowed_response = canonical + ("\nGeneration status: " + result["status"] + ". Factual fallback." if result.get("llm_status") != "ollama_success" else "")
                measured.append({"case_id": case["id"], "repeat": repeat, "seconds": time.monotonic() - started,
                    "failure_control": case["expected"]["expected_service_failure"],
                    "score": score, "insufficient_root_cause_evidence": case["expected"]["insufficient_root_cause_evidence"],
                    "unsupported_generated_claims": 0 if result.get("response_text") == allowed_response else 1,
                    "loaded_vram_bytes": loaded_vram(), "timings": result.get("timings"), "model": result.get("model"),
                    "response_text": result.get("response_text"), "structured_diagnosis": selection})
        rows[model], summaries[model] = measured, aggregate(measured)
        save(args.output / f"{args.stage}-{model.replace(':', '-')}.json", {"model": model, "rows": measured, "summary": summaries[model]})
        ollama("chat", {"model": model, "messages": [], "keep_alive": 0})
    report = {"stage": args.stage, "corpus_sha256": fingerprint(args.corpus), "source_fingerprint": source_fingerprint(),
              "platform": platform.platform(), "python": sys.version, "results": summaries, "cold_start": cold,
              "model_digests": {name: value["digest"] for name, value in cold.items()},
              "latency_measurement": "Each repeat measures live retrieval, selection, validation and rendering. Timings exclude catalog packet assembly, ingestion/download and concurrent production load.",
              "review_status": "Fixtures checked programmatically. Mission-expert review is pending."}
    recall = retrieval["metrics"][split]["recall_at_3"]
    report["selected_model"] = choose_model(summaries, recall)
    report["status"] = "passed" if report["selected_model"] else "no_candidate_passed"
    report["source_fingerprint_at_start"] = tested_source
    if tested_source != source_fingerprint():
        report.update(status="invalid_source_changed", selected_model=None)
    save(args.output / f"{args.stage}.json", report)
    print(json.dumps(report), flush=True)


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((REPO / "src").rglob("*.py")):
        digest.update(path.relative_to(REPO).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def evaluate_retrieval(args):
    from src.rag.semantic import ingest_semantic, retrieve_semantic
    _, cases = load_cases(args.corpus)
    observed = {}
    for case in cases:
        directory = args.output / "retrieval-cases" / case["id"]
        documents = directory / "documents"
        documents.mkdir(parents=True, exist_ok=True)
        for document in case["documents"]:
            (documents / document["source_file"]).write_text(document["text"], encoding="utf-8")
        ingested = ingest_semantic(documents, directory / "index") if case["documents"] else {"status": "no_document_evidence"}
        identifiers = [item["feature"] for item in case["packet"]["top_contributing_features"]]
        query = "Telemetry anomaly " + " ".join(identifiers) + " missing observations operating mode documentation"
        result = retrieve_semantic(query, directory / "index", identifiers=identifiers,
                                   min_similarity=-1, include_candidates=True)
        # Indexing generates content-addressed IDs: gold identity is source filename.
        expected_files = {document["source_file"] for document in case["documents"]
                          if document["chunk_id"] in case["expected"]["relevant_chunk_ids"]}
        observed[case["id"]] = {"candidates": result.get("candidates", []), "query": query,
                                 "identifiers": identifiers, "directory": str(directory),
                                 "expected_files": sorted(expected_files), "ingestion": ingested}
    thresholds = [index / 100 for index in range(0, 101, 5)]
    def selected(entry, threshold):
        return [item for item in entry["candidates"] if item.get("exact_identifier_match") or
                (item.get("semantic_similarity") is not None and item["semantic_similarity"] >= threshold)][:3]
    def metrics(split, threshold):
        tp = fp = total = answerable = covered = 0
        for case in cases:
            if case["split"] != split:
                continue
            entry = observed[case["id"]]
            gold = set(entry["expected_files"])
            got = {item["source_file"] for item in selected(entry, threshold)}
            tp += len(got & gold)
            fp += len(got - gold)
            total += len(gold)
            if case["expected"]["answerable_selection"]:
                answerable += 1
                covered += bool(got & gold)
        return {"recall_at_3": tp / total if total else None, "precision_at_3": tp / (tp + fp) if tp + fp else 0,
                "relevant_documents": total, "retrieved_relevant": tp, "retrieved_irrelevant": fp,
                "answerable_coverage": covered / answerable if answerable else 0}
    feasible = [(value, metrics("development", value)) for value in thresholds]
    passing = [(value, score) for value, score in feasible if score["recall_at_3"] >= .9 and score["answerable_coverage"] >= .95]
    if not passing:
        save(args.output / "retrieval.json", {"status": "retrieval_gate_failed", "corpus_sha256": fingerprint(args.corpus), "calibration": feasible})
        raise RuntimeError("No development-only relevance threshold reaches recall@3 >= .90")
    threshold = max(passing, key=lambda pair: (pair[1]["precision_at_3"], pair[0]))[0]
    entries = {}
    for case in cases:
        entry = observed[case["id"]]
        result = retrieve_semantic(entry["query"], Path(entry["directory"]) / "index",
            identifiers=entry["identifiers"], min_similarity=threshold, deadline=time.monotonic() + 5)
        chunks = result["chunks"]
        # Gold references adapt only content IDs, never relevance decisions.
        for chunk in chunks:
            chunk["benchmark_relevant"] = chunk["source_file"] in entry["expected_files"]
        entries[case["id"]] = {"chunks": chunks, "status": result["status"], "seconds": result["seconds"],
                                "index_metadata": result.get("index_metadata")}
    measured_metrics = {}
    for split in ("development", "acceptance"):
        tp = fp = total = 0
        for case in cases:
            if case["split"] != split:
                continue
            gold = set(observed[case["id"]]["expected_files"])
            got = {chunk["source_file"] for chunk in entries[case["id"]]["chunks"]}
            tp += len(got & gold)
            fp += len(got - gold)
            total += len(gold)
        measured_metrics[split] = {"recall_at_3": tp / total if total else None,
                                   "precision_at_3": tp / (tp + fp) if tp + fp else 0,
                                   "relevant_documents": total, "retrieved_relevant": tp, "retrieved_irrelevant": fp}
    report = {"status": "complete", "corpus_sha256": fingerprint(args.corpus), "min_similarity": threshold,
              "calibration_split": "development", "calibration": feasible,
              "metrics": measured_metrics,
              "cases": entries, "limitations": "Each fixture has its own index for testing relevance. Retrieval on real mission documents needs mission-expert review."}
    save(args.output / "retrieval.json", report)
    print(json.dumps({key: report[key] for key in ("status", "min_similarity", "metrics")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["retrieval", "development", "acceptance", "original-baseline"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "original-baseline":
        _, cases = load_cases(args.corpus)
        rows = [original_baseline(case, "gemma:2b") for case in cases if case["split"] == "development"]
        save(args.output / "original-baseline.json", {"model": "gemma:2b", "cases": rows, "requires_manual_review": True})
    elif args.stage == "retrieval":
        evaluate_retrieval(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
