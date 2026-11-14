# Grounded local diagnoses

COMET asks the local model to select evidence identifiers. Python checks selections and renders six sections. Telemetry values and timestamps come from the packet. If evidence is missing or generation fails, the report uses a factual fallback and states why. The same fallback handles malformed output, a full context budget or an expired deadline.

Review subsystem mappings yourself. A channel's `documented_mappings` entry must supply `subsystem`, `source_file`, `document_version`, `verified_by` and an exact `evidence_quote`, with an optional `unit`. COMET checks that the cited version and quote appear in retrieved evidence. Your review establishes the mapping's meaning. Mentioning a channel and subsystem in the same document is insufficient, and root cause remains unknown.

## Use

Set `COMET_STORAGE_DIR` to your runtime directory, place mission `.txt`, `.md` or `.pdf` documents under `rag/documents`, then run these commands.

```powershell
comet ingest-docs --reindex-semantic
comet diagnose --anomaly-id ANOMALY_ID
```

Install local `embeddinggemma:300m` before indexing. Each reindex builds a versioned Chroma collection and keeps earlier semantic and hash-embedding collections. Changed embedding weights, dimensions or chunking need reindexing. If the index is missing or incompatible, diagnosis reports the problem and uses a factual fallback.

Retrieval combines semantic similarity with exact identifier matching. It considers eight candidates and returns up to three complete excerpts within a 1,200 UTF-8-byte budget, keeping quotes intact. Development cases set the relevance threshold. Python reads the packet and context before inference. Retrieved documents supply quoted evidence, and instructions inside them have no authority over the report.

## Model profile and timing

The deployment profile lives at `STORAGE/config/diagnosis-profile.json`. It pins the accepted model digest, embedding digest and similarity threshold. Use `--llm-model` to choose another local model explicitly. Without a profile, COMET defaults to the corrected Gemma 2b pipeline. A profile with `selected_model: null` uses factual reports without automatic generation. COMET rejects unsupported cloud models and keeps historical diagnoses.

This machine's accepted profile selects `gemma4:e2b` with relevance threshold `0.45`. Temperature is zero. Output is capped at 256 tokens, context at 4,096 tokens and residency at ten minutes, with thinking disabled when supported. Selections must use known fact/excerpt IDs and missing-information categories. Input-size checks guard against context truncation. If a local service stalls, the caller gets a deadline fallback while the worker retains its slot until it finishes.

The five-second target applies to warmed requests. Cold loading may exceed it and return a factual fallback. Preload the selected model with this PowerShell command.

```powershell
Invoke-RestMethod http://localhost:11434/api/chat -Method Post -ContentType application/json -Body '{"model":"gemma4:e2b","messages":[],"keep_alive":"10m"}'
```

The response includes `response_text`, structured selections, status, retrieval/generation timing and model identity. Diagnosis builds its selection prompt in Python. Stored prompt metadata identifies the selection protocol used to produce the report.

## Reproduce evaluation

```powershell
python scripts/benchmark_diagnosis.py --stage retrieval --output E:\COMET\runtime-data\diagnosis-benchmark
python scripts/benchmark_diagnosis.py --stage development --output E:\COMET\runtime-data\diagnosis-benchmark
python scripts/benchmark_diagnosis.py --stage acceptance --output E:\COMET\runtime-data\diagnosis-benchmark
```

The corpus has 40 development fixtures and 40 acceptance fixtures, with related events and document variants kept in the same split. The fixtures were generated programmatically and checked with assertions. Mission-expert review is pending. Development results select a model, which locked acceptance then tests three times per case. Timings include live retrieval, selection, validation and rendering. Cold loads are recorded separately. Catalog assembly, CLI startup and concurrent production load are outside these timings. Separate real ESA CLI checks cover the complete warmed diagnosis function.

On November 5, 2026, Gemma 4 e2b achieved evidence-selection F1 `1.0`, no answerable-case fallbacks and final warm acceptance p95 `0.578 s` across 120 case/repeat combinations after the local-only embedding check. The preceding acceptance run measured `0.547 s` and remains archived. Three warm CLI checks on the earlier real ESA failure case took `1.735`, `1.688` and `1.750 s`, preserved exact values and reported unknown subsystem/root cause without the earlier invented power-system claims. A first cold request correctly returned a deadline fallback.

These results measure evidence selection on synthetic fixtures. Mission diagnostic accuracy needs separate evaluation. Python validates model selections and renders the reports, and the models received no fine-tuning. Artifacts and PNG/SVG/PDF graphs are under `E:\COMET\runtime-data\grounded-diagnosis-20261105-final`. Independent review completed after usage-limit interruptions. Its final finding led to a regression-tested check that embeddings stay local. No critical or high findings remained in the reviewed scope, and the full suite passed 236 tests. Embedding and generation both verify local GGUF weights and reject remote aliases before sending document or query text.
