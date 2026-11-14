# COMET

COMET trains an LSTM autoencoder on telemetry and flags observations it struggles to reconstruct. Start with a telemetry table. Score another dataset with compatible features. Supply mission documents to help investigate alerts. Supported inputs include CSV/TSV, JSON arrays/JSONL, Parquet, and trusted raw or processed ESA Mission 1/2 data. Use feature columns or numeric timestamp/channel/value rows.

Use structured telemetry with finite observed values. If COMET cannot identify timestamps, entities or feature roles, supply a manifest that records those choices for reuse. Units and channel meanings come from you. Live streams, binary protocols, images and prose need separate adapters.

## Install

COMET requires Python 3.10 or newer. The CI workflow targets 3.11 and 3.12. From the checkout, create an environment and install the modeling and ESA extras.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[modeling,esa]"
```

On Linux, activate with `source .venv/bin/activate`. For ingestion alone, use `pip install -e .`. Add `[rag]` for local document ingestion and diagnosis, `[reference]` for the ESA evaluation environment, or `[dev]` for tests. Install CPU-only PyTorch first if needed, using its [official installation selector](https://pytorch.org/get-started/locally/).

Use the editable checkout to access the repository's default YAML files. Standalone wheels currently need an explicit `--config PATH` for training because those files are outside the Python package.

Run commands with `comet` or `python -m src.cli`. To store runtime data on another drive, set these paths first.

```powershell
$env:COMET_STORAGE_DIR = 'E:\COMET\runtime-data'
$env:COMET_SCRATCH_DIR = 'E:\COMET\runtime-data\scratch'
```

The checkout remains the default storage root, with `--storage-dir`, `--scratch-dir` and `--db` available before the CLI command to override paths. Keep raw data, databases, Parquet artifacts, models and memmaps out of commits.

## Use a new dataset

```powershell
comet inspect telemetry.csv --output-manifest telemetry.manifest.json
# Review timestamp, roles, entity IDs, units and gap assumptions in the mapping.
comet run telemetry.csv --manifest telemetry.manifest.json
```

Check the mapping before importing. `inspect` checks a bounded sample, while `ingest` validates every record, so resolve any assumptions inspection leaves open before starting the full import.

With `run`, COMET imports and trains, calibrates the threshold on validation scores, then scores the test partition and prints the run ID and coverage. It also records and prints chronological boundaries. Generic datasets use 70% training, 15% validation and 15% test rows within each entity. Windows stay within their partition and entity, with declared gaps and excluded intervals breaking them.

To run the stages separately, use the following commands.

```powershell
comet ingest telemetry.csv --manifest telemetry.manifest.json
comet profile --dataset-id 1
comet prepare --dataset-id 1
comet train --dataset-id 1 --config config/model_configs/default.yaml
comet ingest later-telemetry.parquet --manifest later.manifest.json
comet detect --dataset-id 2 --model-id 1
comet model-diagnostics --dataset-id 2 --model-id 1 --run-id RUN_ID
comet export-predictions --run-id RUN_ID --output predictions.csv
comet list-anomalies --run-id RUN_ID
comet anomaly-packet --anomaly-id ANOMALY_ID
```

`prepare` records structural boundaries. Training then fits scaling, category vocabularies and missingness encoding using training rows only. The saved model includes those settings, the feature order, fingerprints and the threshold calibrated on validation scores.

Column order can change between datasets. Detection matches feature identities and ignores unrelated extras. Missing required features and conflicting roles/units stop scoring. For missing values, COMET carries earlier observations forward within the declared limits, then uses fallback values learned during training. Raw ESA alignment already performs this hold, so it disables a second fill.

Omit the path to `ingest` to import CSVs from `STORAGE/data/`, with any failed file causing the command to return a failing exit code. COMET sorts each entity on disk and collapses identical observations, stopping on conflicting duplicates unless the manifest specifies `first` or `last`.

The [original workflow compatibility guide](docs/COMPATIBILITY.md) covers editable feature YAML, default training configuration, restored diagnostic plots and Python helpers. Original commands retain text/TSV output. Add `--json` for structured output. CSVs without timestamps remain usable in row order, with unavailable real timestamps recorded explicitly.

Retry with the same source and manifest. Failed or interrupted imports retain staged progress, and the retry can use a different chunk size. ESA output arrives ordered, so COMET verifies it while streaming and skips the sort.

## ESA data

If normal behavior changes over time, try the [operational monitoring workflow](docs/MONITORING.md). It trains from recent history, scores the next recorded batch and records which model produced each result. Use a fixed model for ESA's prescribed test period.

Import ESA data with the built-in adapter. It handles labels without manual removal or conversion scripts. Follow the [step-by-step ESA walkthrough](docs/ESA.md), with the [manifest contract](docs/MANIFEST.md) and [migration guide](docs/MIGRATION.md) for configuration and existing catalogs.

Start with the default **causal** profile, which keeps evaluation annotations separate from telemetry. The **ESA reference** profile follows pinned upstream offline preprocessing, including annotation-aware resampling, and records corrections for edge cases. Either profile still needs evaluation before you can draw conclusions about detection accuracy or benchmark reproduction.

Scores arrive at window end. Exports include warm-up and gap timestamps. These rows have no alert and null scores.

Each detection identifies its run and target rows. Runs are immutable. Select the run in diagnostics and evaluation, or use the latest completed run for that target/model. Feature contributions show which inputs contributed to reconstruction error. They do not establish individual channel detections.

Use `evaluate --metric legacy-overlap` for COMET's interval-overlap diagnostic. It does not measure ESA benchmark performance. The isolated `esa-reference` harness passes binary predictions directly to the pinned ESA global metric, bypassing continuous-score integer coercion. Undefined denominators produce null metrics with reasons, and full channel-aware benchmark metrics remain outside this release.

## Optional document diagnosis

Follow the [local diagnosis guide](docs/GROUNDED_DIAGNOSIS.md) for semantic reindexing, model profiles and measured comparisons, including how reports use checked evidence and leave channel meanings or root causes unknown when documentation is missing.

Install `[rag]` and run [Ollama](https://ollama.com/) locally. Place `.txt`, `.md` or `.pdf` mission documents under `STORAGE/rag/documents/`, then run `comet ingest-docs --reindex-semantic` with local `embeddinggemma:300m` available. Diagnose an alert with `comet diagnose --anomaly-id ID --llm-model MODEL`. Check the cited evidence before acting. Annotation labels are excluded from telemetry tools, packet context and prompts.

## Validation and layout

```powershell
python -m pip install -e ".[modeling,esa,reference,dev]"
python -m pytest -q tests
```

The Windows/Linux CI workflow is configured to run synthetic integration tests. For real data, the optional [validator](scripts/validate_esa.py) imports full verified Mission 1/2 archives and performs bounded functional training and independent inference. It writes local JSON/log files with sampled peak process-tree RSS, disk use, timings, versions and checksums. It does not calculate accuracy metrics.

Read the [validation results](docs/VALIDATION.md) for the executed checks and resource measurements, including versions, checksums and measurement limits.

SQLite holds the catalog and legacy datasets. New immutable telemetry and annotations live in `STORAGE/datasets/`, with DuckDB using `SCRATCH` for temporary generic sorting. ESA alignment scans each decoded channel once in bounded batches.

Look under `STORAGE/models/` for model arrays and lazy window indices, `STORAGE/preparations/` for structural indices, and `STORAGE/runs/` for complete detection scores. The [architecture notes](docs/MANIFEST.md) explain limits, recovery and adapter extension points.
