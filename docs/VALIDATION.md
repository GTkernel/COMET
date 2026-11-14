# Validation results

## PR preparation (November 13, 2026)

After the documentation and display-text cleanup, Windows Python 3.12 passed **236 tests in 41.53 seconds**, including the pinned ESA reference subprocess check. Compilation and Windows-aware Git whitespace checks passed. The cleanup was restricted to new or changed lines, with Python structure checked separately from string values. The 80-case diagnosis corpus was preserved.

The sections below record checks made at different stages. Their source fingerprints identify the tested code. In particular, the November 9 Linux mission runs preceded the wording cleanup. Full-mission training, operational rescoring and benchmark accuracy remain unverified. Hosted CI awaits publication.

## Linux mission validation (November 9, 2026)

Ubuntu 24.04 under WSL2 completed fresh imports of both official v2 archives, followed by bounded training, validation calibration, scoring, diagnostics and prediction export. Each run started with an empty runtime directory and no preparation-cache markers. Both archive SHA-256 values matched the official-source checksums recorded below.

| Check | Mission 1 | Mission 2 |
| --- | ---: | ---: |
| Complete canonical rows | 14,728,321 | 6,129,601 |
| Features, including selected telecommands | 87 | 104 |
| Full import seconds | 1,408.758 | 1,152.795 |
| Sampled import peak process-tree RSS, bytes | 7,195,332,608 | 7,769,702,400 |
| Bounded model-check seconds | 56.037 | 36.182 |
| Sampled bounded model peak process-tree RSS, bytes | 597,893,120 | 608,198,656 |
| Final retained runtime-root disk bytes | 12,118,284,176 | 12,857,155,043 |

The missions ran sequentially. These import figures include raw channel decoding and alignment, while the earlier cached-import figures below exclude that work. Raw decoding still needs memory for one complete channel. RSS was sampled every 0.25 seconds and can miss brief peaks, and disk totals include extraction, caches, telemetry and model artifacts. Download archives were stored outside each runtime root.

Both runs used Python 3.12.3, NumPy 2.5.4, pandas 3.1.1, PyArrow 26.0.1, DuckDB 2.0.0, PyTorch 2.15.0+cpu and psutil 7.2.2. Their shared source fingerprint is `0000bd972ecfb48056bee88d022c21c00563194a844cd4e150a573701169ee7f` (51 Python files), with ESA adapter `1.0.3` and upstream revision `aeebcd9ecd3e7266d6d6a035a8081b3da83dfe33`. The causal profile used 30-second and 18-second grids with observation-age limits of 300 and 180 seconds for Mission 1 and Mission 2 respectively.

The validator used the bounded model settings described below, including independent targets and inference batches of 64 and 17. All stages exited successfully. Reports, logs, checksums and import provenance are under `E:\COMET\runtime-data\linux-mission-validation-20261109\mission1` and `mission2`. The directory date uses Pacific time. Report timestamps use UTC on November 10.

## Grounded model comparison (November 5, 2026)

The evidence-selection implementation passed 236 tests in 46.21 seconds, including the pinned ESA reference test. Independent review found that the embedding path needed a stronger check to keep requests local. That check and remote-alias regression tests were added. No critical or high findings remained in the reviewed scope. Development cases set the retrieval threshold to `0.45`, with recall@3 `1.0` on both 40-case splits. Assertions check these synthetic fixtures. Mission-expert review is pending.

The development comparison measured evidence-selection F1 and warm p95 latency.

| Model | Evidence-selection F1 | Warm p95, seconds |
| --- | ---: | ---: |
| Gemma 2b | 0.8717 | 1.453 |
| Qwen 3.5 2b | 0.5375 | 0.953 |
| Qwen 3.5 4b | 0.3642 | 0.719 |
| Gemma 4 e2b | 1.0 | 0.547 |

Qwen 4b failed the model's insufficient-evidence selection gate, although the Python renderer still abstained. After the embedding check was added, the selected Gemma 4 e2b passed another 120 locked acceptance runs with F1 `1.0`, no fallbacks on answerable cases and p95 `0.578 s`. Only the selected model received this acceptance comparison. The development graph retains its measured source version. The acceptance graph uses the final reviewed implementation.

Timings include live retrieval, selection, validation and rendering. Catalog assembly and process startup are excluded. Three separate warm CLI diagnoses on an existing real ESA anomaly took `1.735`, `1.688` and `1.750 s`. The first cold request reached its deadline and returned a factual fallback. Reports kept the original values and timestamps, left subsystem and root cause unknown, and omitted the earlier fabricated power-system diagnosis. Legacy and semantic Chroma indexes coexisted. The runtime profile pins Gemma 4 and EmbeddingGemma digests, with existing models and historical diagnoses retained.

Evidence and graphs are under `E:\COMET\runtime-data\grounded-diagnosis-20261105-final`. Earlier exploratory runs remain archived separately and were excluded from model selection. The GPU was an RTX 4070 Ti with 12,282 MiB VRAM. Ollama reported loaded-model VRAM after each request, including the resident embedding model. Peak device memory was unmeasured. Independent follow-up review completed after usage-limit interruptions, while static linters remained unavailable. See [grounded diagnosis](GROUNDED_DIAGNOSIS.md) for usage and limits. Diagnostic accuracy and anomaly-detector improvement need separate evaluation.

## Live Chroma and Gemma (November 5, 2026)

Chroma 1.5.9 and local Ollama `gemma:2b` passed live API and CLI integration checks. Two synthetic documents were indexed with an isolated embedded Chroma PersistentClient. The test replaced documents without duplicating chunks and retrieved their source/version metadata, then built a packet from an existing real ESA anomaly and saved an `ollama_success` diagnosis. API generation took about 2.88 seconds. At this stage, Chroma and Ollama were available, while PDF and Linux checks were still pending.

Reviewing the response exposed invented channel meanings and a power-distribution cause. Gemma also treated numeric telemetry values as timestamps despite documentation saying that channel meanings were unknown. That failure prompted the evidence-selection changes described above. Testing with real mission documents is still needed. These checks used COMET's embedded Chroma backend. Evidence is under `E:\COMET\runtime-data\live-rag-validation-20261105`, including `report.json`, the response, retrieved chunks and CLI results. A separate index and copied validation catalog protected the original artifacts.

The local implementation remains uncommitted. The results below measure function and resource use, without establishing detection accuracy or reproducing the ESA benchmark.

## Automated checks

After compatibility restoration, Windows Python 3.12 passed **203 tests in 37.72 seconds**, including the opt-in subprocess test against the pinned ESA metric checkout. The earlier 169-test result belongs to the preceding implementation. Compilation and Windows-aware Git whitespace checks passed during restoration. Python, ML and code reviews previously addressed reference-grid null filling and final annotation restoration, mutable legacy provenance, SQLite cache-spill locking and incomplete score-part reporting.

Tests cover format/chunk equivalence, invalid records, duplicate handling, resume/retry and migration rollback. They also check train-only fitting and causal invariance, entity/gap/partition boundaries, label isolation, independent targets, run isolation, warm-up coverage, legacy checkpoints, part rotation and missing-part rejection.

GitHub CI has not run. Windows/Linux jobs are configured for Python 3.11/3.12, but the work has not been published.

## Audit cleanup

After removing unused modules and helper chains and sharing the duplicate-selection loop, all **157 tests passed in 42.38 seconds**, including the pinned ESA reference subprocess check. A new regression checks legacy profiling and preparation with YAML feature roles and quoted category values. Python and diff reviews found no cleanup regressions. Compilation passed.

The cleanup removed 1116 net production Python lines. Its source fingerprint is `dcafbc8902d94ae224888ec3b8cdd893848aaad9d7b2c5ed00f2874b796d3c1c` (40 Python files). At this cleanup stage, full Mission 1/2 imports had yet to be repeated. The measurements below retain their earlier source identities. Cleanup evidence is in `E:\COMET\runtime-data\reviews\2026-10-01\cleanup-validation.json`.

## Official v2 data

Archives came from the [official ESA-ADB v2 record](https://zenodo.org/records/15237121). Import metadata records both archive SHA-256 values and the individual channel/annotation checksums. The upstream revision is `aeebcd9ecd3e7266d6d6a035a8081b3da83dfe33`. Raw adapter version `1.0.3`, generic storage adapter `2`, and reference-grid fill configuration `3` were used.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| ESA-Mission1.zip | 3,776,246,073 | `ba28f761b1deab4dbba4728793bff139fea39dbf9cf0d9c559d619ffe75d5a72` |
| ESA-Mission2.zip | 4,098,539,932 | `e8a89be1917b6754a10bd323441e87a82c8cf2e84ed162442c2dcf72ecc346d5` |

Every output row was validated. Complete canonical imports used the causal profile with 30-second Mission 1 and 18-second Mission 2 grids, ten-cadence observation-age limits and the prescribed test boundaries. Annotations stayed separate and never entered telemetry tensors.

| Check | Mission 1 | Mission 2 |
| --- | ---: | ---: |
| Complete canonical rows | 14,728,321 | 6,129,601 |
| Channels plus selected telecommands | 87 | 104 |
| Immutable telemetry parts | 335 | 170 |
| Compressed canonical telemetry bytes | 461,057,960 | 402,064,915 |
| Complete canonical import seconds | 188.75 | 112.687 |
| Sampled import peak process-tree RSS, bytes | 471,429,120 | 512,540,672 |
| Final bounded model-check seconds | 81.969 | 47.156 |
| Sampled bounded model peak process-tree RSS, bytes | 465,457,152 | 482,590,720 |
| Retained runtime-root disk bytes | 21,813,118,023 | 21,892,863,162 |

Preparation caches were already present, so import timings describe cached decoded/aligned ESA data. Mission 1 also resumed staging after a Windows sandbox permission failure, with the successful retry using normal host permissions. The timing and memory figures do not measure fresh raw decoding.

Both missions ran concurrently on one drive. Their timings describe those runs without a controlled throughput benchmark, and RSS sampling every 0.25 seconds can miss brief peaks. Disk totals include extracted data, converted channels, staging, historical attempts and model/run artifacts. Shared download archives sit outside those runtime roots.

Part rotation reduced retained footer metadata. Before that change, complete imports sampled 7,036,481,536 bytes for Mission 1 and 7,600,431,104 bytes for Mission 2, and a separate experiment confirmed growing footer retention in a long-lived Parquet writer and its readers.

Canonical and score parts now close at 65,536 rows or 32 MiB of Arrow values, with one row group. A 1,000,000-row, 90-column helper experiment held sampled RSS near 112 MB and peaked at 120 MB. This confirms bounded footer retention. Incoming batches, encoders, feature width and raw channel decoding still need additional memory.

## Bounded model checks

Each check used eight numeric features. Eligibility depended only on usable values in the training portion of the pre-boundary sample. For each mission, the validator retained the last 4,096 rows before the official test boundary and the first 2,048 rows on the test side, preserving structural gap flags.

The pre-boundary sample used chronological 70/15/15 partitions. Independent targets stayed on the actual test side, so these small functional checks do not represent full-mission training or benchmark evaluation.

Training ran one epoch. Model settings were window size 16, hidden size 8, batch size 64 and seed 0. Training, validation calibration, held-out and independent-target scoring, diagnostics and complete prediction export passed, with inference batches of 64 and 17 producing equivalent scores within `rtol=1e-5`, `atol=1e-6`.

Each independent-target export contained all 2,048 timestamps. No-alert warm-up/gap rows were included.

Diagnostics and exports were checked again after the final report-completeness fix, using both missions' existing held-out and independent-target runs. Missing score parts now stop reporting before a CSV is replaced or diagnostic JSON is published.

## Numeric encoding investigation (October 25, 2026)

All **161 tests passed in 36.59 seconds** after adding the optional `asinh` numeric encoding. Python and code reviews approved the change, and compilation and whitespace checks passed. Regression checks cover training-only fitting, future-data invariance, signed extremes, unchanged missingness/categories, inverse original-unit values, old states without an encoding field, independent targets and inference batch equivalence. Standard encoding remains the default.

A bounded numeric batch path, used when forward fill is disabled, matched the scalar path on 10,000 real Mission 1 rows across 87 features (`rtol=1e-6`, `atol=1e-7`). That local check took 0.078 seconds batched versus 1.25 seconds scalar. Full-import timing was outside its scope. The tested source fingerprint is `0016bc35290365c43ae0eb71d562a870bdcfea1760165213049945bbb7a55733`.

The original full Mission 1 run flagged 3,747,995 of 7,362,306 scored test timestamps. Against the union of ESA intervals with `Category=Anomaly`, a direct pointwise check counted TP 249,261, FP 3,498,734, FN 47,167 and TN 3,567,144. This check uses scored timestamps only, does not apply ESA event/affiliation rules or special treatment of other categories, and is not an ESA benchmark result. The maximum score occurred inside labeled event `id_134`. Explaining that score required checking the underlying observations.

An exploratory comparison used 8,192 evenly spaced eligible training windows, 2,048 validation windows and 4,096 test windows, with frozen chronological partitions. Both fresh models used three epochs, 60-row windows, hidden size 32, batch size 128, seed 0 and separate 95th-percentile validation thresholds. Standard encoding produced TP 118, FP 2,022, FN 52 and TN 1,904. With `asinh`, the counts were TP 109, FP 1,405, FN 61 and TN 2,521. It reduced false alerts but also lost sensitivity. These are exploratory sample counts, not an independent accuracy estimate or a reason to replace the default.

The investigation and experiment settings are recorded under `E:\COMET\runtime-data\asinh-validation`. The separate `esa_robust.yaml` configuration can be used for full-dataset candidate training. Existing models, runs and prediction exports are retained.

## Operational monitoring checks (October 25, 2026)

The new recorded-batch monitoring workflow passed **169 tests in 37.59 seconds**, including the pinned ESA reference test. Python and code reviews approved it after fixes for mutable legacy parent identities and incomplete resumed exports. Compilation, whitespace checks and actual CLI replays passed. Regression tests cover past-only model updates, future-value/label invariance, sparse-feature eligibility, numeric window referencing, immutable input arrays, source races, entity selection, export tampering, and separate batches with reordered features, unrelated extras and missing nonstructural context.

Real Mission 1 checks used four weekly updates for January 1-29, 2007, and another four for May 18-June 15, 2009. Each update used up to 90 earlier days, the last 14 for validation, 60-row windows, training stride 60, hidden size 32, three epochs, batch size 128, seed 0 and a 95th-percentile validation threshold. Standard numeric scaling was used. Operational feature eligibility and first-sample numeric window referencing were enabled. Evaluation labels were used only for the comparisons below.

Each four-week export contains 80,640 timestamps. January has 236 unscored warm-up rows. May/June has 1,964 unscored gap/warm-up rows. All unscored rows retain no-alert, null-score records. The January export has 6,487 alerts, and the May/June export has 22,466. Model/run/target IDs and thresholds are recorded per row and in the adjacent JSON sidecar.

The table compares only timestamps scored by both methods. Positive labels are the union of intervals with `Category=Anomaly`. These direct counts do not apply ESA event or affiliation rules, do not evaluate individual channels, and are not ESA benchmark results.

| Range / method | Common scored rows | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: |
| January / original fixed model | 80,403 | 812 | 72,366 | 37 | 7,188 |
| January / operational updates | 80,403 | 76 | 6,411 | 773 | 73,143 |
| May-June / original fixed model | 78,676 | 57,042 | 0 | 13 | 21,621 |
| May-June / operational updates | 78,676 | 20,484 | 1,982 | 36,571 | 19,639 |

On the common January timestamps, the alert rate fell from 91.0% to 8.1%. The earlier 99% observation came from systematic sampling across all of 2007, not this January range. The operational mode also flagged fewer timestamps inside long anomaly intervals, and added false positives in the May/June range. It is not uniformly more accurate.

A global interval-hit check found alerts inside both annotated events in these ranges, `id_115` and `id_134`. For `id_134`, the earliest annotated start was 13:09:02.418 UTC on May 25, 2009. The first alert at or after that start occurred at 13:12:30 UTC. These two exploratory checks are insufficient to estimate event recall or benchmark accuracy.

A separate fresh February 5-12, 2007 update processed 20,160 target rows in 20.093 seconds, with sampled peak process RSS of 707,657,728 bytes using two PyTorch threads. Sampling every 0.05 seconds can miss brief peaks. Source telemetry was already decoded and ingested. This measures one operational update, not raw decoding or a full-mission run.

Window referencing emphasizes local changes and can stop flagging a sustained absolute fault. Unlabeled history can also contain faults that a later update learns. The original model and seven-year prediction export were retained. Full-mission operational rescoring and Linux execution have not been performed.

Evidence is in `E:\COMET\runtime-data\operational-validation\validation.json`, with exported CSV checksums, model/run/source identities, settings, versions, resource samples and detailed comparisons. The final tested source fingerprint is `e941ea6d9e0a401bdbedea1d77412622c63fc75fb21e4e701d7090a84b506454` (41 Python files). Follow [operational monitoring](MONITORING.md) to repeat the workflow.

## Environment and evidence

The checks ran on Windows 11 build 26200 with Python 3.12.14, NumPy 2.5.2, pandas 3.0.6, PyArrow 25.0.1, DuckDB 1.5.5, PyTorch 2.14.0+cpu and psutil 7.2.2. Ruff, mypy, pylint, black and bandit were unavailable. The recorded checks and reviews make no claim that those tools ran.

Source hashes identify the tested code. The validator records them even without a commit, with final completeness-check fingerprint `404b45db70d2f41c386f9a71cb5abafd3739d8243a5200295fd1898f67689ac6` (47 Python files). Earlier training/import reports retain their own source identities, and the final reporting fix left their fitted models and telemetry unchanged.

Find the evidence under `E:\COMET\runtime-data\mission1-validation` and `mission2-validation`. It includes `validation-report-full-import-parts.json`, `validation-report.json`, `import.json`, `bounded.json`, `post-review-validation.json`, prediction exports and stage logs.

To repeat these checks, use [the opt-in validator](../scripts/validate_esa.py) and the commands in [the ESA walkthrough](ESA.md), keeping runtime datasets, databases, models and score artifacts outside the proposed change.

## Post-compatibility real-data checks (October 26, 2026)

The available Windows installation completed the checks below without installing additional packages. Both archive SHA-256 values matched the previously verified v2 sources. Imports reused completed caches and catalog entries. Raw pickle decoding and alignment were outside these checks. Earlier evidence was copied before the validator updated its reports.

| Check | Mission 1 | Mission 2 |
| --- | ---: | ---: |
| Complete stored telemetry rows scanned | 14,728,321 | 6,129,601 |
| Full scan time | 8.42 s | 5.53 s |
| Full scan sampled peak process RSS | 144.0 MiB | 176.5 MiB |
| Fresh bounded model workflow time | 62.97 s | 37.13 s |
| Bounded workflow sampled peak process-tree RSS | 439.1 MiB | 467.9 MiB |
| Independent-target exported rows | 2,048 | 2,048 |
| Scored / explicit no-alert warm-up rows | 2,033 / 15 | 2,033 / 15 |
| Restored diagnostic plots, verified through API and CLI | 5 | 5 |
| One-day operational update exported rows | 2,880 | 4,800 |

Full scans checked every stored row for contiguous IDs, strict timestamp order, finite observed numeric values and annotation-free telemetry schemas. Missing observations remain allowed. The scan checks stored timestamp coordinates and nonmissing timestamp text. The importer handles parsing source timestamps. Each model smoke test trained on 4,096 pre-boundary rows and eight selected numeric channels, then scored the held-out and separate target datasets. Inference batches of 64 and 17 produced equivalent scores. Diagnostics verified replay against immutable scores and generated the five restored plots.

Packets referenced their actual target and run. Synthetic Markdown documents were ingested, retrieved through the JSONL fallback and included in explicit unavailable-Ollama fallback diagnoses. Raw-window tools, context tools and prompts contained no annotation columns. This exercised fallback behavior, not real mission-document guidance or live LLM generation.

Each fresh operational check used three earlier days of history, reserving the final earlier day for validation, and scored the next day. Selected channels with no usable history were explicitly excluded and recorded. Training/validation boundaries preceded the prediction cutoff. Resuming reused the same runs and produced byte-identical combined prediction exports. These were short functional updates, not multi-year operational rescoring.

Both short target ranges contained no selected labeled anomalies. The pinned ESA reference subprocess therefore returned undefined metrics with an explicit reason. A separate check evaluated the existing January Mission 1 operational export against one labeled event. Event precision was approximately 0.000167 and event recall was 1.0, with many unmatched alarms. The limited range supports an integration check only. Mission 2's fixed-model smoke test flagged 2,032 of its 2,033 scored windows. Alert reliability still needs evaluation.

Fresh measured model runs used source fingerprint `28cdf00955b909024d16521ce3fa34bbed2a708d9db97302259506a0ec950435` (49 Python files). A later help-text correction clarified that feature YAML works with SQLite and Parquet. The final automated test suite used fingerprint `b0fa7a8b908b21021b8cd6dc21bbe787ed4bfcd8ea85a3cdf5a54920821b06da`. The validation directories retained about 21.83 GB and 21.95 GB respectively, including earlier caches and artifacts. A fresh run may use a different amount of disk.

Detailed JSON, prior reports, commands, plots, packets and fallback diagnoses are under `E:\COMET\runtime-data\post-compat-validation-20261026`. Its `summary.json` records versions, checksums, source identities, timings, sampled memory, disk totals and limits. Evidence review found no blocking functional-validation findings. Live Chroma/Ollama, PDF parsing, Linux and a separately provisioned reference dependency environment remain untested. No commits or PRs were created.
