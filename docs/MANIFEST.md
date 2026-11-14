# Structured telemetry contract

Start with `version: 1`. A manifest is a JSON or YAML object, as shown in the [wide example](../examples/telemetry.manifest.json) and [long example](../examples/long.manifest.json). Use `inspect INPUT` to get a suggested mapping from a bounded sample. Ingestion checks every record.

| Field | Meaning |
| --- | --- |
| `format` | `auto`, `csv`, `tsv`, `json`, `jsonl`, `parquet` |
| `layout` | `wide` (feature columns) or `long` (numeric channel/value observations) |
| `timestamp` | Exact source timestamp column |
| `ordering` | Default `timestamp`. Set `row` to preserve wide-table source order with real timestamps left empty. Row order supports fractional partitions only and cannot use calendar-based monitoring or interval annotations |
| `timezone` | Timezone for naive timestamps. Default UTC. Ambiguous/nonexistent local times rejected |
| `epoch_unit` | Required for numeric timestamps. Choose `s`, `ms`, `us` or `ns` |
| `entity_columns` | Ordered source identity columns. Observations/windows remain separate per entity |
| `numeric_features`, `categorical_features` | Source columns with explicit roles |
| `context_columns` | Diagnostic-only columns, absent from tensors |
| `label_columns` | Annotation columns, absent from telemetry, diagnostics and prompts |
| `channel_column`, `value_column` | Long-layout channel identity and numeric value |
| `units` | Explicit source/channel name to unit string. Never inferred or converted |
| `cadence_seconds`, `max_gap_seconds` | Sampling expectation and maximum temporal gap/causal fill age |
| `gap_column` | Context column containing `0/1` or boolean source-outage flags. Flagged rows retained with no scores |
| `duplicate_policy` | `error` by default. Explicit `first` or `last` resolves conflicts in source order |
| `splits` | `fractions: [0.7,0.15,0.15]`, or `train_end`/`validation_end`, or `test_start` with pre-test `train_fraction` |
| `preprocessing.forward_fill` | Enable causal fill within gap limits. Default true, disabled after raw ESA alignment |
| `preprocessing.exclude_training_anomalies` | Explicit optional training annotation filter. Excluded intervals break windows |
| `esa` | Mission/version/profile/trust and optional output date/channel selection. See ESA walkthrough |

Source names remain the feature identities. COMET creates internal names that avoid collisions and rejects duplicate/blank headers and reserved `__comet_` names.

Check automatic roles before importing. The first usable observation establishes a schema, and ingestion checks the rest of the column for mixed numeric and nonnumeric values. Map a mixed column as categorical if that is its intended role. Numeric-looking categories, empty sample columns and numeric timestamp units also need an explicit mapping.

Declare unfamiliar annotations in `label_columns`. COMET recognizes known label names automatically and prevents them from being assigned feature or context roles.

NaN and infinity fail validation. Missing fields remain null until model preprocessing applies training-derived numeric fallbacks and missingness indicators. Training rows determine the bounded categorical vocabulary, with separate encodings for missing and unknown categories, and every selected feature needs usable training values or training stops.

Supply consistent units across model and target, because COMET rejects conflicting declarations and performs no unit conversion.

Partition before fitting or creating windows. Generic fractions apply to sorted rows within each entity. ESA calendar boundaries preserve the prescribed test periods, with the final 15% of pre-test rows used for validation by default. The model saves its ordered source feature contract and transformation fingerprints. Causal fills reset at entity and partition boundaries, declared gaps and excluded rows.

## Storage and recovery

SQLite holds the catalog. New telemetry lives in immutable Parquet artifacts with UTC ISO timestamps, integer nanoseconds, entity identity and stable 1-based row IDs. Annotations have separate Parquet artifacts.

Profiling, training, scoring and export read bounded batches. Distinct-value tracking has a cap too. Reports identify cardinalities that are approximate or only lower bounds.

Read parts in sequence. Canonical telemetry, generated point annotations and new scores use directories of numbered Parquet parts, each with one row group and at most 65,536 rows or 32 MiB of uncompressed Arrow values. The first limit reached closes the part. Readers close it before opening the next, which bounds retained footer metadata.

Allow memory for buffers too. The part limit covers uncompressed Arrow values. Encoding, compression and the incoming batch need additional memory, and a record larger than the byte limit stops the import. Readers also accept historical single-file telemetry and score artifacts.

An adapter inspects the source, resolves its schema and yields bounded batches. `store.ingest_source` validates staged parts and fingerprints the source and configuration. DuckDB handles temporary sorting, deduplication and alignment where needed. Once processing finishes, COMET publishes the artifacts and records the completed dataset in SQLite.

Retry with the same source and manifest. The chunk size may change. OS locks prevent concurrent writers for one import/cache and release automatically if a process stops. Correct invalid data and reimport it to create a new fingerprint, leaving staged and published artifacts untouched.

Models read memory-mapped row/value/window indices through PyTorch lazy datasets. Calibration scores stay on disk. An exact in-place percentile selection sets the threshold.

Each run stores complete target scores and no-alert coverage rows, with reconstruction scores timestamped at the window end. The input-window start time serves only as diagnostic context.

Reduce batch size for wide tables. Batch memory grows with both row count and feature count. DuckDB's memory setting limits its managed buffers, while other allocations can raise process RSS beyond that setting. Raw pickle decoding needs a complete channel in a subprocess.

Long-form pivoting caps channel cardinality at 4096. Category statistics/vocabularies also have explicit caps, and source checksums require one streaming pass over every file.

For the underlying APIs and memory limits, see [Arrow batch scanning](https://arrow.apache.org/docs/python/dataset.html), [DuckDB spill behavior](https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads) and [PyTorch datasets/loaders](https://docs.pytorch.org/docs/stable/data.html).
