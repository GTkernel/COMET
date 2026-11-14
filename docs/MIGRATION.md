# Existing catalogs and checkpoints

Back up the catalog and its artifacts. COMET defaults to `db/satellite_operator.db`. If you choose another storage root, it opens that root's catalog unless you supply the old path with `--db`, so check the path before migrating.

Migration 1 runs on first use. It adds preparation and detection-run tables. Anomalies gain nullable run/timestamp columns. Dataset IDs, model IDs, historical anomalies and diagnoses survive the migration. Schema changes share one transaction. Failures roll back. COMET rejects databases from newer schema versions.

Existing `raw_dataset_ID` tables remain readable. New imports write immutable Parquet artifacts, while the legacy reader removes annotations and normalizes timestamps in bounded batches.

Check a legacy table before retraining. It needs an unambiguous profiled timestamp, contiguous row IDs and increasing timestamps. Otherwise, reimport through `ingest INPUT --manifest ...`. That creates a new dataset ID while keeping the old one and its historical results.

Retrain for a reusable feature contract. New training fits preprocessing even for readable legacy raw datasets. Existing checkpoints retain their original prepared, same-dataset workflow and carry a legacy marker, because their fitted preprocessing cannot be reconstructed reliably enough for cross-dataset use.

Legacy SQLite tables can change. New models fingerprint their canonical telemetry content in bounded batches, and detection checks that fingerprint before reusing fitted row boundaries. Retrain or reimport after changing a raw table, even if its dataset ID stays the same.

Content digests are identified separately. Original source checksums remain unavailable if the historical catalog never recorded them.

`profile` writes editable feature YAML for SQLite and Parquet datasets. After editing roles, run `prepare --feature-config PATH`. Training uses those exclusions and numeric, categorical or context choices. For Parquet, preparation records partition/gap boundaries alongside the telemetry files. Legacy SQLite preparation replaces its model/context tables and publishes feature roles in one transaction. Changes to timestamp identity, source alignment or annotation storage require an updated import manifest and a new import.

Training defaults to `config/model_configs/default.yaml`. Python callers can supply a custom `models_dir`. New checkpoints fit reusable preprocessing on training rows only. Old checkpoints continue to use their prepared legacy scaling.

Original CLI commands retain their human-readable summaries and tab-separated anomaly listings. Add `--json` before or after a command for structured output. New commands such as `inspect`, `run` and `monitor` use JSON output by default.

`model-diagnostics` writes the original five plot types and a summary with median, population standard deviation and feature rankings. It reads the completed run's scores and checks them against batched replay. The threshold stays fixed. Figures and summaries identify samples. Statistics and rankings cover all scored windows. Packets and context tools also work with legacy prepared-only tables. Their summaries identify standardized or encoded values whose original units are unavailable.

Original Python imports, including telemetry records/converters, database helpers and training configuration helpers, remain available. Historical anomalies with null run IDs retain their original overlap boundaries. New runs timestamp alerts at window end. With no run specified, the anomaly loader selects the latest completed run for the dataset/model, falling back to historical records only when no completed run exists.

Select runs with `--run-id`. Every new detection creates one, and failed runs stay hidden from diagnostics/evaluation, so use the ID when comparing reports or exporting predictions.

Historical anomalies remain accessible with null run IDs. Legacy overlap evaluation selects them only with a legacy checkpoint, and retraining or new detection leaves those records intact.

Set runtime paths with `COMET_STORAGE_DIR`, `COMET_SCRATCH_DIR` or global CLI options. Catalog artifact paths are absolute. After moving storage, update those references or restore the directory to its original location, because this release has no automatic relocation command.
