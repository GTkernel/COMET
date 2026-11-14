# Update COMET as new observations arrive

Run `monitor` when a new recorded batch arrives and normal behavior has changed over time. It trains on recent history, reserves the later part of that history for threshold calibration, then scores the next batch. Future observations and evaluation labels stay out of model fitting.

For an ESA benchmark comparison, keep one model fixed across the prescribed test period. `monitor` instead measures an operational workflow that updates models between batches. Ordinary `train`, `detect` and `run` retain their existing defaults.

## Try it on the imported ESA mission

Open PowerShell in `comet-source` and set the storage location used for ingestion.

```powershell
$env:COMET_STORAGE_DIR = 'E:\COMET\runtime-data'
$env:COMET_SCRATCH_DIR = 'E:\COMET\runtime-data\scratch'
```

Run this command with your imported mission's dataset ID in place of `1`.

```powershell
.\.venv312\Scripts\comet.exe monitor --dataset-id 1 --start 2007-01-01 --end 2007-01-29 --output E:\COMET\runtime-data\mission1-operational-predictions.csv
```

Use `comet` if your environment is activated, or replace `.\.venv312\Scripts\comet.exe` with your environment's path. Start and end are UTC. The end is exclusive, so this example checks four weeks.

By default, each update uses up to 90 days of history. The final 14 days calibrate the threshold. Earlier rows train the model. A new model scores the following seven days, then the process advances seven days. Training and validation each need complete windows, which stop at gaps and entity boundaries, so a chosen history range must contain enough consecutive observations for both phases.

The operational defaults use 60-row windows, non-overlapping training windows, three epochs, and a 95th-percentile validation threshold. Numerical means and scales are fitted on training rows only. Numeric window values are referenced to the first encoded sample in that window, helping distinguish local changes from steady offsets. Missingness indicators and categorical encodings are unchanged. Packets retain original-unit values.

Features without usable training observations are excluded from that model, with their names and reasons recorded. Future or validation observations cannot make a feature eligible. This handles inactive telecommands without inventing values. An excluded feature is not monitored by that model.

## Read the output

Open the output CSV in a spreadsheet. The timestamp and score columns remain. So do alerts, coverage and row references. Additional fields are `dataset_id`, `model_id`, `run_id` and `threshold`. Identify a row with both `dataset_id` and `row_id`, since row numbers can repeat across target snapshots.

A score above its row's threshold produces `is_anomaly=1`. Scores measure reconstruction error. Compare them against the threshold for the model that produced them, since different models can produce different error scales. Each update retains explicit no-alert warm-up rows with blank scores, plus any gap records.

The adjacent `.csv.json` file records every update's cutoff, training boundaries, excluded features, model/run IDs and individual export. Immutable snapshots preserve the parent dataset identities and selected time ranges. Intermediate files live under `STORAGE/monitoring/`.

## Supply a new batch separately from its history

Ingest the new file normally. Use its printed ID as `--dataset-id` and the ID of an earlier compatible dataset as `--history-dataset-id`.

```powershell
comet ingest new-batch.parquet --manifest telemetry.manifest.json
comet monitor --dataset-id NEW_ID --history-dataset-id HISTORY_ID --start 2026-10-01 --end 2026-10-08 --output latest-predictions.csv
```

Replace the IDs and dates with your actual data. Initial history ends before the first cutoff. At subsequent updates within the command, already observed rows from the new batch can enter history. Required channel roles and units must remain compatible. Monitor one entity at a time. Select it with `--entity-id` in a multi-entity dataset.

Run the command when a recorded batch arrives. It does not start a background watcher or read a live protocol.

## Adjust and retry

Use `--history-days`, `--validation-days` and `--update-days` to change the calendar windows. History must exceed validation, and both need enough observations for the chosen model window. `--config` overrides individual operational model settings. COMET records the resolved settings. Training stride defaults to 60 even if you change only the window size, so set `stride` explicitly for a different spacing.

After an interruption, repeat the command. COMET checks completed runs and their exports before reusing them. If an export has changed, restore it before resuming. The combined CSV is replaced once every requested period finishes successfully.

## Limits

This mode emphasizes local changes. A sustained absolute offset can become unremarkable after the window reference moves, so it does not replace engineering limits or incident tracking. Retraining on unlabeled history can also learn an undetected fault. Review the history and retain model versions when operating conditions matter.

Fewer alerts alone do not demonstrate better fault detection. The checked ranges and both missed-interval and detected-event results are recorded in [validation results](VALIDATION.md).
