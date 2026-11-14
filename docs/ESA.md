# Use ESA-ADB data with COMET

## 1. Install and choose storage

Create a Python environment in the COMET checkout, then run `pip install -e ".[modeling,esa]"`. On Windows, set the runtime paths before importing.

```powershell
$env:COMET_STORAGE_DIR = 'E:\COMET\runtime-data'
$env:COMET_SCRATCH_DIR = 'E:\COMET\runtime-data\scratch'
```

## 2. Obtain and verify the official archives

Download Mission 1 and/or Mission 2 from the [official v2 record](https://zenodo.org/records/15237121). COMET reads ZIPs directly. You can also use the [v1 record](https://zenodo.org/records/12528696) with `esa.version` set to `1`.

| Official v2 archive | MD5 |
| --- | --- |
| ESA-Mission1.zip | `9770ad12ed730238f37c42d5c27ab436` |
| ESA-Mission2.zip | `bfc72012691427d9327eb41f726ce45e` |

```powershell
Get-FileHash E:\COMET\runtime-data\downloads\ESA-Mission1.zip -Algorithm MD5
```

Verify the source first. Raw channels contain pickle data, and [loading pickle can execute code](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html). The ESA example manifests already enable the trusted flag, so use them only with verified official data or another source you explicitly trust.

## 3. Inspect and ingest

```powershell
comet inspect E:\COMET\runtime-data\downloads\ESA-Mission1.zip --manifest examples/esa-mission1-causal.json
comet ingest E:\COMET\runtime-data\downloads\ESA-Mission1.zip --manifest examples/esa-mission1-causal.json
```

For Mission 2, use its example manifest. COMET also accepts an extracted directory containing `channels/`, `labels.csv` and `anomaly_types.csv`, so you can reuse an existing extraction when importing a mission.

COMET decodes one channel per subprocess and saves each completed conversion so a retry can reuse it. Alignment runs on disk. The completed dataset records source checksums, dataset and adapter versions, the pinned upstream revision and preprocessing settings. Each channel must fit in decoder memory even when plenty of disk space is available.

The **causal** profile is the default. Mission 1 uses a 30-second grid, Mission 2 an 18-second grid, with past observations held only within the configured age limit. Labels cannot change telemetry values.

When all telemetry channels are out, COMET records a structural gap that breaks model windows even if telecommands continue to arrive, keeping outages visible to the model. A separate label artifact stores event IDs, channel mappings, categories and interval boundaries.

Mission 1 testing starts January 1, 2007. Mission 2 starts October 1, 2001. COMET trains on the earlier 85% of pre-test rows and uses the final 15% to calibrate the threshold, unless you supply an explicit split mapping.

The **ESA reference** profile runs offline. Copy a manifest and set `esa.profile` to `esa-reference` to apply revision `aeebcd9ecd3e7266d6d6a035a8081b3da83dfe33`, including annotation-aware resampling and upstream derivative conventions. COMET records corrections for empty channels, duplicate observations, unknown categories and duplicated floor boundaries. This profile can use future information. Do not describe it as causal operation.

Limit output with `esa.start` and `esa.end` ISO timestamps or select known channel names in `esa.channels`. Decoding still reads each selected channel completely. Channel selection also restricts associated annotations, so the resulting subset cannot represent a full benchmark run.

## 4. Train and score

Use the dataset and model IDs printed by each command.

```powershell
comet train --dataset-id 1 --config config/model_configs/default.yaml
comet detect --dataset-id 1 --model-id 1 --partition test
comet model-diagnostics --dataset-id 1 --model-id 1 --run-id RUN_ID
comet export-predictions --run-id RUN_ID --output mission1-predictions.csv
```

Start with a documented range/channel subset. Full archives contain long missions and many channels, and COMET rejects a selected feature if the training portion contains no usable values for it. Choose a period/channel contract that contains training observations, keeping test data out of fitted statistics. Window size counts rows. At Mission 1's cadence, 60 rows span 30 minutes.

Set `esa.version` for processed upstream CSVs. `inspect` recognizes the official `ESA-Mission1-semi-supervised` or `ESA-Mission2-semi-supervised` directory and its `*_months.train.csv` / prescribed test CSV pair. If the directory has several training periods, select one with `esa.processed_train`, for example `"3_months"`.

Keep the label columns. COMET stores `is_anomaly_CHANNEL` columns as separate annotations. Processed CSVs require the `esa-reference` profile, because their existing offline preprocessing cannot become causal. For a lone CSV, supply `esa.allow_single`, `esa.single_partition`, and explicit split boundaries. A trained model can score a separate compatible dataset when its manifest retains the same feature roles and source identities.

### Investigating excessive alerts

For incoming batches whose normal behavior changes over time, follow [operational monitoring](MONITORING.md). It updates models using earlier observations and records each scoring period separately.

A large reconstruction error is not automatically a software defect. Compare alerts with annotations and inspect feature contributions before changing the threshold. A channel with little training variation can dominate the squared error after a large change, including a genuine anomaly. A 95th-percentile validation threshold does not guarantee a 5% alert rate on later data.

The optional `config/model_configs/esa_robust.yaml` applies `asinh` after standardization, compressing numeric extremes without clipping them while retaining the training-only means/scales, missingness indicators and categorical encoding needed to reuse the transform. The encoding is reversible. Packet values are converted back to original units. Existing models and the standard default are unchanged.

```powershell
comet train --dataset-id 1 --config config/model_configs/esa_robust.yaml
```

Retrain and calibrate the threshold first. It trains on non-overlapping 60-row windows and scores every eligible detection window. Use the new model ID and save predictions under a new filename. Compare missed anomalies as well as false alerts, since compressing extremes can reduce sensitivity. Changes in operating conditions between training and test periods still need investigation.

## 5. Evaluate a recorded run

Select the run you want to evaluate. Its export contains every target timestamp in the selected partition, with no-alert warm-up and gap records carrying null scores. A detection run created with `--partition test` exports test rows only. Its complete score artifact retains outside-partition coverage records. A run created with `--partition all` exports all target rows, with the harness reporting the exported range and clipping policy.

Create a separate environment and check out the pinned ESA revision.

```powershell
python -m venv .esa-reference
.esa-reference\Scripts\python -m pip install numpy pandas pyarrow scikit-learn portion
git clone https://github.com/kplabs-pl/ESA-ADB.git ESA-ADB
git -C ESA-ADB checkout aeebcd9ecd3e7266d6d6a035a8081b3da83dfe33
comet evaluate --dataset-id 1 --model-id 1 --run-id RUN_ID --metric esa-reference --metadata ANNOTATIONS_PARQUET --esa-checkout ESA-ADB --reference-python .esa-reference\Scripts\python.exe
```

Find `ANNOTATIONS_PARQUET` in the ingestion metadata's `annotations_path`. The harness checks the metric checkout, then sends binary timestamp predictions directly to `ESAScores.score`, bypassing the benchmark runner's conversion of continuous scores to integers.

Undefined metrics are reported explicitly. Empty predictions, no alerts and absent selected ground truth never cause an artificial alert to be inserted. This harness supplies global evaluation only. Contribution scores do not identify channel detections. The workflow makes no claim of a particular accuracy or full ESA benchmark reproduction.

Use `--metric legacy-overlap` for historical COMET overlap diagnostics, with interval metadata in the legacy format (`start_time,end_time,root_cause,affected,category`). Report these results as COMET overlap diagnostics, which do not measure ESA benchmark performance.

## 6. Reproduce operational checks

```powershell
python scripts/validate_esa.py --archive E:\COMET\runtime-data\downloads\ESA-Mission1.zip --mission 1 --output-root E:\COMET\runtime-data\mission1-validation --trust-official-pickle
```

Repeat with Mission 2. The opt-in validator ingests the full archive and runs bounded training, held-out scoring and independent-target inference. It checks scores across batch sizes and exports predictions. Its reports record versions, archive/source checksum identities, upstream revision, preprocessing configuration, local source fingerprint, timings, sampled peak process-tree RSS and disk use.

Check the cache status in the report. A resumed or cached run cannot measure fresh raw decoding, and disk totals include retained caches and earlier attempts. These checks measure function and resource use, without benchmark accuracy results.
