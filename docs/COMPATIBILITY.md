# Compatibility with original COMET workflows

The original eleven CLI commands remain available, including their options and the public telemetry, configuration, CSV and database helpers used by existing Python integrations.

## Profile, select features, train

1. Import data with `comet ingest INPUT`, or use the original no-argument CSV import workflow.
2. Run `comet profile --dataset-id ID`. The returned path identifies editable feature YAML.
3. Edit column roles in that file. Numeric, categorical, context and excluded choices are supported. Annotation columns cannot become telemetry or diagnostic context.
4. Run `comet prepare --dataset-id ID --feature-config PATH`.
5. Edit `config/model_configs/default.yaml`, or supply `comet train --dataset-id ID --config PATH`.
6. Detect, select the resulting run, then run diagnostics or build anomaly packets as before.

SQLite preparation creates the original model/context tables. Parquet preparation saves feature choices and boundaries separately. New training reads the selected raw features and fits preprocessing on training rows only. Scaling in legacy prepared tables is reserved for historical checkpoints.

The no-argument CSV workflow accepts rows without timestamps. These imports record `ordering: row` and preserve file order with real timestamps left empty. For an explicit import, supply `{"ordering": "row"}` in its manifest. Fractional partitions and row-based detection work with row order. Calendar-based monitoring and interval annotations require real timestamps. Ambiguous timestamp columns still require an explicit mapping.

## Output and diagnostics

Original commands print their original human-readable output. For script output, add `--json`. New commands retain JSON by default.

Diagnostics include the error histogram, error time series, top-feature chart, feature heatmap and error quantiles, plus median and standard deviation. Select a run first. Large plots use samples identified in their captions, while statistics and feature rankings cover every scored window. A feature contribution explains reconstruction error. It cannot establish a channel detection.

Packets and diagnosis context also work with legacy prepared-only catalogs. Their summaries identify prepared values when raw observations and original units are unavailable. Annotation names and encoded annotation columns remain excluded.

## Deliberate correctness changes

COMET partitions rows chronologically before fitting preprocessing or creating windows. It calibrates new model thresholds on validation scores and timestamps alerts at the window end. Exports include unscored warm-up rows, and each run keeps its own anomalies. Training and evaluation results can change. Historical overlap evaluation and ESA reference evaluation remain separate choices.

Replace the old `train_test_split` setting with `splits.fractions` in the manifest. COMET warns if the old setting is present. Generic defaults are 70% training, 15% validation and 15% test.

See [migration guidance](MIGRATION.md) for catalog, checkpoint and storage details.
