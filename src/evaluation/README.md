# src/evaluation

`anomaly_evaluation.py` compares recorded runs with known anomaly intervals. `export.py` writes timestamp-aligned predictions, and `reference.py` calls the pinned ESA global metric in a separate environment.

For legacy interval metadata, run `python -m src.cli evaluate --dataset-id DATASET_ID --model-id MODEL_ID --run-id RUN_ID --metric legacy-overlap --metadata path/to/metadata.csv` with the IDs you want to evaluate. The [ESA walkthrough](../../docs/ESA.md) covers reference evaluation and its annotation format.
