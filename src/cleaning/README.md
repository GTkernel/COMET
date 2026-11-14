# src/cleaning

`prepare_dataset.py` selects the preparation path for the dataset's storage type. For canonical Parquet data, it records partition and gap boundaries. Training fits preprocessing later.

Edit the generated YAML under `config/generated/` or pass a custom file with `--feature-config`. Both SQLite and Parquet datasets accept feature-role choices. Legacy SQLite preparation builds model-ready and context-ready tables. Parquet preparation records those choices while leaving observations in their original files.
