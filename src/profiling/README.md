# src/profiling

`dataset_profile.py` dispatches profiling by storage type. Canonical Parquet datasets use bounded descriptive statistics and retain feature roles from their manifest.

Profiling writes editable feature YAML under `config/generated/` for SQLite and Parquet datasets. Pass it to `prepare --feature-config PATH` after changing modeling or context roles. Changes to source mappings require an updated manifest and a new import.
