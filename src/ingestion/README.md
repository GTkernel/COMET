# src/ingestion

`adapters.py` reads structured formats, `manifest.py` resolves their schema, and `store.py` validates and publishes immutable Parquet parts with a SQLite catalog. `esa.py` handles raw and processed ESA layouts. Raw channel decoding runs through `esa_decode.py` in subprocesses.

For `python -m src.cli ingest` without an input path, `csv_ingestion.py` imports CSVs from `data/` through the same canonical store. Existing legacy tables remain readable elsewhere.
