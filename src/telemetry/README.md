# src/telemetry

`types.py` preserves the original `TelemetryRecord` and `TelemetrySource` exports. `csv_adapter.py` preserves the CSV row conversion helpers for Python integrations.

The current ingestion pipeline uses the versioned manifest and batch adapters in `src/ingestion/` to write canonical Parquet datasets. These compatibility helpers do not replace manifest validation.
