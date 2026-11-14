# src/cli

Run commands with `python -m src.cli`. `__main__.py` defines them and dispatches the operations, which validate their own inputs.

The original commands print readable summaries. `list-anomalies` prints tab-separated rows, and `anomaly-packet` prints JSON. Add `--json` before or after the command for machine-readable output, for example `python -m src.cli list-anomalies --json`. New commands (`inspect`, `run`, `monitor`, `export-predictions`) and an explicit-path `ingest` print JSON by default.
