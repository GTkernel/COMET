# data

Put telemetry CSVs here for no-argument ingestion. `python -m src.cli ingest` reads every `.csv` file directly under this directory, which can contain CATs telemetry or real mission telemetry.

Keep mission data out of commits. With a custom storage root, use its `data/` directory. To import another supported format or location, pass the input path explicitly.
