# src/rag

`document_ingestion.py` indexes mission documents. `semantic.py` retrieves excerpts, and `diagnosis.py` asks the local model to select evidence. `evidence.py` validates those selections and renders the report. Python reads telemetry through `telemetry_tools.py` before inference. Annotation labels stay out of telemetry context and prompts.
