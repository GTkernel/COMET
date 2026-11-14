from __future__ import annotations

from pathlib import Path
import os

# Runtime artifacts can live outside the source checkout.
V2_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = Path(os.environ.get("COMET_STORAGE_DIR", V2_ROOT)).expanduser().resolve()
DATA_DIR = STORAGE_ROOT / "data"
DB_DIR = STORAGE_ROOT / "db"
DB_PATH = DB_DIR / "satellite_operator.db"
MODELS_DIR = STORAGE_ROOT / "models"
CONFIG_DIR = V2_ROOT / "config"
GENERATED_CONFIG_DIR = STORAGE_ROOT / "config" / "generated"
MODEL_CONFIG_DIR = CONFIG_DIR / "model_configs"
RAG_DIR = STORAGE_ROOT / "rag"
RAG_DOCS_DIR = RAG_DIR / "documents"
RAG_VECTOR_DIR = RAG_DIR / "vector_store"
LOGS_DIR = STORAGE_ROOT / "logs"
DIAGNOSTICS_DIR = STORAGE_ROOT / "diagnostics"
EVALUATION_DIR = STORAGE_ROOT / "evaluation"
DATASETS_DIR = STORAGE_ROOT / "datasets"
SCRATCH_DIR = Path(os.environ.get("COMET_SCRATCH_DIR", STORAGE_ROOT / "scratch")).expanduser().resolve()


def ensure_base_directories() -> None:
    for p in [
        DATA_DIR,
        DB_DIR,
        MODELS_DIR,
        GENERATED_CONFIG_DIR,
        MODEL_CONFIG_DIR,
        RAG_DOCS_DIR,
        RAG_VECTOR_DIR,
        LOGS_DIR,
        DIAGNOSTICS_DIR,
        EVALUATION_DIR,
        DATASETS_DIR,
        SCRATCH_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
