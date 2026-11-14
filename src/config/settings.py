from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.common.paths import CONFIG_DIR

DEFAULTS: dict[str, Any] = {
    "thresholds": {
        "model_missing_max_ratio": 0.40,
        "context_missing_max_ratio": 0.80,
        "categorical_unique_max": 20,
        "categorical_unique_ratio_max": 0.05,
    },
    "lstm": {
        "window_size": 100,
        "layers": 2,
        "hidden_size": 64,
        "train_split": 0.8,
    },
    "rag": {
        "llm_provider": "ollama",
        "llm_model": "llama3.2",
        "embedding_model": "nomic-embed-text",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
    },
}


def load_app_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or (CONFIG_DIR / "app_config.json")
    if not config_path.exists():
        return deepcopy(DEFAULTS)

    with config_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    merged = deepcopy(DEFAULTS)
    for key, value in parsed.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
