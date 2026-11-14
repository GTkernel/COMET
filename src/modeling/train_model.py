from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass

from src.common.paths import MODELS_DIR, MODEL_CONFIG_DIR
from src.modeling.pipeline import DEFAULT_CONFIG

DEFAULT_TRAINING_CONFIG_PATH = MODEL_CONFIG_DIR / "default.yaml"
DEFAULT_TRAINING_CONFIG = DEFAULT_CONFIG.copy()


@dataclass(frozen=True)
class TrainingData:
    feature_columns: list[str]
    values: Any


def resolve_training_config_path(config_path: str | Path | None = None) -> Path:
    return Path(config_path) if config_path is not None else DEFAULT_TRAINING_CONFIG_PATH


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    from src.modeling.pipeline import _load_config
    return _load_config(config_path) if Path(config_path).exists() else _load_config(defaults=DEFAULT_TRAINING_CONFIG)


def train_model(
    dataset_id: int,
    config_path: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    """Train with persisted, training-only preprocessing, including legacy raw readers."""
    from src.modeling.pipeline import train

    return train(dataset_id, resolve_training_config_path(config_path), db_path=db_path,
                 storage_dir=Path(models_dir).parent, models_dir=models_dir)
