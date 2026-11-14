"""Batched LSTM reconstruction and exact in-place disk percentile calibration."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np


def score_batches(model, windows, batch_size: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    import torch
    from torch.utils.data import DataLoader
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    offset = 0
    with torch.inference_mode():
        for batch in DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=0):
            errors = (model(batch) - batch).square()
            scores = errors.mean(dim=(1, 2)).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("Model produced nonfinite reconstruction errors")
            yield offset, scores, errors.mean(dim=1).cpu().numpy()
            offset += len(batch)


def calibrate(model, windows, path: Path, *, percentile: float, batch_size: int) -> tuple[float, dict]:
    if not len(windows):
        raise ValueError("Validation partition has no complete windows. Reduce window_size or revise splits")
    scores = np.lib.format.open_memmap(path, mode="w+", dtype="float64", shape=(len(windows),))
    total, minimum, maximum = 0.0, float("inf"), -float("inf")
    for offset, values, _ in score_batches(model, windows, batch_size):
        scores[offset:offset + len(values)] = values
        total += float(values.sum(dtype=np.float64))
        minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
    rank = float(percentile) / 100 * (len(scores) - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    # ndarray.partition uses this mmap in place without an N-sized work array.
    # np.percentile would copy the scores into memory by default.
    scores.partition((lower, upper))
    threshold = float(scores[lower] + (scores[upper] - scores[lower]) * (rank - lower))
    scores.flush()
    return threshold, {"count": len(scores), "min": minimum, "max": maximum, "mean": total / len(scores)}
