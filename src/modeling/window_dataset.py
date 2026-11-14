"""Keep windows on disk and load their values a batch at a time."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.modeling.transforms import PARTITIONS


def build_window_index(rows: np.ndarray, path: Path, *, window_size: int, stride: int = 1,
                       partition: str = "all") -> np.ndarray:
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    if partition not in {*PARTITIONS, "all"}:
        raise ValueError("partition must be all, train, validation, or test")
    # Upper bound costs eight bytes per input row, independent of window width.
    starts = np.lib.format.open_memmap(path, mode="w+", dtype="int64", shape=(len(rows),))
    count, segment_start, previous_segment = 0, 0, None
    for begin in range(0, len(rows), 10000):
        for local, row in enumerate(rows[begin:begin + 10000]):
            index = begin + local
            unavailable = bool(row["excluded"] or row["gap"])
            if row["segment"] != previous_segment or unavailable:
                segment_start = index + int(unavailable)
            previous_segment = row["segment"]
            if unavailable or (partition != "all" and row["partition"] != PARTITIONS[partition]):
                continue
            start = index - window_size + 1
            if start >= segment_start and (start - segment_start) % stride == 0:
                starts[count] = start
                count += 1
    starts.flush()
    return starts[:count]


class WindowDataset:
    """PyTorch map-style dataset without importing torch until a sample is read."""

    def __init__(self, values: np.ndarray, starts: np.ndarray, window_size: int,
                 *, numeric_offsets: list[int] | None = None):
        self.values, self.starts, self.window_size = values, starts, window_size
        self.numeric_offsets = numeric_offsets or []

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        import torch
        start = int(self.starts[index])
        # A writable, isolated window prevents torch from mutating a readonly map.
        window = torch.from_numpy(np.asarray(self.values[start:start + self.window_size], dtype=np.float32).copy())
        if self.numeric_offsets:
            window[:, self.numeric_offsets] -= window[0, self.numeric_offsets]
        return window
