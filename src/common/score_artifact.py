"""Validate complete score coverage before publishing derived reports."""
from __future__ import annotations

from typing import Mapping


def validate_score_counts(run: dict, row_count: int, scored_count: int,
                          coverage: Mapping[str, int]) -> None:
    """Compare all source rows, including those outside an exported partition."""
    if run.get("row_count") is not None and row_count != int(run["row_count"]):
        raise ValueError("Incomplete score artifact: row count differs from the completed run")
    if run.get("windows_scored") is not None and scored_count != int(run["windows_scored"]):
        raise ValueError("Inconsistent score artifact: scored count differs from the completed run")
    if run.get("coverage") is not None:
        expected = {name: int(count) for name, count in run["coverage"].items() if count}
        actual = {name: int(count) for name, count in coverage.items() if count}
        if actual != expected:
            raise ValueError("Inconsistent score artifact: coverage differs from the completed run")
