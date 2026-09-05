"""Pipeline runtime benchmarking and bias-correction performance summary."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


class StageTimer:
    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def time(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage_name] = time.perf_counter() - start

    def as_dict(self) -> Dict[str, float]:
        return dict(self.timings)


def summarize_correction_performance(
    benchmark_beta: float, naive_beta: float, corrected_beta: float
) -> Dict[str, float]:
    """How much of the naive model's gap to the benchmark the correction
    actually closed. ``benchmark_beta`` is the real-data (D_true) OLS
    estimate - an empirical benchmark with its own sampling error, not a
    known ground truth (see bias_correction.py's module docstring)."""
    naive_gap = abs(naive_beta - benchmark_beta)
    corrected_gap = abs(corrected_beta - benchmark_beta)
    gap_closed_pct = (1 - corrected_gap / naive_gap) * 100.0 if naive_gap > 0 else float("nan")
    return {
        "attenuation_pct": (
            (1 - naive_beta / benchmark_beta) * 100.0 if benchmark_beta != 0 else float("nan")
        ),
        "naive_gap_to_benchmark": naive_gap,
        "corrected_gap_to_benchmark": corrected_gap,
        "gap_closed_pct": gap_closed_pct,
    }
