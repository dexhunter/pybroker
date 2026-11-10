"""Time cold IndicatorScope.fetch masking long minute series.

Run from the repo root:

    python3 osspr-intake/harness/bench_indicator_scope_fetch.py
"""

from __future__ import annotations

import gc
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from pybroker.common import IndicatorSymbol
from pybroker.scope import IndicatorScope

N_BARS = 160_000
N_SYMBOLS = 48
INDICATOR = "sma"
WARMUP = 1
REPEATS = 5


def _inputs() -> tuple[dict[IndicatorSymbol, pd.Series], np.ndarray, tuple[str, ...]]:
    dates = pd.date_range("2004-01-05 09:30", periods=N_BARS, freq="min")
    # A walkforward window keeps a sorted subset of the history dates.
    filter_dates = dates[::4].to_numpy(dtype="datetime64[ns]")
    rng = np.random.default_rng(2)
    symbols = tuple(f"S{i:02d}" for i in range(N_SYMBOLS))
    series: dict[IndicatorSymbol, pd.Series] = {}
    for symbol in symbols:
        series[IndicatorSymbol(INDICATOR, symbol)] = pd.Series(
            rng.normal(size=N_BARS), index=dates
        )
    return series, filter_dates, symbols


def _run(
    series: dict[IndicatorSymbol, pd.Series],
    filter_dates: np.ndarray,
    symbols: tuple[str, ...],
) -> np.ndarray:
    scope = IndicatorScope(series, filter_dates)
    # Touch the first and last masked values so the reads stay live.
    heads = np.empty(len(symbols), dtype=np.float64)
    for i, symbol in enumerate(symbols):
        values = scope.fetch(symbol, INDICATOR)
        heads[i] = values[0] + values[-1]
    return heads


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    series, filter_dates, symbols = _inputs()
    first = _run(series, filter_dates, symbols)
    second = _run(series, filter_dates, symbols)
    if not np.array_equal(first, second, equal_nan=True):
        raise SystemExit("IndicatorScope.fetch is not deterministic")
    for _ in range(WARMUP):
        _run(series, filter_dates, symbols)
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        _run(series, filter_dates, symbols)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "fetch",
                "samples_ms": [round(v, 4) for v in samples_ms],
                "median_ms": round(median, 4),
                "noise_pct": round(noise, 4),
                "bars": N_BARS,
                "symbols": N_SYMBOLS,
            }
        )
    )


if __name__ == "__main__":
    main()
