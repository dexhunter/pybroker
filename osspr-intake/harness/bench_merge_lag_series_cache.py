"""Time merge_lag_series_cache on a shuffled multi-symbol history.

Run from the repo root:

    python3 osspr-intake/harness/bench_merge_lag_series_cache.py
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

from pybroker.model import merge_lag_series_cache

N_SYMBOLS = 250
N_DAYS = 1000
LAGS = 8
COLUMNS = ("open", "high", "low", "close", "volume")
WARMUP = 2
REPEATS = 5


def _history() -> tuple[pd.DataFrame, tuple[str, ...]]:
    dates = pd.date_range("2012-01-02", periods=N_DAYS, freq="B")
    rng = np.random.default_rng(0)
    frames: list[pd.DataFrame] = []
    for i in range(N_SYMBOLS):
        close = 100.0 + rng.normal(0.0, 1.0, N_DAYS).cumsum()
        frames.append(
            pd.DataFrame(
                {
                    "symbol": f"S{i:03d}",
                    "date": dates,
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": np.abs(rng.normal(1.0e6, 1.0e4, N_DAYS)),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    symbols = tuple(f"S{i:03d}" for i in range(N_SYMBOLS))
    return df, symbols


def _run(df: pd.DataFrame, symbols: tuple[str, ...]):
    return merge_lag_series_cache({}, df, symbols, COLUMNS, LAGS)


def _fingerprint(cache: dict) -> tuple:
    # LagSeriesKey is a tuple; sort so the check does not depend on dict order.
    items = []
    for key in sorted(cache, key=repr):
        values = np.ascontiguousarray(cache[key])
        items.append(
            (
                repr(key),
                values.shape,
                float(np.nansum(values)),
                int(np.isnan(values).sum()),
            )
        )
    return tuple(items)


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    df, symbols = _history()
    first = _fingerprint(_run(df, symbols))
    second = _fingerprint(_run(df, symbols))
    if first != second:
        raise SystemExit("merge_lag_series_cache is not deterministic")
    for _ in range(WARMUP):
        _run(df, symbols)
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        _run(df, symbols)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "merge_lag_series_cache",
                "samples_ms": [round(v, 4) for v in samples_ms],
                "median_ms": round(median, 4),
                "noise_pct": round(noise, 4),
                "rows": int(len(df)),
                "symbols": N_SYMBOLS,
                "lags": LAGS,
            }
        )
    )


if __name__ == "__main__":
    main()
