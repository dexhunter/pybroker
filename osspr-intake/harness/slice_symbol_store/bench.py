"""Time ``slice_symbol_array_store_by_dates`` on one history window.

120 symbols x 2,520 business days. Each sample slices the master
store onto the same 2,000-date history window five times, which is
what a short walkforward does when it builds a history store per
window. One slice is only a couple of milliseconds and sits in
timer noise; the block does not.
"""

from __future__ import annotations

import gc
import os
import statistics
import time

import numpy as np
import pandas as pd

from pybroker.scope import (
    slice_symbol_array_store_by_dates,
    symbol_array_store_from_frame,
)

REPEATS = 31
WARMUP = 4
N_SYMBOLS = 120
N_DAYS = 2520
WINDOW = 2000
SLICES_PER_SAMPLE = 5


def _ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-04", periods=N_DAYS, freq="B")
    frames: list[pd.DataFrame] = []
    for i in range(N_SYMBOLS):
        price = 80.0 + rng.standard_normal(N_DAYS).cumsum() * 0.4
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": np.full(N_DAYS, 1_000_000.0),
                    "symbol": f"SYM{i:03d}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _slice_block(store: object, selected: np.ndarray) -> None:
    for _ in range(SLICES_PER_SAMPLE):
        slice_symbol_array_store_by_dates(store, selected)


def main() -> None:
    os.sched_setaffinity(0, {0})
    df = _ohlcv()
    store = symbol_array_store_from_frame(df)
    dates = np.unique(df["date"].to_numpy(dtype="datetime64[ns]"))
    selected = dates[:WINDOW]
    for _ in range(WARMUP):
        _slice_block(store, selected)
    gc.collect()
    samples: list[float] = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        _slice_block(store, selected)
        samples.append((time.perf_counter() - t0) * 1000.0)
    print(f"median_ms={statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
