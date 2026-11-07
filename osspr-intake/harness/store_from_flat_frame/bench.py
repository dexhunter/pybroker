"""Time ``symbol_array_store_from_flat_frame`` on a flat OHLCV frame.

50 symbols x 1,260 business days (about five years of daily bars) is a
plain multi-symbol backtest input. The frame is built once; each sample
calls the store builder on that same frame.
"""

from __future__ import annotations

import gc
import os
import statistics
import time

import numpy as np
import pandas as pd

from pybroker.scope import symbol_array_store_from_flat_frame

REPEATS = 31
WARMUP = 8
N_SYMBOLS = 50
N_DAYS = 1260


def _ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-02", periods=N_DAYS, freq="B")
    frames: list[pd.DataFrame] = []
    for i in range(N_SYMBOLS):
        price = 100.0 + rng.standard_normal(N_DAYS).cumsum() * 0.5
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "open": price + rng.uniform(-0.5, 0.5, N_DAYS),
                    "high": price + rng.uniform(0.1, 1.0, N_DAYS),
                    "low": price - rng.uniform(0.1, 1.0, N_DAYS),
                    "close": price,
                    "volume": rng.integers(1_000, 100_000, N_DAYS).astype(
                        np.float64
                    ),
                    "symbol": f"SYM{i:03d}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    os.sched_setaffinity(0, {0})
    df = _ohlcv()
    for _ in range(WARMUP):
        symbol_array_store_from_flat_frame(df)
    gc.collect()
    samples: list[float] = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        symbol_array_store_from_flat_frame(df)
        samples.append((time.perf_counter() - t0) * 1000.0)
    print(f"median_ms={statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
