"""Time PriceScope._fetch_bar_ohlc on a 100-name daily book.

Bars are the pinned daily_1.pkl sessions, scaled per symbol and loaded
through the flat store a backtest actually builds. The bar loop calls
fetch_bar_ohlc once per symbol per session from check_stops (a cache miss)
and again from capture_bar (a hit). Both calls are inside the timer.

Run from the repo root:

    python osspr-intake/harness/bench_fetch_bar_ohlc.py
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd

from pybroker.scope import (
    ColumnScope,
    PriceScope,
    symbol_array_store_from_flat_frame,
)

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "tests" / "testdata" / "daily_1.pkl"
N_SYMBOLS = 100
WARMUP = 1
REPEATS = 5


def _frame() -> pd.DataFrame:
    base = pd.read_pickle(DATA_PATH)  # noqa: S301 - pinned test fixture
    base["date"] = pd.to_datetime(base["date"])
    sources = list(base["symbol"].unique())
    frames: list[pd.DataFrame] = []
    for i in range(N_SYMBOLS):
        part = base.loc[base["symbol"] == sources[i % len(sources)]].copy()
        part["symbol"] = f"S{i:04d}"
        scale = 1.0 + (i % 11) * 0.01
        for col in ("open", "high", "low", "close", "adj_close"):
            part[col] = part[col] * scale
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def _run(price_scope: PriceScope, symbols: list[str], dates: np.ndarray) -> float:
    end_index = price_scope._sym_end_index
    total = 0.0
    for i, date in enumerate(dates):
        price_scope.reset_bar()
        bar_index = i + 1
        for symbol in symbols:
            end_index[symbol] = bar_index
            close, _low, _high = price_scope.fetch_bar_ohlc(symbol, date)
            # capture_bar reads the same bar after check_stops has filled
            # the per-bar cache.
            close_again, _, _ = price_scope.fetch_bar_ohlc(symbol, date)
            if close is None or close_again is None:
                raise SystemExit(f"missing bar for {symbol} at {date}")
            total += close
    return total


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    store = symbol_array_store_from_flat_frame(_frame())
    symbols = sorted(store.symbols)
    dates = store.sym_arrays[symbols[0]]["date"]
    end_index: dict[str, int] = defaultdict(int)
    price_scope = PriceScope(ColumnScope(store), end_index, True)
    first = _run(price_scope, symbols, dates)
    second = _run(price_scope, symbols, dates)
    if first != second:
        raise SystemExit("PriceScope.fetch_bar_ohlc is not deterministic")
    for _ in range(WARMUP):
        _run(price_scope, symbols, dates)
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        _run(price_scope, symbols, dates)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "PriceScope._fetch_bar_ohlc",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "samples_ms": samples_ms,
                "checksum": first,
                "n_symbols": len(symbols),
                "n_bars": int(len(dates)),
            }
        )
    )


if __name__ == "__main__":
    main()
