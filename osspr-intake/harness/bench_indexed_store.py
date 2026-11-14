"""Time symbol_array_store_from_indexed_df on a 500-name MultiIndex book.

Rows are the pinned daily_1.pkl sessions, one scaled copy per symbol, then
indexed by (symbol, date) the way ColumnScope accepts a legacy MultiIndex
frame. The store build is the whole timed call.

Run from the repo root:

    python osspr-intake/harness/bench_indexed_store.py
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

from pathlib import Path

import numpy as np
import pandas as pd

from pybroker.scope import symbol_array_store_from_indexed_df

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "tests" / "testdata" / "daily_1.pkl"
N_SYMBOLS = 500
WARMUP = 1
REPEATS = 5


def _indexed_frame() -> pd.DataFrame:
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
    flat = pd.concat(frames, ignore_index=True)
    return flat.set_index(["symbol", "date"]).sort_index()


def _checksum(store: object) -> float:
    total = 0.0
    symbols = sorted(store.symbols)  # type: ignore[attr-defined]
    # First, middle, and last symbol so the read stays cheap and live.
    picks = [symbols[0], symbols[len(symbols) // 2], symbols[-1]]
    for symbol in picks:
        close = store.sym_arrays[symbol]["close"]  # type: ignore[attr-defined]
        total += float(close[0] + close[-1])
    return total


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    frame = _indexed_frame()
    first = symbol_array_store_from_indexed_df(frame)
    second = symbol_array_store_from_indexed_df(frame)
    if _checksum(first) != _checksum(second) or first.symbols != second.symbols:
        raise SystemExit(
            "symbol_array_store_from_indexed_df is not deterministic"
        )
    for _ in range(WARMUP):
        symbol_array_store_from_indexed_df(frame)
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        symbol_array_store_from_indexed_df(frame)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "symbol_array_store_from_indexed_df",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "samples_ms": samples_ms,
                "n_symbols": len(first.symbols),
                "checksum": _checksum(first),
            }
        )
    )


if __name__ == "__main__":
    main()
