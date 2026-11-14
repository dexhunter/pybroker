"""Time YFinance._fetch_data's wide-to-long reshape on 1,000 names.

The price block is tests/testdata/yfinance.pkl (the real download shape:
Date index, MultiIndex columns of field x ticker). Tickers are tiled from
those four series so the frame matches what yfinance.download returns for a
large watchlist. download itself is stubbed; the timed call is _fetch_data.

Run from the repo root:

    python osspr-intake/harness/bench_yfinance_fetch.py
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

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance

import pybroker
from pybroker.data import YFinance

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "tests" / "testdata" / "yfinance.pkl"
N_SYMBOLS = 1000
WARMUP = 1
REPEATS = 5
START = datetime(2020, 1, 2)
END = datetime(2021, 12, 31)


def _wide_frame() -> tuple[pd.DataFrame, frozenset[str]]:
    fixture = pd.read_pickle(DATA_PATH)  # noqa: S301 - pinned test fixture
    fields = list(fixture.columns.get_level_values(0).unique())
    sources = list(fixture.columns.get_level_values(1).unique())
    data: dict[tuple[str, str], np.ndarray] = {}
    symbols: list[str] = []
    for i in range(N_SYMBOLS):
        source = sources[i % len(sources)]
        symbol = f"S{i:04d}"
        symbols.append(symbol)
        for field in fields:
            data[(field, symbol)] = fixture[(field, source)].to_numpy()
    frame = pd.DataFrame(
        data,
        index=fixture.index,
        columns=pd.MultiIndex.from_tuples(data.keys()),
    )
    return frame, frozenset(symbols)


def _checksum(df: pd.DataFrame) -> tuple[int, float, float]:
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    return len(df), float(close[0]), float(close[-1])


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    pybroker.disable_logging()
    pybroker.disable_progress_bar()
    wide, symbols = _wide_frame()
    yfinance.download = lambda *_args, **_kwargs: wide  # type: ignore[assignment]
    source = YFinance(auto_adjust=False)

    def run() -> pd.DataFrame:
        return source._fetch_data(symbols, START, END, "1d", None)

    first = run()
    second = run()
    if not first.equals(second):
        raise SystemExit("YFinance._fetch_data is not deterministic")
    for _ in range(WARMUP):
        run()
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        run()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    n_rows, close_0, close_n = _checksum(first)
    print(
        json.dumps(
            {
                "function": "YFinance._fetch_data",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "samples_ms": samples_ms,
                "n_rows": n_rows,
                "close_0": close_0,
                "close_n": close_n,
            }
        )
    )


if __name__ == "__main__":
    main()
