"""Peak traced memory of _dataframe_records on a long portfolio frame.

Run from the repo root:

    python3 osspr-intake/harness/peak_dataframe_records.py
"""

from __future__ import annotations

import gc
import json
import os
import tracemalloc

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from pybroker.common import _dataframe_records

N_ROWS = 60_000
REPEATS = 5


def _portfolio() -> pd.DataFrame:
    """Float portfolio frame of the shape ``_to_test_result`` hands to JSON.

    Quantize has already turned Decimal money columns into float64. The index
    is the bar date, which ``_dataframe_records`` materializes because it is
    named. A NaN and an infinity sit on ``pnl`` so those JSON branches run.
    """
    dates = pd.date_range("2016-01-04 09:30", periods=N_ROWS, freq="min")
    level = np.linspace(100.0, 180.0, N_ROWS)
    frame = pd.DataFrame(
        {
            "cash": level,
            "equity": level * 1.01,
            "margin": np.zeros(N_ROWS),
            "margin_loan": np.zeros(N_ROWS),
            "net_cash_balance": level,
            "market_value": level * 1.02,
            "pnl": np.linspace(-5.0, 40.0, N_ROWS),
            "unrealized_pnl": np.linspace(-1.0, 4.0, N_ROWS),
            "fees": np.full(N_ROWS, 0.01),
            "symbol": np.resize(
                np.array([f"S{i:02d}" for i in range(30)]), N_ROWS
            ),
        },
        index=pd.Index(dates, name="date"),
    )
    frame.iloc[10, frame.columns.get_loc("pnl")] = np.nan
    frame.iloc[11, frame.columns.get_loc("pnl")] = np.inf
    frame.iloc[12, frame.columns.get_loc("pnl")] = -np.inf
    return frame


def _peak_bytes(frame: pd.DataFrame) -> int:
    gc.collect()
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    try:
        _dataframe_records(frame, max_rows=None)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    frame = _portfolio()
    first = _dataframe_records(frame, max_rows=None)
    second = _dataframe_records(frame, max_rows=None)
    if first != second:
        raise SystemExit("_dataframe_records is not deterministic")
    samples = [_peak_bytes(frame) for _ in range(REPEATS)]
    median = float(np.median(samples))
    noise = (max(samples) - min(samples)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "_dataframe_records",
                "metric_kind": "peak_memory_bytes",
                "samples": samples,
                "median": median,
                "noise_pct": round(noise, 4),
                "rows": N_ROWS,
            }
        )
    )


if __name__ == "__main__":
    main()
