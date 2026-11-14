"""Time _has_pending_order across a 200-name rotational book.

One pass per session of daily_1.pkl, both order sides, matching the calls
_rotation_candidates and _rotation_exits make for every active symbol.
Ten names hold an in-flight buy, as a rotation book does after the first
entries; the rest have an empty pending set.

Run from the repo root:

    python osspr-intake/harness/bench_has_pending_order.py
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

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from pybroker.common import PriceType
from pybroker.scope import PendingOrderScope
from pybroker.strategy import _has_pending_order

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "tests" / "testdata" / "daily_1.pkl"
N_SYMBOLS = 200
N_PENDING = 10
WARMUP = 1
REPEATS = 5


class _Ctx:
    """Same pending_orders path ExecContext uses, without a full backtest."""

    def __init__(self, symbol: str, scope: PendingOrderScope) -> None:
        self.symbol = symbol
        self._pending_order_scope = scope

    def pending_orders(self, symbol: str | None = None):
        yield from self._pending_order_scope.orders(symbol)


def _setup() -> tuple[list[_Ctx], int]:
    base = pd.read_pickle(DATA_PATH)  # noqa: S301 - pinned test fixture
    n_bars = int(base.loc[base["symbol"] == base["symbol"].iloc[0]].shape[0])
    scope = PendingOrderScope()
    created = np.datetime64("2021-06-01")
    exec_date = np.datetime64("2021-06-02")
    for i in range(N_PENDING):
        scope.add(
            "buy",
            f"S{i:04d}",
            created,
            exec_date,
            Decimal(100),
            None,
            PriceType.CLOSE,
            1,
            None,
        )
    ctxs = [_Ctx(f"S{i:04d}", scope) for i in range(N_SYMBOLS)]
    return ctxs, n_bars


def _run(ctxs: list[_Ctx], n_bars: int) -> int:
    hits = 0
    for _ in range(n_bars):
        for ctx in ctxs:
            if _has_pending_order(ctx, "buy"):
                hits += 1
            if _has_pending_order(ctx, "sell"):
                hits += 1
    return hits


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    ctxs, n_bars = _setup()
    first = _run(ctxs, n_bars)
    second = _run(ctxs, n_bars)
    if first != second:
        raise SystemExit("_has_pending_order is not deterministic")
    for _ in range(WARMUP):
        _run(ctxs, n_bars)
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        _run(ctxs, n_bars)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "_has_pending_order",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "samples_ms": samples_ms,
                "hits": first,
                "n_bars": n_bars,
            }
        )
    )


if __name__ == "__main__":
    main()
