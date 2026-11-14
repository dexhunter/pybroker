"""Time PredictionScope._fetch_per_bar on a 20-name daily basket.

The bars are the pinned tests/testdata/daily_1.pkl sessions (SPY, AAPL,
TSLA, MSFT). Each extra symbol is one of those series scaled by a constant,
so every bar stays a valid OHLC row. A per-bar model reads the last close
of the history prefix, which is what fetch() does once per symbol when a
backtest asks for predictions through the last bar.

Run from the repo root:

    python osspr-intake/harness/bench_fetch_per_bar.py
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

import pybroker
from pybroker.common import ModelSymbol, TrainedModel
from pybroker.model import model
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    PredictionScope,
    StaticScope,
)

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "tests" / "testdata" / "daily_1.pkl"
N_SYMBOLS = 20
WARMUP = 1
REPEATS = 5


def _predict_fn(_model: object, data: pd.DataFrame) -> float:
    return float(data["close"].iloc[-1])


def _universe() -> pd.DataFrame:
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


def _scopes(
    df: pd.DataFrame,
) -> list[tuple[str, PredictionScope, int]]:
    StaticScope.instance()._model_sources.clear()
    model(
        "pb",
        lambda _symbol, _train, _test: (object(), ("close",)),
        predict_fn=_predict_fn,
        per_bar=True,
    )
    scopes: list[tuple[str, PredictionScope, int]] = []
    for symbol in sorted(df["symbol"].unique()):
        sym_df = df[df["symbol"] == symbol].set_index(["symbol", "date"])
        dates = list(sym_df.index.get_level_values(1))
        trained = {
            ModelSymbol("pb", symbol): TrainedModel(
                name="pb",
                instance=object(),
                predict_fn=_predict_fn,
                input_cols=("close",),
                per_bar=True,
            )
        }
        col_scope = ColumnScope(sym_df)
        ind_scope = IndicatorScope({}, dates)
        input_scope = ModelInputScope(
            col_scope, ind_scope, trained, {}, dates
        )
        scopes.append(
            (symbol, PredictionScope(trained, input_scope), len(dates))
        )
    return scopes


def _run(scopes: list[tuple[str, PredictionScope, int]]) -> np.ndarray:
    heads = np.empty(len(scopes), dtype=np.float64)
    for i, (symbol, pred_scope, n_bars) in enumerate(scopes):
        pred_scope._per_bar_preds.clear()
        preds = pred_scope.fetch(symbol, "pb", n_bars)
        heads[i] = float(preds[0] + preds[-1])
    return heads


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    pybroker.disable_logging()
    pybroker.disable_progress_bar()
    scopes = _scopes(_universe())
    first = _run(scopes)
    second = _run(scopes)
    if not np.array_equal(first, second):
        raise SystemExit("PredictionScope._fetch_per_bar is not deterministic")
    for _ in range(WARMUP):
        _run(scopes)
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        _run(scopes)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "PredictionScope._fetch_per_bar",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "samples_ms": samples_ms,
                "checksum": float(first.sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
