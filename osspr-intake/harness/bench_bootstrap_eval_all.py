"""Time bootstrap_eval_all on one month of 1-minute bar changes.

Run from the repo root:

    python3 osspr-intake/harness/bench_bootstrap_eval_all.py
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

from pybroker.eval import _seed_bootstrap, bootstrap_eval_all

N_BARS = 20 * 390
N_BOOT = 10_000
BARS_PER_YEAR = 252 * 390
SEED = 42
WARMUP = 1
REPEATS = 5


def _series() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    changes = np.ascontiguousarray(rng.normal(0.05, 2.0, N_BARS))
    returns = np.ascontiguousarray(changes / 100_000.0)
    return changes, returns


def _run(changes: np.ndarray, returns: np.ndarray):
    _seed_bootstrap(SEED)
    return bootstrap_eval_all(changes, returns, N_BOOT, BARS_PER_YEAR)


def _fingerprint(result) -> tuple:
    log_pf, sharpe, drawdown = result
    return (
        tuple(float(v) for v in log_pf),
        tuple(float(v) for v in sharpe),
        tuple(float(v) for v in drawdown.confs),
        tuple(float(v) for v in drawdown.pct_confs),
    )


def main() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})
    changes, returns = _series()
    # Compile before the timed samples. The tiny call still reseeds.
    _seed_bootstrap(SEED)
    bootstrap_eval_all(changes, returns, 50, BARS_PER_YEAR)
    first = _fingerprint(_run(changes, returns))
    second = _fingerprint(_run(changes, returns))
    if first != second:
        raise SystemExit("bootstrap_eval_all is not deterministic")
    for _ in range(WARMUP):
        _run(changes, returns)
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        _run(changes, returns)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(samples_ms))
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(
        json.dumps(
            {
                "function": "bootstrap_eval_all",
                "samples_ms": [round(v, 4) for v in samples_ms],
                "median_ms": round(median, 4),
                "noise_pct": round(noise, 4),
                "bars": N_BARS,
                "bootstrap_samples": N_BOOT,
            }
        )
    )


if __name__ == "__main__":
    main()
