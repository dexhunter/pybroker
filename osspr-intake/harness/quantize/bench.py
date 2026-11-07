"""Time ``pybroker.common.quantize`` on one result-export column.

The column is 20,000 ``Decimal`` values, in the same range as a long
portfolio equity curve (the asv result-export bench uses 12,000 rows
across several money columns). ``round=True`` takes the cent-rounding
path used when ``StrategyConfig.round_test_result`` is on.
"""

from __future__ import annotations

import gc
import os
import statistics
import time
from decimal import Decimal

import numpy as np
import pandas as pd

from pybroker.common import quantize

REPEATS = 31
WARMUP = 8
N_ROWS = 20_000


def _frame() -> pd.DataFrame:
    values = [Decimal(f"{v:.4f}") for v in np.linspace(50.0, 150.0, N_ROWS)]
    return pd.DataFrame({"equity": values})


def main() -> None:
    os.sched_setaffinity(0, {0})
    df = _frame()
    for _ in range(WARMUP):
        quantize(df, "equity", True)
    gc.collect()
    samples: list[float] = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        quantize(df, "equity", True)
        samples.append((time.perf_counter() - t0) * 1000.0)
    print(f"median_ms={statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
