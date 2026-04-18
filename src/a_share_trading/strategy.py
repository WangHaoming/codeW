from __future__ import annotations

import pandas as pd


def ma_crossover_signals(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """
    Generate moving average crossover signals.

    Signal rules:
    - Long (1) when short MA > long MA
    - Flat (0) otherwise
    """
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    out = df.copy()
    out["ma_short"] = out["close"].rolling(short_window).mean()
    out["ma_long"] = out["close"].rolling(long_window).mean()
    out["signal"] = (out["ma_short"] > out["ma_long"]).astype(int)

    return out
