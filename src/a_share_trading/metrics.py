from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame, trading_days: int = 252) -> dict:
    """Compute basic performance metrics."""
    equity = df["equity"].astype(float)
    returns = df["returns"].astype(float)

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (trading_days / len(equity)) - 1.0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    max_drawdown = drawdown.min()

    sharpe = 0.0
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trading_days)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }
