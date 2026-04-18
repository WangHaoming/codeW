from __future__ import annotations

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = 1_000_000.0,
    fee_rate: float = 0.0003,
) -> pd.DataFrame:
    """
    Simple long-only backtest.

    Assumptions:
    - Trades executed at next day's open based on previous day's signal
    - Full allocation when long, all cash when flat
    - Fee applied on notional when changing position
    """
    out = df.copy()
    out = out.dropna(subset=["signal", "open"]).reset_index(drop=True)

    out["position"] = out["signal"].shift(1).fillna(0)

    cash = initial_cash
    shares = 0.0
    equity_curve = []

    prev_position = 0
    for _, row in out.iterrows():
        price = row["open"]
        position = int(row["position"])

        if position != prev_position:
            # Close existing
            if shares > 0:
                cash += shares * price
                cash -= shares * price * fee_rate
                shares = 0.0
            # Open new
            if position == 1:
                shares = cash / price
                cash -= shares * price * fee_rate
                cash = 0.0

        equity = cash + shares * price
        equity_curve.append(equity)
        prev_position = position

    out["equity"] = equity_curve
    out["returns"] = out["equity"].pct_change().fillna(0.0)

    return out
