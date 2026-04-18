#!/usr/bin/env python3
from __future__ import annotations

import argparse

from a_share_trading.data import get_a_share_hist
from a_share_trading.strategy import ma_crossover_signals
from a_share_trading.backtest import run_backtest
from a_share_trading.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A-share backtest")
    parser.add_argument("--symbol", required=True, help="6-digit A-share code")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--short", type=int, default=20, help="Short MA window")
    parser.add_argument("--long", type=int, default=60, help="Long MA window")
    parser.add_argument("--cash", type=float, default=1_000_000, help="Initial cash")
    parser.add_argument("--fee", type=float, default=0.0003, help="Fee rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = get_a_share_hist(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        adjust="qfq",
    )

    df = ma_crossover_signals(df, short_window=args.short, long_window=args.long)
    bt = run_backtest(df, initial_cash=args.cash, fee_rate=args.fee)
    metrics = compute_metrics(bt)

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
