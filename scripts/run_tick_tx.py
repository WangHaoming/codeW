#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import akshare as ak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch A-share tick data (秒级成交) via stock_zh_a_tick_tx"
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="股票代码，示例: 000001 (有些环境可能需要带市场前缀，如 sz000001/sh600000)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="交易日期 YYYYMMDD（可选，留空使用接口默认）",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="展示前 N 行",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(ak, "stock_zh_a_tick_163"):
        kwargs = {"symbol": args.symbol}
        if args.date:
            kwargs["trade_date"] = args.date
        df = ak.stock_zh_a_tick_163(**kwargs)
    elif hasattr(ak, "stock_zh_a_tick_tx_js"):
        if args.date:
            print("Warning: 当前版本仅支持 stock_zh_a_tick_tx_js（当前/最近交易日），忽略 --date")
        df = ak.stock_zh_a_tick_tx_js(symbol=args.symbol)
        print(args.symbol)
    else:
        print("AkShare 版本过旧或接口变更：未找到 stock_zh_a_tick_tx/stock_zh_a_tick_tx_js")
        print("建议升级 AkShare：pip install -U akshare")
        return

    if df is None or df.empty:
        print("No data returned. Check symbol/date and market suffix requirements.")
        return

    print(df.head(args.rows))
    print(f"\nRows: {len(df)}")
    safe_symbol = args.symbol.replace("/", "_")
    date_part = args.date if args.date else "latest"
    out_path = data_dir / f"tick_tx_{safe_symbol}_{date_part}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
