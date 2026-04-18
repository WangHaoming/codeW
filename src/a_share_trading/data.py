from __future__ import annotations

import pandas as pd
import akshare as ak


STANDARD_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def get_a_share_hist(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Fetch A-share historical daily data via AkShare.

    Args:
        symbol: 6-digit A-share code (e.g., 000001)
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        adjust: "qfq" (forward), "hfq" (backward), or "" (none)
    """
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if df.empty:
        raise ValueError("No data returned. Check symbol and date range.")

    df = df.rename(columns=STANDARD_COLUMNS)
    missing = [c for c in STANDARD_COLUMNS.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df
