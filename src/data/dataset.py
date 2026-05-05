"""数据加载与预处理模块

处理1分钟K线CSV数据，输出 [N_days, 240, 5] 的归一化特征数组。
支持中英文列名格式。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl

# 中文列名映射
_CN_COL_MAP = {
    "时间": "timestamp",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
}

_REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
_PRICE_IDX = [0, 1, 2, 3]  # open/high/low/close
_VOL_IDX = 4


def load_raw_data(csv_path: str | Path) -> pl.DataFrame:
    """加载原始1分钟K线CSV，自动识别中英文列名。

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    df = pl.read_csv(str(csv_path))
    # 重命名中文列
    rename_map = {k: v for k, v in _CN_COL_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(rename_map)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少列: {missing}，现有列: {df.columns}")

    df = df.select(_REQUIRED_COLS)
    # 确保时间列为字符串（后续转换）
    if df["timestamp"].dtype != pl.Utf8:
        df = df.with_columns(pl.col("timestamp").cast(pl.Utf8))
    return df


def preprocess_raw_data(df: pl.DataFrame) -> pl.DataFrame:
    """原始数据清洗。

    1. 解析时间戳
    2. 删除停牌日（整天成交量为0）
    3. 删除空值行
    """
    # 解析时间戳
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timestamp")
    )
    df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

    # 删除含空值行
    df = df.drop_nulls()

    # 删除停牌日（当天总成交量=0）
    daily_vol = df.group_by("date").agg(pl.col("volume").sum().alias("daily_vol"))
    valid_dates = daily_vol.filter(pl.col("daily_vol") > 0).select("date")
    df = df.join(valid_dates, on="date", how="inner")

    return df.sort("timestamp")


def detect_bars_per_day(df: pl.DataFrame) -> int:
    """自动检测每个交易日的K线根数（取众数）。"""
    from collections import Counter
    counts = (
        df.group_by("date")
        .agg(pl.len().alias("n"))
        .select("n")
        .to_series()
        .to_list()
    )
    return Counter(counts).most_common(1)[0][0]


def split_by_day(
    df: pl.DataFrame,
    bars_per_day: int | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """按交易日切分K线数据。

    A股数据源不同，每天可能是240或241根K线（取决于是否包含13:00/15:00）。
    bars_per_day=None 时自动检测（取众数）。

    Returns:
        day_data: shape [N_days, bars_per_day, 5]，5维 = [open, high, low, close, volume]
        dates:    对应的日期字符串列表
    """
    if bars_per_day is None:
        bars_per_day = detect_bars_per_day(df)

    days: list[np.ndarray] = []
    dates: list[str] = []

    for date_val, group in df.group_by("date", maintain_order=True):
        group = group.sort("timestamp")
        if len(group) != bars_per_day:
            continue  # 丢弃根数不符的交易日（停牌等）
        ohlcv = group.select(["open", "high", "low", "close", "volume"]).to_numpy()
        days.append(ohlcv.astype(np.float32))
        dates.append(str(date_val))

    if not days:
        raise ValueError(
            f"没有找到完整的{bars_per_day}根交易日，请检查数据格式。"
        )

    return np.stack(days, axis=0), dates


def normalize_features(
    day_data: np.ndarray,
    lookback_window: int = 60,
) -> np.ndarray:
    """因果归一化，严格只用历史数据计算统计量。

    归一化策略：
    - 价格（OHLC）：先转为对数收益率，再用过去lookback_window天的统计量标准化。
      log_return[t] = log(price[t] / price[t-1] + 1e-8)
    - 成交量：log(vol+1)，再标准化。

    Args:
        day_data:        [N_days, 240, 5]，原始OHLCV
        lookback_window: 用多少天历史计算均值/方差

    Returns:
        normalized: [N_days, 240, 5]，归一化后的数据
    """
    N_days, T, F = day_data.shape

    # Step 1: 转换为对数收益率
    returns = np.zeros((N_days, T, F), dtype=np.float32)

    for d in range(N_days):
        for t in range(T):
            if d == 0 and t == 0:
                # 第一个bar，收益率填0
                continue
            if t == 0:
                # 每天第一根bar，参考前一天最后收盘价
                ref = day_data[d - 1, -1, _PRICE_IDX]
            else:
                ref = day_data[d, t - 1, _PRICE_IDX]
            # 价格：对数收益
            returns[d, t, _PRICE_IDX] = np.log(
                day_data[d, t, _PRICE_IDX] / (ref + 1e-8) + 1e-8
            )
        # 成交量：log(vol+1)
        returns[d, :, _VOL_IDX] = np.log1p(day_data[d, :, _VOL_IDX])

    # Step 2: 因果标准化
    normalized = np.zeros_like(returns)

    for d in range(N_days):
        if d == 0:
            # 无历史，直接保留（全0 or raw）
            normalized[d] = returns[d]
            continue
        start = max(0, d - lookback_window)
        hist = returns[start:d]              # [w, 240, 5]
        flat = hist.reshape(-1, F)           # [w*240, 5]
        mean = flat.mean(axis=0)             # [5]
        std  = flat.std(axis=0) + 1e-8       # [5]
        normalized[d] = (returns[d] - mean) / std

    return normalized


def filter_by_year_range(
    day_data: np.ndarray,
    dates: List[str],
    start_year: int,
    end_year: int,
) -> Tuple[np.ndarray, List[str]]:
    """按年份范围过滤。"""
    mask = [start_year <= int(d[:4]) <= end_year for d in dates]
    indices = [i for i, m in enumerate(mask) if m]
    return day_data[indices], [dates[i] for i in indices]
