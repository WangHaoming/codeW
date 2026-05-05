"""财报数据处理模块

从季度财报CSV构建每个交易日的32维财报特征向量。
严格使用 publish_date（实际发布日期），防止数据泄漏。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import polars as pl

_FINANCIAL_FIELDS = [
    "revenue_growth",
    "net_profit_margin",
    "roe",
    "debt_ratio",
    "gross_margin",
    "operating_cashflow",
    "rd_ratio",
    "yoy_net_profit",
]

# 固定随机投影矩阵（seed=42），8维 → 32维
_RNG = np.random.default_rng(42)
_PROJ_W: np.ndarray = (_RNG.standard_normal((8, 32)) / np.sqrt(8)).astype(np.float32)


def load_financial_data(csv_path: str | Path) -> pl.DataFrame:
    """加载财报CSV。

    期望列：stock_code, publish_date, report_period,
            revenue_growth, net_profit_margin, roe, debt_ratio,
            gross_margin, operating_cashflow, rd_ratio, yoy_net_profit
    """
    df = pl.read_csv(str(csv_path))
    df = df.with_columns(
        pl.col("publish_date").str.to_datetime(
            format="%Y-%m-%d", strict=False
        ).dt.date()
    )
    return df.sort("publish_date")


def _encode_8d(raw_8: np.ndarray) -> np.ndarray:
    """8维财报原始值 → 32维特征向量（固定线性投影 + tanh）。

    使用固定随机投影保证可复现性，不引入可训练参数。
    """
    return np.tanh(raw_8 @ _PROJ_W)   # [32]


def build_financial_features(
    financial_df: pl.DataFrame,
    trading_dates: List[str],
) -> np.ndarray:
    """为每个交易日找到截至当天最新发布的财报，编码为32维向量。

    Args:
        financial_df:   财报DataFrame（含 publish_date、各财务字段）
        trading_dates:  交易日日期字符串列表，格式 'YYYY-MM-DD'

    Returns:
        features: [N_days, 32]，上市不足一年的日期填0
    """
    N = len(trading_dates)
    features = np.zeros((N, 32), dtype=np.float32)

    # 提取财报记录为 list of (date, np.ndarray[8])
    dates_col = financial_df["publish_date"].to_list()
    rows_8 = []
    for row in financial_df.iter_rows(named=True):
        vals = np.array(
            [row.get(f, 0.0) or 0.0 for f in _FINANCIAL_FIELDS],
            dtype=np.float32,
        )
        rows_8.append(vals)

    if not rows_8:
        return features  # 无财报数据，全0

    report_dates = [d for d in dates_col]   # list[datetime.date]

    for i, date_str in enumerate(trading_dates):
        from datetime import date as date_type
        try:
            cur_date = date_type.fromisoformat(date_str)
        except Exception:
            continue

        # 找到 publish_date <= cur_date 中最新的一条
        best_idx = -1
        for j, rd in enumerate(report_dates):
            if rd <= cur_date:
                best_idx = j

        if best_idx >= 0:
            features[i] = _encode_8d(rows_8[best_idx])
        # 否则保持0（上市不足一年或无财报）

    return features
