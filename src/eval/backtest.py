"""回测与评估模块

计算并返回标准量化指标：
  年化收益率、Sharpe Ratio、最大回撤、胜率、盈亏比、年化波动率、Calmar Ratio

并提供与基准（买入持有）的对比功能。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.env.trading_env import StockTradingEnv


TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.03,
) -> Dict[str, float]:
    """由每日收益率数组计算量化指标。

    Args:
        daily_returns: 1D array，每日收益率（小数，e.g., 0.01 = 1%）
        risk_free_rate: 年化无风险利率，默认3%

    Returns:
        metrics: dict
    """
    if len(daily_returns) == 0:
        return {k: 0.0 for k in [
            "annualized_return", "sharpe_ratio", "max_drawdown",
            "win_rate", "profit_loss_ratio", "annualized_volatility", "calmar_ratio"
        ]}

    r = np.array(daily_returns, dtype=np.float64)

    # 年化收益率
    total_return = np.prod(1 + r) - 1
    n_days = len(r)
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

    # 年化波动率
    annualized_vol = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR) + 1e-10

    # Sharpe Ratio
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = r - daily_rf
    sharpe = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # 最大回撤
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (running_max - cum) / (running_max + 1e-10)
    max_drawdown = float(drawdowns.max())

    # Calmar Ratio
    calmar = annualized_return / (max_drawdown + 1e-10)

    # 胜率（单笔收益>0的比例）
    win_mask = r > 0
    win_rate = float(win_mask.mean())

    # 盈亏比
    avg_win  = float(r[win_mask].mean()) if win_mask.any() else 0.0
    lose_mask = r < 0
    avg_loss = float(np.abs(r[lose_mask]).mean()) if lose_mask.any() else 1e-10
    profit_loss_ratio = avg_win / avg_loss

    return {
        "annualized_return":    float(annualized_return),
        "sharpe_ratio":         float(sharpe),
        "max_drawdown":         float(max_drawdown),
        "win_rate":             float(win_rate),
        "profit_loss_ratio":    float(profit_loss_ratio),
        "annualized_volatility": float(annualized_vol),
        "calmar_ratio":         float(calmar),
    }


def run_backtest(
    model: PPO,
    features: np.ndarray,
    close_prices: np.ndarray,
    context_window: int = 60,
    initial_capital: float = 1_000_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.001,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """在给定数据集上完整跑一遍回测（非随机起始，从头到尾）。

    Returns:
        metrics:        量化指标字典
        portfolio_vals: [N_steps] 每步组合价值
        daily_returns:  [N_steps] 每步收益率
    """
    episode_length = len(features) - context_window - 1
    env = StockTradingEnv(
        features=features,
        close_prices=close_prices,
        context_window=context_window,
        episode_length=episode_length,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage,
        random_start=False,
    )

    obs, _ = env.reset()
    done = False
    portfolio_vals: list[float] = []
    daily_returns: list[float] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        portfolio_vals.append(info["portfolio_value"])
        daily_returns.append(info["daily_return"])
        done = terminated or truncated

    daily_returns_arr = np.array(daily_returns, dtype=np.float64)
    metrics = compute_metrics(daily_returns_arr)
    return metrics, np.array(portfolio_vals), daily_returns_arr


def compare_with_benchmark(
    model_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> Dict[str, Dict]:
    """对比模型与基准（买入持有）的各项指标。

    Args:
        model_returns:     模型每日收益率
        benchmark_returns: 基准（如沪深300）每日收益率

    Returns:
        {"model": {...}, "benchmark": {...}}
    """
    return {
        "model":     compute_metrics(model_returns),
        "benchmark": compute_metrics(benchmark_returns),
    }


def print_metrics(metrics: Dict[str, float], title: str = "回测结果") -> None:
    """打印格式化指标。"""
    labels = {
        "annualized_return":     "年化收益率",
        "sharpe_ratio":          "Sharpe Ratio",
        "max_drawdown":          "最大回撤",
        "win_rate":              "胜率",
        "profit_loss_ratio":     "盈亏比",
        "annualized_volatility": "年化波动率",
        "calmar_ratio":          "Calmar Ratio",
    }
    print(f"\n{'='*40}")
    print(f"  {title}")
    print(f"{'='*40}")
    for key, label in labels.items():
        val = metrics.get(key, float("nan"))
        if key in ("annualized_return", "max_drawdown", "win_rate", "annualized_volatility"):
            print(f"  {label:<16}: {val*100:>8.2f}%")
        else:
            print(f"  {label:<16}: {val:>8.4f}")
    print(f"{'='*40}\n")
