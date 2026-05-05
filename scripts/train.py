"""训练入口

加载预处理好的特征数据，执行Walk-Forward PPO训练，最后在测试集上回测。

用法：
    python scripts/train.py --data data/processed/sh600318_features.npz
    python scripts/train.py --data data/processed/sh600318_features.npz \
        --config configs/default.yaml \
        --save-dir checkpoints \
        --n-envs 4 \
        --test-only checkpoints/fold_2022.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.backtest import compare_with_benchmark, print_metrics, run_backtest
from src.models.actor_critic import TradingPolicy
from src.train.trainer import load_config, train_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_processed_data(npz_path: str):
    """加载预处理好的 .npz 文件。"""
    data = np.load(npz_path, allow_pickle=True)
    features     = data["features"]       # [N, 244]
    close_prices = data["close_prices"]   # [N]
    dates        = data["dates"].tolist() # list[str]
    return features, close_prices, dates


def get_test_benchmark_returns(close_prices: np.ndarray, dates: list, test_year: int) -> np.ndarray:
    """计算测试年的买入持有基准每日收益率。"""
    mask = [int(d[:4]) == test_year for d in dates]
    idx = [i for i, m in enumerate(mask) if m]
    if len(idx) < 2:
        return np.array([])
    test_close = close_prices[idx]
    returns = np.diff(test_close) / (test_close[:-1] + 1e-8)
    return returns


def main():
    parser = argparse.ArgumentParser(description="股价预测模型 - PPO训练")
    parser.add_argument("--data", required=True, help="预处理好的.npz特征文件")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数")
    parser.add_argument(
        "--test-only",
        default=None,
        help="跳过训练，只在测试集上评估指定模型路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_year = cfg["data"]["test_year"]

    # ── 加载特征数据 ───────────────────────────────────────
    logger.info(f"加载特征数据: {args.data}")
    features, close_prices, dates = load_processed_data(args.data)
    logger.info(f"总天数: {len(dates)}, 特征维度: {features.shape[1]}")

    if args.test_only:
        # ── 仅测试模式 ─────────────────────────────────────
        from stable_baselines3 import PPO
        logger.info(f"加载模型: {args.test_only}")
        model = PPO.load(args.test_only)

        # 过滤测试集
        test_mask = [int(d[:4]) == test_year for d in dates]
        test_idx  = [i for i, m in enumerate(test_mask) if m]
        test_feat  = features[test_idx]
        test_close = close_prices[test_idx]

        logger.info(f"测试集: {len(test_idx)} 天 ({test_year})")
        metrics, _, model_returns = run_backtest(
            model, test_feat, test_close,
            context_window=cfg["data"]["context_window"],
            initial_capital=cfg["env"]["initial_capital"],
            transaction_cost=cfg["env"]["transaction_cost"],
            slippage=cfg["env"]["slippage"],
        )
        print_metrics(metrics, f"测试集回测 ({test_year})")

        bench_returns = get_test_benchmark_returns(close_prices, dates, test_year)
        if len(bench_returns) > 0:
            comparison = compare_with_benchmark(model_returns, bench_returns)
            print_metrics(comparison["benchmark"], "基准（买入持有）")

    else:
        # ── Walk-Forward训练 ──────────────────────────────
        logger.info("开始Walk-Forward训练...")
        summary = train_walk_forward(
            features=features,
            close_prices=close_prices,
            dates=dates,
            cfg=cfg,
            save_dir=args.save_dir,
            n_envs=args.n_envs,
        )
        logger.info(f"Walk-Forward均值Sharpe: {summary['mean_sharpe']:.4f}")
        logger.info(f"最优模型: {summary['best_model_path']}")

        # ── 测试集最终评估 ─────────────────────────────────
        logger.info(f"\n在测试集({test_year})上评估最优模型...")
        from stable_baselines3 import PPO
        model = PPO.load(summary["best_model_path"])

        test_mask = [int(d[:4]) == test_year for d in dates]
        test_idx  = [i for i, m in enumerate(test_mask) if m]
        test_feat  = features[test_idx]
        test_close = close_prices[test_idx]

        if len(test_idx) > cfg["data"]["context_window"] + 10:
            metrics, _, model_returns = run_backtest(
                model, test_feat, test_close,
                context_window=cfg["data"]["context_window"],
            )
            print_metrics(metrics, f"测试集回测 ({test_year})")

            bench_returns = get_test_benchmark_returns(close_prices, dates, test_year)
            if len(bench_returns) > 0:
                comparison = compare_with_benchmark(model_returns, bench_returns)
                print_metrics(comparison["benchmark"], "基准（买入持有）")
        else:
            logger.warning(f"测试集数据不足（{len(test_idx)}天），跳过测试集评估。")


if __name__ == "__main__":
    main()
