"""训练主循环

Walk-Forward验证 + SB3 PPO训练。

数据划分：
  Fold1: Train=2015-2019, Val=2020
  Fold2: Train=2015-2020, Val=2021
  Fold3: Train=2015-2021, Val=2022
  取3个fold的验证集Sharpe均值作为模型选择依据
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.env.trading_env import StockTradingEnv
from src.eval.backtest import compute_metrics
from src.models.actor_critic import TradingPolicy

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Walk-Forward 分折定义
# ──────────────────────────────────────────────

WALK_FORWARD_FOLDS: List[Tuple[Tuple[int, int], int]] = [
    ((2015, 2019), 2020),
    ((2015, 2020), 2021),
    ((2015, 2021), 2022),
]


def _slice_by_year(
    features: np.ndarray,
    close_prices: np.ndarray,
    dates: List[str],
    start_year: int,
    end_year: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """按年份范围切片 features 和 close_prices。"""
    mask = [start_year <= int(d[:4]) <= end_year for d in dates]
    idx = [i for i, m in enumerate(mask) if m]
    return features[idx], close_prices[idx]


def make_env(
    features: np.ndarray,
    close_prices: np.ndarray,
    env_cfg: Dict,
    random_start: bool = True,
):
    """返回一个创建 StockTradingEnv 的工厂函数（供 make_vec_env 使用）。"""
    def _init():
        return StockTradingEnv(
            features=features,
            close_prices=close_prices,
            context_window=env_cfg.get("context_window", 60),
            episode_length=env_cfg.get("episode_length", 250),
            initial_capital=env_cfg.get("initial_capital", 1_000_000),
            transaction_cost=env_cfg.get("transaction_cost", 0.001),
            slippage=env_cfg.get("slippage", 0.001),
            max_dd_penalty=env_cfg.get("max_drawdown_penalty", 0.05),
            random_start=random_start,
        )
    return _init


def evaluate_on_val(
    model: PPO,
    val_features: np.ndarray,
    val_close: np.ndarray,
    env_cfg: Dict,
    n_eval_episodes: int = 5,
) -> Dict:
    """在验证集上跑n_eval_episodes轮，返回平均指标。"""
    env = StockTradingEnv(
        features=val_features,
        close_prices=val_close,
        **{k: env_cfg.get(k, v) for k, v in [
            ("context_window", 60),
            ("episode_length", 250),
            ("initial_capital", 1_000_000),
            ("transaction_cost", 0.001),
            ("slippage", 0.001),
            ("max_drawdown_penalty", 0.05),
        ]},
        random_start=True,
    )

    all_returns: list[list[float]] = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_returns = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_returns.append(info["daily_return"])
            done = terminated or truncated
        all_returns.append(episode_returns)

    # 汇总所有episode的每日收益率
    flat_returns = [r for ep in all_returns for r in ep]
    metrics = compute_metrics(np.array(flat_returns))
    return metrics


def train_fold(
    fold_train: Tuple[int, int],
    fold_val: int,
    features: np.ndarray,
    close_prices: np.ndarray,
    dates: List[str],
    cfg: Dict,
    save_dir: Path,
    n_envs: int = 4,
) -> Tuple[PPO, Dict]:
    """训练单个Walk-Forward折叠，返回模型和验证指标。"""
    train_start, train_end = fold_train

    train_feat, train_close = _slice_by_year(
        features, close_prices, dates, train_start, train_end
    )
    val_feat, val_close = _slice_by_year(
        features, close_prices, dates, fold_val, fold_val
    )

    logger.info(f"Fold train={train_start}-{train_end}, val={fold_val}")
    logger.info(f"  训练集: {len(train_feat)}天, 验证集: {len(val_feat)}天")

    env_cfg = cfg.get("env", {})
    env_cfg["context_window"] = cfg["data"]["context_window"]

    # 创建并行训练环境
    vec_env = make_vec_env(
        make_env(train_feat, train_close, env_cfg, random_start=True),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    ppo_cfg = cfg.get("ppo", {})
    transformer_kwargs = {
        k: cfg["model"]["transformer"][k]
        for k in ["nhead", "num_layers", "dropout", "dim_feedforward"]
    }

    model = PPO(
        policy=TradingPolicy,
        env=vec_env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        verbose=1,
        policy_kwargs={"transformer_kwargs": transformer_kwargs},
    )

    total_timesteps = ppo_cfg.get("total_timesteps", 1_000_000)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # 保存模型
    model_path = save_dir / f"fold_{train_end}"
    model.save(str(model_path))
    logger.info(f"模型已保存至 {model_path}")

    # 验证集评估
    val_metrics = evaluate_on_val(model, val_feat, val_close, env_cfg)
    logger.info(f"  验证集 Sharpe={val_metrics['sharpe_ratio']:.4f}, "
                f"MaxDD={val_metrics['max_drawdown']:.4f}")

    vec_env.close()
    return model, val_metrics


def train_walk_forward(
    features: np.ndarray,
    close_prices: np.ndarray,
    dates: List[str],
    cfg: Dict,
    save_dir: str | Path = "checkpoints",
    n_envs: int = 4,
) -> Dict:
    """执行全部Walk-Forward折叠训练，返回最佳模型路径和汇总指标。

    Args:
        features:    [N_days, 244]
        close_prices:[N_days]
        dates:       N_days 个日期字符串
        cfg:         来自 configs/default.yaml 的配置字典
        save_dir:    模型保存目录
        n_envs:      并行环境数量

    Returns:
        summary: {"best_model_path": str, "fold_sharpes": list, "mean_sharpe": float}
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_sharpes: list[float] = []
    best_sharpe = -np.inf
    best_path = ""

    for (train_start, train_end), val_year in WALK_FORWARD_FOLDS:
        model, metrics = train_fold(
            fold_train=(train_start, train_end),
            fold_val=val_year,
            features=features,
            close_prices=close_prices,
            dates=dates,
            cfg=cfg,
            save_dir=save_dir,
            n_envs=n_envs,
        )
        sharpe = metrics["sharpe_ratio"]
        fold_sharpes.append(sharpe)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_path = str(save_dir / f"fold_{train_end}.zip")

    mean_sharpe = float(np.mean(fold_sharpes))
    logger.info(f"\nWalk-Forward完成: fold Sharpes={fold_sharpes}, 均值={mean_sharpe:.4f}")
    logger.info(f"最优模型: {best_path} (Sharpe={best_sharpe:.4f})")

    return {
        "best_model_path": best_path,
        "fold_sharpes": fold_sharpes,
        "mean_sharpe": mean_sharpe,
    }


def load_config(config_path: str | Path = "configs/default.yaml") -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
