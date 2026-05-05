"""股票交易强化学习环境（Gymnasium接口）

观测空间：过去60天的244维特征矩阵 [60, 244]
动作空间：Discrete(3)  —  0=买入(全仓), 1=卖出(清仓), 2=持有

Reward设计（Sharpe增量）：
  每步 reward = 当日收益率 / 过去30日收益率标准差
  额外惩罚：当日最大回撤 > 5% 时扣分

Episode设计：
  长度 = 一年约250天（可配置）
  随机起始点，避免过拟合特定时段
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    """
    Args:
        features:         [N_days, 244] 预处理好的特征矩阵
        close_prices:     [N_days] 收盘价序列（用于计算收益率）
        context_window:   观测窗口大小，默认60天
        episode_length:   每个episode的交易天数，默认250
        initial_capital:  初始资金（元），默认100万
        transaction_cost: 单边手续费率，默认0.001
        slippage:         单边滑点率，默认0.001
        max_dd_penalty:   最大回撤惩罚阈值，默认0.05（5%）
        random_start:     是否随机选择起始点，默认True
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        context_window: int = 60,
        episode_length: int = 250,
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.001,
        max_dd_penalty: float = 0.05,
        random_start: bool = True,
    ):
        super().__init__()
        assert len(features) == len(close_prices), "features和close_prices长度必须一致"
        assert features.shape[1] == 244, f"features第2维应为244，实际为{features.shape[1]}"

        self.features = features.astype(np.float32)
        self.close_prices = close_prices.astype(np.float32)
        self.context_window = context_window
        self.episode_length = episode_length
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_dd_penalty = max_dd_penalty
        self.random_start = random_start

        self.N = len(features)
        # episode的最早起始点：保证有足够的context_window
        self._min_start = context_window
        # 最晚起始点：保证episode不越界
        self._max_start = self.N - episode_length - 1

        if self._max_start <= self._min_start:
            raise ValueError(
                f"数据量不足：需要至少 {context_window + episode_length + 1} 天，"
                f"实际 {self.N} 天。"
            )

        # Gym接口
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(context_window, 244),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=买, 1=卖, 2=持有

        # 内部状态（由reset初始化）
        self._start_idx: int = 0
        self._current_idx: int = 0
        self._position: int = 0          # 0=空仓，1=满仓
        self._cash: float = 0.0
        self._shares: float = 0.0
        self._portfolio_value: float = 0.0
        self._peak_value: float = 0.0
        self._returns_history: list[float] = []

    # ──────────────────────────────────────────
    # Gym 接口
    # ──────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if self.random_start:
            self._start_idx = int(self.np_random.integers(
                self._min_start, self._max_start + 1
            ))
        else:
            self._start_idx = self._min_start

        self._current_idx = self._start_idx
        self._position = 0
        self._cash = self.initial_capital
        self._shares = 0.0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._returns_history = []

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作，返回 (next_obs, reward, terminated, truncated, info)。

        买入/卖出在 *下一根K线开盘价* 执行（模拟T+1，避免用当日收盘价作弊）。
        已持仓时买入 → no-op；未持仓时卖出 → no-op。
        """
        idx = self._current_idx    # 当前决策日（用当日特征决策）
        next_idx = idx + 1         # 下一交易日

        cur_close  = float(self.close_prices[idx])
        next_open  = self._get_open_price(next_idx)
        next_close = float(self.close_prices[next_idx])

        prev_value = self._portfolio_value

        # ── 执行动作 ──────────────────────────────
        if action == 0 and self._position == 0:  # 买入（空仓 → 满仓）
            cost_rate = 1.0 + self.transaction_cost + self.slippage
            exec_price = next_open * cost_rate
            if exec_price > 0:
                self._shares = self._cash / exec_price
                self._cash = 0.0
                self._position = 1

        elif action == 1 and self._position == 1:  # 卖出（满仓 → 空仓）
            earn_rate = 1.0 - self.transaction_cost - self.slippage
            exec_price = next_open * earn_rate
            self._cash = self._shares * exec_price
            self._shares = 0.0
            self._position = 0

        # action == 2 或 no-op：不操作

        # ── 更新组合价值 ───────────────────────────
        if self._position == 1:
            self._portfolio_value = self._shares * next_close
        else:
            self._portfolio_value = self._cash

        # ── 计算 reward（Sharpe增量）───────────────
        daily_return = (self._portfolio_value - prev_value) / (prev_value + 1e-8)
        self._returns_history.append(daily_return)

        reward = self._compute_reward(daily_return)

        # ── 更新最高水位，计算回撤惩罚 ─────────────
        self._peak_value = max(self._peak_value, self._portfolio_value)
        drawdown = (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-8)
        if drawdown > self.max_dd_penalty:
            reward -= drawdown  # 超出部分按比例惩罚

        # ── 推进时间步 ─────────────────────────────
        self._current_idx += 1
        done = (self._current_idx - self._start_idx) >= self.episode_length

        info = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "daily_return": daily_return,
            "drawdown": drawdown,
        }
        return self._get_obs(), float(reward), done, False, info

    def render(self) -> None:
        pass

    # ──────────────────────────────────────────
    # 内部辅助
    # ──────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """返回过去 context_window 天的特征 [60, 244]。"""
        end   = self._current_idx
        start = end - self.context_window
        return self.features[start:end].copy()

    def _get_open_price(self, idx: int) -> float:
        """从特征矩阵中取近似开盘价（此处用下一日收盘价近似；
        若有单独开盘价数组可替换）。"""
        if idx < self.N:
            return float(self.close_prices[idx])
        return float(self.close_prices[-1])

    def _compute_reward(self, daily_return: float) -> float:
        """Sharpe增量 reward：当日收益 / 过去30日收益标准差。"""
        window = self._returns_history[-30:]
        if len(window) < 2:
            return daily_return
        std = float(np.std(window)) + 1e-8
        return daily_return / std
