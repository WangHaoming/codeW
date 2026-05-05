"""Actor-Critic网络及SB3自定义Policy

ActorCritic:     独立模块，[batch,256] → (action_probs[batch,3], value[batch,1])
TradingPolicyNetwork: SB3 mlp_extractor 兼容类，提供 latent_dim_pi / latent_dim_vf
TradingFeaturesExtractor: SB3 BaseFeaturesExtractor，包装 TradingTransformer
TradingPolicy:   SB3 ActorCriticPolicy 子类，完整端到端策略
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.models.transformer import TradingTransformer


# ──────────────────────────────────────────────
# 1. 独立 ActorCritic（用于单元测试 / 自定义训练）
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Actor（策略头）：Linear(256,128) → GELU → Linear(128,3) → Softmax
    Critic（价值头）：Linear(256,128) → GELU → Linear(128,1)

    动作空间：0=买入，1=卖出，2=持有
    """

    def __init__(self, state_dim: int = 256, hidden_dim: int = 128, n_actions: int = 3):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (action_probs, value)"""
        action_probs = torch.softmax(self.actor(state), dim=-1)
        value = self.critic(state)
        return action_probs, value

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作，返回 (action, log_prob, value)"""
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ──────────────────────────────────────────────
# 2. SB3 兼容组件
# ──────────────────────────────────────────────

class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 特征提取器：obs[batch, 60, 244] → state[batch, 256]"""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        transformer_kwargs: Optional[Dict] = None,
    ):
        super().__init__(observation_space, features_dim)
        kwargs = transformer_kwargs or {}
        input_dim = observation_space.shape[-1]   # 244
        self.transformer = TradingTransformer(input_dim=input_dim, d_model=features_dim, **kwargs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.transformer(observations)


class TradingPolicyNetwork(nn.Module):
    """SB3 mlp_extractor：接受特征提取器输出，分别生成 actor/critic 的隐藏表示。

    SB3 要求：
      - latent_dim_pi: int
      - latent_dim_vf: int
      - forward(features) → (latent_pi, latent_vf)
      - forward_actor(features) → latent_pi
      - forward_critic(features) → latent_vf
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim

        self.actor_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor_net(features), self.critic_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_net(features)


class TradingPolicy(ActorCriticPolicy):
    """完整自定义策略，替换 SB3 默认的 MLP 提取器。

    使用方式：
        model = PPO(TradingPolicy, env, policy_kwargs={"transformer_kwargs": {...}})
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        transformer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self._transformer_kwargs = transformer_kwargs or {}
        # 必须在 super().__init__ 前设置，因为 _build 在 __init__ 中调用
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            ortho_init=False,
            features_extractor_class=TradingFeaturesExtractor,
            features_extractor_kwargs={
                "features_dim": 256,
                "transformer_kwargs": self._transformer_kwargs,
            },
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TradingPolicyNetwork(
            feature_dim=self.features_dim,
            hidden_dim=128,
        )
