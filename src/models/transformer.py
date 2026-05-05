"""Transformer序列建模模块

对过去context_window天的244维特征序列建模，输出当前时刻256维状态表示。

架构：
  输入：[batch, 60, 244]
  线性投影：244 → 256
  可学习位置编码：[60, 256]
  4层因果Transformer Encoder
  取最后时间步输出：[batch, 256]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions).unsqueeze(0)  # broadcast over batch


class TradingTransformer(nn.Module):
    """因果Transformer，输入[batch, seq_len, input_dim]，输出[batch, d_model]。

    超参数（默认值来自 configs/default.yaml）：
        d_model=256, nhead=8, num_layers=4, dropout=0.1, dim_feedforward=512
    """

    def __init__(
        self,
        input_dim: int = 244,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        max_seq_len: int = 60,
    ):
        super().__init__()
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 可学习位置编码
        self.pos_enc = LearnablePositionalEncoding(max_seq_len, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """生成上三角因果mask（True表示被遮蔽）。"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            state: [batch, d_model]，最后时间步的状态表示
        """
        seq_len = x.size(1)

        # 投影 + 位置编码
        x = self.input_proj(x) * math.sqrt(self.d_model)  # scale
        x = self.pos_enc(x)

        # 因果mask：每个时间步只能看到过去
        mask = self._causal_mask(seq_len, x.device)
        x = self.transformer(x, mask=mask)  # [batch, seq_len, d_model]

        return x[:, -1, :]  # 取最后时间步 [batch, d_model]
