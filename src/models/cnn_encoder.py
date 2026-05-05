"""CNN特征提取器

三路并行因果CNN，从1分钟K线数据中提取日/周/月三个时间尺度的特征。

架构：
  输入：[N_days, 240, 5]
  路径1（日）：Conv1d(5→20, kernel=240) → [N_days, 20]
  路径2（周）：因果Conv1d(20→64, kernel=5) → [N_days, 64]
  路径3（月）：因果Conv1d(20→128, kernel=20) → [N_days, 128]
  输出：Concat → [N_days, 212]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果卷积：只看过去，不看未来（左侧padding）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        **kwargs,
    ):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, length]"""
        # 左侧padding，右侧不padding → 因果
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = self.bn(x)
        return F.gelu(x)


class CNNEncoder(nn.Module):
    """三路并行CNN特征提取器。

    输入：[N_days, 240, 5]
    输出：[N_days, 212]  (20 + 64 + 128)
    """

    def __init__(
        self,
        day_out_dim: int = 20,
        week_out_dim: int = 64,
        month_out_dim: int = 128,
        week_kernel: int = 5,
        month_kernel: int = 20,
        intraday_bars: int = 241,  # 自动检测，默认241（含09:30和15:00）
    ):
        super().__init__()
        self.intraday_bars = intraday_bars
        self.out_dim = day_out_dim + week_out_dim + month_out_dim  # 212

        # 路径1：日内特征，每天intraday_bars根K线 → 20维标量
        # 日内CNN无需因果约束（收盘后决策，当天数据全部可见）
        self.day_conv = nn.Sequential(
            nn.Conv1d(5, day_out_dim, kernel_size=intraday_bars, stride=intraday_bars, padding=0),
            nn.BatchNorm1d(day_out_dim),
            nn.GELU(),
        )

        # 路径2：周特征，对日特征序列做因果卷积
        self.week_conv = CausalConv1d(day_out_dim, week_out_dim, kernel_size=week_kernel)

        # 路径3：月特征，对日特征序列做因果卷积
        self.month_conv = CausalConv1d(day_out_dim, month_out_dim, kernel_size=month_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N_days, 240, 5]，归一化后的1分钟OHLCV

        Returns:
            features: [N_days, 212]
        """
        N = x.shape[0]

        # 路径1：日内特征
        x_in = x.permute(0, 2, 1)                     # [N, 5, 240]
        day_feat = self.day_conv(x_in).squeeze(-1)     # [N, 20]

        # 路径2/3：把日特征序列当作一维时序处理
        # 形状 [1, day_dim, N_days]  — batch=1，channel=day_dim，length=N_days
        seq = day_feat.unsqueeze(0).permute(0, 2, 1)  # [1, 20, N]

        week_feat  = self.week_conv(seq).squeeze(0).permute(1, 0)    # [N, 64]
        month_feat = self.month_conv(seq).squeeze(0).permute(1, 0)   # [N, 128]

        return torch.cat([day_feat, week_feat, month_feat], dim=-1)  # [N, 212]
