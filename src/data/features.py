"""特征工程模块

将CNN编码的K线特征与财报特征拼接，得到每天244维的特征向量。
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from src.models.cnn_encoder import CNNEncoder


def build_features(
    day_data_norm: np.ndarray,
    financial_features: np.ndarray,
    cnn_encoder: CNNEncoder | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """构建完整的 [N_days, 244] 特征矩阵。

    Args:
        day_data_norm:      [N_days, 240, 5]，归一化后的1分钟K线数据
        financial_features: [N_days, 32]，财报特征
        cnn_encoder:        CNNEncoder实例；None时自动创建（随机权重）
        device:             'cpu' 或 'cuda'

    Returns:
        features: [N_days, 244]  = 212维K线 + 32维财报
    """
    if cnn_encoder is None:
        cnn_encoder = CNNEncoder()

    cnn_encoder = cnn_encoder.to(device).eval()

    # CNN编码 [N_days, 240, 5] → [N_days, 212]
    x = torch.from_numpy(day_data_norm).to(device)      # [N, 240, 5]
    with torch.no_grad():
        kline_feat = cnn_encoder(x).cpu().numpy()        # [N, 212]

    # 拼接财报特征
    assert kline_feat.shape[0] == financial_features.shape[0], (
        f"K线天数({kline_feat.shape[0]}) ≠ 财报天数({financial_features.shape[0]})"
    )
    features = np.concatenate([kline_feat, financial_features], axis=1)  # [N, 244]
    return features.astype(np.float32)


def build_features_no_financial(
    day_data_norm: np.ndarray,
    cnn_encoder: CNNEncoder | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """无财报数据时的降级版本，财报部分填0，仍返回 [N_days, 244]。"""
    N = day_data_norm.shape[0]
    fin = np.zeros((N, 32), dtype=np.float32)
    return build_features(day_data_norm, fin, cnn_encoder, device)
