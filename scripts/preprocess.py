"""数据预处理入口

从原始1分钟K线CSV生成 [N_days, 244] 特征矩阵并保存到 data/processed/。

用法：
    python scripts/preprocess.py --csv data/raw/sh600318_2026.csv
    python scripts/preprocess.py --csv data/raw/your_data.csv \
        --financial data/financial/000001_financial.csv \
        --output data/processed/features_000001.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    filter_by_year_range,
    load_raw_data,
    normalize_features,
    preprocess_raw_data,
    split_by_day,
)
from src.data.features import build_features, build_features_no_financial
from src.data.financial import build_financial_features, load_financial_data
from src.models.cnn_encoder import CNNEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="股价预测模型 - 数据预处理")
    parser.add_argument("--csv", required=True, help="原始1分钟K线CSV路径")
    parser.add_argument("--financial", default=None, help="季度财报CSV路径（可选）")
    parser.add_argument(
        "--output",
        default=None,
        help="输出.npz路径，默认 data/processed/<stock>_features.npz",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--cnn-weights",
        default=None,
        help="预训练CNN权重路径（.pth），不指定则使用随机初始化",
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    lookback = data_cfg.get("lookback_for_norm", 60)
    cnn_cfg  = cfg.get("model", {}).get("cnn", {})

    # ── Step 1: 加载并清洗原始K线数据 ─────────────────────
    logger.info(f"加载K线数据: {args.csv}")
    raw_df = load_raw_data(args.csv)
    clean_df = preprocess_raw_data(raw_df)
    logger.info(f"清洗后: {len(clean_df)} 条记录")

    # ── Step 2: 按天切分 ───────────────────────────────────
    logger.info("按交易日切分（每天240根K线）...")
    day_data, dates = split_by_day(clean_df)
    logger.info(f"有效交易日: {len(dates)} 天，shape={day_data.shape}")

    # ── Step 3: 因果归一化 ─────────────────────────────────
    logger.info(f"因果归一化（lookback={lookback}天）...")
    day_data_norm = normalize_features(day_data, lookback_window=lookback)

    # ── Step 4: 初始化CNN编码器 ────────────────────────────
    import torch
    cnn = CNNEncoder(
        day_out_dim   = cnn_cfg.get("day_out_dim", 20),
        week_out_dim  = cnn_cfg.get("week_out_dim", 64),
        month_out_dim = cnn_cfg.get("month_out_dim", 128),
        week_kernel   = cnn_cfg.get("week_kernel", 5),
        month_kernel  = cnn_cfg.get("month_kernel", 20),
    )
    if args.cnn_weights:
        logger.info(f"加载CNN权重: {args.cnn_weights}")
        cnn.load_state_dict(torch.load(args.cnn_weights, map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"CNN推理设备: {device}")

    # ── Step 5: 财报特征 ───────────────────────────────────
    if args.financial:
        logger.info(f"加载财报数据: {args.financial}")
        fin_df = load_financial_data(args.financial)
        financial_feat = build_financial_features(fin_df, dates)
        logger.info(f"财报特征 shape={financial_feat.shape}")
        features = build_features(day_data_norm, financial_feat, cnn, device)
    else:
        logger.info("未提供财报数据，财报特征填0")
        features = build_features_no_financial(day_data_norm, cnn, device)

    logger.info(f"最终特征 shape={features.shape}")  # [N_days, 244]

    # ── Step 6: 保存 ───────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        stem = Path(args.csv).stem
        out_path = Path("data/processed") / f"{stem}_features.npz"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 提取收盘价序列（供RL环境使用）
    close_prices = day_data[:, -1, 3]  # 每天最后一根bar的收盘价

    np.savez_compressed(
        str(out_path),
        features=features,
        close_prices=close_prices,
        dates=np.array(dates),
    )
    logger.info(f"特征已保存至: {out_path}")
    logger.info(
        f"  features: {features.shape}, "
        f"close_prices: {close_prices.shape}, "
        f"dates: {len(dates)}"
    )


if __name__ == "__main__":
    main()
