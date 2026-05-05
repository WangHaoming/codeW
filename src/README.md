# 股票交易强化学习系统 — 源码说明

基于10年历史1分钟K线数据 + 季度财报数据，使用 CNN + Transformer + RL（PPO）构建端到端的股票交易决策系统。模型直接输出买/卖/持有动作，以 Sharpe Ratio 作为优化目标。

---

## 目录结构

```
src/
├── data/
│   ├── dataset.py      # 数据加载、清洗、按天切分、因果归一化
│   ├── financial.py    # 财报特征处理（8维→32维，固定线性投影）
│   └── features.py     # CNN编码 + 财报拼接 → [N_days, 244]
├── models/
│   ├── cnn_encoder.py  # 三路并行因果CNN，[N, bars, 5] → [N, 212]
│   ├── transformer.py  # 因果Transformer，[B, 60, 244] → [B, 256]
│   └── actor_critic.py # Actor-Critic网络 + SB3自定义Policy
├── env/
│   └── trading_env.py  # Gym交易环境，Sharpe增量reward
├── train/
│   └── trainer.py      # Walk-Forward PPO训练主循环
└── eval/
    └── backtest.py     # 回测与评估指标
```

---

## 数据流

```
原始CSV（1分钟K线）
  └─ dataset.py: load_raw_data → preprocess_raw_data → split_by_day
        [N_days, bars_per_day, 5]（OHLCV）
        └─ normalize_features（因果归一化）
              [N_days, bars_per_day, 5]（归一化后）
              └─ features.py: build_features
                    ├─ cnn_encoder.py: CNNEncoder → [N_days, 212]
                    └─ financial.py: build_financial_features → [N_days, 32]
                          拼接 → [N_days, 244]  ←── 保存到 data/processed/*.npz

[N_days, 244] 特征矩阵
  └─ trading_env.py: StockTradingEnv
        观测 obs = [60, 244]（过去60天）
        └─ trainer.py: PPO（TradingPolicy）
              ├─ transformer.py: TradingTransformer → [B, 256]
              └─ actor_critic.py: Actor → action {0=买, 1=卖, 2=持有}
                                  Critic → value（状态价值）
```

---

## 模块说明

### `data/dataset.py`

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `load_raw_data(csv_path)` | CSV路径 | `pl.DataFrame` | 自动识别中英文列名 |
| `preprocess_raw_data(df)` | DataFrame | DataFrame | 清洗停牌日、空值 |
| `detect_bars_per_day(df)` | DataFrame | `int` | 自动检测每日K线根数（众数） |
| `split_by_day(df, bars_per_day)` | DataFrame | `(ndarray[N,T,5], dates)` | 按交易日切分，丢弃不完整的日 |
| `normalize_features(day_data, lookback)` | `[N,T,5]` | `[N,T,5]` | 因果归一化，严格只用历史统计量 |
| `filter_by_year_range(...)` | 特征+日期 | 过滤后数组 | 按年份范围切片 |

**归一化策略（防未来泄漏）**：
- 价格（OHLC）：转为对数收益率 `log(p[t]/p[t-1])`，再用过去 `lookback_window` 天的均值/方差标准化
- 成交量：`log(vol+1)`，再标准化
- 第 d 天的统计量仅由第 `[d-lookback, d-1]` 天计算，绝不使用未来数据

---

### `data/financial.py`

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `load_financial_data(csv_path)` | CSV路径 | `pl.DataFrame` | 加载季度财报 |
| `build_financial_features(fin_df, trading_dates)` | 财报DF + 日期列表 | `[N_days, 32]` | 每天取最新发布财报编码为32维 |

**防数据泄漏**：严格使用 `publish_date`（实际发布日期），而非 `report_period`。每个交易日只能看到发布日 ≤ 当天的财报。

8维财报字段：`revenue_growth, net_profit_margin, roe, debt_ratio, gross_margin, operating_cashflow, rd_ratio, yoy_net_profit`

编码方式：固定随机线性投影（seed=42）+ tanh，8→32维，无可训练参数，可复现。

---

### `models/cnn_encoder.py`

三路并行CNN，提取日/周/月三个时间尺度的特征：

```
输入：[N_days, intraday_bars, 5]  (intraday_bars 自动检测，通常241)

路径1（日内特征）：
  Conv1d(5→20, kernel=intraday_bars, stride=intraday_bars) → [N, 20]
  无因果约束（收盘后决策，当天数据全部可见）

路径2（周特征，因果）：
  CausalConv1d(20→64, kernel=5, 左侧padding=4) → [N, 64]

路径3（月特征，因果）：
  CausalConv1d(20→128, kernel=20, 左侧padding=19) → [N, 128]

输出：Concat([20, 64, 128]) = [N, 212]
```

**因果卷积实现**：`F.pad(x, (kernel_size-1, 0))` 左侧填充，右侧不填充，确保每个时间步只能看到过去的信息。

主要参数（可通过 `configs/default.yaml` 配置）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `day_out_dim` | 20 | 日内特征维度 |
| `week_out_dim` | 64 | 周特征维度 |
| `month_out_dim` | 128 | 月特征维度 |
| `week_kernel` | 5 | 周CNN卷积核大小 |
| `month_kernel` | 20 | 月CNN卷积核大小 |
| `intraday_bars` | 241 | 每日K线根数（自动检测） |

---

### `models/transformer.py`

```
输入：[batch, 60, 244]
  └─ 线性投影：244 → 256
  └─ 可学习位置编码：Embedding(60, 256)
  └─ 4层 Transformer Encoder（Pre-LayerNorm，GELU激活）
     因果Mask：每时间步只能看过去，不看未来
  └─ 取最后时间步输出：[batch, 256]
```

| 参数 | 默认值 |
|---|---|
| `d_model` | 256 |
| `nhead` | 8 |
| `num_layers` | 4 |
| `dropout` | 0.1 |
| `dim_feedforward` | 512 |
| `max_seq_len` | 60 |

---

### `models/actor_critic.py`

提供两套接口：

**独立模块 `ActorCritic`**（用于自定义训练/单元测试）：
```
输入：[batch, 256]
Actor：Linear(256,128) → GELU → Linear(128,3) → Softmax → action概率
Critic：Linear(256,128) → GELU → Linear(128,1) → 状态价值
动作：0=买入(全仓), 1=卖出(清仓), 2=持有
```

**SB3兼容组件**：
- `TradingFeaturesExtractor`：继承 `BaseFeaturesExtractor`，obs `[B,60,244]` → `[B,256]`
- `TradingPolicyNetwork`：`mlp_extractor` 兼容类，含 `latent_dim_pi=128` / `latent_dim_vf=128`
- `TradingPolicy`：继承 `ActorCriticPolicy`，重写 `_build_mlp_extractor()`，完整端到端策略

---

### `env/trading_env.py`

```python
observation_space: Box(shape=(60, 244), dtype=float32)
action_space:      Discrete(3)  # 0=买, 1=卖, 2=持有
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `context_window` | 60 | 观测窗口（天） |
| `episode_length` | 250 | 每episode长度（约1年） |
| `initial_capital` | 1,000,000 | 初始资金（元） |
| `transaction_cost` | 0.001 | 单边手续费率 |
| `slippage` | 0.001 | 单边滑点率 |
| `max_dd_penalty` | 0.05 | 回撤惩罚阈值（5%） |
| `random_start` | True | 随机起始点（防过拟合） |

**Reward 设计**：
- 基础 reward = `daily_return / std(returns[-30:])`（Sharpe增量）
- 若当日回撤 > `max_dd_penalty`，额外扣 `drawdown` 分
- 买卖在下一交易日开盘价执行（模拟真实T+1，避免未来泄漏）
- 已持仓时买入 / 未持仓时卖出 → no-op

---

### `train/trainer.py`

**Walk-Forward 分折**：

| 折 | 训练集 | 验证集 |
|---|---|---|
| Fold 1 | 2015–2019 | 2020 |
| Fold 2 | 2015–2020 | 2021 |
| Fold 3 | 2015–2021 | 2022 |

取3折验证集 Sharpe 均值作为模型选择依据，测试集（2024）训练过程中不碰。

**PPO 默认配置**：

| 参数 | 值 |
|---|---|
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `total_timesteps` | 1,000,000 |

---

### `eval/backtest.py`

| 函数 | 说明 |
|---|---|
| `compute_metrics(daily_returns)` | 计算全套指标 |
| `run_backtest(model, features, close_prices)` | 完整回测，从头到尾 |
| `compare_with_benchmark(model_returns, bench_returns)` | 对比买入持有基准 |
| `print_metrics(metrics, title)` | 格式化打印 |

**评估指标**：年化收益率、Sharpe Ratio（目标 >1.5）、最大回撤（目标 <20%）、胜率、盈亏比、年化波动率、Calmar Ratio

---

## 使用流程

### Step 1：安装依赖

```bash
pip install -r requirements.txt
```

### Step 2：数据预处理

```bash
# 仅K线数据
python scripts/preprocess.py \
    --csv data/raw/sh600318_2026.csv \
    --output data/processed/sh600318_features.npz

# K线 + 财报数据
python scripts/preprocess.py \
    --csv data/raw/sh600318_2026.csv \
    --financial data/financial/000001_financial.csv \
    --output data/processed/sh600318_features.npz
```

输出 `.npz` 文件包含：
- `features`：`[N_days, 244]` float32
- `close_prices`：`[N_days]` float32
- `dates`：`[N_days]` 日期字符串

### Step 3：训练

```bash
python scripts/train.py \
    --data data/processed/sh600318_features.npz \
    --config configs/default.yaml \
    --save-dir checkpoints \
    --n-envs 4
```

### Step 4：测试集回测

```bash
python scripts/train.py \
    --data data/processed/sh600318_features.npz \
    --test-only checkpoints/fold_2022.zip
```

---

## 注意事项

1. **数据泄漏检查**：归一化统计量、财报数据，都严格使用发布时间点之前的数据
2. **因果卷积**：周/月CNN使用左侧单向padding，确保不看未来
3. **每日K线根数**：自动检测（本项目数据为241根，09:30–11:30 + 13:01–15:00）
4. **先跑单只股票**：验证pipeline完整性后再扩展到全股票池
5. **GPU支持**：预处理脚本自动检测CUDA，单卡可训练

---

## 文件依赖关系

下图展示各源文件的 import 依赖（箭头方向 = 被依赖方向）：

```
scripts/preprocess.py
  ├──► src/data/dataset.py
  ├──► src/data/financial.py
  ├──► src/data/features.py
  │       └──► src/models/cnn_encoder.py
  └──► src/models/cnn_encoder.py

scripts/train.py
  ├──► src/train/trainer.py
  │       ├──► src/env/trading_env.py
  │       ├──► src/eval/backtest.py
  │       └──► src/models/actor_critic.py
  │               └──► src/models/transformer.py
  └──► src/eval/backtest.py
          └──► src/env/trading_env.py
```

**无环依赖层次（从底层到顶层）**：

```
Layer 0（基础，无内部依赖）
  src/data/dataset.py
  src/data/financial.py
  src/models/transformer.py

Layer 1（依赖 Layer 0）
  src/models/cnn_encoder.py
  src/models/actor_critic.py  ──► transformer.py

Layer 2（依赖 Layer 0/1）
  src/data/features.py        ──► cnn_encoder.py
  src/env/trading_env.py

Layer 3（依赖 Layer 0/1/2）
  src/eval/backtest.py        ──► trading_env.py
  src/train/trainer.py        ──► trading_env.py, backtest.py, actor_critic.py

Layer 4（入口）
  scripts/preprocess.py       ──► dataset, financial, features, cnn_encoder
  scripts/train.py            ──► trainer, backtest
```

---

## 函数调用关系

### `src/data/dataset.py` 内部调用链

```
load_raw_data(csv_path)
  └── pl.read_csv()
  └── df.rename(CN_COL_MAP)          # 中文列名 → 英文

preprocess_raw_data(df)
  └── str.to_datetime()
  └── df.with_columns(dt.date())     # 提取日期列
  └── df.drop_nulls()
  └── group_by("date").agg(sum)      # 计算每日成交量
  └── df.join(valid_dates)           # 过滤停牌日

split_by_day(df, bars_per_day=None)
  ├── detect_bars_per_day(df)        # bars_per_day=None 时自动调用
  │     └── group_by("date").agg(len)
  │     └── Counter(counts).most_common(1)   # 取众数
  └── group_by("date", maintain_order=True)
        └── group.sort("timestamp")
        └── group.select(OHLCV).to_numpy()   # → [241, 5]

normalize_features(day_data, lookback_window=60)
  └── [Loop d=0..N_days]
        └── log(price[t]/price[t-1])         # OHLC → 对数收益率
        └── log1p(volume)                    # 成交量 → log(vol+1)
        └── hist = returns[d-lookback:d]
        └── (returns[d] - hist.mean) / hist.std   # 因果标准化
```

### `src/data/features.py` 内部调用链

```
build_features(day_data_norm, financial_features, cnn_encoder, device)
  ├── CNNEncoder().to(device).eval()          # 若 cnn_encoder=None 则新建
  ├── torch.from_numpy(day_data_norm)         # ndarray → Tensor [N, 241, 5]
  ├── cnn_encoder.forward(x)                 # → [N, 212]
  └── np.concatenate([kline_feat, financial]) # → [N, 244]

build_features_no_financial(day_data_norm, ...)
  └── build_features(day_data_norm, zeros(N,32), ...)  # 财报填0
```

### `src/models/cnn_encoder.py` 内部调用链

```
CNNEncoder.forward(x: [N, 241, 5])
  ├── x.permute(0,2,1)                        # → [N, 5, 241]
  ├── day_conv(x_in).squeeze(-1)              # Conv1d(5→20,k=241) → [N, 20]
  ├── day_feat.unsqueeze(0).permute(0,2,1)    # → [1, 20, N]
  ├── week_conv(seq)                          # CausalConv1d → [1, 64, N]
  │     └── F.pad(x, (kernel-1, 0))           # 左侧因果padding
  │     └── conv(x_padded)                   # Conv1d(无padding)
  │     └── BN → GELU
  ├── month_conv(seq)                         # CausalConv1d → [1, 128, N]
  └── torch.cat([day(20), week(64), month(128)], dim=-1)   # → [N, 212]
```

### `src/models/transformer.py` 内部调用链

```
TradingTransformer.forward(x: [B, 60, 244])
  ├── input_proj(x)                           # Linear(244→256) → [B,60,256]
  ├── * sqrt(d_model)                         # 缩放，稳定训练
  ├── pos_enc(x)                              # Embedding(60,256) broadcast加法
  ├── _causal_mask(seq_len=60, device)        # 上三角bool mask [60,60]
  ├── TransformerEncoder(x, mask=causal_mask) # 4层 Pre-LN Encoder
  │     └── [×4] TransformerEncoderLayer
  │           ├── LayerNorm → MultiheadAttention(mask) → dropout → residual
  │           └── LayerNorm → FFN(Linear→GELU→Linear) → dropout → residual
  └── x[:, -1, :]                             # 取最后时间步 → [B, 256]
```

### `src/models/actor_critic.py` 内部调用链

```
TradingPolicy.__init__()
  └── super().__init__()                      # ActorCriticPolicy
        ├── TradingFeaturesExtractor.__init__()
        │     └── TradingTransformer.__init__()
        └── _build_mlp_extractor()            # 被 super().__init__ 调用
              └── TradingPolicyNetwork.__init__()
                    ├── actor_net: Linear(256,128) → GELU
                    └── critic_net: Linear(256,128) → GELU
        # SB3 自动在上面加：
        # action_net: Linear(128, 3)
        # value_net:  Linear(128, 1)

TradingPolicy.forward(obs)  [SB3内部调用]
  ├── TradingFeaturesExtractor.forward(obs)   # [B,60,244] → [B,256]
  │     └── TradingTransformer.forward(obs)
  ├── TradingPolicyNetwork.forward(features)
  │     └── (actor_net(f), critic_net(f))     # → ([B,128], [B,128])
  ├── action_net(latent_pi)                   # Linear(128,3) → logits
  ├── Categorical(logits).sample()            # 采样动作
  └── value_net(latent_vf)                    # Linear(128,1) → value
```

### `src/env/trading_env.py` 内部调用链

```
StockTradingEnv.reset()
  └── np_random.integers(min_start, max_start)   # 随机起始点
  └── _get_obs()
        └── features[current_idx-60 : current_idx]  # → [60, 244]

StockTradingEnv.step(action)
  ├── _get_open_price(next_idx)               # 下一日收盘价近似开盘价
  ├── [action==0, position==0] 买入
  │     └── exec_price = next_open × (1 + cost + slippage)
  │     └── shares = cash / exec_price
  ├── [action==1, position==1] 卖出
  │     └── exec_price = next_open × (1 - cost - slippage)
  │     └── cash = shares × exec_price
  ├── portfolio_value = shares × next_close   # 或 cash
  ├── daily_return = (value - prev_value) / prev_value
  ├── _compute_reward(daily_return)
  │     └── std(returns[-30:])
  │     └── return daily_return / std          # Sharpe增量
  ├── [drawdown > threshold] reward -= drawdown
  └── _get_obs()                              # 下一观测
```

### `src/train/trainer.py` 内部调用链

```
train_walk_forward(features, close_prices, dates, cfg)
  └── [Loop 3 folds]
        └── train_fold(fold_train, fold_val, ...)
              ├── _slice_by_year(features, close_prices, dates, ...)
              │     └── filter by year → (train_feat, train_close)
              │                        → (val_feat,   val_close)
              ├── make_vec_env(make_env(...), n_envs=4, SubprocVecEnv)
              │     └── make_env() → StockTradingEnv.__init__()
              ├── PPO(TradingPolicy, env, **ppo_cfg)
              │     └── TradingPolicy.__init__() [见上文]
              ├── model.learn(total_timesteps)
              │     └── [SB3 PPO内部]
              │           └── collect_rollouts() → env.step() × n_steps
              │           └── train() → policy.forward() × n_epochs
              ├── model.save(fold_path)
              └── evaluate_on_val(model, val_feat, val_close, env_cfg)
                    └── StockTradingEnv(val_feat, val_close, random_start=True)
                    └── [n_eval_episodes次]
                          └── env.reset() → obs
                          └── [Loop] model.predict(obs) → action → env.step(action)
                    └── compute_metrics(flat_returns)
```

---

## 训练流程完整调用栈

```
scripts/train.py: main()
│
├─① load_processed_data("*.npz")
│     np.load → features[N,244], close_prices[N], dates[N]
│
├─② load_config("configs/default.yaml")
│     yaml.safe_load → cfg dict
│
└─③ train_walk_forward(features, close_prices, dates, cfg)
      │
      ├─[Fold 1/2/3] train_fold(train_years, val_year, ...)
      │   │
      │   ├─ _slice_by_year → train_feat[T1,244], val_feat[T2,244]
      │   │
      │   ├─ make_vec_env × 4 进程
      │   │     └─ StockTradingEnv(train_feat, close_prices)
      │   │           observation_space: Box(60, 244)
      │   │           action_space:      Discrete(3)
      │   │
      │   ├─ PPO(TradingPolicy, vec_env, lr=3e-4, ...)
      │   │     │
      │   │     └─ TradingPolicy 构建
      │   │           ├─ TradingFeaturesExtractor
      │   │           │     └─ TradingTransformer(input=244, d_model=256)
      │   │           │           ├─ Linear(244→256)
      │   │           │           ├─ Embedding(60, 256)   位置编码
      │   │           │           └─ TransformerEncoder × 4层
      │   │           ├─ TradingPolicyNetwork
      │   │           │     ├─ actor_net: Linear(256→128)→GELU
      │   │           │     └─ critic_net: Linear(256→128)→GELU
      │   │           ├─ action_net: Linear(128→3)        [SB3自动]
      │   │           └─ value_net:  Linear(128→1)        [SB3自动]
      │   │
      │   ├─ model.learn(1_000_000 steps)
      │   │     │
      │   │     └─ [PPO 每 2048 步更新一次，共 ~488 次更新]
      │   │           │
      │   │           ├─ collect_rollouts(n_steps=2048)
      │   │           │     └─ [Loop 2048次]
      │   │           │           ├─ obs → TradingFeaturesExtractor.forward
      │   │           │           │         → TradingTransformer.forward
      │   │           │           │           → [B,60,256 proj+pe] → Encoder → [B,256]
      │   │           │           ├─ actor_net → action_net → Categorical.sample()
      │   │           │           ├─ critic_net → value_net → value scalar
      │   │           │           └─ StockTradingEnv.step(action)
      │   │           │                 ├─ 买卖逻辑（next_open执行）
      │   │           │                 ├─ portfolio_value 更新
      │   │           │                 └─ reward = daily_return / std30 - dd_penalty
      │   │           │
      │   │           └─ train(batch_size=64, n_epochs=10)
      │   │                 └─ [每epoch遍历所有mini-batch]
      │   │                       ├─ policy.evaluate_actions(obs, actions)
      │   │                       │     └─ forward → log_probs, values, entropy
      │   │                       ├─ ratio = exp(new_log_prob - old_log_prob)
      │   │                       ├─ L_clip = min(ratio×adv, clip(ratio,0.8,1.2)×adv)
      │   │                       ├─ L_value = 0.5 × (value - return)²
      │   │                       ├─ L_entropy = -0.01 × entropy
      │   │                       └─ loss.backward() → optimizer.step()
      │   │
      │   └─ evaluate_on_val(model, val_feat, val_close)
      │         └─ [5 episodes] model.predict → env.step → daily_returns
      │         └─ compute_metrics → sharpe_ratio
      │
      └─ 返回 best_model_path（验证集Sharpe最高的fold）

      └─④ PPO.load(best_model_path)
            └─ run_backtest(model, test_feat, test_close)  [见推理流程]
```

---

## 推理/回测流程完整调用栈

```
scripts/train.py: main() --test-only  (或训练后自动执行)
│
├─① PPO.load("checkpoints/fold_2022.zip")
│     └─ 恢复 TradingPolicy 所有权重
│
└─② run_backtest(model, test_feat[N,244], test_close[N])
      │
      ├─ StockTradingEnv(features=test_feat, random_start=False)
      │     └─ episode_length = N - context_window - 1（跑完整测试集）
      │
      ├─ env.reset() → obs[60, 244]
      │     └─ _current_idx = context_window（从第60天开始）
      │     └─ _get_obs() → features[0:60]
      │
      └─ [Loop until done]
            │
            ├─ model.predict(obs, deterministic=True)
            │     │
            │     └─ TradingPolicy.forward(obs[1,60,244])
            │           ├─ TradingFeaturesExtractor.forward
            │           │     └─ TradingTransformer.forward
            │           │           ├─ input_proj: [1,60,244]→[1,60,256]
            │           │           ├─ + pos_enc: [1,60,256]
            │           │           ├─ causal_mask [60,60] 上三角遮蔽
            │           │           ├─ TransformerEncoder × 4
            │           │           │     每层: LayerNorm→MHA(causal)→FFN
            │           │           └─ x[:,−1,:] → [1,256]  取最后时间步
            │           ├─ actor_net([1,256]) → [1,128]
            │           ├─ action_net([1,128]) → logits[1,3]
            │           └─ argmax(logits) → action ∈ {0买, 1卖, 2持有}
            │
            ├─ env.step(action)
            │     ├─ next_open = close_prices[idx+1]  (T+1模拟)
            │     ├─ [买] shares = cash / (next_open × 1.002)
            │     ├─ [卖] cash = shares × (next_open × 0.998)
            │     ├─ portfolio_value = shares × next_close  or  cash
            │     ├─ daily_return = (value - prev) / prev
            │     ├─ _compute_reward → daily_return / std(last30)
            │     ├─ drawdown check → extra penalty if > 5%
            │     └─ _get_obs() → features[idx-59:idx+1]  滑动窗口
            │
            └─ [done] compute_metrics(daily_returns)
                  ├─ annualized_return = prod(1+r)^(252/N) - 1
                  ├─ sharpe_ratio      = mean(r-rf) / std(r-rf) × √252
                  ├─ max_drawdown      = max(peak-val)/peak
                  ├─ win_rate          = count(r>0) / N
                  ├─ profit_loss_ratio = mean(r>0) / mean(|r<0|)
                  ├─ annualized_vol    = std(r) × √252
                  └─ calmar_ratio      = ann_return / max_drawdown

      └─③ compare_with_benchmark(model_returns, bench_returns)
            ├─ compute_metrics(model_returns)     → model指标
            └─ compute_metrics(bench_returns)     → 买入持有指标

      └─④ print_metrics(metrics, title)
            格式化输出各项指标
```

---

## 预处理流程完整调用栈

```
scripts/preprocess.py: main()
│
├─① load_raw_data("data/raw/*.csv")
│     ├─ pl.read_csv()
│     ├─ df.rename({中文列名: 英文列名})    # 兼容中英文格式
│     └─ df.select(["timestamp","open","high","low","close","volume"])
│
├─② preprocess_raw_data(df)
│     ├─ str.to_datetime(format="%Y-%m-%d %H:%M:%S")
│     ├─ dt.date() → 提取日期列
│     ├─ drop_nulls()
│     └─ group_by("date").agg(sum("volume"))
│           → join(valid_dates) 过滤停牌日
│
├─③ split_by_day(df, bars_per_day=None)
│     ├─ detect_bars_per_day(df)
│     │     └─ group_by("date").agg(len) → Counter → 众数=241
│     └─ group_by("date") iterate
│           └─ group.sort("timestamp")
│           └─ [len==241] to_numpy() → append
│     返回: day_data[N,241,5], dates[N]
│
├─④ normalize_features(day_data, lookback=60)
│     └─ [Loop d=0..N]
│           ├─ compute log-returns(OHLC): log(p[t]/p[t-1])
│           ├─ log1p(volume)
│           ├─ hist = returns[max(0,d-60):d]
│           └─ normalized[d] = (returns[d] - hist.mean) / hist.std
│     返回: normalized[N,241,5]
│
├─⑤ CNNEncoder(intraday_bars=241)
│     ├─ day_conv:   Conv1d(5→20, k=241, stride=241)
│     ├─ week_conv:  CausalConv1d(20→64, k=5)
│     └─ month_conv: CausalConv1d(20→128, k=20)
│
├─⑥ build_financial_features(fin_df, dates)  [可选]
│     └─ [Loop each trading_date]
│           └─ 找最新 publish_date ≤ trading_date 的财报
│           └─ tanh(raw_8 @ PROJ_W)  → [32]
│     返回: financial_feat[N,32]
│
├─⑦ build_features(normalized, financial_feat, cnn, device)
│     ├─ cnn.eval() → torch.no_grad()
│     ├─ CNNEncoder.forward([N,241,5])
│     │     ├─ day_conv → [N,20]
│     │     ├─ week_conv (causal) → [N,64]
│     │     └─ month_conv (causal) → [N,128]
│     │     cat → [N,212]
│     └─ np.concatenate([kline[N,212], financial[N,32]]) → [N,244]
│
└─⑧ np.savez_compressed(out_path,
        features=features[N,244],
        close_prices=day_data[:,-1,3],   # 每天最后收盘价
        dates=dates[N])
```
