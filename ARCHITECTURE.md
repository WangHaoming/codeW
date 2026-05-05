# 股票交易强化学习系统 — 架构实现文档

## 项目概述

基于10年历史1分钟K线数据 + 季度财报数据，使用 CNN + Transformer + RL（PPO）
构建端到端的股票交易决策系统。模型直接输出买/卖/持有动作，以 Sharpe Ratio 作为优化目标。

---

## 技术栈

- **语言**：Python 3.10+
- **深度学习**：PyTorch 2.x
- **RL框架**：Stable-Baselines3（PPO）
- **回测**：VectorBT
- **数据处理**：Polars + NumPy
- **实验追踪**：Weights & Biases（可选）

---

## 目录结构

```
codeW/
├── data/
│   ├── raw/                  # 原始1分钟K线CSV
│   ├── processed/            # 预处理后的数据
│   └── financial/            # 季度财报数据CSV
├── src/
│   ├── data/
│   │   ├── dataset.py        # 数据加载与预处理
│   │   ├── features.py       # 特征工程
│   │   └── financial.py      # 财报数据处理
│   ├── models/
│   │   ├── cnn_encoder.py    # CNN特征提取器
│   │   ├── transformer.py    # Transformer序列建模
│   │   └── actor_critic.py   # Actor-Critic网络
│   ├── env/
│   │   └── trading_env.py    # 交易环境（Gym接口）
│   ├── train/
│   │   └── trainer.py        # 训练主循环
│   └── eval/
│       └── backtest.py       # 回测与评估
├── configs/
│   └── default.yaml          # 超参数配置
├── scripts/
│   ├── preprocess.py         # 数据预处理入口
│   └── train.py              # 训练入口
└── requirements.txt
```

---

## 数据格式约定

### 1分钟K线数据（CSV）

```
timestamp, open, high, low, close, volume
2015-01-05 09:30:00, 10.5, 10.8, 10.4, 10.7, 1000000
2015-01-05 09:31:00, 10.7, 10.9, 10.6, 10.8, 800000
...
```

### 财报数据（CSV）

```
stock_code, publish_date, report_period, revenue_growth, net_profit_margin,
roe, debt_ratio, gross_margin, operating_cashflow, rd_ratio
000001, 2015-04-30, 2015Q1, 0.12, 0.18, 0.15, 0.45, 0.35, 0.08, 0.03
...
```

**重要**：必须使用 `publish_date`（实际发布日期），不能用 `report_period`，防止数据泄漏。

---

## 数据预处理流程（`src/data/dataset.py`）

### Step 1：原始数据清洗

```python
def preprocess_raw_data(df):
    """
    输入：原始1分钟K线 DataFrame
    输出：清洗后的数据

    处理逻辑：
    1. 删除停牌日（成交量为0的整天）
    2. 删除涨跌停异常值（可选，保留也可）
    3. 按交易日分组，确保每天精确240根K线
       - A股交易时间：09:30-11:30（120分钟）+ 13:00-15:00（120分钟）= 240分钟
    4. 缺失分钟用前值填充（ffill），不超过5分钟
    """
```

### Step 2：按天切分

```python
def split_by_day(df):
    """
    输入：清洗后的连续时序 DataFrame
    输出：shape [N_days, 240, 5] 的 numpy array

    5个特征：[open, high, low, close, volume]

    注意：
    - 按交易日分组，每组必须恰好240行
    - 不足240行的交易日直接丢弃（数据质量问题）
    - 返回对应的日期索引列表
    """
```

### Step 3：归一化（关键：防止未来泄漏）

```python
def normalize_features(day_data, lookback_window=60):
    """
    输入：[N_days, 240, 5] 的原始数据
    输出：[N_days, 240, 5] 的归一化数据

    归一化策略：
    - 第d天的归一化统计量，只用第(d-lookback_window)到第(d-1)天计算
    - 绝对不能用全局均值/方差（会泄漏未来信息）
    - 价格特征（open/high/low/close）：转换为相对收益率再归一化
      close_return[t] = (close[t] - close[t-1]) / close[t-1]
    - 成交量：取log后归一化
      volume_log = log(volume + 1)
    """
```

---

## CNN特征提取器（`src/models/cnn_encoder.py`）

### 架构说明

三路并行CNN，分别提取日、周、月三个时间尺度的特征：

```
输入：[batch, 240, 5]  （一天的1分钟数据）

路径1（日特征）：
  Conv1d(in=5, out=20, kernel=240, stride=240, padding=0) → [batch, 20, 1]
  → Squeeze → [batch, 20]

（路径1的输出 [N_days, 20] 作为后续周/月CNN的输入）

路径2（周特征）：
  输入：[N_days, 20]  （日特征序列）
  Conv1d(in=20, out=64, kernel=5, stride=1, padding=2) → [N_days, 64]
  注：padding=2 保持序列长度不变，causal padding（只看过去）

路径3（月特征）：
  输入：[N_days, 20]  （日特征序列）
  Conv1d(in=20, out=128, kernel=20, stride=1, padding=19) → [N_days, 128]
  注：causal padding，只用过去20天

最终每天特征：Concat([20维, 64维, 128维]) = [N_days, 212维]
```

### 实现要点

```python
class CNNEncoder(nn.Module):
    """
    重要实现细节：

    1. 因果卷积（Causal Convolution）：
       周/月CNN必须只看过去，不看未来
       实现方式：左侧padding，右侧不padding
       kernel=5: 左padding=4，然后output[:, :, :-4] 去掉右侧多余部分
       或者直接用 padding=(kernel_size-1, 0) 的方式

    2. 激活函数：GELU 或 ReLU

    3. Batch Normalization：在每个Conv层后加

    4. 日内CNN（路径1）不需要因果约束，
       因为一天的数据在决策时已全部可见（收盘后决策）
    """
```

---

## 财报特征处理（`src/data/financial.py`）

```python
def build_financial_features(financial_df, trading_dates):
    """
    输入：
      financial_df: 财报原始数据
      trading_dates: 交易日列表

    输出：[N_days, 32] 的财报特征矩阵

    处理逻辑：
    1. 以 publish_date 为准（非report_period）
    2. 每个交易日，找到该日之前最新发布的财报
    3. 将财报数值编码为32维向量（MLP或直接线性层）
    4. 上市不足一年的股票，财报特征全部填0

    财报原始字段（8个）→ MLP → 32维：
    [revenue_growth, net_profit_margin, roe, debt_ratio,
     gross_margin, operating_cashflow, rd_ratio, yoy_net_profit]
    """
```

---

## Transformer建模（`src/models/transformer.py`）

```python
class TradingTransformer(nn.Module):
    """
    输入：[batch, 60, 244]
      - 60：context window（过去60个交易日，约3个月）
      - 244：212维K线特征 + 32维财报特征

    超参数：
      d_model = 256
      nhead = 8
      num_layers = 4
      dropout = 0.1
      dim_feedforward = 512

    输出：[batch, 256]  （当前时刻的状态表示）

    实现要点：
    1. 输入线性投影：244 → 256
    2. 位置编码：使用可学习的位置编码（非固定sin/cos）
    3. 因果Mask：Transformer内部使用causal mask，
       每个时间步只能看到过去的时间步
    4. 取最后一个时间步的输出作为状态表示
    """
```

---

## Actor-Critic网络（`src/models/actor_critic.py`）

```python
class ActorCritic(nn.Module):
    """
    输入：[batch, 256]  （Transformer输出的状态表示）

    Actor（策略网络）：
      Linear(256, 128) → GELU
      Linear(128, 3)   → Softmax
      输出：[buy概率, sell概率, hold概率]

    Critic（价值网络）：
      Linear(256, 128) → GELU
      Linear(128, 1)
      输出：当前状态的预期累积reward（标量）

    动作空间：离散，3个动作
      0 = 买入（全仓）
      1 = 卖出（清仓）
      2 = 持有（不操作）
    """
```

---

## 交易环境（`src/env/trading_env.py`）

### Gym接口实现

```python
class StockTradingEnv(gym.Env):
    """
    observation_space: Box(shape=(60, 244), dtype=float32)
    action_space: Discrete(3)  # 0=买, 1=卖, 2=持有

    初始化参数：
      features: [N_days, 244] 的特征矩阵
      context_window: 60  # 每次观测看过去60天
      initial_capital: 1_000_000  # 初始资金（元）
      transaction_cost: 0.001  # 千分之一手续费（单边）
      slippage: 0.001  # 千分之一滑点

    state（观测）：
      过去60天的特征矩阵 [60, 244]

    step(action)返回：
      next_state, reward, done, info

    reward设计（Sharpe增量）：
      每步reward = 当日收益率 / 过去30日收益率标准差
      惩罚项：如果当日最大回撤超过5%，额外扣分

    episode设计：
      一个episode = 一年的交易数据（约250天）
      随机起始点，避免过拟合特定时段
    """

    def reset(self):
        """随机选择起始点，返回初始观测"""

    def step(self, action):
        """
        执行动作，计算reward
        注意：
        - 买入/卖出在下一根K线开盘价执行（模拟真实情况）
        - 已持仓时再次买入 → 忽略（no-op）
        - 未持仓时执行卖出 → 忽略（no-op）
        """
```

---

## 训练流程（`src/train/trainer.py`）

### Walk-Forward验证（防止过拟合）

```
数据划分：
  训练集：2015-2022（8年）
  验证集：2023（1年）
  测试集：2024（1年，最终评估，训练过程中不碰）

Walk-Forward分折：
  Fold1: Train=2015-2019, Val=2020
  Fold2: Train=2015-2020, Val=2021
  Fold3: Train=2015-2021, Val=2022
  取3个fold的验证集Sharpe均值作为模型选择依据
```

### PPO训练配置

```python
ppo_config = {
    "learning_rate": 3e-4,
    "n_steps": 2048,        # 每次更新前收集的步数
    "batch_size": 64,
    "n_epochs": 10,         # 每批数据的更新轮数
    "gamma": 0.99,          # 折扣因子
    "gae_lambda": 0.95,     # GAE参数
    "clip_range": 0.2,      # PPO clip参数
    "ent_coef": 0.01,       # 熵正则，鼓励探索
    "vf_coef": 0.5,         # Critic loss权重
    "max_grad_norm": 0.5,   # 梯度裁剪
    "total_timesteps": 1_000_000
}
```

### 训练入口

```python
# scripts/train.py 的逻辑

# 1. 加载预处理好的特征数据
# 2. 创建训练环境（VecEnv，多个并行环境加速）
# 3. 初始化自定义Policy（包含CNN+Transformer+ActorCritic）
# 4. 使用Stable-Baselines3的PPO训练
# 5. 每隔N步在验证集上评估Sharpe
# 6. 保存最优模型

# 注意：SB3的自定义Policy需要继承 ActorCriticPolicy
# 并重写 _build_mlp_extractor() 替换为我们的CNN+Transformer
```

---

## 评估指标（`src/eval/backtest.py`）

```python
evaluation_metrics = {
    "年化收益率": "annualized_return",
    "Sharpe Ratio": "sharpe_ratio",          # 主要指标，目标 > 1.5
    "最大回撤": "max_drawdown",              # 目标 < 20%
    "胜率": "win_rate",                      # 盈利交易次数/总交易次数
    "盈亏比": "profit_loss_ratio",
    "年化波动率": "annualized_volatility",
    "Calmar Ratio": "calmar_ratio",          # 年化收益/最大回撤
}

# 基准对比：
# - 沪深300指数买入持有
# - 随机动作基准
```

---

## 超参数配置（`configs/default.yaml`）

```yaml
data:
  context_window: 60          # Transformer输入的天数
  lookback_for_norm: 60       # 归一化用的历史窗口
  train_years: [2015, 2022]
  val_year: 2023
  test_year: 2024

model:
  cnn:
    day_out_dim: 20
    week_out_dim: 64
    month_out_dim: 128
    week_kernel: 5
    month_kernel: 20
  financial:
    input_dim: 8
    output_dim: 32
  transformer:
    d_model: 256
    nhead: 8
    num_layers: 4
    dropout: 0.1
    dim_feedforward: 512
  actor_critic:
    hidden_dim: 128
    n_actions: 3              # buy / sell / hold

env:
  initial_capital: 1_000_000
  transaction_cost: 0.001
  slippage: 0.001
  max_drawdown_penalty: 0.05  # 超过5%回撤开始惩罚

ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  total_timesteps: 1_000_000
```

---

## 实现顺序建议

```
Step 1：数据预处理
  - 实现 dataset.py：读取CSV → 按天切分 → 归一化
  - 验证输出shape：[N_days, 240, 5]
  - 单元测试：检查归一化没有用到未来数据

Step 2：CNN编码器
  - 实现 cnn_encoder.py
  - 输入 [batch, 240, 5] → 输出 [batch, 212]
  - 验证因果卷积的正确性

Step 3：财报特征
  - 实现 financial.py
  - 验证每天对应的财报是发布日之前最新的

Step 4：Transformer
  - 实现 transformer.py
  - 输入 [batch, 60, 244] → 输出 [batch, 256]

Step 5：交易环境
  - 实现 trading_env.py
  - 用随机动作跑通一个episode，检查reward计算正确

Step 6：PPO训练
  - 接入SB3，自定义Policy
  - 先在小数据集（1年数据）跑通训练loop

Step 7：评估与回测
  - 实现 backtest.py
  - 对比基准（买入持有）
```

---

## 注意事项

1. **数据泄漏检查**：归一化统计量、财报数据，都必须严格用发布时间点之前的数据
2. **因果卷积**：周/月CNN的padding必须是左侧单向padding
3. **环境随机性**：episode起始点随机，避免模型记忆特定时段
4. **先跑单只股票**：验证pipeline完整性后再扩展到全股票池
5. **GPU内存**：Transformer的context_window=60，244维输入，内存压力不大，单卡可训练