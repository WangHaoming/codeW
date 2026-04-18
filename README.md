# A-share Quant Research

本项目是一个本地运行的 A 股量化研究与回测脚手架，使用 AkShare 拉取日线数据，支持最小可用的策略研究流程。

## 能干什么
- 拉取 A 股个股日线数据（AkShare）
- 生成策略信号（示例：均线交叉）
- 进行简易回测（长/空仓）
- 输出基础绩效指标（总收益、年化收益、最大回撤、夏普）

## 如何实现
核心流程是“数据 → 信号 → 回测 → 指标”。对应模块如下：

1. 数据层：`data.py`
   - 使用 `ak.stock_zh_a_hist` 拉取指定股票的日线数据
   - 统一列名与日期格式，保证后续计算一致性

2. 策略层：`strategy.py`
   - 以均线交叉为示例策略
   - 当短期均线高于长期均线时持仓，否则空仓

3. 回测层：`backtest.py`
   - 以“次日开盘价”成交作为执行价格
   - 全仓进出，不做部分仓位
   - 变更仓位时计入手续费

4. 指标层：`metrics.py`
   - 根据权益曲线计算总收益、年化收益、最大回撤、夏普

5. 运行脚本：`scripts/run_backtest.py`
   - 串联数据、策略、回测、指标输出
   - 提供命令行参数，便于快速实验

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/run_backtest.py \
  --symbol 000001 \
  --start 20180101 \
  --end 20240101 \
  --short 20 \
  --long 60
```

## 分钟级量价散点图

`scripts/plot_minute_return_volume.py` 支持两个子命令：

### 下载数据

从 AkShare 拉取分钟 K 线并保存到 `data/` 目录：

```bash
# 默认拉取最近一周的 1 分钟数据
python scripts/plot_minute_return_volume.py download --symbol 000001

# 指定周期和时间范围
python scripts/plot_minute_return_volume.py download \
  --symbol 000001 \
  --period 5 \
  --start-date "2026-02-01 09:30:00" \
  --end-date "2026-02-10 15:00:00"
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--symbol` | 6 位股票代码 | 000001 |
| `--period` | 分钟周期（1/5/15/30/60） | 1 |
| `--start-date` | 起始时间 YYYY-MM-DD HH:MM:SS | 最近一周 |
| `--end-date` | 结束时间 YYYY-MM-DD HH:MM:SS | 当天 15:00 |

### 画图

读取已有 CSV 文件，绘制收益率 vs 成交量散点图（点击散点可查看详细信息）：

```bash
# 弹窗显示
python scripts/plot_minute_return_volume.py plot \
  --file data/minute_000001_1min.csv \
  --symbol 000001 \
  --show

python scripts/plot_minute_return_volume.py plot \
  --file data/minute_000001_1min.csv \
  --symbol 000001 \
  --chart price_volume \
  --highlight-start "2026-02-09 09:30:00" \
  --highlight-minutes 30 \
  --show

# 保存为图片
python scripts/plot_minute_return_volume.py plot \
  --file data/minute_000001_1min.csv \
  --symbol 000001 \
  --output data/chart.png
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--file` | CSV 文件路径（必填） | - |
| `--symbol` | 图表标题中的股票代码 | 000001 |
| `--output` | 保存图片路径 | 不保存 |
| `--show` | 弹窗显示图表 | 否 |

## Project Structure
```
codeW/
  src/a_share_trading/
    data.py
    strategy.py
    backtest.py
    metrics.py
  scripts/
    run_backtest.py
    plot_minute_return_volume.py
    run_tick_tx.py
  data/
  notebooks/
  tests/
```

## 注意事项
- 仅用于研究与回测，不适用于实盘交易。
- 回测逻辑极简，未考虑涨跌停、停牌、滑点、交易时段等真实约束。
