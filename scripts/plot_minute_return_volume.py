from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd


def _default_range() -> tuple[str, str]:
    tz = ZoneInfo("Asia/Shanghai")
    now = dt.datetime.now(tz)
    end = now.replace(hour=15, minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(days=7)
    start = start.replace(hour=9, minute=30, second=0, microsecond=0)
    return (
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _last_trading_window(days: int = 5) -> tuple[str, str]:
    trade_df = ak.tool_trade_date_hist_sina()
    if trade_df.empty or "trade_date" not in trade_df.columns:
        raise ValueError("Failed to load trading calendar from AkShare.")
    trade_dates = pd.to_datetime(trade_df["trade_date"]).dt.date.sort_values()
    today = dt.datetime.now(ZoneInfo("Asia/Shanghai")).date()
    valid_dates = [d for d in trade_dates if d <= today]
    if not valid_dates:
        raise ValueError("No valid trading dates found from calendar.")
    end_date = valid_dates[-1]
    start_idx = max(0, len(valid_dates) - days)
    start_date = valid_dates[start_idx]
    return (
        f"{start_date:%Y-%m-%d} 09:30:00",
        f"{end_date:%Y-%m-%d} 15:00:00",
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "日期": "datetime",
        "时间": "datetime",
        "day": "datetime",
        "开盘": "open",
        "open": "open",
        "收盘": "close",
        "close": "close",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "成交量": "volume",
        "volume": "volume",
        "成交额": "amount",
        "amount": "amount",
        "均价": "avg_price",
        "avg_price": "avg_price",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    missing = [c for c in ("datetime", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after rename: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    if "open" in df.columns:
        zero_mask = df["open"] == 0
        if zero_mask.any():
            is_open = df["datetime"].dt.time == dt.time(9, 30)
            df.loc[zero_mask & ~is_open, "open"] = df["close"].shift(1)
            df.loc[zero_mask & is_open, "open"] = df.loc[zero_mask & is_open, "close"]
    return df


def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved raw data to: {out_path}")


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().lower()
    for prefix in ("sh", "sz", "zh"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    if not s.isdigit() or len(s) != 6:
        raise ValueError("Invalid symbol. Use 6-digit code like 000001.")
    return s


def fetch_minute_data(symbol: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol = _normalize_symbol(symbol)
    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        adjust="",
    )
    if df.empty:
        fallback_start, fallback_end = _last_trading_window(days=5)
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            period=period,
            start_date=fallback_start,
            end_date=fallback_end,
            adjust="",
        )
    if df.empty:
        raise ValueError(
            "No data returned from AkShare. "
            "Try explicit recent trading range, e.g. "
            '--start-date "2026-02-13 09:30:00" --end-date "2026-02-20 15:00:00".'
        )
    return _normalize_columns(df)


def compute_minute_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["pct_change"] = df["close"] - df["open"]
    df = df.dropna(subset=["pct_change", "volume"]).copy()
    return df


def plot_scatter(df: pd.DataFrame, symbol: str, output: str | None, show: bool) -> None:
    start = df["datetime"].min().strftime("%Y-%m-%d")
    end = df["datetime"].max().strftime("%Y-%m-%d")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df["pct_change"], df["volume"], alpha=0.45, s=10, edgecolors="none")
    ax.set_title(f"{symbol} 1-min Return vs Volume ({start} to {end})")
    ax.set_xlabel("Minute Return (%)")
    ax.set_ylabel("Volume")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    df_reset = df.reset_index(drop=True)
    cursor = mplcursors.cursor(scatter, hover=False)

    @cursor.connect("add")
    def on_add(sel):
        idx = sel.index
        row = df_reset.iloc[idx]
        text = "\n".join(f"{col}: {row[col]}" for col in df_reset.columns)
        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(alpha=0.9)

    if output:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_price_volume(
    df: pd.DataFrame,
    symbol: str,
    output: str | None,
    show: bool,
    highlight_start: str | None = None,
    highlight_minutes: int | None = None,
) -> None:
    start = df["datetime"].min().strftime("%Y-%m-%d")
    end = df["datetime"].max().strftime("%Y-%m-%d")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_red", ["#006400", "#90EE90", "#D3D3D3", "#FF6666", "#8B0000"]
    )
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-0.03, vmax=0.03)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["volume"], df["close"],
        c=df["pct_change"], cmap=cmap, norm=norm,
        alpha=0.6, s=10, edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, label="pct_change")

    if highlight_start and highlight_minutes:
        hl_start = pd.to_datetime(highlight_start)
        hl_end = hl_start + dt.timedelta(minutes=highlight_minutes)
        hl_mask = (df["datetime"] >= hl_start) & (df["datetime"] < hl_end)
        hl_df = df[hl_mask]
        ax.scatter(
            hl_df["volume"], hl_df["close"],
            color="black", s=15, zorder=5, label=f"{highlight_start} +{highlight_minutes}min",
        )
        ax.legend()

    ax.set_title(f"{symbol} Price vs Volume ({start} to {end})")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    df_reset = df.reset_index(drop=True)
    cursor = mplcursors.cursor(scatter, hover=False)

    @cursor.connect("add")
    def on_add(sel):
        idx = sel.index
        row = df_reset.iloc[idx]
        text = "\n".join(f"{col}: {row[col]}" for col in df_reset.columns)
        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(alpha=0.9)

    if output:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch A-share minute data and plot return vs volume."
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # ---- download 子命令 ----
    dl = sub.add_parser("download", help="下载分钟数据并保存到 data/")
    dl.add_argument("--symbol", default="000001", help="A-share code, e.g. 000001")
    dl.add_argument("--period", default="1", help="Minute period, e.g. 1, 5, 15, 30, 60")
    dl.add_argument("--start-date", default=None, help="YYYY-MM-DD HH:MM:SS")
    dl.add_argument("--end-date", default=None, help="YYYY-MM-DD HH:MM:SS")

    # ---- plot 子命令 ----
    pl = sub.add_parser("plot", help="读取已有 CSV 文件并画图")
    pl.add_argument("--file", required=True, help="CSV 文件路径，如 data/minute_000001_1min.csv")
    pl.add_argument("--symbol", default="000001", help="图表标题中的股票代码")
    pl.add_argument("--output", default=None, help="保存图片路径，如 data/chart.png")
    pl.add_argument("--show", action="store_true", help="弹窗显示图表")
    pl.add_argument("--chart", default="scatter", choices=["scatter", "price_volume"],
                    help="图表类型: scatter(收益率vs成交量) / price_volume(股价vs成交量)")
    pl.add_argument("--highlight-start", default=None,
                    help="高亮起始时间，如 '2026-02-09 09:30:00'")
    pl.add_argument("--highlight-minutes", type=int, default=None,
                    help="高亮时长（分钟）")

    args = parser.parse_args()

    if args.action == "download":
        start_date, end_date = _default_range()
        if args.start_date:
            start_date = args.start_date
        if args.end_date:
            end_date = args.end_date
        df = fetch_minute_data(args.symbol, args.period, start_date, end_date)
        save_to_csv(df, f"minute_{args.symbol}_{args.period}min.csv")

    elif args.action == "plot":
        df = pd.read_csv(args.file, encoding="utf-8-sig")
        df = _normalize_columns(df)
        stats = compute_minute_stats(df)
        if args.chart == "price_volume":
            plot_price_volume(
                stats, args.symbol, args.output, args.show,
                highlight_start=args.highlight_start,
                highlight_minutes=args.highlight_minutes,
            )
        else:
            plot_scatter(stats, args.symbol, args.output, args.show)


if __name__ == "__main__":
    main()
