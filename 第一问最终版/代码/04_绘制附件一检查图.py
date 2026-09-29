from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    CSV_PLOT_PATH, setup_chinese_font,
    FIG_LOAD_PV_PDF, FIG_LOAD_PV_PNG,
    FIG_NET_PDF, FIG_NET_PNG,
    FIG_PRICE_PDF, FIG_PRICE_PNG,
)

X_TICK_STEP_MIN = 120


def _minutes_to_label(m):
    h = int(m) // 60
    return f"{h:02d}:00"


def _style_axis(ax, x, source="附件1"):
    ax.set_xlabel("一天内时刻")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(x.min(), x.max())
    ticks = [m for m in range(0, int(x.max()) + 1, X_TICK_STEP_MIN)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_minutes_to_label(m) for m in ticks], rotation=0)
    ax.set_title(f"（数据来源：{source}）", fontsize=9, pad=8)
    ax.legend(loc="best")


def main():
    font = setup_chinese_font()
    df = pd.read_csv(CSV_PLOT_PATH, encoding="utf-8-sig")
    x = df["minute_of_day"].values

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, df["load_kw"], label="负荷 (kW)", linewidth=1.2, color="#1f77b4")
    ax.plot(x, df["pv_forecast_kw"], label="光伏预测 (kW)", linewidth=1.2, color="#ff7f0e")
    ax.set_ylabel("功率 (kW)")
    _style_axis(ax, x)
    fig.tight_layout()
    fig.savefig(FIG_LOAD_PV_PDF)
    fig.savefig(FIG_LOAD_PV_PNG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, df["net_demand_kw"], label="净需求 (kW)", linewidth=1.2, color="#d62728")
    ax.plot(x, df["pv_surplus_kw"], label="光伏剩余 (kW)", linewidth=1.2, color="#2ca02c")
    ax.set_ylabel("功率 (kW)")
    _style_axis(ax, x)
    fig.tight_layout()
    fig.savefig(FIG_NET_PDF)
    fig.savefig(FIG_NET_PNG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(x, df["price_yuan_per_kwh"], where="post", label="电价 (元/kWh)",
            linewidth=1.4, color="#9467bd")
    ax.set_ylabel("电价 (元/kWh)")
    _style_axis(ax, x)
    fig.tight_layout()
    fig.savefig(FIG_PRICE_PDF)
    fig.savefig(FIG_PRICE_PNG, dpi=150)
    plt.close(fig)

    print(f"[04] 绘图完成（中文字体：{font}）")
    print(f"[04] 已输出：{FIG_LOAD_PV_PDF.name}, {FIG_NET_PDF.name}, {FIG_PRICE_PDF.name}（含 PNG 预览）")
    return {
        "figs": [FIG_LOAD_PV_PDF, FIG_NET_PDF, FIG_PRICE_PDF],
        "font": font,
    }


if __name__ == "__main__":
    main()
