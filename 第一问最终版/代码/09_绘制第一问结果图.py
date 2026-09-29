from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    PERIODS_PER_DAY, DELTA_HOURS, E_MIN, E_MAX, E_INITIAL,
    OPT_SCHEDULE_CSV, BASELINE_CSV, PLOT_DATA_CSV,
    FIG_PRICE_GRID_PDF, FIG_PRICE_GRID_PNG,
    FIG_LOAD_PV_GRID_PDF, FIG_LOAD_PV_GRID_PNG,
    FIG_CD_PDF, FIG_CD_PNG, FIG_E_PDF, FIG_E_PNG,
    FIG_BASELINE_PDF, FIG_BASELINE_PNG,
    setup_chinese_font,
)

N = PERIODS_PER_DAY


def _time_ticks():
    step = 120
    mins = np.arange(0, 1440 + 1, step)
    labels = [f"{m // 60:02d}:{m % 60:02d}" for m in mins]
    return mins, labels


def _load():
    sch = pd.read_csv(OPT_SCHEDULE_CSV, encoding="utf-8-sig")
    base = pd.read_csv(BASELINE_CSV, encoding="utf-8-sig")
    out = {
        "price": sch["price_yuan_per_kwh"].to_numpy(float),
        "load": sch["load_kw"].to_numpy(float),
        "pv": sch["pv_forecast_kw"].to_numpy(float),
        "G": sch["grid_purchase_kw"].to_numpy(float),
        "C": sch["charge_power_kw"].to_numpy(float),
        "D": sch["discharge_power_kw"].to_numpy(float),
        "W": sch["pv_curtailment_kw"].to_numpy(float),
        "E": np.concatenate([sch["energy_start_kwh"].to_numpy(float),
                             [sch["energy_end_kwh"].iloc[-1]]]),
        "interval_start": sch["interval_start"].tolist(),
    }
    out["J1"] = float(sch["period_purchase_cost_yuan"].sum())
    out["J0"] = float(base["period_purchase_cost_yuan"].sum())
    return out


def write_plot_data(d):
    df = pd.DataFrame({
        "interval_index": np.arange(1, N + 1),
        "interval_start": d["interval_start"],
        "price_yuan_per_kwh": d["price"],
        "load_kw": d["load"],
        "pv_forecast_kw": d["pv"],
        "grid_purchase_kw": d["G"],
        "charge_power_kw": d["C"],
        "discharge_power_kw": d["D"],
        "pv_curtailment_kw": d["W"],
    })
    df.to_csv(PLOT_DATA_CSV, index=False, encoding="utf-8-sig", float_format="%.10f")


def plot1(d):
    t = np.arange(N)
    fig, ax1 = plt.subplots(figsize=(9, 3.6))
    ax1.bar(t, d["G"], color="#4C72B0", width=1.0, label="电网购电功率")
    ax1.set_xlabel("时段（10 分钟）")
    ax1.set_ylabel("电网购电功率 (kW)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(t, d["price"], color="#C44E52", lw=1.3, label="交易电价")
    ax2.set_ylabel("电价 (元/kWh)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    mins, labels = _time_ticks()
    ax1.set_xticks([m // 10 for m in mins])
    ax1.set_xticklabels(labels)
    lines = [plt.Line2D([0], [0], color="#4C72B0", lw=6),
             plt.Line2D([0], [0], color="#C44E52", lw=1.5)]
    ax1.legend(lines, ["电网购电功率", "交易电价"], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def plot2(d):
    t = np.arange(N)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, d["load"], color="#2C3E50", lw=1.2, label="居民负荷")
    ax.plot(t, d["pv"], color="#F39C12", lw=1.2, label="光伏预测出力")
    ax.plot(t, d["G"], color="#27AE60", lw=1.2, label="电网购电功率")
    ax.set_xlabel("时段（10 分钟）")
    ax.set_ylabel("功率 (kW)")
    mins, labels = _time_ticks()
    ax.set_xticks([m // 10 for m in mins])
    ax.set_xticklabels(labels)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def plot3(d):
    t = np.arange(N)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(t, d["C"], color="#27AE60", width=1.0, label="充电功率")
    ax.bar(t, -d["D"], color="#C0392B", width=1.0, label="放电功率")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("时段（10 分钟）")
    ax.set_ylabel("充放电功率 (kW)")
    mins, labels = _time_ticks()
    ax.set_xticks([m // 10 for m in mins])
    ax.set_xticklabels(labels)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def plot4(d):
    t_state = np.arange(N + 1) * 10
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t_state, d["E"], color="#16A085", lw=1.4, label="储能电量")
    ax.axhline(E_MAX, color="#C0392B", ls="--", lw=1.0, label=f"上限 {E_MAX} kWh")
    ax.axhline(E_MIN, color="#C0392B", ls="--", lw=1.0, label=f"下限 {E_MIN} kWh")
    ax.axhline(E_INITIAL, color="gray", ls=":", lw=1.0, label=f"初值 {E_INITIAL} kWh")
    ax.set_xlabel("时刻")
    ax.set_ylabel("储能电量 (kWh)")
    mins, labels = _time_ticks()
    ax.set_xticks(mins)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot5(d):
    J0, J1 = d["J0"], d["J1"]
    saving = J0 - J1
    rate = saving / J0 * 100.0
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bars = ax.bar(["无储能 J_0", "有储能 J_1"], [J0, J1],
                  color=["#95A5A6", "#16A085"], width=0.5)
    for b, v in zip(bars, [J0, J1]):
        ax.text(b.get_x() + b.get_width() / 2, v + J0 * 0.01,
                f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel("全天购电费用 (元)")
    ax.set_title(f"节省 {saving:.2f} 元（节省率 {rate:.2f}%）", fontsize=10)
    fig.tight_layout()
    return fig


def make_plots():
    setup_chinese_font()
    d = _load()
    write_plot_data(d)

    jobs = [
        (plot1, FIG_PRICE_GRID_PDF, FIG_PRICE_GRID_PNG),
        (plot2, FIG_LOAD_PV_GRID_PDF, FIG_LOAD_PV_GRID_PNG),
        (plot3, FIG_CD_PDF, FIG_CD_PNG),
        (plot4, FIG_E_PDF, FIG_E_PNG),
        (plot5, FIG_BASELINE_PDF, FIG_BASELINE_PNG),
    ]
    for fn, pdf_path, png_path in jobs:
        fig = fn(d)
        fig.savefig(pdf_path, format="pdf")
        fig.savefig(png_path, format="png", dpi=150)
        plt.close(fig)
        print(f"[09] 已生成 {pdf_path.name} / {png_path.name}")

    print(f"[09] 作图数据已导出：{PLOT_DATA_CSV.name}")
    return d


if __name__ == "__main__":
    make_plots()
