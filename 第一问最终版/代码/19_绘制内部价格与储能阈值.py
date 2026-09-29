from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _comm import setup_chinese_font

PROJECT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT / "模型结果"
FIG_DIR = PROJECT / "模型结果图"

PLOT_DATA_CSV = RESULT_DIR / "第一问_内部价格与阈值作图数据.csv"
FIG_PDF = FIG_DIR / "第一问_内部价格与储能充放电阈值.pdf"
FIG_PNG = FIG_DIR / "第一问_内部价格与储能充放电阈值.png"


def parse_hhmm(s):
    h, m = str(s).split(":")[:2]
    return int(h) + int(m) / 60.0


def main():
    setup_chinese_font()
    df = pd.read_csv(PLOT_DATA_CSV, encoding="utf-8-sig")
    t = np.array([parse_hhmm(x) for x in df["interval_end"]])
    c = df["external_price"].to_numpy(float)
    pi = df["internal_price_pi"].to_numpy(float)
    eta_lam = df["charge_threshold_eta_lambda"].to_numpy(float)
    lam_over_eta = df["discharge_threshold_lambda_over_eta"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(t, c, color="#C44E52", lw=1.4, label="外网电价 $c_t$")
    ax.plot(t, pi, color="#2C3E50", lw=1.4, label=r"内部边际价格 $\pi_t$")
    ax.plot(t, eta_lam, color="#4C72B0", lw=1.3, ls="--", label=r"充电阈值 $\eta\lambda_t$")
    ax.plot(t, lam_over_eta, color="#27AE60", lw=1.3, ls="--", label=r"放电阈值 $\lambda_t/\eta$")

    ax.set_xlim(0.0, 24.0)
    ax.set_xlabel("时刻")
    ax.set_ylabel("价格 (元/kWh)")
    ax.set_ylim(bottom=0.0)

    step = 2.0
    ticks = np.arange(0.0, 24.0 + 1e-9, step)
    labels = [f"{int(h):02d}:00" for h in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, format="pdf")
    fig.savefig(FIG_PNG, format="png", dpi=150)
    plt.close(fig)

    print(f"[19] 已生成 {FIG_PDF.name} / {FIG_PNG.name}")
    return {"pdf": str(FIG_PDF), "png": str(FIG_PNG)}


if __name__ == "__main__":
    main()
