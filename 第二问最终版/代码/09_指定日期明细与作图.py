from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("_comm2.py", "q2_comm")
DP = _load("06_DP价值执行器.py", "q2_dp")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = C.PERIODS_PER_DAY
ETA = C.ETA
DT = C.DELTA_HOURS
TOLL = 1e-6
BLOCK_PERIODS = 24
FIG5_DATE = "2025-06-02"


def save_fig(fig, stem: str) -> None:
    fig.savefig(C.FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(C.FIGURE_DIR / f"{stem}.png", dpi=160)
    plt.close(fig)


def period_label(t: int, use_24: bool = True) -> str:
    a, bb = C.period_bounds(t + 1)
    return f"{C.minutes_to_hhmm(a)}-{C.minutes_to_hhmm(bb, use_24=use_24)}"


def block_slices() -> list:
    return [(i * BLOCK_PERIODS, (i + 1) * BLOCK_PERIODS) for i in range(6)]


def merge_emergency(b: np.ndarray) -> list:
    events = []
    t = 0
    while t < T:
        if b[t] > TOLL:
            s = t
            tot = 0.0
            while t < T and b[t] > TOLL:
                tot += float(b[t])
                t += 1
            events.append((s + 1, t, tot))
        else:
            t += 1
    return events


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg, flush=True)

    t_all = time.perf_counter()
    p("=" * 74)
    p("第二问 09 —— 指定日期明细表与正式图（表 8–12 / 图 5–9）")
    p("=" * 74)

    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    BT = np.load(C.RESULT_DIR / "第二问_执行器回测.npz", allow_pickle=False)
    price = np.asarray(Z["price"], float)
    N_all = np.asarray(Z["net_load_energy_kwh"], float)
    scen_N_all = np.asarray(Zd["scen_L"], float) - np.asarray(Zd["scen_V"], float)
    date_strs = [str(x) for x in Z["dates"]]
    score_idx = np.asarray(Z["score_day_index"], int)
    date_of = {s: i for i, s in enumerate(date_strs)}

    g_all = np.asarray(BT["plan_g"], float)
    b_all = np.asarray(BT["dp_b"], float)
    C_all = np.asarray(BT["dp_C"], float)
    D_all = np.asarray(BT["dp_D"], float)
    E_all = np.asarray(BT["dp_E"], float)
    an_b = np.asarray(BT["an_b"], float)
    an_E = np.asarray(BT["an_E"], float)

    E_start = np.zeros(365)
    prev = float(C.E_INIT)
    for d in score_idx:
        E_start[d] = prev
        prev = float(E_all[d, -1])

    p(f"数据源：第二问_执行器回测.npz（DP 主方案，评分期 {score_idx.size} 天）")
    p(f"日初库存链：首日 {E_start[score_idx[0]]:.6f} kWh → "
      f"末日末 {E_all[score_idx[-1], -1]:.6f} kWh")

    p("")
    p("── 表 8 指定时段计划购电量（kWh；数据源 = DP 主方案日前计划 g）──")
    slot_idx = [C.slot_to_period_index(s) - 1 for s, _ in C.SPEC_SLOTS]
    slot_names = [f"{a}-{bb}" for a, bb in C.SPEC_SLOTS]
    dates = [date_of[ds] for ds in C.SPEC_DATES]
    p(f"{'时段':<14}" + "".join(f"{ds:>18}" for ds in C.SPEC_DATES))
    t8_rows = []
    for name, si in zip(slot_names, slot_idx):
        vals = [float(g_all[d, si]) for d in dates]
        p(f"{name:<14}" + "".join(f"{v:>18.6f}" for v in vals))
        t8_rows.append((name, *vals))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_指定时段计划购电量.csv",
        ("时段", *C.SPEC_DATES), t8_rows)

    p("")
    p("── 表 9 指定日期全天购电量与费用 / 表 10 费用分解 ──")
    p(f"{'日期':<14}{'计划量_kWh':>15}{'紧急量_kWh':>14}{'计费总量_kWh':>15}"
      f"{'结算费用_元':>15}{'计划费_元':>14}{'紧急费_元':>14}")
    t9_rows, t10_rows = [], []
    for d in dates:
        gp = float(g_all[d].sum())
        ek = float(b_all[d].sum())
        plan_fee = float(price @ g_all[d])
        emerg_fee = float((5.0 * price) @ b_all[d])
        p(f"{date_strs[d]:<14}{gp:>15.6f}{ek:>14.6f}{gp + ek:>15.6f}"
          f"{plan_fee + emerg_fee:>15.6f}{plan_fee:>14.6f}{emerg_fee:>14.6f}")
        t9_rows.append((date_strs[d], gp, ek, gp + ek, plan_fee + emerg_fee))
        t10_rows.append((date_strs[d], plan_fee, emerg_fee, plan_fee + emerg_fee))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_指定日期全天购电量与费用.csv",
        ("日期", "全天计划购电量_kWh", "全天紧急购电量_kWh", "全天计费购电量_kWh",
         "全天结算费用_元"), t9_rows)
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_指定日期费用分解.csv",
        ("日期", "计划费_元", "紧急费_元", "合计_元"), t10_rows)

    p("")
    p("── 表 11 紧急购电事件（连续 10 min 合并，跨日分开）──")
    t11_rows = []
    for d in dates:
        ev = merge_emergency(b_all[d])
        if not ev:
            p(f"  {date_strs[d]:<14}无紧急购电事件")
            continue
        for s, e, tot in ev:
            seg = f"{C.minutes_to_hhmm(C.period_bounds(s)[0])}-" \
                  f"{C.minutes_to_hhmm(C.period_bounds(e)[1], use_24=True)}"
            p(f"  {date_strs[d]:<14}{seg:<22}{tot:>12.6f} kWh  （{e - s + 1} 个时段）")
            t11_rows.append((date_strs[d], seg, f"{tot:.6f}", str(e - s + 1)))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_指定日期紧急购电事件.csv",
        ("日期", "购电时间段", "购电量_kWh", "连续时段数"), t11_rows)

    p("")
    p("── 表 12a–12d 储能充放电量与首末/分段储电量（kWh，交流侧口径）──")
    t12_rows = []
    for d in dates:
        e0 = float(E_start[d])
        e1 = float(E_all[d, -1])
        p(f"  【{date_strs[d]}】0:00 {e0:.6f} → 24:00 {e1:.6f}"
          f"（净变化 {e1 - e0:+.6f}，校验 η·ΣC − ΣD/η = "
          f"{ETA * C_all[d].sum() - D_all[d].sum() / ETA:+.6f}）")
        row = [date_strs[d], f"{e0:.6f}"]
        for i, (a, bb) in enumerate(block_slices()):
            ck = float(C_all[d, a:bb].sum())
            dk = float(D_all[d, a:bb].sum())
            p(f"    {C.BATT_BLOCKS[i]:<12}充 {ck:>12.6f}   放 {dk:>12.6f}")
            row += [f"{ck:.6f}", f"{dk:.6f}"]
        row.append(f"{e1:.6f}")
        t12_rows.append(tuple(row))
    header12 = ["日期", "0:00储电量_kWh"]
    for nm in C.BATT_BLOCKS:
        header12 += [f"{nm}_充电量_kWh", f"{nm}_放电量_kWh"]
    header12.append("24:00储电量_kWh")
    C.write_csv_utf8_sig(C.RESULT_DIR / "第二问_指定日期储能充放电与储电量.csv",
                         tuple(header12), t12_rows)

    p("")
    p("── 正式图（图 5–9，各 PDF + PNG）──")
    C.setup_matplotlib()
    tq_end = np.arange(T) * DT + DT / 2
    colors4 = ["#2E75B6", "#548235", "#BF8F00", "#C00000"]

    def plot_power_pair(dlist, stem, title):
        fig, axes = plt.subplots(len(dlist), 1, figsize=(10.2, 3.4 * len(dlist)))
        if len(dlist) == 1:
            axes = [axes]
        for ax, d in zip(axes, dlist):
            p_grid = g_all[d] / DT
            p_emerg = b_all[d] / DT
            p_net = N_all[d] / DT
            ax.bar(tq_end, p_grid, width=DT * 0.92, label="计划购电功率",
                   color="#8FAADC")
            ax.bar(tq_end, p_emerg, width=DT * 0.92, bottom=p_grid,
                   label="紧急购电功率", color="#C00000")
            ax.plot(tq_end, p_net, color="#1F3864", lw=1.1, label="净负荷功率")
            top = float(max(np.max(p_grid + p_emerg), np.max(p_net)))
            bot = float(min(np.min(p_net), 0.0))
            span = max(top - bot, 1.0)
            ax.set_ylim(bot - 0.08 * span, top + 0.46 * span)
            ax.axhline(0.0, color="gray", lw=0.8, alpha=0.8)
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 2))
            ax.set_ylabel("功率（kW）")
            ax.set_title(f"{title}（{date_strs[d]}）")
            ax.grid(ls=":", lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        axes[-1].set_xlabel("时刻（h）")
        fig.tight_layout()
        save_fig(fig, stem)
        p(f"  已保存 模型结果图/{stem}.pdf / .png")

    plot_power_pair(dates[:2], "第二问_图7_0320与0621净负荷及购电功率",
                    "图 7 净负荷及购电功率")
    plot_power_pair(dates[2:], "第二问_图9_0923与1221净负荷及购电功率",
                    "图 9 净负荷及购电功率")

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    for d, col in zip(dates, colors4):
        traj = np.concatenate([[E_start[d]], E_all[d]])
        ax.plot(np.arange(T + 1) * DT, traj, color=col, lw=1.5, label=date_strs[d])
    ax.axhline(C.E_MAX, ls="--", lw=1.0, color="gray")
    ax.axhline(C.E_MIN, ls="--", lw=1.0, color="gray")
    ax.text(0.25, C.E_MAX + 130, f"上限 {C.E_MAX:.0f} kWh", color="gray", fontsize=8)
    ax.text(7.6, C.E_MIN + 130, f"下限 {C.E_MIN:.0f} kWh", color="gray", fontsize=8)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(0, 12000)
    ax.set_xlabel("时刻（h）")
    ax.set_ylabel("内部储电量（kWh）")
    ax.set_title("图 8  四个指定日期内部储电量轨迹")
    ax.grid(ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, ncol=4, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    save_fig(fig, "第二问_图8_四个指定日期内部储电量轨迹")
    p("  已保存 模型结果图/第二问_图8_四个指定日期内部储电量轨迹.pdf / .png")

    p(f"── 图 5：{FIG5_DATE} 储能保留水平及紧急购电时序 ──")
    d5 = date_of[FIG5_DATE]
    vf5 = DP.build_value_functions(scen_N_all[d5], g_all[d5], price)
    R_t = vf5["R"]
    p(f"  DP 动态保留水平 R_t ∈ [{R_t.min():.1f}, {R_t.max():.1f}] kWh；"
      f"解析响应保留水平恒为 E_min={C.E_MIN:.0f} kWh")
    fig, (axU, axL) = plt.subplots(2, 1, figsize=(10.4, 6.8), sharex=False)
    tq = np.arange(T) * DT
    axU.plot(np.arange(T + 1) * DT, np.concatenate([[E_start[d5]], E_all[d5]]),
             color="#1F3864", lw=1.5, label="DP 执行器储电量轨迹")
    axU.step(tq, R_t, where="post", color="#C00000", lw=1.4,
             label="DP 动态保留水平 R_t")
    axU.axhline(C.E_MIN, color="#548235", ls="--", lw=1.2,
                label=f"解析响应保留水平（{C.E_MIN:.0f} kWh）")
    axU.set_ylabel("储电量（kWh）")
    axU.set_xlim(0, 24)
    axU.set_xticks(range(0, 25, 2))
    axU.set_title(f"图 5（上）  {FIG5_DATE} 储能保留水平")
    axU.grid(ls=":", lw=0.6, alpha=0.7)
    axU.set_axisbelow(True)
    axU.legend(fontsize=8, loc="upper left", framealpha=0.95)
    axL.bar(tq_end, an_b[d5], width=DT * 0.9, label="解析响应紧急补购量",
            color="#8FAADC", alpha=0.85)
    axL.bar(tq_end, b_all[d5], width=DT * 0.9, label="DP 紧急补购量",
            color="#C00000", alpha=0.85)
    axL.set_ylabel("紧急补购量（kWh / 10 min）")
    axL.set_xlim(0, 24)
    axL.set_xticks(range(0, 25, 2))
    axL2 = axL.twinx()
    axL2.plot(tq, price, color="#BF8F00", lw=1.3, label="分时电价")
    axL2.set_ylabel("分时电价（元/kWh）")
    axL2.set_ylim(0, float(price.max()) * 1.18)
    axL.set_title(f"图 5（下）  {FIG5_DATE} 紧急购电时序与分时电价")
    axL.grid(ls=":", lw=0.6, alpha=0.7)
    axL.set_axisbelow(True)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axL2.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.95)
    axL.set_xlabel("时刻（h）")
    fig.tight_layout()
    save_fig(fig, "第二问_图5_6月2日储能保留水平及紧急购电时序")
    p("  已保存 模型结果图/第二问_图5_6月2日储能保留水平及紧急购电时序.pdf / .png")

    p("── 图 6：DP 相对解析响应月度费用节省 ──")
    monthly_csv = C.RESULT_DIR / "第二问_月度节省.csv"
    import csv as _csv
    months, saves = [], []
    with open(monthly_csv, encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            months.append(row["月份"].replace("月", ""))
            saves.append(float(row["节省_万元"]))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    xs = np.arange(len(months))
    ax.bar(xs, saves, color="#2E75B6", width=0.62)
    for x, y in zip(xs, saves):
        ax.text(x, y + max(saves) * 0.02, f"{y:.2f}", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{m}月" for m in months])
    ax.set_ylabel("月度节省（万元）")
    ax.set_title("图 6  DP 价值执行器相对解析响应的月度费用节省（2025 年 2—12 月）")
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "第二问_图6_DP相对解析响应月度费用节省")
    p("  已保存 模型结果图/第二问_图6_DP相对解析响应月度费用节省.pdf / .png")

    p("── 内部诊断：四日期六段充放电量（非论文图）──")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    xs = np.arange(6)
    w = 0.2
    for k, (d, col) in enumerate(zip(dates, colors4)):
        ck = [float(C_all[d, a:bb].sum()) for a, bb in block_slices()]
        dk = [float(D_all[d, a:bb].sum()) for a, bb in block_slices()]
        axes[0].bar(xs + (k - 1.5) * w, ck, width=w, color=col, label=date_strs[d])
        axes[1].bar(xs + (k - 1.5) * w, dk, width=w, color=col, label=date_strs[d])
    for ax, ttl in zip(axes, ("充电量（交流侧）", "放电量（交流侧）")):
        ax.set_xticks(xs)
        ax.set_xticklabels(C.BATT_BLOCKS, fontsize=8)
        ax.set_ylabel("电量（kWh）")
        ax.set_title(ttl)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("内部诊断  指定日期储能六段 4 小时充/放电量（交流侧口径）", fontsize=11)
    fig.tight_layout()
    C.INTERNAL_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.INTERNAL_DIAG_DIR / "内部诊断_四日期六段充放电量.png", dpi=160)
    plt.close(fig)
    p("  已保存 内部诊断/非论文图/内部诊断_四日期六段充放电量.png")

    p("── 正式图表数据总表（图 7/9 与表 8/9/12 的逐点数据）──")
    total_rows = []
    for d in dates:
        for t in range(T):
            total_rows.append((
                date_strs[d], t + 1, period_label(t),
                f"{N_all[d, t]:.6f}", f"{g_all[d, t]:.6f}", f"{b_all[d, t]:.6f}",
                f"{N_all[d, t] / DT:.6f}", f"{g_all[d, t] / DT:.6f}",
                f"{b_all[d, t] / DT:.6f}", f"{C_all[d, t]:.6f}",
                f"{D_all[d, t]:.6f}", f"{E_all[d, t]:.6f}"))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_正式图表数据总表.csv",
        ("日期", "时段序号", "时间区间", "实际净负荷_kWh", "计划购电量_kWh",
         "紧急购电量_kWh", "净负荷功率_kW", "计划购电功率_kW", "紧急购电功率_kW",
         "充电量_kWh", "放电量_kWh", "储电量_kWh"), total_rows)

    p("")
    p(f"总用时 {time.perf_counter() - t_all:.1f} s")

    C.write_text_utf8(C.SOLVE_LOG_DIR / "第二问_09指定日期明细日志.txt",
                      "\n".join(log + ["", "[09 完成] 指定日期明细与正式图结束。"]))
    C.write_text_utf8(C.REPORT_DIR / "第二问_指定日期明细报告.md",
                      "\n".join(["# 第二问 指定日期明细报告（表 8–12 / 图 5–9）", ""]
                                + log + [""]))
    p("已保存：求解日志/第二问_09指定日期明细日志.txt、报告/第二问_指定日期明细报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
