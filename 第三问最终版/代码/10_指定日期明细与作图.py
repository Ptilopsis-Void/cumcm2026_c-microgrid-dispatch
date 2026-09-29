#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("_comm3.py", "q3_comm")
S8 = _load("08_全年回测与结算.py", "q3_s08")

T = C.PERIODS_PER_DAY


def merge_events(day_b: np.ndarray, k: int = 3):
    ev = []
    t = 0
    while t < T:
        if day_b[t] > 1e-9:
            t0 = t
            e = 0.0
            while t < T and day_b[t] > 1e-9:
                e += float(day_b[t])
                t += 1
            if t - t0 >= k:
                ev.append((t0 + 1, t, e, t - t0))
            else:
                ev.append((t0 + 1, t, e, t - t0))
        else:
            t += 1
    return ev


def seg_text(t0: int, t1: int) -> str:
    a, _ = C.period_bounds(t0)
    _, b = C.period_bounds(t1)
    return f"{C.minutes_to_hhmm(a)}-{C.minutes_to_hhmm(b, use_24=True)}"


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t00 = time.perf_counter()

    log("=" * 78)
    log("第三问 10 —— 指定日期明细表与作图（表 1–4 / 图 1–3）")
    log("=" * 78)

    RB = np.load(C.BACKTEST_NPZ, allow_pickle=False)
    Z = C.Q2.matrix()
    F = np.load(C.V_FORECAST_NPZ, allow_pickle=False)

    dates = np.asarray(RB["dates"])
    price = np.asarray(RB["price"], float)
    P = np.asarray(RB["P"], float)
    Q = np.asarray(RB["Q"], float)
    b = np.asarray(RB["b"], float)
    Ch = np.asarray(RB["C"], float)
    Dh = np.asarray(RB["D"], float)
    E = np.asarray(RB["E"], float)
    N = np.asarray(RB["N_actual"], float)
    load_kw = np.asarray(Z["load_kw"], float)
    pv_kw = np.asarray(Z["pv_kw"], float)
    score_idx = np.asarray(F["score_day_index"], int)

    didx = {str(d): i for i, d in enumerate(dates)}
    spec = [d for d in C.SPEC_DATES if d in didx]
    if len(spec) != len(C.SPEC_DATES):
        miss = [d for d in C.SPEC_DATES if d not in didx]
        raise AssertionError(f"指定日期缺失：{miss}")
    slots = [(s[0], s[1], C.slot_to_period_index(s[1]) - 1) for s in C.SPEC_SLOTS]

    E_start = np.zeros(365)
    _prev = float(C.E_INIT)
    for _d in score_idx:
        E_start[int(_d)] = _prev
        _prev = float(E[int(_d), -1])

    log(f"指定日期：{', '.join(spec)}")
    log(f"指定时段：{', '.join(a + '-' + c for a, c, _ in slots)}")
    log(f"0:00 库存链：首日 {E_start[int(score_idx[0])]:,.6f} kWh "
        f"→ 末日末 {E[int(score_idx[-1]), -1]:,.6f} kWh")
    log("")
    log("── 表 1 指定时段计划购电量与调整购电量（kWh）──")
    hdr = "时段".ljust(14) + "".join(f"{d + '_计划':>16s}{d + '_调整':>16s}"
                                     for d in spec)
    log(hdr)
    t1_rows = []
    for a, c, tt in slots:
        line = f"{a}-{c}".ljust(14)
        row = [f"{a}-{c}"]
        for d in spec:
            i = didx[d]
            line += f"{P[i, tt]:>16.6f}{Q[i, tt]:>16.6f}"
            row += [f"{P[i, tt]:.6f}", f"{Q[i, tt]:.6f}"]
        log(line)
        t1_rows.append(row)
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_指定时段计划与调整购电量.csv",
        ["时段"] + [f"{d}_{k}" for d in spec for k in ("计划", "调整")],
        t1_rows)

    log("")
    log("── 表 2 指定日期全天购电量与费用三分解 ──")
    log(f"  {'日期':<12s}{'计划量':>13s}{'调整量':>13s}{'紧急量':>11s}"
        f"{'实际净负荷':>13s}{'总费用元':>13s}{'计划费':>12s}"
        f"{'下调':>10s}{'上调':>10s}{'紧急费':>11s}")
    t2_rows = []
    t2_map = {}
    for d in spec:
        i = didx[d]
        dec = S8.decompose(price, P[i][None, :], Q[i][None, :], b[i][None, :])
        p_kwh = float(P[i].sum())
        q_kwh = float(Q[i].sum())
        b_kwh = float(b[i].sum())
        n_kwh = float(N[i].sum())
        plan = float(dec["plan"][0])
        down = float(dec["down"][0])
        up = float(dec["up"][0])
        emer = float(dec["emerg"][0])
        tot = float(dec["total"][0])
        t2_map[d] = dict(P=p_kwh, Q=q_kwh, b=b_kwh, N=n_kwh, plan=plan,
                         down=down, up=up, emer=emer, total=tot)
        log(f"  {d:<12s}{p_kwh:>13.4f}{q_kwh:>13.4f}{b_kwh:>11.4f}"
            f"{n_kwh:>13.4f}{tot:>13.4f}{plan:>12.4f}"
            f"{down:>10.4f}{up:>10.4f}{emer:>11.4f}")
        t2_rows.append([d, f"{p_kwh:.6f}", f"{q_kwh:.6f}", f"{b_kwh:.6f}",
                        f"{n_kwh:.6f}", f"{p_kwh + b_kwh:.6f}", f"{tot:.6f}",
                        f"{plan:.6f}", f"{down:.6f}", f"{up:.6f}",
                        f"{emer:.6f}"])
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_指定日期全天购电量与费用.csv",
        ["日期", "计划量_kWh", "调整量_kWh", "紧急量_kWh", "实际净负荷_kWh",
         "购电总量_kWh", "总费用_元", "计划购电费_元", "下调违约金_元",
         "上调加价_元", "紧急购电费_元"],
        t2_rows)

    log("")
    log("── 表 3 指定日期储能六段充放电量与储电量（kWh，交流侧口径）──")
    t3_rows = []
    for d in spec:
        i = didx[d]
        e0 = float(E_start[i])
        e24 = float(E[i, -1])
        sc = float(Ch[i].sum())
        sd = float(Dh[i].sum())
        chk = C.ETA * sc - sd / C.ETA
        log(f"  【{d}】0:00 储电量 {e0:,.6f} → 24:00 储电量 {e24:,.6f}"
            f"（净变化 {e24 - e0:+,.6f}，校验 ηΣC−ΣD/η = {chk:+,.6f}）")
        for bi, blk in enumerate(C.BATT_BLOCKS):
            k0, k1 = bi * 24, (bi + 1) * 24
            cblk = float(Ch[i, k0:k1].sum())
            dblk = float(Dh[i, k0:k1].sum())
            log(f"    {blk:<12s} 充 {cblk:>12.6f}   放 {dblk:>12.6f}")
            t3_rows.append([d, blk, f"{cblk:.6f}", f"{dblk:.6f}",
                            f"{e0:.6f}", f"{e24:.6f}", f"{chk:.6f}"])
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_指定日期储能充放电与储电量.csv",
        ["日期", "分段", "充电量_kWh", "放电量_kWh", "0点储电量_kWh",
         "24点储电量_kWh", "净变化校验_kWh"],
        t3_rows)

    log("")
    log("── 表 4 指定日期紧急购电事件（连续 10 min 时段合并）──")
    t4_rows = []
    for d in spec:
        i = didx[d]
        ev = merge_events(b[i])
        if not ev:
            log(f"  {d}    无紧急购电事件")
            t4_rows.append([d, "无", "0", "0.000000"])
        for t0, t1_, e, n in ev:
            s = seg_text(t0, t1_)
            log(f"  {d}    {s:<22s} {e:>12.6f} kWh  （{n} 个时段）")
            t4_rows.append([d, s, str(n), f"{e:.6f}"])
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_指定日期紧急购电事件.csv",
        ["日期", "起止时段", "事件时段数", "紧急购电量_kWh"],
        t4_rows)

    C.setup_matplotlib()
    C.setup_chinese_font()
    import matplotlib.pyplot as plt

    hours = (np.arange(T) + 0.5) * C.DELTA_HOURS

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.4), sharex=True)
    for ax, d in zip(axes.ravel(), spec):
        i = didx[d]
        ax.plot(hours, N[i], lw=1.1, color="#222222", label="实际净负荷")
        ax.plot(hours, P[i], lw=1.0, color="#4C78A8", alpha=0.85, label="计划购电量 $p$")
        ax.plot(hours, Q[i], lw=1.0, color="#F2A93B", alpha=0.9, label="调整购电量 $q$")
        ax.bar(hours, b[i], width=C.DELTA_HOURS, color="#E45756", alpha=0.75,
               label="紧急购电量 $b$")
        ax.set_title(f"{d}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylabel("kWh/时段", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("时刻（h）", fontsize=9)
    fig.suptitle("第三问 指定日期：实际净负荷 / 计划购电量 / 调整购电量 / 紧急购电量")
    fig.tight_layout()
    C.save_figure(fig, "第三问_指定日期净负荷与购电量")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    for d in spec:
        i = didx[d]
        ax.plot(hours, E[i], lw=1.3, label=d)
    ax.axhline(C.E_MAX, ls="--", lw=0.9, color="grey")
    ax.axhline(C.E_MIN, ls="--", lw=0.9, color="grey")
    ax.set_xlabel("时刻（h）")
    ax.set_ylabel("储电量（kWh）")
    ax.set_title("第三问 指定日期储电量轨迹（含上下限）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "第三问_指定日期储电量轨迹")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(C.BATT_BLOCKS))
    w = 0.2
    for j, d in enumerate(spec):
        i = didx[d]
        cblk = np.array([Ch[i, k * 24:(k + 1) * 24].sum()
                         for k in range(len(C.BATT_BLOCKS))])
        ax.bar(x + (j - 1.5) * w, cblk, w, label=f"{d} 充")
    ax.set_xticks(x, C.BATT_BLOCKS)
    ax.set_ylabel("充电量（kWh）")
    ax.set_title("第三问 指定日期六段充电量对比")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "第三问_指定日期六段充电量")
    plt.close(fig)

    log("")
    log("── 作图 ──")
    for s in ("第三问_指定日期净负荷与购电量", "第三问_指定日期储电量轨迹",
              "第三问_指定日期六段充电量"):
        log(f"  已保存 模型结果图/{s}.png / .pdf")

    rep = []
    rep.append("# 第三问 指定日期明细报告（表 1–4 / 图 1–3）")
    rep.append("")
    rep.append("```")
    rep.extend(log.lines)
    rep.append("```")
    rep.append("")
    rep.append("## 说明")
    rep.append("")
    rep.append("- 时段标签采用**区间 END** 语义（如 `10:00-10:10` 指 `[10:00, 10:10)` 这个 10 min 时段）。")
    rep.append("- 表 3 为**交流侧（电网侧）口径**：$\\eta\\Sigma C-\\Sigma D/\\eta=\\Delta E$ 已逐日校验。")
    rep.append("- 表 4 中事件量之和等于 `调整购电量` 表中同日的紧急购电合计（过短事件不丢弃，只不拆分展示）。")
    C.write_text_utf8(C.REPORT_SPEC_MD, "\n".join(rep))

    log("")
    log(f"已保存 4 张明细 CSV（模型结果/）、3 张图（模型结果图/）")
    log(f"已保存 {C.REPORT_SPEC_MD.relative_to(C.PROJECT_DIR)}")
    log.dump(C.LOG_DIR / "第三问_10指定日期日志.txt", tail="")
    log("")
    log(f"总用时 {time.perf_counter() - t00:.1f} s")
    log("[10 完成] 指定日期明细与作图结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
