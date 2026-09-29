#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import sys
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


C = _load("_comm4.py", "q4_comm")

T = C.PERIODS_PER_DAY
SPEC_DATES = list(C.SPEC_DATES)
T1_START_MIN = [10 * 60, 12 * 60, 14 * 60, 16 * 60, 18 * 60, 20 * 60]
T1_LEN_MIN = 10
T2_BLOCKS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
TAU_H = list(C.TAU_HOURS)
FONT_CANDIDATES = ("Heiti TC", "Songti SC", "PingFang SC", "STHeiti",
                   "Arial Unicode MS", "SimHei", "Noto Sans CJK SC",
                   "WenQuanYi Zen Hei", "DejaVu Sans")


def load_price_matrix():
    with np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False) as Z:
        return {k: Z[k] for k in Z.files}


def load_branch(branch: str):
    path = C.BACKTEST_42_NPZ if branch == "42" else C.BACKTEST_43_NPZ
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as Z:
        return {k: Z[k] for k in Z.files}


def slot_of_start(slot_start_minute: np.ndarray, minute: int) -> int:
    hit = np.where(np.asarray(slot_start_minute, int) == int(minute))[0]
    if hit.size != 1:
        raise RuntimeError(f"起始分钟 {minute} 在附件四中不唯一（命中 {hit.size} 次）")
    return int(hit[0])


def hhmm(minute: int) -> str:
    m = int(minute) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def bill_42_def(c, g, b) -> dict:
    return {"plan": float((c * g).sum()), "adjust": 0.0,
            "emerg": float((5.0 * c * b).sum()), "total": float((c * g).sum() + (5.0 * c * b).sum())}


def bill_43_def_four(c, g, a, b) -> dict:
    d = g - a
    plan = float((c * np.minimum(g, a)).sum())
    down = float((0.5 * c * np.clip(d, 0.0, None)).sum())
    up = float((1.5 * c * np.clip(-d, 0.0, None)).sum())
    emerg = float((5.0 * c * b).sum())
    return {"plan": plan, "adjust": down + up, "down": down, "up": up,
            "emerg": emerg, "total": plan + down + up + emerg}


def bill_43_def_equiv(c, g, a, b) -> dict:
    eff = a + 0.5 * np.abs(a - g)
    return {"plan": float((c * a).sum()),
            "adjust": float((0.5 * c * np.abs(a - g)).sum()),
            "emerg": float((5.0 * c * b).sum()),
            "total": float((c * eff).sum() + (5.0 * c * b).sum())}


def month_of(date_str: str) -> str:
    return str(date_str)[:7]


def totals_of(d: dict) -> tuple[float, float, float, float]:
    def s(name, alt):
        if name in d:
            v = np.asarray(d[name], float)
            return float(v.sum()) if v.ndim > 0 else float(v.reshape(-1)[0])
        if alt in d:
            return float(np.asarray(d[alt], float).reshape(-1)[0])
        return 0.0
    return (s("bill", "cost_total"), s("plan_cost", "cost_plan"),
            s("adjust_cost", "cost_adjust"), s("emerg_cost", "cost_emerg"))


def table1(branches: dict, pm: dict, log: list, checks: list) -> tuple[list[list], list[list]]:
    ssm = np.asarray(pm["slot_start_minute"], int)
    slots = [slot_of_start(ssm, m) for m in T1_START_MIN]
    rows: list[list] = []
    day_rows: list[list] = []

    def gk(rec, key, di, t):
        return float(np.asarray(rec[key], float)[di, t])

    for date in SPEC_DATES:
        for br in ("42", "43"):
            rec = branches.get(br)
            if rec is None:
                continue
            dates = np.asarray(rec["dates"], dtype="<U10")
            hit = np.where(dates == date)[0]
            if hit.size != 1:
                raise RuntimeError(f"{date} 不在 4-{br[1]} 的回测窗口内")
            di = int(hit[0])
            c = np.asarray(rec["c"], float)[di]
            gg = np.asarray(rec["g"], float)[di]
            aa = np.asarray(rec["a"], float)[di]
            bb = np.asarray(rec["b"], float)[di]
            e0 = float(np.asarray(rec["E_chain"], float)[di])
            e1 = float(np.asarray(rec["E_chain"], float)[di + 1])

            for j, t in enumerate(slots):
                s = int(ssm[t])
                e = s + T1_LEN_MIN
                if br == "42":
                    f_plan = float(c[t] * gg[t])
                    f_adj = 0.0
                else:
                    dd = float(gg[t] - aa[t])
                    f_plan = float(c[t] * min(gg[t], aa[t]))
                    f_adj = float(c[t] * (0.5 * dd if dd > 0 else 1.5 * (-dd)))
                f_em = float(5.0 * c[t] * bb[t])
                rows.append([date, f"4-{br[1]}", j + 1, hhmm(s), hhmm(e), int(t) + 1,
                             f"{gg[t]:.6f}", f"{aa[t]:.6f}", f"{bb[t]:.6f}",
                             f"{c[t]:.6f}", f"{f_plan:.6f}", f"{f_adj:.6f}",
                             f"{f_em:.6f}", f"{f_plan + f_adj + f_em:.6f}"])

            if br == "42":
                bl = bill_42_def(c, gg, bb)
                aeff = gg
            else:
                bl = bill_43_def_four(c, gg, aa, bb)
                bl2 = bill_43_def_equiv(c, gg, aa, bb)
                aeff = aa
                same = abs(bl["total"] - bl2["total"])
                checks.append((f"指定日 {date} 4-43 四式与等价式总值一致", same <= 1e-9,
                               f"总值差 {same:.3e} 元（分项口径不同：四式 plan="
                               f"Σc·min(g,a)，等价式 plan=Σc·a）"))
            day_rows.append([date, f"4-{br[1]}", f"{float(gg.sum()):.6f}",
                             f"{float(aeff.sum()):.6f}", f"{float(bb.sum()):.6f}",
                             f"{float((aeff + bb).sum()):.6f}",
                             f"{bl['plan']:.6f}", f"{bl['adjust']:.6f}",
                             f"{bl['emerg']:.6f}", f"{bl['total']:.6f}",
                             f"{e0:.6f}", f"{e1:.6f}", f"{len(slots)}"])
            nb = float(np.asarray(rec["bill"], float)[di])
            ok = abs(bl["total"] - nb) <= max(1e-5, 1e-10 * abs(nb))
            checks.append((f"指定日 {date} 4-{br[1]} 定义式账单与引擎逐日账单一致", ok,
                           f"定义式 {bl['total']:.6f} 元 vs 引擎 {nb:.6f} 元"))
    log.append(f"  表 1：{len(rows)} 个时段行、{len(day_rows)} 个全天行（4 日期 ×2 分支）")
    return rows, day_rows


def table2(branches: dict, log: list, checks: list) -> list[list]:
    rows: list[list] = []
    for date in SPEC_DATES:
        for br in ("42", "43"):
            rec = branches.get(br)
            if rec is None:
                continue
            dates = np.asarray(rec["dates"], dtype="<U10")
            di = int(np.where(dates == date)[0][0])
            Cc = np.asarray(rec["C"], float)[di]
            Dd = np.asarray(rec["D"], float)[di]
            e0 = float(np.asarray(rec["E_chain"], float)[di])
            e1 = float(np.asarray(rec["E_chain"], float)[di + 1])
            for j, (s, e) in enumerate(T2_BLOCKS):
                rows.append([date, f"4-{br[1]}", j + 1,
                             f"{hhmm(s * 10)}–{hhmm(e * 10)}",
                             f"{float(Cc[s:e].sum()):.6f}", f"{float(Dd[s:e].sum()):.6f}",
                             f"{e0:.6f}", f"{e1:.6f}"])
            cs = float(sum(float(Cc[s:e].sum()) for s, e in T2_BLOCKS))
            ds = float(sum(float(Dd[s:e].sum()) for s, e in T2_BLOCKS))
            checks.append((f"表2 区块合计等于全天充放电（{date} 4-{br[1]}）",
                           abs(cs - float(Cc.sum())) <= 1e-9 and abs(ds - float(Dd.sum())) <= 1e-9,
                           f"Σ区块 充 {cs:.6f}/放 {ds:.6f} kWh vs 全天 充 "
                           f"{float(Cc.sum()):.6f}/放 {float(Dd.sum()):.6f} kWh"))
    log.append(f"  表 2：{len(rows)} 行（4 日期 ×2 分支 ×6 区块）")
    return rows


def table3(branches: dict, pm: dict, log: list, checks: list) -> list[list]:
    ssm = np.asarray(pm["slot_start_minute"], int)
    rows: list[list] = []
    for date in SPEC_DATES:
        for br in ("42", "43"):
            rec = branches.get(br)
            if rec is None:
                continue
            dates = np.asarray(rec["dates"], dtype="<U10")
            di = int(np.where(dates == date)[0][0])
            c = np.asarray(rec["c"], float)[di]
            bb = np.asarray(rec["b"], float)[di]
            pos = np.where(bb > 0.0)[0]
            if pos.size == 0:
                rows.append([date, f"4-{br[1]}", "—", "—", "0.000000",
                             "0.000000", "无紧急购电，0 kWh"])
                continue
            brk = np.where(np.diff(pos) > 1)[0]
            segs = np.split(pos, brk + 1)
            tot = 0.0
            for k, seg in enumerate(segs, start=1):
                q = float(bb[seg].sum())
                cos = float((5.0 * c[seg] * bb[seg]).sum())
                tot += q
                rows.append([date, f"4-{br[1]}", k,
                             f"{hhmm(int(ssm[seg[0]]))}–{hhmm(int(ssm[seg[-1]]) + 10)}",
                             f"{q:.6f}", f"{cos:.6f}",
                             f"{seg.size} 个连续时段"])
            checks.append((f"表3 事件电量合计等于当日紧急量（{date} 4-{br[1]}）",
                           abs(tot - float(bb.sum())) <= 1e-9,
                           f"事件合计 {tot:.6f} kWh vs 当日 {float(bb.sum()):.6f} kWh；"
                           f"{len(segs)} 个事件"))
    log.append(f"  表 3：{len(rows)} 行事件记录")
    return rows


def inventory_chain(rec, di: int) -> np.ndarray:
    Cc = np.asarray(rec["C"], float)[di]
    Dd = np.asarray(rec["D"], float)[di]
    e0 = float(np.asarray(rec["E_chain"], float)[di])
    out = np.empty(T + 1)
    out[0] = e0
    for t in range(T):
        out[t + 1] = out[t] + float(C.ETA_C) * Cc[t] - Dd[t] / float(C.ETA_D)
    return out


def setup_matplotlib() -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    used = "DejaVu Sans"
    for cand in FONT_CANDIDATES:
        if cand in avail:
            used = cand
            break
    plt.rcParams["font.sans-serif"] = [used, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return used


def _save(figdir: Path, stem: str, rows: list[list], header: list[str]) -> None:
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figdir / f"{stem}.png", dpi=160)
    plt.close()
    C.write_csv_utf8_sig(figdir / f"{stem}.csv", header, rows)


def fig1_price(branches, pm, figdir: Path, br_meta: dict, log: list) -> None:
    import matplotlib.pyplot as plt
    pa = np.asarray(pm["price_actual"], float)
    dates365 = np.asarray(pm["dates"], dtype="<U10")
    fp = np.load(C.PRICE_FORECAST_NPZ, allow_pickle=False)
    HALL = np.asarray(fp["H"], int)
    for date in SPEC_DATES:
        d = int(np.where(dates365 == date)[0][0])
        k0 = np.asarray(C.TAU_PERIOD_INDEX, int)
        H = HALL[d]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        x = np.arange(T)
        ax.plot(x, pa[d], color="black", lw=1.0, label="附件四实际价格（事后对照）")
        rows: list[list] = []
        cs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for ti, tau in enumerate(TAU_H):
            chat = np.asarray(fp[f"chat_corr__{br_meta['method']}"], float)[d, ti]
            n = int(H[ti])
            xs = int(k0[ti]) + np.arange(n)
            ax.plot(xs, chat[:n], lw=1.2, color=cs[ti + 1],
                    label=f"{tau} 时节点预测（展望 {n} 段）")
        ax.set_xlim(0, T - 1)
        ticks = np.arange(0, T + 1, 12)
        ax.set_xticks(ticks)
        ax.set_xticklabels([hhmm(int(t) * 10) for t in ticks], rotation=45, fontsize=7)
        ax.set_xlabel("时刻（区间起点）")
        ax.set_ylabel("价格 /（元·kWh⁻¹）")
        ax.set_title(f"{date} 实际价格与 {br_meta['method_label']} 各节点因果预测")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        for t in range(T):
            row = [date, t + 1, hhmm(t * 10), hhmm(t * 10 + 10), f"{pa[d, t]:.6f}"]
            for ti in range(len(TAU_H)):
                chat = np.asarray(fp[f"chat_corr__{br_meta['method']}"], float)[d, ti]
                n = int(H[ti])
                tt = t - int(k0[ti])
                row.append(f"{chat[tt]:.6f}" if 0 <= tt < n else "")
            rows.append(row)
        _save(figdir, f"第四问_图1_价格实际与节点预测_{date}",
              rows, ["日期", "时段序", "时刻起", "时刻止", "价格_实际"]
              + [f"预测_{h}时" for h in TAU_H])
    log.append("  图 1：4 张（价格实际值与各节点预测），含同名 CSV")


def fig2_series(branches, pm, figdir: Path, log: list) -> None:
    import matplotlib.pyplot as plt
    pa = np.asarray(pm["price_actual"], float)
    dates365 = np.asarray(pm["dates"], dtype="<U10")
    for date in SPEC_DATES:
        d = int(np.where(dates365 == date)[0][0])
        fig, axs = plt.subplots(3, 1, figsize=(11, 8.0), sharex=True)
        x = np.arange(T)
        axs[0].plot(x, pa[d], color="black", lw=1.0)
        axs[0].set_ylabel("实际价格\n/(元·kWh⁻¹)")
        axs[0].set_title(f"{date} 价格 / 净负荷 / 购电时序（实际价事后对照）")
        rows: list[list] = []
        net_ref = None
        for br, col in (("42", "tab:blue"), ("43", "tab:red")):
            rec = branches.get(br)
            if rec is None:
                continue
            dates = np.asarray(rec["dates"], dtype="<U10")
            di = int(np.where(dates == date)[0][0])
            nn = np.asarray(rec["N"], float)[di]
            net_ref = nn
            gg = np.asarray(rec["g"], float)[di]
            aa = np.asarray(rec["a"], float)[di]
            bb = np.asarray(rec["b"], float)[di]
            axs[1].step(x, gg, where="post", color=col, lw=1.0,
                        label=f"4-{br[1]} 计划 g")
            if br == "43":
                axs[1].step(x, aa, where="post", color="tab:green", lw=1.0, ls="--",
                            label="4-3 有效 a")
            axs[2].plot(x, bb, color=col, lw=1.2, label=f"4-{br[1]} 紧急 b")
        if net_ref is not None:
            axs[1].step(x, net_ref, where="post", color="gray", lw=0.8,
                        label="净负荷 N（kWh/10min）")
        axs[1].set_ylabel("购电量\n/(kWh·10min⁻¹)")
        axs[2].set_ylabel("紧急购电\n/(kWh·10min⁻¹)")
        axs[1].legend(fontsize=7, ncol=2)
        axs[2].legend(fontsize=7)
        for ax in axs:
            ax.grid(alpha=0.3)
        axs[2].set_xlim(0, T - 1)
        ticks = np.arange(0, T + 1, 12)
        axs[2].set_xticks(ticks)
        axs[2].set_xticklabels([hhmm(int(t) * 10) for t in ticks], rotation=45, fontsize=7)
        axs[2].set_xlabel("时刻（区间起点）")
        for t in range(T):
            row = [date, t + 1, hhmm(t * 10), hhmm(t * 10 + 10), f"{pa[d, t]:.6f}",
                   f"{float(net_ref[t]):.6f}" if net_ref is not None else ""]
            for br in ("42", "43"):
                rec = branches.get(br)
                if rec is None:
                    row += ["", "", ""]
                    continue
                dates = np.asarray(rec["dates"], dtype="<U10")
                di = int(np.where(dates == date)[0][0])
                row += [f"{float(np.asarray(rec['g'], float)[di, t]):.6f}",
                        f"{float(np.asarray(rec['a'], float)[di, t]):.6f}",
                        f"{float(np.asarray(rec['b'], float)[di, t]):.6f}"]
            rows.append(row)
        _save(figdir, f"第四问_图2_价格净负荷购电时序_{date}", rows,
              ["日期", "时段序", "时刻起", "时刻止", "价格_实际", "净负荷_kWh每10min",
               "计划g_42", "有效a_42", "紧急b_42", "计划g_43", "有效a_43", "紧急b_43"])
    log.append("  图 2：4 张（价格/净负荷/购电时序），含同名 CSV")


def fig3_inventory(branches, figdir: Path, log: list, checks: list) -> None:
    import matplotlib.pyplot as plt
    for date in SPEC_DATES:
        fig, ax = plt.subplots(figsize=(11, 4.4))
        rows: list[list] = []
        for br, col in (("42", "tab:blue"), ("43", "tab:red")):
            rec = branches.get(br)
            if rec is None:
                continue
            dates = np.asarray(rec["dates"], dtype="<U10")
            di = int(np.where(dates == date)[0][0])
            ech = inventory_chain(rec, di)
            Rr = np.asarray(rec["R"], float)[di]
            x = np.arange(T + 1)
            ax.plot(x, ech, color=col, lw=1.2, label=f"4-{br[1]} 内部储电量")
            m = np.isfinite(Rr)
            ax.step(np.arange(T)[m], Rr[m], where="post", color=col, lw=0.8, ls=":",
                    label=f"4-{br[1]} 保留水平 R")
        ax.axhspan(float(C.E_MIN), float(C.E_MAX), color="tab:gray", alpha=0.10,
                   label=f"安全区间 [{float(C.E_MIN):.0f}, {float(C.E_MAX):.0f}] kWh")
        ax.set_xlim(0, T)
        ticks = np.arange(0, T + 1, 12)
        ax.set_xticks(ticks)
        ax.set_xticklabels([hhmm(int(t) * 10) for t in ticks], rotation=45, fontsize=7)
        ax.set_xlabel("时刻（区间起点）")
        ax.set_ylabel("电量 / kWh")
        ax.set_title(f"{date} 内部储电量与保留水平")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        for t in range(T + 1):
            row = [date, t, hhmm(t * 10), hhmm(t * 10)]
            for br in ("42", "43"):
                rec = branches.get(br)
                if rec is None:
                    row += ["", ""]
                    continue
                dates = np.asarray(rec["dates"], dtype="<U10")
                di = int(np.where(dates == date)[0][0])
                ech = inventory_chain(rec, di)
                Rr = np.asarray(rec["R"], float)[di]
                row += [f"{ech[t] if t <= T else float('nan'):.6f}",
                        (f"{float(Rr[t]):.6f}" if (t < T and np.isfinite(Rr[t])) else "")]
            rows.append(row)
        _save(figdir, f"第四问_图3_库存与保留水平_{date}", rows,
              ["日期", "库存链序", "时刻起", "时刻止", "库存_42", "保留水平_42",
               "库存_43", "保留水平_43"])
    log.append("  图 3：4 张（库存与保留水平），含同名 CSV")


def fig4_monthly(figdir: Path, log: list) -> None:
    import matplotlib.pyplot as plt
    import csv
    if not C.MONTHLY_CSV.exists():
        log.append("  图 4：跳过（缺 08 的月度费用分解表）")
        return
    with open(C.MONTHLY_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rd = list(csv.DictReader(f))
    if not rd:
        log.append("  图 4：跳过（月度表为空）")
        return
    need = ("branch", "month", "plan_cost", "adjust_cost", "emerg_cost", "total_cost")
    if any(k not in rd[0] for k in need):
        log.append(f"  图 4：跳过（月度表缺列，实有 {list(rd[0].keys())}）")
        return
    by = {"42": [], "43": []}
    for r in rd:
        if r["branch"] in by:
            by[r["branch"]].append(r)
    mon42 = [r["month"] for r in by["42"]]
    mon43 = [r["month"] for r in by["43"]]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
    p42 = [float(r["plan_cost"]) for r in by["42"]]
    e42 = [float(r["emerg_cost"]) for r in by["42"]]
    axs[0].bar(mon42, p42, label="计划费", color="tab:blue")
    axs[0].bar(mon42, e42, bottom=p42, label="紧急费", color="tab:red")
    axs[0].set_title("4-2 月度费用分解")
    p43 = [float(r["plan_cost"]) for r in by["43"]]
    a43 = [float(r["adjust_cost"]) for r in by["43"]]
    e43 = [float(r["emerg_cost"]) for r in by["43"]]
    axs[1].bar(mon43, p43, label="计划费(a)", color="tab:blue")
    axs[1].bar(mon43, a43, bottom=p43, label="调整费", color="tab:orange")
    axs[1].bar(mon43, e43, bottom=[p43[i] + a43[i] for i in range(len(p43))],
               label="紧急费", color="tab:red")
    axs[1].set_title("4-3 月度费用分解")
    for ax in axs:
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("费用 / 元")
        ax.set_xlabel("月份")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    rows = [[r["branch"], r["month"], r["n_days"] if "n_days" in r else "",
             f"{float(r['plan_cost']):.6f}", f"{float(r['adjust_cost']):.6f}",
             f"{float(r['emerg_cost']):.6f}", f"{float(r['total_cost']):.6f}",
             r.get("E_end_kwh", "")] for r in rd]
    _save(figdir, "第四问_图4_月度费用分解", rows,
          ["分支", "月份", "天数", "计划费_元", "调整费_元", "紧急费_元", "合计_元",
           "月末库存_kWh"])
    log.append(f"  图 4：1 张（月度费用分解），{len(rows)} 行同名 CSV")


def fig5_repricing(figdir: Path, log: list) -> list[list]:
    import matplotlib.pyplot as plt
    ab = C.RESULT_DIR / "09_消融"
    main42 = C.BACKTEST_42_NPZ
    if not (ab / "fixed42.npz").exists() or not main42.exists():
        log.append("  图 5：跳过（缺 09 的 fixed42 消融档）")
        return []
    with np.load(main42, allow_pickle=False) as Z:
        main = {k: Z[k] for k in Z.files}
    with np.load(ab / "fixed42.npz", allow_pickle=False) as Z:
        f42 = {k: Z[k] for k in Z.files}
    dates = np.asarray(main["dates"], dtype="<U10")
    months = sorted({month_of(x) for x in dates})
    with np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False) as Z:
        pa = np.asarray(Z["price_actual"], float)
        d365 = list(np.asarray(Z["dates"], dtype="<U10"))
    day_of_42 = np.asarray(main["days"], int)
    day_of_fix = np.asarray(f42["days"], int)
    frow = {int(v): i for i, v in enumerate(day_of_fix)}
    overlap = [d for d in day_of_42 if int(d) in frow]
    rows: list[list] = []
    for m in months:
        sel_m = np.array([i for i, x in enumerate(dates) if month_of(x) == m], int)
        if sel_m.size == 0:
            continue
        sel = np.array([i for i in sel_m if int(day_of_42[i]) in frow], int)
        if sel.size == 0:
            continue
        dd = day_of_42[sel]
        c = pa[dd]
        jf = np.array([frow[int(d)] for d in dd], int)
        gm = np.asarray(main["g"], float)[sel]
        bm = np.asarray(main["b"], float)[sel]
        bill_main = float((c * gm).sum() + (5.0 * c * bm).sum())
        gf = np.asarray(f42["g"], float)[jf]
        bf = np.asarray(f42["b"], float)[jf]
        bill_fix_act = float((c * gf).sum() + (5.0 * c * bf).sum())
        rows.append([m, f"{bill_main:.6f}", f"{bill_fix_act:.6f}",
                     f"{bill_fix_act - bill_main:.6f}",
                     f"{100 * (bill_fix_act - bill_main) / bill_main:.4f}"])
    if len(overlap) != len(day_of_42):
        log.append(f"  图 5：注意——固定价档与主方案可对齐天数 {len(overlap)}/{len(day_of_42)}")
    if not rows:
        log.append("  图 5：跳过（无重叠月份）")
        return []
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [float(r[1]) for r in rows], width=0.4, label="主方案（价格自适应）")
    ax.bar(x + 0.2, [float(r[2]) for r in rows], width=0.4,
           label="固定价动作按附件四重计价")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=45, fontsize=8)
    ax.set_ylabel("费用 / 元")
    ax.set_title("对固定价动作按附件四重计价的月度对比（4-2）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(figdir, "第四问_图5_重计价月度收益", rows,
          ["月份", "主方案费_元", "固定价动作重计价_元", "差额_元", "差额_pct"])
    log.append(f"  图 5：1 张（重计价月度收益），{len(rows)} 行同名 CSV")
    return rows


def fig6_model_update(figdir: Path, log: list) -> list[list]:
    import matplotlib.pyplot as plt
    ab = C.RESULT_DIR / "09_消融"
    items = [("main42", C.BACKTEST_42_NPZ, "4-2 主方案(净负荷回归, 4 次更新)"),
             ("vn42_0", ab / "vn42_0.npz", "4-2 仅 0 时更新"),
             ("method42_ar1", ab / "method42_ar1.npz", "4-2 AR(1)"),
             ("method42_arx", ab / "method42_arx.npz", "4-2 AR-X"),
             ("main43", C.BACKTEST_43_NPZ, "4-3 主方案(AR(1), 4 节点)"),
             ("vn43_0", ab / "vn43_0.npz", "4-3 仅 0 时更新"),
             ("vn43_06", ab / "vn43_06.npz", "4-3 0+6 时更新"),
             ("vn43_0612", ab / "vn43_0612.npz", "4-3 0+6+12 时更新"),
             ("method43_nlr", ab / "method43_nlr.npz", "4-3 净负荷回归"),
             ("method43_arx", ab / "method43_arx.npz", "4-3 AR-X")]
    rows: list[list] = []
    for tag, path, lab in items:
        if not Path(path).exists():
            continue
        with np.load(path, allow_pickle=False) as Z:
            d = {k: Z[k] for k in Z.files}
        tot, plan, adj, em = totals_of(d)
        rows.append([tag, lab, f"{tot:.6f}", f"{plan:.6f}", f"{adj:.6f}", f"{em:.6f}"])
    if not rows:
        log.append("  图 6：跳过（无可用消融档）")
        return []
    mae = {}
    prec = C.PRICE_DIAG_CSV
    if prec.exists():
        import csv
        with open(prec, "r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                k = (r.get("方法") or r.get("method") or "").strip()
                seg = (r.get("区间类型") or "").strip()
                v = r.get("MAE_元每kWh") or r.get("MAE") or ""
                if k and v and ("全部" in seg or r.get("区间", "").strip() == "全年"):
                    mae[k] = v
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [float(r[3]) for r in rows], width=0.4, label="计划费")
    ax.bar(x + 0.2, [float(r[5]) for r in rows], width=0.4, label="紧急费")
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], rotation=35, fontsize=8, ha="right")
    ax.set_ylabel("费用 / 元")
    ax.set_title("价格模型与更新频率对照（同窗口、同初值；费用为实际账单）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(figdir, "第四问_图6_价格模型与更新频率对照", rows,
          ["实验", "说明", "合计_元", "计划费_元", "调整费_元", "紧急费_元"])
    if mae:
        log.append(f"  图 6：1 张（价格模型与更新频率对照），价格精度表 MAE："
                   + "、".join(f"{k}={v}" for k, v in sorted(mae.items())))
    else:
        log.append("  图 6：1 张（价格模型与更新频率对照）")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="第四问 10 指定日期明细与作图")
    ap.add_argument("--no-fig", action="store_true", help="只出表，不出图")
    args = ap.parse_args()

    C.ensure_dirs()
    log: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 78)
    p("第四问 10 —— 指定日期明细与作图（§12.1）")
    p("=" * 78)

    pm = load_price_matrix()
    branches = {"42": load_branch("42"), "43": load_branch("43")}
    have = [b for b in ("42", "43") if branches[b] is not None]
    if not have:
        p("  [错误] 两个分支的回测 npz 均不存在，先运行 06/07。")
        return 2
    p(f"  可用分支：{', '.join('4-' + b[1] for b in have)}；指定日期：{'、'.join(SPEC_DATES)}")

    for br in have:
        ds = list(np.asarray(branches[br]["dates"], dtype="<U10"))
        miss = [x for x in SPEC_DATES if x not in ds]
        checks.append((f"4-{br[1]} 指定日期均落在回测窗口内", not miss,
                       "全部命中" if not miss else f"缺失 {miss}"))

    r1, d1 = table1(branches, pm, log, checks)
    C.write_csv_utf8_sig(C.SPEC_SLOTS_CSV, ["日期", "分支", "序号", "时刻起", "时刻止",
                                            "时段序", "计划g_kWh", "有效a_kWh", "紧急b_kWh",
                                            "实际价_元每kWh", "计划费_元", "调整费_元",
                                            "紧急费_元", "小计_元"], r1)
    C.write_csv_utf8_sig(C.SPEC_DAY_CSV, ["日期", "分支", "计划量_kWh", "有效普通购电量_kWh",
                                          "紧急量_kWh", "实际总取电量_kWh", "计划费_元",
                                          "调整费_元", "紧急费_元", "合计_元",
                                          "0时库存_kWh", "24时库存_kWh", "表1时段数"], d1)
    p(f"  表 1 → {C.SPEC_SLOTS_CSV.name} / {C.SPEC_DAY_CSV.name}")
    p("  说明：4-2 的有效普通购电量即 a≡g；4-3 的「调整费」= 0.5c(g−a)⁺ + 1.5c(a−g)⁺，"
      "「计划费」= Σc·min(g,a)（B 结算分解，总量等于 Σc{a+0.5|a−g|}）。")

    r2 = table2(branches, log, checks)
    C.write_csv_utf8_sig(C.SPEC_BATT_CSV, ["日期", "分支", "区块", "时段区间",
                                           "充电量_kWh", "放电量_kWh",
                                           "0时储电量_kWh", "24时储电量_kWh"], r2)
    p(f"  表 2 → {C.SPEC_BATT_CSV.name}（{len(r2)} 行）")

    r3 = table3(branches, pm, log, checks)
    C.write_csv_utf8_sig(C.SPEC_EMERG_CSV, ["日期", "分支", "事件号", "时间区间",
                                            "紧急电量_kWh", "紧急费_元", "备注"], r3)
    p(f"  表 3 → {C.SPEC_EMERG_CSV.name}（{len(r3)} 行）")

    for br in have:
        rec = branches[br]
        mx = 0.0
        for di in range(len(rec["dates"])):
            ech = inventory_chain(rec, di)
            mx = max(mx, abs(ech[-1] - float(np.asarray(rec["E_chain"], float)[di + 1])))
        checks.append((f"4-{br[1]} 逐日库存链由充放电重建后与 E_chain 一致", mx <= 1e-6,
                       f"最大差 {mx:.3e} kWh"))

    if not args.no_fig:
        font = setup_matplotlib()
        p(f"  作图字体：{font}")
        br_meta = {"method": str(np.asarray(branches["42"]["method"]).reshape(-1)[0])
                   if "method" in branches["42"] else "net_load_regression"}
        br_meta["method_label"] = "净负荷回归" if "net_load" in br_meta["method"] else br_meta["method"]
        fig1_price(branches, pm, C.FIGURE_DIR, br_meta, log)
        fig2_series(branches, pm, C.FIGURE_DIR, log)
        fig3_inventory(branches, C.FIGURE_DIR, log, checks)
        fig4_monthly(C.FIGURE_DIR, log)
        fig5_repricing(C.FIGURE_DIR, log)
        fig6_model_update(C.FIGURE_DIR, log)

    nb = sum(1 for _n, ok, _d in checks if not ok)
    rp: list[str] = []
    rp.append("# 第四问 指定日期明细与作图报告\n")
    rp.append("> 对应流程图 §12.1。两分支均对 "
              + "、".join(f"`{d}`" for d in SPEC_DATES) + " 输出。\n")
    rp.append("## 1. 产出清单\n")
    rp.append(f"- 表 1：`模型结果/{C.SPEC_SLOTS_CSV.name}`（六个 10:00→20:10 起始的 10 分钟段）"
              f"与 `模型结果/{C.SPEC_DAY_CSV.name}`（全天普通/紧急/总取电/费用分列）")
    rp.append(f"- 表 2：`模型结果/{C.SPEC_BATT_CSV.name}`（六个 4 小时区块充放电 + 0 时/24 时储电量）")
    rp.append(f"- 表 3：`模型结果/{C.SPEC_EMERG_CSV.name}`（同日相邻且均为正的时段才合并；"
              "无事件写「无紧急购电，0 kWh」）")
    rp.append(f"- 图 1–6 与同名作图数据 CSV：`模型结果图/`（价格实际与节点预测、"
              "价格/净负荷/购电时序、库存与保留水平、月度费用分解、重计价月度收益、"
              "价格模型与更新频率对照）")
    rp.append("")
    rp.append("## 2. 口径说明\n")
    rp.append("- 时段定位使用**附件四原始起始分钟**匹配，不依赖列序或区间端点约定"
              "（本问时间标签为区间**结束**时刻，10:00–10:10 对应起始分钟 600）。")
    rp.append("- 表 1 的购电量单位为 kWh；价格单位为元/kWh（**不再乘 Δt**）；"
              "费用 = 价格×电量，量纲只乘一次。")
    rp.append("- 4-3 的 `a` 是交付前的最终有效普通购电量，**不是**相对 `g` 的增量；"
              "`a+5b` 不是物理购电量。")
    rp.append("- 图中「附件四实际价格」在决策时点并不可得，仅作事后对照；"
              "CSV 中已单列该列。跨时段窗口按真实时刻刻度显示。")
    rp.append("- 账单均由本脚本的**定义式**独立复算，未调用主结算函数再与自身比较。")
    rp.append("")
    rp.append("## 3. 检查项\n")
    rp.append("| 检查项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    shown = checks if len(checks) <= 40 else checks[:12] + checks[-12:]
    for name, ok, detail in shown:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    if len(checks) > len(shown):
        rp.append(f"| （中间 {len(checks) - len(shown)} 项省略） | | |")
    rp.append("")
    rp.append("## 4. 运行日志\n")
    rp.append("```")
    rp.extend(log)
    rp.append("```")
    C.write_text_utf8(C.REPORT_DIR / "第四问_指定日期结果报告.md", "\n".join(rp))
    C.write_text_utf8(C.LOG_DIR / "10_指定日期日志.txt", "\n".join(log) + "\n")

    p("")
    p("=" * 78)
    p(f"10 完成：检查 {len(checks)} 项，未通过 {nb} 项")
    p("=" * 78)
    return 0 if nb == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
