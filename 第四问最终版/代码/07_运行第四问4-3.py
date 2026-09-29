#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


C = _load("_comm4.py", "q4_comm")
PO = _load("_policy4.py", "q4_policy")
ST = _load("_settlement4.py", "q4_settle")

import numpy as np

T = C.PERIODS_PER_DAY
BRANCH = "43"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="第四问 4-3 全年回测")
    ap.add_argument("--days", type=int, default=C.N_SCORE_DAYS,
                    help="评分期天数（默认 334；小样本调试用，正式交付必须 334）")
    ap.add_argument("--suffix", type=str, default="",
                    help="输出文件名后缀（调试档专用，避免覆盖正式产物）")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    t0 = time.perf_counter()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg, flush=True)

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail):
        checks.append((name, bool(ok), detail))
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    p("=" * 78)
    p("第四问 07 —— 4-3 全年回测（波动价格 / 6、12、18 时节点更新）")
    p("=" * 78)

    suf = str(args.suffix or "").strip()
    if not suf:
        suf = ""

    Zw = np.load(C.WARMUP_NPZ, allow_pickle=False)
    E_start = float(np.asarray(Zw["E_feb1"]).reshape(-1)[0])
    n_days = int(args.days)
    if not 1 <= n_days <= C.N_SCORE_DAYS:
        p(f"  --days 必须在 1..{C.N_SCORE_DAYS} 内")
        return 2
    days = np.arange(C.N_WARMUP_DAYS, C.N_WARMUP_DAYS + n_days)
    prov = ST.SnapshotProviders(BRANCH)
    p("")
    p("── 1. 口径与输入 ──")
    p(f"  共同起点（与 4-2 相同）：E(2025-02-01) = {E_start:,.4f} kWh")
    p(f"  样本窗口：{prov.dates[days[0]]} … {prov.dates[days[-1]]}（{n_days} 天）")
    p(f"  价格预测方法：{prov.method}；采购更新时刻："
      f"{list(C.CONFIG['purchase_update_hours_43'])} 时（相对 0 时原始计划的调整）")
    ck(f"物理/结算/时间口径与 4-2 一致（T=144、η=0.9、S=5000/6、5c 紧急）",
       bool(T == 144 and abs(C.S_PERIOD_KWH - 5000.0 / 6.0) <= 1e-9
            and abs(C.ETA - 0.9) <= 1e-12 and C.E_MIN == 1200.0
            and C.E_MAX == 10800.0 and abs(C.EMERG_MULT - 5.0) <= 1e-12
            and abs(C.RHO_DOWN - 0.5) <= 1e-12 and abs(C.RHO_UP - 1.5) <= 1e-12
            and tuple(C.CONFIG["value_update_hours"]) == (0, 6, 12, 18)
            and tuple(C.CONFIG["purchase_update_hours_43"]) == (6, 12, 18)),
       f"T={T}、S={C.S_PERIOD_KWH:.4f} kWh/段、η={C.ETA}、E∈"
       f"[{C.E_MIN:g},{C.E_MAX:g}]、r_down={C.RHO_DOWN}、r_up={C.RHO_UP}、"
       f"紧急倍数={C.EMERG_MULT:g}；节点 {tuple(C.CONFIG['value_update_hours'])}")

    if C.BACKTEST_42_NPZ.exists():
        Z42 = np.load(C.BACKTEST_42_NPZ, allow_pickle=False)
        d42 = np.asarray(Z42["days"], int).reshape(-1)
        e42 = float(np.asarray(Z42["E_start"]).reshape(-1)[0])
        dt42 = np.asarray(Z42["dates"]).reshape(-1)
        start_ok = abs(e42 - E_start) <= 1e-9
        if n_days == C.N_SCORE_DAYS:
            win_ok = bool(start_ok and d42.size == n_days
                          and np.array_equal(d42, np.asarray(days, int))
                          and np.array_equal(dt42, np.asarray(prov.dates[days])))
            win_note = (f"对照 {C.BACKTEST_42_NPZ.name}：同一起点 E(2025-02-01)="
                        f"{e42:,.4f} kWh（差 {abs(e42 - E_start):.2e}）、同一 "
                        f"{d42.size} 天窗口 {dt42[0]}…{dt42[-1]}")
        else:
            win_ok = bool(start_ok)
            win_note = (f"调试档 {n_days} 天（非正式 {C.N_SCORE_DAYS} 天）："
                        f"仅对照同一起点 {e42:,.4f} kWh（差 {abs(e42 - E_start):.2e}）")
    else:
        win_ok, win_note = False, f"4-2 正式产物缺失：{C.BACKTEST_42_NPZ}"
    ck("起点与样本窗口与 4-2 严格一致（逐项对照 4-2 产物）", win_ok, win_note)

    p("")
    p("── 2. 闭环运行 ──")
    tm = time.perf_counter()
    r = ST.run_policy4(BRANCH, prov, prov, initial_state=E_start, days=days)
    wall = time.perf_counter() - tm
    p(f"  用时 {wall:.2f} s；LP {r['n_lp']} 次（{wall / max(r['n_lp'], 1) * 1000:.0f} ms/次）")
    p(f"  账单合计 {r['cost_total']:,.2f} 元 = 计划 {r['cost_plan']:,.2f}"
      f" + 调整 {r['cost_adjust']:,.2f} + 紧急 {r['cost_emerg']:,.2f}")
    p(f"  期末库存 E({prov.dates[days[-1]]} 24:00) = {r['E_final']:,.2f} kWh")

    p("")
    p("── 3. 节点更新与同目标比价 ──")
    acc = r["accept_rows"]
    tol_rule = all(
        a["accepted"] == bool(a["fun_new"] <= a["fun_old"] + 1e-6 * max(1.0, abs(a["fun_old"])))
        for a in acc)
    n_acc = sum(1 for a in acc if a["accepted"])
    by_tau = {}
    for a in acc:
        s = by_tau.setdefault(a["tau"], {"n": 0, "acc": 0, "l1": 0.0, "mx": 0.0})
        s["n"] += 1
        s["acc"] += int(bool(a["accepted"]))
        s["l1"] += float(a["d_a_l1"])
        s["mx"] = max(s["mx"], float(a["d_a_max"]))
    p("  | τ | 更新次数 | 接受 | 接受率 | Σ‖Δa‖₁/kWh | max|Δa|/kWh |")
    p("  |---|---|---|---|---|---|")
    for tau in sorted(by_tau):
        s = by_tau[tau]
        p(f"  | {tau}:00 | {s['n']} | {s['acc']} | {100 * s['acc'] / s['n']:.1f}% | "
          f"{s['l1']:,.1f} | {s['mx']:,.2f} |")
    ck("每次节点更新都在**同一目标函数**下与新计划比价（旧计划始终为候选）",
       len(acc) == n_days * 3,
       f"{len(acc)} 次更新 = {n_days} 天 × 3 节点；接受 {n_acc} 次"
       f"（{100 * n_acc / max(len(acc), 1):.1f}%）")
    ck("接受判定与留档目标值自洽（不劣才接受）", tol_rule,
       "逐行核对 accepted ⇔ fun_new ≤ fun_old + 1e-6·max(1,|fun_old|)")
    ck("原始计划 g 在 0 时确定后不被改写（结构性）",
       bool(np.all(r["exec_cnt"] == 1)),
       f"已执行 {int(r['exec_cnt'].sum())} 段，每段恰好写入一次，"
       "后续节点只改尚未执行时段")
    d_ga = float(np.abs(r["a"] - r["g"]).max())
    ck("4-3 确实发生了计划调整（否则与 4-2 同构）", d_ga > 1e-9,
       f"max|a−g| = {d_ga:,.4f} kWh；Σ|a−g| = "
       f"{float(np.abs(r['a'] - r['g']).sum()):,.1f} kWh")

    p("")
    p("── 4. 校验 ──")
    N_ref = np.asarray(prov.net_act[r["days"]], float)

    bal = float(np.abs(r["a"] + r["b"] + r["D"] - r["C"] - r["U"] - N_ref).max())
    ck("A07 逐时段能量平衡 a+b+D−C−U = N", bal < 1e-6,
       f"{n_days * T} 段最大绝对残差 {bal:.3e} kWh")

    E_chain = np.concatenate([[E_start], r["E_end"]])
    E = E_start
    rec_err = 0.0
    e_lo, e_hi = E_start, E_start
    for di in range(n_days):
        for t in range(T):
            E = E + C.ETA * r["C"][di, t] - r["D"][di, t] / C.ETA
            e_lo, e_hi = min(e_lo, E), max(e_hi, E)
        rec_err = max(rec_err, abs(E - r["E_end"][di]))
        E = r["E_end"][di]
    ck("A06 库存递推与日末库存逐日一致", rec_err < 1e-6, f"最大误差 {rec_err:.3e} kWh")
    ck("A06 跨日状态连续（日初=前日末）", rec_err < 1e-6,
       f"库存链 {E_chain.size} 点自洽，E(2/1) = {E_chain[0]:,.4f} → "
       f"E({prov.dates[days[-1]]}) = {E_chain[-1]:,.4f}")
    ck("A08 逐时段库存轨迹在安全区间内",
       bool(e_lo >= C.E_MIN - 1e-9 and e_hi <= C.E_MAX + 1e-9),
       f"全期轨迹 [{e_lo:.2f}, {e_hi:.2f}] ⊂ [{C.E_MIN:g}, {C.E_MAX:g}] kWh")
    ck("A08 充放电量不超功率上限",
       bool(r["C"].max() <= C.S_PERIOD_KWH + 1e-9 and r["D"].max() <= C.S_PERIOD_KWH + 1e-9),
       f"max C {r['C'].max():.4f}、max D {r['D'].max():.4f} ≤ {C.S_PERIOD_KWH:.4f} kWh/段")
    ck("A08 执行层充放互斥 C·D = 0", float(np.minimum(r["C"], r["D"]).sum()) <= 1e-6,
       f"Σmin(C,D) = {float(np.minimum(r['C'], r['D']).sum()):.3e} kWh")
    ck("A08 弃电量非负", float(r["U"].min()) >= -1e-9, f"min U = {float(r['U'].min()):.3e} kWh")
    deg_cd = max(n["deg_cd"] for n in r["node_rows"])
    deg_bc = max(n["deg_bc"] for n in r["node_rows"])
    cap_v = max(n["cap_viol"] for n in r["node_rows"])
    n_pin = sum(1 for n in r["node_rows"] if n["cap_pinned"])
    conv = [n for n in r["node_rows"] if not n["cap_pinned"]]
    deg_cd_conv = max(n["deg_cd"] for n in conv)
    pin_bad = [n for n in r["node_rows"] if n["cap_pinned"] and n["deg_cd"] > 1e-6]
    pin_ratio = max([n["deg_cd"] / max(1.0, float(n["sum_C"] or 0.0))
                     for n in pin_bad] or [0.0])
    ck("A08 规划层充放互斥（收敛节点，严格）", deg_cd_conv <= 1e-6,
       f"非钉住节点 {len(conv)} 个，max $\\sum\\min(C,D)$ = {deg_cd_conv:.3e} kWh")
    ck("A08 规划层退化幅度已受限并披露（钉住节点）", pin_ratio <= 1e-3,
       f"钉住节点 {n_pin} 个，其中 {len(pin_bad)} 个存在等价最优取点差异，"
       f"max $\\sum\\min(C,D)$ = {deg_cd:.3f} kWh，"
       f"相对本节点充放电总量 ≤ {pin_ratio:.3e}（阈值 1e-3）")
    ck("A08 规划层无「紧急购电为电池充电」", deg_bc <= 1e-6 and cap_v <= 1e-6,
       f"max Σmin(b,C) = {deg_bc:.3e} kWh；一致性上界最大越界 {cap_v:.3e} kWh")
    rg = [n["relax_gap"] for n in r["node_rows"] if n["relax_gap"] is not None]
    ck("A08 执行一致性上界的代价已量化", max(rg) <= 0.05,
       f"相对缺口最大 {max(rg) * 100:.3f}%、均值 {float(np.mean(rg)) * 100:.3f}%；"
       f"固定 a 重解 {n_pin}/{len(r['node_rows'])} 节点")

    ck("A11 LP 全部取得最优解（无静默回退）", r["lp_status_codes"] == [0],
       f"状态码集合 {r['lp_status_codes']}")
    n_exec_bad = int((r["exec_cnt"] != 1).sum())
    ck("每日 144 段各被恰好执行一次", bool(n_exec_bad == 0 and r["n_unexec"] == 0),
       f"异常段数 {n_exec_bad}；未执行段数 {r['n_unexec']}")

    c = r["c"]
    g_v, a_v, b_v = r["g"], r["a"], r["b"]
    down = 0.5 * c * np.maximum(g_v - a_v, 0.0)
    up = 1.5 * c * np.maximum(a_v - g_v, 0.0)
    lim = c * np.minimum(g_v, a_v)
    emerg = 5.0 * c * b_v
    f1 = float((c * (a_v + 0.5 * np.abs(a_v - g_v))).sum() + emerg.sum())
    f2 = float((lim + down + up).sum() + emerg.sum())
    ck("A09 账单定义式与 B 口径等价式独立复算一致",
       abs(f1 - f2) < max(1e-5, 1e-10 * abs(f1)),
       f"定义式 {f1:,.6f} 元 vs 等价式 {f2:,.6f} 元（差 {abs(f1 - f2):.3e}）")
    ck("A09 引擎合计 = 独立复算（定义式）",
       abs(f1 - r["cost_total"]) < max(1e-5, 1e-10 * abs(r["cost_total"])),
       f"引擎 {r['cost_total']:,.6f} 元 vs 复算 {f1:,.6f} 元")
    bill_err = float(np.abs(r["bill"] - (r["plan_cost"] + r["adjust_cost"]
                                        + r["emerg_cost"])).max())
    ck("A09 逐日账单 = 计划 + 调整 + 紧急", bill_err < 1e-6, f"最大误差 {bill_err:.3e} 元")
    p(f"  调整费分解：下调 Σ0.5c(g−a)⁺ = {float(down.sum()):,.2f} 元；"
      f"上调 Σ1.5c(a−g)⁺ = {float(up.sum()):,.2f} 元；"
      f"合计 {float((down + up).sum()):,.2f} 元 = 引擎调整费 {r['cost_adjust']:,.2f} 元")
    ck("A14 终端续存价值 −νE 未计入实际账单",
       abs(r["cost_total"] - f1) < max(1e-5, 1e-10 * abs(r["cost_total"])),
       "账单只含普通采购 + 调整 + 紧急三项实际现金项")

    naive = float((c * np.maximum(N_ref, 0.0)).sum())
    naive_signed = float((c * N_ref).sum())
    ck("量级 sanity：账单不严重劣于无储能事后基准（Σ c_t·[N_t]⁺）",
       r["cost_total"] <= naive * 1.05,
       f"4-3 {r['cost_total']:,.2f} 元 vs 基准 {naive:,.2f} 元"
       f"（差 {r['cost_total'] - naive:+,.2f} 元；基准为事后口径、非可执行策略；"
       f"旧写法 Σc·N = {naive_signed:,.2f} 元，差 {naive - naive_signed:,.2f} 元 = "
       f"富余段被当成售电收益，已废弃）")

    K0_BY_TAU = {int(h): int(h) * 6 for h in C.TAU_HOURS}
    hor_bad = [(n["date"], n["tau"], int(n["H"]), int(n["H_full"]))
               for n in r["node_rows"]
               if int(n["H"]) != (int(n["H_full"]) if n["tail"]
                                   else min(int(n["H_full"]),
                                            T - K0_BY_TAU[int(n["tau"])]))]
    vo_bad = [(n["date"], n["tau"]) for n in r["node_rows"]
              if int(n["value_origin"]) != (0 if int(n["tau"]) == 0
                                             else K0_BY_TAU[int(n["tau"])])]
    n_tail = sum(1 for n in r["node_rows"] if n["tail"])
    ck("A03 决策展望 = min(库内可用, 当日剩余时段)；价值函数原点 = 节点全局时段号",
       (not hor_bad) and (not vo_bad) and n_tail == 0,
       f"{len(r['node_rows'])} 个节点：H 取值集合 "
       f"{sorted({int(n['H']) for n in r['node_rows']})}；"
       f"越界 {len(hor_bad)} 项、原点异常 {len(vo_bad)} 项；跨日 tail 节点 {n_tail} 个"
       + (f"；示例 {hor_bad[:2]}" if hor_bad else ""))

    cpos = np.array([float(x) for x in np.sort(c.reshape(-1))])
    q90 = float(cpos[int(0.90 * (cpos.size - 1))])
    m_hi = (c >= q90) & (b_v > 1e-9)
    hi_share = float(b_v[m_hi].sum()) / max(float(b_v.sum()), 1e-9)
    p(f"  高价缺电重合：价格 ≥ 90 分位（{q90:.4f} 元/kWh）时段的紧急电量占 "
      f"{100 * hi_share:.1f}%")

    cmp_rows: list[tuple[str, str, str, str]] = []
    p("")
    p("── 5. 与 4-2 对照 ──")
    ref42 = None
    p42 = C.BACKTEST_42_NPZ
    if not suf and p42.is_file():
        try:
            ref42 = np.load(p42, allow_pickle=False)
            if np.asarray(ref42["days"], int).size == n_days and \
                    np.asarray(ref42["dates"]).size == n_days and \
                    abs(float(np.asarray(ref42["E_start"]).reshape(-1)[0]) - E_start) < 1e-9:
                pass
            else:
                ref42 = None
        except Exception as exc:
            p(f"  （读取 4-2 产物失败，跳过对照：{exc}）")
            ref42 = None
    if ref42 is None:
        p("  （未找到可比的 4-2 全轨迹，本段仅报告 4-3 自身；正式对照在 08/09）")
    else:
        rows = [
            ("账单合计/元", r["cost_total"], float(np.asarray(ref42["bill"]).sum())),
            ("计划费/元", r["cost_plan"], float(np.asarray(ref42["plan_cost"]).sum())),
            ("调整费/元", r["cost_adjust"], float(np.asarray(ref42["adjust_cost"]).sum())),
            ("紧急费/元", r["cost_emerg"], float(np.asarray(ref42["emerg_cost"]).sum())),
            ("紧急电量/kWh", float(r["b"].sum()), float(np.asarray(ref42["b"]).sum())),
            ("普通购电 a/kWh", float(r["a"].sum()), float(np.asarray(ref42["a"]).sum())),
            ("放电电量/kWh", float(r["D"].sum()), float(np.asarray(ref42["D"]).sum())),
            ("充电电量/kWh", float(r["C"].sum()), float(np.asarray(ref42["C"]).sum())),
            ("弃电量/kWh", float(r["U"].sum()), float(np.asarray(ref42["U"]).sum())),
            ("期末库存/kWh", float(r["E_final"]),
             float(np.asarray(ref42["E_final"]).reshape(-1)[0])),
        ]
        p("  | 指标 | 4-3 | 4-2 | 4-3 − 4-2 |")
        p("  |---|---|---|---|")
        for name, v43, v42 in rows:
            p(f"  | {name} | {v43:,.2f} | {v42:,.2f} | {v43 - v42:+,.2f} |")
            cmp_rows.append((name, f"{v43:,.2f}", f"{v42:,.2f}", f"{v43 - v42:+,.2f}"))

    p("")
    p("── 6. 产物 ──")
    npz_path = C.BACKTEST_43_NPZ if not suf else \
        C.BACKTEST_43_NPZ.with_name(C.BACKTEST_43_NPZ.stem + f"_{suf}.npz")
    node_csv = C.RESULT_DIR / f"第四问_4-3节点诊断{suf}.csv"
    upd_csv = C.RESULT_DIR / f"第四问_4-3节点更新日志{suf}.csv"
    day_csv = C.RESULT_DIR / f"第四问_4-3逐日汇总{suf}.csv"

    np.savez_compressed(
        npz_path,
        branch=np.asarray(BRANCH), days=np.asarray(r["days"], int),
        dates=r["dates"], method=np.asarray(str(r["method"])),
        E_start=np.asarray(E_start), E_final=np.asarray(float(r["E_final"])),
        E_chain=E_chain,
        g=r["g"], a=r["a"], b=r["b"], C=r["C"], D=r["D"], U=r["U"],
        c=r["c"], r_net=r["r"], R=r["R"], N=N_ref,
        plan_cost=r["plan_cost"], adjust_cost=r["adjust_cost"],
        emerg_cost=r["emerg_cost"], bill=r["bill"],
        E_end=r["E_end"], exec_cnt=r["exec_cnt"].astype(np.int64),
        n_lp=np.asarray(int(r["n_lp"])),
        lp_status_codes=np.asarray(r["lp_status_codes"], int),
        wall_seconds=np.asarray(wall),
        nu_scale=np.asarray(float(r["config"]["nu_scale"])),
        soc_grid_kwh=np.asarray(float(r["config"]["delta"] or C.SOC_GRID_KWH)),
        node_H=np.asarray([int(n["H"]) for n in r["node_rows"]], int),
        node_H_full=np.asarray([int(n["H_full"]) for n in r["node_rows"]], int),
        node_value_origin=np.asarray([int(n["value_origin"]) for n in r["node_rows"]],
                                     int),
        node_tail=np.asarray([bool(n["tail"]) for n in r["node_rows"]], bool),
        horizon_note=np.asarray(str(r["horizon_note"])),
    )
    p(f"  全轨迹：{npz_path.relative_to(C.PROJECT_DIR)}"
      f"（{npz_path.stat().st_size / 1e6:.1f} MB）")

    C.write_csv_utf8_sig(
        upd_csv,
        ["date", "tau", "M", "fun_old", "fun_new", "accepted", "d_a_l1", "d_a_max",
         "iters_old", "iters_new", "converged_new"],
        [[a["date"], a["tau"], a["M"], f"{a['fun_old']:.6f}", f"{a['fun_new']:.6f}",
          int(bool(a["accepted"])), f"{a['d_a_l1']:.6f}", f"{a['d_a_max']:.6f}",
          a["iters_old"], a["iters_new"], int(bool(a["converged_new"]))]
         for a in acc],
    )
    p(f"  节点更新日志：{upd_csv.relative_to(C.PROJECT_DIR)}（{len(acc)} 行）")

    C.write_csv_utf8_sig(
        node_csv,
        ["date", "tau", "M", "H", "H_full", "value_origin", "tail", "nu", "E_in",
         "relax_gap", "fun", "fun_uncapped",
         "cap_iters", "cap_converged", "cap_pinned", "cap_viol", "deg_cd", "deg_bc",
         "sum_C", "sum_D", "fallback", "skip"],
        [[n["date"], n["tau"], n["M"], n["H"], n["H_full"], n["value_origin"],
          int(bool(n["tail"])),
          "" if n["nu"] is None or (isinstance(n["nu"], float) and np.isnan(n["nu"]))
          else f"{n['nu']:.8f}",
          f"{n['E_in']:.6f}",
          "" if n["relax_gap"] is None else f"{n['relax_gap']:.8f}",
          "" if n["fun"] is None or (isinstance(n["fun"], float) and np.isnan(n["fun"]))
          else f"{n['fun']:.6f}",
          "" if n["fun_uncapped"] is None
          or (isinstance(n["fun_uncapped"], float) and np.isnan(n["fun_uncapped"]))
          else f"{n['fun_uncapped']:.6f}",
          n["cap_iters"], int(bool(n["cap_converged"])), int(bool(n["cap_pinned"])),
          f"{n['cap_viol']:.3e}", f"{n['deg_cd']:.3e}", f"{n['deg_bc']:.3e}",
          "" if n.get("sum_C") is None else f"{n['sum_C']:.4f}",
          "" if n.get("sum_D") is None else f"{n['sum_D']:.4f}",
          n["fallback"], int(bool(n["skip"]))]
         for n in r["node_rows"]],
    )
    p(f"  节点诊断：{node_csv.relative_to(C.PROJECT_DIR)}"
      f"（{len(r['node_rows'])} 行）")

    rows = []
    for di in range(n_days):
        rows.append([
            str(r["dates"][di]), int(r["days"][di]),
            f"{r['plan_cost'][di]:.6f}", f"{r['adjust_cost'][di]:.6f}",
            f"{r['emerg_cost'][di]:.6f}", f"{r['bill'][di]:.6f}",
            f"{float(r['g'][di].sum()):.6f}", f"{float(r['a'][di].sum()):.6f}",
            f"{float(r['b'][di].sum()):.6f}", f"{float(r['C'][di].sum()):.6f}",
            f"{float(r['D'][di].sum()):.6f}", f"{float(r['U'][di].sum()):.6f}",
            f"{float(N_ref[di].sum()):.6f}",
            f"{float((c[di] * N_ref[di]).sum()):.6f}",
            f"{float(c[di].mean()):.6f}", f"{float(c[di].max()):.6f}",
            f"{E_chain[di]:.6f}", f"{E_chain[di + 1]:.6f}",
            int((r["b"][di] > 1e-9).sum()),
            f"{float(np.abs(r['a'][di] - r['g'][di]).max()):.6f}",
            f"{float(np.abs(r['a'][di] - r['g'][di]).sum()):.6f}",
            f"{float(np.minimum(r['C'][di], r['D'][di]).sum()):.3e}",
        ])
    C.write_csv_utf8_sig(
        day_csv,
        ["date", "day_index", "plan_cost", "adjust_cost", "emerg_cost", "bill",
         "g_kwh", "a_kwh", "b_kwh", "C_kwh", "D_kwh", "U_kwh", "N_kwh",
         "naive_cost", "price_mean", "price_max", "E_start_kwh", "E_end_kwh",
         "n_emerg_slots", "max_abs_a_minus_g", "sum_abs_a_minus_g", "sum_min_CD"],
        rows,
    )
    p(f"  逐日汇总：{day_csv.relative_to(C.PROJECT_DIR)}（{n_days} 行）")

    dates_arr = np.asarray(r["dates"]).astype(str)
    months = np.array([s[5:7] for s in dates_arr])
    mon_rows = []
    p("")
    p("  月度预览（正式月度/年度文件由 08 生成）：")
    p("  | 月份 | 天数 | 计划费/元 | 调整费/元 | 紧急费/元 | 合计/元 | 紧急电量/kWh | 日均价 |")
    p("  |---|---|---|---|---|---|---|---|")
    for mm in sorted(set(months.tolist())):
        msk = months == mm
        mon_rows.append((mm, int(msk.sum()), float(r["plan_cost"][msk].sum()),
                         float(r["adjust_cost"][msk].sum()),
                         float(r["emerg_cost"][msk].sum()),
                         float(r["bill"][msk].sum()),
                         float(r["b"][msk].sum()), float(c[msk].mean())))
        mm_, nd, pl, ad, em, tot, bk, pr = mon_rows[-1]
        p(f"  | 2025-{mm_} | {nd} | {pl:,.2f} | {ad:,.2f} | {em:,.2f} | {tot:,.2f} "
          f"| {bk:,.1f} | {pr:.4f} |")

    rp: list[str] = []
    rp.append("# 第四问 4-3 结果报告（波动价格 / 6、12、18 时节点更新）\n")
    rp.append("> 本报告由 `代码/07_运行第四问4-3.py` 自动生成；全部数值来自唯一闭环入口 "
              "`_settlement4.run_policy4(\"43\", …)`，未做二次实现或手工修数。\n")
    rp.append("## 1. 口径与输入\n")
    rp.append("| 项目 | 取值 |")
    rp.append("|---|---|")
    rp.append(f"| 样本窗口 | {prov.dates[days[0]]} … {prov.dates[days[-1]]}（{n_days} 天）|")
    rp.append(f"| 共同起点 $E(2025\\!-\\!02\\!-\\!01)$ | {E_start:,.4f} kWh（与 4-2 相同）|")
    rp.append(f"| 价格预测方法 | `{prov.method}`（03 定标）|")
    rp.append("| 采购更新 | $\\tau=6,12,18$ 时；0 时为原始计划 $g$，更新得到最终有效 $a$ |")
    rp.append("| 更新接受规则 | 同一目标函数下 $f_{\\rm new}\\le f_{\\rm old}+10^{-6}\\max(1,|f_{\\rm old}|)$ |")
    rp.append("| 结算 | B 口径相对 0 时计划：$\\sum c[a+0.5|a-g|]+5\\sum cb$ |")
    rp.append("| 规划层一致性上界 | $C_{u,\\omega}\\le\\max(0,a_u-N_{u,\\omega})$，迭代上限 4 轮 |")
    rp.append("")
    rp.append("> 已执行时段写入后**不再被后续节点改写**（每段恰好执行一次），"
              "因此 $\\tau>0$ 的调整只影响尚未执行的时段；$g$ 自 0 时起不再变化。\n")
    rp.append("## 2. 结果总览\n")
    rp.append(f"- 普通购电（原始计划）费用：**{r['cost_plan']:,.2f} 元**")
    rp.append(f"- 计划调整费用：**{r['cost_adjust']:,.2f} 元**"
              f"（下调 {float(down.sum()):,.2f} + 上调 {float(up.sum()):,.2f}）")
    rp.append(f"- 紧急购电费用：**{r['cost_emerg']:,.2f} 元**")
    rp.append(f"- **合计账单：{r['cost_total']:,.2f} 元**")
    rp.append(f"- 原始计划购电 $\\sum g$ = {float(g_v.sum()):,.1f} kWh；最终有效购电 "
              f"$\\sum a$ = {float(a_v.sum()):,.1f} kWh；紧急 {float(b_v.sum()):,.1f} kWh")
    rp.append(f"- 期末库存 $E$({prov.dates[days[-1]]} 24:00) = **{r['E_final']:,.2f} kWh**；"
              f"全期轨迹 [{e_lo:,.2f}, {e_hi:,.2f}] kWh")
    rp.append(f"- 求解：LP {r['n_lp']} 次，全部最优（{r['lp_status_codes']}）；总耗时 {wall:.2f} s")
    rp.append(f"- 高价缺电重合：价格 ≥ 90 分位（{q90:.4f} 元/kWh）的时段贡献了 "
              f"{100 * hi_share:.1f}% 的紧急电量")
    rp.append("")
    rp.append("## 3. 节点更新与同目标比价\n")
    rp.append(f"- 更新次数：{len(acc)}（{n_days} 天 × 6/12/18 三节点），接受 **{n_acc}** 次"
              f"（{100 * n_acc / max(len(acc), 1):.1f}%）")
    rp.append("")
    rp.append("| τ | 更新次数 | 接受 | 接受率 | $\\sum\\|\\Delta a\\|_1$ /kWh | $\\max\\|\\Delta a\\|$ /kWh |")
    rp.append("|---|---|---|---|---|---|")
    for tau in sorted(by_tau):
        s = by_tau[tau]
        rp.append(f"| {tau}:00 | {s['n']} | {s['acc']} | {100 * s['acc'] / s['n']:.1f}% | "
                  f"{s['l1']:,.1f} | {s['mx']:,.2f} |")
    rp.append("")
    rp.append("> 未接受即「候选不优于旧计划」，说明该节点新信息未能改进同口径目标函数；"
              "这不是失败，须与计划费/调整费/紧急费一并解释（§11.1）。\n")
    if cmp_rows:
        rp.append("## 4. 与 4-2 对照（同起点、同窗口、同结算口径）\n")
        rp.append("| 指标 | 4-3 | 4-2 | 差值 |")
        rp.append("|---|---|---|---|")
        for name, v43, v42, dv in cmp_rows:
            rp.append(f"| {name} | {v43} | {v42} | {dv} |")
        rp.append("")
        rp.append("> 期末库存不同时，账单差不能直接解释为「更优」：低末库存等于透支储能（§11.2）。\n")
        rp.append("## 5. 月度费用分解（预览，正式文件见 `08`）\n")
    else:
        rp.append("## 4. 月度费用分解（预览，正式文件见 `08`）\n")
    rp.append("| 月份 | 天数 | 计划费/元 | 调整费/元 | 紧急费/元 | 合计/元 | "
              "紧急电量/kWh | 日均价/(元/kWh) |")
    rp.append("|---|---|---|---|---|---|---|---|")
    for mm, nd, pl, ad, em, tot, bk, pr in mon_rows:
        rp.append(f"| 2025-{mm} | {nd} | {pl:,.2f} | {ad:,.2f} | {em:,.2f} | "
                  f"{tot:,.2f} | {bk:,.1f} | {pr:.4f} |")
    rp.append(f"| **合计** | **{n_days}** | **{r['cost_plan']:,.2f}** | "
              f"**{r['cost_adjust']:,.2f}** | **{r['cost_emerg']:,.2f}** | "
              f"**{r['cost_total']:,.2f}** | **{float(b_v.sum()):,.1f}** | — |")
    rp.append("")
    rp.append("## 6. 规划层诊断（一致性上界）\n")
    rp.append(f"- 相对缺口 $(f-f_0)/\\max(1,|f_0|)$：最大 {max(rg) * 100:.3f}%、"
              f"均值 {float(np.mean(rg)) * 100:.3f}%"
              f"（$f_0$ = **同节点去掉一致性上界**的 LP 目标，$f$ = 含上界后的目标；"
              f"**计数单位为节点**，非时段/检查项；该量**不是**全年费用误差，"
              f"也**不是**纯算法误差——它是约束收紧与迭代近似的混合量）")
    rp.append(f"- 收敛 {sum(1 for n in r['node_rows'] if n['cap_converged'])}"
              f"/{len(r['node_rows'])} 节点；钉住 $a$ 重解 {n_pin} 个节点"
              "（**保守可行兜底**：钉住后重解只保证**返回解自身**满足一致性上界，"
              "**不是**该节点自由 $a$ 下的最优解，其目标值与 $relax\\_gap$ 是该节点真实代价的**上界**）")
    rp.append(f"- 规划层 $\\sum\\min(C,D)$ 最大 {deg_cd:.3e} kWh、$\\sum\\min(b,C)$ 最大 "
              f"{deg_bc:.3e} kWh、最大越界 {cap_v:.3e} kWh")
    rp.append(f"- **规划层退化（已披露）**：{len(conv)} 个非钉住节点上"
              f" $\\sum\\min(C,D)$ 最大 {deg_cd_conv:.3e} kWh（严格互斥）；"
              f"{n_pin} 个钉住节点中 {len(pin_bad)} 个出现同时充/放，最大"
              f" {deg_cd:.3f} kWh，相对本节点充放电总量 ≤ {pin_ratio:.3e}")
    if pin_bad:
        rp.append("  - 退化节点：" + "、".join(
            f"{n['date']}（$\\tau$={n['tau']} h，$\\sum\\min(C,D)$={n['deg_cd']:.3f} kWh）"
            for n in pin_bad))
    rp.append("  - 说明：该层 $C/D$ 为逐情景**内部规划变量**，不进入价值函数"
              "（价值函数仅由 $a$ 构造），也不进入执行层（执行量由"
              " `execute_one_slot` 依真实库存重算），故对账单与库存轨迹无影响；"
              "相同最优解集合不要求退化 LP 返回逐点完全相同的动作。"
              "**执行层**互斥已在全部执行段上严格成立（见校验表）。")
    rp.append("")
    rp.append("## 7. 校验\n")
    rp.append("| 校验项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")

    rep = C.REPORT_DIR / f"第四问_4-3结果报告{suf}.md"
    C.write_text_utf8(rep, "\n".join(rp))
    p(f"  报告：{rep.relative_to(C.PROJECT_DIR)}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    p("")
    p("=" * 78)
    p(f"07 完成：4-3 全年 {n_days} 天闭环；校验 {len(checks)} 项，未通过 {n_fail} 项。"
      f"用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.LOG_DIR / f"07_4-3运行日志{suf}.txt", "\n".join(log) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
