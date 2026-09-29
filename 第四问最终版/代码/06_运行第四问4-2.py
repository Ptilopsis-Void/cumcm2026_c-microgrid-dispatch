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
BRANCH = "42"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="第四问 4-2 全年回测")
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
    p("第四问 06 —— 4-2 全年回测（固定价格 / 一天一次计划）")
    p("=" * 78)

    Zw = np.load(C.WARMUP_NPZ, allow_pickle=False)
    E_start = float(np.asarray(Zw["E_feb1"]).reshape(-1)[0])
    E_init_declared = float(np.asarray(Zw["E_init"]).reshape(-1)[0])
    n_days = int(args.days)
    if not 1 <= n_days <= C.N_SCORE_DAYS:
        p(f"  --days 必须在 1..{C.N_SCORE_DAYS} 内")
        return 2
    days = np.arange(C.N_WARMUP_DAYS, C.N_WARMUP_DAYS + n_days)
    prov = ST.SnapshotProviders(BRANCH)
    p("")
    p("── 1. 口径与输入 ──")
    p(f"  共同起点（02 热启动终点）：E(2025-02-01) = {E_start:,.4f} kWh"
      f"（题面声明初始 {E_init_declared:,.4f} kWh，已在 1 月按同一套口径演化）")
    p(f"  样本窗口：{prov.dates[days[0]]} … {prov.dates[days[-1]]}（{n_days} 天，"
      f"日索引 {days[0]}–{days[-1]}）")
    p(f"  价格预测方法：{prov.method}；采购更新时刻："
      f"{list(C.CONFIG['purchase_update_hours_42']) or '无（全天一次计划）'}")
    ck("共同起点来自第二问口径热启动（非题面初值直接用）",
       abs(E_start - E_init_declared) > 1e-9,
       f"E(2/1) = {E_start:,.4f} ≠ 声明初值 {E_init_declared:,.4f}（1 月热启动后到达）")

    p("")
    p("── 2. 闭环运行 ──")
    tm = time.perf_counter()
    r = ST.run_policy4(BRANCH, prov, prov, initial_state=E_start, days=days)
    wall = time.perf_counter() - tm
    p(f"  用时 {wall:.2f} s；LP {r['n_lp']} 次（{wall / max(r['n_lp'], 1) * 1000:.0f} ms/次）；"
      f"{n_days} 天 × 4 节点 = {n_days * 4} 个节点")
    p(f"  账单合计 {r['cost_total']:,.2f} 元 = 计划 {r['cost_plan']:,.2f}"
      f" + 调整 {r['cost_adjust']:,.2f} + 紧急 {r['cost_emerg']:,.2f}")
    p(f"  期末库存 E({prov.dates[days[-1]]} 24:00) = {r['E_final']:,.2f} kWh")

    p("")
    p("── 3. 校验 ──")
    N_ref = np.asarray(prov.net_act[r["days"]], float)

    d_ag = float(np.abs(r["a"] - r["g"]).max())
    ck("A05 4-2 全时段 a ≡ g（不借规划加购电）", d_ag <= 1e-9,
       f"max|a−g| = {d_ag:.3e} kWh；调整费 {r['cost_adjust']:,.6f} 元")

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
       f"{sorted({int(n['H']) for n in r['node_rows']})}（库内 144 → 18 时节点收窄到 36）；"
       f"越界 {len(hor_bad)} 项、原点异常 {len(vo_bad)} 项；跨日 tail 节点 {n_tail} 个"
       + (f"；示例 {hor_bad[:2]}" if hor_bad else ""))

    bal = float(np.abs(r["a"] + r["b"] + r["D"] - r["C"] - r["U"] - N_ref).max())
    ck("A07 逐时段能量平衡 a+b+D−C−U = N", bal < 1e-6,
       f"48096 段最大绝对残差 {bal:.3e} kWh")

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
       f"状态码集合 {r['lp_status_codes']}；无 fallback 节点 {n_days * 4} 个 = 全部")
    n_exec_bad = int((r["exec_cnt"] != 1).sum())
    ck("每日 144 段各被恰好执行一次", bool(n_exec_bad == 0 and r["n_unexec"] == 0),
       f"异常段数 {n_exec_bad}；未执行段数 {r['n_unexec']}")

    bill_err = float(np.abs(r["bill"] - (r["plan_cost"] + r["adjust_cost"]
                                        + r["emerg_cost"])).max())
    ck("A09 逐日账单 = 计划 + 调整 + 紧急", bill_err < 1e-6,
       f"最大误差 {bill_err:.3e} 元")
    ck("A09 年度合计 = 逐日求和", abs(float(r["bill"].sum()) - r["cost_total"]) < 1e-6,
       f"{r['cost_total']:,.6f} 元")

    bill_def = (float((r["c"] * np.minimum(r["g"], r["a"])).sum())
                + float((C.RHO_DOWN * r["c"] * np.maximum(r["g"] - r["a"], 0)).sum())
                + float((C.RHO_UP * r["c"] * np.maximum(r["a"] - r["g"], 0)).sum())
                + float((C.EMERG_MULT * r["c"] * r["b"]).sum()))
    ck("A14 终端续存价值 −νE 未计入实际账单",
       abs(r["cost_total"] - bill_def) < 1e-6,
       "账单 = Σc·min(g,a) + Σ0.5c(g−a)⁺ + Σ1.5c(a−g)⁺ + Σ5c·b（独立公式复算；"
       "本分支 a≡g，故调整费两项恒 0）")

    naive = float((r["c"] * np.maximum(N_ref, 0.0)).sum())
    naive_signed = float((r["c"] * N_ref).sum())
    ck("量级 sanity：账单不劣于无储能事后基准（Σ c_t·[N_t]⁺）",
       r["cost_total"] <= naive * 1.0 + 1e-6,
       f"4-2 {r['cost_total']:,.2f} 元 vs 基准 {naive:,.2f} 元"
       f"（较基准少 {naive - r['cost_total']:,.2f} 元，"
       f"{100 * (naive - r['cost_total']) / naive:.2f}%，事后口径不作可执行性比较；"
       f"旧写法 Σc·N = "
       f"{naive_signed:,.2f} 元，差 {naive - naive_signed:,.2f} 元 = 富余段被当成售电收益，已废弃）")

    p("")
    p("── 4. 产物 ──")
    suf = str(args.suffix or "").strip()
    npz_path = C.BACKTEST_42_NPZ if not suf else \
        C.BACKTEST_42_NPZ.with_name(C.BACKTEST_42_NPZ.stem + f"_{suf}.npz")
    node_csv = C.RESULT_DIR / f"第四问_4-2节点诊断{suf}.csv"
    day_csv = C.RESULT_DIR / f"第四问_4-2逐日汇总{suf}.csv"

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
      f"（{len(r['node_rows'])} 行 = {n_days} 天 × 4 节点）")

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
            f"{float((r['c'][di] * N_ref[di]).sum()):.6f}",
            f"{float(r['c'][di].mean()):.6f}", f"{float(r['c'][di].max()):.6f}",
            f"{E_chain[di]:.6f}", f"{E_chain[di + 1]:.6f}",
            int((r["b"][di] > 1e-9).sum()),
            f"{float(np.abs(r['a'][di] - r['g'][di]).max()):.3e}",
            f"{float(np.minimum(r['C'][di], r['D'][di]).sum()):.3e}",
        ])
    C.write_csv_utf8_sig(
        day_csv,
        ["date", "day_index", "plan_cost", "adjust_cost", "emerg_cost", "bill",
         "g_kwh", "a_kwh", "b_kwh", "C_kwh", "D_kwh", "U_kwh", "N_kwh",
         "naive_cost", "price_mean", "price_max", "E_start_kwh", "E_end_kwh",
         "n_emerg_slots", "max_abs_a_minus_g", "sum_min_CD"],
        rows,
    )
    p(f"  逐日汇总：{day_csv.relative_to(C.PROJECT_DIR)}（{n_days} 行）")

    dates_arr = np.asarray(r["dates"]).astype(str)
    months = np.array([s[5:7] for s in dates_arr])
    p("")
    p("  月度预览（正式月度/年度文件由 08 生成）：")
    p("  | 月份 | 天数 | 计划费/元 | 调整费/元 | 紧急费/元 | 合计/元 | 紧急电量/kWh | 日均价 |")
    p("  |---|---|---|---|---|---|---|---|")
    mon_rows = []
    for mm in sorted(set(months.tolist())):
        msk = months == mm
        mon_rows.append((mm, int(msk.sum()), float(r["plan_cost"][msk].sum()),
                         float(r["adjust_cost"][msk].sum()),
                         float(r["emerg_cost"][msk].sum()),
                         float(r["bill"][msk].sum()),
                         float(r["b"][msk].sum()),
                         float(r["c"][msk].mean())))
        mm_, nd, pl, ad, em, tot, bk, pr = mon_rows[-1]
        p(f"  | 2025-{mm_} | {nd} | {pl:,.2f} | {ad:,.2f} | {em:,.2f} | {tot:,.2f} "
          f"| {bk:,.1f} | {pr:.4f} |")

    rp: list[str] = []
    rp.append("# 第四问 4-2 结果报告（固定价格 / 一天一次计划）\n")
    rp.append("> 本报告由 `代码/06_运行第四问4-2.py` 自动生成；全部数值来自唯一闭环入口 "
              "`_settlement4.run_policy4(\"42\", …)`，未做二次实现或手工修数。\n")
    rp.append("## 1. 口径与输入\n")
    rp.append("| 项目 | 取值 |")
    rp.append("|---|---|")
    rp.append(f"| 样本窗口 | {prov.dates[days[0]]} … {prov.dates[days[-1]]}（{n_days} 天）|")
    rp.append(f"| 共同起点 $E(2025\\!-\\!02\\!-\\!01)$ | {E_start:,.4f} kWh（02 热启动终点）|")
    rp.append(f"| 价格预测方法 | `{prov.method}`（03 定标，评分期 MAE 见 03 报告）|")
    rp.append("| 采购更新 | 无，全天一次计划，故 $a\\equiv g$ |")
    rp.append(f"| 情景数 | 每节点按合法来源取 $M\\le${C.M_SCENARIOS}，"
              f"实际均值 {float(np.mean([n['M'] for n in r['node_rows']])):.2f} |")
    rp.append(f"| 价值网格 | $\\delta$ = {C.SOC_GRID_KWH:g} kWh |")
    rp.append("| 结算 | 分项口径：$\\sum c\\min(g,a)+\\sum 0.5c(g-a)^+"
              "+\\sum 1.5c(a-g)^++\\sum 5cb$（按附件四实际价）；本分支 $a\\equiv g$，"
              "故调整费两项恒 $0$，退化为 $\\sum cg+\\sum 5cb$ |")
    rp.append("| 规划层一致性上界 | $C_{u,\\omega}\\le\\max(0,a_u-N_{u,\\omega})$，"
              "不动点迭代上限 4 轮 |")
    rp.append("")
    rp.append("## 2. 结果总览\n")
    rp.append(f"- 评分期普通购电（计划）费用：**{r['cost_plan']:,.2f} 元**")
    rp.append(f"- 调整费用：**{r['cost_adjust']:,.6f} 元**（4-2 恒为 0，$a\\equiv g$）")
    rp.append(f"- 紧急购电费用：**{r['cost_emerg']:,.2f} 元**")
    rp.append(f"- **合计账单：{r['cost_total']:,.2f} 元**")
    rp.append(f"- **无储能事后基准（非可执行策略）**：{naive:,.2f} 元 → "
              f"差额 {naive - r['cost_total']:,.2f} 元"
              f"（{100 * (naive - r['cost_total']) / naive:.2f}%）")
    rp.append("  - 信息口径：该基准 $\\sum_t c_t[N_t]^+$ 使用**事后已知**的净负荷 "
              "$N_t$（含实际光伏），完全信息、不可在线执行，只用于**量级 sanity**；"
              "又因合同只允许购电，富余段按 0 计（**不把负净负荷当售电收益**）。"
              "旧写法 $\\sum c_t N_t$ 会把富余段算成售电收益，"
              f"较本口径低 {naive - naive_signed:,.2f} 元，已废弃。")
    rp.append(f"- 普通购电合计 {float(r['g'].sum()):,.1f} kWh；紧急购电合计 "
              f"{float(r['b'].sum()):,.1f} kWh；放电 {float(r['D'].sum()):,.1f} kWh；"
              f"充电 {float(r['C'].sum()):,.1f} kWh；弃电 {float(r['U'].sum()):,.1f} kWh")
    rp.append(f"- 期末库存 $E$({prov.dates[days[-1]]} 24:00) = "
              f"**{r['E_final']:,.2f} kWh**；全期轨迹范围 "
              f"[{e_lo:,.2f}, {e_hi:,.2f}] kWh")
    rp.append(f"- 求解：LP {r['n_lp']} 次，全部最优（状态码 "
              f"{r['lp_status_codes']}）；总耗时 {wall:.2f} s")
    rp.append("")
    rp.append("> **期末库存须与 4-3 对照**：低末库存等于透支储能换来的低账单，"
              "不能单独解释为「更省」（§11.2）。\n")
    rp.append("## 3. 月度费用分解（预览，正式文件见 `08`）\n")
    rp.append("| 月份 | 天数 | 计划费/元 | 调整费/元 | 紧急费/元 | 合计/元 | "
              "紧急电量/kWh | 日均价/(元/kWh) |")
    rp.append("|---|---|---|---|---|---|---|---|")
    for mm, nd, pl, ad, em, tot, bk, pr in mon_rows:
        rp.append(f"| 2025-{mm} | {nd} | {pl:,.2f} | {ad:,.2f} | {em:,.2f} | "
                  f"{tot:,.2f} | {bk:,.1f} | {pr:.4f} |")
    rp.append(f"| **合计** | **{n_days}** | **{r['cost_plan']:,.2f}** | "
              f"**{r['cost_adjust']:,.6f}** | **{r['cost_emerg']:,.2f}** | "
              f"**{r['cost_total']:,.2f}** | **{float(r['b'].sum()):,.1f}** | — |")
    rp.append("")
    rp.append("## 4. 规划层诊断（一致性上界）\n")
    rp.append(f"- 一致性上界的相对缺口 $\\Delta=(f-f_0)/\\max(1,|f_0|)$："
              f"最大 {max(rg) * 100:.3f}%、均值 {float(np.mean(rg)) * 100:.3f}%"
              "（$f_0$ = **同节点去掉上界**的 LP 目标值，$f$ = 加界后目标值；"
              "**计数单位为节点**，非时段/检查项；该量**不是**全年费用误差，"
              "也**不是**纯算法误差，而是约束收紧与迭代近似的混合量）")
    rp.append(f"- 不动点迭代收敛 {sum(1 for n in r['node_rows'] if n['cap_converged'])}"
              f"/{len(r['node_rows'])} 节点；其余 {n_pin} 个节点采用「钉住 $a$ 后重解」"
              "以保证**返回解自身**满足上界（**保守可行兜底**，"
              "**不是**该节点自由 $a$ 下的最优解，其目标值是真实代价的**上界**）")
    rp.append(f"- 上界生效后：规划层 $\\sum\\min(C,D)$ 最大 {deg_cd:.3e} kWh、"
              f"$\\sum\\min(b,C)$ 最大 {deg_bc:.3e} kWh，最大越界 {cap_v:.3e} kWh")
    rp.append(f"- **规划层退化（已披露）**：{len(conv)} 个非钉住节点上"
              f" $\\sum\\min(C,D)$ 最大 {deg_cd_conv:.3e} kWh（严格互斥）；"
              f"{n_pin} 个钉住节点中 {len(pin_bad)} 个出现同时充/放，最大"
              f" {deg_cd:.3f} kWh，相对本节点充放电总量 ≤ {pin_ratio:.3e}")
    if pin_bad:
        rp.append("  - 退化节点：" + "、".join(
            f"{n['date']}（$\\tau$={n['tau']} h，$\\sum\\min(C,D)$={n['deg_cd']:.3f} kWh）"
            for n in pin_bad))
    rp.append("  - 说明：该层 $C/D$ 为逐情景**内部规划变量**，既不进入价值函数"
              "（价值函数仅由 $a$ 构造），也不进入执行层（执行量由"
              " `execute_one_slot` 依真实库存重算），故对账单与库存轨迹无影响；"
              "按赛题口径，相同最优解集合不要求退化 LP 返回逐点完全相同的动作。"
              "**执行层**互斥已在全部 48096 段严格成立（见上表 A08 执行层检查）。")
    rp.append("")
    rp.append("## 5. 校验\n")
    rp.append("| 校验项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")

    rep = C.REPORT_DIR / (f"第四问_4-2结果报告{suf}.md")
    C.write_text_utf8(rep, "\n".join(rp))
    p(f"  报告：{rep.relative_to(C.PROJECT_DIR)}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    p("")
    p("=" * 78)
    p(f"06 完成：4-2 全年 {n_days} 天闭环；校验 {len(checks)} 项，未通过 {n_fail} 项。"
      f"用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.LOG_DIR / f"06_4-2运行日志{suf}.txt", "\n".join(log) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
