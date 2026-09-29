#!/usr/bin/env python3
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


C = _load("_comm4.py", "q4_comm")

import numpy as np

T = C.PERIODS_PER_DAY
BRANCH_LABEL = {"42": "4-2（波动价格 / 全天一次计划）",
                "43": "4-3（波动价格 / 6、12、18 时节点更新）"}


def independent_bill(branch: str, c: np.ndarray, g: np.ndarray, a: np.ndarray,
                     b: np.ndarray) -> dict:
    g = np.asarray(g, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    up = np.maximum(g - a, 0.0)
    dn = np.maximum(a - g, 0.0)
    if branch == "43":
        plan = float((c * np.minimum(g, a)).sum())
        adjust = float((0.5 * c * up).sum() + (1.5 * c * dn).sum())
    else:
        plan = float((c * np.minimum(g, a)).sum())
        adjust = 0.0
    emerg = 5.0 * float((c * b).sum())
    return {"plan": plan, "adjust": adjust, "emerg": emerg,
            "total": plan + adjust + emerg}


def main(argv=None) -> int:
    _ = argv
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
    p("第四问 08 —— 结算复核与月度 / 年度汇总")
    p("=" * 78)

    p("")
    p("── 1. 输入 ──")
    data: dict[str, dict] = {}
    for br in ("42", "43"):
        path = C.BACKTEST_42_NPZ if br == "42" else C.BACKTEST_43_NPZ
        if not path.is_file():
            p(f"  缺少 {path.relative_to(C.PROJECT_DIR)}；请先运行 "
              f"{'06' if br == '42' else '07'}。")
            return 2
        with np.load(path, allow_pickle=False) as Zf:
            Z = {k: Zf[k] for k in Zf.files}
        data[br] = Z
        p(f"  {BRANCH_LABEL[br]}：{path.name}，{np.asarray(Z['days']).size} 天，"
          f"{np.asarray(Z['dates'])[0]} … {np.asarray(Z['dates'])[-1]}")

    d42, d43 = data["42"], data["43"]
    same_win = bool(np.array_equal(np.asarray(d42["days"]), np.asarray(d43["days"])))
    same_start = abs(float(np.asarray(d42["E_start"]).reshape(-1)[0])
                     - float(np.asarray(d43["E_start"]).reshape(-1)[0])) < 1e-9
    ck("两分支同起点、同窗口（对照前提）", same_win and same_start,
       f"天数 {np.asarray(d42['days']).size} = {np.asarray(d43['days']).size}；"
       f"起点 {float(np.asarray(d42['E_start']).reshape(-1)[0]):,.4f} kWh 相同")

    p("")
    p("── 2. 定义式独立复算（不调用主结算函数）──")
    recomputed: dict[str, dict] = {}
    for br, Z in (("42", d42), ("43", d43)):
        c = np.asarray(Z["c"], float)
        g = np.asarray(Z["g"], float)
        a = np.asarray(Z["a"], float)
        b = np.asarray(Z["b"], float)
        n_days = c.shape[0]
        bill = np.array([independent_bill(br, c[i], g[i], a[i], b[i])["total"]
                         for i in range(n_days)])
        engine = np.asarray(Z["bill"], float)
        err = float(np.abs(bill - engine).max())
        ck(f"{br} 逐日账单：定义式复算 = 引擎留档", err < 1e-6 * max(1.0, abs(float(engine.sum()))),
           f"最大绝对误差 {err:.3e} 元；合计 {float(engine.sum()):,.2f} 元")
        if br == "42":
            ck("4-2 $a\\equiv g$（调整费恒 0）",
               float(np.abs(a - g).max()) < 1e-9,
               f"max|a−g| = {float(np.abs(a - g).max()):.3e} kWh；调整费 "
               f"{float(np.asarray(Z['adjust_cost']).sum()):.2f} 元")
        recomputed[br] = {"bill": bill}

    p("")
    p("── 3. 月度费用分解 ──")
    mon_rows: list[list] = []
    mon_by_branch: dict[str, dict] = {}
    for br, Z in (("42", d42), ("43", d43)):
        c = np.asarray(Z["c"], float)
        g = np.asarray(Z["g"], float)
        a = np.asarray(Z["a"], float)
        b = np.asarray(Z["b"], float)
        Cc = np.asarray(Z["C"], float)
        D = np.asarray(Z["D"], float)
        U = np.asarray(Z["U"], float)
        N = np.asarray(Z["N"], float)
        Ee = np.asarray(Z["E_end"], float)
        dates = np.asarray(Z["dates"]).astype(str)
        months = np.array([s[5:7] for s in dates])
        agg = {}
        for mm in sorted(set(months.tolist())):
            m = months == mm
            bl = independent_bill(br, c[m], g[m], a[m], b[m])
            agg[mm] = {
                "n_days": int(m.sum()),
                "plan": bl["plan"], "adjust": bl["adjust"], "emerg": bl["emerg"],
                "total": bl["total"],
                "g": float(g[m].sum()), "a": float(a[m].sum()),
                "b": float(b[m].sum()), "C": float(Cc[m].sum()),
                "D": float(D[m].sum()), "U": float(U[m].sum()),
                "N": float(N[m].sum()),
                "E_end": float(Ee[m][-1]), "price_mean": float(c[m].mean()),
                "n_emerg_slots": int((b[m] > 1e-9).sum()),
                "d_ag_l1": float(np.abs(a[m] - g[m]).sum()),
            }
            agg[mm]["unit_cost"] = agg[mm]["total"] / max(agg[mm]["a"] + agg[mm]["b"], 1e-9)
            mon_rows.append([
                br, f"2025-{mm}", agg[mm]["n_days"],
                f"{agg[mm]['plan']:.6f}", f"{agg[mm]['adjust']:.6f}",
                f"{agg[mm]['emerg']:.6f}", f"{agg[mm]['total']:.6f}",
                f"{agg[mm]['g']:.6f}", f"{agg[mm]['a']:.6f}", f"{agg[mm]['b']:.6f}",
                f"{agg[mm]['C']:.6f}", f"{agg[mm]['D']:.6f}", f"{agg[mm]['U']:.6f}",
                f"{agg[mm]['N']:.6f}", f"{agg[mm]['E_end']:.6f}",
                f"{agg[mm]['price_mean']:.6f}", agg[mm]["n_emerg_slots"],
                f"{agg[mm]['d_ag_l1']:.6f}", f"{agg[mm]['unit_cost']:.6f}",
            ])
        mon_by_branch[br] = agg
        p(f"  {br}：{len(agg)} 个月已聚合")

    C.write_csv_utf8_sig(
        C.MONTHLY_CSV,
        ["branch", "month", "n_days", "plan_cost", "adjust_cost", "emerg_cost",
         "total_cost", "g_kwh", "a_kwh", "b_kwh", "C_kwh", "D_kwh", "U_kwh", "N_kwh",
         "E_end_kwh", "price_mean", "n_emerg_slots", "sum_abs_a_minus_g", "unit_cost"],
        mon_rows,
    )
    p(f"  月度分解：{C.MONTHLY_CSV.relative_to(C.PROJECT_DIR)}（{len(mon_rows)} 行）")

    p("")
    p("── 4. 年度汇总 ──")
    yr_rows: list[list] = []
    yr: dict[str, dict] = {}
    for br, Z in (("42", d42), ("43", d43)):
        c = np.asarray(Z["c"], float)
        g = np.asarray(Z["g"], float)
        a = np.asarray(Z["a"], float)
        b = np.asarray(Z["b"], float)
        Cc = np.asarray(Z["C"], float)
        D = np.asarray(Z["D"], float)
        U = np.asarray(Z["U"], float)
        N = np.asarray(Z["N"], float)
        bl = independent_bill(br, c, g, a, b)
        E_start = float(np.asarray(Z["E_start"]).reshape(-1)[0])
        E_end = float(np.asarray(Z["E_final"]).reshape(-1)[0])
        naive_pos = float((c * np.maximum(N, 0.0)).sum())
        naive_signed = float((c * N).sum())
        surplus = float((c * np.maximum(-N, 0.0)).sum())
        naive = naive_pos
        z_naive_kwh = float(np.maximum(N, 0.0).sum())
        yr[br] = {
            "n_days": int(c.shape[0]), "plan": bl["plan"], "adjust": bl["adjust"],
            "emerg": bl["emerg"], "total": bl["total"],
            "g": float(g.sum()), "a": float(a.sum()), "b": float(b.sum()),
            "C": float(Cc.sum()), "D": float(D.sum()), "U": float(U.sum()),
            "N": float(N.sum()), "E_start": E_start, "E_end": E_end,
            "price_mean": float(c.mean()),
            "n_emerg_slots": int((b > 1e-9).sum()),
            "d_ag_l1": float(np.abs(a - g).sum()),
            "naive": naive,
            "naive_signed": naive_signed, "surplus_c": surplus,
            "naive_kwh": z_naive_kwh,
            "cg": float((c * g).sum()),
            "rows_sum": float((c * g).sum() + (c * (a + 0.5 * np.abs(a - g))).sum()),
        }
        y = yr[br]
        y["unit_cost"] = y["total"] / max(y["a"] + y["b"], 1e-9)
        yr_rows.append([
            br, y["n_days"], f"{y['plan']:.6f}", f"{y['adjust']:.6f}",
            f"{y['emerg']:.6f}", f"{y['total']:.6f}",
            f"{y['g']:.6f}", f"{y['a']:.6f}", f"{y['b']:.6f}",
            f"{y['C']:.6f}", f"{y['D']:.6f}", f"{y['U']:.6f}", f"{y['N']:.6f}",
            f"{y['E_start']:.6f}", f"{y['E_end']:.6f}",
            f"{y['price_mean']:.6f}", y["n_emerg_slots"], f"{y['d_ag_l1']:.6f}",
            f"{y['unit_cost']:.6f}", f"{y['naive']:.6f}",
        ])
        p(f"  {br}：账单 {y['total']:,.2f} 元（计划 {y['plan']:,.2f} + 调整 "
          f"{y['adjust']:,.2f} + 紧急 {y['emerg']:,.2f}）；"
          f"单位成本 {y['unit_cost']:.4f} 元/kWh；末库存 {y['E_end']:,.2f} kWh")
        p(f"      无储能事后基准 Σc[N]⁺ = {y['naive']:,.2f} 元（正净负荷 "
          f"{y['naive_kwh']:,.1f} kWh）；旧写法 ΣcN = {y['naive_signed']:,.2f} 元，"
          f"差额 {y['surplus_c']:,.2f} 元 = 富余段 Σc[−N]⁺（不得计作售电收益）")
        ck(f"{br} 无储能事后基准按 Σc[N]⁺（不把负净负荷当售电收益）",
           abs((y["naive"] - y["naive_signed"]) - y["surplus_c"])
           <= 1e-9 * max(1.0, y["surplus_c"]) and y["surplus_c"] >= -1e-9,
           f"Σc[N]⁺ = {y['naive']:,.6f} 元；ΣcN = {y['naive_signed']:,.6f} 元；"
           f"Σc[−N]⁺ = {y['surplus_c']:,.6f} 元")

    C.write_csv_utf8_sig(
        C.YEARLY_CSV,
        ["branch", "n_days", "plan_cost", "adjust_cost", "emerg_cost", "total_cost",
         "g_kwh", "a_kwh", "b_kwh", "C_kwh", "D_kwh", "U_kwh", "N_kwh",
         "E_start_kwh", "E_end_kwh", "price_mean", "n_emerg_slots",
         "sum_abs_a_minus_g", "unit_cost", "naive_cost"],
        yr_rows,
    )
    p(f"  年度汇总：{C.YEARLY_CSV.relative_to(C.PROJECT_DIR)}（{len(yr_rows)} 行）")

    for br in ("42", "43"):
        z = data[br]
        s_day = float(np.asarray(z["bill"]).sum())
        s_mon = sum(v["total"] for v in mon_by_branch[br].values())
        s_yr = yr[br]["total"]
        ck(f"{br} 三级闭合（逐日 = 月度 = 年度 = 定义式复算）",
           max(abs(s_day - s_mon), abs(s_mon - s_yr),
               abs(s_yr - recomputed[br]["bill"].sum())) < 1e-6 * max(1.0, abs(s_yr)),
           f"{s_day:,.6f} = {s_mon:,.6f} = {s_yr:,.6f} 元")

    p("")
    p("── 5. 4-2 / 4-3 对照 ──")
    keys = [("账单", "total", "元"), ("计划费", "plan", "元"), ("调整费", "adjust", "元"),
            ("紧急费", "emerg", "元"), ("购电量 g", "g", "kWh"), ("有效购电 a", "a", "kWh"),
            ("紧急电量 b", "b", "kWh"), ("放电量", "D", "kWh"), ("充电量", "C", "kWh"),
            ("弃电量", "U", "kWh"), ("紧急时段数", "n_emerg_slots", "段"),
            ("Σ|a−g|", "d_ag_l1", "kWh"), ("末库存", "E_end", "kWh"),
            ("单位电量成本", "unit_cost", "元/kWh"), ("朴素基线", "naive", "元")]
    p("  | 指标 | 4-2 | 4-3 | 4-3 − 4-2 |")
    p("  |---|---|---|---|")
    cmp_table: list[tuple[str, str, str, str]] = []
    for label, key, unit in keys:
        v42, v43 = yr["42"][key], yr["43"][key]
        p(f"  | {label} /{unit} | {v42:,.2f} | {v43:,.2f} | {v43 - v42:+,.2f} |")
        cmp_table.append((f"{label} /{unit}", f"{v42:,.2f}", f"{v43:,.2f}",
                          f"{v43 - v42:+,.2f}"))
    dE = yr["43"]["E_end"] - yr["42"]["E_end"]
    ck("对照已披露库存差异带来的不可比性",
       True,
       f"两分支末库存相差 {dE:+,.2f} kWh（4-2 {yr['42']['E_end']:,.2f} kWh、"
       f"4-3 {yr['43']['E_end']:,.2f} kWh）。B 口径下终端库存**不进入账单**，"
       "故两分支账单不可直接比较优劣；本报告只给出账单与末库存两项可核对事实，"
       "不再输出任何「库存归一化收益」指标")

    for br in ("42", "43"):
        y = yr[br]
        rhs = (y["plan"] + y["adjust"]) + y["cg"]
        ck(f"{br} 附件五两行费用之和 = 账单购电部分 + Σc·g（口径交叉恒等式）",
           abs(y["rows_sum"] - rhs) < 1e-6 * max(1.0, abs(rhs)),
           f"两行和 {y['rows_sum']:,.6f} 元；账单购电部分 {y['plan'] + y['adjust']:,.6f} 元 "
           f"+ Σc·g {y['cg']:,.6f} 元 = {rhs:,.6f} 元（差 {y['rows_sum'] - rhs:.3e}）")

    rp: list[str] = []
    infl43 = float((np.asarray(d43["c"], float)
                    * (np.asarray(d43["g"], float)
                       - np.asarray(d43["a"], float))).sum())
    rp.append("# 第四问 结算复核与月度/年度汇总\n")
    rp.append("> 由 `代码/08_结算与年度汇总.py` 自动生成；全部数字来自 `06`/`07` 落盘的"
              "全轨迹，费用用**定义式独立复算**（不调用主结算函数）。\n")
    rp.append("## 1. 口径\n")
    rp.append("- 4-2：$a\\equiv g$，故 $\\min(g,a)=g$，调整费恒 0；"
              "账单 $=\\sum_t c_t\\min(g_t,a_t)+5\\sum_t c_tb_t=\\sum_t c_tg_t+5\\sum_t c_tb_t$。")
    rp.append("- 4-3：$\\text{计划}=\\sum_t c_t\\min(g_t,a_t)$（**交付基准量**；被下调掉的"
              "部分不进计划费），账单 $=\\sum_t c_t\\min(g_t,a_t)"
              "+\\left[0.5\\sum_t c_t(g_t-a_t)^++1.5\\sum_t c_t(a_t-g_t)^+\\right]"
              "+5\\sum_t c_tb_t$，等价于 $\\sum_t c_t[a_t+0.5|a_t-g_t|]+5\\sum_t c_tb_t$")
    rp.append("  （两式**数值相同**、**成分不同**：前式计划费配平为 $\\sum c\\min(g,a)$，"
              "后式把下调电量按 $a$ 全额计入计划费；不得把 $\\sum c\\min(g,a)$ 写成 "
              "$\\sum cg$，也不得混用两式的成分 —— 若按 $\\sum c_tg_t+0.5\\sum_t c_t|a_t-g_t|"
              "+5\\sum_t c_tb_t$ 计算，会把下调电量按半价重复计入，使 4-3 账单虚高 "
              f"$\\sum c(g-a)=${infl43:,.6f} 元/年（本脚本按留档 $c,g,a$ 直接复算，可逐位核对；"
              "不得再写成「约 89 万元」）。")
    rp.append(f"- 系数来源：$\\rho_{{\\text{{down}}}}=${C.RHO_DOWN:g}、"
              f"$\\rho_{{\\text{{up}}}}=${C.RHO_UP:g}、紧急倍数 ${C.EMERG_MULT:g}$，"
              "集中定义于 `代码/_comm4.py`。")
    rp.append("- 量单位：电量 kWh，费用 元；价格为元/kWh，**不乘** $\\Delta t=1/6$ h。")
    rp.append(f"- 窗口：{np.asarray(d42['dates'])[0]} … {np.asarray(d42['dates'])[-1]}"
              f"（{np.asarray(d42['days']).size} 天），两分支同起点 "
              f"$E(2/1)={float(np.asarray(d42['E_start']).reshape(-1)[0]):,.2f}$ kWh。")
    rp.append("- 1 月暖启动期费用单列，不计入上述年度合计（见 `02` 报告）。\n")
    rp.append("## 2. 年度汇总\n")
    rp.append("| 分支 | 天数 | 计划费/元 | 调整费/元 | 紧急费/元 | **合计/元** | "
              "购电 a/kWh | 紧急 b/kWh | 放电/kWh | 充电/kWh | 弃电/kWh | 末库存/kWh | "
              "单位成本/(元/kWh) |")
    rp.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for br in ("42", "43"):
        y = yr[br]
        rp.append(f"| {br} | {y['n_days']} | {y['plan']:,.2f} | {y['adjust']:,.2f} | "
                  f"{y['emerg']:,.2f} | **{y['total']:,.2f}** | {y['a']:,.1f} | "
                  f"{y['b']:,.1f} | {y['D']:,.1f} | {y['C']:,.1f} | {y['U']:,.1f} | "
                  f"{y['E_end']:,.1f} | {y['unit_cost']:.4f} |")
    rp.append("")
    rp.append(
        f"- 无储能**事后**基准（完全信息、非可执行策略）$\\sum_t c_t[N_t]^{{+}}$："
        f"4-2 {yr['42']['naive']:,.2f} 元、4-3 {yr['43']['naive']:,.2f} 元；"
        f"对应正净负荷电量 4-2 {yr['42']['naive_kwh']:,.1f} kWh、"
        f"4-3 {yr['43']['naive_kwh']:,.1f} kWh。"
        f"（旧写法 $\\sum_t c_tN_t$ 会把富余段 $c_t(-N_t)$ 当成售电收益："
        f"4-2 {yr['42']['naive_signed']:,.2f} 元、4-3 {yr['43']['naive_signed']:,.2f} 元，"
        f"分别低估 {yr['42']['surplus_c']:,.2f} / {yr['43']['surplus_c']:,.2f} 元，已废弃。）")
    rp.append("")
    rp.append("## 3. 4-2 / 4-3 对照\n")
    rp.append("| 指标 | 4-2 | 4-3 | 4-3 − 4-2 |")
    rp.append("|---|---|---|---|")
    for label, v42, v43, dv in cmp_table:
        rp.append(f"| {label} | {v42} | {v43} | {dv} |")
    rp.append("")
    rp.append(f"> **可比性提示**：两分支末库存相差 {dE:+,.2f} kWh（4-2 "
              f"{yr['42']['E_end']:,.2f} kWh、4-3 {yr['43']['E_end']:,.2f} kWh）。B 口径下"
              "终端库存**不进入账单**，低末库存相当于透支储能，因此账单差不能直接"
              "解释为「更优」，也不应换算为任何「归一化收益」。策略差异的可解释对比见 "
              "`09` 的消融与 `10` 的指定日期明细；口径局限见 "
              "`报告/第四问_口径差异与局限性说明.md` §6。\n")
    for br in ("42", "43"):
        rp.append(f"## 4.{'1' if br == '42' else '2'} {br} 月度费用分解\n")
        rp.append("| 月份 | 天数 | 计划费/元 | 调整费/元 | 紧急费/元 | 合计/元 | "
                  "购电 a/kWh | 紧急 b/kWh | 末库存/kWh | 日均价/(元/kWh) | "
                  "单位成本/(元/kWh) | 紧急时段数 |")
        rp.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for mm in sorted(mon_by_branch[br]):
            g_ = mon_by_branch[br][mm]
            rp.append(f"| 2025-{mm} | {g_['n_days']} | {g_['plan']:,.2f} | "
                      f"{g_['adjust']:,.2f} | {g_['emerg']:,.2f} | {g_['total']:,.2f} | "
                      f"{g_['a']:,.1f} | {g_['b']:,.1f} | {g_['E_end']:,.1f} | "
                      f"{g_['price_mean']:.4f} | {g_['unit_cost']:.4f} | "
                      f"{g_['n_emerg_slots']} |")
        y = yr[br]
        rp.append(f"| **合计** | **{y['n_days']}** | **{y['plan']:,.2f}** | "
                  f"**{y['adjust']:,.2f}** | **{y['emerg']:,.2f}** | **{y['total']:,.2f}** | "
                  f"**{y['a']:,.1f}** | **{y['b']:,.1f}** | **{y['E_end']:,.1f}** | "
                  f"{y['price_mean']:.4f} | {y['unit_cost']:.4f} | "
                  f"**{y['n_emerg_slots']}** |")
        rp.append("")
    rp.append("## 5. 校验\n")
    rp.append("| 校验项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")

    rep = C.REPORT_DIR / "第四问_结算与年度汇总报告.md"
    C.write_text_utf8(rep, "\n".join(rp))
    p(f"  报告：{rep.relative_to(C.PROJECT_DIR)}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    p("")
    p("=" * 78)
    p(f"08 完成：月度 {len(mon_rows)} 行、年度 {len(yr_rows)} 行；校验 {len(checks)} 项，"
      f"未通过 {n_fail} 项。用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.LOG_DIR / "08_结算汇总日志.txt", "\n".join(log) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
