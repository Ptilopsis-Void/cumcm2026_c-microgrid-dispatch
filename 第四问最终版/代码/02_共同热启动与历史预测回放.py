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
PO = _load("_policy4.py", "q4_policy")

import numpy as np

T = C.PERIODS_PER_DAY
S = C.S_PERIOD_KWH
ETA = C.ETA
E_MIN, E_MAX = C.E_MIN, C.E_MAX
E_INIT = C.E_INIT
WARMUP_DAYS = C.N_WARMUP_DAYS
SCORE_START_IDX = WARMUP_DAYS
WARMUP_CSV = C.RESULT_DIR / "第四问_热启动逐日.csv"

ZERO_LEDGER_KEYS = ("g", "a", "b", "C", "D", "U")


def fallback_day(N_day: np.ndarray, price_day: np.ndarray, E0: float) -> dict:
    H = N_day.size
    led = {k: np.zeros(H) for k in ZERO_LEDGER_KEYS}
    E = np.empty(H + 1)
    E[0] = E0
    for t in range(H):
        r = float(N_day[t])
        if r > 0.0:
            D = max(0.0, min(r, S, ETA * (E[t] - E_MIN)))
            b = r - D
            Cc = U = 0.0
        else:
            Cc = max(0.0, min(-r, S, (E_MAX - E[t]) / ETA))
            U = max(0.0, -r - Cc)
            D = b = 0.0
        led["D"][t], led["b"][t], led["C"][t], led["U"][t] = D, b, Cc, U
        E[t + 1] = E[t] + ETA * Cc - D / ETA
    led["E"] = E
    led["price"] = price_day
    led["N"] = N_day
    led["cost_normal"] = 0.0
    led["cost_emerg"] = float((5.0 * price_day * led["b"]).sum())
    led["nu"] = 0.0
    led["g"] = np.zeros(H)
    led["a"] = np.zeros(H)
    return led


def simple_policy_day(N_day: np.ndarray, price_day: np.ndarray, E0: float,
                      N_prev: np.ndarray, price_prev: np.ndarray,
                      nu: float) -> dict:
    H = N_day.size
    plan = PO.solve_midnight_plan(N_prev[None, :], price_prev[None, :],
                                  np.array([1.0]), E0, nu)
    g = np.array(plan["a"], float)
    vf = PO.build_value_functions(N_prev[None, :], g[None, :], price_prev[None, :], nu)
    Hbar, grid = vf["Hbar"], vf["grid"]

    led = {k: np.zeros(H) for k in ZERO_LEDGER_KEYS}
    E = np.empty(H + 1)
    E[0] = E0
    led["g"] = g
    led["a"] = g.copy()
    R = np.full(H, np.nan)
    for t in range(H):
        out = PO.execute_one_slot(E[t], g[t], float(price_day[t]), float(N_day[t]),
                                  Hbar[t + 1], grid)
        led["C"][t] = out["C"]
        led["D"][t] = out["D"]
        led["b"][t] = out["b"]
        led["U"][t] = out["U"]
        R[t] = out["R"]
        E[t + 1] = out["E_next"]
    led["E"] = E
    led["R"] = R
    led["price"] = price_day
    led["N"] = N_day
    led["cost_normal"] = float((price_day * led["g"]).sum())
    led["cost_emerg"] = float((5.0 * price_day * led["b"]).sum())
    led["nu"] = float(nu)
    led["lp_status"] = int(plan["status"])
    return led


def main() -> int:
    C.ensure_dirs()
    t0 = time.perf_counter()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 78)
    p("第四问 02 —— 1 月共同因果热启动 + 价格候选预测历史回放")
    p("=" * 78)

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail):
        checks.append((name, bool(ok), detail))
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    p("")
    p("── 0. 读入附件四电价、附件二实际负荷/光伏（只读）──")
    Z4 = np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False)
    price_act = np.asarray(Z4["price_actual"], float)
    dates = np.array([str(s) for s in Z4["dates"]], dtype="<U10")
    Z = C.load_q2_matrix()
    L_act = np.asarray(Z["load_energy_kwh"], float)
    V_act = np.asarray(Z["pv_energy_kwh"], float)
    N_act = np.asarray(Z["net_load_energy_kwh"], float)
    dates2 = np.array([str(s) for s in Z["dates"]], dtype="<U10")
    p(f"  price_actual {price_act.shape}；N_act {N_act.shape}；"
      f"日期 {dates[0]} … {dates[-1]}（{dates.size} 天）")
    ck("附件四与附件二日期一一对应", bool(np.array_equal(dates, dates2)),
       f"{dates.size} 天逐日一致")
    ck("净负荷 = 负荷 − 光伏（能量口径自洽）",
       float(np.abs(N_act - (L_act - V_act)).max()) < 1e-6,
       f"最大偏差 {np.abs(N_act - (L_act - V_act)).max():.3e} kWh")
    ck("全表价格有限且为正",
       bool(np.isfinite(price_act).all() and (price_act > 0).all()),
       f"最小值 {price_act.min():.4f} 元/kWh")

    import datetime as _dt
    L_day = L_act.sum(axis=1)
    med = float(np.median(L_day))
    ratio = L_day / med
    wd = np.array([_dt.date.fromisoformat(s).strftime("%a") for s in dates])
    low_idx = np.where(ratio < 0.8)[0]
    low_wd = {w: int(((wd[low_idx] == w)).sum()) for w in
              ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")}
    p(f"  日负荷中位数 {med:,.0f} kWh；低于中位数 80% 的天数 {low_idx.size}，"
      f"星期分布 {low_wd}")
    p("  → 附件二存在**周五/周六低负荷模式**（约为中位数的 0.61–0.69），"
      "周一至周四午后无此模式。")
    p("    这使「前一日同刻持续性」预测在 周四→周五 与 周六→周日 两个方向上系统偏错，"
      "是 1 月公共简单策略误差的主要来源（1 月费用单列，不影响评分）。")
    ck("附件二存在周五/周六低负荷模式（已独立核验）",
       low_wd["Fri"] + low_wd["Sat"] == low_idx.size and low_idx.size > 0,
       f"{low_idx.size} 个低负荷日全部落在周五/周六；比率中位 {np.median(ratio):.4f}")

    p("")
    p("── 1. 1 月共同因果热启动（31 天，E₀ = 6000 kWh）──")
    p("  规则：1/1 保底策略（g=0，闲余充电、缺额先放电后紧急）；")
    p("        1/2–1/31 固定公共简单策略（前一日同刻预测 + 无日循环日前计划 + 解析执行）。")
    diag_led = {}
    for k in ("g", "a", "b", "C", "D", "U"):
        diag_led[k] = np.zeros((WARMUP_DAYS, T))
    E_chain = np.zeros((WARMUP_DAYS, T + 1))
    cost_norm = np.zeros(WARMUP_DAYS)
    cost_emerg = np.zeros(WARMUP_DAYS)
    nu_used = np.zeros(WARMUP_DAYS)
    R_chain = np.full((WARMUP_DAYS, T), np.nan)
    lp_status = np.zeros(WARMUP_DAYS, dtype=int)

    E = float(E_INIT)
    for d in range(WARMUP_DAYS):
        if d == 0:
            led = fallback_day(N_act[d], price_act[d], E)
        else:
            nu = float(price_act[:d, :C.NU_PRICE_POINTS].mean() / ETA)
            led = simple_policy_day(N_act[d], price_act[d], E,
                                    N_act[d - 1], price_act[d - 1], nu)
            nu_used[d] = nu
            lp_status[d] = led["lp_status"]
            R_chain[d] = led["R"]
        for k in ("g", "a", "b", "C", "D", "U"):
            diag_led[k][d] = led[k]
        E_chain[d] = led["E"]
        cost_norm[d] = led["cost_normal"]
        cost_emerg[d] = led["cost_emerg"]
        E = float(led["E"][-1])

    E_FEB1 = E
    p(f"  1 月总费用（单列，不计入评分期）：普通 {cost_norm.sum():,.2f} 元 + "
      f"紧急 {cost_emerg.sum():,.2f} 元 = {cost_norm.sum() + cost_emerg.sum():,.2f} 元")
    p(f"  2 月 1 日库存起点 E(2/1) = {E_FEB1:,.4f} kWh（共同分叉点）")
    p(f"  1 月库存范围：[{E_chain.min():,.2f}, {E_chain.max():,.2f}] kWh；"
      f"日末库存首/末 = {E_chain[0, -1]:,.1f} / {E_chain[-1, -1]:,.1f} kWh")

    dE = np.abs(np.diff(E_chain, axis=1)
                - (ETA * diag_led["C"] - diag_led["D"] / ETA))
    ck("库存递推 E' = E + ηC − D/η", float(dE.max()) < 1e-8,
       f"最大误差 {dE.max():.3e} kWh")
    U_impl = diag_led["a"] + diag_led["b"] + diag_led["D"] - diag_led["C"] - N_act[:WARMUP_DAYS]
    ck("逐时段能量平衡（弃光作为松弛 = 严格非负）",
       float(np.abs(U_impl - diag_led["U"]).max()) < 1e-8 and bool((diag_led["U"] >= -1e-9).all()),
       f"|U_impl−U|max {np.abs(U_impl - diag_led['U']).max():.3e} kWh")
    ck("充放互斥（C·D = 0）",
       float(np.minimum(diag_led["C"], diag_led["D"]).sum()) < 1e-9,
       f"Σmin(C,D) = {np.minimum(diag_led['C'], diag_led['D']).sum():.3e} kWh")
    ck("库存始终在安全区间内",
       bool((E_chain >= E_MIN - 1e-6).all() and (E_chain <= E_MAX + 1e-6).all()),
       f"[{E_chain.min():.2f}, {E_chain.max():.2f}] ⊂ [1200, 10800]")
    ck("充电功率不超过 P_max", bool((diag_led["C"] <= S + 1e-6).all()),
       f"max C = {diag_led['C'].max():.2f} ≤ {S:.2f} kWh/段")
    ck("放电功率不超过 P_max", bool((diag_led["D"] <= S + 1e-6).all()),
       f"max D = {diag_led['D'].max():.2f} ≤ {S:.2f} kWh/段")
    ck("1/2–1/31 日前计划 LP 全部最优", bool((lp_status[1:] == 0).all()),
       f"状态码集合 {sorted(set(lp_status[1:].tolist()))}")
    ck("1 月首日采用保底策略（g ≡ 0）", bool(np.allclose(diag_led["g"][0], 0.0)),
       "1/1 普通购电量为 0")

    p("")
    p("── 2. 价格候选预测的 1 月回放（仅用已完成日的因果窗口）──")
    nhat_snap = PO.P.net_load_feature_all(N_act, np.asarray(C.load_q2_scenarios()["Nhat"], float))
    p(f"  节点净负荷特征 nhat_snap {nhat_snap.shape}（节点前用实际、节点后用第二问预测）")
    jan_days = np.arange(0, WARMUP_DAYS)
    jan_scores: dict[str, dict] = {}
    for method in PO.P.METHODS:
        rep = PO.P.replay_forecasts(price_act, nhat_snap, method)
        sc = PO.P.score_forecasts(price_act, rep["chat_corr"], rep["H"], jan_days)
        jan_scores[method] = sc
        p(f"  {PO.P.METHOD_LABEL.get(method, method):<12s} "
          f"MAE {sc['mae']:.4f}  RMSE {sc['rmse']:.4f}  RMSE' {sc['rmse_dtrend']:.4f}  "
          f"偏差 {sc['bias']:+.4f} 元/kWh（n={sc['n']}）")
    p("  说明：1 月样本最终**不参与**主口径选择（选择在 03 的评分期诊断之外另立纪律），")
    p("        此处只作为「历史不足时回退行为」是否按流程 §5.3 触发的证据留档。")

    p("")
    p("── 3. 落盘 ──")
    np.savez_compressed(
        C.WARMUP_NPZ,
        dates=dates[:WARMUP_DAYS],
        N_actual=N_act[:WARMUP_DAYS],
        price_actual=price_act[:WARMUP_DAYS],
        g=diag_led["g"], a=diag_led["a"], b=diag_led["b"],
        C=diag_led["C"], D=diag_led["D"], U=diag_led["U"],
        E_chain=E_chain,
        retention=R_chain,
        nu_used=nu_used,
        cost_normal=cost_norm, cost_emerg=cost_emerg,
        lp_status=lp_status,
        E_feb1=np.array([E_FEB1]),
        E_init=np.array([E_INIT]),
        warmup_days=np.array([WARMUP_DAYS]),
        score_start_index=np.array([SCORE_START_IDX]),
        nhat_snap=nhat_snap,
    )
    p(f"  已保存：{C.WARMUP_NPZ.relative_to(C.ROOT_DIR)}")

    C.write_csv_utf8_sig(
        WARMUP_CSV,
        ["日期", "普通购电_元", "紧急购电_元", "当日合计_元", "净负荷_kWh",
         "光伏_kWh", "充电_kWh", "放电_kWh", "弃光_kWh", "紧急购电_kWh",
         "日末库存_kWh", "ν_元每kWh"],
        [[dates[d], C.fmt_num(cost_norm[d], 2), C.fmt_num(cost_emerg[d], 2),
          C.fmt_num(cost_norm[d] + cost_emerg[d], 2),
          C.fmt_num(N_act[d].sum(), 2), C.fmt_num(V_act[d].sum(), 2),
          C.fmt_num(diag_led["C"][d].sum(), 2), C.fmt_num(diag_led["D"][d].sum(), 2),
          C.fmt_num(diag_led["U"][d].sum(), 2), C.fmt_num(diag_led["b"][d].sum(), 2),
          C.fmt_num(E_chain[d, -1], 2), C.fmt_num(nu_used[d], 6)]
         for d in range(WARMUP_DAYS)])
    p(f"  已保存：{WARMUP_CSV.relative_to(C.ROOT_DIR)}")

    rp: list[str] = []
    rp.append("# 第四问 · 共同因果热启动报告（1 月）\n")
    rp.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"- 运行耗时：{time.perf_counter() - t0:.2f} s")
    rp.append(f"- 脚本：`第四问最终版/代码/02_共同热启动与历史预测回放.py`")
    rp.append(f"- 依据：《第四问详细流程图.md》§5.3\n")
    rp.append("## 1. 为什么需要共同热启动\n")
    rp.append("1 月作为**所有对照臂共用的状态链**：2 月 1 日从该链末态分叉，之后各臂独立运行。"
              "这样固定电价基线、4-2、4-3 的差异只来自策略本身。1 月费用**单列**，"
              "不计入官方评分期（2 月 1 日–12 月 31 日）总费用。\n")
    rp.append("## 2. 规则与防泄漏\n")
    rp.append("| 日期 | 策略 | 可用信息 | 是否读未来 |")
    rp.append("|---|---|---|---|")
    rp.append("| 2025-01-01 | 明示保底：`g=0`，富余光伏尽量充电、缺额先放电再紧急补足 | 无历史 | 否 |")
    rp.append("| 2025-01-02 – 01-31 | 固定公共简单策略：前一日同刻负荷/光伏/价格作预测 → "
              "无日循环约束日前计划 → 解析响应执行 | 仅前一日及更早 | 否 |\n")
    rp.append("- 续存价值 ν 只取**此前已结束日**的 `[00:00,05:00)` 谷段实际价均值 ÷ η。")
    rp.append("- 本脚本不读附件三预报（4-3 专用），1 月链两分支共用，故只用附件二实际光伏。\n")
    rp.append("### 2.1 附件二负荷的周内模式（关键数据事实，已独立核验）\n")
    rp.append("| 统计量 | 数值 |")
    rp.append("|---|---|")
    rp.append(f"| 日负荷中位数 | {med:,.0f} kWh |")
    rp.append(f"| 低于中位数 80% 的天数 | {low_idx.size} 天（占 {low_idx.size / 365 * 100:.1f}%）|")
    rp.append(f"| 这些低负荷日的星期分布 | Fri {low_wd['Fri']} 天、Sat {low_wd['Sat']} 天，其余 0 天 |")
    rp.append(f"| 低负荷比率区间 | {ratio[low_idx].min():.3f} – {ratio[low_idx].max():.3f}（相对中位数）|")
    rp.append("")
    rp.append("> **结论**：附件二负荷存在明确的**周五/周六低负荷模式**（约降至中位数的 0.61–0.69），"
              "周一–周四无此模式。该模式使「前一日同刻持续性」预测在 周四→周五（低估）与 "
              "周六→周日（高估）两个方向上系统偏错，是 1 月公共简单策略误差与紧急购电量的主要来源。"
              "1 月费用单列，不影响评分期结果；但该模式在 2–12 月同样存在，"
              "因此 4-2/4-3 的日前计划必须使用第二问已给出的预测负荷（`Nhat`）而非持续性外推。\n")
    rp.append("## 3. 结果\n")
    rp.append(f"- 1 月普通购电费用：**{cost_norm.sum():,.2f} 元**")
    rp.append(f"- 1 月紧急购电费用：**{cost_emerg.sum():,.2f} 元**")
    rp.append(f"- 1 月合计（单列，不计入评分期）：**{cost_norm.sum() + cost_emerg.sum():,.2f} 元**")
    rp.append(f"- 2 月 1 日共同起点库存：**E(2/1) = {E_FEB1:,.4f} kWh**")
    rp.append(f"- 1 月库存轨迹范围：[{E_chain.min():,.2f}, {E_chain.max():,.2f}] kWh")
    rp.append(f"- 1 月弃光合计：{diag_led['U'].sum():,.2f} kWh；紧急购电合计：{diag_led['b'].sum():,.2f} kWh\n")
    rp.append("| 日期 | 普通购电/元 | 紧急购电/元 | 净负荷/kWh | 弃光/kWh | 紧急购电/kWh | 日末库存/kWh | ν |")
    rp.append("|---|---|---|---|---|---|---|---|")
    for d in range(WARMUP_DAYS):
        rp.append(f"| {dates[d]} | {cost_norm[d]:,.2f} | {cost_emerg[d]:,.2f} | "
                  f"{N_act[d].sum():,.1f} | {diag_led['U'][d].sum():,.1f} | "
                  f"{diag_led['b'][d].sum():,.1f} | {E_chain[d, -1]:,.1f} | "
                  f"{nu_used[d] if d else float('nan'):.6f} |")
    rp.append("\n## 4. 价格候选预测的 1 月回放（校准样本）\n")
    rp.append("| 方法 | MAE/(元/kWh) | RMSE/(元/kWh) | 去漂移 RMSE | 偏差 | 有效点数 |")
    rp.append("|---|---|---|---|---|---|")
    for method in PO.P.METHODS:
        sc = jan_scores[method]
        rp.append(f"| {PO.P.METHOD_LABEL.get(method, method)} | {sc['mae']:.4f} | "
                  f"{sc['rmse']:.4f} | {sc['rmse_dtrend']:.4f} | {sc['bias']:+.4f} | {sc['n']} |")
    rp.append("\n> 1 月样本主要用于验证「历史不足 → 回退」链路是否按 §5.3 触发；"
              "主口径方法的选择在 03 中按评分期诊断（并为 4-2/4-3 分别固定），"
              "**不允许**用评分期结果反向挑方法。\n")
    rp.append("## 5. 校验\n")
    rp.append("| 校验项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")
    C.write_text_utf8(C.WARMUP_REPORT_MD, "\n".join(rp))
    p(f"  已保存：{C.WARMUP_REPORT_MD.relative_to(C.ROOT_DIR)}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    p("")
    p("=" * 78)
    p(f"02 完成：1 月 31 天链 + 5 个价格候选回放；校验 {len(checks)} 项，未通过 {n_fail} 项。"
      f"用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.LOG_DIR / "02_热启动日志.txt", "\n".join(log) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
