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


C = _load("_comm3.py", "q3_comm")
M05 = _load("05_求解阶段0计划.py", "q3_05")
TwoStageSP = M05.TwoStageSP

import numpy as np

T = C.PERIODS_PER_DAY


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t00 = time.perf_counter()

    log("=" * 78)
    log("第三问 06 —— 滚动调整规划（MPC：τ = 6:00 / 12:00 / 18:00）")
    log("=" * 78)
    log("")
    log("⚠ 历史实现：产物 `第三问_滚动调整.npz` **不参与正式结果**")
    log("  原因：以「规划库存」而非真实库存作为滚动起点（P0-1/P0-3）。")
    log("  正式路径：`_policy3.run_policy`（真实标量 SOC 闭环）→ `08` 结算。")
    log("")

    Z = C.Q2.matrix()
    S2 = C.Q2.scenarios()
    F = np.load(C.V_FORECAST_NPZ, allow_pickle=False)
    Zd = np.load(C.V_SCENARIO_NPZ, allow_pickle=False)
    D0 = np.load(C.STAGE0_NPZ, allow_pickle=False)

    price = np.asarray(Z["price"], float)
    scen_L = np.asarray(S2["scen_L"], float)
    scen_V_h = np.asarray(Zd["scen_V_hourly"], float)
    scen_V_abs = C.lead_to_absolute(scen_V_h)
    P = np.asarray(D0["P"], float)
    Ebar0 = np.asarray(D0["Ebar"], float)
    E_end0 = np.asarray(D0["E_end"], float)
    sc = np.asarray(F["score_day_index"], int)
    dates = [str(x) for x in F["dates"]]
    M = C.M_SCENARIOS
    nu = C.NU_VALUE

    taus = list(C.TAU_HOURS)
    k0s = list(C.TAU_PERIOD_INDEX)
    log(f"情景数 M = {M}；调整节点 τ = {taus}（时段下标 {k0s}）")

    for ti in range(1, 4):
        L = T - k0s[ti]
        lp = TwoStageSP(price[k0s[ti]:], M, nu, mode="adjust", p_seg=price[k0s[ti]:] * 0)
        log(f"  τ={taus[ti]}:00 段：剩余 {L} 时段，LP 变量 {lp.n_var}"
            f"（阶段1 = 2×{L}，阶段2 = {M}×4×{L}）")

    d0 = int(sc[0])
    LV = C.scenario_pv_energy(scen_V_abs, d0, 1)
    SN6 = scen_L[d0] - LV
    E6 = float(Ebar0[d0, k0s[1] - 1]) if Ebar0[d0, k0s[1] - 1] > 0 else C.E_INIT
    lp6 = TwoStageSP(price[k0s[1]:], M, nu, mode="adjust", p_seg=P[d0, k0s[1]:])
    t0 = time.perf_counter()
    r6 = lp6.solve(SN6[:, k0s[1]:], E6)
    dt = time.perf_counter() - t0
    log("")
    log(f"单日试算（{dates[d0]}，τ=6:00）：{dt:.2f} s，初始库存 {E6:.2f} kWh")
    log(f"  调整后购电量合计（36:144）= {r6['q'].sum():.2f} kWh，"
        f"计划量同段 = {P[d0, k0s[1]:].sum():.2f} kWh")
    log(f"  下调量 Σu = {r6['u'].sum():.2f} kWh，上调量 Σv = {r6['v'].sum():.2f} kWh")
    log(f"  → 预计全年滚动耗时 ≈ {dt * len(sc) * 0.55 / 60:.1f} 分钟")

    log("")
    log("全年 334 天滚动调整（每天 3 次调整，冻结过去、执行到下一节点）…")
    Q = np.zeros((365, T))
    U = np.zeros((365, T))
    V = np.zeros((365, T))
    SRC = np.zeros((365, T), int)
    E_node = np.zeros((365, 4))
    seg_fee = np.zeros((365, 4))
    seg_energy = np.zeros((365, 4))

    def seg_fee_of(d, k0, k1, q):
        px = price[k0:k1]
        pq = P[d, k0:k1]
        c = C.adjust_cost(px, pq, q)
        return c["total"], float(q.sum())

    t0 = time.perf_counter()
    n_done = 0
    for k, d in enumerate(sc):
        E_cur = float(E_end0[d - 1]) if (d > 0 and E_end0[d - 1] > 0) else C.E_INIT
        E_node[d, 0] = E_cur

        q_day = P[d].copy()
        SRC[d, :k0s[1]] = 0
        LV = C.scenario_pv_energy(scen_V_abs, d, 1)
        SN6 = scen_L[d] - LV
        lp = TwoStageSP(price[k0s[1]:], M, nu, mode="adjust", p_seg=P[d, k0s[1]:])
        r = lp.solve(SN6[:, k0s[1]:], E_cur)
        q_day[k0s[1]:] = r["q"]
        SRC[d, k0s[1]:] = 1
        E_cur = float(r["E"][:, 35].mean())
        E_node[d, 1] = E_cur
        f_, e_ = seg_fee_of(d, k0s[1], k0s[2], q_day[k0s[1]:k0s[2]])
        seg_fee[d, 1], seg_energy[d, 1] = f_, e_

        LV = C.scenario_pv_energy(scen_V_abs, d, 2)
        SN12 = scen_L[d] - LV
        lp = TwoStageSP(price[k0s[2]:], M, nu, mode="adjust", p_seg=P[d, k0s[2]:])
        r = lp.solve(SN12[:, k0s[2]:], E_cur)
        q_day[k0s[2]:] = r["q"]
        SRC[d, k0s[2]:] = 2
        E_cur = float(r["E"][:, 35].mean())
        E_node[d, 2] = E_cur
        f_, e_ = seg_fee_of(d, k0s[2], k0s[3], q_day[k0s[2]:k0s[3]])
        seg_fee[d, 2], seg_energy[d, 2] = f_, e_

        LV = C.scenario_pv_energy(scen_V_abs, d, 3)
        SN18 = scen_L[d] - LV
        lp = TwoStageSP(price[k0s[3]:], M, nu, mode="adjust", p_seg=P[d, k0s[3]:])
        r = lp.solve(SN18[:, k0s[3]:], E_cur)
        q_day[k0s[3]:] = r["q"]
        SRC[d, k0s[3]:] = 3
        E_cur = float(r["E"][:, 35].mean())
        E_node[d, 3] = E_cur
        f_, e_ = seg_fee_of(d, k0s[3], T, q_day[k0s[3]:T])
        seg_fee[d, 3], seg_energy[d, 3] = f_, e_
        f_, e_ = seg_fee_of(d, 0, k0s[1], q_day[0:k0s[1]])
        seg_fee[d, 0], seg_energy[d, 0] = f_, e_
        Q[d] = q_day
        U[d] = np.maximum(P[d] - q_day, 0.0)
        V[d] = np.maximum(q_day - P[d], 0.0)

        cA = C.adjust_cost(price, P[d], q_day)
        idl = cA["plan_fee"] + cA["adj_fee_down"] + cA["adj_fee_up"]
        idl2 = (float(price @ q_day) - float(price @ V[d])
                + 0.5 * float(price @ U[d]) + 1.5 * float(price @ V[d]))
        idl3 = float(price @ q_day) + 0.5 * float(price @ (U[d] + V[d]))
        for tag, val in (("(1)", idl2), ("(2)", idl3)):
            if abs(idl - val) > 1e-4 * max(1.0, abs(idl)):
                raise AssertionError(
                    f"{dates[d]} 成本恒等式{tag}不成立：{idl:.6f} vs {val:.6f}")
        n_done += 1
        if n_done % 40 == 0 or n_done == len(sc):
            el = time.perf_counter() - t0
            log(f"  {n_done}/{len(sc)} 天  用时 {el:.1f} s  "
                f"预计剩余 {el / n_done * (len(sc) - n_done) / 60:.1f} 分钟  "
                f"日末库存 {E_cur:.1f} kWh")
    el = time.perf_counter() - t0
    log(f"滚动调整完成，总用时 {el:.1f} s")
    log("  ✔ 逐日成本恒等式 c·min(p,q) + 0.5c·u + 1.5c·v ≡ c·q − c·v + 0.5c·u + 1.5c·v "
        "≡ c·q + 0.5c·(u+v) 全部通过")

    pf = np.zeros(365); ad = np.zeros(365); au = np.zeros(365)
    for d in sc:
        cA = C.adjust_cost(price, P[d], Q[d])
        pf[d], ad[d], au[d] = cA["plan_fee"], cA["adj_fee_down"], cA["adj_fee_up"]
    plan_sum = pf[sc].sum() + ad[sc].sum() + au[sc].sum()

    log("")
    log("── 评分期滚动调整汇总 ──")
    log(f"  最终调整购电量合计 = {Q[sc].sum():.2f} kWh")
    log(f"  计划购电量合计     = {P[sc].sum():.2f} kWh（差 {Q[sc].sum() - P[sc].sum():+.2f} kWh）")
    log(f"  下调量 Σu = {U[sc].sum():.2f} kWh，上调量 Σv = {V[sc].sum():.2f} kWh")
    log(f"  计划购电费（Σc·min(p,q)） = {pf[sc].sum():.2f} 元")
    log(f"  下调补偿（0.5c·u）        = {ad[sc].sum():.2f} 元")
    log(f"  上调加价（1.5c·v）        = {au[sc].sum():.2f} 元")
    log(f"  计划+调整合计             = {plan_sum:.2f} 元")
    log(f"  其中若完全不调整（q≡p）   = {float((P[sc] * price).sum()):.2f} 元")
    log(f"  → 调整带来的净费用变化    = {plan_sum - float((P[sc] * price).sum()):+.2f} 元")
    log("")
    for ti, tau in enumerate(taus):
        k0 = k0s[ti]
        k1 = k0s[ti + 1] if ti + 1 < 4 else T
        log(f"  τ={tau:2d}:00 段決策覆盖 [{k0:3d},{T:3d})，"
            f"实际执行 [{k0:3d},{k1:3d})，"
            f"执行段购电量 {seg_energy[sc, ti].sum():.2f} kWh，"
            f"执行段费用 {seg_fee[sc, ti].sum():.2f} 元")
    log(f"  日末库存（节点均值）= {E_node[sc, 3].mean():.2f} kWh，"
        f"范围 [{E_node[sc, 3].min():.1f}, {E_node[sc, 3].max():.1f}]")
    log(f"  4 个节点库存均值 = "
        + "，".join(f"{t}:00 {E_node[sc, i].mean():.1f}" for i, t in enumerate(taus)))

    N_act = np.asarray(Z["net_load_energy_kwh"], float)[sc]
    defc = np.maximum(N_act, 0.0)
    base_cost = float((defc @ price).sum())
    log("")
    log("── 与「完全不调整」对照 ──")
    log(f"  完全不调整（q≡p）计划购电费 = {float((P[sc] * price).sum()):.2f} 元")
    log(f"  滚动调整后计划+调整费       = {plan_sum:.2f} 元")
    log(f"  差额 = {plan_sum - float((P[sc] * price).sum()):+.2f} 元"
        f"（占 {100 * (plan_sum / float((P[sc] * price).sum()) - 1):+.3f}%）")
    log(f"  说明：调整费与紧急费的最终合成口径在 08 全年回测后给出；"
        f"此处金额未含 5c 紧急购电费。")

    np.savez_compressed(
        C.ROLLING_NPZ,
        Q=Q, U=U, V=V, SRC=SRC, E_node=E_node,
        seg_fee=seg_fee, seg_energy=seg_energy,
        plan_fee=pf, adj_fee_down=ad, adj_fee_up=au,
        score_day_index=sc, tau_period_index=np.asarray(C.TAU_PERIOD_INDEX, int),
        n_scenarios=np.asarray([M], int),
    )
    rows = []
    for d in sc:
        rows.append([dates[d],
                     f"{P[d].sum():.6f}", f"{Q[d].sum():.6f}",
                     f"{U[d].sum():.6f}", f"{V[d].sum():.6f}",
                     f"{pf[d]:.6f}", f"{ad[d]:.6f}", f"{au[d]:.6f}",
                     f"{pf[d] + ad[d] + au[d]:.6f}",
                     f"{E_node[d, 1]:.6f}", f"{E_node[d, 2]:.6f}", f"{E_node[d, 3]:.6f}"])
    roll_csv = C.RESULT_DIR / "第三问_滚动调整日汇总.csv"
    C.write_csv_utf8_sig(
        roll_csv,
        ["日期", "计划购电量_kWh", "调整购电量_kWh", "下调量_kWh", "上调量_kWh",
         "计划购电费_元", "下调补偿_元", "上调加价_元", "计划加调整_元",
         "6点储电量_kWh", "12点储电量_kWh", "18点储电量_kWh"], rows)
    log("")
    log(f"已保存：{C.ROLLING_NPZ.relative_to(C.PROJECT_DIR)}")
    log(f"已保存：{roll_csv.relative_to(C.PROJECT_DIR)}")

    log("")
    log(f"总用时 {time.perf_counter() - t00:.1f} s")
    log("[06 完成] 滚动调整规划结束。")
    log.dump(C.SOLVE_LOG_DIR / "第三问_06滚动调整日志.txt", "[06 完成] 滚动调整结束。")
    C.write_text_utf8(C.REPORT_ROLL_MD, "\n".join([
        "# 第三问 滚动调整规划报告（τ = 6:00 / 12:00 / 18:00）", "",
        "> ⚠ **本报告为历史实现留存件，其中的全部数字都不参与第三问正式结果。**",
        "> 该实现存在两处已确认缺陷（见 `第三问模型问题汇总与整改建议.md` 的 P0-1 / P0-3）：",
        "> ① 节点间不传递真实库存（等价于每段独立决策）；",
        "> ② 调整量口径未与 `_policy3.decompose` 的 `q = p − u + v` 对齐同名约束。",
        "> 正式结果一律取自 `08_全年回测与结算.py`；本文件仅用于对照说明。",
        "> 严禁把本报告中的任何金额、电量抄入论文或提交结果。", "",
        "> 脚本：`第三问最终版/代码/06_滚动调整规划.py`；"
        "产出：`模型结果/第三问_滚动调整.npz`", "",
        "## 0. 结论速览（⚠ 非正式结论，仅供对照）", "",
        f"* 最终调整购电量合计 **{Q[sc].sum():.2f} kWh**，"
        f"计划购电量 {P[sc].sum():.2f} kWh，"
        f"净差 {Q[sc].sum() - P[sc].sum():+.2f} kWh；",
        f"* 下调量 Σu = {U[sc].sum():.2f} kWh，上调量 Σv = {V[sc].sum():.2f} kWh；",
        f"* 计划 + 调整合计 **{plan_sum:.2f} 元**"
        f"（计划购电费 {pf[sc].sum():.2f} + 下调补偿 {ad[sc].sum():.2f} "
        f"+ 上调加价 {au[sc].sum():.2f}）；",
        f"* 相对「完全不调整」（{float((P[sc] * price).sum()):.2f} 元）"
        f"变化 {plan_sum - float((P[sc] * price).sum()):+.2f} 元；",
        "* 每个 τ 节点只调整**剩余**时段、冻结已过去时段、只执行到下一节点（标准 MPC）；",
        "* 逐日通过调整费恒等式校验。", ""] + log.lines + [""]))
    log("已保存：报告/第三问_滚动调整报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
