from pathlib import Path
import csv

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT / "模型结果"
SOLUTION_CSV = RESULT_DIR / "最终LP_原始最优解.csv"
SUMMARY_CSV = RESULT_DIR / "最终LP_求解摘要.csv"
MILP_SUMMARY = PROJECT / "历史模型" / "MILP历史基准校验摘要.csv"

OUT_ACCEPT = RESULT_DIR / "最终LP_约束验收.csv"
OUT_CROSS = RESULT_DIR / "最终LP与历史MILP互证.csv"

N = 144
DT = 1.0 / 6.0
ETA_C = 0.9
ETA_D = 0.9
E_MIN = 1200.0
E_MAX = 10800.0
E_INIT = 6000.0
E_TERM = 6000.0
S = 5000.0 * DT


def read_solver_objective():
    with open(SUMMARY_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["指标"] == "最优目标值_元":
                return float(r["数值"])
    raise ValueError("未在求解摘要中找到最优目标值")


def read_milp_benchmark():
    if not MILP_SUMMARY.is_file():
        return None
    j, q = None, None
    with open(MILP_SUMMARY, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["指标"] == "最低费用 J_1":
                j = float(r["数值"])
            elif r["指标"] == "总购电量":
                q = float(r["数值"])
    assert j is not None and q is not None, "未在冻结摘要中找到MILP基准值"
    return j, q


def main():
    df = pd.read_csv(SOLUTION_CSV, encoding="utf-8-sig")
    g = df["grid_purchase_energy_kwh"].to_numpy(float)
    C = df["charge_energy_kwh"].to_numpy(float)
    D = df["discharge_energy_kwh"].to_numpy(float)
    U = df["unused_pv_energy_kwh"].to_numpy(float)
    L = df["load_energy_kwh"].to_numpy(float)
    V = df["pv_forecast_energy_kwh"].to_numpy(float)
    price = df["price_yuan_per_kwh"].to_numpy(float)

    E_start = df["energy_start_kwh"].to_numpy(float)
    E_end = df["energy_end_kwh"].to_numpy(float)
    E_all = np.concatenate([E_start[:1], E_end])

    checks = []

    r_bal = g + V + D - L - C - U
    mb = float(np.max(np.abs(r_bal)))
    checks.append(("供需平衡残差", "max|r| ≤ 1e-06", f"max|r_balance| = {mb:.3e}", mb <= 1e-6))

    r_energy = E_end - E_start - ETA_C * C + D / ETA_D
    me = float(np.max(np.abs(r_energy)))
    checks.append(("储能递推残差", "max|r| ≤ 1e-06", f"max|r_energy| = {me:.3e}", me <= 1e-6))

    lo_viol = int((E_all < E_MIN - 1e-6).sum())
    hi_viol = int((E_all > E_MAX + 1e-6).sum())
    checks.append(("储能边界", f"{E_MIN} ≤ E ≤ {E_MAX}（145 个状态）",
                   f"min={E_all.min():.6f}, max={E_all.max():.6f}, 下越界={lo_viol}, 上越界={hi_viol}",
                   lo_viol == 0 and hi_viol == 0))

    E0_v = float(E_all[0])
    E144_v = float(E_all[-1])
    checks.append(("首末储电量", f"E_0 = E_144 = {E_INIT}",
                   f"E_0={E0_v:.6f}, E_144={E144_v:.6f}",
                   abs(E0_v - E_INIT) < 1e-6 and abs(E144_v - E_TERM) < 1e-6))

    c_viol = int((C > S + 1e-6).sum()) + int((C < -1e-6).sum())
    d_viol = int((D > S + 1e-6).sum()) + int((D < -1e-6).sum())
    checks.append(("充放电上限", f"0 ≤ C ≤ {S:.10f}, 0 ≤ D ≤ {S:.10f}",
                   f"max C={C.max():.6f}, max D={D.max():.6f}, C越界={c_viol}, D越界={d_viol}",
                   c_viol == 0 and d_viol == 0))

    gmin = float(g.min())
    umin = float(U.min())
    checks.append(("非负约束", "g_t ≥ 0, U_t ≥ 0",
                   f"min g={gmin:.3e}, min U={umin:.3e}",
                   gmin >= -1e-6 and umin >= -1e-6))

    j_check = float(np.sum(price * g))
    j_solver = read_solver_objective()
    checks.append(("目标函数复算", "Σ c_t·g_t = 求解器 J_1",
                   f"J_check={j_check:.8f} vs 求解器 {j_solver:.8f}，偏差 {abs(j_check - j_solver):.3e}",
                   abs(j_check - j_solver) < 1e-6))

    lhs = float(g.sum() + V.sum() + D.sum())
    rhs = float(L.sum() + C.sum() + U.sum())
    checks.append(("总能量平衡", "购电+光伏+放电 = 负荷+充电+弃光",
                   f"左={lhs:.6f}, 右={rhs:.6f}, 差={abs(lhs - rhs):.3e}",
                   abs(lhs - rhs) < 1e-4))

    n_1e7 = int(((C > 1e-7) & (D > 1e-7)).sum())
    n_1e5 = int(((C > 1e-5) & (D > 1e-5)).sum())
    n_1e3 = int(((C > 1e-3) & (D > 1e-3)).sum())
    sim_idx = np.where((C > 1e-7) & (D > 1e-7))[0]
    sim_detail = "; ".join(
        f"t={int(df['interval_index'].iloc[i])}(C={C[i]:.6f},D={D[i]:.6f})"
        for i in sim_idx[:20])
    if len(sim_idx) > 20:
        sim_detail += f" ...共{len(sim_idx)}个"
    checks.append(("同时充放电诊断", "非失败项，保留原始解",
                   f"1e-7:{n_1e7}个，1e-5:{n_1e5}个，1e-3:{n_1e3}个 ; {sim_detail if n_1e7 else '无'}",
                   True))

    header = ["序号", "校验项", "要求", "计算结果", "结果"]
    accept_rows = []
    hard_pass = 0
    hard_total = 0
    for i, (name, req, val, ok) in enumerate(checks, 1):
        if name == "同时充放电诊断":
            status = "诊断（非失败项）"
        else:
            status = "通过" if ok else "未通过"
            hard_total += 1
            hard_pass += 1 if ok else 0
        accept_rows.append((i, name, req, val, status))
    pd.DataFrame(accept_rows, columns=header).to_csv(OUT_ACCEPT, index=False, encoding="utf-8-sig")

    q_lp = float(g.sum())
    benchmark = read_milp_benchmark()
    if benchmark is None:
        j_milp = q_milp = None
        obj_abs = obj_rel = q_abs = 0.0
        cross = pd.DataFrame([
            ("历史MILP互证", f"{j_solver:.10f}", "未提供", "—", "—", "—",
             "开发期历史基准为可选材料", "跳过（不影响LP约束验收）"),
        ], columns=["指标", "LP值", "MILP值", "单位", "绝对误差", "相对误差(费用)",
                    "验收准则", "结果"])
    else:
        j_milp, q_milp = benchmark
        obj_abs = abs(j_solver - j_milp)
        obj_rel = obj_abs / max(1.0, abs(j_milp))
        q_abs = abs(q_lp - q_milp)
        cross = pd.DataFrame([
            ("最优费用 J_1", f"{j_solver:.10f}", f"{j_milp:.10f}", "元",
             f"{obj_abs:.6e}", f"{obj_rel:.6e}", "objective_abs_error ≤ 1e-5 元",
             "通过" if obj_abs <= 1e-5 else "未通过"),
            ("总购电量", f"{q_lp:.10f}", f"{q_milp:.10f}", "kWh",
             f"{q_abs:.6e}", "-", "purchase_abs_error ≤ 1e-4 kWh",
             "通过" if q_abs <= 1e-4 else "未通过"),
        ], columns=["指标", "LP值", "MILP值", "单位", "绝对误差", "相对误差(费用)",
                    "验收准则", "结果"])
    cross.to_csv(OUT_CROSS, index=False, encoding="utf-8-sig")

    print(f"[14] 硬约束验收：{hard_pass} / {hard_total} 通过")
    print(f"[14] 同时充放电诊断：1e-7阈值 {n_1e7} 个；1e-5 {n_1e5} 个；1e-3 {n_1e3} 个")
    if benchmark is None:
        print("[14] 与旧MILP互证：未提供开发期历史基准，已跳过。")
    else:
        print(f"[14] 与旧MILP互证：费用误差={obj_abs:.3e}，购电量误差={q_abs:.3e}")

    return {
        "accept": accept_rows,
        "hard_pass": hard_pass,
        "hard_total": hard_total,
        "max_balance_resid": mb,
        "max_energy_resid": me,
        "E_min": float(E_all.min()),
        "E_max": float(E_all.max()),
        "E0": E0_v,
        "E144": E144_v,
        "j_check": j_check,
        "j_solver": j_solver,
        "lhs": lhs,
        "rhs": rhs,
        "n_sim_1e7": n_1e7,
        "n_sim_1e5": n_1e5,
        "n_sim_1e3": n_1e3,
        "sim_detail": sim_detail,
        "j_milp": j_milp,
        "q_milp": q_milp,
        "q_lp": q_lp,
        "obj_abs": obj_abs,
        "obj_rel": obj_rel,
        "q_abs": q_abs,
        "benchmark_available": benchmark is not None,
        "cross": cross,
    }


if __name__ == "__main__":
    main()
