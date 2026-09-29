from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    PERIODS_PER_DAY, DELTA_HOURS,
    ETA_CHARGE, ETA_DISCHARGE, P_CHARGE_MAX, P_DISCHARGE_MAX,
    E_MIN, E_MAX, E_INITIAL, E_TERMINAL,
    load_formal_input, OPT_SCHEDULE_CSV, VALIDATION_CSV,
    compute_sha256, ATTACHMENT1_PATH, CONFIG_YAML, minutes_to_hhmm,
)

N = PERIODS_PER_DAY
DT = DELTA_HOURS
EPS_RESIDUAL = 1e-6
EPS_NEAR_BOUND = 1e-3
EPS_ZERO = 1e-6


def _config_sha256_baseline():
    if not CONFIG_YAML.exists():
        return ""
    record = CONFIG_YAML.read_text(encoding="utf-8")
    for line in record.splitlines():
        if "attachment1_sha256" in line and ":" in line:
            return line.split(":", 1)[1].strip().strip('"').strip()
    return ""


def validate(solver_obj=None):
    df = pd.read_csv(OPT_SCHEDULE_CSV, encoding="utf-8-sig")
    inp = load_formal_input()

    price = df["price_yuan_per_kwh"].to_numpy(float)
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_forecast_kw"].to_numpy(float)
    G = df["grid_purchase_kw"].to_numpy(float)
    C = df["charge_power_kw"].to_numpy(float)
    D = df["discharge_power_kw"].to_numpy(float)
    W = df["pv_curtailment_kw"].to_numpy(float)
    z = df["charge_state"].to_numpy(int)
    E_start = df["energy_start_kwh"].to_numpy(float)
    E_end = df["energy_end_kwh"].to_numpy(float)

    E_seq = np.concatenate([E_start, [E_end[-1]]])
    boundary_labels = [minutes_to_hhmm((i - 1) * 10) for i in range(1, N + 2)]

    r_power = G + pv + D - load - C - W
    r_soc = E_end - (E_start + ETA_CHARGE * C * DT - D * DT / ETA_DISCHARGE)

    checks = []

    def add(no, name, req, val, ok):
        checks.append({"序号": no, "校验项": name, "要求": req,
                       "计算结果": val, "结果": "通过" if ok else "不通过"})

    max_rp = float(np.max(np.abs(r_power)))
    add(1, "功率平衡残差", f"max|r| ≤ {EPS_RESIDUAL}",
        f"max|r_power| = {max_rp:.3e}", max_rp <= EPS_RESIDUAL + 1e-9)

    max_rs = float(np.max(np.abs(r_soc)))
    add(2, "SOC递推残差", f"max|r| ≤ {EPS_RESIDUAL}",
        f"max|r_soc| = {max_rs:.3e}", max_rs <= EPS_RESIDUAL + 1e-9)

    e_min = float(E_seq.min())
    e_max = float(E_seq.max())
    viol_low = int((E_seq < E_MIN - EPS_NEAR_BOUND).sum())
    viol_high = int((E_seq > E_MAX + EPS_NEAR_BOUND).sum())
    low_idx = [i + 1 for i in range(N + 1) if abs(E_seq[i] - E_MIN) <= EPS_NEAR_BOUND]
    high_idx = [i + 1 for i in range(N + 1) if abs(E_seq[i] - E_MAX) <= EPS_NEAR_BOUND]
    low_lab = "、".join(boundary_labels[i - 1] for i in low_idx) if low_idx else "无"
    high_lab = "、".join(boundary_labels[i - 1] for i in high_idx) if high_idx else "无"
    ok3 = (e_min >= E_MIN - 1e-9) and (e_max <= E_MAX + 1e-9) and viol_low == 0 and viol_high == 0
    add(3, "SOC范围", f"{E_MIN} ≤ E ≤ {E_MAX}",
        f"min={e_min:.6f}, max={e_max:.6f}, 越界={viol_low + viol_high}"
        f"；达下限[{low_lab}]；达上限[{high_lab}]", ok3)

    max_c = float(C.max())
    max_d = float(D.max())
    ok4 = bool((C >= -EPS_ZERO).all() and (C <= P_CHARGE_MAX + 1e-9).all()
               and (D >= -EPS_ZERO).all() and (D <= P_DISCHARGE_MAX + 1e-9).all())
    add(4, "充放电功率范围", f"0 ≤ C ≤ {P_CHARGE_MAX}, 0 ≤ D ≤ {P_DISCHARGE_MAX}",
        f"max C={max_c:.6f} kW, max D={max_d:.6f} kW", ok4)

    okC_z = bool((C <= P_CHARGE_MAX * z + 1e-9).all())
    okD_z = bool((D <= P_DISCHARGE_MAX * (1 - z) + 1e-9).all())
    sim = int(((C > EPS_ZERO) & (D > EPS_ZERO)).sum())
    ok5 = okC_z and okD_z and sim == 0
    add(5, "充放电互斥", "C ≤ 5000z, D ≤ 5000(1−z), 同时充放电=0",
        f"同时充放电时段数 = {sim}", ok5)

    ok6 = bool((G >= -EPS_ZERO).all() and (W >= -EPS_ZERO).all())
    add(6, "购电与弃光非负", "G ≥ 0, W ≥ 0",
        f"min G={G.min():.3e}, min W={W.min():.3e}", ok6)

    ok7 = abs(E_seq[0] - E_INITIAL) <= 1e-6 and abs(E_seq[-1] - E_TERMINAL) <= 1e-6
    add(7, "首末储能量", f"E_1 = E_145 = {E_INITIAL}",
        f"E_1={E_seq[0]:.6f}, E_145={E_seq[-1]:.6f}", ok7)

    J1_check = float(df["period_purchase_cost_yuan"].sum())
    if solver_obj is None:
        ok8 = True
        info8 = f"J_1_check = {J1_check:.8f} 元（未提供求解器目标值，仅复算）"
    else:
        ok8 = abs(J1_check - solver_obj) <= 1e-4
        info8 = (f"J_1_check = {J1_check:.8f} 元 vs 求解器 {solver_obj:.8f} 元，"
                 f"偏差 {abs(J1_check - solver_obj):.3e}")
    add(8, "目标函数复算", "Σ period_purchase_cost = 求解器 J_1", info8, ok8)

    total_grid = float((G * DT).sum())
    total_pv = float((pv * DT).sum())
    total_dout = float((D * DT).sum())
    total_load = float((load * DT).sum())
    total_cin = float((C * DT).sum())
    total_w = float((W * DT).sum())
    lhs = total_grid + total_pv + total_dout
    rhs = total_load + total_cin + total_w
    ok9 = abs(lhs - rhs) <= 1e-4
    add(9, "总能量平衡", "购电+光伏+放电输出 = 负荷+充电输入+弃光",
        f"左={lhs:.6f}, 右={rhs:.6f}, 差={lhs - rhs:.3e}", ok9)

    ok_rows = len(inp) == N and len(df) == N
    ok_missing = not inp[["price_yuan_per_kwh", "load_kw", "pv_forecast_kw"]].isna().any().any()
    baseline = _config_sha256_baseline()
    actual = compute_sha256(ATTACHMENT1_PATH) if ATTACHMENT1_PATH.exists() else ""
    ok_sha = (baseline == "") or (baseline == actual)
    ok10 = ok_rows and ok_missing and ok_sha
    add(10, "原始数据一致性",
        "144行、输入字段无缺失、附件SHA-256未变",
        f"行数={len(inp)}、缺失={not ok_missing}、SHA一致={ok_sha}", ok10)

    vdf = pd.DataFrame(checks)
    vdf.to_csv(VALIDATION_CSV, index=False, encoding="utf-8-sig")

    core = [c for c in checks if isinstance(c["序号"], int)]
    n_pass = sum(1 for c in core if c["结果"] == "通过")
    n_total = len(core)
    all_pass = n_pass == n_total

    summary = {
        "all_pass": all_pass,
        "n_pass": n_pass,
        "n_total": n_total,
        "max_r_power": max_rp,
        "max_r_soc": max_rs,
        "e_min": e_min, "e_max": e_max,
        "low_idx": low_idx, "high_idx": high_idx,
        "max_charge_kw": max_c, "max_discharge_kw": max_d,
        "simultaneous_cnt": sim,
        "J1_check": J1_check,
        "total_grid_energy": total_grid, "total_pv_energy": total_pv,
        "total_charge_input_energy": total_cin,
        "total_discharge_output_energy": total_dout,
        "total_load_energy": total_load, "total_curtail_energy": total_w,
        "E1": float(E_seq[0]), "E145": float(E_seq[-1]),
        "n_rows": len(inp),
    }

    print(f"[07] 约束验收：{n_pass}/{n_total} 项通过")
    print(f"[07] 功率平衡 max|r|={max_rp:.3e}；SOC 递推 max|r|={max_rs:.3e}")
    print(f"[07] 储能量范围 [{e_min:.4f}, {e_max:.4f}]；同时充放电时段数={sim}")
    print(f"[07] 已导出：{VALIDATION_CSV.name}")
    return summary


if __name__ == "__main__":
    validate()
