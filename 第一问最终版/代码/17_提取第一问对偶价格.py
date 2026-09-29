from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT / "模型结果"

SOLUTION_CSV = RESULT_DIR / "最终LP_原始最优解.csv"
DUAL_EQ_CSV = RESULT_DIR / "最终LP_原始对偶数据.csv"
DUAL_BD_CSV = RESULT_DIR / "最终LP_变量边界对偶数据.csv"
ROW_MAP_CSV = RESULT_DIR / "最终LP_约束行映射.csv"
SUMMARY_CSV = RESULT_DIR / "最终LP_求解摘要.csv"

OUT_TABLE = RESULT_DIR / "第一问_内部价格与储能阈值.csv"
OUT_PLOT_DATA = RESULT_DIR / "第一问_内部价格与阈值作图数据.csv"

ETA = 0.9
ETA2 = ETA * ETA
S = 5000.0 * (1.0 / 6.0)
E_MIN, E_MAX = 1200.0, 10800.0
J_LOCK = 35126.9485892896
Q_LOCK = 59482.6989983539

EPS_VAR = 1e-7
EPS_PRICE = 1e-7
EPS_RC = 1e-7
EPS_BOUND = 1e-5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def parse_hhmm_to_hours(s):
    s = str(s).strip()
    h, m = s.split(":")[:2]
    return int(h) + int(m) / 60.0


def read_inputs():
    sol = pd.read_csv(SOLUTION_CSV, encoding="utf-8-sig")
    dual = pd.read_csv(DUAL_EQ_CSV, encoding="utf-8-sig")
    rowmap = pd.read_csv(ROW_MAP_CSV, encoding="utf-8-sig")
    summary = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    return sol, dual, rowmap, summary


def _pi_kkt_violation(pi_cand, c, g, U):
    return float(max(
        np.maximum(-pi_cand, 0.0).max(),
        np.maximum(pi_cand - c, 0.0).max(),
        np.abs(g * (c - pi_cand)).max(),
        np.abs(pi_cand * U).max(),
    ))


def _lam_kkt_violation(pi, lam_cand, C, D):
    eta_lam = ETA * lam_cand
    lam_over = lam_cand / ETA
    rc_c = pi - eta_lam
    rc_d = -pi + lam_over
    v = 0.0
    for t in range(len(pi)):
        if C[t] <= EPS_VAR:
            v = max(v, float(-rc_c[t]))
        elif C[t] >= S - EPS_BOUND:
            v = max(v, float(rc_c[t]))
        else:
            v = max(v, float(abs(rc_c[t])))
        if D[t] <= EPS_VAR:
            v = max(v, float(-rc_d[t]))
        elif D[t] >= S - EPS_BOUND:
            v = max(v, float(rc_d[t]))
        else:
            v = max(v, float(abs(rc_d[t])))
    return v


def extract_duals(dual, c, g, C, D, U):
    bal = dual[dual["constraint_type"] == "供需平衡"].sort_values("interval_index")
    sto = dual[dual["constraint_type"] == "储能递推"].sort_values("interval_index")
    assert len(bal) == 144 and len(sto) == 144, "对偶数据应为供需平衡144 + 储能递推144"

    raw_bal = bal["raw_marginal"].to_numpy(float)
    raw_sto = sto["raw_marginal"].to_numpy(float)

    vA = _pi_kkt_violation(raw_bal, c, g, U)
    vB = _pi_kkt_violation(-raw_bal, c, g, U)
    pi_neg = vB <= vA
    pi = -raw_bal if pi_neg else raw_bal
    pi_v, pi_alt = (vB, vA) if pi_neg else (vA, vB)
    assert pi_v < 1e-3, f"π 两种候选均违反式(9)：+raw={vA:.3e}, -raw={vB:.3e}"

    vA = _lam_kkt_violation(pi, raw_sto, C, D)
    vB = _lam_kkt_violation(pi, -raw_sto, C, D)
    lam_neg = vB <= vA
    lam = -raw_sto if lam_neg else raw_sto
    lam_v, lam_alt = (vB, vA) if lam_neg else (vA, vB)
    assert lam_v < 1e-3, f"λ 两种候选均违反充放电 KKT：+raw={vA:.3e}, -raw={vB:.3e}"

    decision = {
        "pi_chosen": "-raw_marginal" if pi_neg else "+raw_marginal",
        "pi_violation": pi_v,
        "pi_alt_violation": pi_alt,
        "lambda_chosen": "-raw_marginal" if lam_neg else "+raw_marginal",
        "lambda_violation": lam_v,
        "lambda_alt_violation": lam_alt,
    }
    return pi, lam, raw_bal, raw_sto, decision


def build_table(sol, pi, lam):
    N = len(sol)
    c = sol["price_yuan_per_kwh"].to_numpy(float)
    g = sol["grid_purchase_energy_kwh"].to_numpy(float)
    C = sol["charge_energy_kwh"].to_numpy(float)
    D = sol["discharge_energy_kwh"].to_numpy(float)
    U = sol["unused_pv_energy_kwh"].to_numpy(float)
    E_start = sol["energy_start_kwh"].to_numpy(float)
    E_end = sol["energy_end_kwh"].to_numpy(float)

    eta_lam = ETA * lam
    lam_over_eta = lam / ETA
    rc_c = pi - eta_lam
    rc_d = -pi + lam_over_eta
    grid_comp = g * (c - pi)
    curt_comp = pi * U

    rows = []
    kkt_flags = []
    for t in range(N):
        if C[t] >= S - EPS_BOUND:
            c_state = "upper"
        elif C[t] <= EPS_VAR:
            c_state = "zero"
        else:
            c_state = "interior"

        if D[t] >= S - EPS_BOUND:
            d_state = "upper"
        elif D[t] <= EPS_VAR:
            d_state = "zero"
        else:
            d_state = "interior"

        soc_lo = (abs(E_start[t] - E_MIN) <= EPS_BOUND) or (abs(E_end[t] - E_MIN) <= EPS_BOUND)
        soc_hi = (abs(E_start[t] - E_MAX) <= EPS_BOUND) or (abs(E_end[t] - E_MAX) <= EPS_BOUND)

        pi_range_ok = (-EPS_PRICE <= pi[t]) and (pi[t] <= c[t] + EPS_PRICE)
        grid_comp_ok = abs(grid_comp[t]) <= 1e-6
        curt_comp_ok = abs(curt_comp[t]) <= 1e-6
        if c_state == "upper":
            c_ok = rc_c[t] <= EPS_RC
        elif c_state == "interior":
            c_ok = abs(rc_c[t]) <= EPS_RC
        else:
            c_ok = rc_c[t] >= -EPS_RC
        if d_state == "upper":
            d_ok = rc_d[t] <= EPS_RC
        elif d_state == "interior":
            d_ok = abs(rc_d[t]) <= EPS_RC
        else:
            d_ok = rc_d[t] >= -EPS_RC
        ok = pi_range_ok and grid_comp_ok and curt_comp_ok and c_ok and d_ok
        kkt_flags.append(bool(ok))

        rows.append({
            "interval_index": int(sol["interval_index"].iloc[t]),
            "interval_start": sol["interval_start"].iloc[t],
            "interval_end": sol["interval_end"].iloc[t],
            "external_price": c[t],
            "internal_price_pi": pi[t],
            "storage_value_lambda": lam[t],
            "charge_threshold_eta_lambda": eta_lam[t],
            "discharge_threshold_lambda_over_eta": lam_over_eta[t],
            "grid_purchase_energy_kwh": g[t],
            "charge_energy_kwh": C[t],
            "discharge_energy_kwh": D[t],
            "unused_pv_energy_kwh": U[t],
            "energy_start_kwh": E_start[t],
            "energy_end_kwh": E_end[t],
            "charge_reduced_cost": rc_c[t],
            "discharge_reduced_cost": rc_d[t],
            "grid_complementarity": grid_comp[t],
            "curtailment_complementarity": curt_comp[t],
            "charge_state": c_state,
            "discharge_state": d_state,
            "soc_lower_active": bool(soc_lo),
            "soc_upper_active": bool(soc_hi),
            "kkt_pass": bool(ok),
        })
    df = pd.DataFrame(rows)
    return df, dict(c=c, g=g, C=C, D=D, U=U,
                    pi=pi, lam=lam, eta_lam=eta_lam, lam_over_eta=lam_over_eta,
                    rc_c=rc_c, rc_d=rc_d, grid_comp=grid_comp, curt_comp=curt_comp,
                    kkt_flags=kkt_flags)


def compute_price_stats(arr):
    c = arr["c"]; g = arr["g"]; U = arr["U"]; pi = arr["pi"]
    j_total = float((c * g).sum())
    grid_comp_abs = np.abs(g * (c - pi))
    curt_comp_abs = np.abs(pi * U)
    stats = {
        "max_pi_lower_violation": float(np.maximum(-pi, 0.0).max()),
        "max_pi_upper_violation": float(np.maximum(pi - c, 0.0).max()),
        "max_grid_comp": float(grid_comp_abs.max()),
        "max_grid_comp_norm": float(grid_comp_abs.max() / j_total),
        "max_curt_comp": float(curt_comp_abs.max()),
        "max_curt_comp_norm": float(curt_comp_abs.max() / j_total),
        "j_total": j_total,
    }
    pos_g = g > EPS_VAR
    pos_u = U > EPS_VAR
    if pos_g.any():
        stats["max_err_pi_eq_c_when_g_pos"] = float(np.max(np.abs(pi[pos_g] - c[pos_g])))
        stats["num_g_pos"] = int(pos_g.sum())
    else:
        stats["max_err_pi_eq_c_when_g_pos"] = 0.0
        stats["num_g_pos"] = 0
    if pos_u.any():
        stats["max_err_pi_eq_0_when_U_pos"] = float(np.max(np.abs(pi[pos_u])))
        stats["num_u_pos"] = int(pos_u.sum())
    else:
        stats["max_err_pi_eq_0_when_U_pos"] = 0.0
        stats["num_u_pos"] = 0
    stats["pi_min"] = float(pi.min())
    stats["pi_max"] = float(pi.max())
    stats["lam_min"] = float(arr["lam"].min())
    stats["lam_max"] = float(arr["lam"].max())
    return stats


def compute_arbitrage(arr):
    c = arr["c"]; g = arr["g"]; C = arr["C"]; D = arr["D"]
    buy_charge = np.where((g > EPS_VAR) & (C > EPS_VAR))[0]
    discharge = np.where(D > EPS_VAR)[0]
    results = []
    if buy_charge.size and discharge.size:
        night = buy_charge[buy_charge < 12]
        a_low = int(night[np.argmin(c[night])]) if night.size else int(buy_charge[np.argmin(c[buy_charge])])
        b_high = int(discharge[np.argmax(c[discharge])])
        for a, b, tag in [(a_low, b_high, "夜充→晚峰放电（最低购电价→最高放电价）")]:
            ca, cb = c[a], c[b]
            cond = ETA2 * cb - ca
            results.append({
                "charge_interval": a + 1,
                "discharge_interval": b + 1,
                "charge_slot": (a + 1), "c_a": ca,
                "discharge_slot": b + 1, "c_b": cb,
                "eta2_c_b_minus_c_a": cond,
                "arbitrage_feasible": bool(cond > 0),
                "tag": tag,
            })
    return results


def write_plot_data(sol, arr):
    t_end_h = np.array([parse_hhmm_to_hours(x) for x in sol["interval_end"]])
    df = pd.DataFrame({
        "interval_index": sol["interval_index"].to_numpy(int),
        "interval_end": sol["interval_end"].to_numpy(),
        "interval_end_hours": t_end_h,
        "external_price": arr["c"],
        "internal_price_pi": arr["pi"],
        "charge_threshold_eta_lambda": arr["eta_lam"],
        "discharge_threshold_lambda_over_eta": arr["lam_over_eta"],
    })
    df.to_csv(OUT_PLOT_DATA, index=False, encoding="utf-8-sig", float_format="%.10f")
    return t_end_h


def main():
    sol, dual, rowmap, summary = read_inputs()
    j = float(summary[summary["指标"] == "最优目标值_元"]["数值"].iloc[0])
    q_real = float(sol["grid_purchase_energy_kwh"].sum())
    j_real = float((sol["price_yuan_per_kwh"] * sol["grid_purchase_energy_kwh"]).sum())
    assert len(dual) == 288, "原始对偶数据应为 288 行"

    c_arr = sol["price_yuan_per_kwh"].to_numpy(float)
    g_arr = sol["grid_purchase_energy_kwh"].to_numpy(float)
    C_arr = sol["charge_energy_kwh"].to_numpy(float)
    D_arr = sol["discharge_energy_kwh"].to_numpy(float)
    U_arr = sol["unused_pv_energy_kwh"].to_numpy(float)
    pi, lam, raw_bal, raw_sto, decision = extract_duals(
        dual, c_arr, g_arr, C_arr, D_arr, U_arr)
    df, arr = build_table(sol, pi, lam)
    price_stats = compute_price_stats(arr)
    arb = compute_arbitrage(arr)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig", float_format="%.10g")
    t_end_h = write_plot_data(sol, arr)

    print(f"[17] 对偶提取完成：π∈[{price_stats['pi_min']:.6f},{price_stats['pi_max']:.6f}]，"
          f"λ∈[{price_stats['lam_min']:.6f},{price_stats['lam_max']:.6f}]")
    print(f"[17] π = {decision['pi_chosen']}（违例 {decision['pi_violation']:.2e} vs 备选 {decision['pi_alt_violation']:.2e}）")
    print(f"[17] λ = {decision['lambda_chosen']}（违例 {decision['lambda_violation']:.2e} vs 备选 {decision['lambda_alt_violation']:.2e}）")
    print(f"[17] 已导出 {OUT_TABLE.name} / {OUT_PLOT_DATA.name}")
    return {
        "sol": sol, "dual": dual, "rowmap": rowmap, "summary": summary,
        "pi": pi, "lam": lam, "raw_bal": raw_bal, "raw_sto": raw_sto,
        "decision": decision, "df": df, "arr": arr,
        "price_stats": price_stats, "arb": arb,
        "t_end_h": t_end_h, "j": j, "j_real": j_real, "q_real": q_real,
    }


if __name__ == "__main__":
    main()
