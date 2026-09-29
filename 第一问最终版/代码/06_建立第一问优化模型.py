from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    PERIODS_PER_DAY, DELTA_HOURS,
    ETA_CHARGE, ETA_DISCHARGE, P_CHARGE_MAX, P_DISCHARGE_MAX,
    E_MIN, E_MAX, E_INITIAL, E_TERMINAL,
    load_formal_input, OPT_SCHEDULE_CSV,
)

N = PERIODS_PER_DAY
DT = DELTA_HOURS

TOL = 1e-7


def build_milp(price, load, pv):
    n = N
    nvar = 6 * n + 1
    g0, c0, d0, w0 = 0, n, 2 * n, 3 * n
    z0, e0 = 4 * n, 5 * n

    c = np.zeros(nvar)
    c[g0:g0 + n] = price * DT

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    lb[z0:z0 + n] = 0.0
    ub[z0:z0 + n] = 1.0
    lb[e0:e0 + n + 1] = E_MIN
    ub[e0:e0 + n + 1] = E_MAX

    integrality = np.zeros(nvar, dtype=int)
    integrality[z0:z0 + n] = 1

    rows, cols, vals = [], [], []
    lb_c, ub_c = [], []

    def add_row(entries, lo, hi):
        r = len(lb_c)
        for idx, v in entries:
            rows.append(r)
            cols.append(idx)
            vals.append(v)
        lb_c.append(lo)
        ub_c.append(hi)

    for t in range(n):
        add_row([(g0 + t, 1.0), (d0 + t, 1.0), (c0 + t, -1.0), (w0 + t, -1.0)],
                float(load[t] - pv[t]), float(load[t] - pv[t]))

    for t in range(n):
        add_row([(e0 + t + 1, 1.0), (e0 + t, -1.0),
                 (c0 + t, -ETA_CHARGE * DT), (d0 + t, DT / ETA_DISCHARGE)],
                0.0, 0.0)

    add_row([(e0, 1.0)], E_INITIAL, E_INITIAL)
    add_row([(e0 + n, 1.0)], E_TERMINAL, E_TERMINAL)

    for t in range(n):
        add_row([(c0 + t, 1.0), (z0 + t, -P_CHARGE_MAX)], -np.inf, 0.0)
        add_row([(d0 + t, 1.0), (z0 + t, P_DISCHARGE_MAX)], -np.inf, P_DISCHARGE_MAX)

    nrows = len(lb_c)
    A = coo_matrix((vals, (rows, cols)), shape=(nrows, nvar)).tocsr()
    lb_c = np.array(lb_c, dtype=float)
    ub_c = np.array(ub_c, dtype=float)

    idx = {"g0": g0, "c0": c0, "d0": d0, "w0": w0, "z0": z0, "e0": e0}
    return c, A, lb_c, ub_c, lb, ub, integrality, idx


def solve_milp(price, load, pv):
    c, A, lb_c, ub_c, lb, ub, integrality, idx = build_milp(price, load, pv)
    constraints = LinearConstraint(A, lb_c, ub_c)
    bounds = Bounds(lb, ub)
    t0 = time.perf_counter()
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints,
               options={"mip_rel_gap": 1e-9, "time_limit": 600.0, "disp": False})
    elapsed = time.perf_counter() - t0
    return res, elapsed, idx


def extract_solution(res, idx, price, load, pv):
    x = res.x
    g0, c0, d0, w0 = idx["g0"], idx["c0"], idx["d0"], idx["w0"]
    z0, e0 = idx["z0"], idx["e0"]
    n = N

    G = np.clip(x[g0:g0 + n].astype(float), 0.0, None)
    C = np.clip(x[c0:c0 + n].astype(float), 0.0, None)
    D = np.clip(x[d0:d0 + n].astype(float), 0.0, None)
    W = np.clip(x[w0:w0 + n].astype(float), 0.0, None)
    z = np.round(x[z0:z0 + n].astype(float)).astype(int)
    E = x[e0:e0 + n + 1].astype(float)

    G[G < TOL] = 0.0
    C[C < TOL] = 0.0
    D[D < TOL] = 0.0
    W[W < TOL] = 0.0
    E = np.clip(E, E_MIN, E_MAX)

    return {"G": G, "C": C, "D": D, "W": W, "z": z, "E": E}


def build_schedule_df(df_input, sol):
    price = df_input["price_yuan_per_kwh"].to_numpy(float)
    load = df_input["load_kw"].to_numpy(float)
    pv = df_input["pv_forecast_kw"].to_numpy(float)
    G, C, D, W, z, E = sol["G"], sol["C"], sol["D"], sol["W"], sol["z"], sol["E"]

    rows = []
    for t in range(N):
        Et, Etn = E[t], E[t + 1]
        r_power = G[t] + pv[t] + D[t] - load[t] - C[t] - W[t]
        r_soc = Etn - (Et + ETA_CHARGE * C[t] * DT - D[t] * DT / ETA_DISCHARGE)
        rows.append({
            "interval_index": int(df_input["interval_index"].iloc[t]),
            "original_time_label": df_input["original_time_label"].iloc[t],
            "interval_start": df_input["interval_start"].iloc[t],
            "interval_end": df_input["interval_end"].iloc[t],
            "delta_hours": float(df_input["delta_hours"].iloc[t]),
            "price_yuan_per_kwh": price[t],
            "load_kw": load[t],
            "pv_forecast_kw": pv[t],
            "grid_purchase_kw": G[t],
            "charge_power_kw": C[t],
            "discharge_power_kw": D[t],
            "pv_curtailment_kw": W[t],
            "charge_state": int(z[t]),
            "energy_start_kwh": Et,
            "energy_end_kwh": Etn,
            "grid_purchase_energy_kwh": G[t] * DT,
            "charge_input_energy_kwh": C[t] * DT,
            "battery_energy_increase_kwh": ETA_CHARGE * C[t] * DT,
            "discharge_output_energy_kwh": D[t] * DT,
            "battery_energy_decrease_kwh": D[t] * DT / ETA_DISCHARGE,
            "period_purchase_cost_yuan": price[t] * G[t] * DT,
            "power_balance_residual_kw": r_power,
            "soc_transition_residual_kwh": r_soc,
        })
    return pd.DataFrame(rows)


def run_solve():
    df = load_formal_input()
    price = df["price_yuan_per_kwh"].to_numpy(float)
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_forecast_kw"].to_numpy(float)

    res, elapsed, idx = solve_milp(price, load, pv)

    if res.status != 0 or not res.success:
        raise RuntimeError(
            f"MILP 求解未取得最优解：status={res.status}, success={res.success}, "
            f"message={res.message}"
        )

    sol = extract_solution(res, idx, price, load, pv)

    if not np.all(np.isin(sol["z"], [0, 1])):
        raise RuntimeError("充放电互斥二进制变量 z_t 存在非 0/1 取值")

    schedule = build_schedule_df(df, sol)
    schedule.to_csv(OPT_SCHEDULE_CSV, index=False, encoding="utf-8-sig",
                    float_format="%.10f")

    result = {
        "solver": "HiGHS MIP（scipy.optimize.milp 接口）",
        "solver_version": "",
        "status": res.message,
        "status_code": int(res.status),
        "success": bool(res.success),
        "objective": float(res.fun),
        "mip_gap": float(res.mip_gap),
        "mip_dual_bound": float(res.mip_dual_bound),
        "mip_node_count": int(getattr(res, "mip_node_count", -1)),
        "solve_time_s": float(elapsed),
        "schedule": schedule,
        "sol": sol,
        "df_input": df,
    }
    try:
        import scipy
        result["solver_version"] = f"scipy {scipy.__version__}"
    except Exception:
        result["solver_version"] = "scipy（版本未知）"

    print(f"[06] 求解状态：{result['status']}")
    print(f"[06] 目标函数 J_1 = {result['objective']:.8f} 元")
    print(f"[06] 求解耗时 = {result['solve_time_s']:.4f} s ; MIP Gap = {result['mip_gap']:.3e}")
    print(f"[06] 已导出：{OPT_SCHEDULE_CSV.name}")
    return result


if __name__ == "__main__":
    r = run_solve()
    print(f"[06] 完成，J_1={r['objective']:.8f}")
