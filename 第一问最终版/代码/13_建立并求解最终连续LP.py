from pathlib import Path
import re
import time
import hashlib
import csv

import sys
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
import scipy

PROJECT = Path(__file__).resolve().parent.parent
CSV_BASE = PROJECT / "处理后数据" / "附件一_第一问基础数据.csv"
CONFIG = PROJECT / "配置" / "第一问数据处理配置.yaml"
RESULT_DIR = PROJECT / "模型结果"
MANIFEST = PROJECT / "历史模型" / "MILP历史基准文件清单.csv"

N = 144
DT = 1.0 / 6.0
ETA_C = 0.9
ETA_D = 0.9
E_MIN = 1200.0
E_MAX = 10800.0
E_INIT = 6000.0
E_TERM = 6000.0
P_CHARGE_MAX = 5000.0
P_DISCHARGE_MAX = 5000.0
S = P_CHARGE_MAX * DT

FORMAL = ["interval_index", "original_time_label", "interval_start", "interval_end",
          "delta_hours", "price_yuan_per_kwh", "load_energy_kwh", "pv_forecast_energy_kwh"]

G0, C0, D0, U0, E0 = 0, N, 2 * N, 3 * N, 4 * N
NVAR = 4 * N + (N + 1)
N_EQ = 2 * N

OUT_SOLUTION = RESULT_DIR / "最终LP_原始最优解.csv"
OUT_SUMMARY = RESULT_DIR / "最终LP_求解摘要.csv"
OUT_ROW_MAP = RESULT_DIR / "最终LP_约束行映射.csv"
OUT_DUAL_EQ = RESULT_DIR / "最终LP_原始对偶数据.csv"
OUT_DUAL_BD = RESULT_DIR / "最终LP_变量边界对偶数据.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def read_config():
    d = {}
    section = None
    for raw in open(CONFIG, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip() or line.strip().startswith("#"):
            continue
        s = line.lstrip()
        ind = len(line) - len(s)
        if ind == 0:
            section = s.rstrip(":").strip()
            d[section] = {}
        else:
            k, _, v = s.partition(":")
            d[section][k.strip()] = v.strip().strip('"')
    return d


def detect_highs_version():
    import re as _re
    code = ("import scipy.optimize as so; "
            "so.linprog(c=[1.0], bounds=[(0, None)], method='highs', "
            "options={'disp': True})")
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=60)
        m = _re.search(r"HiGHS\s+(\d+\.\d+\.\d+)", out.stdout + out.stderr)
        return m.group(1) if m else "1.8.0"
    except Exception:
        return "1.8.0"


def load_input():
    df = pd.read_csv(CSV_BASE, encoding="utf-8-sig")
    cur_sha = sha256(CSV_BASE)
    if MANIFEST.is_file():
        frozen_sha = None
        with open(MANIFEST, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["原路径"] == "处理后数据/附件一_第一问基础数据.csv":
                    frozen_sha = r["SHA-256"]
        assert frozen_sha is not None, "历史冻结清单存在，但未登记清洗CSV"
        assert cur_sha == frozen_sha, \
            f"清洗CSV SHA-256 与第一阶段冻结值不一致：{cur_sha} != {frozen_sha}"
    else:
        print("[SKIP] 未提供开发期历史冻结清单；仅记录本次清洗CSV的SHA-256。")

    assert len(df) == N, f"CSV 应为 {N} 行，实际 {len(df)}"
    assert list(df["interval_index"]) == list(range(1, N + 1)), "interval_index 应为 1..144"
    assert not df[FORMAL].isna().any().any(), "正式输入字段存在缺失"
    d = df["delta_hours"].to_numpy(float)
    assert np.max(np.abs(d - DT)) < 1e-12, "delta_hours 应全部为 1/6"
    assert np.max(np.abs(df["load_energy_kwh"] - df["load_kw"] * DT)) < 1e-8, \
        "load_energy_kwh 应等于 load_kw×Δt"
    assert np.max(np.abs(df["pv_forecast_energy_kwh"] - df["pv_forecast_kw"] * DT)) < 1e-8, \
        "pv_forecast_energy_kwh 应等于 pv_forecast_kw×Δt"
    return df, cur_sha


def build_lp(price, L, V):
    c = np.zeros(NVAR)
    c[G0:G0 + N] = price

    lb = np.zeros(NVAR)
    ub = np.full(NVAR, np.inf)
    lb[C0:C0 + N] = 0.0
    ub[C0:C0 + N] = S
    lb[D0:D0 + N] = 0.0
    ub[D0:D0 + N] = S
    lb[E0:E0 + N + 1] = E_MIN
    ub[E0:E0 + N + 1] = E_MAX
    lb[E0] = E_INIT
    ub[E0] = E_INIT
    lb[E0 + N] = E_TERM
    ub[E0 + N] = E_TERM

    rows, cols, vals = [], [], []
    b_eq = np.zeros(N_EQ)
    for t in range(N):
        r = t
        rows += [r, r, r, r]
        cols += [G0 + t, C0 + t, D0 + t, U0 + t]
        vals += [-1.0, 1.0, -1.0, 1.0]
        b_eq[r] = V[t] - L[t]
    for t in range(N):
        r = N + t
        rows += [r, r, r, r]
        cols += [E0 + t + 1, E0 + t, C0 + t, D0 + t]
        vals += [1.0, -1.0, -ETA_C, 1.0 / ETA_D]
        b_eq[r] = 0.0

    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(N_EQ, NVAR)).tocsr()
    return c, A_eq, b_eq, lb, ub


def extract_solution(x):
    g = x[G0:G0 + N].astype(float)
    C = x[C0:C0 + N].astype(float)
    D = x[D0:D0 + N].astype(float)
    U = x[U0:U0 + N].astype(float)
    E = x[E0:E0 + N + 1].astype(float)
    return g, C, D, U, E


def build_solution_df(df, g, C, D, U, E):
    price = df["price_yuan_per_kwh"].to_numpy(float)
    L = df["load_energy_kwh"].to_numpy(float)
    V = df["pv_forecast_energy_kwh"].to_numpy(float)
    TOL = 1e-7
    rows = []
    for t in range(N):
        r_bal = g[t] + V[t] + D[t] - L[t] - C[t] - U[t]
        r_energy = E[t + 1] - E[t] - ETA_C * C[t] + D[t] / ETA_D
        rows.append({
            "interval_index": int(df["interval_index"].iloc[t]),
            "original_time_label": df["original_time_label"].iloc[t],
            "interval_start": df["interval_start"].iloc[t],
            "interval_end": df["interval_end"].iloc[t],
            "price_yuan_per_kwh": price[t],
            "load_energy_kwh": L[t],
            "pv_forecast_energy_kwh": V[t],
            "grid_purchase_energy_kwh": g[t],
            "charge_energy_kwh": C[t],
            "discharge_energy_kwh": D[t],
            "unused_pv_energy_kwh": U[t],
            "energy_start_kwh": E[t],
            "energy_end_kwh": E[t + 1],
            "grid_purchase_power_kw": g[t] / DT,
            "charge_power_kw": C[t] / DT,
            "discharge_power_kw": D[t] / DT,
            "unused_pv_power_kw": U[t] / DT,
            "period_purchase_cost_yuan": price[t] * g[t],
            "power_balance_residual_kwh": r_bal,
            "energy_transition_residual_kwh": r_energy,
            "simultaneous_charge_discharge_flag": int(C[t] > TOL and D[t] > TOL),
        })
    return pd.DataFrame(rows)


def build_row_map():
    rows = []
    for t in range(N):
        rows.append({
            "row_index": t + 1,
            "constraint_type": "供需平衡",
            "interval_index": t + 1,
            "constraint_name": f"balance_t{t + 1}",
            "equation_orientation": "L_t + C_t + U_t - g_t - V_t - D_t = 0  (即 -g_t+C_t-D_t+U_t = V_t-L_t)",
            "unit": "kWh",
        })
    for t in range(N):
        rows.append({
            "row_index": N + t + 1,
            "constraint_type": "储能递推",
            "interval_index": t + 1,
            "constraint_name": f"soc_transition_t{t + 1}",
            "equation_orientation": "E_t - E_{t-1} - eta_c*C_t + D_t/eta_d = 0",
            "unit": "kWh",
        })
    return pd.DataFrame(rows)


def build_dual_eq(res):
    eq = getattr(res, "eqlin", None)
    marg = np.asarray(getattr(eq, "marginals", [np.nan] * N_EQ)) if eq is not None else np.full(N_EQ, np.nan)
    resid = np.asarray(getattr(eq, "residual", [np.nan] * N_EQ)) if eq is not None else np.full(N_EQ, np.nan)
    rows = []
    for i in range(N_EQ):
        if i < N:
            ctype, tint = "供需平衡", i + 1
            orient = "L_t + C_t + U_t - g_t - V_t - D_t = 0"
        else:
            ctype, tint = "储能递推", i - N + 1
            orient = "E_t - E_{t-1} - eta_c*C_t + D_t/eta_d = 0"
        rows.append({
            "constraint_row": i + 1,
            "constraint_type": ctype,
            "interval_index": tint,
            "raw_marginal": float(marg[i]),
            "residual": float(resid[i]),
            "equation_orientation": orient,
            "solver": "HiGHS (scipy.optimize.linprog)",
        })
    return pd.DataFrame(rows)


def build_dual_bounds(res, lb, ub):
    lo = getattr(res, "lower", None)
    up = getattr(res, "upper", None)
    lo_m = np.asarray(getattr(lo, "marginals", [np.nan] * NVAR)) if lo is not None else np.full(NVAR, np.nan)
    up_m = np.asarray(getattr(up, "marginals", [np.nan] * NVAR)) if up is not None else np.full(NVAR, np.nan)
    lo_r = np.asarray(getattr(lo, "residual", [np.nan] * NVAR)) if lo is not None else np.full(NVAR, np.nan)
    up_r = np.asarray(getattr(up, "residual", [np.nan] * NVAR)) if up is not None else np.full(NVAR, np.nan)

    def desc(i):
        if i < C0:
            return "g_t", i - G0 + 1
        if i < D0:
            return "C_t", i - C0 + 1
        if i < U0:
            return "D_t", i - D0 + 1
        if i < E0:
            return "U_t", i - U0 + 1
        return "E_t", i - E0

    rows = []
    for i in range(NVAR):
        grp, tint = desc(i)
        rows.append({
            "var_index": i,
            "var_group": grp,
            "interval_index": tint,
            "lower_bound": float(lb[i]),
            "upper_bound": float(ub[i]),
            "lower_marginal": float(lo_m[i]),
            "upper_marginal": float(up_m[i]),
            "lower_residual": float(lo_r[i]),
            "upper_residual": float(up_r[i]),
        })
    return pd.DataFrame(rows)


def main():
    cfg = read_config()
    op = cfg.get("operation", {})
    df, cur_sha = load_input()

    price = df["price_yuan_per_kwh"].to_numpy(float)
    L = df["load_energy_kwh"].to_numpy(float)
    V = df["pv_forecast_energy_kwh"].to_numpy(float)

    c, A_eq, b_eq, lb, ub = build_lp(price, L, V)

    highs_ver = detect_highs_version()

    t0 = time.perf_counter()
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    elapsed = time.perf_counter() - t0

    if not res.success:
        raise RuntimeError(f"LP 求解失败：status={res.status}, message={res.message}")

    g, C, D, U, E = extract_solution(res.x)
    sol_df = build_solution_df(df, g, C, D, U, E)

    summary = pd.DataFrame([
        ("模型类型", "连续线性规划（LP）", "-"),
        ("求解器", "scipy.optimize.linprog(method='highs')", "-"),
        ("scipy版本", scipy.__version__, "-"),
        ("HiGHS版本", highs_ver, "-"),
        ("求解状态码", int(res.status), "-"),
        ("求解消息", str(res.message), "-"),
        ("求解成功", bool(res.success), "-"),
        ("求解耗时_秒", f"{elapsed:.6f}", "s"),
        ("迭代次数", int(getattr(res, "nit", -1)), "次"),
        ("最优目标值_元", f"{float(res.fun):.12f}", "元"),
        ("变量数", NVAR, "个"),
        ("等式约束数", N_EQ, "条"),
        ("等式约束构成", "供需平衡144 + 储能递推144", "-"),
    ], columns=["指标", "数值", "单位"])

    totals = {
        "grid": float(g.sum()),
        "charge": float(C.sum()),
        "discharge": float(D.sum()),
        "unused_pv": float(U.sum()),
        "E_min": float(E.min()),
        "E_max": float(E.max()),
        "max_charge_energy": float(C.max()),
        "max_discharge_energy": float(D.max()),
        "max_charge_power": float(C.max()) / DT,
        "max_discharge_power": float(D.max()) / DT,
        "max_balance_resid": float(np.max(np.abs(sol_df["power_balance_residual_kwh"]))),
        "max_energy_resid": float(np.max(np.abs(sol_df["energy_transition_residual_kwh"]))),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sol_df.to_csv(OUT_SOLUTION, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    build_row_map().to_csv(OUT_ROW_MAP, index=False, encoding="utf-8-sig")
    build_dual_eq(res).to_csv(OUT_DUAL_EQ, index=False, encoding="utf-8-sig")
    build_dual_bounds(res, lb, ub).to_csv(OUT_DUAL_BD, index=False, encoding="utf-8-sig")

    result = {
        "cfg": cfg,
        "op": op,
        "csv_sha": cur_sha,
        "highs_version": highs_ver,
        "scipy_version": scipy.__version__,
        "status_code": int(res.status),
        "message": str(res.message),
        "success": bool(res.success),
        "elapsed": float(elapsed),
        "nit": int(getattr(res, "nit", -1)),
        "objective": float(res.fun),
        "n_var": NVAR,
        "n_eq": N_EQ,
        "totals": totals,
        "sol_df": sol_df,
    }
    print(f"[13] 求解状态：{result['message']}")
    print(f"[13] J_1 = {result['objective']:.10f} 元")
    print(f"[13] 耗时 = {elapsed:.5f} s，迭代 = {result['nit']}，HiGHS = {highs_ver}")
    print(f"[13] 已导出：原始最优解 / 求解摘要 / 约束行映射 / 原始对偶数据 / 变量边界对偶数据")
    return result


if __name__ == "__main__":
    main()
