from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse

PROJECT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT / "模型结果"
SOLUTION_CSV = RESULT_DIR / "最终LP_原始最优解.csv"

T = 144
ETA = 0.9
S = 5000.0 / 6.0
ETA_S = ETA * S
S_OVER_ETA = S / ETA
E_MIN, E_MAX = 1200.0, 10800.0
E_INIT = E_TERM = 6000.0

LP_TOL = 1e-7
SPLIT_TOL = 1e-6
INF = 1e18

REP_TIMES = [
    (1, "0:00"),
    (37, "6:00"),
    (73, "12:00"),
    (109, "18:00"),
    (144, "23:50（临近24:00）"),
]

OUT_BREAK = RESULT_DIR / "连续DP_价值函数折点.csv"
OUT_MARG = RESULT_DIR / "连续DP_边际价值.csv"

QUERY_COUNT = 0


def reset_query_count():
    global QUERY_COUNT
    QUERY_COUNT = 0


def read_inputs():
    sol = pd.read_csv(SOLUTION_CSV, encoding="utf-8-sig")
    price = sol["price_yuan_per_kwh"].to_numpy(float)
    L = sol["load_energy_kwh"].to_numpy(float)
    V = sol["pv_forecast_energy_kwh"].to_numpy(float)
    return price, L, V


def reachable_domain(t):
    h = T - t + 1
    lo = max(E_MIN, E_INIT - h * ETA_S)
    hi = min(E_MAX, E_INIT + h * S_OVER_ETA)
    return lo, hi


def psi(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(x / ETA, ETA * x)


def purchase(x, n_t):
    return np.maximum(n_t + psi(x), 0.0)


def period_cost(x, c_t, n_t):
    return c_t * purchase(x, n_t)


def _build_tail_lp(price, L, V, t, e):
    m = T - t + 1
    G0, C0, D0, U0, E0 = 0, m, 2 * m, 3 * m, 4 * m
    nvar = 5 * m

    c = np.zeros(nvar)
    c[G0:G0 + m] = price[t - 1:T]

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    lb[C0:C0 + m] = 0.0
    ub[C0:C0 + m] = S
    lb[D0:D0 + m] = 0.0
    ub[D0:D0 + m] = S
    lb[E0:E0 + m] = E_MIN
    ub[E0:E0 + m] = E_MAX
    lb[E0 + m - 1] = E_TERM
    ub[E0 + m - 1] = E_TERM

    n_eq = 2 * m
    rows, cols, vals = [], [], []
    b_eq = np.zeros(n_eq)

    for k in range(m):
        r = k
        rows += [r, r, r, r]
        cols += [G0 + k, C0 + k, D0 + k, U0 + k]
        vals += [-1.0, 1.0, -1.0, 1.0]
        b_eq[r] = V[t - 1 + k] - L[t - 1 + k]

    for k in range(m):
        r = m + k
        rows += [r, r, r]
        cols += [E0 + k, C0 + k, D0 + k]
        vals += [1.0, -ETA, 1.0 / ETA]
        if k == 0:
            b_eq[r] = e
        else:
            rows += [r]
            cols += [E0 + k - 1]
            vals += [-1.0]
            b_eq[r] = 0.0

    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(n_eq, nvar)).tocsr()
    return c, A_eq, b_eq, lb, ub, m


def tail_lp_value(price, L, V, t, e):
    global QUERY_COUNT
    QUERY_COUNT += 1
    lo, hi = reachable_domain(t)
    if e < lo - 1e-9 or e > hi + 1e-9:
        return INF
    if t == T + 1:
        return 0.0
    c, A_eq, b_eq, lb, ub, m = _build_tail_lp(price, L, V, t, e)
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(f"可达状态LP求解失败 t={t}, e={e}: {res.message}")
    return float(res.fun)


def tail_lp_solution(price, L, V, t, e):
    lo, hi = reachable_domain(t)
    if e < lo - 1e-9 or e > hi + 1e-9:
        return None
    if t == T + 1:
        return 0.0
    c, A_eq, b_eq, lb, ub, m = _build_tail_lp(price, L, V, t, e)
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        return None
    x = res.x
    g_t = float(x[0])
    C_t = float(x[m])
    D_t = float(x[2 * m])
    U_t = float(x[3 * m])
    E_t = float(x[4 * m])
    return {
        "F": float(res.fun), "g": g_t, "C": C_t, "D": D_t, "U": U_t,
        "E_next": E_t, "x": E_t - e,
    }


def build_cpl(price, L, V, t, tol=SPLIT_TOL, max_depth=55, min_width=1e-4):
    lo, hi = reachable_domain(t)
    if t == T:
        net = L[-1] - V[-1]
        crossing = E_TERM + (net / ETA if net >= 0 else ETA * net)
        states = sorted(set([lo, hi] + [v for v in (E_TERM, crossing) if lo < v < hi]))
        return [(e, float(period_cost(E_TERM-e, price[-1], net))) for e in states]
    f_lo = tail_lp_value(price, L, V, t, lo)
    f_hi = tail_lp_value(price, L, V, t, hi)

    pts = [(lo, f_lo), (hi, f_hi)]
    stack = [(lo, f_lo, hi, f_hi, 0)]
    while stack:
        a, fa, b, fb, depth = stack.pop()
        if (b - a) <= min_width:
            continue
        m = 0.5 * (a + b)
        fm = tail_lp_value(price, L, V, t, m)
        if abs(fm - 0.5 * (fa + fb)) > tol and depth < max_depth:
            stack.append((a, fa, m, fm, depth + 1))
            stack.append((m, fm, b, fb, depth + 1))
            pts.append((m, fm))

    pts.sort(key=lambda p: p[0])
    out = []
    for e, f in pts:
        if out and abs(e - out[-1][0]) <= 1e-9:
            continue
        out.append((e, f))
    return out


def knot_slopes(knots):
    e = np.array([p[0] for p in knots], dtype=float)
    F = np.array([p[1] for p in knots], dtype=float)
    n = len(knots)
    slope_L = np.full(n, np.nan)
    slope_R = np.full(n, np.nan)
    for i in range(1, n):
        s = (F[i] - F[i - 1]) / (e[i] - e[i - 1])
        slope_L[i] = s
        slope_R[i - 1] = s
    return e, F, slope_L, slope_R


def recover_trajectory(price, L, V):
    e = E_INIT
    rows = []
    for t in range(1, T + 1):
        dec = tail_lp_solution(price, L, V, t, e)
        assert dec is not None, f"t={t} 状态 {e:.6f} 不可达"
        rows.append({
            "t": t,
            "E_prev": e,
            "E_next": dec["E_next"],
            "x": dec["x"],
            "g": dec["g"],
            "C": dec["C"],
            "D": dec["D"],
            "U": dec["U"],
        })
        e = dec["E_next"]
    return pd.DataFrame(rows), e


def main():
    price, L, V = read_inputs()
    n = L - V

    break_rows = []
    marg_rows = []
    cpl_cache = {}
    for t, label in REP_TIMES:
        knots = build_cpl(price, L, V, t)
        cpl_cache[t] = knots
        e, F, slope_L, slope_R = knot_slopes(knots)
        for i in range(len(knots)):
            is_bp = (i > 0 and i < len(knots) - 1 and
                     np.isfinite(slope_L[i]) and np.isfinite(slope_R[i]) and
                     abs(slope_L[i] - slope_R[i]) > 1e-4)
            break_rows.append({
                "t": t, "代表时刻": label,
                "e_kwh": e[i], "F_yuan": F[i],
                "slope_left": float(slope_L[i]) if np.isfinite(slope_L[i]) else np.nan,
                "slope_right": float(slope_R[i]) if np.isfinite(slope_R[i]) else np.nan,
                "is_slope_change_sample": bool(is_bp),
            })
        for i in range(len(knots) - 1):
            s = (F[i + 1] - F[i]) / (e[i + 1] - e[i])
            marg_rows.append({
                "t": t, "代表时刻": label,
                "e_from": e[i], "e_to": e[i + 1],
                "slope": float(s), "lambda": float(-s),
            })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(break_rows).to_csv(OUT_BREAK, index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(marg_rows).to_csv(OUT_MARG, index=False, encoding="utf-8-sig", float_format="%.10g")

    traj_df, E_end = recover_trajectory(price, L, V)

    print(f"[21] 价值函数折点已导出 {OUT_BREAK.name} / {OUT_MARG.name}")
    for t, label in REP_TIMES:
        print(f"      t={t:3d} ({label:16s}) 采样点数={len(cpl_cache[t])}")
    print(f"[21] 轨迹恢复完成，终端 E_144={E_end:.6f}，Σg={traj_df['g'].sum():.10f} kWh")
    return {
        "price": price, "L": L, "V": V, "n": n,
        "cpl_cache": cpl_cache,
        "traj_df": traj_df, "E_end": E_end,
        "REP_TIMES": REP_TIMES,
    }


if __name__ == "__main__":
    main()
