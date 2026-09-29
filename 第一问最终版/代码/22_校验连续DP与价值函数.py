from pathlib import Path
import sys
import importlib.util
import hashlib

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parent
RESULT_DIR = PROJECT / "模型结果"
sys.path.insert(0, str(CODE_DIR))

J_LOCK = 35126.9485892896
Q_LOCK = 59482.6989983539
E_TERM = 6000.0
SRC_SHA_LOCK = "6c1bcd91155d71ca5ae0f09762b40a6d5b98781454c6730bbf8fcafda17ced0a"

TOL_COST = 1e-5
TOL_PURCHASE = 1e-4
TOL_BELLMAN = 1e-3
TOL_CONVEX = 1e-3
TOL_RESID = 1e-7

OUT_MUTUAL = RESULT_DIR / "连续DP_LP互证.csv"
OUT_BELLMAN = RESULT_DIR / "连续DP_贝尔曼残差.csv"
OUT_ACCEPT = RESULT_DIR / "连续DP_性质验收.csv"


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, str(CODE_DIR / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def golden_min(func, a, b, tol=1e-7, maxiter=400):
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c)
    fd = func(d)
    for _ in range(maxiter):
        if (b - a) <= tol:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = func(d)
    return min(fc, fd)


def bellman_rhs(m21, price, L, V, n, t, e, next_knots):
    ETA, S = m21.ETA, m21.S
    S_OVER_ETA, ETA_S = m21.S_OVER_ETA, m21.ETA_S
    if t == m21.T:
        lo_n, hi_n = m21.reachable_domain(t + 1)
        lo = max(lo_n, e - S_OVER_ETA)
        hi = min(hi_n, e + ETA_S)
        if lo > hi + 1e-9:
            return m21.INF
        ep = 6000.0
        if ep < lo - 1e-9 or ep > hi + 1e-9:
            return m21.INF
        x = ep - e
        return price[t - 1] * max(n[t - 1] + m21.psi(x), 0.0)

    lo_n, hi_n = m21.reachable_domain(t + 1)
    lo = max(lo_n, e - S_OVER_ETA)
    hi = min(hi_n, e + ETA_S)
    if lo > hi + 1e-9:
        return m21.INF

    en = np.array([p[0] for p in next_knots], dtype=float)
    Fn = np.array([p[1] for p in next_knots], dtype=float)
    c_t = price[t - 1]
    n_t = n[t - 1]

    def obj(ep):
        x = ep - e
        return c_t * max(n_t + m21.psi(x), 0.0) + float(np.interp(ep, en, Fn))

    return golden_min(obj, lo, hi)


def main(ctx=None, m21=None):
    sha_before = sha256(RESULT_DIR / "最终LP_原始最优解.csv")
    if m21 is None:
        m21 = load_mod("21_构造连续状态价值函数")
    if ctx is None:
        ctx = m21.main()

    price = ctx["price"]
    L = ctx["L"]
    V = ctx["V"]
    n = ctx["n"]
    cpl_cache = ctx["cpl_cache"]
    traj_df = ctx["traj_df"]
    E_end = ctx["E_end"]
    REP_TIMES = ctx["REP_TIMES"]

    F1 = m21.tail_lp_value(price, L, V, 1, m21.E_INIT)
    cost_err = abs(F1 - J_LOCK)
    cost_pass = cost_err <= TOL_COST

    g = traj_df["g"].to_numpy(float)
    C = traj_df["C"].to_numpy(float)
    D = traj_df["D"].to_numpy(float)
    U = traj_df["U"].to_numpy(float)
    E_prev = traj_df["E_prev"].to_numpy(float)
    E_next = traj_df["E_next"].to_numpy(float)

    g_sum = float(g.sum())
    q_err = abs(g_sum - Q_LOCK)
    q_pass = q_err <= TOL_PURCHASE

    traj_cost = float((g * price).sum())
    traj_cost_err = abs(traj_cost - J_LOCK)

    balance_resid = np.abs(L + C + U - g - V - D).max()
    storage_resid = np.abs(E_next - E_prev - m21.ETA * C + D / m21.ETA).max()
    min_g = float(g.min())
    min_U = float(U.min())
    max_CD_over = max(float(C.max() - m21.S), float(D.max() - m21.S), 0.0)
    E_term_err = abs(float(E_end) - E_TERM)

    next_cache = {}
    for t, _ in REP_TIMES:
        if t < m21.T:
            next_cache[t] = m21.build_cpl(price, L, V, t + 1)

    bell_rows = []
    max_bell = 0.0
    invalid_bell = 0
    max_bell_info = ""
    for t, label in REP_TIMES:
        knots = cpl_cache[t]
        lo, hi = m21.reachable_domain(t)
        nk = next_cache.get(t)
        e_test = [p[0] for p in knots] + [0.5 * (lo + hi)]
        e_test = sorted(set(round(float(e), 10) for e in e_test))
        for e in e_test:
            if e < lo - 1e-9 or e > hi + 1e-9:
                continue
            lhs = m21.tail_lp_value(price, L, V, t, e)
            rhs = bellman_rhs(m21, price, L, V, n, t, e, nk)
            resid = float(np.inf) if (lhs >= m21.INF - 1 or rhs >= m21.INF - 1) else abs(lhs - rhs)
            bell_rows.append({
                "t": t, "代表时刻": label, "e_kwh": e,
                "F_t_e_yuan": lhs, "rhs_yuan": rhs, "residual_yuan": resid,
            })
            if not np.isfinite(resid):
                invalid_bell += 1
                max_bell = float("inf")
            elif resid > max_bell:
                max_bell = resid
                max_bell_info = f"t={t}({label}), e={e:.6f}"

    max_conv = 0.0
    max_conv_info = ""
    knot_count = {}
    bp_count = {}
    BP_SLOPE_TOL = 1e-4
    for t, label in REP_TIMES:
        knots = cpl_cache[t]
        knot_count[t] = len(knots)
        e = np.array([p[0] for p in knots], dtype=float)
        F = np.array([p[1] for p in knots], dtype=float)
        s = (F[1:] - F[:-1]) / (e[1:] - e[:-1])
        bp_count[t] = int((np.abs(np.diff(s)) > BP_SLOPE_TOL).sum()) if len(s) >= 2 else 0
        if len(s) >= 2:
            gap = s[:-1] - s[1:]
            if gap.max() > max_conv:
                max_conv = float(gap.max())
                i = int(gap.argmax())
                max_conv_info = f"t={t}({label}), e∈[{e[i]:.6f},{e[i+1]:.6f}]→[{e[i+1]:.6f},{e[i+2]:.6f}]"

    I_Tp1_lo, I_Tp1_hi = m21.reachable_domain(m21.T + 1)
    I_Tp1_exact = (abs(I_Tp1_lo - 6000.0) < 1e-12 and abs(I_Tp1_hi - 6000.0) < 1e-12)

    domain_in_range = True
    max_width_grow = 0.0
    prev_hi, prev_lo = None, None
    for t in range(1, m21.T + 2):
        lo, hi = m21.reachable_domain(t)
        if lo < m21.E_MIN - 1e-9 or hi > m21.E_MAX + 1e-9:
            domain_in_range = False
        if prev_hi is not None and (hi - lo) - (prev_hi - prev_lo) > max_width_grow:
            max_width_grow = (hi - lo) - (prev_hi - prev_lo)
        prev_lo, prev_hi = lo, hi

    lo_last, hi_last = m21.reachable_domain(m21.T)
    last_err = max(abs(m21.tail_lp_value(price,L,V,m21.T,e) - m21.period_cost(6000-e,price[-1],n[-1])) for e in np.linspace(lo_last,hi_last,21))
    sol_sha = sha256(RESULT_DIR / "最终LP_原始最优解.csv")
    sha_unchanged = sol_sha == sha_before

    checks = [
        ("整日LP与后续时域LP费用一致性 F_1(6000)=J_LP", cost_err, TOL_COST, cost_pass,
         f"F_1(6000)={F1:.10f} vs {J_LOCK}"),
        ("DP轨迹总购电量与LP一致", q_err, TOL_PURCHASE, q_pass,
         f"Σg={g_sum:.10f} vs {Q_LOCK}"),
        ("DP轨迹目标值与LP一致", traj_cost_err, TOL_PURCHASE, traj_cost_err <= TOL_PURCHASE,
         f"Σ c_t g_t={traj_cost:.8f} vs {J_LOCK}"),
        ("Bellman递推最大残差", max_bell, TOL_BELLMAN, invalid_bell == 0 and max_bell <= TOL_BELLMAN,
         max_bell_info),
        ("凸性（割线斜率单调不减）", max_conv, TOL_CONVEX, max_conv <= TOL_CONVEX,
         max_conv_info),
        ("边际价值单调不增（λ=-斜率）", max_conv, TOL_CONVEX, max_conv <= TOL_CONVEX,
         "与凸性同源：λ非增 ⟺ 割线斜率不减"),
        ("终端条件精确执行 E_144=6000", E_term_err, 1e-6, E_term_err <= 1e-6,
         f"E_144={float(E_end):.10f}"),
        ("可达状态域不超设备范围", 0.0, 0.0, domain_in_range,
         "1200≤I_t≤10800 恒成立"),
        ("可达域临近终点收缩且 I_{T+1}={6000}", max(0.0, max_width_grow), 0.0,
         I_Tp1_exact and max_width_grow <= 0.0,
         f"I_{{T+1}}=[{I_Tp1_lo:.6f},{I_Tp1_hi:.6f}], 宽度最大回弹={max_width_grow:.3e}"),
        ("轨迹可行（供需平衡残差）", float(balance_resid), TOL_RESID, float(balance_resid) <= TOL_RESID,
         "max|L+C+U-g-V-D|"),
        ("轨迹可行（储能递推残差）", float(storage_resid), TOL_RESID, float(storage_resid) <= TOL_RESID,
         "max|E_t-E_{t-1}-ηC+D/η|"),
        ("全程无售电（g_t≥0 恒成立）", max(0.0, -min_g), 1e-8, min_g >= -1e-8,
         f"min g={min_g:.3e}（模型无售电变量）"),
        ("弃光非负 U_t≥0", max(0.0, -min_U), 1e-8, min_U >= -1e-8,
         f"min U={min_U:.3e}"),
        ("充放电量不超过 S", max_CD_over, 1e-7, max_CD_over <= 1e-7,
         f"max C={C.max():.6f}, max D={D.max():.6f}, S={m21.S:.6f}"),
        ("原始LP文件SHA-256未改动", 0.0, 0.0, sha_unchanged,
         f"SHA-256={sol_sha[:16]}…"),
    ]
    checks.extend([
        ("Bellman测试点均有限", invalid_bell, 0, invalid_bell == 0, "可达测试点出现非有限残差即失败"),
        ("最后时段解析解与LP一致", float(last_err), 1e-7, last_err <= 1e-7, "21个状态含有效域两端"),
        ("轨迹状态范围", 0.0, 1e-7, bool(np.all((E_prev>=m21.E_MIN-1e-7)&(E_prev<=m21.E_MAX+1e-7)&(E_next>=m21.E_MIN-1e-7)&(E_next<=m21.E_MAX+1e-7))), "首末及所有中间状态"),
        ("充放电非负", max(0.0,-float(C.min()),-float(D.min())), 1e-7, bool(C.min()>=-1e-7 and D.min()>=-1e-7), "C,D非负"),
        ("轨迹相邻状态连续", float(np.max(abs(E_prev[1:]-E_next[:-1]))),1e-7,bool(np.max(abs(E_prev[1:]-E_next[:-1]))<=1e-7), "前一时段终态等于后一时段初态"),
    ])
    accept_df = pd.DataFrame(checks, columns=["check_name", "max_violation", "tolerance", "pass", "explanation"])
    n_pass = int(accept_df["pass"].sum())
    n_total = len(accept_df)

    mut_df = pd.DataFrame([
        {"metric": "最优费用 F_1(6000)", "dp_value": F1, "lp_value": J_LOCK,
         "abs_error": cost_err, "tolerance": TOL_COST, "pass": cost_pass},
        {"metric": "总购电量 Σg_t", "dp_value": g_sum, "lp_value": Q_LOCK,
         "abs_error": q_err, "tolerance": TOL_PURCHASE, "pass": q_pass},
        {"metric": "轨迹费用 Σc_t g_t", "dp_value": traj_cost, "lp_value": J_LOCK,
         "abs_error": traj_cost_err, "tolerance": TOL_PURCHASE, "pass": traj_cost_err <= TOL_PURCHASE},
    ])

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mut_df.to_csv(OUT_MUTUAL, index=False, encoding="utf-8-sig", float_format="%.12g")
    pd.DataFrame(bell_rows).to_csv(OUT_BELLMAN, index=False, encoding="utf-8-sig", float_format="%.10g")
    accept_df.to_csv(OUT_ACCEPT, index=False, encoding="utf-8-sig", float_format="%.10g")

    return {
        "F1": F1, "cost_err": cost_err, "cost_pass": cost_pass,
        "g_sum": g_sum, "q_err": q_err, "q_pass": q_pass,
        "traj_cost": traj_cost, "traj_cost_err": traj_cost_err,
        "balance_resid": float(balance_resid), "storage_resid": float(storage_resid),
        "min_g": min_g, "min_U": min_U, "max_C": float(C.max()), "max_D": float(D.max()),
        "max_CD_over": max_CD_over, "E_term_err": E_term_err, "E_end": float(E_end),
        "max_bell": max_bell, "max_bell_info": max_bell_info,
        "max_conv": max_conv, "max_conv_info": max_conv_info,
        "I_Tp1": (I_Tp1_lo, I_Tp1_hi), "I_Tp1_exact": I_Tp1_exact,
        "domain_in_range": domain_in_range, "max_width_grow": max_width_grow,
        "knot_count": knot_count, "bp_count": bp_count, "bp_tol": BP_SLOPE_TOL,
        "sol_sha": sol_sha, "sha_unchanged": sha_unchanged,
        "accept_df": accept_df, "n_pass": n_pass, "n_total": n_total,
        "all_pass": n_pass == n_total,
        "bell_tol": TOL_BELLMAN, "conv_tol": TOL_CONVEX,
    }


if __name__ == "__main__":
    r = main()
    print(f"[22] 验收 {r['n_pass']}/{r['n_total']} 通过")
    print(f"     F_1(6000)={r['F1']:.10f}, 与LP误差={r['cost_err']:.2e}")
    print(f"     轨迹总购电={r['g_sum']:.10f}, 误差={r['q_err']:.2e}")
    print(f"     Bellman最大残差={r['max_bell']:.3e}, 最大凸性违例={r['max_conv']:.3e}")
