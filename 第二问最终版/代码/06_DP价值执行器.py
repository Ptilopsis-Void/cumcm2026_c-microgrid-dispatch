from __future__ import annotations

import importlib.util
import sys
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


C = _load("_comm2.py", "q2_comm")

import numpy as np

NU = 0.4815481481
M = C.M_SCENARIOS
T = C.PERIODS_PER_DAY
S = C.S_PERIOD_KWH
ETA = C.ETA
DELTA = 6.0


def make_grid(delta: float = DELTA) -> np.ndarray:
    n = int(round((C.E_MAX - C.E_MIN) / delta))
    return np.linspace(C.E_MIN, C.E_MAX, n + 1)


def _interp_rows(grid: np.ndarray, vals: np.ndarray, e_new: np.ndarray) -> np.ndarray:
    g = grid.ravel()
    delta = g[1] - g[0]
    n = g.size
    pos = np.clip((e_new - g[0]) / delta, 0.0, n - 1.0)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
    fr = np.clip(pos - i0, 0.0, 1.0)
    vals = vals.reshape(-1, n)
    return vals[np.arange(vals.shape[0])[:, None], i0] * (1.0 - fr) \
        + vals[np.arange(vals.shape[0])[:, None], i0 + 1] * fr


def max_argmin(a: np.ndarray) -> int:
    return a.size - 1 - int(np.argmin(a[::-1]))


def build_value_functions(scen_N_day: np.ndarray, g_day: np.ndarray,
                          price: np.ndarray, nu: float = NU,
                          delta: float = DELTA) -> dict:
    M_, T_ = scen_N_day.shape
    grid = make_grid(delta)
    n = grid.size
    r = scen_N_day - g_day[None, :]

    Hs = np.tile(-nu * grid, (M_, 1))
    Hbar = np.empty((T_ + 1, n))
    Hbar[T_] = -nu * grid
    R = np.zeros(T_)
    Rw = np.zeros((M_, T_))
    idx_all = np.arange(M_)[:, None]
    jall = np.arange(n)[None, :]

    for t in range(T_ - 1, -1, -1):
        cc = 5.0 * price[t]
        rt = r[:, t]

        tm = np.minimum(S, np.maximum(rt, 0.0)) / ETA
        F = Hs + cc * ETA * grid[None, :]
        jstar = (n - 1) - np.argmin(F[:, ::-1], axis=1)
        m = np.floor(tm / delta + 1e-12).astype(np.int64)
        lo = np.maximum(0, jall - m[:, None])
        jsel = np.minimum(np.maximum(jstar[:, None], lo), jall)
        new_def = F[idx_all, jsel] - cc * ETA * grid[None, :]

        s = ETA * np.minimum(S, np.maximum(-rt, 0.0))
        enew = np.minimum(grid[None, :] + s[:, None], C.E_MAX)
        new_chg = _interp_rows(grid, Hs, enew)

        Gw = cc * ETA * grid[None, :] + Hs
        Rw[:, t] = grid[(n - 1) - np.argmin(Gw[:, ::-1], axis=1)]
        Hs = np.where((rt <= 0.0)[:, None], new_chg, new_def)
        Hbar[t] = Hs.mean(axis=0)

        Gt = cc * ETA * grid + Hbar[t + 1]
        R[t] = grid[max_argmin(Gt)]

    return {"Hbar": Hbar, "R": R, "Rw": Rw, "grid": grid, "r": r}


def dp_execute(Hbar: np.ndarray, R: np.ndarray, N_act: np.ndarray,
               g_day: np.ndarray, E0: float) -> dict:
    T_ = N_act.size
    Cch = np.zeros(T_)
    D = np.zeros(T_)
    b = np.zeros(T_)
    U = np.zeros(T_)
    E = np.zeros(T_)
    e = float(E0)
    for t in range(T_):
        rt = float(N_act[t] - g_day[t])
        if rt > 0.0:
            d = min(rt, S, ETA * (e - R[t]))
            d = max(d, 0.0)
            D[t] = d
            b[t] = rt - d
            e = e - d / ETA
        else:
            cch = min(-rt, S, (C.E_MAX - e) / ETA)
            cch = max(cch, 0.0)
            Cch[t] = cch
            U[t] = -rt - cch
            e = e + ETA * cch
        E[t] = e
    return {"C": Cch, "D": D, "b": b, "U": U, "E": E}


def analytic_execute(N_act: np.ndarray, g_day: np.ndarray, E0: float) -> dict:
    T_ = N_act.size
    Cch = np.zeros(T_)
    D = np.zeros(T_)
    b = np.zeros(T_)
    U = np.zeros(T_)
    E = np.zeros(T_)
    e = float(E0)
    for t in range(T_):
        rt = float(N_act[t] - g_day[t])
        if rt > 0.0:
            d = max(min(rt, S, ETA * (e - C.E_MIN)), 0.0)
            D[t] = d
            b[t] = rt - d
            e = e - d / ETA
        else:
            cch = max(min(-rt, S, (C.E_MAX - e) / ETA), 0.0)
            Cch[t] = cch
            U[t] = -rt - cch
            e = e + ETA * cch
        E[t] = e
    return {"C": Cch, "D": D, "b": b, "U": U, "E": E}


def direct_opt_action(Hbar_next: np.ndarray, grid: np.ndarray, r: float,
                      c: float, e: float) -> float:
    cc = 5.0 * c
    delta = grid[1] - grid[0]
    if r > 0.0:
        x_lo = max(-min(S, r) / ETA, C.E_MIN - e)
        x_hi = 0.0
    elif r < 0.0:
        x_lo = 0.0
        x_hi = min(ETA * min(S, -r), C.E_MAX - e)
    else:
        return 0.0
    if x_hi - x_lo < -1e-9:
        return 0.0
    n0 = int(np.ceil(x_lo / delta - 1e-12))
    n1 = int(np.floor(x_hi / delta + 1e-12))
    xs = np.arange(n0, n1 + 1) * delta
    if xs.size == 0:
        xs = np.array([0.0])
    en = e + xs
    psi = np.where(xs >= 0, xs / ETA, ETA * xs)
    val = cc * np.maximum(r + psi, 0.0) + _interp_rows(
        grid[None, :], Hbar_next[None, :], en[None, :])[0]
    return float(xs[max_argmin(val)])


def main() -> int:
    C.ensure_dirs()
    import time

    log: list = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 74)
    p("第二问 06 —— DP 未来价值执行器（式 29~34，数值近似：均匀网格）")
    p("=" * 74)

    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    price = Z["price"]
    N_act_all = Z["net_load_energy_kwh"]
    scen_N_all = Zd["scen_L"] - Zd["scen_V"]
    date_strs = [str(s) for s in Z["dates"]]
    score_idx = Z["score_day_index"]
    grid = make_grid(DELTA)
    TOL_VAL = DELTA * 5.0 * float(price.max()) * ETA
    p(f"内部电量网格：{grid.size} 个节点，步长 {DELTA} kWh，"
      f"区间 [{grid[0]:.1f}, {grid[-1]:.1f}]（数值近似，非连续折点递推）")

    from scipy.optimize import linprog
    lp_mod = _load("05_求解日前计划.py", "q2_lp")
    lp = lp_mod.DayPlanLP(price)

    d1 = score_idx[10]
    g_self = lp.solve(scen_N_all[d1], C.E_INIT)["g"]

    p("")
    p("── 自检 1：动态保留水平 R_t（式 33/34）与式 32 直接最小化的一致性 ──")
    vf = build_value_functions(scen_N_all[d1], g_self, price)
    E0 = float(C.E_INIT)
    r_true = N_act_all[d1] - g_self
    n_chk = 0
    n_bad = 0
    max_val_dev = 0.0
    for t in range(T):
        if r_true[t] > 0:
            d_rule = max(min(r_true[t], S, ETA * (E0 - vf["R"][t])), 0.0)
            x_rule = -d_rule / ETA
        else:
            c_rule = max(min(-r_true[t], S, (C.E_MAX - E0) / ETA), 0.0)
            x_rule = c_rule * ETA
        x_direct = direct_opt_action(vf["Hbar"][t + 1], grid, float(r_true[t]),
                                     float(price[t]), float(E0))

        def _obj(x, _t=t):
            psi = x / ETA if x >= 0 else ETA * x
            return 5.0 * price[_t] * max(r_true[_t] + psi, 0.0) + float(
                _interp_rows(grid[None, :], vf["Hbar"][_t + 1][None, :],
                             np.array([[E0 + x]]))[0, 0])

        val_dev = abs(_obj(x_rule) - _obj(x_direct))
        max_val_dev = max(max_val_dev, val_dev)
        n_chk += 1
        if val_dev > TOL_VAL:
            n_bad += 1
        E0 = float(np.clip(E0 + x_rule, C.E_MIN, C.E_MAX))
    p(f"  一致性容差 = δ·max(5c_t·η) = {TOL_VAL:.6f} 元（单步量化误差上界）")
    p(f"  逐时段比较（{date_strs[d1]}）：共 {n_chk} 个时段，"
      f"超出容差 {n_bad} 个，最大目标差 {max_val_dev:.3e} 元")
    p(f"  → {'✔ 通过（式 33/34 与式 32 等价）' if n_bad == 0 else '✘ 未通过'}")

    p("")
    p("── 自检 2：网格步长收敛性（δ = 6 / 3 / 1.5 kWh，数值近似三档对照）──")
    res_by_d = {}
    for dl in (6.0, 3.0, 1.5):
        t0 = time.perf_counter()
        vf = build_value_functions(scen_N_all[d1], g_self, price, delta=dl)
        o = dp_execute(vf["Hbar"], vf["R"], N_act_all[d1], g_self, C.E_INIT)
        k = float(price @ g_self) + float((5.0 * price) @ o["b"])
        res_by_d[dl] = (k, float(o["b"].sum()))
        p(f"  δ = {dl:<4} 节点 {vf['grid'].size:<5} 费用 {k:.4f} 元，"
          f"紧急量 {o['b'].sum():.4f} kWh，耗时 {time.perf_counter() - t0:.2f} s（"
          f"{date_strs[d1]}）")
    d63 = abs(res_by_d[6.0][0] - res_by_d[3.0][0])
    d31 = abs(res_by_d[3.0][0] - res_by_d[1.5][0])
    p(f"  相邻网格费用差：δ6→δ3 = {d63:.6f} 元；δ3→δ1.5 = {d31:.6f} 元")
    p(f"  → {'✔ 收敛（相邻差 ≤ 1e-2 元）' if max(d63, d31) <= 1e-2 else '⚠ 未充分收敛，但保留为数值近似'}")

    log_path = C.SOLVE_LOG_DIR / "第二问_06DP自检日志.txt"
    C.write_text_utf8(log_path, "\n".join(
        log + ["", "[06 完成] DP 价值执行器自检结束。"]))
    C.write_text_utf8(C.REPORT_DIR / "第二问_DP执行器自检报告.md", "\n".join(
        ["# 第二问 DP 未来价值执行器 自检报告（数值近似）", ""] + [f"{x}" for x in log] + [""]))
    p("")
    p("已保存：求解日志/第二问_06DP自检日志.txt、报告/第二问_DP执行器自检报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
