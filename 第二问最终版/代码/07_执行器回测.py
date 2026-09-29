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


C = _load("_comm2.py", "q2_comm")
P5 = _load("05_求解日前计划.py", "q2_plan")
DP = _load("06_DP价值执行器.py", "q2_dp")

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = C.M_SCENARIOS
T = C.PERIODS_PER_DAY
S = C.S_PERIOD_KWH
ETA = C.ETA
NU = P5.NU
TOLL = 1e-6


def mpc_first_action(price_seg: np.ndarray, r_hat: np.ndarray,
                     E_now: float, nu: float = NU) -> tuple:
    n = int(r_hat.size)
    nv = 4 * n
    ob, oc, od, oe = 0, n, 2 * n, 3 * n
    defc = r_hat > 0.0

    rows_ub, cols_ub, vals_ub, b_ub = [], [], [], np.zeros(n)
    for s in range(n):
        if defc[s]:
            rows_ub += [s, s]
            cols_ub += [ob + s, od + s]
            vals_ub += [-1.0, -1.0]
            b_ub[s] = -r_hat[s]
        else:
            rows_ub += [s]
            cols_ub += [oc + s]
            vals_ub += [1.0]
            b_ub[s] = -r_hat[s]
    A_ub = sp.csr_matrix((np.array(vals_ub), (np.array(rows_ub), np.array(cols_ub))),
                         shape=(n, nv))

    rows_eq, cols_eq, vals_eq, b_eq = [], [], [], np.zeros(n)
    for s in range(n):
        rows_eq.append(s)
        cols_eq.append(oe + s)
        vals_eq.append(1.0)
        rows_eq.append(s)
        if defc[s]:
            cols_eq.append(od + s)
            vals_eq.append(1.0 / ETA)
        else:
            cols_eq.append(oc + s)
            vals_eq.append(-ETA)
        if s > 0:
            rows_eq.append(s)
            cols_eq.append(oe + s - 1)
            vals_eq.append(-1.0)
        else:
            b_eq[s] = E_now
    A_eq = sp.csr_matrix((np.array(vals_eq), (np.array(rows_eq), np.array(cols_eq))),
                         shape=(n, nv))

    cvec = np.zeros(nv)
    cvec[ob:ob + n] = np.where(defc, 5.0 * price_seg[:n], 0.0)
    cvec[oe + n - 1] = -nu

    bnds = ([(0.0, None) if defc[s] else (0.0, 0.0) for s in range(n)]
            + [(0.0, S) if not defc[s] else (0.0, 0.0) for s in range(n)]
            + [(0.0, S) if defc[s] else (0.0, 0.0) for s in range(n)]
            + [(C.E_MIN, C.E_MAX)] * n)

    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bnds, method="highs")
    if not res.success:
        raise RuntimeError(f"MPC 滚动 LP 求解失败：{res.message}")
    x = res.x
    if defc[0]:
        return 0.0, float(x[od])
    return float(x[oc]), 0.0


def mpc_execute(price: np.ndarray, N_act: np.ndarray, Nhat: np.ndarray,
                rho: float, g_day: np.ndarray, E0: float) -> dict:
    T_ = N_act.size
    Cch = np.zeros(T_)
    D = np.zeros(T_)
    b = np.zeros(T_)
    U = np.zeros(T_)
    E = np.zeros(T_)
    e = float(E0)
    for t in range(T_):
        j = np.arange(T_ - t)
        rho_p = np.power(float(np.clip(rho, 0.0, 0.99)), j)
        Nfc = Nhat[t:] + rho_p * (float(N_act[t]) - float(Nhat[t]))
        r_hat = Nfc - g_day[t:]
        cch, dd = mpc_first_action(price[t:], r_hat, e)
        Cch[t], D[t] = cch, dd
        rt = float(N_act[t] - g_day[t])
        if rt > 0.0:
            b[t] = max(rt - dd, 0.0)
            e = e - dd / ETA
        else:
            U[t] = max(-rt - cch, 0.0)
            e = e + ETA * cch
        E[t] = e
    return {"C": Cch, "D": D, "b": b, "U": U, "E": E}


def eval_plan_causal(Rw: np.ndarray, scen_N_day: np.ndarray, g_day: np.ndarray,
                     price: np.ndarray, E0: float, nu: float = NU) -> np.ndarray:
    M_, T_ = scen_N_day.shape
    J = np.zeros(M_)
    for w in range(M_):
        e = float(E0)
        tot = 0.0
        r = scen_N_day[w] - g_day
        for t in range(T_):
            rt = r[t]
            if rt > 0.0:
                dd = max(min(rt, S, ETA * (e - Rw[w, t])), 0.0)
                tot += 5.0 * price[t] * max(rt - dd, 0.0)
                e = e - dd / ETA
            else:
                cch = max(min(-rt, S, (C.E_MAX - e) / ETA), 0.0)
                e = e + ETA * cch
        J[w] = tot - nu * e
    return J


def new_result():
    return {
        "g": np.zeros((365, T)), "C": np.zeros((365, T)),
        "D": np.zeros((365, T)), "b": np.zeros((365, T)),
        "U": np.zeros((365, T)), "E": np.zeros((365, T)),
        "plan_cost": np.zeros(365), "emerg_cost": np.zeros(365),
        "emerg_kwh": np.zeros(365),
        "E0_plan": np.zeros(365), "E0_exec": np.zeros(365),
    }


def run_executor_chain(executor, lp30, N_act_all, scen_N_all, Nhat_all,
                       rho_all, price, score_idx, date_strs, log_p):
    R = new_result()
    g_out = np.zeros((365, T))
    E_current = float(C.E_INIT)
    for d in score_idx:
        r = lp30.solve(scen_N_all[d], E_current)
        g_d = r["g"]
        g_out[d] = g_d
        R["g"][d] = g_d
        R["E0_plan"][d] = E_current
        R["E0_exec"][d] = E_current
        if executor == "解析响应":
            o = DP.analytic_execute(N_act_all[d], g_d, E_current)
        elif executor == "MPC":
            o = mpc_execute(price, N_act_all[d], Nhat_all[d],
                            float(rho_all[d]), g_d, E_current)
        elif executor == "DP价值执行器":
            vf = DP.build_value_functions(scen_N_all[d], g_d, price)
            o = DP.dp_execute(vf["Hbar"], vf["R"], N_act_all[d], g_d, E_current)
        else:
            raise ValueError(executor)
        R["C"][d], R["D"][d], R["b"][d], R["U"][d], R["E"][d] = (
            o["C"], o["D"], o["b"], o["U"], o["E"])
        R["plan_cost"][d] = float(price @ g_d)
        R["emerg_cost"][d] = float((5.0 * price) @ o["b"])
        R["emerg_kwh"][d] = float(o["b"].sum())
        E_current = float(o["E"][-1])
    return R, g_out


def run_executor_fixed(executor, g_fixed, N_act_all, scen_N_all, Nhat_all,
                       rho_all, price, score_idx, date_strs):
    R = new_result()
    E_current = float(C.E_INIT)
    for d in score_idx:
        g_d = g_fixed[d]
        R["g"][d] = g_d
        R["E0_plan"][d] = E_current
        R["E0_exec"][d] = E_current
        if executor == "解析响应":
            o = DP.analytic_execute(N_act_all[d], g_d, E_current)
        elif executor == "MPC":
            o = mpc_execute(price, N_act_all[d], Nhat_all[d],
                            float(rho_all[d]), g_d, E_current)
        elif executor == "DP价值执行器":
            vf = DP.build_value_functions(scen_N_all[d], g_d, price)
            o = DP.dp_execute(vf["Hbar"], vf["R"], N_act_all[d], g_d, E_current)
        else:
            raise ValueError(executor)
        R["C"][d], R["D"][d], R["b"][d], R["U"][d], R["E"][d] = (
            o["C"], o["D"], o["b"], o["U"], o["E"])
        R["plan_cost"][d] = float(price @ g_d)
        R["emerg_cost"][d] = float((5.0 * price) @ o["b"])
        R["emerg_kwh"][d] = float(o["b"].sum())
        E_current = float(o["E"][-1])
    return R


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg, flush=True)

    t_all = time.perf_counter()
    p("=" * 74)
    p("第二问 07 —— 跨日闭环三执行器连续回测（固定计划 / 各自重订）")
    p("=" * 74)

    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    price = np.asarray(Z["price"], float)
    N_act_all = np.asarray(Z["net_load_energy_kwh"], float)
    scen_N_all = np.asarray(Zd["scen_L"], float) - np.asarray(Zd["scen_V"], float)
    Nhat_all = np.asarray(Zd["Nhat"], float)
    rho_all = np.asarray(Zd["rho"], float)
    date_strs = [str(x) for x in Z["dates"]]
    score_idx = np.asarray(Z["score_day_index"], int)
    nD = score_idx.size
    p(f"评分期 {date_strs[score_idx[0]]} … {date_strs[score_idx[-1]]}，{nD} 天 / {nD * T} 时段")
    p(f"ν = {NU:.10f} 元/kWh；电价 ∈ [{price.min():.6f}, {price.max():.6f}]")
    p("MPC 滚动时域 = 完整剩余时域 t..144（式 28，非固定 36 步）")

    lp30 = P5.DayPlanLP(price, nu=NU, m=M)

    p("")
    p("── 基准执行方案：解析响应跨日闭环（生成固定计划 g_fixed）──")
    t0 = time.perf_counter()
    R_an, g_fixed = run_executor_chain(
        "解析响应", lp30, N_act_all, scen_N_all, Nhat_all, rho_all,
        price, score_idx, date_strs, p)
    p(f"  g_fixed 全年计划购电量 {g_fixed[score_idx].sum():.4f} kWh，"
      f"计划费 {R_an['plan_cost'][score_idx].sum():.4f} 元，"
      f"年末库存 {R_an['E'][score_idx[-1], -1]:.6f} kWh，"
      f"用时 {time.perf_counter() - t0:.1f} s")

    p("")
    p("── 固定计划组：三执行器共用 g_fixed，各自从 6000 独立执行 ──")
    t0 = time.perf_counter()
    R_fix_an = R_an
    R_fix_mpc = run_executor_fixed("MPC", g_fixed, N_act_all, scen_N_all,
                                   Nhat_all, rho_all, price, score_idx, date_strs)
    R_fix_dp = run_executor_fixed("DP价值执行器", g_fixed, N_act_all, scen_N_all,
                                  Nhat_all, rho_all, price, score_idx, date_strs)
    p(f"  用时 {time.perf_counter() - t0:.1f} s")

    p("")
    p("── 各自重订计划组：三执行器各维护真实库存、各自重订全天计划 ──")
    t0 = time.perf_counter()
    R_an_self = R_an
    R_mpc, g_mpc = run_executor_chain(
        "MPC", lp30, N_act_all, scen_N_all, Nhat_all, rho_all,
        price, score_idx, date_strs, p)
    R_dp, g_dp = run_executor_chain(
        "DP价值执行器", lp30, N_act_all, scen_N_all, Nhat_all, rho_all,
        price, score_idx, date_strs, p)
    p(f"  用时 {time.perf_counter() - t0:.1f} s")

    groups = {"固定计划": ("解析响应", "MPC", "DP价值执行器"),
              "各自重订": ("解析响应", "MPC", "DP价值执行器")}
    RES = {
        ("固定计划", "解析响应"): R_fix_an,
        ("固定计划", "MPC"): R_fix_mpc,
        ("固定计划", "DP价值执行器"): R_fix_dp,
        ("各自重订", "解析响应"): R_an_self,
        ("各自重订", "MPC"): R_mpc,
        ("各自重订", "DP价值执行器"): R_dp,
    }
    GPLAN = {
        ("固定计划", "解析响应"): g_fixed, ("固定计划", "MPC"): g_fixed,
        ("固定计划", "DP价值执行器"): g_fixed,
        ("各自重订", "解析响应"): g_fixed, ("各自重订", "MPC"): g_mpc,
        ("各自重订", "DP价值执行器"): g_dp,
    }

    p("")
    p("── 表 7 执行器对照（万元 / 万 kWh）──")
    p(f"{'比较方式':<10}{'执行器':<14}{'计划费':>12}{'紧急费':>12}{'总费用':>12}"
      f"{'紧急量':>10}")
    tbl_rows = []
    for gname, enames in groups.items():
        for ename in enames:
            R = RES[(gname, ename)]
            pc = float(R["plan_cost"][score_idx].sum())
            ec = float(R["emerg_cost"][score_idx].sum())
            ek = float(R["emerg_kwh"][score_idx].sum())
            p(f"{gname:<10}{ename:<14}{pc / 1e4:>12.3f}{ec / 1e4:>12.3f}"
              f"{(pc + ec) / 1e4:>12.3f}{ek / 1e4:>10.3f}")
            tbl_rows.append((gname, ename, pc, ec, pc + ec, ek))

    def _total(gname, ename):
        R = RES[(gname, ename)]
        return float((R["plan_cost"] + R["emerg_cost"])[score_idx].sum())

    p("")
    p(f"各自重订：DP 总费用 {_total('各自重订', 'DP价值执行器'):.4f} 元，"
      f"解析响应 {_total('各自重订', '解析响应'):.4f} 元，"
      f"DP 省 {_total('各自重订', '解析响应') - _total('各自重订', 'DP价值执行器'):.4f} 元（"
      f"{(_total('各自重订', 'DP价值执行器') / _total('各自重订', '解析响应') - 1.0) * 100:+.4f}%）")
    p(f"固定计划：DP 总费用 {_total('固定计划', 'DP价值执行器'):.4f} 元，"
      f"解析响应 {_total('固定计划', '解析响应'):.4f} 元，"
      f"DP 省 {_total('固定计划', '解析响应') - _total('固定计划', 'DP价值执行器'):.4f} 元（"
      f"{(_total('固定计划', 'DP价值执行器') / _total('固定计划', '解析响应') - 1.0) * 100:+.4f}%）")
    p(f"→ 最终主结果取【各自重订 / DP 价值执行器】："
      f"{_total('各自重订', 'DP价值执行器'):.4f} 元")

    p("")
    p("── 跨日闭环连续性校验（E_{d,0}^plan = E_{d,0}^exec = E_{d-1,144}^actual）──")
    cont_rows = []
    max_res_plan = 0.0
    max_res_exec = 0.0
    for gname, enames in groups.items():
        for ename in enames:
            R = RES[(gname, ename)]
            rp = 0.0
            re = 0.0
            for i, d in enumerate(score_idx):
                prev_end = float(R["E"][score_idx[i - 1], -1]) if i > 0 else float(C.E_INIT)
                rp = max(rp, abs(R["E0_plan"][d] - prev_end))
                re = max(re, abs(R["E0_exec"][d] - prev_end))
            max_res_plan = max(max_res_plan, rp)
            max_res_exec = max(max_res_exec, re)
            p(f"  [{gname}/{ename}] 计划初值链残差 {rp:.3e}，执行初值链残差 {re:.3e} kWh")
            cont_rows.append((gname, ename, f"{rp:.3e}", f"{re:.3e}"))
    p(f"  最大跨日连续残差：计划链 {max_res_plan:.3e}，执行链 {max_res_exec:.3e} kWh")

    dp_key = ("各自重订", "DP价值执行器")
    an_key = ("各自重订", "解析响应")
    mpc_key = ("各自重订", "MPC")
    bill_dp = RES[dp_key]["plan_cost"] + RES[dp_key]["emerg_cost"]
    bill_an = RES[an_key]["plan_cost"] + RES[an_key]["emerg_cost"]
    bill_mpc = RES[mpc_key]["plan_cost"] + RES[mpc_key]["emerg_cost"]
    ddiff = bill_an[score_idx] - bill_dp[score_idx]
    n_down = int((ddiff > 1.0).sum())
    n_up = int((ddiff < -1.0).sum())
    n_same = int(nD - n_down - n_up)
    p("")
    p(f"逐日比较（DP − 解析响应，正数 = DP 更省）：降 {n_down} 天、"
      f"升 {n_up} 天、基本相同 {n_same} 天；最大省 {ddiff.max():.4f} 元，"
      f"最大亏 {-ddiff.min():.4f} 元")
    months = np.array([int(date_strs[d][5:7]) for d in score_idx])
    m_save = {mm: float(ddiff[months == mm].sum()) for mm in range(2, 13)}
    p("  月度节省（万元）：" + "、".join(
        f"{mm}月 {m_save[mm] / 1e4:.2f}" for mm in range(2, 13)))
    p(f"  月度节省全为正：{'✔ 是' if all(v > 0 for v in m_save.values()) else '✘ 否'}")

    p("")
    p("── 诊断 ①：日前情景 LP 的乐观偏差（紧急购电与充电同时发生）──")
    t0 = time.perf_counter()
    bad_days = 0
    bad_periods = 0
    sum_sig = 0.0
    sum_abs = 0.0
    n_gap = 0
    diag_rows = []
    for d in score_idx:
        r = lp30.solve(scen_N_all[d], RES[an_key]["E0_plan"][d])
        bL, CL, EL = r["b"], r["C"], r["E"]
        co = (bL > TOLL) & (CL > TOLL)
        bad_days += int(co.any())
        bad_periods += int(co.sum())
        vf = DP.build_value_functions(scen_N_all[d], GPLAN[an_key][d], price)
        J_lp = 5.0 * (price @ bL.T) - NU * EL[:, -1]
        J_ev = eval_plan_causal(vf["Rw"], scen_N_all[d], GPLAN[an_key][d],
                                price, RES[an_key]["E0_plan"][d])
        gap_sig = float(J_ev.mean() - J_lp.mean())
        gap_abs = float(np.mean(np.abs(J_lp - J_ev)))
        sum_sig += gap_sig
        sum_abs += gap_abs
        n_gap += 1
        diag_rows.append((date_strs[d], int(co.any()), int(co.sum()),
                          float(J_lp.mean()), float(J_ev.mean()),
                          gap_sig, gap_abs))
    p(f"  出现「紧急购电与充电同时发生」的天数：{bad_days} / {nD}，"
      f"累计情景时段数：{bad_periods}（M = {M}）")
    p(f"  日前情景目标 vs 逐情景已知轨迹重评（含 −ν·E_T）：")
    p(f"    有符号偏差（情景内重评 − 日前，> 0 = 日前乐观低估）：{sum_sig / n_gap:.4f} 元/天")
    p(f"    平均绝对差（逐情景 |Δ| 再取日均）：{sum_abs / n_gap:.4f} 元/天"
      f"（用时 {time.perf_counter() - t0:.1f} s）")

    ann_rows = []
    for gname, ename, pc, ec, tt, ek in tbl_rows:
        R = RES[(gname, ename)]
        ann_rows.append((gname, ename, pc, ec, tt, ek,
                         float(GPLAN[(gname, ename)][score_idx].sum()),
                         float(R["C"][score_idx].sum()), float(R["D"][score_idx].sum()),
                         float(R["E"][score_idx[-1], -1])))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_执行器对照表.csv",
        ("比较方式", "执行器", "计划费_元", "紧急费_元", "总费用_元", "紧急量_kWh",
         "计划购电量_kWh", "充电量_kWh", "放电量_kWh", "年末储电量_kWh"),
        [tuple(r_) for r_ in ann_rows])

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_逐日费用对照.csv",
        ("日期", "解析响应总费用_元", "MPC总费用_元", "DP总费用_元",
         "DP较解析响应节省_元", "DP紧急量_kWh", "解析响应紧急量_kWh"),
        [(date_strs[d], f"{bill_an[d]:.6f}", f"{bill_mpc[d]:.6f}",
          f"{bill_dp[d]:.6f}", f"{ddiff[i]:.6f}",
          f"{RES[dp_key]['emerg_kwh'][d]:.6f}",
          f"{RES[an_key]['emerg_kwh'][d]:.6f}")
         for i, d in enumerate(score_idx)])

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_月度节省.csv",
        ("月份", "DP较解析响应节省_元", "节省_万元"),
        [(f"{mm}月", f"{m_save[mm]:.6f}", f"{m_save[mm] / 1e4:.6f}")
         for mm in range(2, 13)])

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_日前乐观偏差诊断.csv",
        ("日期", "是否出现同时购电与充电", "同时发生情景时段数",
         "日前情景目标均值_元", "逐情景重评目标均值_元",
         "有符号偏差_元", "平均绝对差_元"),
        [(x[0], str(x[1]), str(x[2]), f"{x[3]:.6f}", f"{x[4]:.6f}", f"{x[5]:.6f}",
          f"{x[6]:.6f}")
         for x in diag_rows])

    np.savez_compressed(
        C.RESULT_DIR / "第二问_固定计划.npz",
        plan_g=g_fixed, plan_cost=R_fix_an["plan_cost"],
        plan_E_mean=np.array([R_fix_an["E"][d, -1] for d in range(365)]),
        plan_b=R_fix_an["b"], score_day_index=score_idx, nu=np.array([NU]),
    )
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_固定计划日汇总.csv",
        ("日期", "全天计划购电量_kWh", "全天计划购电费_元", "日末计划储电量_kWh"),
        [(date_strs[d], f"{g_fixed[d].sum():.6f}", f"{R_fix_an['plan_cost'][d]:.6f}",
          f"{R_fix_an['E'][d, -1]:.6f}") for d in score_idx])

    np.savez_compressed(
        C.RESULT_DIR / "第二问_执行器回测.npz",
        score_day_index=score_idx,
        g_fixed=g_fixed,
        plan_g=g_dp,
        dp_C=RES[dp_key]["C"], dp_D=RES[dp_key]["D"], dp_b=RES[dp_key]["b"],
        dp_E=RES[dp_key]["E"], dp_U=RES[dp_key]["U"],
        an_g=g_fixed, an_C=RES[an_key]["C"], an_D=RES[an_key]["D"],
        an_b=RES[an_key]["b"], an_E=RES[an_key]["E"], an_U=RES[an_key]["U"],
        mpc_g=g_mpc, mpc_C=RES[mpc_key]["C"], mpc_D=RES[mpc_key]["D"],
        mpc_b=RES[mpc_key]["b"], mpc_E=RES[mpc_key]["E"], mpc_U=RES[mpc_key]["U"],
        g_analytic=g_fixed, g_mpc_var=g_mpc, g_dp_var=g_dp,
    )

    C.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=150)
    xs = np.arange(2, 13)
    ys = np.array([m_save[mm] / 1e4 for mm in xs])
    ax.bar(xs, ys, color="#2E75B6", width=0.62)
    for x, y in zip(xs, ys):
        ax.text(x, y + max(ys) * 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{mm}月" for mm in xs])
    ax.set_ylabel("月度节省（万元）")
    ax.set_title("图 6  DP 价值执行器相对解析响应的月度节省（2025 年 2—12 月）")
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(C.FIGURE_DIR / "第二问_图6_DP相对解析响应月度费用节省.png")
    plt.close(fig)
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_图6_作图数据.csv",
        ("月份", "节省_万元"), [(f"{mm}月", f"{m_save[mm] / 1e4:.6f}") for mm in xs])

    p("")
    p(f"已保存：模型结果/第二问_执行器对照表.csv、第二问_逐日费用对照.csv、"
      f"第二问_月度节省.csv、第二问_日前乐观偏差诊断.csv、第二问_固定计划.npz、第二问_执行器回测.npz")
    p(f"已保存：模型结果图/第二问_图6_DP相对解析响应月度费用节省.png")
    p(f"总用时 {time.perf_counter() - t_all:.1f} s")

    C.write_text_utf8(C.SOLVE_LOG_DIR / "第二问_07回测日志.txt",
                      "\n".join(log + ["", "[07 完成] 跨日闭环三执行器回测结束。"]))
    C.write_text_utf8(C.REPORT_DIR / "第二问_执行器回测报告.md",
                      "\n".join(["# 第二问 跨日闭环三执行器回测报告", ""] + log + [""]))
    p("已保存：求解日志/第二问_07回测日志.txt、报告/第二问_执行器回测报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
