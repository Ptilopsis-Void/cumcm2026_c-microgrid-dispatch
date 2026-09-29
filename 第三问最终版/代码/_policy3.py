#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("_comm3.py", "q3_comm_policy")
S5 = _load("05_求解阶段0计划.py", "q3_05_policy")
S7 = _load("07_阶段内DP执行器.py", "q3_07_policy")

T = C.PERIODS_PER_DAY
TAUS = tuple(C.TAU_HOURS)
K0S = tuple(C.TAU_PERIOD_INDEX)
NODE_TAU_LABEL = {t: f"{t}:00" for t in TAUS}

ALL_ARMS = tuple(C.ABLATION_ARMS)


_FACTS: dict | None = None


def facts() -> dict:
    global _FACTS
    if _FACTS is not None:
        return _FACTS
    F = np.load(C.V_FORECAST_NPZ, allow_pickle=False)
    Zd = np.load(C.V_SCENARIO_NPZ, allow_pickle=False)
    Z = C.Q2.matrix()
    S2 = C.Q2.scenarios()
    warm = (np.asarray(F["warmup_day_index"], int)
            if "warmup_day_index" in F else np.array([], dtype=int))
    _FACTS = {
        "price": np.asarray(Z["price"], float),
        "load_kw": np.asarray(Z["load_kw"], float),
        "pv_kw": np.asarray(Z["pv_kw"], float),
        "scen_L": np.asarray(S2["scen_L"], float),
        "scen_V_abs": C.lead_to_absolute(
            np.asarray(Zd["scen_V_hourly"], float)),
        "score_day_index": np.asarray(F["score_day_index"], int),
        "warmup_day_index": warm,
        "dates": [str(x) for x in F["dates"]],
    }
    _FACTS["N_actual_all"] = ((_FACTS["load_kw"] - _FACTS["pv_kw"])
                              * C.DELTA_HOURS)
    return _FACTS


def reset_cache() -> None:
    global _FACTS
    _FACTS = None


def node_plan(update_times=()) -> list[tuple[int, int]]:
    us = sorted({int(t) for t in update_times})
    for t in us:
        if t not in TAUS:
            raise ValueError(
                f"update_times 只能取自 {TAUS}（附件 3 的发布时刻），收到 {t}")
    return [(0, 0)] + [(t, K0S[TAUS.index(t)]) for t in us]


def segment_bounds(nodes) -> list[tuple[int, int, int]]:
    out = []
    for i, (tau, k0) in enumerate(nodes):
        k1 = nodes[i + 1][1] if i + 1 < len(nodes) else T
        out.append((tau, k0, k1))
    return out


def seg_for_tau(update_times, tau: int) -> tuple[int, int]:
    for (t, k0, k1) in segment_bounds(node_plan(update_times)):
        if t == tau:
            return k0, k1
    raise ValueError(f"τ={tau} 不在 update_times={tuple(update_times)} 中")


def decompose(price: np.ndarray, P: np.ndarray, Q: np.ndarray,
              b: np.ndarray) -> dict:
    u = np.maximum(P - Q, 0.0)
    v = np.maximum(Q - P, 0.0)
    plan = (price * np.minimum(P, Q)).sum(axis=1)
    down = (C.RHO_DOWN * price * u).sum(axis=1)
    up = (C.RHO_UP * price * v).sum(axis=1)
    emerg = (C.EMERG_MULT * price * b).sum(axis=1)
    return {"plan": plan, "down": down, "up": up, "emerg": emerg,
            "total": plan + down + up + emerg,
            "u": u.sum(axis=1), "v": v.sum(axis=1), "b": b.sum(axis=1)}


def exante_cost_seg(price_seg, p_seg, q_seg, N_seg, mult=5.0):
    u = np.maximum(p_seg - q_seg, 0.0)
    v = np.maximum(q_seg - p_seg, 0.0)
    base = float((price_seg * (np.minimum(p_seg, q_seg)
                               + 0.5 * u + 1.5 * v)).sum())
    gap = np.maximum(N_seg - q_seg[None, :], 0.0)
    em = mult * float((price_seg[None, :] * gap).mean(0).sum())
    return base + em, base, em


def run_policy(*, update_times=(6, 12, 18), days=None, m=None, nu=None,
               delta: float = 6.0, executor: str = "dp",
               stage0_mode: str = "plan", stage0_recourse: bool = True,
               e_init: float | None = None, with_analytic: bool = True,
               forecast_mode: str = "attachment3", perfect_taus=(),
               perfect_force: bool = False,
               eval_full: bool = False,
               log=None, progress_every: int = 40,
               collect_expected: bool = True) -> dict:
    f = facts()
    price = f["price"]
    sc_all = f["score_day_index"]
    if days is None:
        days = sc_all
    days = np.asarray(days, int)
    mm = int(m or C.M_SCENARIOS)
    _M_LIB = int(np.asarray(f["scen_L"]).shape[1])
    if mm > _M_LIB:
        raise ValueError(f"m={mm} 超过情景库可用情景数 {_M_LIB}")
    nuv = float(C.NU_VALUE if nu is None else nu)
    nodes = node_plan(update_times)
    segs = segment_bounds(nodes)
    n_node = len(nodes)
    if executor not in ("dp", "analytic"):
        raise ValueError("executor 只能为 'dp' 或 'analytic'")
    if forecast_mode not in ("attachment3", "actual"):
        raise ValueError("forecast_mode 只能为 'attachment3' 或 'actual'")
    _seed = {int(t) for t in perfect_taus}
    for t in _seed:
        if t not in TAUS:
            raise ValueError(f"perfect_taus 只能取自 {TAUS}，收到 {t}")
    _all_actual = (forecast_mode == "actual")

    def _is_perf(tau: int) -> bool:
        return _all_actual or (int(tau) in _seed)

    n_day = 365
    P = np.zeros((n_day, T))
    Q = np.zeros((n_day, T))
    SRC = np.full((n_day, T), -1, dtype=int)
    SRC_MIN = np.full((n_day, T), -1, dtype=int)
    Cch = np.zeros((n_day, T))
    Dh = np.zeros((n_day, T))
    Bm = np.zeros((n_day, T))
    Curt = np.zeros((n_day, T))
    Em = np.zeros((n_day, T))
    Can = np.zeros((n_day, T))
    Dan = np.zeros((n_day, T))
    Ban = np.zeros((n_day, T))
    CurtA = np.zeros((n_day, T))
    Ean = np.zeros((n_day, T))
    Rmat = np.zeros((n_day, T))
    E_in_node = np.full((n_day, len(TAUS)), np.nan)
    E_out_node = np.full((n_day, len(TAUS)), np.nan)
    obj_lp0 = np.full(n_day, np.nan)
    lp_stats = {"n_var": 0, "n_ub": 0, "n_eq": 0}
    E_plan_chain = np.full(n_day, np.nan)
    EM_model = np.zeros((n_day, T))
    n_declined = 0
    n_declined_perf = 0
    n_forced = 0

    e = float(C.E_INIT if e_init is None else e_init)
    e_an = float(e)
    t00 = time.perf_counter()
    for i, d in enumerate(days):
        d = int(d)
        E_start = float(e)
        E_start_an = float(e_an)
        V0_all = C.scenario_pv_energy(f["scen_V_abs"], d, 0)[:mm]
        N_comm = f["scen_L"][d][:mm] - V0_all
        if _all_actual:
            N_eval = f["N_actual_all"][d][None, :]
        elif eval_full and mm < _M_LIB:
            N_eval = f["scen_L"][d] - C.scenario_pv_energy(f["scen_V_abs"], d, 0)
        else:
            N_eval = N_comm
        if _is_perf(0):
            L = (f["load_kw"][d] * C.DELTA_HOURS)[None, :]
            V0 = (f["pv_kw"][d] * C.DELTA_HOURS)[None, :]
            mm_n = 1
        else:
            L = f["scen_L"][d][:mm]
            V0 = V0_all
            mm_n = mm
        N0 = L - V0
        if stage0_mode == "plan":
            lp0 = S5.TwoStageSP(price, mm_n, nuv, mode="plan")
            r0 = lp0.solve(N0, E_start)
            P[d] = r0["p"]
            Q[d] = r0["p"]
        elif stage0_mode == "plan_sp_rc":
            lp0 = S5.TwoStageSP(price, mm_n, nuv, mode="plan_sp_rc",
                                recourse=stage0_recourse)
            r0 = lp0.solve(N0, E_start)
            P[d] = r0["p"]
            Q[d] = r0["q"]
        else:
            raise ValueError(f"stage0_mode 只能为 'plan' 或 'plan_sp_rc'，"
                             f"收到 {stage0_mode!r}")
        obj_lp0[d] = r0["objective"]
        lp_stats = {"n_var": lp0.n_var, "n_ub": lp0.n_ub, "n_eq": lp0.n_eq}
        E_in_node[d, 0] = E_start
        SRC_MIN[d, :] = 0
        E_plan_chain[d] = float(np.asarray(r0["E"], float)[:, -1].mean())

        e_node = E_start
        e_node_an = E_start_an
        for (tau, k0, k1) in segs:
            ti = TAUS.index(tau)
            if tau == 0:
                r_node = N0 - Q[d][None, :]
            else:
                cand: list[tuple[str, np.ndarray, np.ndarray]] = []
                if _is_perf(tau):
                    N_perf = f["N_actual_all"][d][None, k0:]
                    lp_p = S5.TwoStageSP(price[k0:], 1, nuv, mode="adjust",
                                         p_seg=P[d, k0:])
                    r_perf = lp_p.solve(N_perf, e_node)
                    cand.append(("完美信息", N_perf,
                                 np.asarray(r_perf["q"], float)))
                    if not _all_actual and not perfect_force:
                        vin = C.scenario_pv_energy(f["scen_V_abs"], d, ti)[:mm]
                        N_reg = (L - vin)[:, k0:]
                        lp_r = S5.TwoStageSP(price[k0:], mm, nuv, mode="adjust",
                                             p_seg=P[d, k0:])
                        r_reg = lp_r.solve(N_reg, e_node)
                        cand.append(("常规预报", N_reg,
                                     np.asarray(r_reg["q"], float)))
                else:
                    vin = C.scenario_pv_energy(f["scen_V_abs"], d, ti)[:mm]
                    N_rem = (L - vin)[:, k0:]
                    lp = S5.TwoStageSP(price[k0:], mm, nuv, mode="adjust",
                                       p_seg=P[d, k0:])
                    r_adj = lp.solve(N_rem, e_node)
                    cand.append(("本节点", N_rem,
                                 np.asarray(r_adj["q"], float)))
                N_eval_t = N_eval[:, k0:]
                q_old = Q[d, k0:].copy()
                best_tag = "维持旧计划"
                best_net = None
                best_q = q_old
                best_c = exante_cost_seg(price[k0:], P[d, k0:], q_old,
                                         N_eval_t)[0]
                for _tag, _net, _q in cand:
                    _c = exante_cost_seg(price[k0:], P[d, k0:], _q, N_eval_t)[0]
                    if _c < best_c - 1e-9:
                        best_tag, best_net, best_q, best_c = _tag, _net, _q, _c
                if _is_perf(tau) and perfect_force:
                    Q[d, k0:] = cand[0][2]
                    best_net = cand[0][1]
                    n_forced += 1
                elif best_tag == "维持旧计划":
                    n_declined += 1
                else:
                    Q[d, k0:] = best_q
                if _is_perf(tau) and not perfect_force and best_tag != "完美信息":
                    n_declined_perf += 1
                r_node = ((cand[0][1] if best_net is None else best_net)
                          - Q[d, k0:][None, :])
            SRC[d, k0:k1] = tau
            SRC_MIN[d, k0:] = tau
            if collect_expected:
                EM_model[d, k0:k1] = np.maximum(
                    N_eval[:, k0:k1] - Q[d, k0:k1][None, :], 0.0).mean(0)

            bv = S7.build_value_functions(r_node, price[k0:], nuv, delta)
            R_seg = bv["R"][: (k1 - k0)]
            if tau == 0:
                Rmat[d] = bv["R"]
            N_act_seg = f["N_actual_all"][d, k0:k1]
            q_seg = Q[d, k0:k1]
            if executor == "dp":
                ex = S7.dp_execute(R_seg, N_act_seg, q_seg, e_node)
            else:
                ex = S7.analytic_execute(N_act_seg, q_seg, e_node)
            Cch[d, k0:k1] = ex["C"]
            Dh[d, k0:k1] = ex["D"]
            Bm[d, k0:k1] = ex["b"]
            Curt[d, k0:k1] = ex["U"]
            Em[d, k0:k1] = ex["E"]
            E_out_node[d, ti] = float(ex["E"][-1])
            e_node = float(ex["E"][-1])
            if with_analytic:
                an = S7.analytic_execute(N_act_seg, q_seg, e_node_an)
                Can[d, k0:k1] = an["C"]
                Dan[d, k0:k1] = an["D"]
                Ban[d, k0:k1] = an["b"]
                CurtA[d, k0:k1] = an["U"]
                Ean[d, k0:k1] = an["E"]
                e_node_an = float(an["E"][-1])
        for ti_n in range(1, n_node):
            E_in_node[d, ti_n] = E_out_node[d, ti_n - 1]
        e = float(Em[d, -1])
        e_an = float(Ean[d, -1])
        if progress_every and ((i + 1) % progress_every == 0 or i == len(days) - 1):
            if log is not None:
                el = time.perf_counter() - t00
                log(f"  {i + 1}/{len(days)} 天  用时 {el:.1f} s  "
                    f"预计剩余 {el / (i + 1) * (len(days) - i - 1) / 60:.1f} 分钟  "
                    f"日末库存 {e:.1f} kWh")

    sc = np.asarray(days, int)
    dec = decompose(price, P[sc], Q[sc], Bm[sc])
    exp_dec = decompose(price, P[sc], Q[sc], EM_model[sc])

    E_day_start_real = E_in_node[sc, 0].copy()
    E_00_plan_input = E_in_node[sc, 0].copy()
    E_day_end_real = Em[sc, -1].copy()
    nxt = np.empty_like(E_day_end_real)
    nxt[:-1] = E_day_start_real[1:]
    nxt[-1] = np.nan
    E_next_day_start_real = nxt

    resid = _residuals(f, sc, P, Q, Bm, Cch, Dh, Curt, Em, E_in_node,
                       E_out_node, SRC, SRC_MIN,
                       e_init=float(C.E_INIT if e_init is None else e_init),
                       E_plan_chain=E_plan_chain)

    out = {
        "update_times": tuple(update_times),
        "nodes": nodes,
        "segments": segs,
        "days": sc,
        "dates": [f["dates"][d] for d in sc],
        "score_day_index": sc,
        "price": price,
        "P": P, "Q": Q, "SRC": SRC, "SRC_MIN": SRC_MIN,
        "N_actual": f["N_actual_all"],
        "C": Cch, "D": Dh, "b": Bm, "curt": Curt, "E": Em,
        "C_analytic": Can, "D_analytic": Dan, "b_analytic": Ban,
        "curt_analytic": CurtA, "E_analytic": Ean,
        "E_in_node": E_in_node, "E_out_node": E_out_node,
        "E_plan_chain_virtual": E_plan_chain,
        "E_day_start_real": E_day_start_real,
        "E_00_plan_input": E_00_plan_input,
        "E_day_end_real": E_day_end_real,
        "E_next_day_start_real": E_next_day_start_real,
        "R": Rmat,
        "dec": dec,
        "exp_dec": exp_dec,
        "EM_model": EM_model,
        "resid": resid,
        "info": {
            "executor": executor,
            "stage0_mode": stage0_mode,
            "stage0_recourse": bool(stage0_recourse),
            "forecast_mode": forecast_mode,
            "perfect_taus": tuple(sorted(_seed)),
            "perfect_force": bool(perfect_force),
            "n_forced_seg": int(n_forced),
            "m_scenarios": mm,
            "nu": nuv,
            "delta_kwh": float(delta),
            "eval_full": bool(eval_full),
            "m_lib": int(_M_LIB),
            "s_period_kwh": C.S_PERIOD_KWH,
            "n_days": int(sc.size),
            "n_node": n_node,
            "lp_stats": lp_stats,
            "grid_size": int(S7.make_grid(delta).size),
            "elapsed_s": time.perf_counter() - t00,
            "n_declined_seg": int(n_declined),
            "n_declined_perf_seg": int(n_declined_perf),
            "expected_measure": ("全情景集（P1-3 收敛性检验用同一把尺子）" if eval_full
                                 else "τ=0 (0:00) 情景集（共同 ex-ante 测度）"),
        },
    }
    return out


def _residuals(f, sc, P, Q, Bm, Cch, Dh, Curt, Em, E_in_node, E_out_node,
               SRC, SRC_MIN, e_init, E_plan_chain=None) -> dict:
    r = {}

    if E_in_node.shape[1] >= 2 and np.any(np.isfinite(E_in_node[sc, 1:])):
        r["node_chain"] = float(
            np.nanmax(np.abs(E_in_node[sc, 1:] - E_out_node[sc, :-1])))
    else:
        r["node_chain"] = 0.0

    if sc.size >= 2:
        r["cross_day"] = float(
            np.nanmax(np.abs(E_in_node[sc[1:], 0] - Em[sc[:-1], -1])))
    else:
        r["cross_day"] = 0.0

    r["plan_input_vs_real"] = 0.0
    if E_plan_chain is None:
        r["plan_chain_gap_max"] = float("nan")
        r["plan_chain_gap_mean"] = float("nan")
    elif sc.size < 2:
        r["plan_chain_gap_max"] = 0.0
        r["plan_chain_gap_mean"] = 0.0
    else:
        gap = np.abs(E_plan_chain[sc[:-1]] - Em[sc[:-1], -1])
        r["plan_chain_gap_max"] = float(np.nanmax(gap))
        r["plan_chain_gap_mean"] = float(np.nanmean(gap))

    E = Em[sc]
    Cp = Cch[sc]
    Dp = Dh[sc]
    rec = E[:, 1:] - E[:, :-1] - C.ETA * Cp[:, 1:] + Dp[:, 1:] / C.ETA
    first = (E[:, :1] - E_in_node[sc, 0][:, None]
             - C.ETA * Cp[:, :1] + Dp[:, :1] / C.ETA)
    r["soc_recursion"] = float(max(np.abs(rec).max(), np.abs(first).max()))

    bal = f["N_actual_all"][sc] - Q[sc] - Dp - Bm[sc] + Cp + Curt[sc]
    r["balance"] = float(np.abs(bal).max())

    r["power_cap"] = float(max(np.maximum(Cp, 0).max() - C.S_PERIOD_KWH,
                               np.maximum(Dp, 0).max() - C.S_PERIOD_KWH))

    r["soc_lower"] = float(C.E_MIN - min(float(E.min()),
                                         float(E_in_node[sc, 0].min())))
    r["soc_upper"] = float(float(E.max()) - C.E_MAX)

    r["simul_charge_discharge"] = int(np.sum((Cp > 1e-9) & (Dp > 1e-9)))

    both = (Bm[sc] > 1e-9) & (Cp > 1e-9)
    r["emerg_with_charge_cells"] = int(both.sum())
    r["emerg_with_charge_days"] = int(np.sum(both.sum(axis=1) > 0))
    r["emerg_days"] = int(np.sum(Bm[sc].sum(axis=1) > 1e-9))
    r["emerg_cells"] = int(np.sum(Bm[sc] > 1e-9))

    S = SRC[sc]
    tt = np.arange(T)[None, :]
    used = np.where(S >= 0, S, 10 ** 6)
    r["info_leak_cells"] = int(np.sum((used * 6 > tt) & (S >= 0)))
    r["info_src_undef"] = int(np.sum(S < 0))
    r["info_min_gt_src"] = int(np.sum(
        (SRC_MIN[sc] >= 0) & (S >= 0) & (SRC_MIN[sc] > S)))
    r["info_nodes_seen"] = int(np.unique(S[S >= 0]).size)

    r["max_abs"] = float(max(abs(r["node_chain"]), abs(r["cross_day"]),
                             r["soc_recursion"], r["balance"]))
    return r


_ARM_ALIAS = {"S_all+_UB": "S_all_plus", "S_all+": "S_all_plus"}


def run_arm(arm: str, *, log=None, **kw) -> dict:
    arm = _ARM_ALIAS.get(arm, arm)
    if arm not in C.ARM_TAUS:
        raise ValueError(f"未知档位 {arm!r}，可选 {tuple(C.ARM_TAUS)}")
    taus = tuple(C.ARM_TAUS[arm])
    if arm == "S_all_plus":
        return run_policy(update_times=taus, perfect_taus=(18,), log=log, **kw)
    if arm == "S_all+_raw":
        return run_policy(update_times=taus, perfect_taus=(18,),
                          perfect_force=True, log=log, **kw)
    if arm == "PF":
        return run_policy(update_times=taus, forecast_mode="actual",
                          log=log, **kw)
    return run_policy(update_times=taus, log=log, **kw)


def summarise(res: dict, *, label: str = "") -> dict:
    sc = res["score_day_index"]
    dec = res["dec"]
    exp = res["exp_dec"]
    P = res["P"]; Q = res["Q"]; Bm = res["b"]
    u = np.maximum(P - Q, 0.0); v = np.maximum(Q - P, 0.0)
    N = res["N_actual"]; Em = res["E"]
    tot_exp = float(exp["total"].sum())
    tot_imp = float(dec["total"].sum())
    return {
        "档位": label or "-",
        "更新时刻": ",".join(str(t) for t in res["update_times"]) or "无(仅0:00)",
        "总费用_元": tot_imp,
        "模型内期望总费用_元": tot_exp,
        "实现减期望_元": tot_imp - tot_exp,
        "计划购电费_元": float(dec["plan"].sum()),
        "下调违约金_元": float(dec["down"].sum()),
        "上调加价_元": float(dec["up"].sum()),
        "调整相关费用_元": float(dec["down"].sum() + dec["up"].sum()),
        "紧急购电费_元": float(dec["emerg"].sum()),
        "计划购电量_kWh": float(P[sc].sum()),
        "调整购电量_kWh": float(Q[sc].sum()),
        "下调量_kWh": float(u[sc].sum()),
        "上调量_kWh": float(v[sc].sum()),
        "紧急购电量_kWh": float(Bm[sc].sum()),
        "实际净负荷_kWh": float(N[sc].sum()),
        "充电量_kWh": float(res["C"][sc].sum()),
        "放电量_kWh": float(res["D"][sc].sum()),
        "弃电量_kWh": float(res["curt"][sc].sum()),
        "日末库存_末值_kWh": float(Em[sc[-1], -1]),
        "均价_元每kWh": tot_imp / float(N[sc].sum()),
        "闭环残差最大值": float(res["resid"]["max_abs"]),
        "信息泄露时段数": int(res["resid"]["info_leak_cells"]),
    }
