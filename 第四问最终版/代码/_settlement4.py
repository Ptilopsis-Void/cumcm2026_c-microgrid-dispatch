#!/usr/bin/env python3
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


C = _load("_comm4.py", "q4_comm")
PO = _load("_policy4.py", "q4_policy")

import numpy as np

T = C.PERIODS_PER_DAY
K0 = C.TAU_PERIOD_INDEX
N_TAU = len(C.TAU_HOURS)
N_DAY = 365
NODE_BOUNDS = [(0, 6), (6, 12), (12, 18), (18, 24)]


def bill_42_slots(c, g, b) -> dict:
    c = np.asarray(c, float)
    g = np.asarray(g, float)
    b = np.asarray(b, float)
    plan = float(np.sum(c * g))
    emerg = float(5.0 * np.sum(c * b))
    return {"plan": plan, "adjust": 0.0, "emerg": emerg, "total": plan + emerg}


def bill_43_slots(c, g, a, b) -> dict:
    c = np.asarray(c, float)
    g = np.asarray(g, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    base = float(np.sum(c * np.minimum(g, a)))
    up = float(0.5 * np.sum(c * np.maximum(g - a, 0.0)))
    dn = float(1.5 * np.sum(c * np.maximum(a - g, 0.0)))
    emerg = float(5.0 * np.sum(c * b))
    return {"plan": base, "adjust": up + dn, "emerg": emerg,
            "total": base + up + dn + emerg}


def bill_43_equiv(c, g, a, b) -> float:
    c = np.asarray(c, float)
    g = np.asarray(g, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sum(c * (a + 0.5 * np.abs(a - g) + 5.0 * b)))


def minimal_bill_test(tol: float = 1e-9) -> dict:
    c = np.ones(1)
    g = np.array([100.0])
    b0 = np.zeros(1)
    cases = {}
    for a_val, want in ((100.0, 100.0), (80.0, 90.0), (120.0, 130.0)):
        got = bill_43_slots(c, g, np.array([a_val]), b0)["total"]
        cases[a_val] = {"got": got, "want": want, "ok": abs(got - want) <= tol}
    for a_val, want in ((100.0, 150.0), (80.0, 140.0), (120.0, 180.0)):
        got = bill_43_slots(c, g, np.array([a_val]), np.array([10.0]))["total"]
        cases[f"b10_a{a_val}"] = {"got": got, "want": want, "ok": abs(got - want) <= tol}
    got = bill_43_slots(c, g, np.array([90.0]), b0)["total"]
    cases["chain_final_a90"] = {"got": got, "want": 95.0, "ok": abs(got - 95.0) <= tol}
    for a_val in (70.0, 100.0, 140.0):
        seg = bill_43_slots(c, g, np.array([a_val]), np.array([3.0]))["total"]
        eqv = bill_43_equiv(c, g, np.array([a_val]), np.array([3.0]))
        cases[f"equiv_a{a_val}"] = {"got": eqv, "want": seg, "ok": abs(eqv - seg) <= tol}
    return {"cases": cases, "ok": all(v["ok"] for v in cases.values())}


class SnapshotProviders:

    def __init__(self, branch: str):
        self.branch = branch
        Z4 = np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False)
        self.dates = np.array([str(s) for s in Z4["dates"]], dtype="<U10")
        self.price_actual = np.asarray(Z4["price_actual"], float)
        Zf = np.load(C.PRICE_FORECAST_NPZ, allow_pickle=False)
        method = str(Zf[f"adopted_{branch}"][0])
        self.method = method
        self.chat = np.asarray(Zf[f"chat_corr__{method}"], float)
        self.chat_base = np.asarray(Zf[f"chat_base__{method}"], float)
        self.nhat_snap = np.asarray(Zf["nhat_snap"], float)
        Zj = np.load(C.JOINT_SCENARIO_NPZ, allow_pickle=False)
        self._J = {k: Zj[k] for k in Zj.files}
        Z2 = C.load_q2_matrix()
        self.net_act = np.asarray(Z2["net_load_energy_kwh"], float)
        self.load_act = np.asarray(Z2["load_energy_kwh"], float)
        self.pv_act = np.asarray(Z2["pv_energy_kwh"], float)

    def price(self, d: int, t: int) -> float:
        return float(self.price_actual[d, t])

    def forecast(self, d: int, ti: int, H: int) -> np.ndarray:
        return self.chat[d, ti, :H].copy()

    def scenarios(self, d: int, ti: int) -> dict:
        idx = d * N_TAU + ti
        M = int(self._J[f"M__{self.branch}"][idx])
        H = int(self._J["H"][d, ti])
        if M <= 0 or H <= 0:
            raise ValueError(f"节点 ({d},{ti}) 无合法情景")
        return {"N": self._J[f"N_scen__{self.branch}"][idx, :M, :H].copy(),
                "c": self._J[f"price_scen__{self.branch}"][idx, :M, :H].copy(),
                "w": np.full(M, 1.0 / M),
                "origin": self._J[f"origin__{self.branch}"][idx, :M].copy(),
                "M": M, "H": H,
                "fallback": int(self._J[f"fallback__{self.branch}"][idx])}


def plan_node(mode, scen_N, scen_c, w, E0, nu, g_today=None, fix_a=None,
              max_iter: int = 4, tol: float = 1e-9):
    M, H = scen_N.shape
    nt = H if mode == "midnight" else int(np.asarray(g_today).size)

    def _solve(caps, pin):
        if mode == "midnight":
            return PO.solve_midnight_plan(scen_N, scen_c, w, E0, nu,
                                          cap_C=caps, fix_a=pin)
        return PO.solve_revised_plan(scen_N, scen_c, w, g_today, E0, nu,
                                     fix_a=pin, cap_C=caps)

    def _check(res, a_use):
        cap = np.maximum(0.0, np.asarray(a_use, float)[None, :] - scen_N[:, :nt])
        Cc = np.asarray(res["C"], float)[:, :nt]
        viol = float(np.maximum(0.0, Cc - cap).max())
        if viol > 1e-6:
            raise AssertionError(f"规划层一致性上界被违反 {viol:.3e} kWh")
        return viol

    def _flow(res):
        Cc = np.asarray(res["C"], float)[:, :nt]
        Dd = np.asarray(res["D"], float)[:, :nt]
        return {"sum_C": float(Cc.sum()), "sum_D": float(Dd.sum())}

    info = {"mode": mode, "nt": int(nt), "M": int(M), "iters": 0,
            "converged": False, "pinned_final": False, "fun_uncapped": None,
            "fun": None, "relax_gap": None, "cap_sum": None, "n_lp": 0,
            "cap_viol": 0.0, "sum_C": None, "sum_D": None}

    if fix_a is not None:
        a0 = np.asarray(fix_a, float).reshape(-1)
        caps = np.maximum(0.0, a0[None, :] - scen_N[:, :nt])
        res = _solve(caps, a0)
        if res["status"] != 0:
            raise RuntimeError(f"节点规划 LP 失败：{res['message']}")
        info.update(iters=1, converged=True, fun=float(res["fun"]),
                    cap_sum=float(caps.sum()), n_lp=1)
        info.update(_flow(res))
        info["cap_viol"] = _check(res, a0)
        return res, a0, info

    res0 = _solve(None, None)
    if res0["status"] != 0:
        raise RuntimeError(f"节点规划 LP（无上界）失败：{res0['message']}")
    f0 = float(res0["fun"])
    info["fun_uncapped"] = f0
    a = np.asarray(res0["a"], float).reshape(-1)
    best = (float("inf"), a, None)

    for k in range(1, max_iter + 1):
        caps = np.maximum(0.0, a[None, :] - scen_N[:, :nt])
        res = _solve(caps, None)
        if res["status"] != 0:
            raise RuntimeError(f"节点规划 LP（含一致性上界）失败：{res['message']}")
        a_new = np.asarray(res["a"], float).reshape(-1)
        f = float(res["fun"])
        info["iters"] = k
        if f < best[0]:
            best = (f, a_new, caps)
        if float(np.max(np.abs(a_new - a))) <= tol:
            info.update(converged=True, fun=f, cap_sum=float(caps.sum()),
                        n_lp=1 + k)
            info.update(_flow(res))
            info["relax_gap"] = (f - f0) / max(1.0, abs(f0))
            info["cap_viol"] = _check(res, a_new)
            return res, a_new, info
        a = a_new

    a_fix = best[1]
    caps = np.maximum(0.0, a_fix[None, :] - scen_N[:, :nt])
    res = _solve(caps, a_fix)
    if res["status"] != 0:
        raise RuntimeError(f"节点规划 LP（固定 a 重解）失败：{res['message']}")
    info.update(converged=False, pinned_final=True, fun=float(res["fun"]),
                cap_sum=float(caps.sum()), n_lp=2 + max_iter)
    info.update(_flow(res))
    info["relax_gap"] = (float(res["fun"]) - f0) / max(1.0, abs(f0))
    a_ret = np.asarray(res["a"], float).reshape(-1)
    info["cap_viol"] = _check(res, a_ret)
    return res, a_ret, info


def node_nu(ti: int, sc: dict, H: int, tail: bool = True,
            nu_day: float | None = None) -> float:
    if not tail:
        if nu_day is None:
            raise ValueError("日末截断口径下必须显式传入 nu_day（当日 00:00 可见的谷段预测均价）")
        return float(nu_day)
    n = C.NU_PRICE_POINTS
    k0 = K0[ti]
    if ti == 0:
        return PO.nu_tau(0, sc["c"][:, :n])
    start = T - k0
    if start + n > H:
        return 0.0
    return PO.nu_tau(ti, None, sc["c"][:, start:start + n])


def run_policy4(branch, price_provider, forecast_provider, config=None,
                initial_state=float(C.E_INIT), days=None, log=None):
    cfg = {"delta": None, "nu_scale": 1.0, "value_nodes": (0, 6, 12, 18),
           "tail": False, "grid_1d": True, "cap_iter": 4, "verify_exec": False}
    cfg.update(config or {})
    branch = str(branch)
    if branch not in ("42", "43"):
        raise ValueError("branch 必须为 '42' 或 '43'")
    if days is None:
        days = np.arange(C.N_WARMUP_DAYS, N_DAY)

    prov = forecast_provider
    n_days = len(days)
    rec = {k: np.zeros((n_days, T)) for k in ("g", "a", "b", "C", "D", "U", "c", "r", "R")}
    rec["E_end"] = np.zeros(n_days)
    rec["plan_cost"] = np.zeros(n_days)
    rec["adjust_cost"] = np.zeros(n_days)
    rec["emerg_cost"] = np.zeros(n_days)
    rec["bill"] = np.zeros(n_days)
    node_rows = []
    accept_rows = []
    E = float(initial_state)
    E_start = E
    n_lp = 0
    lp_status = set()
    exec_gap = 0.0
    exec_gap_info = None
    vg = 0

    exec_done = np.zeros((n_days, T), bool)
    exec_cnt = np.zeros((n_days, T), np.int64)

    for di, d in enumerate(days):
        d = int(d)
        c_day = np.asarray(price_provider.price_actual[d], float)
        N_day = np.asarray(prov.net_act[d], float)
        g_day = np.zeros(T)
        a_day = np.zeros(T)
        act = {k: np.zeros(T) for k in ("b", "C", "D", "U", "R", "r")}
        Hbar0 = None
        grid0 = None
        value_origin = 0

        for ti in range(N_TAU):
            k0 = K0[ti]
            sc = prov.scenarios(d, ti)
            M, H_full = sc["M"], sc["H"]
            H = H_full if cfg["tail"] else min(int(H_full), int(T - k0))
            if H <= 0:
                raise RuntimeError(f"节点 ({prov.dates[d]}, τ={C.TAU_HOURS[ti]}) 展望长度为 0")
            nt = min(H, T - k0)
            nz = H - nt
            if not cfg["tail"] and nz != 0:
                raise AssertionError("截断口径下不应存在尾段")

            n_exec = min((K0[ti + 1] if ti + 1 < N_TAU else T) - k0, nt)
            E_in = E
            skip = (ti > 0) and (C.TAU_HOURS[ti] not in cfg["value_nodes"])

            if ti == 0:
                nu_day = PO.nu_tau(0, sc["c"][:, :C.NU_PRICE_POINTS])
                nu = nu_day * float(cfg["nu_scale"])
                scN, scC = sc["N"][:, :H], sc["c"][:, :H]
                res, a_node, pinfo = plan_node("midnight", scN, scC, sc["w"],
                                               E, nu, max_iter=cfg["cap_iter"])
                n_lp += pinfo["n_lp"]
                lp_status.add(int(res["status"]))
                g_node = np.asarray(res["g"], float)
                plan = np.broadcast_to(a_node, (M, nt)).copy()
                dg = PO.degeneracy_diag(res)
                vf = PO.build_value_functions(scN, plan[:, :H], scC, nu,
                                              delta=cfg["delta"])
                Hbar, grid = vf["Hbar"], vf["grid"]
                Hbar0 = Hbar
                grid0 = grid
                value_origin = 0
                g_day[:nt] = g_node
                a_day = g_day.copy()
            elif skip:
                nu = float("nan")
                a_node = a_day[k0:k0 + nt].copy()
                g_node = g_day[k0:k0 + nt].copy()
                Hbar, grid = Hbar0, grid0
                pinfo = {"iters": 0, "converged": True, "pinned_final": False,
                         "relax_gap": 0.0, "fun": float("nan"), "n_lp": 0,
                         "cap_viol": 0.0, "fun_uncapped": float("nan")}
                dg = {"ok": True, "sim_charge_discharge": 0.0,
                      "sim_emergency_charge_inplan": 0.0, "skipped": True}
            else:
                nu = node_nu(ti, sc, H, tail=cfg["tail"],
                             nu_day=nu_day) * float(cfg["nu_scale"])
                scN, scC = sc["N"][:, :H], sc["c"][:, :H]
                g_node = g_day[k0:k0 + nt].copy()
                if branch == "42":
                    res, a_node, pinfo = plan_node("revise", scN, scC, sc["w"],
                                                   E, nu, g_today=g_node, fix_a=g_node)
                    n_lp += pinfo["n_lp"]
                    lp_status.add(int(res["status"]))
                    z = np.asarray(res["z"], float) if res.get("z") is not None \
                        else np.zeros((M, nz))
                    dg = PO.degeneracy_diag(res)
                else:
                    a_old = a_day[k0:k0 + nt].copy()
                    res_old, _, pinfo_old = plan_node("revise", scN, scC, sc["w"],
                                                      E, nu, g_today=g_node, fix_a=a_old)
                    n_lp += pinfo_old["n_lp"]
                    lp_status.add(int(res_old["status"]))
                    res, a_new, pinfo = plan_node("revise", scN, scC, sc["w"],
                                                  E, nu, g_today=g_node,
                                                  max_iter=cfg["cap_iter"])
                    n_lp += pinfo["n_lp"]
                    lp_status.add(int(res["status"]))
                    f_new = float(res["fun"])
                    f_old = float(res_old["fun"])
                    tol = 1e-6 * max(1.0, abs(f_old))
                    accepted = bool(f_new <= f_old + tol)
                    a_node = a_new if accepted else a_old
                    z_src = res if accepted else res_old
                    z = np.asarray(z_src["z"], float) if z_src.get("z") is not None \
                        else np.zeros((M, nz))
                    dg = PO.degeneracy_diag(res)
                    accept_rows.append({
                        "date": str(prov.dates[d]), "tau": int(C.TAU_HOURS[ti]),
                        "fun_old": f_old, "fun_new": f_new, "accepted": accepted,
                        "d_a_l1": float(np.abs(a_node - a_old).sum()),
                        "d_a_max": float(np.abs(a_node - a_old).max()) if nt else 0.0,
                        "M": M, "iters_old": pinfo_old["iters"],
                        "iters_new": pinfo["iters"],
                        "converged_new": pinfo["converged"]})
                plan = np.empty((M, nt + nz)) if nz else np.broadcast_to(a_node, (M, nt)).copy()
                if nz:
                    plan[:, :nt] = a_node[None, :]
                    plan[:, nt:] = z
                vf = PO.build_value_functions(scN, plan[:, :H], scC, nu,
                                              delta=cfg["delta"])
                Hbar, grid = vf["Hbar"], vf["grid"]
                value_origin = k0

            for h in range(n_exec):
                t = k0 + h
                j = t - value_origin + 1
                if not (0 <= j < Hbar.shape[0]):
                    raise AssertionError(
                        f"价值函数下标越界：t={t} origin={value_origin} j={j} "
                        f"H={Hbar.shape[0] - 1}")
                E_before = E
                ex = PO.execute_one_slot(E, float(a_node[h]), float(c_day[t]),
                                         float(N_day[t]), Hbar[j], grid)
                if cfg["verify_exec"] and ex["r"] > 0.0:
                    Dmax = min(ex["r"], C.S_PERIOD_KWH, C.ETA * (E_before - C.E_MIN))
                    cand = np.asarray(grid[grid <= E_before + 1e-12], float)
                    Dc = C.ETA * (E_before - cand)
                    msk = Dc <= Dmax + 1e-9
                    cand, Dc = cand[msk], Dc[msk]
                    if cand.size:
                        intr = np.asarray(PO._interp_rows(grid, Hbar[j], cand),
                                          float).reshape(-1)
                        obj = 5.0 * float(c_day[t]) * (ex["r"] - Dc) + intr
                        jbest = int(np.argmin(obj))
                        best = float(obj.reshape(-1)[jbest])
                        rule = 5.0 * float(c_day[t]) * ex["b"] + float(np.asarray(
                            PO._interp_rows(grid, Hbar[j],
                                            np.asarray([ex["E_next"]], float)),
                            float).reshape(-1)[0])
                        vg += 1
                        if rule - best > exec_gap:
                            exec_gap = rule - best
                            exec_gap_info = (str(prov.dates[d]), int(C.TAU_HOURS[ti]),
                                             int(h), float(ex["r"]), float(E_before),
                                             float(ex["E_next"]), float(cand[jbest]))
                E = float(ex["E_next"])
                act["b"][t] = ex["b"]
                act["C"][t] = ex["C"]
                act["D"][t] = ex["D"]
                act["U"][t] = ex["U"]
                act["R"][t] = ex["R"]
                act["r"][t] = ex["r"]
                a_day[t] = a_node[h]
                exec_done[di, t] = True
                exec_cnt[di, t] += 1
            node_rows.append({"date": str(prov.dates[d]), "tau": int(C.TAU_HOURS[ti]),
                              "M": M, "H": H, "H_full": int(H_full), "nu": nu,
                              "value_origin": int(value_origin),
                              "tail": bool(cfg["tail"]), "E_in": E_in, "skip": bool(skip),
                              "deg_ok": bool(dg["ok"]),
                              "deg_cd": float(dg["sim_charge_discharge"]),
                              "deg_bc": float(dg.get("sim_emergency_charge_inplan", 0.0)),
                              "fallback": int(sc["fallback"]),
                              "cap_iters": int(pinfo["iters"]),
                              "cap_converged": bool(pinfo["converged"]),
                              "cap_pinned": bool(pinfo["pinned_final"]),
                              "cap_viol": float(pinfo.get("cap_viol", 0.0)),
                              "relax_gap": pinfo["relax_gap"],
                              "fun": pinfo["fun"],
                              "fun_uncapped": pinfo["fun_uncapped"],
                              "sum_C": pinfo.get("sum_C"),
                              "sum_D": pinfo.get("sum_D")})

        if branch == "42":
            bl = bill_42_slots(c_day, g_day, act["b"])
        else:
            bl = bill_43_slots(c_day, g_day, a_day, act["b"])
            eq = bill_43_equiv(c_day, g_day, a_day, act["b"])
            if abs(eq - bl["total"]) > 1e-6 * max(1.0, abs(bl["total"])):
                raise RuntimeError(f"{prov.dates[d]} 4-3 账单等价式不成立")
        for k, v in (("g", g_day), ("a", a_day), ("b", act["b"]), ("C", act["C"]),
                     ("D", act["D"]), ("U", act["U"]), ("c", c_day), ("r", act["r"]),
                     ("R", act["R"])):
            rec[k][di] = v
        rec["E_end"][di] = E
        rec["plan_cost"][di] = bl["plan"]
        rec["adjust_cost"][di] = bl["adjust"]
        rec["emerg_cost"][di] = bl["emerg"]
        rec["bill"][di] = bl["total"]

    out = {
        "branch": branch, "days": np.asarray(days, int),
        "dates": prov.dates[np.asarray(days, int)],
        "method": getattr(prov, "method", None),
        "E_start": E_start, "config": cfg,
        "tail": bool(cfg["tail"]),
        "horizon_note": ("day_end_truncated" if not cfg["tail"]
                         else "24h_cross_day(historical_reference)"),
        "n_lp": int(n_lp), "lp_status_codes": sorted(int(s) for s in lp_status),
        "node_rows": node_rows, "accept_rows": accept_rows,
        "exec_gap": float(exec_gap), "exec_gap_info": exec_gap_info,
        "exec_checked": int(vg), "exec_done": exec_done, "exec_cnt": exec_cnt,
        "n_unexec": int((~exec_done).sum()),
    }
    out.update(rec)
    out["E_final"] = float(E)
    out["cost_total"] = float(rec["bill"].sum())
    out["cost_plan"] = float(rec["plan_cost"].sum())
    out["cost_adjust"] = float(rec["adjust_cost"].sum())
    out["cost_emerg"] = float(rec["emerg_cost"].sum())
    return out
