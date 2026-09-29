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
P = _load("_price4.py", "q4_price")

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

T = C.PERIODS_PER_DAY
K0 = C.TAU_PERIOD_INDEX
N_TAU = len(C.TAU_HOURS)
S = C.S_PERIOD_KWH
ETA = C.ETA
E_MIN, E_MAX = C.E_MIN, C.E_MAX
EPS_THROUGHPUT = 1e-9
LP_TOL = 1e-9


def _lp_options() -> dict:
    return {"primal_feasibility_tolerance": LP_TOL,
            "dual_feasibility_tolerance": LP_TOL}


def make_grid(delta: float = None) -> np.ndarray:
    delta = float(delta or C.SOC_GRID_KWH)
    n = int(round((E_MAX - E_MIN) / delta))
    return np.linspace(E_MIN, E_MAX, n + 1)


def _interp_rows(grid, vals, e_new):
    g = grid.ravel()
    delta = g[1] - g[0]
    n = g.size
    pos = np.clip((e_new - g[0]) / delta, 0.0, n - 1.0)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
    fr = np.clip(pos - i0, 0.0, 1.0)
    vals = vals.reshape(-1, n)
    rows = np.arange(vals.shape[0])[:, None]
    return vals[rows, i0] * (1.0 - fr) + vals[rows, i0 + 1] * fr


def max_argmin(a: np.ndarray) -> int:
    return a.size - 1 - int(np.argmin(a[::-1]))


def build_value_functions(scen_N: np.ndarray, plan: np.ndarray, scen_c: np.ndarray,
                          nu: float, delta: float = None) -> dict:
    M_, H = scen_N.shape
    grid = make_grid(delta)
    n = grid.size
    r = scen_N - plan

    Hs = np.tile(-nu * grid, (M_, 1))
    Hbar = np.empty((H + 1, n))
    Hbar[H] = -nu * grid
    jall = np.arange(n)[None, :]
    rows = np.arange(M_)[:, None]

    for t in range(H - 1, -1, -1):
        cc = 5.0 * scen_c[:, t]
        rt = r[:, t]

        step = float(grid[1] - grid[0])
        tm = np.minimum(S, np.maximum(rt, 0.0)) / ETA
        F = Hs + cc[:, None] * ETA * grid[None, :]
        jstar = (n - 1) - np.argmin(F[:, ::-1], axis=1)
        m = np.floor(tm / step + 1e-12).astype(np.int64)
        lo = np.maximum(0, jall - m[:, None])
        jsel = np.minimum(np.maximum(jstar[:, None], lo), jall)
        new_def = F[rows, jsel] - cc[:, None] * ETA * grid[None, :] \
            + cc[:, None] * rt[:, None]
        e_lo = np.maximum(grid[0], grid[None, :] - tm[:, None])
        new_def = np.minimum(
            new_def,
            _interp_rows(grid, Hs, e_lo) + cc[:, None] * ETA * e_lo
            - cc[:, None] * ETA * grid[None, :] + cc[:, None] * rt[:, None])

        s = ETA * np.minimum(S, np.maximum(-rt, 0.0))
        enew = np.minimum(grid[None, :] + s[:, None], E_MAX)
        new_chg = _interp_rows(grid, Hs, enew)

        Hs = np.where((rt <= 0.0)[:, None], new_chg, new_def)
        Hbar[t] = Hs.mean(axis=0)

    return {"Hbar": Hbar, "grid": grid, "r": r}


def retention_level(Hbar_next: np.ndarray, c_now: float, grid: np.ndarray) -> float:
    G = 5.0 * float(c_now) * ETA * grid + Hbar_next
    return float(grid[max_argmin(G)])


def execute_one_slot(E: float, a_t: float, c_now: float, N_now: float,
                     Hbar_next: np.ndarray, grid: np.ndarray) -> dict:
    r = float(N_now) - float(a_t)
    Rv = float("nan")
    if r > 0.0:
        Rv = retention_level(Hbar_next, c_now, grid)
        D = max(0.0, min(r, S, ETA * (E - Rv)))
        b = r - D
        Cc = 0.0
        U = 0.0
    else:
        Cc = max(0.0, min(-r, S, (E_MAX - E) / ETA))
        U = max(0.0, -r - Cc)
        D = 0.0
        b = 0.0
    E_next = E + ETA * Cc - D / ETA
    return {"C": Cc, "D": D, "b": b, "U": U, "E_next": E_next, "R": Rv, "r": r}


class NodePlanLP:

    _cache: dict = {}

    def __init__(self, H: int, n_today: int, M: int, mode: str):
        self.H, self.n_today, self.M, self.mode = H, n_today, M, mode
        self.n_tail = H - n_today
        self.rev = (mode == "revise")
        nt = n_today
        self.off_a = 0
        self.off_up = nt if self.rev else -1
        self.off_dn = 2 * nt if self.rev else -1
        self.off_z = 3 * nt if self.rev else nt
        self.off_b = self.off_z + M * self.n_tail
        self.off_C = self.off_b + M * H
        self.off_D = self.off_C + M * H
        self.off_E = self.off_D + M * H
        self.nvar = self.off_E + M * H
        self._build()

    def _build(self):
        H, M, nt, nz = self.H, self.M, self.n_today, self.n_tail
        rows, cols, vals = [], [], []
        for w in range(M):
            for u in range(H):
                rr = w * H + u
                if u < nt:
                    rows.append(rr); cols.append(self.off_a + u); vals.append(-1.0)
                else:
                    rows.append(rr); cols.append(self.off_z + w * nz + (u - nt)); vals.append(-1.0)
                rows.append(rr); cols.append(self.off_b + rr); vals.append(-1.0)
                rows.append(rr); cols.append(self.off_D + rr); vals.append(-1.0)
                rows.append(rr); cols.append(self.off_C + rr); vals.append(1.0)
        self._brows, self._bcols, self._bvals = rows, cols, vals

        rrows, rcols, rvals = [], [], []
        for w in range(M):
            for u in range(H):
                rr = w * H + u
                rrows.append(rr); rcols.append(self.off_E + w * H + u); rvals.append(1.0)
                if u > 0:
                    rrows.append(rr); rcols.append(self.off_E + w * H + u - 1); rvals.append(-1.0)
                rrows.append(rr); rcols.append(self.off_C + w * H + u); rvals.append(-ETA)
                rrows.append(rr); rcols.append(self.off_D + w * H + u); rvals.append(1.0 / ETA)
        self._erows, self._ecols, self._evals = rrows, rcols, rvals

        self.n_eq_adj = nt if self.rev else 0
        if self.n_eq_adj:
            arows, acols, avals = [], [], []
            for u in range(nt):
                rr = u
                arows.append(rr); acols.append(self.off_a + u); avals.append(1.0)
                arows.append(rr); acols.append(self.off_up + u); avals.append(-1.0)
                arows.append(rr); acols.append(self.off_dn + u); avals.append(1.0)
            A_eq = vstack([csr_matrix((rvals, (rrows, rcols)),
                                      shape=(M * H, self.nvar)),
                           csr_matrix((avals, (arows, acols)),
                                      shape=(self.n_eq_adj, self.nvar))]).tocsr()
        else:
            A_eq = csr_matrix((rvals, (rrows, rcols)), shape=(M * H, self.nvar))

        self.A_ub = csr_matrix((vals, (rows, cols)), shape=(M * H, self.nvar))
        self.A_eq = A_eq

        lb = np.zeros(self.nvar)
        ub = np.full(self.nvar, np.inf)
        ub[self.off_C:self.off_C + M * H] = S
        ub[self.off_D:self.off_D + M * H] = S
        lb[self.off_E:self.off_E + M * H] = E_MIN
        ub[self.off_E:self.off_E + M * H] = E_MAX
        self.bounds = list(zip(lb, ub))

    @classmethod
    def get(cls, H, n_today, M, mode):
        key = (H, n_today, M, mode)
        if key not in cls._cache:
            cls._cache[key] = cls(H, n_today, M, mode)
        return cls._cache[key]

    def solve(self, scen_N, scen_c, w, g_today, E0, nu, fix_a=None, cap_C=None):
        H, M, nt, nz = self.H, self.M, self.n_today, self.n_tail
        scen_N = np.asarray(scen_N, float)
        scen_c = np.asarray(scen_c, float)
        w = np.asarray(w, float)
        w = w / w.sum()
        cbar = w @ scen_c

        c = np.zeros(self.nvar)
        c[self.off_a:self.off_a + nt] = cbar[:nt]
        if self.rev:
            c[self.off_up:self.off_up + nt] = 0.5 * cbar[:nt]
            c[self.off_dn:self.off_dn + nt] = 0.5 * cbar[:nt]
        for wi in range(M):
            for u in range(nz):
                c[self.off_z + wi * nz + u] = w[wi] * scen_c[wi, nt + u]
            base = wi * H
            c[self.off_b + base:self.off_b + base + H] = w[wi] * 5.0 * scen_c[wi, :]
            c[self.off_E + base + H - 1] -= nu * w[wi]
        for wi in range(M):
            base = wi * H
            c[self.off_C + base:self.off_C + base + H] += EPS_THROUGHPUT
            c[self.off_D + base:self.off_D + base + H] += EPS_THROUGHPUT
            c[self.off_b + base:self.off_b + base + H] += EPS_THROUGHPUT

        b_ub = -scen_N.reshape(-1).copy()
        b_eq = np.zeros(M * H)
        for wi in range(M):
            b_eq[wi * H] = E0
        if self.n_eq_adj:
            gt = np.asarray(g_today, float).reshape(-1)
            if gt.size != nt:
                raise ValueError(f"g_today 长度 {gt.size} ≠ n_today {nt}")
            b_eq = np.concatenate([b_eq, gt.copy()])

        bnds = self.bounds
        if fix_a is not None:
            fa = np.asarray(fix_a, float).reshape(-1)
            if fa.size != nt:
                raise ValueError(f"fix_a 长度 {fa.size} ≠ n_today {nt}")
            bnds = list(self.bounds)
            for u in range(nt):
                bnds[self.off_a + u] = (float(fa[u]), float(fa[u]))

        A_ub = self.A_ub
        if cap_C is not None:
            cap = np.asarray(cap_C, float)
            if cap.shape != (M, nt):
                raise ValueError(f"cap_C 形状 {cap.shape} ≠ ({M}, {nt})")
            if (cap < 0).any():
                raise ValueError("cap_C 不得为负")
            rrows, rcols, rvals = [], [], []
            for wi in range(M):
                for u in range(nt):
                    rrows.append(wi * nt + u)
                    rcols.append(self.off_C + wi * H + u)
                    rvals.append(1.0)
            A_cap = csr_matrix((rvals, (rrows, rcols)), shape=(M * nt, self.nvar))
            A_ub = vstack([self.A_ub, A_cap]).tocsr()
            b_ub = np.concatenate([b_ub, cap.reshape(-1)])

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=self.A_eq, b_eq=b_eq,
                      bounds=bnds, method="highs", options=_lp_options())
        out = {"status": int(res.status), "message": str(res.message),
               "fun": float(res.fun) if res.fun is not None else float("nan"),
               "a": None, "z": None, "b": None, "C": None, "D": None, "E": None,
               "cbar": cbar}
        if res.x is None:
            return out
        x = res.x
        out["b"] = x[self.off_b:self.off_b + M * H].reshape(M, H)
        out["C"] = x[self.off_C:self.off_C + M * H].reshape(M, H)
        out["D"] = x[self.off_D:self.off_D + M * H].reshape(M, H)
        out["E"] = x[self.off_E:self.off_E + M * H].reshape(M, H)
        if nz:
            out["z"] = x[self.off_z:self.off_z + M * nz].reshape(M, nz)
        out["a"] = x[self.off_a:self.off_a + nt]
        if self.rev:
            out["up"] = x[self.off_up:self.off_up + nt]
            out["dn"] = x[self.off_dn:self.off_dn + nt]
        return out


def solve_midnight_plan(scen_N, scen_c, w, E0, nu, delta=None, cap_C=None,
                        fix_a=None) -> dict:
    H = scen_N.shape[1]
    lp = NodePlanLP.get(H, H, scen_N.shape[0], "midnight")
    out = lp.solve(scen_N, scen_c, w, np.zeros(H), E0, nu, fix_a=fix_a, cap_C=cap_C)
    out["g"] = out["a"]
    return out


def solve_revised_plan(scen_N, scen_c, w, g_today, E0, nu, fix_a=None,
                       cap_C=None) -> dict:
    H = scen_N.shape[1]
    lp = NodePlanLP.get(H, g_today.size, scen_N.shape[0], "revise")
    return lp.solve(scen_N, scen_c, w, g_today, E0, nu, fix_a=fix_a, cap_C=cap_C)


def degeneracy_diag(res: dict, tol: float = 1e-6) -> dict:
    if res.get("b") is None:
        return {"ok": False, "reason": "无解"}
    b, Cc, D = res["b"], res["C"], res["D"]
    sim_cd = float(np.minimum(Cc, D).sum())
    sim_bc = float(np.minimum(b, Cc).sum())
    return {"ok": bool(sim_cd <= tol),
            "sim_charge_discharge": sim_cd,
            "sim_emergency_charge_inplan": sim_bc}


def nu_tau(ti: int, scen_c_today, scen_c_next=None) -> float:
    n = C.NU_PRICE_POINTS
    if ti == 0:
        arr = np.asarray(scen_c_today, float)
    else:
        arr = np.asarray(scen_c_next, float)
        if arr.size < n:
            raise ValueError("次日谷段预测不可用")
    return float(np.mean(arr[:n]) / ETA)
