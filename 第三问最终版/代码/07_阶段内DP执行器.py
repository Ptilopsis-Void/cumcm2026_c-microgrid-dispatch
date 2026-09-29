#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import sys
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


C = _load("_comm3.py", "q3_comm")

T = C.PERIODS_PER_DAY
M = C.M_SCENARIOS
S = C.S_PERIOD_KWH
ETA = C.ETA
TAUS = list(C.TAU_HOURS)
K0S = list(C.TAU_PERIOD_INDEX)


def stage_bounds() -> list[tuple[int, int]]:
    b = []
    for i, k0 in enumerate(K0S):
        k1 = K0S[i + 1] if i + 1 < len(K0S) else T
        b.append((k0, k1))
    return b


STAGES = stage_bounds()


def make_grid(delta: float) -> np.ndarray:
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
    rows = np.arange(vals.shape[0])[:, None]
    return vals[rows, i0] * (1.0 - fr) + vals[rows, i0 + 1] * fr


def max_argmin(a: np.ndarray) -> int:
    return a.size - 1 - int(np.argmin(a[::-1]))


def build_value_functions(r_seg: np.ndarray, price_seg: np.ndarray,
                          nu: float = None, delta: float = 6.0) -> dict:
    if nu is None:
        nu = C.NU_VALUE
    r = np.asarray(r_seg, float)
    px = np.asarray(price_seg, float)
    m_, t_ = r.shape
    grid = make_grid(delta)
    n = grid.size

    Hs = np.tile(-nu * grid, (m_, 1))
    Hbar = np.empty((t_ + 1, n))
    Hbar[t_] = -nu * grid
    R = np.zeros(t_)
    idx_all = np.arange(m_)[:, None]
    jall = np.arange(n)[None, :]

    for t in range(t_ - 1, -1, -1):
        cc = 5.0 * px[t]
        rt = r[:, t]

        tm = np.minimum(S, np.maximum(rt, 0.0)) / ETA
        F = Hs + cc * ETA * grid[None, :]
        jstar = (n - 1) - np.argmin(F[:, ::-1], axis=1)
        mm = np.floor(tm / delta + 1e-12).astype(np.int64)
        lo = np.maximum(0, jall - mm[:, None])
        jsel = np.minimum(np.maximum(jstar[:, None], lo), jall)
        new_def = F[idx_all, jsel] - cc * ETA * grid[None, :]

        s = ETA * np.minimum(S, np.maximum(-rt, 0.0))
        enew = np.minimum(grid[None, :] + s[:, None], C.E_MAX)
        new_chg = _interp_rows(grid, Hs, enew)

        Hs = np.where((rt <= 0.0)[:, None], new_chg, new_def)
        Hbar[t] = Hs.mean(axis=0)

        Gt = cc * ETA * grid + Hbar[t + 1]
        R[t] = grid[max_argmin(Gt)]

    return {"Hbar": Hbar, "R": R, "grid": grid}


def dp_execute(R: np.ndarray, N_act: np.ndarray, q_day: np.ndarray,
               E0: float) -> dict:
    t_ = N_act.size
    Cch = np.zeros(t_)
    D = np.zeros(t_)
    b = np.zeros(t_)
    U = np.zeros(t_)
    E = np.zeros(t_)
    e = float(E0)
    for t in range(t_):
        rt = float(N_act[t] - q_day[t])
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


def analytic_execute(N_act: np.ndarray, q_day: np.ndarray, E0: float) -> dict:
    t_ = N_act.size
    Cch = np.zeros(t_)
    D = np.zeros(t_)
    b = np.zeros(t_)
    U = np.zeros(t_)
    E = np.zeros(t_)
    e = float(E0)
    for t in range(t_):
        rt = float(N_act[t] - q_day[t])
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


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    log("=" * 78)
    log("第三问 07 —— 阶段内 DP 未来价值执行器（**函数库**）")
    log("=" * 78)
    log("")
    log("本文件现在只作为 DP 执行内核（函数库）提供：")
    log("  build_value_functions(r_seg, price_seg, nu, delta) → {Hbar, R, grid}")
    log("  dp_execute(R, N_act, q_day, E0)                    → {C, D, b, U, E}")
    log("  analytic_execute(N_act, q_day, E0)                 → {C, D, b, U, E}")
    log("")
    log("正式执行路径（P0-7 选择 A：逐发布节点分层，严禁跨节点复用 R_t）：")
    log("  `_policy3.run_policy` → 对 τ ∈ {0,6,12,18} 各自")
    log("    ① 用**该节点自己的**预报集在 [k0,144) 上 build_value_functions()")
    log("    ② 只取 [0, k1−k0) 段执行 dp_execute()")
    log("    ③ 用真实负荷/光伏因果推进 → 该节点末端的**真实** SOC")
    log("  再由 `08_全年回测与结算.py` 汇总结算并写报告。")
    log("")
    log("⚠ 旧的独立驱动（把 4 段净缺口拼接成 144 时段后仅做一次反向递推）")
    log("  已**删除**：它等价于 0:00 即已知 6:00/12:00/18:00 的预报，")
    log("  构成跨阶段信息泄露（整改建议 P0-7）。其产物 `第三问_执行器结果.npz`")
    log("  不再生成，内容等价地并入 `第三问_全年回测.npz`。")
    log("")
    log("自检入口：`00_接口契约自检.py`（K17–K19、D9–D10）与 `08` 的闭环残差表。")
    log.dump(C.LOG_DIR / "第三问_07执行器日志.txt", tail="")
    log("")
    log("[07] 函数库就绪（独立驱动已退役，P0-7 选择 A 生效）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
