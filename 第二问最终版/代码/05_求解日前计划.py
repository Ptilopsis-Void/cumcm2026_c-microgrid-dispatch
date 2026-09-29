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
import scipy.sparse as sp
from scipy.optimize import linprog


def compute_nu() -> float:
    import pandas as pd
    df = pd.read_excel(C.ATTACHMENT1_PATH, sheet_name=0, header=0)
    price = df.iloc[:, 1].to_numpy(float)[:C.NU_PRICE_POINTS]
    return float(price.mean() / C.ETA)


NU = compute_nu()

M = C.M_SCENARIOS
T = C.PERIODS_PER_DAY
S = C.S_PERIOD_KWH


class DayPlanLP:

    def __init__(self, price: np.ndarray, nu: float = NU, m: int = M):
        self.price = np.asarray(price, float)
        self.nu = float(nu)
        self.m = int(m)
        self.T = T
        self.n_var = T + self.m * 4 * T
        self.n_con = self.m * 2 * T
        self._build()

    def off(self, w: int, name: str) -> int:
        base = T + w * 4 * T
        return base + {"b": 0, "C": T, "D": 2 * T, "E": 3 * T}[name]

    def _build(self):
        m, nT = self.m, self.T
        nv = self.n_var
        rows_i, cols_i, vals = [], [], []

        for w in range(m):
            r = np.arange(w * 2 * nT, w * 2 * nT + nT)
            t = np.arange(nT)
            rows_i += [r, r, r]
            cols_i += [np.full(nT, 0) + t,
                       np.full(nT, self.off(w, "b")) + t,
                       np.full(nT, self.off(w, "C")) + t]
            vals += [-np.ones(nT), -np.ones(nT), np.ones(nT)]
            rows_i.append(r)
            cols_i.append(np.full(nT, self.off(w, "D")) + t)
            vals.append(-np.ones(nT))

        self.A_ub = sp.csr_matrix((np.concatenate(vals),
                                   (np.concatenate(rows_i), np.concatenate(cols_i))),
                                  shape=(self.n_con, nv))

        rows_i, cols_i, vals = [], [], []
        for w in range(m):
            r = np.arange(w * 2 * nT + nT, w * 2 * nT + 2 * nT)
            t = np.arange(nT)
            Eo, Co, Do = self.off(w, "E"), self.off(w, "C"), self.off(w, "D")
            rows_i += [r, r, r]
            cols_i += [np.full(nT, Eo) + t, np.full(nT, Co) + t, np.full(nT, Do) + t]
            vals += [np.ones(nT), -C.ETA_C * np.ones(nT), (1.0 / C.ETA_D) * np.ones(nT)]
            rows_i.append(r[1:])
            cols_i.append(np.full(nT - 1, Eo) + t[:-1])
            vals.append(-np.ones(nT - 1))
        self.A_eq = sp.csr_matrix((np.concatenate(vals),
                                   (np.concatenate(rows_i), np.concatenate(cols_i))),
                                  shape=(self.n_con, nv))

        cvec = np.zeros(nv)
        cvec[:nT] = self.price
        for w in range(m):
            cvec[self.off(w, "b"):self.off(w, "b") + nT] = 5.0 * self.price / m
            cvec[self.off(w, "E") + nT - 1] = -self.nu / m
        self.c = cvec

        bnds = [(0.0, None)] * nT
        for _ in range(m):
            bnds += [(0.0, None)] * nT
            bnds += [(0.0, S)] * nT
            bnds += [(0.0, S)] * nT
            bnds += [(C.E_MIN, C.E_MAX)] * nT
        self.bounds = bnds

    def solve(self, scen_N: np.ndarray, E_init: float):
        m, nT = self.m, self.T
        scen_N = np.asarray(scen_N, float).reshape(m, nT)
        b_ub = np.zeros(self.n_con)
        for w in range(m):
            b_ub[w * 2 * nT: w * 2 * nT + nT] = -scen_N[w]
        b_eq = np.zeros(self.n_con)
        for w in range(m):
            b_eq[w * 2 * nT + nT] = E_init
        res = linprog(self.c, A_ub=self.A_ub, b_ub=b_ub,
                      A_eq=self.A_eq, b_eq=b_eq, bounds=self.bounds,
                      method="highs")
        if not res.success:
            raise RuntimeError(f"日前计划 LP 求解失败：{res.message}")
        x = res.x
        g = x[:nT]
        b = np.stack([x[self.off(w, "b"):self.off(w, "b") + nT] for w in range(m)])
        Cc = np.stack([x[self.off(w, "C"):self.off(w, "C") + nT] for w in range(m)])
        D = np.stack([x[self.off(w, "D"):self.off(w, "D") + nT] for w in range(m)])
        E = np.stack([x[self.off(w, "E"):self.off(w, "E") + nT] for w in range(m)])
        return {"g": g, "b": b, "C": Cc, "D": D, "E": E, "status": res.status,
                "objective": float(res.fun)}


def make_scen_N(scen_L, scen_V):
    return scen_L - scen_V


def main() -> int:
    C.ensure_dirs()
    import time
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 74)
    p("第二问 05 —— 日前情景规划（单日试算 + ν 验证 + 无储能基准）")
    p("=" * 74)
    p(f"ν（期末库存续存价值系数）= {NU:.10f} 元/kWh")
    p(f"  = 附件一前 {C.NU_PRICE_POINTS} 个时段电价均值 ÷ η，"
      f"与图片参考值 0.481548148148 一致："
      f"{'✔ 是' if abs(NU - 0.4815481481481481) < 1e-9 else '✘ 否'}")

    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    price = Z["price"]
    scen_L, scen_V = Zd["scen_L"], Zd["scen_V"]
    scen_N = scen_L - scen_V
    date_strs = [str(s) for s in Z["dates"]]
    score_idx = Z["score_day_index"]

    lp = DayPlanLP(price)
    p(f"LP 规模：变量 {lp.n_var}，等式约束 {lp.n_con}，不等式约束 {lp.n_con}")

    t0 = time.perf_counter()
    r0 = lp.solve(make_scen_N(scen_L[score_idx[0]], scen_V[score_idx[0]]), C.E_INIT)
    dt0 = time.perf_counter() - t0
    p(f"单日试算（{date_strs[score_idx[0]]}）：用时 {dt0:.3f} s，"
      f"目标值 {r0['objective']:.4f}")
    p(f"  计划购电量合计 = {r0['g'].sum():.4f} kWh")
    p(f"  计划成本 = {price @ r0['g']:.4f} 元")
    bal_slack = scen_N[score_idx[0]] - (r0['g'][None, :] + r0['b'] + r0['D'] - r0['C'])
    p(f"  最大供需平衡约束残差 = {np.maximum(bal_slack, 0.0).max():.6e} kWh")
    E_rec = np.empty_like(r0['E'])
    E_rec[:, 0] = C.E_INIT + C.ETA_C * r0['C'][:, 0] - r0['D'][:, 0] / C.ETA_D
    E_rec[:, 1:] = r0['E'][:, :-1] + C.ETA_C * r0['C'][:, 1:] - r0['D'][:, 1:] / C.ETA_D
    p(f"  最大储能递推残差 = {np.abs(E_rec - r0['E']).max():.6e} kWh")

    N_act = Z["net_load_energy_kwh"][score_idx]
    defc = np.maximum(N_act, 0.0)
    base_e = float(defc.sum())
    base_cost = float((defc @ price).sum())
    p("")
    p("── 基准对照 ──")
    p(f"  评分期实际净负荷合计      = {float(N_act.sum()):.4f} kWh")
    p(f"  评分期净缺额合计（无储能）= {base_e:.4f} kWh，购电费 {base_cost:.4f} 元，"
      f"均价 {base_cost / base_e:.6f} 元/kWh")
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_无储能基准.csv",
        ("项目", "数值", "单位"),
        [("评分期实际净负荷合计", f"{float(N_act.sum()):.6f}", "kWh"),
         ("评分期净缺额合计", f"{base_e:.6f}", "kWh"),
         ("无储能购电费", f"{base_cost:.6f}", "元"),
         ("无储能购电均价", f"{base_cost / base_e:.6f}", "元/kWh")])

    p("")
    p("说明：全年跨日闭环计划（每天用真实日初库存）与三执行器回测统一在 07 中完成。")
    C.write_text_utf8(C.SOLVE_LOG_TXT, "\n".join(log + ["", "[05 完成] 日前情景规划试算校验结束。"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
