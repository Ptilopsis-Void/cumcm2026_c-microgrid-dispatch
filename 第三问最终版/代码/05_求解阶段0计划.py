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


C = _load("_comm3.py", "q3_comm")

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

T = C.PERIODS_PER_DAY
S = C.S_PERIOD_KWH
ETA = C.ETA


class TwoStageSP:

    _SLOTS = {
        "lead":   ("b", "C", "D", "E"),
        "puv":    ("b", "C", "D", "E"),
        "uv":     ("b", "C", "D", "E"),
        "puv_rc": ("u", "v", "b", "C", "D", "E"),
    }

    def __init__(self, price, m, nu=C.NU_VALUE, mode="plan", p_seg=None,
                 recourse=True):
        self.price = np.asarray(price, float)
        self.L = int(self.price.size)
        self.m = int(m)
        self.nu = float(nu)
        self.mode = mode
        self.recourse = bool(recourse)
        self.p_seg = None if p_seg is None else np.asarray(p_seg, float)
        if mode == "plan":
            self.n_lead, self.kind = self.L, "lead"
        elif mode == "plan_sp":
            self.n_lead, self.kind = 3 * self.L, "puv"
        elif mode == "plan_sp_rc":
            self.n_lead, self.kind = self.L, "puv_rc"
        elif mode == "adjust":
            if self.p_seg is None or self.p_seg.size != self.L:
                raise ValueError("mode='adjust' 必须提供等长 p_seg")
            self.n_lead, self.kind = 2 * self.L, "uv"
        else:
            raise ValueError(mode)
        self.slots = self._SLOTS[self.kind]
        self.slot_off = {nm: i * self.L for i, nm in enumerate(self.slots)}
        self.per_w = len(self.slots) * self.L
        self.n_var = self.n_lead + self.m * self.per_w
        if self.kind in ("uv", "puv"):
            self.n_uub = self.L
        elif self.kind == "puv_rc":
            self.n_uub = self.m * self.L
        else:
            self.n_uub = 0
        self.n_ub = self.m * self.L + self.n_uub
        self.n_eq = self.m * self.L
        self._build()

    def off(self, w: int, name: str) -> int:
        return self.n_lead + w * self.per_w + self.slot_off[name]

    def _build(self):
        L, m = self.L, self.m
        p = self.p_seg
        pr = self.price

        ri, ci, vv = [], [], []
        for w in range(m):
            r = np.arange(w * L, w * L + L)
            t = np.arange(L)
            if self.kind == "lead":
                ri.append(r); ci.append(t); vv.append(-np.ones(L))
            elif self.kind == "puv":
                ri.append(r); ci.append(t); vv.append(-np.ones(L))
                ri.append(r); ci.append(L + t); vv.append(np.ones(L))
                ri.append(r); ci.append(2 * L + t); vv.append(-np.ones(L))
            elif self.kind == "puv_rc":
                ri.append(r); ci.append(t); vv.append(-np.ones(L))
                ri.append(r); ci.append(self.off(w, "u") + t); vv.append(np.ones(L))
                ri.append(r); ci.append(self.off(w, "v") + t); vv.append(-np.ones(L))
            else:
                ri.append(r); ci.append(t); vv.append(np.ones(L))
                ri.append(r); ci.append(L + t); vv.append(-np.ones(L))
            ri.append(r); ci.append(self.off(w, "b") + t); vv.append(-np.ones(L))
            ri.append(r); ci.append(self.off(w, "C") + t); vv.append(np.ones(L))
            ri.append(r); ci.append(self.off(w, "D") + t); vv.append(-np.ones(L))

        if self.kind == "uv":
            r0 = m * L
            r = np.arange(r0, r0 + L)
            t = np.arange(L)
            ri.append(r); ci.append(t); vv.append(np.ones(L))
        elif self.kind == "puv":
            r0 = m * L
            r = np.arange(r0, r0 + L)
            t = np.arange(L)
            ri.append(r); ci.append(t); vv.append(-np.ones(L))
            ri.append(r); ci.append(L + t); vv.append(np.ones(L))
        elif self.kind == "puv_rc":
            r0 = m * L
            t = np.arange(L)
            for w in range(m):
                r = np.arange(r0 + w * L, r0 + (w + 1) * L)
                ri.append(r); ci.append(t); vv.append(-np.ones(L))
                ri.append(r); ci.append(self.off(w, "u") + t); vv.append(np.ones(L))

        self.A_ub = sp.csr_matrix(
            (np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))),
            shape=(self.n_ub, self.n_var))

        ri, ci, vv = [], [], []
        for w in range(m):
            r = np.arange(w * L, w * L + L)
            t = np.arange(L)
            Eo, Co, Do = self.off(w, "E"), self.off(w, "C"), self.off(w, "D")
            ri += [r, r, r]
            ci += [Eo + t, Co + t, Do + t]
            vv += [np.ones(L), -ETA * np.ones(L), (1.0 / ETA) * np.ones(L)]
            if L > 1:
                ri.append(r[1:])
                ci.append(Eo + t[:-1])
                vv.append(-np.ones(L - 1))
        self.A_eq = sp.csr_matrix(
            (np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))),
            shape=(self.n_eq, self.n_var))

        c = np.zeros(self.n_var)
        if self.kind == "lead":
            c[:L] = pr
        elif self.kind == "puv":
            c[:L] = pr
            c[L:2 * L] = -0.5 * pr
            c[2 * L:3 * L] = 1.5 * pr
        elif self.kind == "puv_rc":
            c[:L] = pr
        else:
            c[:L] = -0.5 * pr
            c[L:2 * L] = 1.5 * pr
        for w in range(m):
            if self.kind == "puv_rc":
                c[self.off(w, "u"):self.off(w, "u") + L] = -0.5 * pr / m
                c[self.off(w, "v"):self.off(w, "v") + L] = 1.5 * pr / m
            c[self.off(w, "b"):self.off(w, "b") + L] = 5.0 * pr / m
            c[self.off(w, "E") + L - 1] = -self.nu / m
        self.c = c

        ub_max = None if self.recourse else 0.0
        if self.kind == "lead":
            bnds = [(0.0, None)] * L
        elif self.kind == "puv":
            bnds = [(0.0, None)] * L + [(0.0, ub_max)] * L + [(0.0, ub_max)] * L
        elif self.kind == "puv_rc":
            bnds = [(0.0, None)] * L
        else:
            bnds = [(0.0, None)] * L + [(0.0, None)] * L
        for _ in range(m):
            if self.kind == "puv_rc":
                bnds += [(0.0, ub_max)] * L + [(0.0, ub_max)] * L
            bnds += [(0.0, None)] * L
            bnds += [(0.0, S)] * L
            bnds += [(0.0, S)] * L
            bnds += [(C.E_MIN, C.E_MAX)] * L
        self.bounds = bnds

    def solve(self, scen_N, E_init):
        L, m = self.L, self.m
        scen_N = np.asarray(scen_N, float).reshape(m, L)

        b_ub = np.zeros(self.n_ub)
        for w in range(m):
            rhs = (self.p_seg - scen_N[w]) if self.kind == "uv" else (-scen_N[w])
            b_ub[w * L: w * L + L] = rhs
        if self.kind == "uv":
            b_ub[m * L: m * L + L] = self.p_seg
        elif self.kind in ("puv", "puv_rc"):
            b_ub[m * L:] = 0.0

        b_eq = np.zeros(self.n_eq)
        for w in range(m):
            b_eq[w * L] = E_init

        res = linprog(self.c, A_ub=self.A_ub, b_ub=b_ub, A_eq=self.A_eq,
                      b_eq=b_eq, bounds=self.bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"两阶段 SP 求解失败：{res.message}")
        x = res.x
        out = {
            "status": res.status,
            "objective": float(res.fun),
            "b": np.stack([x[self.off(w, "b"):self.off(w, "b") + L] for w in range(m)]),
            "C": np.stack([x[self.off(w, "C"):self.off(w, "C") + L] for w in range(m)]),
            "D": np.stack([x[self.off(w, "D"):self.off(w, "D") + L] for w in range(m)]),
            "E": np.stack([x[self.off(w, "E"):self.off(w, "E") + L] for w in range(m)]),
        }
        if self.kind == "lead":
            out["p"] = x[:L]
            out["q"] = x[:L].copy()
            out["u"] = out["u_bar"] = np.zeros(L)
            out["v"] = out["v_bar"] = np.zeros(L)
        elif self.kind == "puv":
            p, u, v = x[:L], x[L:2 * L], x[2 * L:3 * L]
            out["p"], out["u"], out["v"] = p, u, v
            out["u_bar"], out["v_bar"] = u, v
            out["q"] = p - u + v
        elif self.kind == "puv_rc":
            p = x[:L]
            U = np.stack([x[self.off(w, "u"):self.off(w, "u") + L]
                          for w in range(m)])
            V = np.stack([x[self.off(w, "v"):self.off(w, "v") + L]
                          for w in range(m)])
            out["p"] = p
            out["u_s"], out["v_s"] = U, V
            out["q_s"] = p - U + V
            out["u_bar"], out["v_bar"] = U.mean(0), V.mean(0)
            out["u"], out["v"] = out["u_bar"], out["v_bar"]
            out["q"] = out["q_s"].mean(0)
        else:
            u, v = x[:L], x[L:2 * L]
            out["u"], out["v"] = u, v
            out["u_bar"], out["v_bar"] = u, v
            out["q"] = self.p_seg - u + v
            out["p"] = self.p_seg.copy()

        pr = self.price
        pterm = 0.0 if self.kind == "uv" else float(pr @ out["p"])
        first = (pterm - 0.5 * float(pr @ out["u_bar"])
                 + 1.5 * float(pr @ out["v_bar"]))
        rec = 5.0 * (out["b"] @ pr).sum() / m \
            - self.nu * out["E"][:, -1].sum() / m
        gap = abs(first + rec - float(res.fun))
        if gap > 1e-6 * max(1.0, abs(float(res.fun))):
            raise AssertionError(
                "LP 目标与费用口径不一致：LHS=%.6f，LP objective=%.6f，差=%.3e。"
                "请检查 u / v 的系数是否为 −0.5c / +1.5c（README §5.5、§5.8）。"
                % (first + rec, float(res.fun), gap))
        out["first_stage_fee"] = float(first)
        out["recourse_fee"] = float(rec)
        return out


def scen_net_load(scen_L, scen_V):
    return scen_L - scen_V


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t00 = time.perf_counter()

    log("=" * 78)
    log("第三问 05 —— 阶段 0 两阶段随机规划（日前计划购电量 p）")
    log("=" * 78)

    F = np.load(C.V_FORECAST_NPZ, allow_pickle=False)
    Zd = np.load(C.V_SCENARIO_NPZ, allow_pickle=False)
    Z = C.Q2.matrix()
    price = np.asarray(Z["price"], float)
    scen_L = np.asarray(Zd["scen_L"], float) if "scen_L" in Zd else None
    if scen_L is None:
        S2 = C.Q2.scenarios()
        scen_L = np.asarray(S2["scen_L"], float)
    scen_V_h = np.asarray(Zd["scen_V_hourly"], float)
    scen_V_abs = C.lead_to_absolute(scen_V_h)
    score_idx = np.asarray(F["score_day_index"], int)
    dates = [str(x) for x in F["dates"]]
    n_day = len(dates)
    M = C.M_SCENARIOS

    nu = C.NU_VALUE
    log(f"ν = {nu:.10f} 元/kWh（复算 {C.compute_nu():.10f}，"
        f"{'✔ 一致' if abs(nu - C.compute_nu()) < 1e-9 else '✘ 不一致'}）")
    log(f"情景数 M = {M}，段长 T = {T}，评分期天数 = {len(score_idx)}")

    lp = TwoStageSP(price, M, nu, mode="plan")
    log(f"LP 规模：变量 {lp.n_var}，不等式 {lp.n_ub}，等式 {lp.n_eq}")

    def scen_N_of(d: int) -> np.ndarray:
        return scen_L[d] - C.scenario_pv_energy(scen_V_abs, d, 0)

    d0 = int(score_idx[0])
    t0 = time.perf_counter()
    r0 = lp.solve(scen_N_of(d0), C.E_INIT)
    dt0 = time.perf_counter() - t0
    log("")
    log(f"单日试算（{dates[d0]}）：{dt0:.2f} s，目标 {r0['objective']:.4f}")
    log(f"  计划购电量合计 = {r0['p'].sum():.4f} kWh")
    log(f"  计划购电费 = {float(price @ r0['p']):.4f} 元")
    log(f"  紧急购电量（情景均）= {r0['b'].mean(0).sum():.4f} kWh")
    log(f"  期末储电量（情景均）= {r0['E'][:, -1].mean():.4f} kWh")
    log(f"  → 预计阶段 0 全年耗时 ≈ {dt0 * len(score_idx) / 60:.1f} 分钟")

    log("")
    log("前向生成评分期 334 天的计划轨迹（跨日连续：E0 = 前一日情景末储电量均值）…")
    P = np.zeros((n_day, T))
    Cbar = np.zeros((n_day, T))
    Dbar = np.zeros((n_day, T))
    Ebar = np.zeros((n_day, T))
    Bbar = np.zeros((n_day, T))
    E_end = np.zeros(n_day)
    obj = np.zeros(n_day)
    t0 = time.perf_counter()
    E_warm = C.E_INIT
    for k, d in enumerate(score_idx):
        E0 = E_end[d - 1] if (d > 0 and E_end[d - 1] > 0) else E_warm
        r = lp.solve(scen_N_of(d), E0)
        P[d] = r["p"]
        Bbar[d] = r["b"].mean(0)
        Cbar[d] = r["C"].mean(0)
        Dbar[d] = r["D"].mean(0)
        Ebar[d] = r["E"].mean(0)
        E_end[d] = float(r["E"][:, -1].mean())
        obj[d] = r["objective"]
        if (k + 1) % 40 == 0 or k == len(score_idx) - 1:
            el = time.perf_counter() - t0
            log(f"  {k + 1}/{len(score_idx)} 天  用时 {el:.1f} s  "
                f"预计剩余 {el / (k + 1) * (len(score_idx) - k - 1) / 60:.1f} 分钟  "
                f"日末库存均值 {E_end[d]:.1f} kWh")
    el = time.perf_counter() - t0
    log(f"阶段 0 完成，总用时 {el:.1f} s")

    sc = score_idx
    plan_energy = float(P[sc].sum())
    plan_fee_traj = (P[sc] * price).sum()
    log("")
    log(f"  评分期计划购电量合计 = {plan_energy:.4f} kWh")
    log(f"  评分期计划购电费合计 = {plan_fee_traj:.4f} 元")
    log(f"  评分期计划均价 = {plan_fee_traj / plan_energy:.6f} 元/kWh")
    log(f"  日末库存均值范围 = [{E_end[sc].min():.1f}, {E_end[sc].max():.1f}] kWh，"
        f"年末 {E_end[sc[-1]]:.4f} kWh")
    log(f"  计划视角紧急购电量（情景均）= {float(Bbar[sc].sum()):.2f} kWh")

    log("")
    log("── 口径自检：调整费对称性 ⇒ 最优计划量应贴近需求分布的中位数 ──")
    N0 = scen_net_load(scen_L, C.lead_to_absolute(scen_V_h)[:, 0]
                       .repeat(6, axis=-1)[..., :C.PERIODS_PER_DAY]
                       * C.DELTA_HOURS)
    npos = [int(d) for d in sc]
    med = np.median(N0[npos], axis=1)
    q80 = np.percentile(N0[npos], 80, axis=1)
    log(f"  情景净负荷中位数（逐时段平均）= {med.mean():.2f} kWh/时段")
    log(f"  情景净负荷 80 分位（逐时段平均）= {q80.mean():.2f} kWh/时段")
    log(f"  本模型给出的计划量（逐时段平均）= {P[sc].mean():.2f} kWh/时段")
    log(f"  中位数正部合计 = {float(np.maximum(med, 0).sum()):.0f} kWh/天，"
        f"80 分位正部合计 = {float(np.maximum(q80, 0).sum()):.0f} kWh/天，"
        f"计划量合计 = {float(P[sc].sum() / len(sc)):.0f} kWh/天")
    log("  → 阶段 0 无调整机制，计划量只需覆盖“情景均值”意义上的缺口；"
        "真正的“中位数 vs 80 分位”分水岭出现在 06 的调整费里。")

    N_act = np.asarray(Z["net_load_energy_kwh"], float)[sc]
    defc = np.maximum(N_act, 0.0)
    base_e = float(defc.sum())
    base_cost = float((defc @ price).sum())
    log("")
    log("── 基准对照（无储能、按时段净缺额直接购电）──")
    log(f"  净缺额合计 = {base_e:.4f} kWh，购电费 = {base_cost:.4f} 元，"
        f"均价 = {base_cost / base_e:.6f} 元/kWh")
    log(f"  计划 vs 无储能：购电量 {plan_energy - base_e:+.2f} kWh，"
        f"计划购电费 {plan_fee_traj - base_cost:+.2f} 元 "
        f"({(plan_fee_traj / base_cost - 1) * 100:+.3f}%)")
    log(f"  注：第三问的『计划购电量』阶段尚未计入调整费与紧急费，"
        f"费用对比在 08 全年回测后给出。")

    np.savez_compressed(
        C.STAGE0_NPZ,
        P=P, Cbar=Cbar, Dbar=Dbar, Ebar=Ebar, Bbar=Bbar,
        E_end=E_end, objective=obj, score_day_index=sc,
        price=price, nu=np.asarray([nu], float), n_scenarios=np.asarray([M], int),
        base_energy=np.asarray([base_e]), base_cost=np.asarray([base_cost]),
    )
    rows = []
    for d in sc:
        rows.append([dates[d], f"{P[d].sum():.6f}", f"{float(price @ P[d]):.6f}",
                     f"{E_end[d]:.6f}", f"{Bbar[d].sum():.6f}",
                     f"{Cbar[d].sum():.6f}", f"{Dbar[d].sum():.6f}"])
    day_csv = C.RESULT_DIR / "第三问_阶段0日汇总.csv"
    C.write_csv_utf8_sig(
        day_csv,
        ["日期", "计划购电量_kWh", "计划购电费_元", "日末储电量_kWh",
         "计划视角紧急量_kWh", "充电量_kWh", "放电量_kWh"], rows)
    log("")
    log(f"已保存：{C.STAGE0_NPZ.relative_to(C.PROJECT_DIR)}")
    log(f"已保存：{day_csv.relative_to(C.PROJECT_DIR)}")

    log("")
    log(f"总用时 {time.perf_counter() - t00:.1f} s")
    log("[05 完成] 阶段 0 两阶段随机规划结束。")
    log.dump(C.SOLVE_LOG_DIR / "第三问_05阶段0求解日志.txt", "[05 完成] 阶段 0 结束。")
    C.write_text_utf8(C.REPORT_STAGE0_MD, "\n".join([
        "# 第三问 阶段 0 计划报告（两阶段随机规划）", "",
        "> 脚本：`第三问最终版/代码/05_求解阶段0计划.py`；"
        "产出：`模型结果/第三问_阶段0计划.npz`", "",
        "## 0. 结论速览", "",
        f"* 评分期计划购电量合计 **{plan_energy:.2f} kWh**，"
        f"计划购电费 **{plan_fee_traj:.2f} 元**，均价 {plan_fee_traj / plan_energy:.4f} 元/kWh；",
        f"* 计划量均值 {P[sc].mean():.2f} kWh/时段，"
        f"情景净负荷逐时段中位数均值 {med.mean():.2f} kWh/时段、"
        f"80 分位均值 {q80.mean():.2f} kWh/时段；",
        "* 跨日库存连续（E0 = 前一日情景末储电量均值），预热期保持 6000 kWh；",
        "* 本步**只**产生计划轨迹 p，调整费与紧急费在 06/08 计入。", ""]
        + log.lines + [""]))
    log("已保存：报告/第三问_阶段0计划报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
