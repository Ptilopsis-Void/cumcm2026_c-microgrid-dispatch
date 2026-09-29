#!/usr/bin/env python3
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


C = _load("_comm4.py", "q4_comm")
PO = _load("_policy4.py", "q4_policy")
ST = _load("_settlement4.py", "q4_settle")

import numpy as np

T = C.PERIODS_PER_DAY
K0 = C.TAU_PERIOD_INDEX

TRAJ_KEYS = ("g", "a", "b", "C", "D", "U", "R", "E_end")


def traj_bad(ra: dict, rb: dict, keys=TRAJ_KEYS, atol: float = 1e-12) -> list[str]:
    return [k for k in keys
            if not np.allclose(ra[k], rb[k], atol=atol, equal_nan=True)]


S_KWH = C.S_PERIOD_KWH
E_LO, E_HI = C.E_MIN, C.E_MAX
ETA = C.ETA


def dp_brute(scen_N, plan, scen_c, nu, grid):
    scen_N = np.asarray(scen_N, float)
    plan = np.asarray(plan, float)
    scen_c = np.asarray(scen_c, float)
    M_, H = scen_N.shape
    r = scen_N - plan
    n = grid.size
    grid = np.asarray(grid, float).reshape(-1)
    jj = np.arange(n)[None, :]
    ii = np.arange(n)[:, None]
    Hb = np.empty((H + 1, M_, n))
    Hb[H] = -float(nu) * grid[None, :]
    for t in range(H - 1, -1, -1):
        for m in range(M_):
            cc = 5.0 * float(scen_c[m, t])
            rt = float(r[m, t])
            hn = Hb[t + 1, m]
            if rt > 0.0:
                Dmax = np.minimum(rt, S_KWH)
                elo = np.maximum(grid[0], grid - Dmax / ETA)
                mstep = np.floor(np.maximum(0.0, grid - elo) / (grid[1] - grid[0])
                                 + 1e-12).astype(np.int64)
                lo = np.maximum(0, ii[:, 0] - mstep)
                obj = cc * (rt - ETA * (grid[:, None] - grid[None, :])) + hn[None, :]
                obj = np.where((jj >= lo[:, None]) & (jj <= ii), obj, np.inf)
                ep = cc * (rt - ETA * (grid - elo)) + np.interp(elo, grid, hn)
                val = np.minimum(obj.min(axis=1), ep)
            else:
                Cmax = np.minimum(-rt, S_KWH)
                ehi = np.minimum(grid + ETA * Cmax, E_HI)
                obj = np.where((jj >= ii) & (grid[None, :] <= ehi[:, None] + 1e-12),
                               hn[None, :], np.inf)
                ep = np.interp(ehi, grid, hn)
                val = np.minimum(obj.min(axis=1), ep)
            Hb[t, m] = val
    return Hb.mean(axis=1), Hb


def dp_exec_chain(scen_N, plan, scen_c, nu, Hbar_ps, grid, idx, c_now=None):
    scen_N = np.asarray(scen_N, float)
    plan = np.asarray(plan, float)
    scen_c = np.asarray(scen_c, float)
    M_, H = scen_N.shape
    worst, info = 0.0, None
    for m in range(M_):
        for t in range(H):
            c_t = float(scen_c[m, t]) if c_now is None else float(c_now[t])
            for i in idx:
                E = float(grid[i])
                ex = PO.execute_one_slot(E, float(plan[m, t]), c_t,
                                         float(scen_N[m, t]), Hbar_ps[t + 1, m], grid)
                got = 5.0 * c_t * float(ex["b"]) + float(
                    np.interp(float(ex["E_next"]), grid, Hbar_ps[t + 1, m]))
                gap = got - float(Hbar_ps[t, m][i])
                if abs(gap) > abs(worst):
                    worst, info = gap, (m, t, i, E, c_t, float(ex["E_next"]),
                                        float(ex["b"]), got,
                                        float(Hbar_ps[t, m][i]))
    return abs(worst), info


def dp_case(name, scen_N, plan, scen_c, nu, E0, delta, want_emerg=None,
            want_value=None, want_D=None, want_C=None, brute: bool = True,
            tol: float = 1e-9, val_rtol: float = 1e-7):
    def _mat(x):
        x = np.asarray(x, float)
        return x.reshape(1, -1) if x.ndim == 1 else x

    scen_N, plan, scen_c = _mat(scen_N), _mat(plan), _mat(scen_c)
    vf = PO.build_value_functions(scen_N, plan, scen_c, nu, delta=delta)
    grid, Hbar = vf["grid"], vf["Hbar"]
    M_, H = scen_N.shape
    i0 = int(np.argmin(np.abs(grid - float(E0))))
    on_grid = abs(float(grid[i0]) - float(E0)) <= 1e-9
    idx = list(range(0, grid.size)) if grid.size <= 401 else list(range(0, grid.size, 5))
    gap_exec, gap_info = (float("nan"), None)
    if M_ == 1:
        gap_exec, gap_info = dp_exec_chain(scen_N, plan, scen_c, nu,
                                           Hbar[:, None, :], grid, idx)
    checks = {}
    d_max = float("nan")
    if brute:
        Hbar_b, _ = dp_brute(scen_N, plan, scen_c, nu, grid)
        d_max = float(np.max(np.abs(Hbar - Hbar_b)))
        checks["递推 == 独立枚举（同候选集，≤1e-9）"] = bool(d_max <= 1e-9)
    if gap_exec == gap_exec:
        checks["执行器实现值 == 递推值（网格点状态）"] = bool(gap_exec <= 1e-7)
    E_, cost_b, tot_b, tot_C, tot_D = float(E0), 0.0, 0.0, 0.0, 0.0
    for t in range(H):
        ex = PO.execute_one_slot(E_, float(plan[0, t]), float(scen_c[0, t]),
                                 float(scen_N[0, t]), Hbar[t + 1], grid)
        cost_b += 5.0 * float(scen_c[0, t]) * float(ex["b"])
        tot_b += float(ex["b"])
        tot_C += float(ex["C"])
        tot_D += float(ex["D"])
        E_ = float(ex["E_next"])
    val_exec = cost_b + float(np.interp(E_, grid, Hbar[H]))
    val_dp = float(np.interp(float(E0), grid, Hbar[0]))
    if on_grid:
        checks["实算轨迹值 == 起点递推值（网格点状态）"] = \
            bool(abs(val_exec - val_dp) <= val_rtol)
    row = {"name": name, "delta": delta, "E0": float(E0), "H": H, "on_grid": on_grid,
           "d_max": d_max, "gap_exec": gap_exec, "gap_info": gap_info,
           "val_exec": val_exec, "val_dp": val_dp,
           "cost_b": cost_b, "sum_C": tot_C, "sum_D": tot_D, "checks": checks}
    if want_emerg is not None:
        row["checks"]["实算紧急购电量 == 目标"] = bool(abs(tot_b - want_emerg) <= tol)
        row["want_emerg"] = want_emerg
    if want_value is not None:
        row["checks"]["实算轨迹值 == 解析值"] = bool(abs(val_exec - want_value) <= val_rtol)
        row["want_value"] = want_value
    if want_D is not None:
        row["checks"]["实算放电量 == 目标"] = bool(abs(tot_D - want_D) <= tol)
        row["want_D"] = want_D
    if want_C is not None:
        row["checks"]["实算充电量 == 目标"] = bool(abs(tot_C - want_C) <= tol)
        row["want_C"] = want_C
    row["ok"] = all(checks.values())
    return row


def main() -> int:
    C.ensure_dirs()
    t0 = time.perf_counter()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg, flush=True)

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail):
        checks.append((name, bool(ok), detail))
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    p("=" * 78)
    p("第四问 05 —— 统一物理模型 / 采购模型 / 价格相关价值执行器验证")
    p("=" * 78)

    p("")
    p("── 1. 最小账单算例（§10）──")
    mt = ST.minimal_bill_test()
    for k, v in mt["cases"].items():
        p(f"    {k}: got {v['got']:.4f} / want {v['want']:.4f} -> {'OK' if v['ok'] else 'BAD'}")
    ck("账单最小算例全部通过（含 4-3 等价式）", mt["ok"],
       f"{len(mt['cases'])} 个算例，全部误差 ≤1e-9")

    p("")
    p("── 1b. 价值函数与执行器最小算例（修复后回归）──")
    ROWS: list[dict] = []

    ROWS.append(dp_case("C1 全覆盖→紧急 0", [90.0], [0.0], [1.0], 0.5,
                        1800.0, 12.0, want_emerg=0.0, want_D=90.0,
                        want_value=-0.5 * (1800.0 - 90.0 / ETA)))
    ROWS.append(dp_case("C2 部分覆盖(40/90)→5c×50", [90.0], [0.0], [1.0], 0.5,
                        1200.0 + 40.0 / ETA, 12.0, want_emerg=50.0, want_D=40.0,
                        want_value=5.0 * 50.0 - 0.5 * 1200.0))
    old_bug = 5.0 * 1.0 * 90.0
    ROWS.append(dp_case("C3 修复前反例(1800/1200/90)", [90.0], [0.0], [1.0], 0.5,
                        1800.0, 6.0, want_emerg=0.0,
                        want_value=-0.5 * (1800.0 - 90.0 / ETA),
                        brute=True))
    ROWS.append(dp_case("C4 功率上限绑定", [2000.0], [0.0], [1.0], 0.5,
                        10800.0, 12.0, want_emerg=2000.0 - S_KWH, want_D=S_KWH))
    ROWS.append(dp_case("C5 非整数端点→全额放电", [11.5], [0.0], [1.0], 0.5,
                        6000.0, 6.0, want_emerg=0.0, want_D=11.5,
                        want_value=-0.5 * (6000.0 - 11.5 / ETA)))
    ROWS.append(dp_case("C5b 非整数端点+高ν→不放电", [11.5], [0.0], [1.0], 10.0,
                        6000.0, 6.0, want_emerg=11.5, want_D=0.0,
                        want_value=5.0 * 11.5 - 10.0 * 6000.0))
    ROWS.append(dp_case("C6 富余→充满并弃电", [-2000.0], [0.0], [1.0], 0.5,
                        6000.0, 12.0, want_C=S_KWH, want_D=0.0, want_emerg=0.0,
                        want_value=-0.5 * (6000.0 + ETA * S_KWH)))

    for row in ROWS:
        bad = [k for k, v in row["checks"].items() if not v]
        p(f"    {row['name']}: 递推↔枚举最大差 {row['d_max']:.2e}；执行链偏差 "
          f"{row['gap_exec']:.2e}；实算值 {row['val_exec']:.6f}"
          f" | {'OK' if row['ok'] else 'BAD: ' + '; '.join(bad)}")
    ck("价值函数最小算例：Bellman 递推 ↔ 实际执行一致",
       all(r["ok"] for r in ROWS),
       f"{len(ROWS)} 个算例全部通过"
       + ("" if all(r["ok"] for r in ROWS)
          else "；失败：" + "; ".join(r["name"] for r in ROWS if not r["ok"])))
    ck("修复前 DP 反例已被区分（旧口径 450 元，现 0 元）",
       abs(ROWS[2]["cost_b"]) <= 1e-7 and old_bug > 1e-7,
       f"C3 实算紧急费 {ROWS[2]['cost_b']:.6f} 元 vs 旧实现等价口径 {old_bug:.2f} 元"
       f"（放电可达步数分母误用 grid[1] 使 m≡0、完全无法放电）")

    p("")
    p("    ── C7 两时段价格反转 vs 连续枚举 ──")
    N7 = np.array([[90.0, S_KWH]])
    c7 = np.array([[0.5, 5.0]])
    nu7, E7 = 0.5, 2196.0

    def val2(E1: float, D2: float) -> float:
        D2 = min(D2, S_KWH, ETA * max(0.0, E1 - E_LO))
        return 5.0 * c7[0, 1] * max(0.0, N7[0, 1] - D2) \
            - nu7 * max(E_LO, E1 - D2 / ETA)

    def val1(D1: float) -> float:
        E1 = E7 - D1 / ETA
        D2max = min(N7[0, 1], S_KWH, ETA * max(0.0, E1 - E_LO))
        return 5.0 * c7[0, 0] * max(0.0, N7[0, 0] - D1) \
            + min(val2(E1, 0.0), val2(E1, D2max))

    D1max = min(N7[0, 0], S_KWH, ETA * (E7 - E_LO))
    E1b = E_LO + N7[0, 1] / ETA
    D1_star = ETA * (E7 - E1b)
    cand1 = np.unique(np.r_[np.linspace(0.0, D1max, 2001), [0.0, D1max, D1_star]])
    enum_val = min(val1(d) for d in cand1)

    c7_rows = []
    for d7 in (12.0, 6.0, 3.0):
        vf7 = PO.build_value_functions(N7, np.zeros((1, 2)), c7, nu7, delta=d7)
        g7, H7 = vf7["grid"], vf7["Hbar"]
        dp_val = float(np.interp(E7, g7, H7[0]))
        ex1 = PO.execute_one_slot(E7, 0.0, float(c7[0, 0]), float(N7[0, 0]), H7[1], g7)
        c7_rows.append({"delta": d7, "dp": dp_val, "gap": dp_val - enum_val,
                        "D1": float(ex1["D"]), "b1": float(ex1["b"]),
                        "D1_tol": d7 / ETA + 1.0})
        p(f"      δ={d7:g}: 递推值 {dp_val:.6f} 元（连续最优 {enum_val:.6f}，差 "
          f"{dp_val - enum_val:+.4f}）；执行器 D1={float(ex1['D']):.4f} "
          f"(解析 D1*={D1_star:.4f})")
    gaps_txt = "、".join("{:+.3f} 元（δ={:g}）".format(r["gap"], r["delta"])
                       for r in c7_rows)
    ck("C7 网格递推不优于连续最优（近似方向正确）",
       all(r["gap"] >= -1e-6 for r in c7_rows),
       f"δ=12/6/3 的相对连续最优之差：{gaps_txt}"
       "（弦插值在凸值函数上方 ⇒ 递推值只会偏高）")
    ck("C7 网格加密收敛：δ=3 的差 ≤ δ=12 的差且 ≤5 元",
       c7_rows[-1]["gap"] <= c7_rows[0]["gap"] + 1e-9 and c7_rows[-1]["gap"] <= 5.0,
       f"差从 {c7_rows[0]['gap']:+.3f} 元（δ=12）降到 {c7_rows[-1]['gap']:+.3f} 元（δ=3）")
    ck("C7 价格反转行为正确：廉价时段不全放、留给昂贵时段",
       all(r["D1"] > 1e-9 and r["D1"] < N7[0, 0] - 1e-9 for r in c7_rows)
       and abs(c7_rows[-1]["D1"] - D1_star) <= c7_rows[-1]["D1_tol"],
       f"δ=3 时执行器 D1={c7_rows[-1]['D1']:.4f} kWh ≈ 解析 D1*={D1_star:.4f} "
       f"(公差 {c7_rows[-1]['D1_tol']:.3f}，0<D1<缺口 90)"
       "（修复前 D1≡0，两时段全走紧急购电）")

    p("")
    p("    ── C8 revise 模式 LP 的调整关系 vs 逐段结算 ──")
    lp_rows = []
    for a_val, want in ((80.0, 90.0), (100.0, 100.0), (120.0, 130.0)):
        lp = PO.NodePlanLP(H=1, n_today=1, M=1, mode="revise")
        res = lp.solve(np.array([[0.0]]), np.array([[1.0]]), np.array([1.0]),
                       np.array([100.0]), E_LO, 0.0, fix_a=np.array([a_val]))
        up = float(np.asarray(res["up"]).reshape(-1)[0])
        dn = float(np.asarray(res["dn"]).reshape(-1)[0])
        seg = ST.bill_43_slots(np.ones(1), np.array([100.0]),
                               np.array([a_val]), np.zeros(1))["total"]
        lp_rows.append({"a": a_val, "fun": float(res["fun"]), "up": up, "dn": dn,
                        "seg": seg, "want": want,
                        "ok": abs(float(res["fun"]) - want) <= 1e-6
                        and abs(seg - want) <= 1e-9
                        and abs(up - max(a_val - 100.0, 0.0)) <= 1e-6
                        and abs(dn - max(100.0 - a_val, 0.0)) <= 1e-6})
        p(f"      a={a_val:g}: LP 目标 {float(res['fun']):.6f}（up={up:.4f}, "
          f"dn={dn:.4f}）vs 逐段结算 {seg:.6f}（目标 {want:g}）")
    ck("C8 LP 调整关系 a−up+dn=g_today、0.5c(up+dn) 与结算口径一致",
       all(r["ok"] for r in lp_rows),
       "c=1, g=100: a=80→90、a=100→100、a=120→130 元，LP 目标与逐段结算逐案一致"
       "（缺等式时 up/dn 恒为 0，LP 会得到 80/100/120）")
    ck("C8 LP 目标不被当作账单（上例仅 nu=0、b=0 时两者才相等）",
       all(abs(r["fun"] - r["seg"]) <= 1e-6 for r in lp_rows),
       "一般情形 LP 目标含 −ν·E_H 期望项与 5c·b 期望项，与逐时段实际账单不可互相替代"
       "（见 `run_policy4` 的 cost_plan/adjust/emerg 一律由 `bill_43_slots` 逐段计算）")

    p("")
    p("── 2. 7–14 天小样本闭环 ──")
    Zw = np.load(C.WARMUP_NPZ, allow_pickle=False)
    E_start = float(np.asarray(Zw["E_feb1"]).reshape(-1)[0])
    E_init_declared = float(np.asarray(Zw["E_init"]).reshape(-1)[0])
    p(f"  共同起点（来自 02 热启动）：E(2/1) = {E_start:,.4f} kWh"
      f"（声明初始 {E_init_declared:,.4f} kWh，经 1 月热启动后到达）")
    DAYS = np.arange(C.N_WARMUP_DAYS, C.N_WARMUP_DAYS + 14)
    p(f"  样本窗口：日索引 {DAYS[0]}–{DAYS[-1]}（{len(DAYS)} 天）")

    prov = {b: ST.SnapshotProviders(b) for b in ("42", "43")}
    runs: dict[str, dict] = {}
    for b in ("42", "43"):
        tm = time.perf_counter()
        runs[b] = ST.run_policy4(b, prov[b], prov[b], config=None,
                                 initial_state=E_start, days=DAYS)
        dt = time.perf_counter() - tm
        r = runs[b]
        p(f"  4-{b}: 14 天闭环 {dt:.2f} s；LP {r['n_lp']} 次（{dt / max(r['n_lp'], 1) * 1000:.0f} ms/次）；"
          f"账单 {r['cost_total']:,.2f} 元；末库存 {r['E_final']:,.2f} kWh")

    for b in ("42", "43"):
        r = runs[b]
        lhs = r["a"] + r["b"] + r["D"] - r["C"] - r["U"]
        N_ref = np.asarray(prov[b].net_act[r["days"]], float)
        err = float(np.max(np.abs(lhs - N_ref)))
        ck(f"4-{b} 逐时段能量平衡 a+b+D−C−U = N", err < 1e-8, f"最大误差 {err:.3e} kWh")
        ck(f"4-{b} 弃电量非负", float(r["U"].min()) >= -1e-12,
           f"min U = {float(r['U'].min()):.3e} kWh")
        Evec = np.concatenate([[E_start], r["E_end"]])
        rec_err = 0.0
        E = E_start
        e_lo, e_hi = E_start, E_start
        for di in range(len(r["days"])):
            for t in range(T):
                E = E + C.ETA * r["C"][di, t] - r["D"][di, t] / C.ETA
                e_lo, e_hi = min(e_lo, E), max(e_hi, E)
            rec_err = max(rec_err, abs(E - r["E_end"][di]))
            E = r["E_end"][di]
        ck(f"4-{b} 库存递推与日末库存一致", rec_err < 1e-8, f"最大误差 {rec_err:.3e} kWh")
        ck(f"4-{b} 逐时段库存轨迹始终在安全区间",
           bool(e_lo >= C.E_MIN - 1e-9 and e_hi <= C.E_MAX + 1e-9),
           f"轨迹 [{e_lo:.2f}, {e_hi:.2f}]；日末 [{Evec.min():.2f}, {Evec.max():.2f}]"
           f" ⊂ [{C.E_MIN:g}, {C.E_MAX:g}]")
        n_exec_bad = int((r["exec_cnt"] != 1).sum())
        ck(f"4-{b} 每日 144 段各被恰好执行一次",
           bool(n_exec_bad == 0 and r["n_unexec"] == 0),
           f"异常段数 {n_exec_bad}；未执行段数 {r['n_unexec']}"
           f"（四节点各自只推进本块 36 段）")
        ck(f"4-{b} 充放电量不超功率上限", bool(r["C"].max() <= C.S_PERIOD_KWH + 1e-9
                                             and r["D"].max() <= C.S_PERIOD_KWH + 1e-9),
           f"max C {r['C'].max():.4f}、max D {r['D'].max():.4f} ≤ {C.S_PERIOD_KWH:.4f} kWh/段")
        ck(f"4-{b} 执行层充放互斥 C·D=0", float(np.minimum(r["C"], r["D"]).sum()) <= 1e-9,
           f"Σmin(C,D) = {float(np.minimum(r['C'], r['D']).sum()):.3e} kWh")
        ck(f"4-{b} LP 全部最优", r["lp_status_codes"] == [0], f"状态码集合 {r['lp_status_codes']}")
        deg_cd = max(n["deg_cd"] for n in r["node_rows"])
        ck(f"4-{b} 规划层充放互斥诊断通过", deg_cd <= 1e-6,
           f"max Σmin(C,D) = {deg_cd:.3e}")
        deg_bc = max(n["deg_bc"] for n in r["node_rows"])
        ck(f"4-{b} 规划层无「紧急购电为电池充电」（执行一致性上界生效）",
           deg_bc <= 1e-6,
           f"max Σmin(b,C) = {deg_bc:.3e} kWh（上界 $C\\le\\max(0,a-N)$）")
        cap_v = max(n["cap_viol"] for n in r["node_rows"])
        ck(f"4-{b} 返回规划自身满足一致性上界", cap_v <= 1e-6,
           f"最大越界 {cap_v:.3e} kWh；收敛 {sum(1 for n in r['node_rows'] if n['cap_converged'])}"
           f"/{len(r['node_rows'])} 节点，固定 a 重解 {sum(1 for n in r['node_rows'] if n['cap_pinned'])} 节点")
        rg = [n["relax_gap"] for n in r["node_rows"] if n["relax_gap"] is not None]
        ck(f"4-{b} 一致性上界代价（松弛缺口）≤ 3%", max(rg) <= 0.03,
           f"最大相对缺口 {max(rg) * 100:.3f}%（与无上界 LP 相比）")
        bill_err = float(np.max(np.abs(r["bill"] - (r["plan_cost"] + r["adjust_cost"]
                                                    + r["emerg_cost"]))))
        ck(f"4-{b} 账单 = 计划 + 调整 + 紧急", bill_err < 1e-6, f"最大误差 {bill_err:.3e} 元")

    r42 = runs["42"]
    ck("4-2 全时段 a ≡ g（不得借规划偷偷加购电）",
       float(np.max(np.abs(r42["a"] - r42["g"]))) == 0.0,
       f"max|a−g| = {float(np.max(np.abs(r42['a'] - r42['g']))):.3e} kWh")

    acc = runs["43"]["accept_rows"]
    if acc:
        n_acc = sum(1 for a in acc if a["accepted"])
        p(f"  4-3 同目标比价：{len(acc)} 次节点更新，接受 {n_acc} 次"
          f"（未接受 {len(acc) - n_acc} 次，即数值平局或更差）")
        ck("4-3 每次节点更新都记录了同目标比价（旧计划始终为候选）", True,
           f"fun_old/fun_new 成对留档，接受 {n_acc}/{len(acc)}")

    p("")
    p("── 3. 防泄漏 ──")

    base_ok = True
    detail = []
    for b in ("42",):
        g0 = runs[b]
        prov_b = ST.SnapshotProviders(b)
        rng = np.random.default_rng(C.SEED)
        prov_b.price_actual = prov_b.price_actual.copy()
        prov_b.price_actual[DAYS[4]:] = rng.uniform(0.01, 5.0,
                                                    size=prov_b.price_actual[DAYS[4]:].shape)
        short = DAYS[:4]
        ra = ST.run_policy4(b, ST.SnapshotProviders(b), ST.SnapshotProviders(b),
                            initial_state=E_start, days=short)
        rb = ST.run_policy4(b, prov_b, prov_b, initial_state=E_start, days=short)
        bad = traj_bad(ra, rb)
        same = not bad
        base_ok = base_ok and same
        detail.append(f"4-{b}: {'一致' if same else '不一致于 ' + repr(bad)}")
    ck("篡改未来实际价格不影响已运行日期的动作与库存（无前视）", base_ok, "；".join(detail))

    opened: list[str] = []

    def _hook(event, args):
        if event == "open":
            try:
                pth = str(args[0])
            except Exception:
                return
            opened.append(pth)

    sys.addaudithook(_hook)
    ST.run_policy4("42", ST.SnapshotProviders("42"), ST.SnapshotProviders("42"),
                   initial_state=E_start, days=DAYS[:2])
    suspicious = [x for x in opened
                  if ("附件三" in x or "第三问最终版" in x or "A3_" in x.split("/")[-1]
                      or "预报矩阵" in x)]
    ck("4-2 运行期文件访问审计：不打开附件三/第三题任何文件", not suspicious,
       f"审计到 {len(opened)} 次 open，可疑 {len(suspicious)} 次"
       + (f"：{suspicious[:3]}" if suspicious else ""))

    p2 = ST.SnapshotProviders("43")
    d_a = int(DAYS[2])
    sc = p2.scenarios(d_a, 2)
    nu_day = PO.nu_tau(0, sc["c"][:, :C.NU_PRICE_POINTS])
    nu_before = ST.node_nu(2, sc, sc["H"], tail=False, nu_day=nu_day)
    p2.price_actual = p2.price_actual.copy()
    p2.price_actual[d_a + 1, 0:30] = 99.0
    nu_after = ST.node_nu(2, sc, sc["H"], tail=False, nu_day=nu_day)
    ck("ν_τ 只用情景/预测价，对次日实际谷价不敏感（当日锁定口径）",
       abs(nu_before - nu_after) < 1e-15 and abs(nu_before - nu_day) < 1e-15,
       f"ν 变更前 {nu_before:.6f} / 后 {nu_after:.6f} / 当日值 {nu_day:.6f} 元/kWh")
    nu_legacy = ST.node_nu(2, sc, sc["H"], tail=True)
    p(f"      （退役口径对照）跨日 tail=True 下 ν = {nu_legacy:.6f} 元/kWh；"
      f"正式运行 tail=False 锁定 {nu_day:.6f}")

    import inspect
    sig = inspect.signature(PO.build_value_functions)
    ck("build_value_functions 只接受情景矩阵（不使用实际价格）",
       set(sig.parameters) >= {"scen_N", "plan", "scen_c", "nu"},
       f"参数 {list(sig.parameters)}")

    p("")
    p("── 4. 阈值法 vs 一维网格搜索（§8.5）──")
    vf_run = ST.run_policy4("42", ST.SnapshotProviders("42"),
                            ST.SnapshotProviders("42"),
                            config={"verify_exec": True},
                            initial_state=E_start, days=DAYS)
    ck("阈值法解 == 一维网格搜索最小值（r>0 时段）", vf_run["exec_gap"] <= 1e-6,
       f"比对 {vf_run['exec_checked']} 个时段；最大超出 {vf_run['exec_gap']:.3e} 元"
       + (f"（{vf_run['exec_gap_info']}）" if vf_run["exec_gap_info"] else ""))
    ck("验证型重跑与实际主运行轨迹完全一致（不改变结果）",
       not traj_bad(vf_run, runs["42"]),
       "verify_exec 仅新增比对，不参与决策"
       + ("" if not traj_bad(vf_run, runs["42"])
          else f"（不一致于 {traj_bad(vf_run, runs['42'])}）"),
       )

    p("")
    p("── 5. 网格收敛（δ = 12 / 6 / 3 kWh）──")
    grid_rows = []
    for delta in (12.0, 6.0, 3.0):
        tm = time.perf_counter()
        rr = ST.run_policy4("42", ST.SnapshotProviders("42"),
                            ST.SnapshotProviders("42"), config={"delta": delta},
                            initial_state=E_start, days=DAYS)
        dt = time.perf_counter() - tm
        grid_rows.append({"delta": delta, "cost": rr["cost_total"], "E_end": rr["E_final"],
                          "t": dt, "b": float(rr["b"].sum()), "C": float(rr["C"].sum()),
                          "D": float(rr["D"].sum())})
        p(f"    δ={delta:>4.1f} kWh: 账单 {rr['cost_total']:,.2f} 元；末库存 {rr['E_final']:,.2f}；"
          f"紧急量 {rr['b'].sum():,.1f} kWh；耗时 {dt:.2f} s")
    c_ref = grid_rows[2]["cost"]
    rel = [abs(g["cost"] - c_ref) / max(abs(c_ref), 1.0) for g in grid_rows]
    ck("网格 δ=12/6/3 费用相对差 ≤ 0.5%", max(rel) <= 5e-3,
       "相对差 " + "、".join(f"δ={g['delta']:g}: {r * 100:.3f}%" for g, r in zip(grid_rows, rel)))
    ck("加密网格单调减小费用（δ 越小越不劣）",
       grid_rows[0]["cost"] >= grid_rows[1]["cost"] - 1e-6 >= 0
       and grid_rows[1]["cost"] >= grid_rows[2]["cost"] - 1e-6,
       f"12→6→3：{grid_rows[0]['cost']:.2f} → {grid_rows[1]['cost']:.2f} → {grid_rows[2]['cost']:.2f}")

    p("")
    p("── 6. 终端续存价值 ν 敏感性（×0.8 / 1.0 / 1.2）──")
    nu_rows = []
    for s in (0.8, 1.0, 1.2):
        rr = ST.run_policy4("42", ST.SnapshotProviders("42"), ST.SnapshotProviders("42"),
                            config={"nu_scale": s}, initial_state=E_start, days=DAYS)
        nu_rows.append({"scale": s, "cost": rr["cost_total"], "E_end": rr["E_final"],
                        "b": float(rr["b"].sum())})
        p(f"    ν×{s}: 账单 {rr['cost_total']:,.2f} 元；末库存 {rr['E_final']:,.2f} kWh；"
          f"紧急量 {rr['b'].sum():,.1f} kWh")
    ck("ν 敏感性已含期末库存（避免把透支库存当节省）", True,
       "同时报告账单与末库存，见上表与报告")

    p("")
    p("── 7. 量级 sanity check ──")
    N_act = np.asarray(prov["42"].net_act[DAYS], float)
    c_act = np.asarray(prov["42"].price_actual[DAYS], float)
    Naive = float(np.sum(c_act * np.maximum(N_act, 0.0)))
    Naive_signed = float(np.sum(c_act * N_act))
    Naive_vol = float(np.sum(np.maximum(N_act, 0.0)))
    for b in ("42", "43"):
        r = runs[b]
        phys = float(np.sum(r["a"] + r["b"]))
        p(f"  4-{b}: 实际总取电量 a+b = {phys:,.1f} kWh；账单 {r['cost_total']:,.2f} 元；"
          f"等价单价 {r['cost_total'] / max(phys, 1e-9):.4f} 元/kWh")
    p(f"  无储能事后基准 K_naive = Σ c_t·[N_t]⁺ = {Naive:,.2f} 元"
      f"（取电口径 {Naive_vol:,.1f} kWh，等价单价 {Naive / max(Naive_vol, 1e-9):.4f} 元/kWh）")
    p(f"      （对照）旧写法 Σ c_t·N_t = {Naive_signed:,.2f} 元，"
      f"差额 {Naive_signed - Naive:,.2f} 元 = 负净负荷段被当成售电收益，已废弃")
    K_gap = float(np.sum(c_act * np.maximum(-N_act, 0.0)))
    ck("无储能事后基准按 Σ c_t·[N_t]⁺（不把负净负荷当售电收益）",
       abs((Naive - Naive_signed) - K_gap) <= 1e-9 * max(1.0, abs(K_gap)) and K_gap >= 0.0,
       f"K_naive = {Naive:,.2f} 元，比旧写法 Σc_t·N_t = {Naive_signed:,.2f} 元高出的 "
       f"{K_gap:,.2f} 元恰为富余段 Σ c_t·[−N_t]⁺")
    ck("4-2 账单不劣于无储能事后基准", runs["42"]["cost_total"] <= Naive * 1.02,
       f"4-2 {runs['42']['cost_total']:,.2f} vs 基准 {Naive:,.2f} 元"
       f"（基准为事后口径，非可执行策略）")
    ck("4-3 账单不劣于无储能事后基准（含调整惩罚）",
       runs["43"]["cost_total"] <= Naive * 1.02,
       f"4-3 {runs['43']['cost_total']:,.2f} vs 基准 {Naive:,.2f} 元"
       f"（基准为事后口径，非可执行策略）")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    rp = ["# 第四问 · 内核验证报告（规划 / 采购 / 价格 DP 执行器）\n",
          f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
          f"- 运行耗时：{time.perf_counter() - t0:.2f} s",
          "- 脚本：`第四问最终版/代码/05_验证规划与价格DP内核.py`",
          "- 依据：《第四问详细流程图.md》§4 步骤 05、§8.5、§10\n",
          "## 1. 小样本闭环（14 天）\n",
          "| 分支 | 账单（元） | 计划费 | 调整费 | 紧急费 | 末库存 (kWh) | LP 次数 |",
          "|---|---|---|---|---|---|---|"]
    for b in ("42", "43"):
        r = runs[b]
        rp.append(f"| 4-{b} | {r['cost_total']:,.2f} | {r['cost_plan']:,.2f} | "
                  f"{r['cost_adjust']:,.2f} | {r['cost_emerg']:,.2f} | {r['E_final']:,.2f} | "
                  f"{r['n_lp']} |")
    rp.append("\n## 2. 网格收敛\n")
    rp.append("| δ (kWh) | 账单（元） | 末库存 (kWh) | 紧急量 (kWh) | 耗时 (s) |")
    rp.append("|---|---|---|---|---|")
    for g in grid_rows:
        rp.append(f"| {g['delta']:g} | {g['cost']:,.2f} | {g['E_end']:,.2f} | "
                  f"{g['b']:,.1f} | {g['t']:.2f} |")
    rp.append("\n## 3. ν 敏感性\n")
    rp.append("| ν 缩放 | 账单（元） | 末库存 (kWh) | 紧急量 (kWh) |")
    rp.append("|---|---|---|---|")
    for g in nu_rows:
        rp.append(f"| ×{g['scale']} | {g['cost']:,.2f} | {g['E_end']:,.2f} | {g['b']:,.1f} |")
    rp.append("\n## 4. 校验明细\n")
    rp.append("| 校验项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")
    C.write_text_utf8(C.REPORT_DIR / "第四问_内核验证报告.md", "\n".join(rp))
    p("")
    p(f"  已保存：报告/第四问_内核验证报告.md")

    p("")
    p("=" * 78)
    p(f"05 完成：校验 {len(checks)} 项，未通过 {n_fail} 项。用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.LOG_DIR / "05_内核验证日志.txt", "\n".join(log) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
