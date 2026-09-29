#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
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
P = _load("_price4.py", "q4_price")
PO = _load("_policy4.py", "q4_policy")
ST = _load("_settlement4.py", "q4_settle")

import numpy as np

T = C.PERIODS_PER_DAY
K0 = C.TAU_PERIOD_INDEX
TAU = list(C.TAU_HOURS)

CONFIG_PATH = C.CONFIG_DIR / "第四问_受控对照配置.json"
OUT_DIR = C.RESULT_DIR / "受控对照"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class UnifiedProvider:

    def __init__(self, method: str, max_m: int, window: int):
        Z4 = np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False)
        self.dates = np.array([str(s) for s in Z4["dates"]], dtype="<U10")
        self.price_actual = np.asarray(Z4["price_actual"], float)
        Zf = np.load(C.PRICE_FORECAST_NPZ, allow_pickle=False)
        self.method = str(method)
        self.chat_corr = np.asarray(Zf[f"chat_corr__{self.method}"], float)
        Z2 = C.load_q2_matrix()
        S2 = C.load_q2_scenarios()
        self.net_act = np.asarray(Z2["net_load_energy_kwh"], float)
        self.load_act = np.asarray(Z2["load_energy_kwh"], float)
        self.pv_act = np.asarray(Z2["pv_energy_kwh"], float)
        self.ctx = {
            "price_act": self.price_actual,
            "chat_corr": self.chat_corr,
            "L_act": self.load_act,
            "Lhat": np.asarray(S2["Lhat"], float),
            "Vhat_q2": np.asarray(S2["Vhat"], float),
            "V_act": self.pv_act,
        }
        self.max_m = int(max_m)
        self.window = int(window)
        self._cache: dict = {}

    def price(self, d: int, t: int) -> float:
        return float(self.price_actual[d, t])

    def scenarios(self, d: int, ti: int) -> dict:
        key = (int(d), int(ti), self.max_m, self.window)
        got = self._cache.get(key)
        if got is None:
            raw = P.build_node_scenarios(d, ti, self.ctx, "42",
                                         max_m=self.max_m, window=self.window)
            got = {
                "N": np.asarray(raw["N_scen"], float).copy(),
                "c": np.asarray(raw["price_scen"], float).copy(),
                "w": np.full(int(raw["M"]), 1.0 / int(raw["M"])),
                "origin": np.asarray(raw["origin"], int).copy(),
                "M": int(raw["M"]), "H": int(raw["H"]),
                "fallback": 0 if not raw["fallbacks"] else 1,
            }
            self._cache[key] = got
        return got


class FixedPriceEnv:

    _CURVE = None

    def __init__(self, base: UnifiedProvider):
        if FixedPriceEnv._CURVE is None:
            FixedPriceEnv._CURVE = np.asarray(C.ref_price144(), float).reshape(-1)
        self._b = base
        self.c_fix = FixedPriceEnv._CURVE
        assert self.c_fix.size == T
        pa = np.asarray(base.price_actual, float)
        self.price_actual = np.tile(self.c_fix, (pa.shape[0], 1))
        self.method = f"{base.method}+fixed_price"
        self.max_m = base.max_m
        self.window = base.window

    def __getattr__(self, key):
        return getattr(self._b, key)

    def _win(self, k0: int, H: int) -> np.ndarray:
        return self.c_fix[(k0 + np.arange(H)) % T]

    def price(self, d: int, t: int) -> float:
        return float(self.c_fix[t])

    def scenarios(self, d: int, ti: int) -> dict:
        sc = self._b.scenarios(d, ti)
        M, H = int(sc["M"]), int(sc["H"])
        out = dict(sc)
        out["c"] = np.broadcast_to(self._win(K0[ti], H), (M, H)).copy()
        return out


def official_day_start_map() -> tuple[dict, dict]:
    maps = {}
    for tag, path in (("42", C.BACKTEST_42_NPZ), ("43", C.BACKTEST_43_NPZ)):
        Z = np.load(path, allow_pickle=False)
        days = np.asarray(Z["days"], int)
        Ech = np.asarray(Z["E_chain"], float)
        maps[tag] = {int(days[i]): float(Ech[i]) for i in range(days.size)}
    return maps["42"], maps["43"]


def date_to_index(dates: np.ndarray, s: str) -> int:
    hit = np.where(dates == s)[0]
    if hit.size == 0:
        raise KeyError(f"日期 {s} 不在附件四日期序列中")
    return int(hit[0])


def run_arm(branch: str, env, days, E0: float, cfg_over: dict | None = None) -> dict:
    over = {"tail": False, "value_nodes": tuple(load_config()["unified_conditions"]["value_nodes"]),
            "cap_iter": int(load_config()["unified_conditions"]["cap_iter"])}
    over.update(cfg_over or {})
    return ST.run_policy4(branch, env, env, config=over,
                          initial_state=float(E0), days=np.asarray(days, int))


def recompute(r: dict, branch: str, env) -> dict:
    n = len(r["days"])
    N = np.asarray(env.net_act[np.asarray(r["days"], int)], float)
    g, a, b = r["g"], r["a"], r["b"]
    Cc, Dd, U = r["C"], r["D"], r["U"]
    c = r["c"]
    out = {}
    out["bal_max"] = float(np.abs(a + b + Dd - Cc - U - N).max())
    err = 0.0
    E = float(r["E_start"])
    lo = hi = E
    for di in range(n):
        for t in range(T):
            E = E + C.ETA * Cc[di, t] - Dd[di, t] / C.ETA
            lo, hi = min(lo, E), max(hi, E)
        err = max(err, abs(E - float(r["E_end"][di])))
        E = float(r["E_end"][di])
    out["soc_err"] = float(err)
    out["E_min"], out["E_max"] = float(lo), float(hi)
    out["E_under"] = float(max(0.0, C.E_MIN - lo))
    out["E_over"] = float(max(0.0, hi - C.E_MAX))
    out["E_final_chain"] = float(E)
    out["C_max"] = float(Cc.max()); out["D_max"] = float(Dd.max())
    out["mutex"] = float(np.minimum(Cc, Dd).sum())
    out["neg_min"] = float(min(g.min(), a.min(), b.min(), Cc.min(), Dd.min(), U.min()))
    if branch == "42":
        plan = float((c * g).sum()); adj = 0.0
    else:
        plan = float((c * np.minimum(g, a)).sum())
        adj = float((0.5 * c * np.maximum(g - a, 0)).sum()
                    + (1.5 * c * np.maximum(a - g, 0)).sum())
    emg = float((5.0 * c * b).sum())
    out["bill_indep"] = plan + adj + emg
    out["plan_indep"] = plan; out["adjust_indep"] = adj; out["emerg_indep"] = emg
    out["bill_engine"] = float(r["cost_total"])
    out["max_ag"] = float(np.abs(a - g).max())
    out["n_days"] = n
    return out


def summarize(r: dict, branch: str, env, arm: str, case: str, E0: float,
              wall: float) -> dict:
    v = recompute(r, branch, env)
    rg = [n["relax_gap"] for n in r["node_rows"] if n["relax_gap"] is not None]
    n_cap = len(r["node_rows"])
    over5 = [x for x in rg if x > 0.05]
    return {
        "arm": arm, "case": case, "branch": branch,
        "n_days": len(r["days"]),
        "dates": f"{env.dates[int(r['days'][0])]}..{env.dates[int(r['days'][-1])]}",
        "E_start": float(E0), "E_final": float(r["E_final"]),
        "bill_total": float(r["cost_total"]),
        "bill_plan": float(r["cost_plan"]),
        "bill_adjust": float(r["cost_adjust"]),
        "bill_emerg": float(r["cost_emerg"]),
        "sum_g": float(r["g"].sum()), "sum_a": float(r["a"].sum()),
        "sum_b": float(r["b"].sum()), "sum_C": float(r["C"].sum()),
        "sum_D": float(r["D"].sum()), "sum_U": float(r["U"].sum()),
        "n_lp": int(r["n_lp"]), "wall_s": float(wall),
        "bal_max": v["bal_max"], "soc_err": v["soc_err"],
        "E_min": v["E_min"], "E_max": v["E_max"],
        "E_under": v["E_under"], "E_over": v["E_over"],
        "C_max": v["C_max"], "D_max": v["D_max"], "mutex": v["mutex"],
        "neg_min": v["neg_min"], "max_ag": v["max_ag"],
        "bill_indep": v["bill_indep"], "bill_engine": v["bill_engine"],
        "bill_gap": abs(v["bill_indep"] - v["bill_engine"]),
        "relax_gap_max": float(max(rg)) if rg else 0.0,
        "relax_gap_mean": float(np.mean(rg)) if rg else 0.0,
        "n_nodes": n_cap, "n_nodes_over5": len(over5),
        "cap_pinned": int(sum(1 for x in r["node_rows"] if x["cap_pinned"])),
    }


def print_summary_row(s: dict) -> str:
    return (f"  [{s['arm']}] {s['case']:<26s} {s['dates']:<25s} "
            f"账单 {s['bill_total']:>15,.2f} 元  "
            f"计划 {s['bill_plan']:>13,.2f} 调整 {s['bill_adjust']:>12,.2f} "
            f"紧急 {s['bill_emerg']:>11,.2f}  "
            f"E末 {s['E_final']:>9,.2f}  LP {s['n_lp']:>4d}  {s['wall_s']:>6.1f}s")


FIELDS = ["arm", "case", "branch", "n_days", "dates", "E_start", "E_final",
          "bill_total", "bill_plan", "bill_adjust", "bill_emerg",
          "sum_g", "sum_a", "sum_b", "sum_C", "sum_D", "sum_U",
          "n_lp", "wall_s", "bal_max", "soc_err",
          "E_min", "E_max", "E_under", "E_over",
          "C_max", "D_max", "mutex", "neg_min", "max_ag",
          "bill_indep", "bill_engine", "bill_gap",
          "relax_gap_max", "relax_gap_mean", "n_nodes", "n_nodes_over5",
          "cap_pinned"]


def write_rows(path: Path, rows: list[dict]) -> None:
    merged: dict = {}
    if path.exists():
        try:
            old = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
            for r in old:
                merged[(r.get("arm"), r.get("case"))] = r
        except Exception:
            pass
    for r in rows:
        merged[(r.get("arm"), r.get("case"))] = r
    ordered = sorted(merged.values(), key=lambda r: (str(r.get("case")), str(r.get("arm"))))
    C.write_csv_utf8_sig(path, FIELDS, [[r.get(k) for k in FIELDS] for r in ordered])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="第四问 2x2 受控对照 + 单因素敏感性")
    ap.add_argument("--plan", action="store_true", help="只打印方案")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 日期 × 四臂")
    ap.add_argument("--run", choices=["all", "sensitivity", "arms", "cont", "year"],
                    default=None, help="运行范围")
    ap.add_argument("--only-arms", default="", help="只跑指定臂，逗号分隔，如 C,D")
    args = ap.parse_args(argv)

    cfg = load_config()
    uc = cfg["unified_conditions"]
    C.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Z4 = np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False)
    dates = np.array([str(s) for s in Z4["dates"]], dtype="<U10")
    e42, _e43 = official_day_start_map()

    plan_dates = cfg["sampling_plan"]
    spec = list(plan_dates["spec_dates"])
    worst = plan_dates["relax_gap_worst_date"]
    cont = plan_dates["contiguous_14d"]

    def P_(*a):
        print(*a, flush=True)

    P_("=" * 100)
    P_("第四问 20 —— 2×2 受控对照与单因素敏感性（独立实验）")
    P_("=" * 100)
    P_(f"  统一条件：预测器 = {uc['price_forecaster']}；情景源 = 附件二因果链路；"
       f"M = {uc['scenario_max_count_M']}；δ = {uc['soc_grid_kwh_delta']}；"
       f"值节点 = {uc['value_nodes']}；tail(跨日) = {uc['tail']}")
    P_(f"  采样：指定日期 {spec}；连续14天 {cont['start']}…{cont['end']}")
    P_(f"  输出：{OUT_DIR.relative_to(C.PROJECT_DIR)}/")

    def date_di(s: str) -> int:
        return date_to_index(dates, s)

    arms_all = ["A", "B", "C", "D"]
    if args.only_arms:
        arms_all = [x.strip().upper() for x in args.only_arms.split(",") if x.strip()]

    def env_for(arm: str, max_m=None, window=None, method=None):
        base = UnifiedProvider(method or uc["price_forecaster"],
                               max_m or uc["scenario_max_count_M"],
                               window or uc["scenario_window_days"])
        if cfg["arms"][arm]["price_env"] == "fixed":
            return FixedPriceEnv(base)
        return base

    def branch_for(arm: str) -> str:
        return "43" if cfg["arms"][arm]["purchase_adjust"] else "42"

    if args.plan or (args.run is None and not args.smoke):
        P_("")
        P_("── 计划（--plan；未做任何计算）──")
        for arm in arms_all:
            for s in spec + [worst]:
                P_(f"  臂 {arm}（{cfg['arms'][arm]['_label']}）单日独立重放 {s}；"
                   f"共同日初库存 = {e42[date_di(s)]:,.4f} kWh")
            P_(f"  臂 {arm} 连续14天 {cont['start']}…{cont['end']}，"
               f"共同起点 {cont['start']} 日初 = {e42[date_di(cont['start'])]:,.4f} kWh")
        P_("  敏感性（单因素）：δ 6 → 3；M 30 → 50（嵌套），"
           f"日期 {plan_dates['sensitivity_dates']}，只跑 C/D 两臂")
        return 0

    rows: list[dict] = []
    t_all = time.perf_counter()

    def do_case(arm: str, case: str, days, E0, env=None, cfg_over=None) -> dict:
        branch = branch_for(arm)
        env = env or env_for(arm)
        t0 = time.perf_counter()
        r = run_arm(branch, env, days, E0, cfg_over)
        wall = time.perf_counter() - t0
        s = summarize(r, branch, env, arm, case, E0, wall)
        rows.append(s)
        P_(print_summary_row(s))
        write_rows(OUT_DIR / "受控对照_结果汇总.csv", rows)
        tag = f"{arm}_{case}".replace(".", "").replace(" ", "_").replace("…", "-")
        np.savez_compressed(
            OUT_DIR / f"{tag}.npz",
            arm=np.asarray(arm), case=np.asarray(case), branch=np.asarray(branch),
            days=np.asarray(r["days"], int), dates=r["dates"],
            E_start=np.asarray(float(E0)), E_final=np.asarray(float(r["E_final"])),
            g=r["g"], a=r["a"], b=r["b"], C=r["C"], D=r["D"], U=r["U"],
            c=r["c"], N=np.asarray(env.net_act[np.asarray(r["days"], int)], float),
            bill=r["bill"], plan_cost=r["plan_cost"], adjust_cost=r["adjust_cost"],
            emerg_cost=r["emerg_cost"], E_end=r["E_end"],
            node_relax_gap=np.asarray([x["relax_gap"] if x["relax_gap"] is not None
                                       else np.nan for x in r["node_rows"]], float),
            node_tau=np.asarray([x["tau"] for x in r["node_rows"]], int),
            node_date=np.asarray([x["date"] for x in r["node_rows"]], "<U10"),
            node_cap_pinned=np.asarray(
                [bool(x["cap_pinned"]) for x in r["node_rows"]], bool),
        )
        return s

    if args.smoke:
        d0 = spec[0]
        P_("")
        P_(f"── 冒烟：{d0} × 四臂 ──")
        for arm in arms_all:
            do_case(arm, f"smoke_{d0}", [date_di(d0)], e42[date_di(d0)])
        write_rows(OUT_DIR / "受控对照_冒烟.csv", rows)
        P_(f"\n  冒烟完成，{time.perf_counter() - t_all:.1f}s → {OUT_DIR.name}/受控对照_冒烟.csv")
        return 0

    if args.run in ("all", "arms"):
        P_("")
        P_("── 1. 指定日期独立重放（四臂 × 5 日期，共用同一真实日初库存）──")
        for arm in arms_all:
            for s in spec + [worst]:
                do_case(arm, f"day_{s}", [date_di(s)], e42[date_di(s)])

    if args.run in ("all", "cont"):
        P_("")
        P_(f"── 2. 连续14天对照 {cont['start']}…{cont['end']}（四臂各自连续递推）──")
        ci = date_di(cont["start"]); cj = date_di(cont["end"])
        days14 = np.arange(ci, cj + 1, dtype=int)
        assert days14.size == int(cont["days"]), days14.size
        E0 = e42[ci]
        for arm in arms_all:
            do_case(arm, f"cont14_{cont['start']}_{cont['end']}", days14, E0)

    if args.run in ("all", "sensitivity"):
        P_("")
        P_("── 3. 单因素敏感性（C/D 两臂 × 5 日期 × {δ, M}）──")
        sens_dates = list(plan_dates["sensitivity_dates"])
        delta0 = float(uc["soc_grid_kwh_delta"]); delta1 = float(cfg["sensitivity_plan"]["grid"]["finer"])
        M0 = int(uc["scenario_max_count_M"]); M1 = int(cfg["sensitivity_plan"]["scenario_count"]["larger_M"])
        W1 = int(cfg["sensitivity_plan"]["scenario_count"]["window_for_larger"])
        for arm in [a for a in ("C", "D") if a in arms_all]:
            env_b = env_for(arm)
            env_d = env_for(arm)
            env_m = env_for(arm, max_m=M1, window=W1)
            for s in sens_dates:
                d = date_di(s); E0 = e42[d]
                do_case(arm, f"sens_base_{s}", [d], E0, env=env_b, cfg_over={})
                do_case(arm, f"sens_delta3_{s}", [d], E0, env=env_d,
                        cfg_over={"delta": delta1})
                do_case(arm, f"sens_M50_{s}", [d], E0, env=env_m, cfg_over={})

    if args.run in ("all", "year"):
        P_("")
        P_("── 4. 四臂全年（评分期 334 天，统一条件下年度口径）──")
        y0 = date_di("2025-02-01"); y1 = date_di("2025-12-31")
        daysY = np.arange(y0, y1 + 1, dtype=int)
        P_(f"  评分期 {daysY.size} 天；共同起点 {env_for('A').dates[y0]} 日初 = "
           f"{e42[y0]:,.4f} kWh")
        for arm in arms_all:
            do_case(arm, "year_2025-02-01_2025-12-31", daysY, e42[y0])

    if rows:
        write_rows(OUT_DIR / "受控对照_结果汇总.csv", rows)
        P_("")
        P_(f"  汇总：{OUT_DIR.relative_to(C.PROJECT_DIR)}/受控对照_结果汇总.csv"
           f"（{len(rows)} 行）")
    P_(f"\n  总耗时 {time.perf_counter() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
