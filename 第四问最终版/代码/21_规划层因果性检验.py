#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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
E20 = _load("20_受控对照实验.py", "q4_ctrl")

import numpy as np

T = C.PERIODS_PER_DAY
OUT_DIR = C.RESULT_DIR / "受控对照"
FIELDS = ["arm", "date", "variant", "meaning", "expect",
          "dg_max", "da_max", "db_max", "planned_same", "bill_base", "bill_var",
          "verdict", "ok"]


def _diff(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(np.asarray(x, float) - np.asarray(y, float)).max())


def variants(pa: np.ndarray, d: int) -> list[tuple[str, str, str, str]]:
    out = []
    z = pa.copy(); z[d + 1:] *= 3.0
    out.append(("V_FUT", "次日及以后『实际价格』×3（尚未发生）", "计划不变", z))
    z = pa.copy(); z[d] *= 3.0
    out.append(("V_TODAY", "当日整日『实际价格』×3", "计划不变（结算可变）", z))
    z = pa.copy(); z[d, 1:] *= 3.0
    out.append(("V_TAIL", "当日 t>=1『实际价格』×3（当日尚未发生的时段）",
                "计划不变", z))
    z = pa.copy(); z[:d] *= 3.0
    out.append(("V_PAST", "d-1 及更早『实际价格』×3（历史，合规可用）",
                "允许变化", z))
    return out


def run_variant(arm: str, d: int, E0: float, tag: str,
                mutate) -> tuple[dict, dict]:
    uc = E20.load_config()["unified_conditions"]

    def fresh():
        env = E20.UnifiedProvider(uc["price_forecaster"],
                                  uc["scenario_max_count_M"],
                                  uc["scenario_window_days"])
        if E20.load_config()["arms"][arm]["price_env"] == "fixed":
            env = E20.FixedPriceEnv(env)
        return env

    branch = "43" if E20.load_config()["arms"][arm]["purchase_adjust"] else "42"
    cfg = {"cap_iter": int(uc["cap_iter"])}

    eb = fresh()
    rb = E20.ST.run_policy4(branch, eb, eb, config=dict(cfg),
                            initial_state=float(E0), days=np.asarray([d], int))

    ev = fresh()
    mutate(ev)
    ev._cache.clear() if hasattr(ev, "_cache") else None
    if hasattr(ev, "_b") and hasattr(ev._b, "_cache"):
        ev._b._cache.clear()
    rv = E20.ST.run_policy4(branch, ev, ev, config=dict(cfg),
                            initial_state=float(E0), days=np.asarray([d], int))
    return rb, rv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="第四问 规划层因果性检验")
    ap.add_argument("--plan", action="store_true", help="只打印检验项")
    ap.add_argument("--arms", default="C,D", help="参与检验的臂，默认 C,D")
    ap.add_argument("--dates", default="2025-03-20,2025-12-21",
                    help="目标日期，逗号分隔")
    args = ap.parse_args(argv)

    cfg = E20.load_config()
    uc = cfg["unified_conditions"]
    C.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    e42, _ = E20.official_day_start_map()

    Z4 = np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False)
    dates = np.array([str(s) for s in Z4["dates"]], dtype="<U10")
    Zf = np.load(C.PRICE_FORECAST_NPZ, allow_pickle=False)
    fkey = f"chat_corr__{uc['price_forecaster']}"
    chat = np.asarray(Zf[fkey], float)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    day_list = [s.strip() for s in args.dates.split(",") if s.strip()]

    P_ = print
    P_("=" * 100)
    P_("第四问 21 —— 规划层因果性与信息可用性检验")
    P_("=" * 100)
    P_(f"  预测器 = {uc['price_forecaster']}；M = {uc['scenario_max_count_M']}；"
       f"窗口 = {uc['scenario_window_days']}；臂 = {arms}")
    P_(f"  价格矩阵 price_actual 形状 = {np.asarray(Z4['price_actual']).shape}；"
       f"预测项 {fkey} 形状 = {chat.shape}")
    P_(f"  目标日期 = {day_list}")

    if args.plan:
        for arm in arms:
            for s in day_list:
                d = E20.date_to_index(dates, s)
                P_(f"  臂 {arm} @ {s}（日索引 {d}，日初 {e42[d]:,.4f} kWh）："
                   f"{[v[0] for v in variants(np.asarray(Z4['price_actual'], float), d)]}")
        return 0

    rows = []
    t_all = time.perf_counter()
    P_("")
    P_("── 扰动检验（只替换 ctx['price_act'] 引用；plan/结算分离）──")
    for arm in arms:
        for s in day_list:
            d = E20.date_to_index(dates, s)
            E0 = e42[d]
            pa = np.asarray(Z4["price_actual"], float)
            for tag, meaning, expect, arr in variants(pa, d):
                def mk(a=arr):
                    return lambda env: env.ctx.__setitem__("price_act", a)
                rb, rv = run_variant(arm, d, E0, tag, mk())
                dg = _diff(rb["g"], rv["g"])
                da = _diff(rb["a"], rv["a"])
                db = _diff(rb["b"], rv["b"])
                same = (dg <= 1e-9)
                if tag in ("V_FUT", "V_TODAY", "V_TAIL"):
                    ok = bool(same)
                elif tag == "V_PAST":
                    ok = True
                else:
                    ok = bool(same)
                verdict = ("不变（合规）" if same else "计划改变（信息穿越）")
                rows.append({
                    "arm": arm, "date": s, "variant": tag, "meaning": meaning,
                    "expect": expect, "dg_max": dg, "da_max": da, "db_max": db,
                    "planned_same": same, "bill_base": float(rb["cost_total"]),
                    "bill_var": float(rv["cost_total"]),
                    "verdict": verdict, "ok": ok,
                })
                P_(f"  [{arm}] {s} {tag:<8s} Δg={dg:>10.6f} Δa={da:>10.6f} "
                   f"Δb={db:>10.6f}  账单 {rb['cost_total']:>12,.2f} → "
                   f"{rv['cost_total']:>12,.2f}  {verdict}")

    P_("")
    P_("── 未来日『预测』扰动（改 chat_corr 的 d+1.. 行）──")
    for arm in arms:
        for s in day_list:
            d = E20.date_to_index(dates, s)
            E0 = e42[d]
            def mk(_d=d):
                def f(env):
                    cc = np.asarray(env.ctx["chat_corr"], float)
                    key = "chat_corr" if cc.shape == chat.shape else "chat_corr"
                    z = cc.copy()
                    if z.ndim == 2:
                        z[_d + 1:] *= 3.0
                    else:
                        z[_d + 1:] *= 3.0
                    env.ctx[key] = z
                return f
            rb, rv = run_variant(arm, d, E0, "V_FCST_FUT", mk())
            dg = _diff(rb["g"], rv["g"]); da = _diff(rb["a"], rv["a"])
            db = _diff(rb["b"], rv["b"])
            same = (dg <= 1e-9)
            rows.append({
                "arm": arm, "date": s, "variant": "V_FCST_FUT",
                "meaning": "次日及以后『预测值』×3（尚未公开）",
                "expect": "计划不变", "dg_max": dg, "da_max": da, "db_max": db,
                "planned_same": same, "bill_base": float(rb["cost_total"]),
                "bill_var": float(rv["cost_total"]),
                "verdict": ("不变（合规）" if same else "计划改变（信息穿越）"),
                "ok": bool(same),
            })
            P_(f"  [{arm}] {s} V_FCST_FUT Δg={dg:>10.6f} Δa={da:>10.6f} "
               f"Δb={db:>10.6f}  账单 {rb['cost_total']:>12,.2f} → "
               f"{rv['cost_total']:>12,.2f}  "
               f"{'不变（合规）' if same else '计划改变（信息穿越）'}")

    C.write_csv_utf8_sig(OUT_DIR / "规划层因果性检验.csv",
                         FIELDS, [[r.get(k) for k in FIELDS] for r in rows])
    nab = sum(1 for r in rows if not r["ok"])
    P_("")
    P_(f"  合计 {len(rows)} 项；不合格 {nab} 项；"
       f"耗时 {time.perf_counter() - t_all:.1f}s")
    P_(f"  明细：{OUT_DIR.relative_to(C.PROJECT_DIR)}/规划层因果性检验.csv")
    return 0 if nab == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
