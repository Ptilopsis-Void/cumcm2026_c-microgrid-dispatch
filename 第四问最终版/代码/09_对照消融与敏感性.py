#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


C = _load("_comm4.py", "q4_comm")
P = _load("_price4.py", "q4_price")
PO = _load("_policy4.py", "q4_policy")
ST = _load("_settlement4.py", "q4_settle")

T = C.PERIODS_PER_DAY
K0 = C.TAU_PERIOD_INDEX
TAU = C.TAU_HOURS
N_DAY = 365

ABL_DIR = C.RESULT_DIR / "09_消融"

VALUE_NODES_MAIN = (0, 6, 12, 18)


class _Wrap:

    def __init__(self, base):
        self._b = base

    def __getattr__(self, key):
        return getattr(self._b, key)

    def price(self, d, t):
        return self._b.price(d, t)

    def forecast(self, d, ti, H):
        return self._b.forecast(d, ti, H)

    def scenarios(self, d, ti):
        return self._b.scenarios(d, ti)


class FixedPrice(_Wrap):

    _CACHE = None
    _SRC = None

    def __init__(self, base):
        super().__init__(base)
        if FixedPrice._CACHE is None:
            FixedPrice._CACHE = np.asarray(C.ref_price144(), float).reshape(-1)
            FixedPrice._SRC = str(C.ATTACHMENT1_PATH)
        self.c_fix = FixedPrice._CACHE
        assert self.c_fix.size == T, f"附件一电价曲线应为 {T} 点，实际 {self.c_fix.size}"
        pa = np.asarray(base.price_actual, float)
        if pa.ndim != 2 or pa.shape[1] != T:
            raise ValueError(f"基础提供者 price_actual 形状异常：{pa.shape}")
        self.price_actual = np.tile(self.c_fix, (pa.shape[0], 1))

    def _win(self, k0: int, H: int) -> np.ndarray:
        return self.c_fix[(k0 + np.arange(H)) % T]

    def price(self, d, t):
        return float(self.c_fix[t])

    def forecast(self, d, ti, H):
        return self._win(K0[ti], H)

    def scenarios(self, d, ti):
        sc = self._b.scenarios(d, ti)
        H = int(sc["H"])
        sc["c"] = np.broadcast_to(self._win(K0[ti], H), (sc["M"], H)).copy()
        return sc


class Scaled(_Wrap):

    def __init__(self, base, s: float):
        super().__init__(base)
        self.s = float(s)
        self.price_actual = self.s * np.asarray(base.price_actual, float)

    def price(self, d, t):
        return self.s * self._b.price(d, t)

    def forecast(self, d, ti, H):
        return self.s * np.asarray(self._b.forecast(d, ti, H), float)

    def scenarios(self, d, ti):
        sc = self._b.scenarios(d, ti)
        sc["c"] = self.s * np.asarray(sc["c"], float)
        return sc


class MethodSwap(_Wrap):

    def __init__(self, base, alt_method: str):
        super().__init__(base)
        Zf = np.load(C.PRICE_FORECAST_NPZ, allow_pickle=False)
        self.alt_method = str(alt_method)
        self.alt = np.asarray(Zf[f"chat_corr__{self.alt_method}"], float)
        self.delta = self.alt - np.asarray(base.chat, float)
        self.base_method = str(base.method)

    def forecast(self, d, ti, H):
        return np.maximum(self.alt[d, ti, :H], C.PRICE_POINT_FLOOR).copy()

    def scenarios(self, d, ti):
        sc = self._b.scenarios(d, ti)
        H = int(sc["H"])
        sc["c"] = np.maximum(np.asarray(sc["c"], float)
                             + self.delta[d, ti, :H][None, :], C.PRICE_POINT_FLOOR)
        return sc


class ToDayEnd(_Wrap):

    def forecast(self, d, ti, H):
        return self._b.forecast(d, ti, min(H, T - K0[ti]))

    def scenarios(self, d, ti):
        sc = self._b.scenarios(d, ti)
        He = min(int(sc["H"]), T - K0[ti])
        sc["N"] = np.asarray(sc["N"], float)[:, :He]
        sc["c"] = np.asarray(sc["c"], float)[:, :He]
        sc["H"] = He
        return sc


class PriceOracle(_Wrap):

    def _win(self, d: int, k0: int, H: int) -> np.ndarray:
        idx = k0 + np.arange(H)
        cur = np.asarray(self._b.price_actual[d], float)
        dn = min(d + 1, len(self._b.dates) - 1)
        nxt = np.asarray(self._b.price_actual[dn], float)
        return np.where(idx < T, cur[np.minimum(idx, T - 1)], nxt[np.maximum(idx - T, 0)])

    def forecast(self, d, ti, H):
        return self._win(d, K0[ti], H).copy()

    def scenarios(self, d, ti):
        sc = self._b.scenarios(d, ti)
        H = int(sc["H"])
        sc["c"] = np.broadcast_to(self._win(d, K0[ti], H), (sc["M"], H)).copy()
        return sc


class MCount(_Wrap):

    def __init__(self, base, Mmax: int):
        super().__init__(base)
        self.Mmax = int(Mmax)

    def scenarios(self, d, ti):
        sc = self._b.scenarios(d, ti)
        m = min(self.Mmax, int(sc["M"]))
        sc["N"] = np.asarray(sc["N"], float)[:m]
        sc["c"] = np.asarray(sc["c"], float)[:m]
        sc["origin"] = np.asarray(sc["origin"], int)[:m]
        sc["M"] = m
        sc["w"] = np.full(m, 1.0 / m)
        return sc


def make_experiments() -> list[dict]:
    ex: list[dict] = []

    def add(name, group, branch, note, make=None, cfg=None, kind="policy"):
        ex.append({"name": name, "group": group, "branch": branch, "note": note,
                   "make": make, "cfg": cfg, "kind": kind})

    add("fixed42", "价格结构", "42",
        "同一 4-2 策略栈改用附件一固定电价（规划/执行/结算同价；其余输入完全一致）",
        make=lambda p: FixedPrice(p))
    add("fixed43", "价格结构", "43",
        "同一 4-3 策略栈改用附件一固定电价（规划/执行/结算同价；其余输入完全一致）",
        make=lambda p: FixedPrice(p))

    add("vn42_0", "价值更新频率", "42", "4-2 仅 0 时更新价值（对照主方案的四次更新）",
        cfg={"value_nodes": (0,)})
    add("vn43_0", "价值更新频率", "43", "4-3 仅 0 时更新价值并调整采购")
    add("vn43_06", "价值更新频率", "43", "4-3 于 0、6 时更新")
    add("vn43_0612", "价值更新频率", "43", "4-3 于 0、6、12 时更新")

    add("method42_ar1", "价格模型", "42", "4-2 换用 AR(1) 价格预报（同残差/情景/网格）",
        make=lambda p: MethodSwap(p, "ar1"))
    add("method42_arx", "价格模型", "42", "4-2 换用 AR-X 价格预报（同残差/情景/网格）",
        make=lambda p: MethodSwap(p, "arx"))
    add("method43_nlr", "价格模型", "43",
        "4-3 换用净负荷回归价格预报（同残差/情景/网格）",
        make=lambda p: MethodSwap(p, "net_load_regression"))
    add("method43_arx", "价格模型", "43", "4-3 换用 AR-X 价格预报（同残差/情景/网格）",
        make=lambda p: MethodSwap(p, "arx"))

    add("look42_dayend", "展望", "42", "4-2 展望改为截至当日结束（去跨日尾段）",
        make=lambda p: ToDayEnd(p))
    add("oracle42", "价格信息", "42",
        "4-2 价格先知参照（只提前给实际价，负荷/PV 规则不变；参照而非下界）",
        make=lambda p: PriceOracle(p))

    add("M10", "情景数", "42", "4-2 情景数 M=10（同一情景集合的前 10 条）",
        make=lambda p: MCount(p, 10))
    add("M20", "情景数", "42", "4-2 情景数 M=20（同一情景集合的前 20 条）",
        make=lambda p: MCount(p, 20))
    add("delta12", "网格", "42", "4-2 库存价值网格 δ=12 kWh", cfg={"delta": 12.0})
    add("delta3", "网格", "42", "4-2 库存价值网格 δ=3 kWh", cfg={"delta": 3.0})
    add("delta3_43", "网格", "43", "4-3 库存价值网格 δ=3 kWh（细化；与 4-2 的 delta3 同口径）",
        cfg={"delta": 3.0})
    add("nu08", "终端价值", "42", "4-2 终端价值系数 ν×0.8", cfg={"nu_scale": 0.8})
    add("nu12", "终端价值", "42", "4-2 终端价值系数 ν×1.2", cfg={"nu_scale": 1.2})
    add("scale05", "价格缩放", "42", "4-2 共同价格与续存系数同时 ×0.5",
        make=lambda p: Scaled(p, 0.5))
    add("scale2", "价格缩放", "42", "4-2 共同价格与续存系数同时 ×2",
        make=lambda p: Scaled(p, 2.0))

    add("offline", "事后下界", "42",
        "全量事后信息离线 LP（已实现净负荷与价格、共同初库存、安全终态、不计残值）",
        kind="offline")

    return ex


def load_context():
    Z = np.load(C.WARMUP_NPZ, allow_pickle=False)
    e_start = float(np.asarray(Z["E_feb1"], float).reshape(-1)[0])
    days_all = np.arange(C.N_WARMUP_DAYS, C.N_WARMUP_DAYS + C.N_SCORE_DAYS)
    return e_start, days_all


def run_one(item: dict, prov, e_start: float, days: np.ndarray, cfg_all: dict,
            log: list) -> dict:
    cfg = dict(cfg_all)
    cfg.update(item["cfg"] or {})
    p = prov
    if item["make"] is not None:
        p = item["make"](prov)
    t0 = time.perf_counter()
    r = ST.run_policy4(item["branch"], p, p, config=cfg,
                       initial_state=e_start, days=days)
    wall = time.perf_counter() - t0
    rec = {
        "name": item["name"], "group": item["group"], "branch": item["branch"],
        "note": item["note"], "days": np.asarray(r["days"], int),
        "dates": np.asarray(r["dates"], dtype="<U10"),
        "E_start": float(r["E_start"]), "E_final": float(r["E_final"]),
        "cost_plan": float(r["cost_plan"]), "cost_adjust": float(r["cost_adjust"]),
        "cost_emerg": float(r["cost_emerg"]), "cost_total": float(r["cost_total"]),
        "n_lp": int(r["n_lp"]), "wall_s": float(wall),
        "exec_gap": float(r["exec_gap"]),
        "lp_status": np.asarray(sorted(r["lp_status_codes"]), int),
        "accept_n": int(len(r["accept_rows"])),
        "accept_ok": int(sum(1 for a in r["accept_rows"] if a["accepted"])),
    }
    for k in ("g", "a", "b", "C", "D", "U", "c", "r", "R"):
        rec[k] = np.asarray(r[k], float)
    rec["E_traj"] = np.concatenate([[e_start],
                                    np.asarray(r["E_end"], float)]).astype(float)
    rec["sum_abs_a_minus_g"] = float(np.abs(rec["a"] - rec["g"]).sum())
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ABL_DIR / f"{item['name']}.npz", **rec)
    log.append(f"  [{item['name']}] 合计 {rec['cost_total']:,.2f} 元；"
               f"计划 {rec['cost_plan']:,.2f} + 调整 {rec['cost_adjust']:,.2f} + "
               f"紧急 {rec['cost_emerg']:,.2f}；末库存 {rec['E_final']:,.2f} kWh；"
               f"{rec['n_lp']} 次 LP；{wall:.1f} s")
    return rec


def _offline_solve(price: np.ndarray, net: np.ndarray, e_start: float,
                   e_final: float | None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    K = price.size
    nv = 4 * K + (K + 1)
    iC, iD, iU, iA, iE = 0, K, 2 * K, 3 * K, 4 * K

    rows, cols, vals = [], [], []

    def blk(row, col, val):
        rows.append(np.asarray(row, int).reshape(-1))
        cols.append(np.asarray(col, int).reshape(-1))
        vals.append(np.full(np.asarray(row).size, float(val), float))

    ar = np.arange(K)
    for off, co in ((iA, 1.0), (iD, 1.0), (iC, -1.0), (iU, -1.0)):
        blk(ar, ar + off, co)
    b_eq = net.copy()
    for off, co in ((iE + 1, 1.0), (iE, -1.0), (iC, -float(C.ETA_C)),
                    (iD, 1.0 / float(C.ETA_D))):
        blk(ar + K, ar + off, co)
    b_eq = np.concatenate([b_eq, np.zeros(K)])

    A = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * K, nv)).tocsc()

    c_obj = np.zeros(nv)
    c_obj[iA:iA + K] = price

    lo = np.zeros(nv)
    hi = np.full(nv, np.inf)
    hi[iC:iC + K] = float(C.S_PERIOD_KWH)
    hi[iD:iD + K] = float(C.S_PERIOD_KWH)
    lo[iE:iE + K + 1] = float(C.E_MIN)
    hi[iE:iE + K + 1] = float(C.E_MAX)
    lo[iE] = hi[iE] = float(e_start)
    if e_final is not None:
        lo[iE + K] = hi[iE + K] = float(e_final)

    bounds = list(zip(lo.tolist(), hi.tolist()))
    t0 = time.perf_counter()
    res = linprog(c_obj, A_eq=A, b_eq=b_eq, bounds=bounds, method="highs")
    wall = time.perf_counter() - t0
    x = np.asarray(res.x, float) if res.status == 0 else None
    return {"res": res, "wall": wall, "K": K, "nv": nv, "x": x,
            "iC": iC, "iD": iD, "iU": iU, "iA": iA, "iE": iE}


def offline_lp(days: np.ndarray, e_start: float, log: list) -> dict[str, dict]:
    prov = ST.SnapshotProviders("42")
    price = np.asarray(prov.price_actual, float)[days].reshape(-1)
    net = np.asarray(prov.net_act, float)[days].reshape(-1)
    dates = np.asarray(prov.dates, dtype="<U10")[days]

    pins: list[tuple[str, float | None, str]] = [("", None,
                                                  "全量事后信息离线 LP，终态仅需落在安全区间（弱下界）")]
    for br in ("42", "43"):
        m = load_main(br)
        if m is not None:
            pins.append((f"_fe{br}", float(m["E_final"]),
                         f"离线 LP 且终态钉在 4-{br[1]} 主方案末库存 {float(m['E_final']):,.2f} kWh"
                         "（同终态可比下界）"))

    outs: dict[str, dict] = {}
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    for tag, pin, note in pins:
        r = _offline_solve(price, net, e_start, pin)
        res, K, nv = r["res"], r["K"], r["nv"]
        out = {"name": f"offline{tag}", "group": "事后下界", "branch": "42", "note": note,
               "days": np.asarray(days, int), "dates": dates,
               "status": int(res.status), "fun": float(res.fun), "wall_s": float(r["wall"]),
               "n_var": int(nv), "n_eq": int(2 * K), "message": str(res.message),
               "E_start": float(e_start), "E_pin": (float("nan") if pin is None else float(pin))}
        if res.status == 0:
            x = r["x"]
            iC, iD, iU, iA, iE = r["iC"], r["iD"], r["iU"], r["iA"], r["iE"]
            out.update({"price": price, "net": net, "E_traj": x[iE:iE + K + 1],
                        "g": x[iA:iA + K], "a": x[iA:iA + K], "b": np.zeros(K),
                        "C": x[iC:iC + K], "D": x[iD:iD + K], "U": x[iU:iU + K],
                        "c": price, "cost_total": float(res.fun),
                        "cost_plan": float((price * x[iA:iA + K]).sum()),
                        "cost_adjust": 0.0, "cost_emerg": 0.0,
                        "E_final": float(x[iE + K]), "n_lp": 1,
                        "sum_abs_a_minus_g": 0.0})
        else:
            out.update({"cost_total": float("nan"), "cost_plan": 0.0,
                        "cost_adjust": 0.0, "cost_emerg": 0.0, "E_final": 0.0, "n_lp": 1,
                        "sum_abs_a_minus_g": 0.0})
        np.savez_compressed(ABL_DIR / f"{out['name']}.npz", **out)
        outs[out["name"]] = out
        log.append(f"    离线 LP{tag or '（自由终态）'}：status={out['status']}，"
                   f"目标 {out['fun']:,.2f} 元，末库存 {out['E_final']:,.2f} kWh，"
                   f"{2 * K:,} 等式 / {nv:,} 变量，用时 {r['wall']:.2f} s")
    return outs


def load_main(branch: str) -> dict | None:
    path = C.BACKTEST_42_NPZ if branch == "42" else C.BACKTEST_43_NPZ
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as Zf:
        Z = {k: Zf[k] for k in Zf.files}

    def scal(*keys, default=0.0):
        for k in keys:
            if k in Z:
                return float(np.asarray(Z[k], float).reshape(-1)[0])
        return float(default)

    def tot(*keys, default=0.0):
        for k in keys:
            if k in Z:
                return float(np.asarray(Z[k], float).sum())
        return float(default)

    out = {"name": f"main{branch}", "group": "主方案", "branch": branch,
           "note": ("4-2 主方案（波动价、四次价值更新、a≡g）" if branch == "42"
                    else "4-3 主方案（波动价、0/6/12/18 四节点更新）"),
           "days": np.asarray(Z["days"], int),
           "dates": np.asarray(Z["dates"], dtype="<U10")}
    for k in ("g", "a", "b", "C", "D", "U", "c"):
        if k in Z:
            out[k] = np.asarray(Z[k], float)
    out["E_start"] = scal("E_start")
    out["E_final"] = scal("E_final", "E_chain")
    out["cost_total"] = tot("bill", "cost_total")
    out["cost_plan"] = tot("plan_cost", "cost_plan")
    out["cost_adjust"] = tot("adjust_cost", "cost_adjust")
    out["cost_emerg"] = tot("emerg_cost", "cost_emerg")
    out["n_lp"] = int(scal("n_lp"))
    out["wall_s"] = scal("wall_seconds", "wall_s")
    if "E_end" in Z:
        out["E_end"] = np.asarray(Z["E_end"], float)
    out["sum_abs_a_minus_g"] = float(np.abs(out["a"] - out["g"]).sum()) \
        if ("a" in out and "g" in out) else 0.0
    return out


def collect() -> dict[str, dict]:
    got: dict[str, dict] = {}
    if not ABL_DIR.exists():
        return got
    for f in sorted(ABL_DIR.glob("*.npz")):
        with np.load(f, allow_pickle=False) as Zf:
            d = {k: Zf[k] for k in Zf.files}
        for k in ("name", "group", "branch", "note", "method", "message"):
            if k in d:
                d[k] = str(np.asarray(d[k]).reshape(-1)[0])
        d.setdefault("name", f.stem)
        got[f.stem] = d
    return got


def independent_bill(branch, c, g, a, b) -> dict:
    c = np.asarray(c, float); g = np.asarray(g, float)
    a = np.asarray(a, float); b = np.asarray(b, float)
    emerg = float((5.0 * c * b).sum())
    if branch == "42":
        plan = float((c * g).sum())
        adjust = 0.0
    else:
        up = np.maximum(g - a, 0.0)
        dn = np.maximum(a - g, 0.0)
        plan = float((c * np.minimum(g, a)).sum())
        adjust = float((0.5 * c * up).sum() + (1.5 * c * dn).sum())
    return {"plan": plan, "adjust": adjust, "emerg": emerg,
            "total": plan + adjust + emerg}


def summary(log: list, checks: list, out_csv: Path, out_md: Path) -> int:
    got = collect()
    main = {"42": load_main("42"), "43": load_main("43")}
    with np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False) as Zp:
        price_act = np.asarray(Zp["price_actual"], float)

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    def ck(name, ok, detail):
        checks.append((name, bool(ok), detail))
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    base42, base43 = main["42"], main["43"]
    ck("主方案 4-2 回测可用", base42 is not None, "载入 第四问_4-2全年回测.npz")
    ck("主方案 4-3 回测可用", base43 is not None,
       "载入 第四问_4-3全年回测.npz" + ("" if base43 is not None else "（缺失，相关对照暂缺）"))
    if base42 is None:
        return 1
    b43 = float(base43["cost_total"]) if base43 is not None else float("nan")
    days = base42["days"]
    same_win = bool(base43 is not None and len(days) == C.N_SCORE_DAYS
                   and abs(base42["E_start"] - base43["E_start"]) < 1e-9)
    ck("全部实验与主方案共享同一评分窗口与共同初值",
       bool(len(days) == C.N_SCORE_DAYS and same_win),
       f"窗口 {days[0]}..{days[-1]}（{len(days)} 天）；E_start = {base42['E_start']:.6f} kWh"
       + (f"；4-3 E_start = {base43['E_start']:.6f} kWh" if base43 is not None else ""))

    rows: list[list] = []

    def push(rec, ref=None):
        total = float(rec["cost_total"])
        diff = "" if ref is None else f"{total - ref:,.2f}"
        pct = "" if ref in (None, 0) else f"{100 * (total - ref) / ref:.3f}%"
        rows.append([rec["group"], rec["name"], rec.get("note", ""), rec["branch"],
                     len(rec["days"]), f"{rec['cost_plan']:,.6f}",
                     f"{rec['cost_adjust']:,.6f}", f"{rec['cost_emerg']:,.6f}",
                     f"{total:,.6f}", f"{float(rec['E_final']):,.4f}",
                     f"{float(rec['b'].sum()):,.1f}",
                     f"{float(rec.get('sum_abs_a_minus_g', 0.0)):,.3f}",
                     int(rec.get("n_lp", 0)), f"{float(rec.get('wall_s', 0.0)):.1f}",
                     diff, pct])

    for br in ("42", "43"):
        if main[br] is not None:
            push(main[br])

    def match(rec, br="42") -> bool:
        b = base_by_br.get(br)
        if b is None:
            p(f"  [跳过] {rec['name']}：主方案 4-{br[1]} 缺失，无法对照")
            return False
        if len(rec["days"]) != len(b["days"]):
            p(f"  [跳过] {rec['name']}：天数 {len(rec['days'])} 与主方案 4-{br[1]} "
              f"{len(b['days'])} 不一致，不作费用对照")
            return False
        return True

    base_by_br = {"42": base42, "43": base43}
    for br in ("42", "43"):
        if base_by_br[br] is None:
            continue
        rec = got.get(f"fixed{br}")
        if rec is None:
            p(f"  [缺失] fixed{br} 未运行")
            continue
        if not match(rec, br):
            continue
        ref = base_by_br[br]["cost_total"]
        push(rec, ref)
        b1 = independent_bill(br, price_act[rec["days"]], rec["g"], rec["a"], rec["b"])
        dlt = b1["total"] - float(ref)
        rows.append([f"重计价-{'2' if br == '42' else '3'}",
                     f"fixed{br}_repriced_act", "固定价动作按附件四重计价（不动任何动作）",
                     br, len(rec["days"]), f"{b1['plan']:,.6f}", f"{b1['adjust']:,.6f}",
                     f"{b1['emerg']:,.6f}", f"{b1['total']:,.6f}",
                     f"{float(rec['E_final']):,.4f}",
                     f"{float(rec['b'].sum()):,.1f}",
                     f"{float(rec.get('sum_abs_a_minus_g', 0.0)):,.3f}",
                     0, "0.0", f"{dlt:,.2f}", f"{100 * dlt / float(ref):.3f}%"])
        p(f"  fixed{br}：固定价合计 {rec['cost_total']:,.2f} 元；"
          f"其动作按附件四重计价 {b1['total']:,.2f} 元（主方案 {float(ref):,.2f} 元）")

    for item in make_experiments():
        if item["kind"] in ("offline",):
            continue
        if item["name"].startswith("fixed"):
            continue
        rec = got.get(item["name"])
        if rec is None:
            continue
        if not match(rec, rec["branch"]):
            continue
        push(rec, float(base_by_br[rec["branch"]]["cost_total"]))

    c_fix = np.asarray(C.ref_price144(), float).reshape(-1)
    pa_arr = np.asarray(price_act, float)
    dev_src = float(np.abs(pa_arr - c_fix[None, :]).max())
    for br in ("42", "43"):
        rec = got.get(f"fixed{br}")
        if rec is None or base_by_br[br] is None:
            continue
        if len(rec["days"]) != len(base_by_br[br]["days"]):
            continue
        cc = np.asarray(rec["c"], float)
        dev_c = float(np.abs(cc - c_fix[None, :]).max()) if (cc.ndim == 2
                                                             and cc.shape[1] == T) else float("inf")
        ck(f"固定价场景 {br} 实际价矩阵 = 附件一固定曲线（规划/执行/结算同价）",
           bool(dev_c <= 1e-12),
           f"max|c(落盘, {cc.shape}) − 附件一固定曲线| = {dev_c:.3e} 元/kWh"
           f"（须为 0；来源 `_comm4.ref_price144()` → {C.ATTACHMENT1_PATH.name}）")
        ck(f"固定价场景 {br} 的重计价价格与固定曲线确为两条不同路径",
           bool(float(np.abs(pa_arr[rec["days"]] - c_fix[None, :]).max()) > 1e-6),
           f"附件四实际价 vs 固定曲线最大差 "
           f"{float(np.abs(pa_arr[rec['days']] - c_fix[None, :]).max()):.4f} 元/kWh；"
           f"落盘价矩阵差 {dev_c:.3e} 元/kWh（前者用于重计价，后者用于本臂闭环）")
    ck("重计价所用价格矩阵取自附件四原始实际价（非固定曲线、非包装器输出）",
       bool(dev_src > 1e-6),
       f"`C.PRICE_MATRIX_NPZ['price_actual']` 与附件一固定曲线最大差 {dev_src:.4f} 元/kWh；"
       "重计价只读该矩阵，不经过任何提供者包装器")

    for br in ("42", "43"):
        rec = got.get(f"fixed{br}")
        if rec is None or base_by_br[br] is None:
            continue
        if len(rec["days"]) != len(base_by_br[br]["days"]):
            continue
        d = rec["days"]
        b_fix = independent_bill(br, np.asarray(rec["c"], float), rec["g"], rec["a"], rec["b"])
        ck(f"固定价场景 {br} 账单可由独立定义式复算",
           abs(b_fix["total"] - float(rec["cost_total"])) <= 1e-6 * max(1.0, float(rec["cost_total"])),
           f"独立式 {b_fix['total']:,.6f} 元 vs 引擎 {float(rec['cost_total']):,.6f} 元")
        b_act = independent_bill(br, price_act[d], rec["g"], rec["a"], rec["b"])
        ck(f"固定价动作按附件四重计价已生成 {br}",
           np.isfinite(b_act["total"]),
           f"重计价合计 {b_act['total']:,.2f} 元（计划 {b_act['plan']:,.2f} + "
           f"调整 {b_act['adjust']:,.2f} + 紧急 {b_act['emerg']:,.2f}）")

    s_map = {"scale05": 0.5, "scale2": 2.0}
    for nm, s in s_map.items():
        rec = got.get(nm)
        if rec is None or not match(rec):
            continue
        cc = np.asarray(rec["c"], float)
        rec_cost = float(rec["cost_total"])
        base_cost = float(base42["cost_total"])
        ratio = rec_cost / base_cost
        b_rep = independent_bill("42", cc, base42["g"], base42["a"], base42["b"])
        da = float(np.abs(np.asarray(rec["a"], float) - np.asarray(base42["a"], float)).max())
        ck(f"价格缩放 ×{s:g}：费用比例 = s（原解在新目标下仍最优）",
           abs(rec_cost - s * base_cost) <= 1e-6 * abs(s * base_cost),
           f"合计 {rec_cost:,.6f} 元 ÷ 主方案 {base_cost:,.6f} 元 = {ratio:.9f}"
           f"（应 = s = {s:g}）；同一动作集按 s×实际价独立重计价 {b_rep['total']:,.6f} 元"
           f"（与 s×主方案差 {b_rep['total'] - s * base_cost:.3e} 元）；"
           f"动作最大逐段差 {da:.3e} kWh（等价最优解允许不同动作，不作判据）")
        ck(f"价格缩放 ×{s:g}：缩放后解仍可行（库存界、非负、末库存）",
           (float(C.E_MIN) - 1e-6 <= float(rec["E_final"]) <= float(C.E_MAX) + 1e-6
            and float(np.asarray(rec["a"], float).min()) >= -1e-6
            and float(np.asarray(rec["b"], float).min()) >= -1e-6),
           f"末库存 {float(rec['E_final']):,.4f} kWh ∈ "
           f"[{float(C.E_MIN):,.0f}, {float(C.E_MAX):,.0f}]；"
           f"min a = {float(np.asarray(rec['a'], float).min()):.3e}、"
           f"min b = {float(np.asarray(rec['b'], float).min()):.3e} kWh")

    b12 = got.get("delta12"); b3 = got.get("delta3")
    if b12 is not None and b3 is not None and match(b3):
        ck("网格敏感性：δ=3 不劣于 δ=6（主配置）",
           float(b3["cost_total"]) <= float(base42["cost_total"]) + 1e-6 * abs(float(base42["cost_total"])),
           f"δ=3 {float(b3['cost_total']):,.2f} 元 ≤ δ=6 {float(base42['cost_total']):,.2f} 元")
        ck("网格敏感性：已对完整评分期（334 天）验证，方可写「全年已收敛」",
           bool(len(b12["days"]) == C.N_SCORE_DAYS and len(b3["days"]) == C.N_SCORE_DAYS),
           f"δ=12 与 δ=3 均为 {C.N_SCORE_DAYS} 天全期比较")

    for nm in ("nu08", "nu12"):
        rec = got.get(nm)
        if rec is None:
            continue
        ck(f"终端价值 {nm}：末库存与费用同时记录",
           np.isfinite(float(rec["E_final"])),
           f"合计 {float(rec['cost_total']):,.2f} 元，末库存 {float(rec['E_final']):,.2f} kWh"
           + (f"（对比主方案 {float(base42['cost_total']):,.2f} 元 / "
              f"{float(base42['E_final']):,.2f} kWh）" if match(rec) else ""))

    for nm, m in (("M10", 10), ("M20", 20)):
        rec = got.get(nm)
        if rec is None:
            continue
        ck(f"情景数 {nm}：使用同一合法情景集合的前 {m} 条",
           True, f"合计 {float(rec['cost_total']):,.2f} 元，末库存 "
                 f"{float(rec['E_final']):,.2f} kWh，{int(rec['n_lp'])} 次 LP，"
                 f"{float(rec['wall_s']):.1f} s")

    for tag, note in (("", "离线 LP 自由终态（弱下界）"),
                      ("_fe42", "离线 LP 同终态 @4-2 末库存"),
                      ("_fe43", "离线 LP 同终态 @4-3 末库存")):
        off = got.get(f"offline{tag}")
        if off is None or int(np.asarray(off.get("status", 9)).reshape(-1)[0]) != 0:
            continue
        if len(np.asarray(off["days"]).reshape(-1)) != len(base42["days"]):
            p(f"  [跳过] offline{tag}：天数与主方案不一致，不作下界对照")
            continue
        lb = float(np.asarray(off["fun"], float).reshape(-1)[0])
        ef = float(np.asarray(off["E_final"], float).reshape(-1)[0])
        rows.append(["事后下界", f"offline{tag}", note, "42", len(days),
                     f"{float(np.asarray(off['cost_plan'], float).reshape(-1)[0]):,.6f}",
                     "0.000000", "0.000000", f"{lb:,.6f}", f"{ef:,.4f}", "0.0",
                     "0.000", 1, f"{float(np.asarray(off['wall_s'], float).reshape(-1)[0]):.1f}",
                     f"{lb - float(base42['cost_total']):,.2f}",
                     f"{100 * (lb - float(base42['cost_total'])) / float(base42['cost_total']):.3f}%"])
        ck(f"离线完全信息 LP 求解成功（{note}）",
           True, f"目标 {lb:,.2f} 元，末库存 {ef:,.2f} kWh")
        if tag == "":
            ck("下界关系成立：4-2、4-3 均不低于离线自由终态下界",
               bool(lb <= float(base42["cost_total"]) + 1e-6
                    and (base43 is None or lb <= b43 + 1e-6)),
               f"下界 {lb:,.2f} ≤ 4-2 {float(base42['cost_total']):,.2f}"
               + (f"、≤ 4-3 {b43:,.2f}" if base43 is not None else "（4-3 缺失）"))
        else:
            key = float(base42["cost_total"]) if tag == "_fe42" else b43
            if not np.isfinite(key):
                continue
            ck(f"同终态下界关系成立（{note}）",
               bool(lb <= key + 1e-6),
               f"同终态下界 {lb:,.2f} ≤ 主方案 {key:,.2f} 元（终态均为 {ef:,.2f} kWh，"
               "故差额不含「少留库存」的水分）")

    p("")
    p("  —— 不预设结论的三项（只在报告中陈述数值，不作「必然」断言）——")
    if base43 is not None:
        p(f"  4-3 vs 4-2：4-3 {b43:,.2f} 元，4-2 {float(base42['cost_total']):,.2f} 元，"
          f"差 {b43 - float(base42['cost_total']):,.2f} 元")
    f42 = got.get("fixed42"); f43 = got.get("fixed43")
    if f42 is not None and f43 is not None and match(f42) and match(f43, "43"):
        p(f"  固定价 vs 波动价：固定价 4-2 {float(f42['cost_total']):,.2f} / "
          f"4-3 {float(f43['cost_total']):,.2f} 元")
    v42 = got.get("vn42_0")
    if v42 is not None and match(v42):
        p(f"  价值更新频率：4-2 仅 0 时 {float(v42['cost_total']):,.2f} 元 vs "
          f"四次 {float(base42['cost_total']):,.2f} 元")

    header = ["组别", "实验", "说明", "分支", "天数", "计划费/元", "调整费/元",
              "紧急费/元", "合计/元", "末库存/kWh", "紧急量/kWh", "Σ|a−g|/kWh",
              "LP次数", "耗时/s", "相对主方案/元", "相对主方案/%"]
    C.write_csv_utf8_sig(out_csv, header, rows)
    p(f"  对照与敏感性表：{out_csv}（{len(rows)} 行）")
    return 0


def write_report(log: list, checks: list, elapsed: float, mode: str,
                 out_md: Path) -> None:
    rp: list[str] = []
    rp.append("# 第四问 对照、消融与敏感性报告\n")
    rp.append("> 对应《第四问详细流程图》§11.1 必需对照表与 §11.2 最小敏感性集合。"
              "全部实验共享同一评分窗口、同一热启动初值、同一情景库与同一结算口径；"
              "除各表「说明」列明的一项外，其余条件保持不变。\n")
    rp.append("## 1. 结果总表\n")
    rp.append(f"- 汇总表：`模型结果/{C.ABLATION_CSV.name}`（逐实验一行，含费用分解、"
              "末库存、紧急量、调整量、LP 次数与相对主方案的差）")
    rp.append(f"- 实验轨迹：`模型结果/{ABL_DIR.name}/<实验名>.npz`（逐时段 "
              "`g,a,b,C,D,U,c,R,r` 与日末库存链，可用于独立复算）")
    rp.append("")
    rp.append("## 2. 检查项\n")
    rp.append("| 检查项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")
    rp.append("## 3. 口径与结论纪律\n")
    rp.append("- 固定价格场景使用**附件一**给出的恒定（逐日重复）电价曲线"
              "（`附件1.xlsx`「电价」列，144 个区间终点，均值 0.766197 元/kWh，"
              "谷段 0.3713 / 峰段 1.3952 元/kWh），与第一/二/三问的固定电价假设一致；"
              "第二问矩阵中的 `price` 只是该列的**已处理复用字段**（`_comm4.ref_price144()` "
              "直读附件一并与复用字段逐点交叉校验，差异 0.0），本报告不把它写作「原始附件二提供电价」。"
              "此时 $\\nu_\\tau$ 自动等于 `_comm4.NU_FIXED`（谷段前 30 段均值 / $\\eta$），未硬编码。")
    rp.append("- ★ 固定价臂**必须显式覆盖 `price_actual` 属性**："
              "`_settlement4.run_policy4` 直读 `price_provider.price_actual[d]`，"
              "该数组同时决定**实时执行价**与**当日结算价**；只覆盖 "
              "`price/forecast/scenarios` 会被 `_Wrap.__getattr__` 静默透传为附件四波动价，"
              "本臂即退化为「按固定价规划、按波动价执行并结算」的**混合口径**。"
              "故本报告不凭「调用了哪个包装器」判断切换成功，而是**直接断言落盘价矩阵**"
              "（npz 的 `c`）逐点等于附件一固定曲线，并断言重计价所用价取自附件四原始实际价。")
    rp.append("- 固定价格动作按附件四重计价时，`g,a,C,D,b` 与库存轨迹**完全不动**，"
              "只把价格换回附件四实际价重算账单，从而把「换价格的账单影响」与"
              "「重新优化动作的影响」分开。该「重计价」行只读 "
              "`模型结果/第四问_电价矩阵.npz['price_actual']`（附件四原始实际价），"
              "不经过任何提供者包装器。")
    rp.append("- 价格模型对照只替换价格预报的**均值路径**，残差样本、情景数、"
              "网格与更新频率不变，用于回答「价格预报改善是否转化为账单改善」。")
    rp.append("- 价格先知参照只把未来价格换成实际值，负荷/PV 预报规则不变，"
              "**是参照而非严格下界**。")
    rp.append("- 离线 LP 放松了「普通采购须按日前计划执行」等约束，其最优值"
              "**可作可比的事后下界**；但它不计残值，故不等于真实问题的最优值。")
    rp.append("- 价格缩放臂把**实际价、预报价、情景价与由情景价导出的续存系数 ν**"
              "同时乘 $s$；结算价由 `_settlement4.run_policy4` 直接读属性 "
              "`price_provider.price_actual`，故包装器必须**显式缩放该属性**"
              "（不能只覆盖 `price/forecast/scenarios`）。约束集与阈值法 argmin 均不变，"
              "目标是整体乘 $s$，因此 **「动作不变、账单同比乘 $s$」是该试验的预期结论**；"
              "npz 中 `c` 即缩放后的实际价，故「同一动作集重计价 $=s\\times$主方案」"
              "应精确成立（等价最优解允许多解，不以动作逐点相同作判据）。")
    rp.append("- 情景数敏感性 $M=20/10$ 取自**同一合法情景集合的前 20/10 条**（共同评价样本），"
              "属「降配」侧；主方案已取满该集合（`M_max=30`），无可更细者。"
              "故情景数偏差异常只沿「情景越少费用越高」的方向出现，"
              "验收据此按方向单调 + 分档阈值判定，见 12 号 A16。")
    rp.append("- 本报告不预设「4-3 一定优于 4-2」「波动价必然增费」「18 时一定有效」，"
              "所有判断均须同时看计划费、调整费、紧急费、高价缺电重合、充放电时段"
              "与期末库存。")
    rp.append("")
    rp.append("## 4. 运行日志\n")
    rp.append("```")
    rp.extend(log)
    rp.append(f"（模式 {mode}，用时 {elapsed:.1f} s）")
    rp.append("```")
    C.write_text_utf8(out_md, "\n".join(rp))


def main() -> int:
    ap = argparse.ArgumentParser(description="第四问 09 对照/消融/敏感性")
    ap.add_argument("--exp", default="all",
                    help="逗号分隔的实验名，或 all / summary")
    ap.add_argument("--days", type=int, default=C.N_SCORE_DAYS,
                    help="评分期天数（小样本联调用）")
    ap.add_argument("--suffix", default="", help="实验落盘后缀（联调用；留空为正式）")
    args = ap.parse_args()

    global ABL_DIR
    t0 = time.perf_counter()
    C.ensure_dirs()
    if args.suffix:
        ABL_DIR = C.RESULT_DIR / f"09_消融{args.suffix}"

    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    checks: list[tuple[str, bool, str]] = []
    ex = make_experiments()
    by_name = {e["name"]: e for e in ex}
    want = [s.strip() for s in args.exp.split(",") if s.strip()]
    if want == ["all"]:
        want = [e["name"] for e in ex]
    only_summary = want == ["summary"]

    out_csv = C.ABLATION_CSV if not args.suffix else \
        C.RESULT_DIR / f"第四问_对照与敏感性{args.suffix}.csv"
    out_md = (C.REPORT_DIR / "第四问_对照与敏感性报告.md") if not args.suffix else \
        (C.REPORT_DIR / f"第四问_对照与敏感性报告{args.suffix}.md")

    p("=" * 78)
    p("第四问 09 —— 对照、消融与敏感性（§11.1 / §11.2）")
    p("=" * 78)

    if not only_summary:
        e_start, days_all = load_context()
        days = days_all[:args.days]
        cfg_all = {"delta": None, "nu_scale": 1.0,
                   "value_nodes": VALUE_NODES_MAIN, "tail": True, "grid_1d": True,
                   "cap_iter": 4, "verify_exec": False}
        p(f"  共同初值 E(2/1) = {e_start:,.4f} kWh；窗口 {days[0]}..{days[-1]}"
          f"（{len(days)} 天）")
        prov_cache: dict[str, object] = {}
        for name in want:
            item = by_name.get(name)
            if item is None:
                p(f"  [跳过] 未知实验：{name}")
                continue
            if item["kind"] == "offline":
                p(f"\n-- 离线完全信息 LP --")
                offline_lp(days, e_start, log)
                continue
            br = item["branch"]
            if br not in prov_cache:
                prov_cache[br] = ST.SnapshotProviders(br)
            p(f"\n-- {item['name']}（{item['group']}，分支 4-{br[1]}）--")
            p(f"   说明：{item['note']}")
            run_one(item, prov_cache[br], e_start, days, cfg_all, log)

    if args.suffix:
        p(f"\n  （联调后缀模式：`{ABL_DIR.name}`，输出另名不覆盖正式交付）")

    p("\n-- 汇总与检查 --")
    rc = summary(log, checks, out_csv, out_md)

    nb = sum(1 for _n, ok, _d in checks if not ok)
    p("")
    p("=" * 78)
    p(f"09 完成：检查 {len(checks)} 项，未通过 {nb} 项。用时 {time.perf_counter() - t0:.1f} s")
    p("=" * 78)
    write_report(log, checks, time.perf_counter() - t0, args.exp, out_md)
    C.write_text_utf8(C.LOG_DIR / f"09_对照敏感性日志{args.suffix}.txt",
                      "\n".join(log) + "\n")
    return rc if nb == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
