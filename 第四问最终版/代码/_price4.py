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

import numpy as np

T = C.PERIODS_PER_DAY
N_DAY = 365
N_FLAT = N_DAY * T
K0 = C.TAU_PERIOD_INDEX
N_TAU = len(C.TAU_HOURS)

METHODS = ("prev_day", "mean7", "ar1", "net_load_regression", "arx")
METHOD_LABEL = {
    "prev_day": "昨日水平",
    "mean7": "七日均值",
    "ar1": "AR(1)",
    "net_load_regression": "净负荷回归",
    "arx": "AR-X",
}
METHOD_SPEC = {
    "prev_day": (False, False, False),
    "mean7": (False, False, False),
    "ar1": (True, True, False),
    "net_load_regression": (True, False, True),
    "arx": (True, True, True),
}
MIN_SAMPLES = {"ar1": 8, "net_load_regression": 8, "arx": 12}
ORIGIN_MIN_DAY = int(max(MIN_SAMPLES.values()))
PRICE_LEVEL_PRIOR = 0.50


def to_flat(a2d) -> np.ndarray:
    return np.asarray(a2d, float).reshape(-1)


def daily_level(price_act) -> np.ndarray:
    return np.asarray(price_act, float).mean(axis=1)


def intraday_shape(price_act, days) -> np.ndarray:
    a = np.asarray(price_act, float)
    idx = np.asarray(list(days), int)
    if idx.size == 0:
        return np.ones(T)
    lev = a[idx].mean(axis=1)
    lev = np.where(lev > 1e-12, lev, 1.0)
    return (a[idx] / lev[:, None]).mean(axis=0)


def net_load_feature(N_act, Nhat_q2, k0: int) -> np.ndarray:
    Na = np.asarray(N_act, float)
    Nh = np.asarray(Nhat_q2, float)
    if k0 <= 0:
        return Nh.sum(axis=1)
    return np.concatenate([Na[:, :k0].sum(axis=1, keepdims=True),
                           Nh[:, k0:].sum(axis=1, keepdims=True)], axis=1).sum(axis=1)


def net_load_feature_all(N_act, Nhat_q2) -> np.ndarray:
    return np.stack([net_load_feature(N_act, Nhat_q2, k0) for k0 in K0], axis=1)


def _ols(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.shape[0] == 0:
        return None, 0
    beta, _res, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(beta)):
        return None, int(rank)
    return beta, int(rank)


def fit_level(method: str, lev_hist: np.ndarray, nhat_hist: np.ndarray,
              nhat_now: float) -> dict:
    lev_hist = np.asarray(lev_hist, float)
    nhat_hist = np.asarray(nhat_hist, float)
    n = lev_hist.size
    prev = float(lev_hist[-1]) if n >= 1 else None
    out = {"method": method, "n_samples": int(n), "fallback": None,
           "coef": {}, "rank": None}

    if method == "prev_day":
        if prev is None:
            out["fallback"] = "no_history→固定正价先验"
            return {**out, "level": PRICE_LEVEL_PRIOR}
        return {**out, "level": prev}

    if method == "mean7":
        if n == 0:
            out["fallback"] = "no_history→固定正价先验"
            return {**out, "level": PRICE_LEVEL_PRIOR}
        k = min(7, n)
        return {**out, "level": float(lev_hist[-k:].mean())}

    has_int, has_ar, has_nl = METHOD_SPEC[method]
    need = MIN_SAMPLES.get(method, 8)
    if n < need or prev is None:
        out["fallback"] = f"样本不足({n}<{need})→昨日水平"
        lv = prev if prev is not None else PRICE_LEVEL_PRIOR
        return {**out, "level": float(lv)}

    lag = np.concatenate([[np.nan], lev_hist[:-1]])
    ok = np.isfinite(lag)
    cols, names = [], []
    if has_int:
        cols.append(np.ones(n)); names.append("intercept")
    if has_ar:
        cols.append(lag); names.append("rho_c")
    if has_nl:
        cols.append(nhat_hist / 1e5); names.append("b_c")
    X = np.stack(cols, axis=1)[ok]
    y = lev_hist[ok]
    beta, rank = _ols(X, y)
    out["rank"] = rank
    if beta is None or rank < X.shape[1]:
        out["fallback"] = f"秩亏(rank={rank}<{X.shape[1]})→昨日水平"
        return {**out, "level": float(prev)}

    a_c, rho_c, b_c = 0.0, 0.0, 0.0
    for nm, v in zip(names, beta):
        if nm == "intercept":
            a_c = float(v)
        elif nm == "rho_c":
            rho_c = float(v)
        else:
            b_c = float(v)
    lo, hi = C.AR_RHO_BOUNDS
    rho_c = float(min(max(rho_c, lo), hi))
    coef = {"a_c": a_c, "rho_c": rho_c, "b_c": b_c}
    return {**out, "coef": coef,
            "level": predict_level(coef, prev, float(nhat_now))}


def predict_level(coef: dict, prev_level: float, nhat_now: float) -> float:
    return (float(coef["a_c"]) + float(coef["rho_c"]) * float(prev_level)
            + float(coef["b_c"]) * float(nhat_now) / 1e5)


def window_index(d: int, k0: int, h: int) -> int:
    return d * T + k0 + h


def horizon_len(d: int, k0: int, n_flat: int = N_FLAT) -> int:
    return int(max(0, min(T, n_flat - (d * T + k0))))


def estimate_rho_p(price_act, chat_base0, days) -> float:
    idx = np.asarray(list(days), int)
    if idx.size == 0:
        return 0.0
    e = np.asarray(price_act, float)[idx] - np.asarray(chat_base0, float)[idx]
    x = e[:, :-1].ravel()
    y = e[:, 1:].ravel()
    if x.size < 30 or np.std(x) < 1e-12:
        return 0.0
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    lo, hi = C.AR_RHO_BOUNDS
    return float(min(max(beta, lo), hi))


def replay_forecasts(price_act, nhat_snap, method: str,
                     window: int = None, bias_correct: bool = True,
                     verbose_log=None) -> dict:
    window = int(window or C.PRICE_HISTORY_DAYS)
    price = np.asarray(price_act, float)
    nhat = np.asarray(nhat_snap, float)
    lev_act = daily_level(price)

    chat_base = np.zeros((N_DAY, N_TAU, T))
    chat_corr = np.zeros((N_DAY, N_TAU, T))
    level = np.zeros((N_DAY, N_TAU))
    Hv = np.zeros((N_DAY, N_TAU), dtype=int)
    rho_p = np.zeros(N_DAY)
    diag: list[dict] = []
    shape_all = np.zeros((N_DAY, T))

    for d in range(N_DAY):
        lo = max(0, d - window)
        hist = np.arange(lo, d)
        shape = intraday_shape(price, hist)
        shape_all[d] = shape
        fit = fit_level(method, lev_act[hist] if hist.size else np.array([]),
                        nhat[hist, 0] if hist.size else np.array([]),
                        float(nhat[d, 0]))
        ell = max(float(fit["level"]), C.PRICE_LEVEL_FLOOR)
        level[d, 0] = ell
        k0 = 0
        H = horizon_len(d, k0)
        Hv[d, 0] = H
        if H > 0:
            idx = [window_index(d, k0, h) for h in range(H)]
            p = np.array([idx[h] % T for h in range(H)])
            chat_base[d, 0, :H] = ell * shape[p]
        diag.append({"day": d, "tau": 0, "method": method, "H": H,
                     "n_samples": fit["n_samples"], "level": ell,
                     "fallback": fit["fallback"], "coef": fit["coef"]})

    for d in range(N_DAY):
        lo = max(0, d - window)
        rho_p[d] = estimate_rho_p(price, chat_base[:, 0, :], np.arange(lo, d)) \
            if d > 1 else 0.0

    for d in range(N_DAY):
        hist = np.arange(max(0, d - window), d)
        shape = shape_all[d]
        levs = np.zeros(N_TAU)
        for ti, k0 in enumerate(K0):
            fit = fit_level(method, lev_act[hist] if hist.size else np.array([]),
                            nhat[hist, ti] if hist.size else np.array([]),
                            float(nhat[d, ti]))
            ell = max(float(fit["level"]), C.PRICE_LEVEL_FLOOR)
            levs[ti] = ell
            level[d, ti] = ell
            H = horizon_len(d, k0)
            Hv[d, ti] = H
            if H <= 0:
                continue
            g = np.array([window_index(d, k0, h) for h in range(H)])
            p = g % T
            day_off = (k0 + np.arange(H)) // T
            base = np.empty(H)
            same = day_off == 0
            base[same] = ell * shape[p[same]]
            if (~same).any():
                nxt_nhat = nhat[d + 1, 0] if d + 1 < N_DAY else nhat[d, 0]
                if method in ("ar1", "arx"):
                    spec = METHOD_SPEC[method]
                    coef_now = fit.get("coef") or {}
                    if coef_now and spec[1]:
                        ell_next = max(predict_level(coef_now, ell, nxt_nhat),
                                       C.PRICE_LEVEL_FLOOR)
                    else:
                        ell_next = ell
                else:
                    ell_next = ell
                base[~same] = ell_next * shape[p[~same]]
            chat_base[d, ti, :H] = base

            if not bias_correct or ti == 0 or H == 0:
                chat_corr[d, ti, :H] = base
            else:
                c_last = float(price[d, k0 - 1])
                tilde_last = ell * shape[k0 - 1]
                err_last = c_last - tilde_last
                rp = rho_p[d]
                j = np.arange(1, H + 1)
                chat_corr[d, ti, :H] = np.maximum(
                    C.PRICE_POINT_FLOOR, base + (rp ** j) * err_last)

            diag.append({"day": d, "tau": int(C.TAU_HOURS[ti]), "method": method,
                         "H": H, "n_samples": fit["n_samples"], "level": ell,
                         "fallback": fit["fallback"], "coef": fit["coef"]})

    return {"chat_base": chat_base, "chat_corr": chat_corr, "level": level,
            "H": Hv, "rho_p": rho_p, "diag": diag, "shape": shape_all,
            "method": method, "window": window}


def score_forecasts(price_act, chat_corr, Hv, days) -> dict:
    price = np.asarray(price_act, float)
    chat = np.asarray(chat_corr, float)
    err_all = []
    per_tau = {}
    for ti in range(N_TAU):
        e = []
        for d in days:
            H = int(Hv[d, ti])
            if H <= 0:
                continue
            g = np.array([window_index(d, K0[ti], h) for h in range(H)])
            ok = g < N_FLAT
            e.append(chat[d, ti, :H][ok] - price.reshape(-1)[g[ok]])
        if e:
            e = np.concatenate(e)
        else:
            e = np.zeros(0)
        err_all.append(e)
        per_tau[C.TAU_HOURS[ti]] = {
            "mae": float(np.mean(np.abs(e))) if e.size else float("nan"),
            "rmse": float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
            "bias": float(np.mean(e)) if e.size else float("nan"),
            "n": int(e.size),
        }
    e = np.concatenate([a for a in err_all if a.size]) if any(
        a.size for a in err_all) else np.zeros(0)
    if e.size:
        ed = e - e.mean()
        rmse_dt = float(np.sqrt(np.mean(ed ** 2)))
        mae_dt = float(np.mean(np.abs(ed)))
    else:
        rmse_dt = mae_dt = float("nan")
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)), "rmse_dtrend": rmse_dt, "mae_dtrend": mae_dt,
            "n": int(e.size), "per_tau": per_tau}


def legal_origins(d: int, max_m: int, window: int) -> list[int]:
    lo = max(ORIGIN_MIN_DAY, d - window)
    if d - 1 < lo:
        return []
    return list(range(d - 1, lo - 1, -1))[:max_m]


def build_node_scenarios(d, ti, ctx, branch,
                         max_m: int = None, window: int = None) -> dict:
    max_m = int(max_m or C.M_SCENARIOS)
    window = int(window or 30)
    k0 = K0[ti]
    H = horizon_len(d, k0)

    price = np.asarray(ctx["price_act"], float).reshape(-1)
    chat = np.asarray(ctx["chat_corr"], float)
    L_act = np.asarray(ctx["L_act"], float).reshape(-1)
    Lhat = np.asarray(ctx["Lhat"], float).reshape(-1)
    Vhat_q2 = np.asarray(ctx["Vhat_q2"], float).reshape(-1)

    out = {"day": d, "tau": int(C.TAU_HOURS[ti]), "k0": k0, "H": H,
           "branch": branch, "fallbacks": []}
    if H <= 0:
        out.update({"M": 0, "origin": np.zeros(0, int), "price_scen": np.zeros((0, 0)),
                    "N_scen": np.zeros((0, 0)), "price_hat": np.zeros(0)})
        out["fallbacks"].append("H=0（数据终点外）")
        return out

    g = np.array([window_index(d, k0, h) for h in range(H)])
    p = g % T
    day_off = (k0 + np.arange(H)) // T

    price_hat = chat[d, ti, :H].copy()

    if branch == "42":
        Vhat_now = Vhat_q2[g]
        V_act_win = np.asarray(ctx["V_act"], float).reshape(-1)[g]
    else:
        Vatt = np.asarray(ctx["V_att3_win"], float)
        Vhat_now = Vatt[d, ti, :H].copy()
        pv_h = np.asarray(ctx["pv_act_hour"], float)
        V_act_win = pv_h[g // T, (g % T) // 6] * C.DELTA_HOURS
    Lhat_now = Lhat[g]

    origins = legal_origins(d, max_m, window)
    legal = []
    for i in origins:
        gi = np.array([window_index(i, k0, h) for h in range(H)])
        if gi.max() >= N_FLAT:
            continue
        legal.append(i)
    out["legal_count"] = len(legal)

    if not legal:
        out["fallbacks"].append("无已闭合历史窗口→零残差暖启动")
        M = 1
        price_scen = price_hat[None, :]
        L_scen = Lhat_now[None, :]
        V_scen = Vhat_now[None, :]
        N_scen = (L_scen - V_scen) * 1.0
        out.update({"M": M, "origin": np.full(1, -1),
                    "price_scen": price_scen, "N_scen": N_scen,
                    "L_scen": L_scen, "V_scen": V_scen, "price_hat": price_hat,
                    "weights": np.ones(1)})
        return out

    M = len(legal)
    origin = np.array(legal, int)
    price_scen = np.empty((M, H))
    L_scen = np.empty((M, H))
    V_scen = np.empty((M, H))
    for wi, i in enumerate(legal):
        gi = np.array([window_index(i, k0, h) for h in range(H)])
        price_scen[wi] = np.maximum(
            C.PRICE_POINT_FLOOR,
            price_hat + (price[gi] - chat[i, ti, :H]))
        eL = L_act[gi] - Lhat[gi]
        L_scen[wi] = np.maximum(0.0, Lhat_now + eL)
        if branch == "42":
            Vhat_hist = Vhat_q2[gi]
            Vact_hist = np.asarray(ctx["V_act"], float).reshape(-1)[gi]
        else:
            Vatt = np.asarray(ctx["V_att3_win"], float)
            pv_h = np.asarray(ctx["pv_act_hour"], float)
            Vhat_hist = Vatt[i, ti, :H]
            Vact_hist = pv_h[gi // T, (gi % T) // 6] * C.DELTA_HOURS
        eV = Vact_hist - Vhat_hist
        V_scen[wi] = np.maximum(0.0, Vhat_now + eV)

    N_scen = L_scen - V_scen

    out.update({"M": M, "origin": origin, "price_scen": price_scen,
                "N_scen": N_scen, "L_scen": L_scen, "V_scen": V_scen,
                "price_hat": price_hat, "weights": np.full(M, 1.0 / M),
                "day_off": day_off})
    return out


def scenario_correlation_diag(sc) -> dict:
    if sc["M"] < 3:
        return {"corr_price_N": float("nan"), "M": sc["M"]}
    ph = sc["price_scen"].mean(axis=0)
    Nh = sc["N_scen"].mean(axis=0)
    ep = (sc["price_scen"] - ph[None, :]).ravel()
    eN = (sc["N_scen"] - Nh[None, :]).ravel()
    if np.std(ep) < 1e-12 or np.std(eN) < 1e-12:
        return {"corr_price_N": float("nan"), "M": sc["M"]}
    return {"corr_price_N": float(np.corrcoef(ep, eN)[0, 1]), "M": sc["M"]}
