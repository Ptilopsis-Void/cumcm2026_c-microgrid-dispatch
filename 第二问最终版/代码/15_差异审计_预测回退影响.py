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


C = _load("_comm2.py", "q2_comm")
P5 = _load("05_求解日前计划.py", "q2_plan")
DP = _load("06_DP价值执行器.py", "q2_dp")

import numpy as np

T = C.PERIODS_PER_DAY
M = C.M_SCENARIOS


def causal_forecast(load_e, wd, min_same: int) -> np.ndarray:
    n_day = load_e.shape[0]
    Lhat = np.zeros_like(load_e)
    for d in range(n_day):
        cand = [i for i in range(max(0, d - C.LOAD_LOOKBACK_DAYS), d) if wd[i] == wd[d]]
        if len(cand) >= min_same:
            Lhat[d] = load_e[cand].mean(0)
        else:
            lo = max(0, d - C.LOAD_FALLBACK_DAYS)
            if d - lo > 0:
                Lhat[d] = load_e[lo:d].mean(0)
            else:
                Lhat[d] = load_e[0]
    return Lhat


def run_dp_chain(scen_N_all, N_act_all, price, score_idx, lp30):
    E_current = float(C.E_INIT)
    g_sum = 0.0
    b_sum = 0.0
    pc = 0.0
    ec = 0.0
    tend = float("nan")
    for d in score_idx:
        g_d = lp30.solve(scen_N_all[d], E_current)["g"]
        vf = DP.build_value_functions(scen_N_all[d], g_d, price, delta=6.0)
        o = DP.dp_execute(vf["Hbar"], vf["R"], N_act_all[d], g_d, E_current)
        g_sum += float(g_d.sum())
        b_sum += float(o["b"].sum())
        pc += float(price @ g_d)
        ec += float((5.0 * price) @ o["b"])
        E_current = float(o["E"][-1])
    tend = E_current
    return g_sum, b_sum, pc, ec, pc + ec, tend


def main() -> int:
    C.ensure_dirs()
    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    price = np.asarray(Z["price"], float)
    N_act_all = np.asarray(Z["net_load_energy_kwh"], float)
    load_e = np.asarray(Z["load_energy_kwh"], float)
    pv_e = np.asarray(Z["pv_energy_kwh"], float)
    date_strs = [str(s) for s in Z["dates"]]
    dates = [__import__("pandas").Timestamp(s).date() for s in date_strs]
    wd = np.array([d.weekday() for d in dates])
    score_idx = np.asarray(Z["score_day_index"], int)

    Vhat = Zd["Vhat"]
    resid_V = Zd["resid_V"]

    lp30 = P5.DayPlanLP(price, nu=P5.NU, m=M)

    results = {}
    print("=" * 72)
    print("负荷预测回退规则只读对照（min_same=1 vs min_same=2）")
    print("=" * 72)
    for ms, label in ((1, "现代码(min_same=1)"), (2, "图片(min_same=2)")):
        Lhat = causal_forecast(load_e, wd, min_same=ms)
        resid_L = load_e - Lhat
        scen_L = np.zeros((load_e.shape[0], M, T))
        for d in range(load_e.shape[0]):
            pool = list(range(max(1, d - C.RESIDUAL_WINDOW_DAYS), d))
            if not pool:
                pool = [0]
            for w in range(M):
                i = pool[w % len(pool)]
                scen_L[d, w] = np.maximum(0.0, Lhat[d] + resid_L[i])
        scen_V = Zd["scen_V"]
        scen_N = scen_L - scen_V

        dt = C.DELTA_HOURS
        maeL = np.abs(load_e - Lhat)[score_idx].mean() / dt
        maeV = np.abs(pv_e - Vhat)[score_idx].mean() / dt
        maeN = np.abs((load_e - pv_e) - (Lhat - Vhat))[score_idx].mean() / dt
        n_diff_resid = int((np.abs(resid_L - (load_e - Zd["Lhat"])) > 1e-9).sum())
        print(f"\n[{label}] 评分期 MAE：负载 {maeL:.3f} kW，光伏 {maeV:.3f} kW，"
              f"净负荷 {maeN:.3f} kW")
        print(f"  与现代码的残差差异：{n_diff_resid} 个（日×时段），"
              f"影响历史日索引 7..13")

        t0 = time.perf_counter()
        r = run_dp_chain(scen_N, N_act_all, price, score_idx, lp30)
        dtc = time.perf_counter() - t0
        results[ms] = r
        print(f"  各自重订/DP(δ=6)：计划量 {r[0]:.3f} kWh，计划费 {r[2]:.3f} 元，"
              f"紧急量 {r[1]:.3f} kWh，紧急费 {r[3]:.3f} 元，总费用 {r[4]:.3f} 元，"
              f"年末库存 {r[5]:.3f} kWh（{dtc:.0f}s）")

    r1 = results[1]
    r2 = results[2]
    print("\n" + "=" * 72)
    print("两口径年度差异（min_same=2 − min_same=1）")
    for name, a, b in (("年计划购电量", r1[0], r2[0]),
                       ("年紧急购电量", r1[1], r2[1]),
                       ("年计划费", r1[2], r2[2]),
                       ("年紧急费", r1[3], r2[3]),
                       ("年总费用", r1[4], r2[4]),
                       ("年末库存", r1[5], r2[5])):
        print(f"  {name}: {a:.6f} → {b:.6f}  (Δ={b-a:+.6f})")
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_预测回退规则影响对照.csv",
        ("口径", "年计划购电量_kWh", "年紧急购电量_kWh", "年计划费_元",
         "年紧急费_元", "年总费用_元", "年末库存_kWh"),
        [(f"min_same={ms}", f"{results[ms][0]:.6f}", f"{results[ms][1]:.6f}",
          f"{results[ms][2]:.6f}", f"{results[ms][3]:.6f}", f"{results[ms][4]:.6f}",
          f"{results[ms][5]:.6f}") for ms in (1, 2)])
    print("已保存：第二问_预测回退规则影响对照.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
