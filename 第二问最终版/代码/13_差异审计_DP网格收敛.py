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
DELTAS = (6.0, 3.0, 1.5)
SPEC_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def run_dp_chain(delta, lp30, N_act_all, scen_N_all, price, score_idx):
    E_current = float(C.E_INIT)
    g_chain = np.zeros((365, T))
    b_chain = np.zeros((365, T))
    C_chain = np.zeros((365, T))
    D_chain = np.zeros((365, T))
    E_chain = np.zeros((365, T))
    plan_cost = np.zeros(365)
    emerg_cost = np.zeros(365)
    for d in score_idx:
        r = lp30.solve(scen_N_all[d], E_current)
        g_d = r["g"]
        g_chain[d] = g_d
        vf = DP.build_value_functions(scen_N_all[d], g_d, price, delta=delta)
        o = DP.dp_execute(vf["Hbar"], vf["R"], N_act_all[d], g_d, E_current)
        b_chain[d] = o["b"]
        C_chain[d] = o["C"]
        D_chain[d] = o["D"]
        E_chain[d] = o["E"]
        plan_cost[d] = float(price @ g_d)
        emerg_cost[d] = float((5.0 * price) @ o["b"])
        E_current = float(o["E"][-1])
    return dict(g=g_chain, b=b_chain, C=C_chain, D=D_chain, E=E_chain,
                plan_cost=plan_cost, emerg_cost=emerg_cost)


def main() -> int:
    C.ensure_dirs()
    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    price = np.asarray(Z["price"], float)
    N_act_all = np.asarray(Z["net_load_energy_kwh"], float)
    scen_N_all = np.asarray(Zd["scen_L"], float) - np.asarray(Zd["scen_V"], float)
    date_strs = [str(s) for s in Z["dates"]]
    date_of = {s: i for i, s in enumerate(date_strs)}
    score_idx = np.asarray(Z["score_day_index"], int)

    lp30 = P5.DayPlanLP(price, nu=P5.NU, m=M)

    d0 = int(score_idx[0])
    g_ref = None
    decouple_rows = []
    for delta in DELTAS:
        r = lp30.solve(scen_N_all[d0], C.E_INIT)
        g = r["g"]
        if g_ref is None:
            g_ref = g
        decouple_rows.append((f"δ={delta}", date_strs[d0], f"{float(C.E_INIT):.4f}",
                              "LP(不含δ)", f"{float(np.abs(g - g_ref).max()):.2e}"))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_DP网格与日前计划解耦审计.csv",
        ("网格档", "日期", "固定E_init", "日前计划来源", "g与δ=6逐位最大差_kWh"),
        decouple_rows)

    ann_rows = []
    spec_rows = []
    print("=" * 74)
    print("第二问 13 —— DP 三档网格年度收敛审计（各自重订/DP 链）")
    print("=" * 74)
    for delta in DELTAS:
        t0 = time.perf_counter()
        R = run_dp_chain(delta, lp30, N_act_all, scen_N_all, price, score_idx)
        dt = time.perf_counter() - t0
        g_sum = float(R["g"][score_idx].sum())
        b_sum = float(R["b"][score_idx].sum())
        pc = float(R["plan_cost"][score_idx].sum())
        ec = float(R["emerg_cost"][score_idx].sum())
        tc = pc + ec
        tend = float(R["E"][score_idx[-1], -1])
        print(f"  δ={delta:<4} 计划量 {g_sum:.6f} kWh，计划费 {pc:.4f}，"
              f"紧急量 {b_sum:.6f} kWh，紧急费 {ec:.4f}，总费用 {tc:.4f}，"
              f"年末库存 {tend:.6f} kWh，耗时 {dt:.1f} s")
        ann_rows.append((f"δ={delta}", f"{g_sum:.6f}", f"{pc:.6f}", f"{b_sum:.6f}",
                         f"{ec:.6f}", f"{tc:.6f}", f"{tend:.6f}", f"{dt:.1f}"))
        for ds in SPEC_DATES:
            d = date_of[ds]
            gd = float(R["g"][d].sum())
            bd = float(R["b"][d].sum())
            cd = float(price @ R["g"][d]) + float((5.0 * price) @ R["b"][d])
            spec_rows.append((f"δ={delta}", ds, f"{gd:.6f}", f"{bd:.6f}",
                              f"{gd + bd:.6f}", f"{cd:.6f}"))

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_DP三档网格年度收敛.csv",
        ("网格档", "年计划购电量_kWh", "年计划费_元", "年紧急购电量_kWh",
         "年紧急费_元", "年总费用_元", "年末库存_kWh", "运行时间_s"),
        ann_rows)
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_DP三档网格指定日期收敛.csv",
        ("网格档", "日期", "计划量_kWh", "紧急量_kWh", "总计费量_kWh", "总费用_元"),
        spec_rows)

    print("=" * 74)
    print("已保存：DP 三档网格年度收敛 / 指定日期收敛 / δ解耦审计")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
