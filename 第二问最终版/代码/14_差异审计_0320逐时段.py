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
P5 = _load("05_求解日前计划.py", "q2_plan")
DP = _load("06_DP价值执行器.py", "q2_dp")

import numpy as np


def main() -> int:
    C.ensure_dirs()
    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    B = np.load(C.RESULT_DIR / "第二问_执行器回测.npz", allow_pickle=False)

    price = np.asarray(Z["price"], float)
    N_act_all = np.asarray(Z["net_load_energy_kwh"], float)
    scen_N_all = np.asarray(Zd["scen_L"], float) - np.asarray(Zd["scen_V"], float)
    date_strs = [str(s) for s in Z["dates"]]
    date_of = {s: i for i, s in enumerate(date_strs)}

    g_dp_var = np.asarray(B["g_dp_var"], float)
    dp_b = np.asarray(B["dp_b"], float)
    dp_C = np.asarray(B["dp_C"], float)
    dp_D = np.asarray(B["dp_D"], float)
    dp_E = np.asarray(B["dp_E"], float)

    d = date_of["2025-03-20"]
    g_day = g_dp_var[d]
    N_act = N_act_all[d]
    price_d = price

    def slot(ti):
        return f"{C.minutes_to_hhmm((ti-1)*10)}-{C.minutes_to_hhmm(ti*10)}"

    E0_actual = float(B["dp_E"][d - 1, -1]) if d - 1 >= 0 else float(C.E_INIT)
    rows = []
    for delta in (6.0, 3.0, 1.5):
        vf = DP.build_value_functions(scen_N_all[d], g_day, price_d, delta=delta)
        o = DP.dp_execute(vf["Hbar"], vf["R"], N_act, g_day, E0_actual)
        for t in range(C.PERIODS_PER_DAY):
            rows.append((
                f"δ={delta}", t + 1, slot(t + 1), f"{price_d[t]:.4f}",
                f"{g_day[t]:.6f}", f"{N_act[t]:.6f}", f"{N_act[t]-g_day[t]:.6f}",
                f"{o['C'][t]:.6f}", f"{o['D'][t]:.6f}", f"{o['b'][t]:.6f}",
                f"{o['E'][t]:.6f}",
                f"{price_d[t]*g_day[t]:.6f}", f"{5.0*price_d[t]*o['b'][t]:.6f}",
            ))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_0320逐时段差异审计.csv",
        ("网格档", "时段序号", "实际区间", "电价c_t", "计划购电量g_t", "实际净负荷N_t",
         "缺口r_t", "充电量C_t", "放电量D_t", "紧急购电量b_t", "时段末库存E_t",
         "计划费c_t*g_t", "紧急费5c_t*b_t"),
        rows)

    print("=" * 72)
    print("2025-03-20 三档 DP 汇总（同一 g_day，E0 来自 07 闭环）")
    print(f"  E0(真实日初库存) = {E0_actual:.6f} kWh")
    print(f"  计划购电量合计 = {g_day.sum():.6f} kWh，计划费 = {price@g_day:.6f} 元")
    for delta in (6.0, 3.0, 1.5):
        vf = DP.build_value_functions(scen_N_all[d], g_day, price_d, delta=delta)
        o = DP.dp_execute(vf["Hbar"], vf["R"], N_act, g_day, E0_actual)
        bc = float((5.0 * price_d) @ o["b"])
        print(f"  δ={delta:<4} 紧急量 {o['b'].sum():.6f} kWh，紧急费 {bc:.6f} 元，"
              f"总费用 {price@g_day + bc:.6f} 元，末库存 {o['E'][-1]:.6f}")
    print("  参考值：计划量 65969.247，紧急量 176.258，总费用 41450.015 元")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
