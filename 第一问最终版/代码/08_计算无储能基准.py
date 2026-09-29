from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    PERIODS_PER_DAY, DELTA_HOURS, load_formal_input, BASELINE_CSV,
)

N = PERIODS_PER_DAY
DT = DELTA_HOURS


def compute_baseline(df=None):
    if df is None:
        df = load_formal_input()

    load = df["load_kw"].to_numpy(float)
    pv = df["pv_forecast_kw"].to_numpy(float)
    price = df["price_yuan_per_kwh"].to_numpy(float)

    G0 = np.maximum(load - pv, 0.0)
    W0 = np.maximum(pv - load, 0.0)
    J0 = float(np.sum(price * G0 * DT))

    rows = []
    for t in range(N):
        rows.append({
            "interval_index": int(df["interval_index"].iloc[t]),
            "original_time_label": df["original_time_label"].iloc[t],
            "interval_start": df["interval_start"].iloc[t],
            "interval_end": df["interval_end"].iloc[t],
            "price_yuan_per_kwh": price[t],
            "load_kw": load[t],
            "pv_forecast_kw": pv[t],
            "grid_purchase_kw": G0[t],
            "pv_curtailment_kw": W0[t],
            "grid_purchase_energy_kwh": G0[t] * DT,
            "period_purchase_cost_yuan": price[t] * G0[t] * DT,
        })
    base_df = pd.DataFrame(rows)
    base_df.to_csv(BASELINE_CSV, index=False, encoding="utf-8-sig",
                   float_format="%.10f")

    return {
        "J0": J0,
        "G0": G0,
        "W0": W0,
        "total_grid_energy": float(np.sum(G0 * DT)),
        "total_curtail_energy": float(np.sum(W0 * DT)),
        "baseline_df": base_df,
    }


if __name__ == "__main__":
    r = compute_baseline()
    print(f"[08] 无储能购电费用 J_0 = {r['J0']:.8f} 元")
    print(f"[08] 无储能总购电量 = {r['total_grid_energy']:.4f} kWh")
    print(f"[08] 无储能总弃光量 = {r['total_curtail_energy']:.4f} kWh")
    print(f"[08] 已导出：{BASELINE_CSV.name}")
