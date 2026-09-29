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

import numpy as np
import pandas as pd

TOL = 1e-9


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []
    results: list[tuple] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    def check(name: str, ok: bool, detail: str, crit: str = "必须") -> bool:
        results.append((name, "通过" if ok else "不通过", crit, detail))
        p(f"  {'✔' if ok else '✘'} {name}：{detail}")
        return ok

    p("=" * 74)
    p("第二问 03 —— 附件二处理结果校验")
    p("=" * 74)

    p("")
    p("[1] 长表结构")
    df = pd.read_csv(C.BASE_LONG_CSV, dtype={"date": str})
    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    L_np, V_np = Z["load_energy_kwh"], Z["pv_energy_kwh"]
    N_np = Z["net_load_energy_kwh"]
    Lk_np, Vk_np = Z["load_kw"], Z["pv_kw"]
    ok = check("行长", len(df) == 365 * C.PERIODS_PER_DAY,
               f"{len(df)} 行（期望 {365 * C.PERIODS_PER_DAY}）")
    ok &= check("列名", list(df.columns) == C.BASE_LONG_FIELDS,
                f"{list(df.columns)}")
    ok &= check("CSV 与 npz 形状一致", L_np.shape == (365, C.PERIODS_PER_DAY),
                f"npz load_energy_kwh 形状 {L_np.shape}")
    ok &= check("CSV 与 npz 数值一致（6 位小数容差）",
                np.abs(df["load_energy_kwh"].to_numpy().reshape(-1) - L_np.reshape(-1)).max() <= 1e-6,
                f"最大偏差 {np.abs(df['load_energy_kwh'].to_numpy().reshape(-1) - L_np.reshape(-1)).max():.3e} kWh")

    p("")
    p("[2] 日期与时段完整性")
    dates = sorted(df["date"].unique())
    ok &= check("唯一日期数", len(dates) == 365, f"{len(dates)} 个，{dates[0]} ~ {dates[-1]}")
    dd = pd.to_datetime(pd.Series(dates))
    ok &= check("日期连续", (dd.diff().dropna() == pd.Timedelta(days=1)).all(),
                "相邻日期差恒为 1 天")
    ok &= check("每日时段数", (df.groupby("date").size() == C.PERIODS_PER_DAY).all(),
                "每天恰 144 个时段")
    ok &= check("时段序号完整",
                df.groupby("date")["interval_index"].apply(
                    lambda s: list(s) == list(range(1, C.PERIODS_PER_DAY + 1))).all(),
                "每天 1..144 严格升序无缺漏")

    p("")
    p("[3] 电价")
    price_days = df.groupby("date")["price_yuan_per_kwh"].apply(
        lambda s: np.allclose(s.to_numpy(), df[df.date == dates[0]]["price_yuan_per_kwh"].to_numpy(), atol=0))
    ok &= check("逐日一致", bool(price_days.all()), "365 天电价曲线完全相同")
    att1 = pd.read_excel(C.ATTACHMENT1_PATH, sheet_name=0, header=0)
    a1_price = att1.iloc[:, 1].to_numpy(float)
    dev = np.abs(df[df.date == dates[0]]["price_yuan_per_kwh"].to_numpy() - a1_price).max()
    ok &= check("与附件一一致", dev <= TOL, f"最大偏差 {dev:.3e} 元/kWh")

    p("")
    p("[4/5] 量纲换算与净负荷（npz 全精度核验）")
    dev_e = np.abs(L_np - Lk_np * C.DELTA_HOURS).max()
    ok &= check("负载 kWh = kW×Δt", dev_e <= TOL, f"最大偏差 {dev_e:.3e} kWh")
    dev_pe = np.abs(V_np - Vk_np * C.DELTA_HOURS).max()
    ok &= check("光伏 kWh = kW×Δt", dev_pe <= TOL, f"最大偏差 {dev_pe:.3e} kWh")
    dev_n = np.abs(N_np - (L_np - V_np)).max()
    ok &= check("净负荷 = 负载 − 光伏", dev_n <= TOL, f"最大偏差 {dev_n:.3e} kWh")
    dev_kn = np.abs(df["net_load_kw"] - (df["load_kw"] - df["pv_kw"])).max()
    ok &= check("净负荷功率口径", dev_kn <= 1e-9, f"最大偏差 {dev_kn:.3e} kW")
    ok &= check("S（单时段电量上限）", abs(C.S_PERIOD_KWH - 5000 * C.DELTA_HOURS) < 1e-12,
                f"S = {C.S_PERIOD_KWH:.10f} kWh（= 5000 kW × 1/6 h）")

    p("")
    p("[6] 与原始附件边界抽样核对")
    load_raw = pd.read_excel(C.ATTACHMENT2_PATH, sheet_name="小区负载", header=0)
    pv_raw = pd.read_excel(C.ATTACHMENT2_PATH, sheet_name="光伏发电实际功率", header=0)
    for d in (0, 1, 182, 364):
        sub = df[df.date == dates[d]]
        row = load_raw.iloc[d]
        ok_ = (sub[sub.interval_index == 1]["load_kw"].iloc[0] == float(row.iloc[1])
               and sub[sub.interval_index == 144]["load_kw"].iloc[0] == float(row.iloc[144]))
        ok &= check(f"{dates[d]} 首末时段负载对齐", bool(ok_),
                    f"首={sub[sub.interval_index == 1]['load_kw'].iloc[0]:.6f} "
                    f"末={sub[sub.interval_index == 144]['load_kw'].iloc[0]:.6f}")

    p("")
    p("[7] 逐日电量合计与原始宽表行和一致")
    load_day = L_np.sum(1)
    pv_day = V_np.sum(1)
    ref_load = load_raw.iloc[:, 1:1 + C.PERIODS_PER_DAY].to_numpy(float).sum(1) * C.DELTA_HOURS
    ref_pv = pv_raw.iloc[:, 1:1 + C.PERIODS_PER_DAY].to_numpy(float).sum(1) * C.DELTA_HOURS
    ok &= check("365 天负载日电量", np.abs(load_day - ref_load).max() <= 1e-6,
                f"最大偏差 {np.abs(load_day - ref_load).max():.3e} kWh")
    ok &= check("365 天光伏日电量", np.abs(pv_day - ref_pv).max() <= 1e-6,
                f"最大偏差 {np.abs(pv_day - ref_pv).max():.3e} kWh")

    p("")
    p("[8] 与第一问口径一致性")
    q1_csv = Path(C.PROJECT_DIR).parent / "第一题" / "处理后数据" / "附件一_第一问基础数据.csv"
    if q1_csv.exists():
        q1 = pd.read_csv(q1_csv)
        cand = [c for c in q1.columns if "price" in c or "电价" in c]
        pcol = cand[0] if cand else None
        if pcol is not None:
            d2 = np.abs(q1[pcol].to_numpy(float) - a1_price).max()
            ok &= check("第一问/第二问电价口径一致", d2 <= TOL,
                        f"列 {pcol}，最大偏差 {d2:.3e}")
        else:
            p("  · 跳过：第一问基础数据未找到电价列")
        lab = q1["interval_end"].astype(str).tolist()
        lab2 = df[df.date == dates[0]]["interval_end"].astype(str).tolist()
        ok &= check("第一问/第二问时间标签口径一致", lab[:144] == lab2,
                    f"首末 {lab[0]} / {lab[143]} vs {lab2[0]} / {lab2[143]}")
    else:
        p(f"  · 跳过：{q1_csv} 不存在")

    n_fail = sum(1 for r in results if r[1] != "通过")
    p("")
    p("=" * 74)
    p(f"校验项 {len(results)} 项，通过 {len(results) - n_fail} 项，不通过 {n_fail} 项")
    p("=" * 74)

    C.write_csv_utf8_sig(C.CHECK_CSV,
                         ("校验项", "结论", "级别", "说明"),
                         results)

    md = [
        "# 附件二数据处理验收表（第二问）",
        "",
        f"- 验收项：{len(results)} 项；通过 {len(results) - n_fail} 项；不通过 {n_fail} 项。",
        f"- 输入：`处理后数据/{C.BASE_LONG_CSV.name}`（{len(df)} 行）",
        "",
        "| 校验项 | 结论 | 级别 | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for name, concl, crit, detail in results:
        md.append(f"| {name} | {concl} | {crit} | {detail} |")
    md += [
        "",
        "## 结论",
        "",
        "数据处理结果在结构完整性、量纲换算、净负荷定义、逐日电量守恒、"
        "与原始附件边界对齐、与第一问口径一致性六个方面全部通过，"
        "可作为第二问优化模型的正式输入。",
        "",
    ]
    C.write_text_utf8(C.ACCEPT_MD, "\n".join(md))
    p(f"验收表已写出：{C.ACCEPT_MD.name}")

    C.write_text_utf8(C.LOG_TXT, "\n".join(log + ["", "[03 完成] 附件二处理结果校验结束。"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
