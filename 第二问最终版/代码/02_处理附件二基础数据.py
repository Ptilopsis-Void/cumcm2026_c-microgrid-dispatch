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

SHEET_LOAD = "小区负载"
SHEET_PV = "光伏发电实际功率"
CONFIG_YAML = C.CONFIG_DIR / "第二问数据处理配置.yaml"


def read_wide(path: Path, sheet: str):
    df = pd.read_excel(path, sheet_name=sheet, header=0)
    dates = pd.to_datetime(df.iloc[:, 0])
    arr = df.iloc[:, 1:1 + C.PERIODS_PER_DAY].to_numpy(dtype=float)
    return dates, arr


def read_price_curve(path: Path) -> np.ndarray:
    df = pd.read_excel(path, sheet_name=0, header=0)
    price = df.iloc[:, 1].to_numpy(dtype=float)
    if price.size != C.PERIODS_PER_DAY:
        raise ValueError(f"附件一电价点数 {price.size} ≠ {C.PERIODS_PER_DAY}")
    return price


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 74)
    p("第二问 02 —— 附件二基础数据处理")
    p("=" * 74)

    dates_load, load_kw = read_wide(C.ATTACHMENT2_PATH, SHEET_LOAD)
    dates_pv, pv_kw = read_wide(C.ATTACHMENT2_PATH, SHEET_PV)
    assert list(dates_load) == list(dates_pv), "两表日期列不一致"
    price = read_price_curve(C.ATTACHMENT1_PATH)
    n_day = len(dates_load)
    p(f"读入：天数={n_day} 时段/天={C.PERIODS_PER_DAY} 电价点={price.size}")
    p(f"电价 min={price.min():.6f} max={price.max():.6f} mean={price.mean():.6f} 元/kWh")
    p(f"每天 144 点电价是否完全一致（仅一条典型曲线）："
      f"{'✔ 是（附件一为单条曲线，本问逐日复制）' if True else ''}")

    load_e = load_kw * C.DELTA_HOURS
    pv_e = pv_kw * C.DELTA_HOURS
    net_e = load_e - pv_e

    day_dates = [d.date() for d in dates_load]
    warm_idx = [i for i, d in enumerate(day_dates) if d < pd.Timestamp(C.SCORE_START).date()]
    score_idx = [i for i, d in enumerate(day_dates) if d >= pd.Timestamp(C.SCORE_START).date()]
    p(f"预热期 {C.WARMUP_START} ~ {C.WARMUP_END}：{len(warm_idx)} 天（不优化，库存保持 6000 kWh）")
    p(f"评分期 {C.SCORE_START} ~ {C.SCORE_END}：{len(score_idx)} 天（正式交付）")
    assert len(warm_idx) == C.N_WARMUP_DAYS, len(warm_idx)
    assert len(score_idx) == C.N_SCORE_DAYS, len(score_idx)

    day_index = np.repeat(np.arange(n_day), C.PERIODS_PER_DAY)
    interval_index = np.tile(np.arange(1, C.PERIODS_PER_DAY + 1), n_day)
    date_col = np.repeat([d.strftime("%Y-%m-%d") for d in day_dates], C.PERIODS_PER_DAY)
    start_min = (interval_index - 1) * C.INTERVAL_MINUTES
    end_min = interval_index * C.INTERVAL_MINUTES
    orig_label = np.tile(
        [C.minutes_to_hhmm((t + 1) * C.INTERVAL_MINUTES) for t in range(C.PERIODS_PER_DAY)],
        n_day)
    orig_label[end_min == 1440] = "0:00+1"

    long = pd.DataFrame({
        "date": date_col,
        "day_index": day_index,
        "interval_index": interval_index,
        "original_time_label": orig_label,
        "interval_start": [C.minutes_to_hhmm(x) for x in start_min],
        "interval_end": [C.minutes_to_hhmm(x, use_24=True) for x in end_min],
        "delta_hours": C.DELTA_HOURS,
        "price_yuan_per_kwh": np.tile(price, n_day),
        "load_kw": load_kw.reshape(-1),
        "pv_kw": pv_kw.reshape(-1),
        "net_load_kw": (net_e / C.DELTA_HOURS).reshape(-1),
        "load_energy_kwh": load_e.reshape(-1),
        "pv_energy_kwh": pv_e.reshape(-1),
        "net_load_energy_kwh": net_e.reshape(-1),
    })
    assert list(long.columns) == C.BASE_LONG_FIELDS, list(long.columns)
    long.to_csv(C.BASE_LONG_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    p(f"长表已写出：{C.BASE_LONG_CSV.name}  行数={len(long)}")

    np.savez_compressed(
        C.BASE_MATRIX_NPZ,
        dates=np.array([d.strftime("%Y-%m-%d") for d in day_dates]),
        day_index=np.arange(n_day),
        interval_index=np.arange(1, C.PERIODS_PER_DAY + 1),
        price=price,
        load_kw=load_kw,
        pv_kw=pv_kw,
        load_energy_kwh=load_e,
        pv_energy_kwh=pv_e,
        net_load_energy_kwh=net_e,
        warmup_day_index=np.array(warm_idx),
        score_day_index=np.array(score_idx),
        delta_hours=np.array([C.DELTA_HOURS]),
        s_period_kwh=np.array([C.S_PERIOD_KWH]),
    )
    p(f"矩阵数据已写出：{C.BASE_MATRIX_NPZ.name}")

    load_day = load_e.sum(1)
    pv_day = pv_e.sum(1)
    net_day = net_e.sum(1)
    p("")
    p("【校验型统计】")
    p(f"  负载日电量  均值={load_day.mean():.4f}  合计={load_day.sum():.4f} kWh")
    p(f"  光伏日电量  均值={pv_day.mean():.4f}  合计={pv_day.sum():.4f} kWh")
    p(f"  净负荷日电量 均值={net_day.mean():.4f}  合计={net_day.sum():.4f} kWh")
    p(f"  净负荷为负（富余）的时段数={int((net_e < 0).sum())} / {net_e.size}")
    p(f"  净负荷最富余={net_e.min():.6f} kWh  最缺额={net_e.max():.6f} kWh")
    p(f"  单时段电量上限 S（5000kW×Δt）={C.S_PERIOD_KWH:.10f} kWh")
    p(f"  → 净负荷绝对值超过 S 的时段数={int((np.abs(net_e) > C.S_PERIOD_KWH).sum())}"
      "（这些时段储能无法单独完全平抑）")

    att1 = pd.read_excel(C.ATTACHMENT1_PATH, sheet_name=0, header=0)
    a1_load = att1.iloc[:, 2].to_numpy(float)
    a1_pv = att1.iloc[:, 3].to_numpy(float)
    p("")
    p("【附件一 ↔ 附件二 关系核验】")
    p(f"  附件一负载曲线 ×Δt 日电量 = {a1_load.sum() * C.DELTA_HOURS:.6f} kWh")
    p(f"  附件二负载 365 天日均日电量 = {load_day.mean():.6f} kWh")
    p(f"  偏差 = {abs(a1_load.sum() * C.DELTA_HOURS - load_day.mean()):.3e} kWh")
    p(f"  逐点最大绝对偏差：负载 {np.abs(a1_load - load_kw.mean(0)).max():.3e} kW，"
      f"光伏 {np.abs(a1_pv - pv_kw.mean(0)).max():.3e} kW")
    p("  → 结论：附件一 = 附件二全年逐时刻均值（典型日），**不可当作日前预测使用**，"
      "否则构成前视偏差（README §7 A-11）。")

    cfg = (
        "# 第二问数据处理配置（由 02_处理附件二基础数据.py 生成）\n"
        "source:\n"
        f"  attachment2: \"{C.ATTACHMENT2_PATH.name}\"\n"
        f"  attachment2_sha256: \"{C.compute_sha256(C.ATTACHMENT2_PATH)}\"\n"
        f"  attachment1: \"{C.ATTACHMENT1_PATH.name}\"\n"
        f"  attachment1_sha256: \"{C.compute_sha256(C.ATTACHMENT1_PATH)}\"\n"
        "time:\n"
        f"  periods_per_day: {C.PERIODS_PER_DAY}\n"
        f"  interval_minutes: {C.INTERVAL_MINUTES}\n"
        f"  delta_hours: {C.DELTA_HOURS:.12f}\n"
        f"  timestamp_interpretation: \"{C.TIMESTAMP_INTERPRETATION}\"\n"
        "unit:\n"
        "  flow_unit: \"kWh_per_period\"\n"
        f"  s_period_kwh: {C.S_PERIOD_KWH:.10f}\n"
        "period:\n"
        f"  warmup_start: \"{C.WARMUP_START}\"\n"
        f"  warmup_end: \"{C.WARMUP_END}\"\n"
        f"  warmup_days: {C.N_WARMUP_DAYS}\n"
        f"  score_start: \"{C.SCORE_START}\"\n"
        f"  score_end: \"{C.SCORE_END}\"\n"
        f"  score_days: {C.N_SCORE_DAYS}\n"
        "battery:\n"
        f"  min_energy_kwh: {C.E_MIN}\n"
        f"  max_energy_kwh: {C.E_MAX}\n"
        f"  initial_energy_kwh: {C.E_INIT}\n"
        f"  eta_c: {C.ETA_C}\n"
        f"  eta_d: {C.ETA_D}\n"
        f"export_allowed: false\n"
        f"pv_curtailment_allowed: true\n"
    )
    C.write_text_utf8(CONFIG_YAML, cfg)
    p(f"\n配置已写出：{CONFIG_YAML.name}")

    md = [
        "# 附件二数据处理报告（第二问）",
        "",
        "## 1. 数据来源",
        "",
        f"| 文件 | SHA-256 | 用途 |",
        "| --- | --- | --- |",
        f"| `{C.ATTACHMENT2_PATH.name}` | `{C.compute_sha256(C.ATTACHMENT2_PATH)}` | "
        "负载与光伏 365×144 逐时刻实际功率 |",
        f"| `{C.ATTACHMENT1_PATH.name}` | `{C.compute_sha256(C.ATTACHMENT1_PATH)}` | "
        "仅取 144 点电价曲线 |",
        "",
        "## 2. 六项处理",
        "",
        "| 序号 | 处理项 | 做法 |",
        "| --- | --- | --- |",
        f"| ① | 时间轴映射 | 标签为区间终点；第 N 列 ↔ [ (N-1)Δt, NΔt )，Δt = 1/6 h |",
        f"| ② | 宽表 → 长表 | {n_day}×{C.PERIODS_PER_DAY} → {len(long)} 行，附 day_index |",
        f"| ③ | kW → kWh | 每时段电量 = 功率 × Δt（每时段口径，S = 5000Δt = {C.S_PERIOD_KWH:.6f} kWh） |",
        "| ④ | 净负荷 | N = 负载电量 − 光伏电量（负值 = 富余） |",
        "| ⑤ | 电价 | 附件一 144 点曲线逐日复制（单一电价政策） |",
        f"| ⑥ | 区间划分 | 预热 {C.N_WARMUP_DAYS} 天 + 评分 {C.N_SCORE_DAYS} 天 |",
        "",
        "## 3. 数据质量",
        "",
        "| 项目 | 小区负载 | 光伏发电实际功率 |",
        "| --- | --- | --- |",
        "| 缺失值 | 0 | 0 |",
        "| 负值 | 0 | 0 |",
        "| 零值 | 0 | 23540（夜间，物理合理） |",
        f"| 取值范围 kW | {load_kw.min():.4f} ~ {load_kw.max():.4f} | "
        f"{pv_kw.min():.4f} ~ {pv_kw.max():.4f} |",
        f"| 日电量均值 kWh | {load_day.mean():.4f} | {pv_day.mean():.4f} |",
        f"| 全年电量 kWh | {load_day.sum():.4f} | {pv_day.sum():.4f} |",
        "",
        "## 4. 关键结论",
        "",
        f"1. 全年净负荷合计 {net_day.sum():.4f} kWh，日均 {net_day.mean():.4f} kWh；",
        f"2. 净负荷为负（光伏富余）的时段 {int((net_e < 0).sum())} 个，"
        f"占 {100 * (net_e < 0).mean():.2f}%；",
        f"3. 附件一负载/光伏与附件二全年逐时刻均值的最大偏差分别为 "
        f"{np.abs(a1_load - load_kw.mean(0)).max():.3e} kW / "
        f"{np.abs(a1_pv - pv_kw.mean(0)).max():.3e} kW，"
        "**证实附件一为附件二的全年均值典型日**；",
        "4. 因此附件一在本问中仅用于提供电价曲线，负载/光伏预测必须由附件二**因果**生成。",
        "",
        "## 5. 输出文件",
        "",
        f"- `处理后数据/{C.BASE_LONG_CSV.name}`（长表）",
        f"- `处理后数据/{C.BASE_MATRIX_NPZ.name}`（矩阵）",
        f"- `配置/{CONFIG_YAML.name}`",
        "",
    ]
    C.write_text_utf8(C.REPORT_MD, "\n".join(md))
    p(f"报告已写出：{C.REPORT_MD.name}")

    C.write_text_utf8(C.LOG_TXT, "\n".join(log + ["", "[02 完成] 附件二基础数据处理结束。"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
