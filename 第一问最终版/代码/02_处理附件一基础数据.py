from pathlib import Path
import sys
from datetime import time as dt_time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    ATTACHMENT1_PATH, CSV_BASE_PATH, CSV_PLOT_PATH, TIME_MD, PROCESS_MD,
    DELTA_HOURS, INTERVAL_MINUTES, PERIODS_PER_DAY, TIMESTAMP_INTERPRETATION,
    parse_time_label_to_minutes, minutes_to_hhmm, label_to_text,
)

COL_MAP = {
    "电价": "price_yuan_per_kwh",
    "小区负载": "load_kw",
    "光伏发电预测功率": "pv_forecast_kw",
}

BASE_COLS = [
    "interval_index", "original_time_label", "interval_start", "interval_end",
    "delta_hours", "price_yuan_per_kwh", "load_kw", "pv_forecast_kw",
    "net_load_kw", "net_demand_kw", "pv_surplus_kw",
    "load_energy_kwh", "pv_forecast_energy_kwh",
    "net_demand_energy_kwh", "pv_surplus_energy_kwh",
]

PLOT_COLS = [
    "interval_index", "minute_of_day", "interval_start", "interval_end",
    "price_yuan_per_kwh", "load_kw", "pv_forecast_kw",
    "net_demand_kw", "pv_surplus_kw",
]


def write_time_mapping_report(first_label, last_label, n, pv_first, pv_last):
    lines = [
        "# 附件一时间标签映射说明",
        "",
        "## 原始时间标签概况",
        f"- 数据条数：{n}",
        f"- 首条标签：`{first_label}`",
        f"- 末条标签：`{last_label}`",
        "",
        "## 待判断问题",
        "附件1中 `00:10` 等标签，究竟是：",
        "1. 以 00:10 为区间**起点**（区间 [00:10, 00:20)）；",
        "2. 还是以 00:10 为区间**终点**（区间 [00:00, 00:10)）。",
        "",
        "## 两种解释的差异",
        "",
        "| 解释 | 第1区间 | 第144区间 | 末条标签含义 |",
        "|---|---|---|---|",
        "| 起点 | [00:10, 00:20) | [24:00, 24:10) | 跨入第二天 |",
        "| 终点（采用） | [00:00, 00:10) | [23:50, 24:00) | 当天 24:00 |",
        "",
        "## 证据与判断",
        "",
        f"1. **首行/末行**：首条为 `{first_label}`（不是 00:00），末条为 `{last_label}`。",
        "   只有“终点”解释能构成完整一整天；若为“起点”，首区间将缺 [00:00,00:10)。",
        f"2. **144 条 = 24h × 6**：{n} 条恰好覆盖 24 小时，“终点”解释下无缺漏、无越界。",
        "3. **题目文字**：附录2 明确“`0:00+1` 表示第二天凌晨 0:00，即当天 24:00”，",
        "   本文据此采用末条作为当天最后一个区间终点的离散化约定；题面未直接规定所有行的区间含义。",
        "4. **结果模板写法**：`result*.xlsx` 用 `10:00-10:10` 这类“[起点]-[终点]”区间，",
        "   原始模板从0:10-0:20至次日0:00-0:10，本文将其调整为覆盖当天0:00-24:00并明确说明。",
        f"5. **光伏日出日落**：光伏>0 首段标签 `{pv_first}`、末段标签 `{pv_last}`，",
        "   落在合理日照时段，不因解释偏移而产生明显偏差。",
        "",
        "## 结论",
        "",
        f"采用**解释 2（标签为区间终点）**：`{TIMESTAMP_INTERPRETATION}`。",
        f"第 t 个区间为 [标签(t)−10min, 标签(t))，约定区间内功率恒定并取该标签所在行；此为本文数据离散化假设。",
        "",
        "| 时段 | 区间 | 标签 |",
        "|---|---|---|",
        f"| 1 | [00:00, 00:10) | {first_label} |",
        "| 2 | [00:10, 00:20) | 00:20 |",
        "| … | … | … |",
        f"| 144 | [23:50, 24:00) | {last_label} (=24:00) |",
        "",
        "> 同时保留 `original_time_label` 原始文本字段，便于追溯。",
        "",
    ]
    TIME_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    raw = pd.read_excel(ATTACHMENT1_PATH)
    for cn in COL_MAP:
        if cn not in raw.columns:
            raise KeyError(f"附件1 缺少列：{cn}；实际列：{list(raw.columns)}")

    time_labels = list(raw.iloc[:, 0])
    minutes = [parse_time_label_to_minutes(v) for v in time_labels]

    first_label = label_to_text(time_labels[0])
    last_label = label_to_text(time_labels[-1])
    n = len(raw)

    pv_raw = raw["光伏发电预测功率"].values
    pv_pos_idx = [i for i, v in enumerate(pv_raw) if v > 0]
    pv_first = label_to_text(time_labels[pv_pos_idx[0]]) if pv_pos_idx else "无"
    pv_last = label_to_text(time_labels[pv_pos_idx[-1]]) if pv_pos_idx else "无"

    write_time_mapping_report(first_label, last_label, n, pv_first, pv_last)

    d = DELTA_HOURS
    df = pd.DataFrame()
    df["interval_index"] = np.arange(1, n + 1, dtype=int)
    df["original_time_label"] = [label_to_text(v) for v in time_labels]
    df["interval_start"] = [minutes_to_hhmm(m - INTERVAL_MINUTES) for m in minutes]
    df["interval_end"] = [minutes_to_hhmm(m, use_24=True) for m in minutes]
    df["delta_hours"] = d

    df["price_yuan_per_kwh"] = raw["电价"].values
    df["load_kw"] = raw["小区负载"].values
    df["pv_forecast_kw"] = raw["光伏发电预测功率"].values

    df["net_load_kw"] = df["load_kw"] - df["pv_forecast_kw"]
    df["net_demand_kw"] = np.maximum(df["net_load_kw"], 0.0)
    df["pv_surplus_kw"] = np.maximum(-df["net_load_kw"], 0.0)

    df["load_energy_kwh"] = df["load_kw"] * d
    df["pv_forecast_energy_kwh"] = df["pv_forecast_kw"] * d
    df["net_demand_energy_kwh"] = df["net_demand_kw"] * d
    df["pv_surplus_energy_kwh"] = df["pv_surplus_kw"] * d

    df = df[BASE_COLS]
    df.to_csv(CSV_BASE_PATH, index=False, encoding="utf-8-sig")

    plot = pd.DataFrame()
    plot["interval_index"] = df["interval_index"]
    plot["minute_of_day"] = df["interval_index"].values * INTERVAL_MINUTES
    plot["interval_start"] = df["interval_start"]
    plot["interval_end"] = df["interval_end"]
    plot["price_yuan_per_kwh"] = df["price_yuan_per_kwh"]
    plot["load_kw"] = df["load_kw"]
    plot["pv_forecast_kw"] = df["pv_forecast_kw"]
    plot["net_demand_kw"] = df["net_demand_kw"]
    plot["pv_surplus_kw"] = df["pv_surplus_kw"]
    plot = plot[PLOT_COLS]
    plot.to_csv(CSV_PLOT_PATH, index=False, encoding="utf-8-sig")

    load = df["load_kw"].values
    pv = df["pv_forecast_kw"].values
    nd = df["net_demand_kw"].values
    ps = df["pv_surplus_kw"].values
    nl = df["net_load_kw"].values
    price = df["price_yuan_per_kwh"].values

    total_load_energy = float((load * d).sum())
    total_pv_energy = float((pv * d).sum())
    total_nd_energy = float((nd * d).sum())
    total_ps_energy = float((ps * d).sum())

    n_short = int((nl > 0).sum())
    n_surplus = int((nl < 0).sum())
    n_equal = int((np.abs(nl) < 1e-9).sum())

    i_min_price = int(np.argmin(price))
    i_max_price = int(np.argmax(price))

    over_idx = [i for i in range(n) if ps[i] > 5000.0]
    theoretical_charge = float((0.9 * ps * d).sum())

    lines = [
        "# 附件一数据处理报告",
        "",
        "## 数据来源", f"- 文件：`{ATTACHMENT1_PATH}`（仅第一问使用，未使用附件2）",
        "",
        "## 输出文件",
        f"- 基础数据：`{CSV_BASE_PATH.name}`（{len(df)} 行 × {len(BASE_COLS)} 列，UTF-8-SIG）",
        f"- 作图数据：`{CSV_PLOT_PATH.name}`（{len(plot)} 行 × {len(PLOT_COLS)} 列，UTF-8-SIG）",
        "",
        "## 基本统计（功率 kW；能量 kWh；时段 = 10 分钟 = 1/6 小时）",
        "| 指标 | 数值 |", "|---|---|",
        f"| 一天负荷总电量 | {total_load_energy:.4f} |",
        f"| 一天光伏预测总电量 | {total_pv_energy:.4f} |",
        f"| 一天净需求总电量 | {total_nd_energy:.4f} |",
        f"| 一天光伏剩余总电量 | {total_ps_energy:.4f} |",
        f"| 最大负荷功率 (kW) | {load.max():.4f} |",
        f"| 最小负荷功率 (kW) | {load.min():.4f} |",
        f"| 最大光伏预测功率 (kW) | {pv.max():.4f} |",
        f"| 最小光伏预测功率 (kW) | {pv.min():.4f} |",
        f"| 最大净需求功率 (kW) | {nd.max():.4f} |",
        f"| 最大光伏剩余功率 (kW) | {ps.max():.4f} |",
        f"| 光伏不足时段数 | {n_short} |",
        f"| 光伏充足时段数 | {n_surplus} |",
        f"| 光伏恰好等于负荷时段数 | {n_equal} |",
        "",
        "## 电价相关信息",
        f"- 最低电价：{price[i_min_price]:.4f} 元/kWh，第 {i_min_price+1} 时段"
        f"（标签 `{df['original_time_label'][i_min_price]}`）",
        f"- 最高电价：{price[i_max_price]:.4f} 元/kWh，第 {i_max_price+1} 时段"
        f"（标签 `{df['original_time_label'][i_max_price]}`）",
        "",
        "## 光伏剩余功率超过 5000 kW 的时段",
    ]
    if over_idx:
        for i in over_idx:
            lines.append(f"- 第 {i+1} 时段（标签 `{df['original_time_label'][i]}`）：{ps[i]:.4f} kW")
    else:
        lines.append("- 无（没有任何时段光伏剩余功率超过 5000 kW）")

    lines += [
        "",
        "## 模型口径说明（最终模型）",
        "- 最终模型**允许外部电网给储能充电**（`allow_grid_charging: true`），不限于仅用剩余光伏充电。",
        "- `pv_surplus_kw = max(pv_forecast_kw - load_kw, 0)` 仅是原始负荷与光伏的**静态差值**，",
        "  属于描述性统计字段，**不等于**最终模型的弃光功率 `W_t`。",
        "- 弃光功率 `W_t` 是优化决策变量，由功率平衡 `G_t + P_t^pv + D_t = L_t + C_t + W_t`",
        "  与储能约束、目标函数共同确定，不在数据处理阶段计算。",
        "",
        "## 光伏剩余静态统计（仅描述性，不代表实际充入或弃光）",
        f"- sum(0.9 × pv_surplus_kw × 1/6) = {theoretical_charge:.4f} kWh（仅静态统计）",
        "",
        "## 光伏出力起止",
        f"- 首个光伏>0：第 {pv_pos_idx[0]+1} 段（标签 `{pv_first}`）",
        f"- 末个光伏>0：第 {pv_pos_idx[-1]+1} 段（标签 `{pv_last}`）",
        "",
    ]
    PROCESS_MD.write_text("\n".join(lines), encoding="utf-8")

    stats = {
        "rows": int(len(df)),
        "total_load_energy": total_load_energy,
        "total_pv_energy": total_pv_energy,
        "total_nd_energy": total_nd_energy,
        "total_ps_energy": total_ps_energy,
        "n_short": n_short, "n_surplus": n_surplus, "n_equal": n_equal,
        "max_load": float(load.max()), "min_load": float(load.min()),
        "max_pv": float(pv.max()), "min_pv": float(pv.min()),
        "max_nd": float(nd.max()), "max_ps": float(ps.max()),
        "min_price": float(price[i_min_price]), "min_price_idx": i_min_price + 1,
        "max_price": float(price[i_max_price]), "max_price_idx": i_max_price + 1,
        "over_5000": len(over_idx), "over_5000_list": over_idx,
        "theoretical_charge": theoretical_charge,
    }
    print(f"[02] 处理完成：{CSV_BASE_PATH.name} 与 {CSV_PLOT_PATH.name} 已生成")
    print(f"[02] 已输出：{TIME_MD.name}, {PROCESS_MD.name}")
    return stats


if __name__ == "__main__":
    main()
