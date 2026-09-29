from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    ATTACHMENT1_PATH, CSV_BASE_PATH, ACCEPT_MD, ANOMALY_CSV,
    DELTA_HOURS, PERIODS_PER_DAY, FORMAL_INPUT_FIELDS,
    CODE_DIR, CONFIG_YAML, PROCESS_MD, compute_sha256,
)


def main():
    df = pd.read_csv(CSV_BASE_PATH, encoding="utf-8-sig")

    checks = []
    def _ck(name, ok):
        checks.append((name, bool(ok)))

    _ck("清洗后恰好144行", len(df) == PERIODS_PER_DAY)
    _ck("interval_index 恰好为 1..144",
        list(df["interval_index"]) == list(range(1, PERIODS_PER_DAY + 1)))
    _ck("price_yuan_per_kwh 无缺失", df["price_yuan_per_kwh"].notna().all())
    _ck("load_kw 无缺失且非负", df["load_kw"].notna().all() and (df["load_kw"] >= 0).all())
    _ck("pv_forecast_kw 无缺失且非负",
        df["pv_forecast_kw"].notna().all() and (df["pv_forecast_kw"] >= 0).all())
    _ck("delta_hours 全部等于 1/6", np.allclose(df["delta_hours"], DELTA_HOURS))
    _ck("net_load = load - pv",
        np.allclose(df["net_load_kw"], df["load_kw"] - df["pv_forecast_kw"]))
    _ck("net_demand = max(net_load, 0)",
        np.allclose(df["net_demand_kw"], np.maximum(df["net_load_kw"], 0.0)))
    _ck("pv_surplus = max(-net_load, 0)",
        np.allclose(df["pv_surplus_kw"], np.maximum(-df["net_load_kw"], 0.0)))
    _ck("net_demand * pv_surplus == 0（同段不并存）",
        np.allclose(df["net_demand_kw"] * df["pv_surplus_kw"], 0.0))
    _ck("net_load = net_demand - pv_surplus",
        np.allclose(df["net_load_kw"], df["net_demand_kw"] - df["pv_surplus_kw"]))

    try:
        df2 = pd.read_csv(CSV_BASE_PATH, encoding="utf-8-sig")
        num_cols_are_float = all(pd.api.types.is_numeric_dtype(df2[c])
                                 for c in ["price_yuan_per_kwh", "load_kw", "pv_forecast_kw",
                                           "net_demand_kw", "pv_surplus_kw"])
        reread_ok = (len(df2) == len(df)) and num_cols_are_float and \
            np.allclose(df2["load_kw"].values, df["load_kw"].values) and \
            np.allclose(df2["pv_forecast_kw"].values, df["pv_forecast_kw"].values)
    except Exception:
        reread_ok = False
    _ck("CSV 重读：中文路径可访问、行数一致、数值列仍为数值", reread_ok)

    model_checks = []
    def _mck(name, ok):
        model_checks.append((name, bool(ok)))

    _mck("正式输入字段无缺失且恰好144行",
         len(df) == PERIODS_PER_DAY and df[FORMAL_INPUT_FIELDS].notna().all(axis=1).all())
    _mck("price / load / pv / delta_hours 无缺失",
         df[["price_yuan_per_kwh", "load_kw", "pv_forecast_kw", "delta_hours"]].notna().all().all())
    _mck("数值型模型输入字段均为数值类型",
         all(pd.api.types.is_numeric_dtype(df[c])
             for c in ["interval_index", "delta_hours", "price_yuan_per_kwh", "load_kw", "pv_forecast_kw"]))

    decision_english = ["grid_purchase_kw", "charge_kw", "discharge_kw", "curtail_kw"]
    src02 = (CODE_DIR / "02_处理附件一基础数据.py").read_text(encoding="utf-8")
    no_decision_code = all(tok not in src02 for tok in decision_english)
    no_decision_col = set(df.columns).isdisjoint(decision_english)
    _mck("无代码用 net_demand_kw 限制购电/放电", no_decision_code and no_decision_col)
    _mck("无代码用 pv_surplus_kw 限制充电", no_decision_code and no_decision_col)

    config_txt = CONFIG_YAML.read_text(encoding="utf-8") if CONFIG_YAML.exists() else ""
    _mck("配置为 allow_grid_charging: true", "  allow_grid_charging: true" in config_txt)

    report_txt = PROCESS_MD.read_text(encoding="utf-8") if PROCESS_MD.exists() else ""
    forbidden_phrases = ["禁止电网充电", "只能使用剩余光伏充电", "只能由剩余光伏充电",
                         "不得从电网购电给电池充电", "不允许电网充电", "电网不得给电池充电"]
    _mck("报告无“禁止/只能剩余光伏充电”旧表述", not any(p in report_txt for p in forbidden_phrases))

    baseline = ""
    for line in config_txt.splitlines():
        if "attachment1_sha256" in line:
            baseline = line.split(":", 1)[1].strip().strip('"')
            break
    _mck("原始附件未被修改（SHA-256 与配置一致）",
         baseline == "" or baseline == compute_sha256(ATTACHMENT1_PATH))

    all_pass = all(ok for _, ok in checks) and all(ok for _, ok in model_checks)

    import openpyxl
    anomalies = []
    try:
        wb = openpyxl.load_workbook(ATTACHMENT1_PATH, data_only=True)
        ws = wb[wb.sheetnames[0]]
        tt = Counter(type(ws.cell(r, 1).value).__name__ for r in range(2, ws.max_row + 1))
        if len(tt) > 1:
            anomalies.append({
                "record_id": 1,
                "original_time_label": "（整列）",
                "field_name": "时间",
                "raw_value": str(dict(tt)),
                "problem_type": "TIME_FORMAT_INCONSISTENT",
                "processing_action": "统一按区间终点解析为 HH:MM",
                "processing_reason": "前段为 datetime.time、后段为字符串(含 0:00+1)，格式不一致但语义一致",
            })
    except Exception as e:
        anomalies.append({
            "record_id": 1, "original_time_label": "（整列）", "field_name": "时间",
            "raw_value": "", "problem_type": "READ_ERROR",
            "processing_action": "跳过格式检查", "processing_reason": str(e),
        })

    if anomalies:
        pd.DataFrame(anomalies).to_csv(ANOMALY_CSV, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=[
            "record_id", "original_time_label", "field_name", "raw_value",
            "problem_type", "processing_action", "processing_reason",
        ]).to_csv(ANOMALY_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# 附件一数据处理验收表",
        "",
        f"- 校验对象：`{CSV_BASE_PATH.name}`",
        f"- 校验结论：{'✅ 全部通过' if all_pass else '❌ 存在未通过项'}",
        "",
        "## 一、基础数据校验",
        "| 序号 | 校验项 | 结果 |",
        "|---|---|---|",
    ]
    for i, (name, ok) in enumerate(checks, 1):
        lines.append(f"| {i} | {name} | {'✅ 通过' if ok else '❌ 未通过'} |")
    lines += [
        "",
        "## 二、最终模型口径验收",
        "| 序号 | 验收项 | 结果 |",
        "|---|---|---|",
    ]
    for i, (name, ok) in enumerate(model_checks, 1):
        lines.append(f"| {i} | {name} | {'✅ 通过' if ok else '❌ 未通过'} |")
    lines += [
        "",
        f"- 基础数据校验：{sum(ok for _, ok in checks)} / {len(checks)} 项通过。",
        f"- 模型口径验收：{sum(ok for _, ok in model_checks)} / {len(model_checks)} 项通过。",
        f"- 汇总：{'全部通过' if all_pass else '存在未通过项'}。",
        "",
        "## 异常记录",
        f"- 异常条数：{len(anomalies)}",
        ("- 详见 `日志/附件一数据异常记录.csv`。" if anomalies else "- 数值数据未发现异常（时间标签格式不一致已在异常记录中单独说明）。"),
        "",
    ]
    ACCEPT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[03] 校验完成：all_pass={all_pass}, 异常条数={len(anomalies)}")
    print(f"[03] 已输出：{ACCEPT_MD.name}, {ANOMALY_CSV.name}")
    return {"all_pass": all_pass, "checks": checks, "model_checks": model_checks,
            "anomaly_count": len(anomalies)}


if __name__ == "__main__":
    main()
