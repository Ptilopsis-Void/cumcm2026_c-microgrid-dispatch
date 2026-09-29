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

import openpyxl
import numpy as np

SHEET_LOAD = "小区负载"
SHEET_PV = "光伏发电实际功率"

EXPECT_DAYS = 365
EXPECT_COLS = 144


def read_sheet(path: Path, sheet: str):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = list(rows[0])
    labels = [C.label_to_text(v) for v in header[1:1 + EXPECT_COLS]]
    dates, data, anomalies = [], [], []
    for i, r in enumerate(rows[1:], start=2):
        if r[0] is None:
            continue
        dates.append(r[0])
        vals = np.full(EXPECT_COLS, np.nan, dtype=float)
        for j in range(EXPECT_COLS):
            v = r[1 + j] if 1 + j < len(r) else None
            if v is None or (isinstance(v, str) and not v.strip()):
                anomalies.append((sheet, i, j + 1, "空值", "", ""))
                continue
            try:
                vals[j] = float(v)
            except (TypeError, ValueError):
                anomalies.append((sheet, i, j + 1, "非数值", repr(v), ""))
        data.append(vals)
    return labels, dates, np.vstack(data), anomalies


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 74)
    p("第二问 01 —— 附件二原始数据体检查")
    p("=" * 74)
    p(f"原始文件：{C.ATTACHMENT2_PATH}")
    p(f"文件哈希：{C.compute_sha256(C.ATTACHMENT2_PATH)}")

    wb = openpyxl.load_workbook(C.ATTACHMENT2_PATH, read_only=True, data_only=True)
    sheet_names = list(wb.sheetnames)
    dims = {}
    for sh in sheet_names:
        ws = wb[sh]
        dims[sh] = (ws.max_row, ws.max_column)
    wb.close()
    p(f"工作表：{sheet_names}")

    struct_rows, stat_rows, anomaly_rows = [], [], []

    for sh in sheet_names:
        mr, mc = dims[sh]
        struct_rows.append((sh, mr, mc, mr - 1, mc - 1, "日期+144个时段"))
        p(f"  [{sh}] max_row={mr} max_col={mc} → 数据行={mr - 1}，时段列={mc - 1}")

    labels_ref = None
    dates_ref = None
    for sh in sheet_names:
        labels, dates, arr, anomal = read_sheet(C.ATTACHMENT2_PATH, sh)
        anomaly_rows.extend(anomal)
        if labels_ref is None:
            labels_ref, dates_ref = labels, dates
        else:
            assert labels == labels_ref, f"{sh} 表头标签与首表不一致"

        n_day, n_col = arr.shape
        nan_cnt = int(np.isnan(arr).sum())
        neg_cnt = int((arr < 0).sum())
        zero_cnt = int((arr == 0).sum())
        p(f"  [{sh}] 形状={arr.shape} 缺失={nan_cnt} 负值={neg_cnt} 零值={zero_cnt}")
        p(f"          负载/功率  min={np.nanmin(arr):.6f}  max={np.nanmax(arr):.6f}  "
          f"mean={np.nanmean(arr):.6f}  sd={np.nanstd(arr):.6f}")
        p(f"          日电量      min={np.nanmin(arr.sum(1)) * C.DELTA_HOURS:.6f}  "
          f"max={np.nanmax(arr.sum(1)) * C.DELTA_HOURS:.6f}  "
          f"全天合计={np.nansum(arr) * C.DELTA_HOURS:.6f} kWh")
        stat_rows.append((
            sh, n_day, n_col, nan_cnt, neg_cnt, zero_cnt,
            f"{np.nanmin(arr):.6f}", f"{np.nanmax(arr):.6f}", f"{np.nanmean(arr):.6f}",
            f"{np.nanstd(arr):.6f}",
            f"{np.nanmean(arr.sum(1)) * C.DELTA_HOURS:.6f}",
            f"{np.nansum(arr) * C.DELTA_HOURS:.6f}",
        ))

    expect_labels = [C.minutes_to_hhmm((t + 1) * C.INTERVAL_MINUTES, use_24=False)
                     for t in range(C.PERIODS_PER_DAY)]
    expect_labels[-1] = "23:50"
    norm = [("23:50" if s in ("0:00+1", "24:00") else s) for s in labels_ref]
    ok_labels = norm == expect_labels
    p("")
    p(f"时间标签序列严格匹配 00:10…23:50 末列 '0:00+1'：{'✔ 是' if ok_labels else '✘ 否'}")
    if not ok_labels:
        for a, b in zip(norm, expect_labels):
            if a != b:
                p(f"   首个不一致：实际 {a!r} vs 期望 {b!r}")
                break

    from datetime import date, timedelta
    d0, d1 = dates_ref[0], dates_ref[-1]
    d0 = d0.date() if hasattr(d0, "date") else d0
    d1 = d1.date() if hasattr(d1, "date") else d1
    expect_dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    ok_dates = expect_dates == [(x.date() if hasattr(x, "date") else x) for x in dates_ref]
    p(f"日期行数={len(dates_ref)}  首={d0}  末={d1}")
    p(f"日期严格连续且无重复：{'✔ 是' if ok_dates else '✘ 否'}")
    p(f"天数是否为 365：{'✔ 是' if len(dates_ref) == EXPECT_DAYS else '✘ 否'}")

    C.write_csv_utf8_sig(C.RAW_STRUCT_CSV,
                         ("工作表", "max_row", "max_col", "数据行数", "时段列数", "说明"),
                         struct_rows)
    C.write_csv_utf8_sig(C.RAW_STAT_CSV,
                         ("工作表", "天数", "时段列数", "缺失数", "负值数", "零值数",
                          "行/列最小值", "行/列最大值", "均值", "标准差",
                          "日电量均值_kWh", "全年电量合计_kWh"),
                         stat_rows)
    C.write_csv_utf8_sig(C.RAW_MAPPING_CSV,
                         ("原始列", "原始表头", "解析含义", "本问字段", "单位", "备注"),
                         [
                             ("A", "日期\\时间", "日期", "date", "—", "2025-01-01 ~ 2025-12-31"),
                             ("B..EO", "0:10 … 0:00+1", "10 分钟区间终点标签",
                              "interval_index 1..144", "—",
                              "标签为区间终点；第 N 列对应区间 [(N-1)×10min, N×10min)"),
                             ("B..EO", "—", "功率", "load_kw / pv_kw", "kW", "附件二两张表分别给出"),
                         ])
    C.write_csv_utf8_sig(C.ANOMALY_CSV,
                         ("工作表", "Excel行号", "时段列序号", "异常类型", "原始值", "处理方式"),
                         anomaly_rows or [("—", "", "", "无异常", "", "")])

    time_md = [
        f"# 附件二时间标签映射说明（第二问）",
        "",
        f"- 原始文件：`{C.ATTACHMENT2_PATH.name}`，SHA-256：`{C.compute_sha256(C.ATTACHMENT2_PATH)}`",
        f"- 时间标签口径：**{C.TIMESTAMP_INTERPRETATION}**",
        f"- 全天时段数：{C.PERIODS_PER_DAY}，每时段 {C.INTERVAL_MINUTES} 分钟，"
        f"Δt = {C.DELTA_HOURS:.12f} h",
        "",
        "## 1. 标签 → 区间映射",
        "",
        "| 时段序号 t | 原始标签（第 t 列） | 实际区间 | 起点/终点（分钟） |",
        "| --- | --- | --- | --- |",
    ]
    demo = list(range(1, 4)) + [142, 143, 144]
    for t in demo:
        lab = labels_ref[t - 1]
        s, e = C.period_bounds(t)
        time_md.append(
            f"| {t} | `{lab}` | [{C.minutes_to_hhmm(s)}, {C.minutes_to_hhmm(e)}) | {s}/{e} |")
    time_md += [
        "",
        "末列标签写作 `0:00+1`，表示次日 0:00，即区间 [23:50, 24:00)。",
        "",
        "## 2. 关键结论",
        "",
        "1. 第 N 个数据列（B..EO，N=1..144）对应区间 `[(N-1)×10min, N×10min)`；",
        "2. 原文标签比「起点式」标签早一格，故不能直接照抄列名标签作为起点；",
        "3. 该口径与附件一完全一致，已由「ν = 0.481548 元/kWh」反推验证（见 README §6）。",
        "",
    ]
    C.write_text_utf8(C.TIME_MD, "\n".join(time_md))

    n_anom = len(anomaly_rows)
    p("")
    p(f"异常记录条数：{n_anom}")
    p(f"输出：{C.RAW_STRUCT_CSV.name} / {C.RAW_STAT_CSV.name} / {C.RAW_MAPPING_CSV.name} / "
      f"{C.ANOMALY_CSV.name} / {C.TIME_MD.name}")

    C.write_text_utf8(C.LOG_TXT,
                      "\n".join(log + ["", "[01 完成] 附件二原始数据体检结束。"]))

    return 0 if (n_anom == 0 and ok_labels and ok_dates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
