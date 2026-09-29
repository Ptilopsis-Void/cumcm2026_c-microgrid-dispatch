from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from datetime import datetime, time as dt_time
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
import openpyxl
from openpyxl.utils import get_column_letter

T = C.PERIODS_PER_DAY
ETA = C.ETA
TOLL = 1e-6
DATE_FMT = "mm-dd-yy"


def merge_emergency(b: np.ndarray) -> list:
    events = []
    t = 0
    while t < T:
        if b[t] > TOLL:
            s = t
            tot = 0.0
            while t < T and b[t] > TOLL:
                tot += float(b[t])
                t += 1
            events.append((s + 1, t, tot))
        else:
            t += 1
    return events


def seg_text(s: int, e: int) -> str:
    a = C.minutes_to_hhmm(C.period_bounds(s)[0])
    bb = C.minutes_to_hhmm(C.period_bounds(e)[1], use_24=True)
    return f"{a}-{bb}"


def validate_emergency_sheet(ws, days, date_strs, b_all):
    row = 2
    ok = True
    worst = 0.0
    for day in days:
        values = b_all[day]
        mask = values > TOLL
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
        value = ws.cell(row, 1).value
        ok &= hasattr(value, "strftime") and value.strftime("%Y-%m-%d") == date_strs[day]
        for j in range(max(3, len(starts))):
            rr = row + j
            if j:
                ok &= ws.cell(rr, 1).value is None
            if j < len(starts):
                start, end = int(starts[j]), int(ends[j])
                ok &= ws.cell(rr, 2).value == seg_text(start + 1, end)
                val = ws.cell(rr, 3).value
                ok &= isinstance(val, (int, float))
                if isinstance(val, (int, float)):
                    worst = max(worst, abs(val - values[start:end].sum()))
            else:
                ok &= ws.cell(rr, 2).value in (None, "") and ws.cell(rr, 3).value is None
        row += max(3, len(starts))
    return bool(ok and worst < 1e-6), row - 1, worst


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg, flush=True)

    t_all = time.perf_counter()
    p("=" * 74)
    p("第二问 10 —— 生成并回读校验 提交结果/result2.xlsx")
    p("=" * 74)

    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Pf = np.load(C.RESULT_DIR / "第二问_固定计划.npz", allow_pickle=False)
    BT = np.load(C.RESULT_DIR / "第二问_执行器回测.npz", allow_pickle=False)
    price = np.asarray(Z["price"], float)
    date_strs = [str(x) for x in Z["dates"]]
    score_idx = np.asarray(Z["score_day_index"], int)

    g_all = np.asarray(BT["plan_g"], float)
    b_all = np.asarray(BT["dp_b"], float)
    C_all = np.asarray(BT["dp_C"], float)
    D_all = np.asarray(BT["dp_D"], float)
    E_all = np.asarray(BT["dp_E"], float)

    E_start = np.zeros(365)
    prev = float(C.E_INIT)
    for d in score_idx:
        E_start[d] = prev
        prev = float(E_all[d, -1])

    days = [int(d) for d in score_idx]
    day_dates = [date_strs[d] for d in days]
    p(f"数据源：第二问_执行器回测.npz（DP 主方案）；提交 {len(days)} 天："
      f"{day_dates[0]} … {day_dates[-1]}")

    C.SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.SUBMIT_DIR / "result2_最终核验候选.xlsx"
    shutil.copyfile(C.ATTACHMENT5_PATH, out_path)
    p(f"已复制模板：{C.ATTACHMENT5_PATH.name} → {out_path}（最终核验候选，不覆盖正式 result2.xlsx）")

    wb = openpyxl.load_workbook(out_path)
    p(f"工作表：{wb.sheetnames}")

    ws1 = wb[C.SHEET_PLAN]
    hdr1 = [ws1.cell(1, c).value for c in range(1, ws1.max_column + 1)]
    ncol = len(hdr1)
    p("")
    p(f"── 工作表『{C.SHEET_PLAN}』：{ws1.max_row} 行 × {ncol} 列（表头保持原样）──")
    p(f"  表头首列 = {hdr1[0]!r}；第 2 列 = {hdr1[1]!r}；"
      f"第 {ncol - 2} 列 = {hdr1[ncol - 3]!r}；末两列 = {hdr1[-2]!r}, {hdr1[-1]!r}")
    assert ncol == T + 3, f"意外的列数：{ncol}"
    ep_col, eq_col = T + 2, T + 3

    for i, d in enumerate(days):
        row = 2 + i
        av = ws1.cell(row, 1).value
        av_s = av.strftime("%Y-%m-%d") if hasattr(av, "strftime") else str(av)
        if av_s[:10] != day_dates[i]:
            p(f"  ⚠ 第 {row} 行日期不符：模板 {av_s} vs 预期 {day_dates[i]}，已覆盖写入。")
            ws1.cell(row, 1).value = datetime.strptime(day_dates[i], "%Y-%m-%d")
            ws1.cell(row, 1).number_format = DATE_FMT
        for t in range(T):
            ws1.cell(row, 2 + t).value = float(g_all[d, t])
        ws1.cell(row, ep_col).value = float(g_all[d].sum())
        ws1.cell(row, eq_col).value = float(price @ g_all[d])
    p(f"  已填写第 2..{1 + len(days)} 行的列 B..{get_column_letter(T + 1)}"
      f"（{T} 个时段）与 {get_column_letter(ep_col)}(全天购电量)、"
      f"{get_column_letter(eq_col)}(全天购电费)")

    ws2 = wb[C.SHEET_BATT]
    hdr2 = [ws2.cell(1, c).value for c in range(1, 7)]
    p("")
    p(f"── 工作表『{C.SHEET_BATT}』：重建 334 个 6 行区块（表头 {hdr2}）──")
    if ws2.max_row >= 2:
        ws2.delete_rows(2, ws2.max_row - 1)
    r = 2
    for i, d in enumerate(days):
        base = r + 6 * i
        dtv = datetime.strptime(day_dates[i], "%Y-%m-%d")
        for j in range(6):
            rr = base + j
            ws2.cell(rr, 2).value = C.BATT_BLOCKS[j]
            a, bb = j * 24, (j + 1) * 24
            ws2.cell(rr, 3).value = float(C_all[d, a:bb].sum())
            ws2.cell(rr, 4).value = float(D_all[d, a:bb].sum())
            if j == 0:
                c = ws2.cell(rr, 1)
                c.value = dtv
                c.number_format = DATE_FMT
                c = ws2.cell(rr, 5)
                c.value = dt_time(0, 0)
                c.number_format = "h:mm"
                ws2.cell(rr, 6).value = float(E_start[d])
            elif j == 1:
                ws2.cell(rr, 5).value = "24:00"
                ws2.cell(rr, 6).value = float(E_all[d, -1])
    p(f"  已写入第 2..{1 + 6 * len(days)} 行（{len(days)} 天 × 6 段）")
    p(f"  列 E/F 仅在每个区块前两行填 时刻 与 储电量（0:00 / 24:00）")

    ws3 = wb[C.SHEET_EMERG]
    hdr3 = [ws3.cell(1, c).value for c in range(1, 4)]
    p("")
    p(f"── 工作表『{C.SHEET_EMERG}』：重建 334 个可扩展日期区块（表头 {hdr3}）──")
    if ws3.max_row >= 2:
        ws3.delete_rows(2, ws3.max_row - 1)
    r = 2
    n_ev_raw = 0
    n_expanded = 0
    for i, d in enumerate(days):
        raw = merge_emergency(b_all[d])
        n_ev_raw += len(raw)
        ev = raw
        block_rows = max(3, len(ev))
        if len(raw) > 3:
            n_expanded += 1
        dtv = datetime.strptime(day_dates[i], "%Y-%m-%d")
        for j in range(block_rows):
            rr = r + j
            if j == 0:
                c = ws3.cell(rr, 1)
                c.value = dtv
                c.number_format = DATE_FMT
            if j < len(ev):
                s, e, tot = ev[j]
                ws3.cell(rr, 2).value = seg_text(int(round(s)), int(round(e)))
                ws3.cell(rr, 3).value = float(tot)
            else:
                ws3.cell(rr, 2).value = ""
        r += block_rows
    p(f"  已写入第 2..{r - 1} 行（{len(days)} 天，每天至少 3 行，事件多时扩展）")
    p(f"  原始事件合计 {n_ev_raw} 起；因原始事件 > 3 起而扩展行数的天数：{n_expanded}")
    wb.save(out_path)
    p(f"已保存：{out_path}")

    p("")
    p("── 回读校验 ──")
    wb2 = openpyxl.load_workbook(out_path, data_only=True)
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        p(f"  {'✔' if ok else '✘'} {name}{('：' + detail) if detail else ''}")

    wsA = wb2[C.SHEET_PLAN]
    chk("计划购电量 行数 = 335", wsA.max_row == 335, f"实际 {wsA.max_row}")
    chk("计划购电量 列数 = 147", wsA.max_column == 147, f"实际 {wsA.max_column}")
    chk("计划购电量 表头首列/末两列不变",
        wsA.cell(1, 1).value == hdr1[0] and wsA.cell(1, T + 2).value == hdr1[-2]
        and wsA.cell(1, T + 3).value == hdr1[-1])
    ok_date = True
    for i, dtxt in enumerate(day_dates):
        v = wsA.cell(2 + i, 1).value
        if not hasattr(v, "strftime") or v.strftime("%Y-%m-%d") != dtxt:
            ok_date = False
            break
    chk("计划购电量 列 A 日期 = 2025-02-01..2025-12-31 连续", ok_date)

    worst_g = 0.0
    worst_ep = 0.0
    worst_eq = 0.0
    for i, d in enumerate(days):
        row = 2 + i
        for t in range(T):
            worst_g = max(worst_g, abs(float(wsA.cell(row, 2 + t).value) - g_all[d, t]))
        worst_ep = max(worst_ep, abs(float(wsA.cell(row, T + 2).value) - g_all[d].sum()))
        worst_eq = max(worst_eq, abs(float(wsA.cell(row, T + 3).value) - float(price @ g_all[d])))
    chk("计划购电量 144 时段逐格与模型一致", worst_g < 1e-6, f"最大偏差 {worst_g:.3e}")
    chk("计划购电量 全天购电量列自洽", worst_ep < 1e-6, f"最大偏差 {worst_ep:.3e}")
    chk("计划购电量 全天购电费列自洽", worst_eq < 1e-6, f"最大偏差 {worst_eq:.3e}")

    wsA_2 = wb2[C.SHEET_BATT]
    chk("充放电量 行数 = 1 + 334×6", wsA_2.max_row == 1 + 6 * len(days),
        f"实际 {wsA_2.max_row}")
    ok_blk = True
    worst_batt = 0.0
    for i, d in enumerate(days):
        base = 2 + 6 * i
        for j in range(6):
            rr = base + j
            if wsA_2.cell(rr, 2).value != C.BATT_BLOCKS[j]:
                ok_blk = False
            a, bb = j * 24, (j + 1) * 24
            worst_batt = max(worst_batt,
                             abs(float(wsA_2.cell(rr, 3).value) - C_all[d, a:bb].sum()),
                             abs(float(wsA_2.cell(rr, 4).value) - D_all[d, a:bb].sum()))
    chk("充放电量 六段标签与行序正确", ok_blk)
    chk("充放电量 充电/放电量逐格与模型一致", worst_batt < 1e-6,
        f"最大偏差 {worst_batt:.3e}")
    ok_ef = True
    worst_soc = 0.0
    for i, d in enumerate(days):
        base = 2 + 6 * i
        e0 = wsA_2.cell(base, 6).value
        e1 = wsA_2.cell(base + 1, 6).value
        e0 = float(e0) if e0 is not None else np.nan
        e1 = float(e1) if e1 is not None else np.nan
        worst_soc = max(worst_soc, abs(e0 - E_start[d]), abs(e1 - E_all[d, -1]))
        if wsA_2.cell(base + 1, 5).value != "24:00":
            ok_ef = False
    chk("充放电量 列 E 的 0:00 / 24:00 标签与列 F 储电量自洽", ok_ef and worst_soc < 1e-6,
        f"储电量最大偏差 {worst_soc:.3e}")

    worst_rec = 0.0
    for i, d in enumerate(days):
        worst_rec = max(worst_rec,
                        abs((ETA * C_all[d].sum() - D_all[d].sum() / ETA)
                            - (E_all[d, -1] - E_start[d])))
    chk("储能 SOC 递推自洽（ηΣC − ΣD/η = ΔE）", worst_rec < 1e-6,
        f"最大残差 {worst_rec:.3e}")

    wsA_3 = wb2[C.SHEET_EMERG]
    event_ok, expected_rows, event_error = validate_emergency_sheet(wsA_3, days, date_strs, b_all)
    chk("紧急购电量 行数与实际事件数一致", wsA_3.max_row == expected_rows,
        f"实际 {wsA_3.max_row}，预期 {expected_rows}")
    chk("紧急购电量 每日日期、事件边界及逐事件电量一致", event_ok,
        f"逐事件最大电量偏差 {event_error:.3e}")
    tot_ev = sum(float(b_all[d].sum()) for d in days)
    tot_in = 0.0
    for rr in range(2, wsA_3.max_row + 1):
        v = wsA_3.cell(rr, 3).value
        if isinstance(v, (int, float)):
            tot_in += float(v)
    chk("紧急购电量 求和 = 模型紧急量合计", abs(tot_in - tot_ev) < 1e-4,
        f"表内 {tot_in:.6f} vs 模型 {tot_ev:.6f}")

    cmp_path = C.RESULT_DIR / "第二问_执行器对照表.csv"
    if cmp_path.exists():
        import csv
        with open(cmp_path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("比较方式") == "各自重订" and row.get("执行器") == "DP价值执行器":
                    ref_g = float(row["计划购电量_kWh"])
                    ref_e = float(row["紧急量_kWh"])
                    got_g = sum(float(wsA.cell(2 + i, T + 2).value) for i in range(len(days)))
                    chk("与 07 对照表口径一致（计划购电量）",
                        abs(got_g - ref_g) < 1e-3,
                        f"表内 {got_g:.4f} vs 对照 {ref_g:.4f} kWh")
                    chk("与 07 对照表口径一致（紧急量）",
                        abs(tot_in - ref_e) < 1e-4,
                        f"表内 {tot_in:.4f} vs 对照 {ref_e:.4f} kWh")
                    break
    else:
        p(f"  （未找到 {cmp_path.name}，跳过与对照表的一致性核对）")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    p("")
    p(f"校验结果：{len(checks) - n_fail}/{len(checks)} 通过"
      f"{'，全部通过 ✔' if n_fail == 0 else f'，失败 {n_fail} 项 ✘'}")
    p(f"提交文件：{out_path}（{out_path.stat().st_size / 1024:.1f} KiB）")
    p(f"总用时 {time.perf_counter() - t_all:.1f} s")

    C.write_text_utf8(C.SOLVE_LOG_DIR / "第二问_10结果生成日志.txt",
                      "\n".join(log + ["", "[10 完成] result2_最终核验候选.xlsx 生成与回读校验结束。"]))
    C.write_text_utf8(C.REPORT_DIR / "第二问_result2生成与校验报告.md",
                      "\n".join(["# 第二问 result2_最终核验候选.xlsx 生成与回读校验报告", ""] + log + [""]))
    p("已保存：求解日志/第二问_10结果生成日志.txt、"
      "报告/第二问_result2生成与校验报告.md")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
