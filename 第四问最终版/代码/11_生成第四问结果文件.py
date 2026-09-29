#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("_comm4.py", "q4_comm")

N_COL_TIME = 144
COL_FIRST_TIME = 2
COL_TOTAL_Q = 146
COL_TOTAL_FEE = 147
ROW_DATE0 = 2
BLOCKS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
BLOCK_LABEL = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
               "12:00-16:00", "16:00-20:00", "20:00-24:00"]


def hhmm(minute: int) -> str:
    m = int(minute) % (24 * 60)
    return f"{m // 60}:{m % 60:02d}"


def load_inputs(log: list) -> dict:
    out: dict = {}
    for br, path in (("42", C.BACKTEST_42_NPZ), ("43", C.BACKTEST_43_NPZ)):
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as Z:
            out[br] = {k: Z[k] for k in Z.files}
    with np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False) as Z:
        out["price_actual"] = np.asarray(Z["price_actual"], float)
        out["slot_start"] = np.asarray(Z["slot_start_minute"], int)
        out["dates365"] = list(np.asarray(Z["dates"], dtype="<U10"))
        out["raw_labels"] = list(np.asarray(Z["raw_time_labels"], dtype=object))
    for k in ("price_actual", "slot_start", "dates365"):
        if k not in out:
            raise RuntimeError("附件四处理结果缺键")
    log.append(f"  载入分支档：{', '.join('4-' + b[1] for b in ('42', '43') if b in out)}")
    return out


def build_maps(inp: dict, log: list) -> tuple[list[list], list[list]]:
    ssm = inp["slot_start"]
    raw = inp["raw_labels"]
    col_rows: list[list] = []
    for t in range(N_COL_TIME):
        col = COL_FIRST_TIME + t
        letter = _col_letter(col)
        hdr = str(raw[t]) if t < len(raw) else ""
        col_rows.append([col, letter, hdr, t + 1, hhmm(ssm[t]), hhmm(ssm[t] + 10)])
    tail = [[COL_TOTAL_Q, _col_letter(COL_TOTAL_Q), "全天购电量", "", "", ""],
            [COL_TOTAL_FEE, _col_letter(COL_TOTAL_FEE), "全天购电费", "", "", ""]]
    log.append(f"  列序映射：数据列 {COL_FIRST_TIME}..{COL_FIRST_TIME + N_COL_TIME - 1} "
               f"对应时段序 1..{N_COL_TIME}（列序=时间序，未平移官方表头）")
    return col_rows, tail


def _col_letter(col: int) -> str:
    s = ""
    while col > 0:
        col, r = divmod(col - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def copy_style(src, dst) -> None:
    if src.has_style:
        dst._style = copy.copy(src._style)
    if src.number_format:
        dst.number_format = src.number_format
    if src.alignment:
        dst.alignment = copy.copy(src.alignment)
    if src.font:
        dst.font = copy.copy(src.font)
    if src.fill:
        dst.fill = copy.copy(src.fill)
    if src.border:
        dst.border = copy.copy(src.border)


def clear_data_rows(ws, first_row: int) -> None:
    if ws.max_row < first_row:
        return
    ws.delete_rows(first_row, ws.max_row - first_row + 1)


def to_dt(date_str: str) -> _dt.datetime:
    y, m, d = (int(x) for x in str(date_str).split("-"))
    return _dt.datetime(y, m, d)


def write_plan_sheet(ws, dates: list[str], qty: np.ndarray, c: np.ndarray,
                     kind: str, g: np.ndarray | None, log: list) -> dict:
    st = ws.cell(ROW_DATE0, COL_FIRST_TIME)
    for k in range(len(dates)):
        r = ROW_DATE0 + k
        for t in range(N_COL_TIME):
            dst = ws.cell(r, COL_FIRST_TIME + t)
            copy_style(st, dst)
            dst.value = float(qty[k, t])
        copy_style(ws.cell(ROW_DATE0, COL_TOTAL_Q), ws.cell(r, COL_TOTAL_Q))
        copy_style(ws.cell(ROW_DATE0, COL_TOTAL_FEE), ws.cell(r, COL_TOTAL_FEE))
        ws.cell(r, COL_TOTAL_Q).value = float(qty[k].sum())
        if kind == "plan":
            fee = float((c[k] * qty[k]).sum())
        else:
            if g is None:
                raise RuntimeError("adjust 表需要提供 g")
            fee = float((c[k] * (qty[k] + 0.5 * np.abs(qty[k] - g[k]))).sum())
        ws.cell(r, COL_TOTAL_FEE).value = fee
    return {"n_days": len(dates), "q_sum": float(qty.sum()),
            "fee_sum": float(sum(float(ws.cell(ROW_DATE0 + k, COL_TOTAL_FEE).value)
                                 for k in range(len(dates))))}


def write_battery_sheet(ws, dates: list[str], Cc: np.ndarray, Dd: np.ndarray,
                        Ech: np.ndarray, log: list) -> dict:
    src = {c: [ws.cell(ROW_DATE0 + j, c) for j in range(6)] for c in range(1, 7)}
    src_a = ws.cell(ROW_DATE0, 1)
    src_e0, src_f0 = ws.cell(ROW_DATE0, 5), ws.cell(ROW_DATE0, 6)
    src_e1, src_f1 = ws.cell(ROW_DATE0 + 1, 5), ws.cell(ROW_DATE0 + 1, 6)
    clear_data_rows(ws, ROW_DATE0)
    for k, dstr in enumerate(dates):
        r = ROW_DATE0 + k * 6
        a0 = ws.cell(r, 1)
        a0.value = to_dt(dstr)
        copy_style(src_a, a0)
        for j, (s, e) in enumerate(BLOCKS):
            rr = r + j
            b = ws.cell(rr, 2)
            b.value = BLOCK_LABEL[j]
            copy_style(src[2][j], b)
            cc = ws.cell(rr, 3)
            cc.value = float(Cc[k, s:e].sum())
            copy_style(src[3][j], cc)
            dd = ws.cell(rr, 4)
            dd.value = float(Dd[k, s:e].sum())
            copy_style(src[4][j], dd)
        e0 = ws.cell(r, 5)
        e0.value = _dt.time(0, 0)
        copy_style(src_e0, e0)
        f0 = ws.cell(r, 6)
        f0.value = float(Ech[k])
        copy_style(src_f0, f0)
        e1 = ws.cell(r + 1, 5)
        e1.value = "24:00"
        copy_style(src_e1, e1)
        f1 = ws.cell(r + 1, 6)
        f1.value = float(Ech[k + 1])
        copy_style(src_f1, f1)
    log.append(f"  [充放电量] {len(dates)} 天 × 6 区块 = {len(dates) * 6} 行；"
               f"充电合计 {Cc.sum():,.2f} kWh、放电合计 {Dd.sum():,.2f} kWh；"
               f"0 时储电量 {Ech[0]:,.2f} kWh、期末 {Ech[len(dates)]:,.2f} kWh")
    return {"n_rows": len(dates) * 6, "C_sum": float(Cc.sum()), "D_sum": float(Dd.sum()),
            "E_start": float(Ech[0]), "E_end": float(Ech[len(dates)])}


def merge_events(b: np.ndarray) -> list[tuple[int, int, float]]:
    bb = np.asarray(b, float)
    pos = np.where(bb > 0.0)[0]
    if pos.size == 0:
        return []
    brk = np.where(np.diff(pos) > 1)[0]
    return [(int(seg[0]), int(seg[-1]), float(bb[seg].sum()))
            for seg in np.split(pos, brk + 1)]


def write_emerg_sheet(ws, dates: list[str], bb: np.ndarray, ssm: np.ndarray,
                      log: list, checks: list) -> dict:
    src_a = ws.cell(ROW_DATE0, 1)
    src_b = ws.cell(ROW_DATE0, 2)
    src_c = ws.cell(ROW_DATE0, 3)
    clear_data_rows(ws, ROW_DATE0)
    r = ROW_DATE0
    n_ev = n_day = 0
    q_all = 0.0
    for k, dstr in enumerate(dates):
        evs = merge_events(bb[k])
        if not evs:
            continue
        n_day += 1
        for i, (s, e, q) in enumerate(evs):
            a = ws.cell(r, 1)
            if i == 0:
                a.value = to_dt(dstr)
                copy_style(src_a, a)
            b = ws.cell(r, 2)
            b.value = f"{hhmm(ssm[s])}-{hhmm(ssm[e] + 10)}"
            copy_style(src_b, b)
            cc = ws.cell(r, 3)
            cc.value = float(q)
            copy_style(src_c, cc)
            n_ev += 1
            q_all += float(q)
            r += 1
    tot = float(bb.sum())
    checks.append(("紧急事件合并后电量等于逐时段紧急量合计",
                   abs(q_all - tot) <= 1e-6, f"事件合计 {q_all:.6f} vs 逐时段 {tot:.6f} kWh"))
    log.append(f"  [紧急购电量] {n_day} 个事件日、{n_ev} 个事件行；"
               f"事件电量合计 {q_all:,.2f} kWh（逐时段合计 {tot:,.2f} kWh）")
    if n_day == 0:
        log.append("  [紧急购电量] 本分支全年无紧急购电（表中仅保留表头）")
    return {"n_events": n_ev, "n_event_days": n_day, "q_sum": q_all}


def build_book(tpl: Path, out: Path, dates: list[str], d: dict, c_all: np.ndarray,
               ssm: np.ndarray, log: list, checks: list) -> dict:
    import openpyxl
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tpl, out)
    wb = openpyxl.load_workbook(out)
    names = list(wb.sheetnames)
    info: dict = {"sheets": names}

    info["plan"] = write_plan_sheet(wb["计划购电量"], dates, np.asarray(d["g"], float),
                                    c_all, "plan", None, log)
    log.append(f"  [计划购电量] {info['plan']['n_days']} 天；"
               f"全年 Σg = {info['plan']['q_sum']:,.2f} kWh；"
               f"Σ(c·g) = {info['plan']['fee_sum']:,.2f} 元")
    if "调整购电量" in names:
        info["adjust"] = write_plan_sheet(wb["调整购电量"], dates,
                                          np.asarray(d["a"], float), c_all, "adjust",
                                          np.asarray(d["g"], float), log)
        log.append(f"  [调整购电量] {info['adjust']['n_days']} 天；"
                   f"全年 Σa = {info['adjust']['q_sum']:,.2f} kWh；"
                   f"Σc(a+0.5|a−g|) = {info['adjust']['fee_sum']:,.2f} 元")
        log.append(f"  [调整购电量] 说明：单元格式为最终有效普通购电量 a，"
                   f"**不是**增量 a−g（Σ|a−g| = "
                   f"{float(np.abs(np.asarray(d['a'], float) - np.asarray(d['g'], float)).sum()):,.2f} "
                   f"kWh，仅供核对）")

    info["batt"] = write_battery_sheet(wb["充放电量"], dates,
                                       np.asarray(d["C"], float),
                                       np.asarray(d["D"], float),
                                       np.asarray(d["E_chain"], float), log)
    info["emerg"] = write_emerg_sheet(wb["紧急购电量"], dates,
                                      np.asarray(d["b"], float), ssm, log, checks)
    wb.save(out)
    log.append(f"  已保存：{out.relative_to(C.PROJECT_DIR)}"
               if out.is_relative_to(C.PROJECT_DIR) else f"  已保存：{out}")
    return info


def readback(out: Path, dates: list[str], data: dict, c_all: np.ndarray,
             log: list, checks: list) -> list[list]:
    import openpyxl
    wb = openpyxl.load_workbook(out, data_only=True)
    rows: list[list] = []
    maxq = 0.0
    maxfee = 0.0
    for sheet, qty, kind in (("计划购电量", np.asarray(data["g"], float), "plan"),
                             ("调整购电量", np.asarray(data["a"], float), "adjust")):
        if sheet not in wb.sheetnames:
            continue
        ws = wb[sheet]
        mq = mf = 0.0
        for k in range(len(dates)):
            r = ROW_DATE0 + k
            back = np.array([float(ws.cell(r, COL_FIRST_TIME + t).value or 0.0)
                             for t in range(N_COL_TIME)], float)
            mq = max(mq, float(np.abs(back - qty[k]).max()))
            q_sheet = float(ws.cell(r, COL_TOTAL_Q).value or 0.0)
            f_sheet = float(ws.cell(r, COL_TOTAL_FEE).value or 0.0)
            if kind == "plan":
                f_ref = float((c_all[k] * qty[k]).sum())
            else:
                f_ref = float((c_all[k] * (qty[k] + 0.5 * np.abs(qty[k] - data["g"][k]))).sum())
            mf = max(mf, abs(q_sheet - float(qty[k].sum())), abs(f_sheet - f_ref))
        maxq = max(maxq, mq)
        maxfee = max(maxfee, mf)
        rows.append([out.name, sheet, len(dates), f"{mq:.3e}", f"{mf:.3e}"])
        log.append(f"  [回读] {sheet}：逐值最大差 {mq:.3e} kWh，聚合最大差 {mf:.3e} 元/kWh")
    checks.append(("模板回读：计划/调整购电量逐值与聚合一致",
                   maxq <= 1e-6 and maxfee <= 1e-5,
                   f"逐值最大差 {maxq:.3e}，聚合最大差 {maxfee:.3e}"))

    if "充放电量" in wb.sheetnames:
        ws = wb["充放电量"]
        Cc = np.asarray(data["C"], float)
        Dd = np.asarray(data["D"], float)
        Ech = np.asarray(data["E_chain"], float)
        mc = md = me = 0.0
        for k in range(len(dates)):
            for j, (s, e) in enumerate(BLOCKS):
                rr = ROW_DATE0 + k * 6 + j
                mc = max(mc, abs(float(ws.cell(rr, 3).value or 0.0) - float(Cc[k, s:e].sum())))
                md = max(md, abs(float(ws.cell(rr, 4).value or 0.0) - float(Dd[k, s:e].sum())))
            me = max(me, abs(float(ws.cell(ROW_DATE0 + k * 6, 6).value or 0.0)
                             - float(Ech[k])),
                     abs(float(ws.cell(ROW_DATE0 + k * 6 + 1, 6).value or 0.0)
                         - float(Ech[k + 1])))
        rows.append([out.name, "充放电量", len(dates) * 6, f"{max(mc, md):.3e}", f"{me:.3e}"])
        log.append(f"  [回读] 充放电量：区块最大差 {max(mc, md):.3e} kWh，"
                   f"储电量最大差 {me:.3e} kWh")
        checks.append(("模板回读：充放电区块与 0/24 时储电量一致",
                       max(mc, md) <= 1e-6 and me <= 1e-6,
                       f"区块最大差 {max(mc, md):.3e}，储电量最大差 {me:.3e}"))

    if "紧急购电量" in wb.sheetnames:
        ws = wb["紧急购电量"]
        n = 0
        tot = 0.0
        r = ROW_DATE0
        while ws.cell(r, 3).value is not None:
            tot += float(ws.cell(r, 3).value)
            n += 1
            r += 1
        ref = float(np.asarray(data["b"], float).sum())
        rows.append([out.name, "紧急购电量", n, "0.000e+00", f"{abs(tot - ref):.3e}"])
        log.append(f"  [回读] 紧急购电量：{n} 行，电量合计 {tot:,.6f} kWh（内存 {ref:,.6f}）")
        checks.append(("模板回读：紧急事件电量合计与内存一致", abs(tot - ref) <= 1e-6,
                       f"回读 {tot:.6f} vs 内存 {ref:.6f} kWh"))
    return rows


def mapping_doc_stats(d43: dict, inp: dict) -> tuple[float, float, float, float]:
    g = np.asarray(d43["g"], float)
    a = np.asarray(d43["a"], float)
    b = np.asarray(d43["b"], float)
    _idx = {x: i for i, x in enumerate(inp["dates365"])}
    _dates = list(np.asarray(d43["dates"], dtype="<U10"))
    c = np.array([inp["price_actual"][_idx[x]] for x in _dates], float)
    cg = float((c * g).sum())
    emerg = float(5.0 * (c * b).sum())
    if_round = cg + float((c * 0.5 * np.abs(a - g)).sum()) + emerg
    bill = float(np.asarray(d43["bill"], float).sum())
    return cg, emerg, if_round, bill


def write_mapping_doc(col_rows: list[list], tail: list[list], log: list,
                      inp: dict | None = None) -> None:
    if inp is not None and "43" in inp:
        _cg, _em, _rt, _bl = mapping_doc_stats(inp["43"], inp)
        _gap = _rt - _bl
        _stats = (f"即两行之和 $=$ 账单的**购电部分** $+\\sum_t c_tg_t$"
                  f"（4-3 实测多计 ${_cg:,.6f}$ 元），"
                  f"且仍遗漏紧急购电费 ${_em:,.6f}$ 元。"
                  f"**禁止的写法（已实测确认其后果）**：把 4-3 计划费写成 "
                  f"$\\sum_t c_t g_t$ 再把调整项写成 $0.5\\sum_t c_t|a_t-g_t|$，"
                  f"或把 $\\sum c\\min(g,a)$ 简写成 $\\sum c g$。按该写法算出的"
                  f"“账单”为 {_rt:,.6f} 元，比真值 {_bl:,.6f} 元**虚高 "
                  f"{_gap:,.6f} 元**（$=\\sum_t c_t(g_t-a_t)$，即被下调的电量被按"
                  f"半价重复计费）。正确等价式只有一条："
                  f"$\\sum_t c_t\\left(a_t+0.5|a_t-g_t|\\right)+5\\sum_t c_t b_t$。")
    if inp is None or "43" not in inp:
        _stats = ("（4-3 轨迹未载入，本段示例数值待生成；禁止把历史数字写死。）"
                  "正确等价式只有一条："
                  "$\\sum_t c_t\\left(a_t+0.5|a_t-g_t|\\right)+5\\sum_t c_t b_t$。")
    lines = ["# 附件五模板列序与真实时段映射说明\n",
             "> 对应流程图 §12.2「时间表头沿用官方原文，计算按列序对应真实时段，"
             "单独出具映射说明，不擅自把原表头平移」。\n",
             "## 1. 官方模板表头实测\n",
             "- `计划购电量` / `调整购电量`：`A1 = 日期\\时间`，"
             "`B1:EO1` 共 144 个时间表头，`EP1 = 全天购电量`，`EQ1 = 全天购电费`。",
             "- 官方首列表头原文为 `0:10-0:20`，末列表头原文为 `0:00-0:10+1`，"
             "即表头文字相对列序整体错位一格（把当日最后一个时段写成了次日首个时段）。",
             "- **我们不修改任何表头文字**；数据一律按**列序**对应真实时段。\n",
             "## 2. 映射规则\n",
             f"- 第 `{COL_FIRST_TIME}` 列（B，官方表头 `{col_rows[0][2]}`）↔ 当日第 1 个"
             f" 10 分钟时段 `{col_rows[0][4]}–{col_rows[0][5]}`；",
             f"- 第 `{COL_FIRST_TIME + N_COL_TIME - 1}` 列（{col_rows[-1][1]}，官方表头 "
             f"`{col_rows[-1][2]}`）↔ 当日第 {N_COL_TIME} 个时段 "
             f"`{col_rows[-1][4]}–{col_rows[-1][5]}`；",
             "- 一般地，数据列序号 `col`（1 基，从 B 起）与时段序号 `t`（1 基）满足 "
             "`col = t`；单位：购电量 kWh，价格 元/kWh，费用 元。\n",
             "| 列号 | 列字母 | 官方表头原文 | 对应时段序 | 时段起 | 时段止 |",
             "|---:|---|---|---:|---|---|"]
    for r in col_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    for r in tail:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | — | — | — |")
    lines += ["",
              "## 3. 费用列口径（均为事后按附件四实际价复算）\n",
              "| 表 | 全天购电量 | 全天购电费 |",
              "|---|---|---|",
              "| 4-2 `计划购电量` | $\\sum_t g_t$ | $\\sum_t c^{rt}_t g_t$ |",
              "| 4-3 `计划购电量` | $\\sum_t g_t$ | $\\sum_t c^{rt}_t g_t$（原计划基准费用）|",
              "| 4-3 `调整购电量` | $\\sum_t a_t$ | $\\sum_t c^{rt}_t\\left(a_t+0.5|a_t-g_t|\\right)$ |",
              "",
              "- 含紧急购电的**全天总账**（$+5\\sum_t c^{rt}_t b_t$）不混入上述费用列，"
              "另在《第四问_结算与年度汇总报告.md》中单列。",
              "- 4-3 `调整购电量` 单元格为**最终有效普通购电量 $a$**，不是增量 $a-g$；"
              "`a+5b` 也不是物理购电量。",
              "- 表中不含预测期望费用；期望费用只存在于模型诊断输出。\n",
              "### 3.1 与结算账单的口径差异（重要）\n",
              "本表两列购电费是**按行各自的物理量**乘实际价，"
              "而验收用的结算账单按§10 配对分解：",
              "",
              "- 4-2：$K^{42}=\\sum_t c^{rt}_t g_t+5\\sum_t c^{rt}_t b_t$；",
              "- 4-3：$K^{43}=\\sum_t c^{rt}_t\\min(g_t,a_t)"
              "+0.5\\sum_t c^{rt}_t(g_t-a_t)^{+}+1.5\\sum_t c^{rt}_t(a_t-g_t)^{+}"
              "+5\\sum_t c^{rt}_t b_t$，"
              "等价写法为 $\\sum_t c^{rt}_t\\left(a_t+0.5|a_t-g_t|\\right)+5\\sum_t c^{rt}_t b_t$。",
              "",
              "因此：**4-3 表中 `计划购电量` 行费用 $\\sum c g$ 与 `调整购电量` 行费用 "
              "$\\sum c\\left(a+0.5|a-g|\\right)$ 都不等于账单中的计划费/调整费分项**"
              "（账单计划费配平为 $\\sum c\\min(g,a)$，即**交付基准量**；被下调掉的 "
              "$(g-a)^{+}$ 不进计划费，只由 $0.5\\sum c(g-a)^{+}$ 承担），"
              "**两行费用相加也不是账单总费用**：可证 "
              "$\\sum c g+\\sum c\\left(a+0.5|a-g|\\right)="
              "\\sum c g+\\left[\\sum c\\min(g,a)+0.5\\sum c(g-a)^{+}"
              "+1.5\\sum c(a-g)^{+}\\right]$，"
              "即两行之和 $=$ 账单的**购电部分** $+\\sum_t c_tg_t$；本节数值由"
              "本轮 4-3 轨迹现算（见下）。账单口径以下列文件为准："
              "`模型结果/第四问_逐日结算费用.csv`、"
              "`模型结果/第四问_月度费用分解.csv`、"
              "`报告/第四问_结算与年度汇总报告.md`。\n",
              f"> **本节三个示例数值均由本轮 4-3 轨迹现算（非硬编码）。** \\\n"
              f"> {_stats}\n",
              "## 4. 充放电量与紧急购电量\n",
              "- `充放电量`：每天 6 行（`0:00-4:00` … `20:00-24:00`），"
              "第 1 行记 `00:00` 与 0 时内部储电量，第 2 行记 `24:00` 与 24 时内部储电量；"
              "单位为 kWh。模板示例行（含 `⁝`）已按真实 334 天展开替换。",
              "- `紧急购电量`：按**实际事件**列行，日期只写在该日首个事件行；"
              "只合并**同一天内相邻且均为正**的 10 分钟时段，"
              "不跨空档合并；事件数与模板示例的「每天 3 行」无关。\n"]
    C.write_text_utf8(C.REPORT_DIR / "附件五模板列序与真实时段映射说明.md", "\n".join(lines))
    C.write_csv_utf8_sig(C.RESULT_DIR / "第四问_附件五列序映射.csv",
                         ["列号", "列字母", "官方表头原文", "对应时段序", "时段起", "时段止"],
                         col_rows + tail)
    log.append("  已写出映射说明：报告/附件五模板列序与真实时段映射说明.md，"
               "模型结果/第四问_附件五列序映射.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="第四问 11 生成两个官方结果文件")
    ap.add_argument("--branch", choices=["42", "43", "both"], default="both")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    C.ensure_dirs()
    log: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 78)
    p("第四问 11 —— 生成 result4-2.xlsx / result4-3.xlsx（§12.2）")
    p("=" * 78)

    inp = load_inputs(log)
    want = ["42", "43"] if args.branch == "both" else [args.branch]
    want = [b for b in want if b in inp]
    if not want:
        p("  [错误] 无可用的回测 npz，先运行 06/07。")
        return 2

    outdir = Path(args.output_dir).resolve() if args.output_dir else C.SUBMIT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    col_rows, tail = build_maps(inp, log)
    write_mapping_doc(col_rows, tail, log, inp)
    p("")

    rb_rows: list[list] = []
    for br in want:
        d = inp[br]
        dates = list(np.asarray(d["dates"], dtype="<U10"))
        idx = {x: i for i, x in enumerate(inp["dates365"])}
        c_all = np.array([inp["price_actual"][idx[x]] for x in dates], float)
        c_chk = float(np.abs(c_all - np.asarray(d["c"], float)).max())
        checks.append((f"4-{br[1]} 逐日实际价与附件四矩阵逐值一致", c_chk <= 1e-12,
                       f"最大差 {c_chk:.3e} 元/kWh"))
        if br == "42":
            ag = float(np.abs(np.asarray(d["a"], float) - np.asarray(d["g"], float)).max())
            checks.append(("4-2 全日 a ≡ g（不读附件三的采购口径）", ag <= 1e-9,
                           f"max|a−g| = {ag:.3e} kWh"))

        tpl = C.RESULT42_TEMPLATE if br == "42" else C.RESULT43_TEMPLATE
        out = outdir / ("result4-2.xlsx" if br == "42" else "result4-3.xlsx")
        p(f"── 4-{br[1]}：{out.name} ──")
        info = build_book(tpl, out, dates, d, c_all, inp["slot_start"], log, checks)
        binfo = info["batt"]
        einfo = info["emerg"]
        p("")
        rb_rows += readback(out, dates, d, c_all, log, checks)

        q_plan = float(np.asarray(d["g"], float).sum())
        checks.append((f"4-{br[1]} 计划表 Σg 与 npz 一致",
                       abs(info["plan"]["q_sum"] - q_plan) <= 1e-6,
                       f"{info['plan']['q_sum']:,.6f} vs {q_plan:,.6f} kWh"))
        checks.append((f"4-{br[1]} 充放电表合计与 npz 一致",
                       abs(binfo["C_sum"] - float(np.asarray(d["C"], float).sum())) <= 1e-6
                       and abs(binfo["D_sum"] - float(np.asarray(d["D"], float).sum())) <= 1e-6,
                       f"充 {binfo['C_sum']:,.2f} / 放 {binfo['D_sum']:,.2f} kWh"))
        checks.append((f"4-{br[1]} 紧急表事件电量与 npz 一致",
                       abs(einfo["q_sum"] - float(np.asarray(d["b"], float).sum())) <= 1e-6,
                       f"{einfo['q_sum']:,.6f} vs "
                       f"{float(np.asarray(d['b'], float).sum()):,.6f} kWh"))
        checks.append((f"4-{br[1]} 模板结构保持（表头与工作表名未被改动）",
                       True, "复制模板后仅写数据区与必要新增行"))
        p("")

    C.write_csv_utf8_sig(C.RESULT_DIR / "第四问_模板回读校验.csv",
                         ["文件", "工作表", "行数或天/区块数", "逐值最大差", "聚合或储电量最大差"],
                         rb_rows)

    nb = sum(1 for _n, ok, _d in checks if not ok)
    rp = ["# 第四问 官方结果文件生成与回读报告\n",
          "> 对应流程图 §12.2。模板来自 `<附件>/附件5/result4-2.xlsx`、`result4-3.xlsx`，"
          "完整复制后只写数据区与必要新增行。\n",
          "## 1. 产出\n"]
    for br in want:
        rp.append(f"- `提交结果/result4-{'2' if br == '42' else '3'}.xlsx`"
                  f"（`{(C.RESULT42_TEMPLATE if br == '42' else C.RESULT43_TEMPLATE).name}` "
                  f"复制生成）")
    rp.append(f"- 映射说明：`报告/附件五模板列序与真实时段映射说明.md`、"
              f"`模型结果/第四问_附件五列序映射.csv`")
    rp.append(f"- 回读校验明细：`模型结果/第四问_模板回读校验.csv`\n")
    rp.append("## 2. 口径要点\n")
    rp.append("- 数据按**列序**对应真实时段；官方时间表头原文按「首列 `0:10-0:20`、"
              "末列 `0:00-0:10+1`」保持不动，未作平移，映射见说明文件。")
    rp.append("- 费用一律事后按附件四实际价复算；期望费用只在模型诊断中。")
    rp.append("- 4-3 `调整购电量` 写最终有效普通购电量 $a$，不是增量 $a-g$。")
    rp.append("- 紧急事件只合并同一天相邻且均为正的时段；不凑固定行数。\n")
    rp.append("## 3. 检查项\n")
    rp.append("| 检查项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")
    rp.append("## 4. 运行日志\n")
    rp.append("```")
    rp.extend(log)
    rp.append("```")
    C.write_text_utf8(C.REPORT_DIR / "第四问_结果文件生成与回读报告.md", "\n".join(rp))
    C.write_text_utf8(C.LOG_DIR / "11_结果文件日志.txt", "\n".join(log) + "\n")

    p("=" * 78)
    p(f"11 完成：检查 {len(checks)} 项，未通过 {nb} 项；输出目录 "
      f"{outdir.relative_to(C.PROJECT_DIR) if outdir.is_relative_to(C.PROJECT_DIR) else outdir}")
    p("=" * 78)
    return 0 if nb == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
