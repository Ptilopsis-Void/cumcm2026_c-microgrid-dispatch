#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
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


C = _load("_comm3.py", "q3_comm")

DELTA_GRID = (6.0, 3.0, 1.5)
M_GRID = (10, 20, 30)
NU_GRID = (0.0, 0.25, None, 0.75, 1.0)

UPDATE_TIMES = (6, 12, 18)

CONV_TOL = 0.005

CV_DELTA_CSV = C.RESULT_DIR / "第三问_收敛性_网格步长.csv"
CV_M_CSV = C.RESULT_DIR / "第三问_收敛性_情景数.csv"
SE_NU_CSV = C.RESULT_DIR / "第三问_敏感性_终端价值.csv"
CV_NPZ = C.RESULT_DIR / "第三问_敏感性与收敛性.npz"
REPORT_MD = C.REPORT_DIR / "第三问_敏感性与收敛性报告.md"
FIG_STEM = "第三问_敏感性与收敛性"

COLS = ["总费用_元", "模型内期望总费用_元", "计划购电费_元", "调整相关费用_元",
        "紧急购电费_元", "计划购电量_kWh", "调整购电量_kWh", "紧急购电量_kWh",
        "充电量_kWh", "放电量_kWh", "弃电量_kWh", "日末库存_末值_kWh",
        "闭环残差最大值", "信息泄露时段数"]
COL_LABEL = {
    "总费用_元": "总费用（实现口径）",
    "模型内期望总费用_元": "期望总费用（共同测度）",
    "计划购电费_元": "计划购电费",
    "调整相关费用_元": "调整相关费用",
    "紧急购电费_元": "紧急购电费",
    "计划购电量_kWh": "计划购电量",
    "调整购电量_kWh": "调整购电量",
    "紧急购电量_kWh": "紧急购电量",
    "充电量_kWh": "充电量",
    "放电量_kWh": "放电量",
    "弃电量_kWh": "弃电量",
    "日末库存_末值_kWh": "日末库存",
    "闭环残差最大值": "闭环残差最大值",
    "信息泄露时段数": "信息泄露时段数",
}


def _rel(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    if abs(b) < 1e-12:
        return 0.0 if abs(a) < 1e-12 else float("nan")
    return (a - b) / b


def _pct(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{100.0 * x:+.4f} %"


def _num(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:,.2f}"


def _run_block(log, P3, specs, days_kw, tag):
    rows: dict[str, dict] = {}
    raws: dict[str, dict] = {}
    for label, kw in specs:
        t1 = time.perf_counter()
        r = P3.run_policy(update_times=UPDATE_TIMES, **kw, **days_kw)
        s = P3.summarise(r, label=label)
        rows[label] = s
        raws[label] = r
        log(f"  {label:<22} 实现 {s['总费用_元']:>16,.4f}  期望 "
            f"{s['模型内期望总费用_元']:>18,.4f}  Σb {s['紧急购电量_kWh']:>12,.4f}"
            f"  残差 {s['闭环残差最大值']:.2e}  用时 {time.perf_counter() - t1:.1f} s")
    log(f"{tag} 总用时 {sum(1 for _ in rows)} 档已记录")
    return rows, raws


def _write_table(path, labels, rows, extra_col: str = "", extra_val=None):
    header = ["档位"] + [COL_LABEL[c] for c in COLS]
    if extra_col:
        header.append(extra_col)
    body = []
    for lb in labels:
        s = rows[lb]
        line = [lb] + [f"{float(s[c]):.6f}" if np.isfinite(float(s[c])) else "nan"
                       for c in COLS]
        if extra_col:
            line.append(f"{float(extra_val[lb]):.6f}"
                        if np.isfinite(float(extra_val[lb])) else "nan")
        body.append(line)
    C.write_csv_utf8_sig(path, header, body)


def _md_table(labels, rows, ref: str, extra_col: str = "", extra_val=None) -> list[str]:
    keys = ["总费用_元", "模型内期望总费用_元", "紧急购电量_kWh", "紧急购电费_元",
            "充电量_kWh", "放电量_kWh", "弃电量_kWh", "日末库存_末值_kWh"]
    head = ["档位"] + [COL_LABEL[k] for k in keys] + ["期望费用相对基准"]
    if extra_col:
        head.append(extra_col)
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for lb in labels:
        s = rows[lb]
        cells = [lb] + [_num(float(s[k])) for k in keys]
        cells.append(_pct(_rel(float(s["模型内期望总费用_元"]),
                               float(rows[ref]["模型内期望总费用_元"]))))
        if extra_col:
            cells.append(_num(float(extra_val[lb])))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _verdict(block: str, labels: list[str], vals: list[float],
             extra: dict[str, list[float]] | None = None) -> dict:
    res: dict = {"块": block, "细分档": labels[-1], "次细档": labels[-2],
                 "末次相对变化": _rel(float(vals[-1]), float(vals[-2]))}
    for k, seq in (extra or {}).items():
        res[f"末次相对变化·{k}"] = _rel(float(seq[-1]), float(seq[-2]))
    allv = [abs(v) for k, v in res.items() if k.startswith("末次相对变化")
            and np.isfinite(v)]
    res["最大末次相对变化"] = max(allv) if allv else float("nan")
    res["已收敛"] = bool(np.isfinite(res["最大末次相对变化"])
                        and res["最大末次相对变化"] <= CONV_TOL)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="第三问敏感性与收敛性检验")
    ap.add_argument("--blocks", default="delta,m,nu",
                    help="要运行的块，逗号分隔，取 delta/m/nu")
    args = ap.parse_args()
    want = {b.strip().lower() for b in args.blocks.split(",") if b.strip()}

    C.ensure_dirs()
    log = C.Tee()
    t00 = time.perf_counter()

    log("=" * 78)
    log("第三问 13 —— 敏感性与收敛性检验（P1-2 / P1-3 / P1-4）")
    log("=" * 78)

    P3 = _load("_policy3.py", "q3_policy")
    f = P3.facts()
    sc = np.asarray(f["score_day_index"], int)
    nu = float(C.NU_VALUE)
    M_lib = int(np.asarray(f["scen_L"]).shape[1])

    _maxd = int(os.environ.get("Q3_SENS_MAX_DAYS", "0") or 0)

    def _progress(msg: str) -> None:
        log(msg)

    days_kw: dict = {"log": _progress, "progress_every": 20}
    if _maxd > 0:
        days_kw["days"] = [int(x) for x in sc[:_maxd]]
        log("")
        log(f"⚠⚠⚠ 骨架冒烟模式：Q3_SENS_MAX_DAYS = {_maxd}（只跑前 {_maxd} 个评分日）")
        log("      该模式下的任何数字都不得写入论文、不得提交、不得作为建模结论。")
        log("")
    if C.SANDBOX:
        log("")
        log(f"⚠⚠ 数据源 = {C.Q2_SOURCE}：输出写入沙箱 {C.OUT_ROOT}")
        log("      非 real 数据源下本节所有数值不具任何建模意义，仅供管线自检。")
        log("")

    n_days_used = int(len(days_kw["days"])) if "days" in days_kw else int(sc.size)
    log(f"情景库可用情景数 M_lib = {M_lib}；ν（附件 1 复算）= {nu:.10f} 元/kWh")
    log(f"评分日数 = {n_days_used}；更新时刻 {UPDATE_TIMES}；"
        f"收敛阈值 {100 * CONV_TOL:.2f} %")
    log("口径：ν 仅作规划信号，**从不进入账单**；全部费用由 `_policy3.decompose` 给出。")

    tables: dict[str, dict[str, dict]] = {}
    raws_all: dict[str, dict] = {}

    if "delta" in want:
        log("")
        log("── A 块：内部电量网格步长 δ 收敛性（δ 从粗到细）──")
        specs = [(f"δ={d:g} kWh", {"delta": float(d)}) for d in DELTA_GRID]
        rows, raws = _run_block(log, P3, specs, days_kw, "A 块")
        tables["delta"] = rows
        raws_all.update({f"delta::{k}": v for k, v in raws.items()})

    if "m" in want:
        log("")
        log(f"── B 块：情景数 M 收敛性（情景库上限 {M_lib}）──")
        grid = [int(x) for x in M_GRID if int(x) <= M_lib]
        if M_lib not in grid:
            grid.append(M_lib)
        specs = [(f"M={int(x)}", {"m": int(x), "eval_full": True}) for x in grid]
        rows, raws = _run_block(log, P3, specs, days_kw, "B 块")
        tables["m"] = rows
        raws_all.update({f"m::{k}": v for k, v in raws.items()})

    if "nu" in want:
        log("")
        log("── C 块：终端价值系数 ν 敏感性（ν 只进规划，不进账单）──")
        specs = []
        for v in NU_GRID:
            val = nu if v is None else float(v)
            tag = (f"ν={val:.10f}（附件 1 复算值）" if v is None
                   else f"ν={val:.4g}")
            specs.append((tag, {"nu": val}))
        rows, raws = _run_block(log, P3, specs, days_kw, "C 块")
        tables["nu"] = rows
        raws_all.update({f"nu::{k}": v for k, v in raws.items()})

    verdicts: list[dict] = []
    if "delta" in tables:
        lb = list(tables["delta"])
        verdicts.append(_verdict(
            "A 网格步长 δ", lb,
            [float(tables["delta"][k]["模型内期望总费用_元"]) for k in lb],
            {"紧急电量": [float(tables["delta"][k]["紧急购电量_kWh"]) for k in lb],
             "充电量": [float(tables["delta"][k]["充电量_kWh"]) for k in lb]}))
    if "m" in tables:
        lb = list(tables["m"])
        verdicts.append(_verdict(
            "B 情景数 M", lb,
            [float(tables["m"][k]["模型内期望总费用_元"]) for k in lb],
            {"紧急电量": [float(tables["m"][k]["紧急购电量_kWh"]) for k in lb],
             "充电量": [float(tables["m"][k]["充电量_kWh"]) for k in lb]}))

    _ev = lambda lb: {k: n_days_used for k in lb}
    if "delta" in tables:
        lb = list(tables["delta"])
        _write_table(CV_DELTA_CSV, lb, tables["delta"],
                     extra_col="评分日数", extra_val=_ev(lb))
    if "m" in tables:
        lb = list(tables["m"])
        _write_table(CV_M_CSV, lb, tables["m"],
                     extra_col="评分日数", extra_val=_ev(lb))
    if "nu" in tables:
        lb = list(tables["nu"])
        _write_table(SE_NU_CSV, lb, tables["nu"],
                     extra_col="评分日数", extra_val=_ev(lb))
    log("")
    log("CSV 已写出：")
    for p in (CV_DELTA_CSV, CV_M_CSV, SE_NU_CSV):
        if p.exists():
            log(f"  {p.relative_to(C.OUT_ROOT)}")

    npz: dict = {"nu_ref": np.asarray([nu], float),
                 "update_times": np.asarray(UPDATE_TIMES, int),
                 "delta_grid": np.asarray(DELTA_GRID, float),
                 "m_grid": np.asarray(M_GRID, int),
                 "nu_grid": np.asarray([nu if v is None else v for v in NU_GRID], float),
                 "conv_tol": np.asarray([CONV_TOL], float)}
    for blk, rows in tables.items():
        lb = list(rows)
        npz[f"{blk}_labels"] = np.asarray(lb, dtype=str)
        for j, c in enumerate(COLS):
            npz[f"{blk}_col{j}"] = np.asarray([float(rows[k][c]) for k in lb], float)
        npz[f"{blk}_cols"] = np.asarray(COLS, dtype=str)
    np.savez_compressed(CV_NPZ, **npz)
    log(f"  {CV_NPZ.relative_to(C.OUT_ROOT)}")

    try:
        import matplotlib.pyplot as plt
        C.setup_matplotlib()
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        _axes = [("delta", "A 网格步长 δ (kWh)", axes[0], [float(d) for d in DELTA_GRID]),
                 ("m", "B 情景数 M", axes[1], None),
                 ("nu", "C 终端价值 ν (元/kWh)", axes[2], None)]
        for key, title, ax, _ in _axes:
            rows = tables.get(key)
            if not rows:
                ax.set_visible(False)
                continue
            lb = list(rows)
            xs = np.arange(len(lb))
            ax.plot(xs, [float(rows[k]["模型内期望总费用_元"]) for k in lb],
                    "o-", label="期望总费用（共同测度）")
            ax.set_xticks(xs)
            ax.set_xticklabels(lb, rotation=30, ha="right", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("元")
            ax.grid(alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(xs, [float(rows[k]["日末库存_末值_kWh"]) for k in lb],
                     "s--", color="C1", label="日末库存（末值）")
            ax2.set_ylabel("kWh", color="C1")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
        fig.suptitle("第三问：敏感性与收敛性（P1-2 / P1-3 / P1-4）", fontsize=11)
        fig.tight_layout()
        png = C.save_figure(fig, FIG_STEM)
        plt.close(fig)
        log(f"  图 {png.relative_to(C.OUT_ROOT)}")
    except Exception as exc:
        log(f"  ⚠ 作图失败（不影响数值）：{type(exc).__name__}: {exc}")

    L: list[str] = []
    L.append(f"数据源 = **{C.Q2_SOURCE}**，输出根目录 = `{C.OUT_ROOT.name}/`。")
    L.append("")
    if C.SANDBOX:
        L.append("> ⚠⚠ **非 real 数据源**：本节所有数值不具任何建模意义，"
                 "仅供管线自检（K15）。**不得写入论文、不得提交。**")
        L.append("")
    L.append(f"- 唯一策略入口：`_policy3.run_policy(update_times={UPDATE_TIMES}, …)`"
             "（与本问正式主结果同源）")
    L.append(f"- 情景库可用情景数：$M_{{lib}} = {M_lib}$")
    L.append(f"- 终端价值系数基准：$\\nu = {nu:.10f}$ 元/kWh（附件 1 复算）")
    L.append(f"- 收敛阈值：末次相对变化 $\\le {100 * CONV_TOL:.2f}\\%$")
    L.append(f"- 评分日数：{n_days_used}"
             + (f"（**冒烟模式仅 {_maxd} 日**，非正式）" if _maxd > 0 else ""))
    L.append("")

    if "delta" in tables:
        lb = list(tables["delta"])
        L.append("## 1. A 块：内部电量网格步长 $\\delta$ 收敛性")
        L.append("")
        L.append(f"网格范围 $[E_{{\\min}}, E_{{\\max}}] = [{C.E_MIN:g}, {C.E_MAX:g}]$ kWh，"
                 f"步长 $\\delta$ 决定网格点数 $n = (E_{{\\max}}-E_{{\\min}})/\\delta$：")
        L.append("")
        L.append("| δ (kWh) | 网格点数 | 说明 |")
        L.append("|---:|---:|---|")
        for d in DELTA_GRID:
            n = int(round((C.E_MAX - C.E_MIN) / float(d)))
            L.append(f"| {d:g} | {n} | 越细越精确、越慢 |")
        L.append("")
        L += _md_table(lb, tables["delta"], lb[0])
        L.append("")

    if "m" in tables:
        lb = list(tables["m"])
        L.append("## 2. B 块：情景数 $M$ 收敛性")
        L.append("")
        L.append(f"情景取该日情景库的**前 $M$ 个**（分层分位数抽样，$M \\le M_{{lib}} = {M_lib}$）；"
                 "LP 目标的 $1/M$ 权重与实际情景数严格一致。")
        L.append("")
        L += _md_table(lb, tables["m"], lb[0])
        L.append("")

    if "nu" in tables:
        lb = list(tables["nu"])
        L.append("## 3. C 块：终端价值系数 $\\nu$ 敏感性")
        L.append("")
        L.append("$\\nu$ 是日末库存的**续存价值**（规划信号）："
                 "$H_{T+1,\\omega}(e) = -\\nu e$，")
        L.append("即「多留 1 kWh 库存被记为节省 $\\nu$ 元」。")
        L.append("")
        L.append("> **口径铁律**：$\\nu$ **从不进入账单**。表中「总费用（实现口径）」"
                 "与 $\\nu$ 的关系完全来自**改变后的充放电行为**，不是记账项。")
        L.append("")
        L += _md_table(lb, tables["nu"], [k for k in lb if "复算值" in k][0]
                       if any("复算值" in k for k in lb) else lb[0])
        L.append("")

    L.append("## 4. 判定与结论")
    L.append("")
    if verdicts:
        L.append("| 块 | 末次相对变化 | 阈值 | 判定 |")
        L.append("|---|---:|---:|---|")
        for v in verdicts:
            keys = [k for k in v if k.startswith("末次相对变化")]
            L.append(f"| {v['块']} | {_pct(v['末次相对变化'])} | "
                     f"{100 * CONV_TOL:.2f} % | "
                     f"{'✅ 已收敛' if v['已收敛'] else '⚠ 尚未收敛'} |")
            for k in keys[1:]:
                L.append(f"| ↳ {k} | {_pct(v[k])} | — | — |")
        L.append("")
        L.append("> 判定规则：对**主指标（总费用）与全部附列指标**取末次相对变化的"
                 "最大绝对值，只要有一项超过阈值即判「尚未收敛」（保守口径）。"
                 "阈值只约束**是否可声称收敛**，不构成对模型的否决。")
        L.append("")
        for v in verdicts:
            if not v["已收敛"]:
                drv = [k for k in v if k.startswith("末次相对变化")
                       and np.isfinite(v[k]) and abs(v[k]) > CONV_TOL]
                L.append(f"- ⚠ **{v['块']} 尚未收敛**：超阈值的指标为 "
                         + "、".join(f"`{k}`（{_pct(v[k])}）" for k in drv)
                         + f"，其中主指标「总费用」的末次相对变化为 "
                           f"{_pct(v['末次相对变化'])}。论文中**不得**声称该维度"
                           "已收敛，应把剩余漂移作为一种不确定性来源报告；"
                           "若主指标本身已在阈值内，可表述为"
                           "「总费用对该维度已稳定，但次级指标（如充/放电量、"
                           "紧急购电量）仍有余量漂移」。")
    else:
        L.append("（本次未运行收敛性块。）")
    L.append("")

    if "nu" in tables:
        lb = list(tables["nu"])
        tot = [float(tables["nu"][k]["总费用_元"]) for k in lb]
        soc = [float(tables["nu"][k]["日末库存_末值_kWh"]) for k in lb]
        q = [float(tables["nu"][k]["调整购电量_kWh"]) for k in lb]
        span = (max(tot) - min(tot))
        L.append("### 4.1 $\\nu$ 的结论稳健性")
        L.append("")
        L.append(f"- 总费用区间：{_num(min(tot))} – {_num(max(tot))} 元"
                 f"（极差 {_num(span)} 元，占均值 "
                 f"{_pct(span / (sum(tot) / len(tot))) if abs(sum(tot)) > 0 else '—'}）")
        L.append(f"- 日末库存（末值）区间：{_num(min(soc))} – {_num(max(soc))} kWh")
        L.append(f"- 调整购电总量区间：{_num(min(q))} – {_num(max(q))} kWh")
        L.append("")
        L.append("⇒ 若总费用极差相对均值很小，则**购电与调整策略对 $\\nu$ 稳健**；"
                 "若明显变化，说明终端库存计价对结论有实质影响，"
                 "论文须**同时报告 $\\nu$ 的取值依据**（附件 1 复算值）与该敏感性区间。")
        L.append("")

    L.append("## 5. 复现命令")
    L.append("")
    L.append("```bash")
    L.append(f".venv/bin/python 代码/13_敏感性与收敛性.py --blocks "
             f"{','.join(k for k in ('delta', 'm', 'nu') if k in tables)}")
    L.append("```")
    L.append("")
    L.append("> 冒烟自检：`Q3_SENS_MAX_DAYS=6` 只跑前 6 个评分日，产物落在"
             f" `{C.out_root('stub').name}/`。")
    L.append("")

    body = "\n".join(L)
    log("")
    log("── 报告草稿 ──")
    log(body)

    C.write_text_utf8(REPORT_MD, log.markdown(
        "第三问 · 敏感性与收敛性报告（P1-2 / P1-3 / P1-4）"))
    log("")
    log(f"报告已写出：{REPORT_MD.relative_to(C.OUT_ROOT)}")
    log(f"13 总用时 {time.perf_counter() - t00:.1f} s")

    log.dump(C.LOG_DIR / "第三问_13_日志.txt")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
