#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
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


C = _load("_comm3.py", "q3_comm")

STEPS: tuple[tuple[str, str, float], ...] = (
    ("00", "接口契约与骨架自检", 10.0),
    ("01", "检查附件三原始数据", 20.0),
    ("02", "处理附件三基础数据", 30.0),
    ("03", "校验附件三处理结果", 20.0),
    ("04", "构建情景库", 30.0),
    ("05", "求解阶段 0 计划（LP，被 _policy3 调用）", 120.0),
    ("06", "滚动调整规划（⚠ 历史实现·产物不参与正式结果，见 P0-1/P0-3）", 120.0),
    ("07", "阶段内 DP 执行器（**函数库**，P0-7 选择 A；独立驱动已退役）", 3.0),
    ("08", "★ 全年回测与结算（唯一主结果入口）", 260.0),
    ("09", "消融实验", 600.0),
    ("10", "指定日期明细与作图", 30.0),
    ("11", "生成并校验 result3.xlsx", 30.0),
    ("13", "敏感性与收敛性检验（P1-2/P1-3/P1-4；非交付主结果，可单独跑）",
     2400.0),
    ("14", "★ 最终一致性校验（只读交叉复核，秒级；全部主结果入账前守门）",
     30.0),
)


def get_deliverables(source: str) -> tuple[Path, ...]:
    root = C.out_root(source)
    proc = root / "处理后数据"
    res = root / "模型结果"
    rep = root / "报告"
    sub = root / "提交结果"
    return (
        proc / C.V_FORECAST_NPZ.name,
        proc / C.V_SCENARIO_NPZ.name,
        res / C.STAGE0_NPZ.name,
        res / C.BACKTEST_NPZ.name,
        res / "第三问_结算口径对照.csv",
        res / "第三问_闭环审计.csv",
        res / C.ABLATION_CSV.name,
        res / "第三问_收敛性_网格步长.csv",
        res / "第三问_收敛性_情景数.csv",
        res / "第三问_敏感性_终端价值.csv",
        rep / C.REPORT_EXEC_MD.name,
        rep / "第三问_敏感性与收敛性报告.md",
        res / "第三问_最终一致性校验.csv",
        rep / "第三问_最终一致性校验报告.md",
        sub / C.RESULT3_XLSX.name,
    )

ANCHOR_KEYS: tuple[str, ...] = (
    "计划购电量_kWh",
    "调整购电量_kWh",
    "下调量_kWh",
    "上调量_kWh",
    "计划购电费_元",
    "调整相关费用_元",
    "紧急购电量_kWh",
    "紧急购电费_元",
    "总费用_元",
    "无储能基准购电费_元",
    "储能相对基准节省_元",
    "储能相对基准节省_%",
    "总费用均价_元每kWh",
)


LIB_FILES: tuple[str, ...] = ("_comm3.py", "_q2_adapter.py", "_policy3.py")


def _script_sha(script: Path) -> str:
    try:
        return C.compute_sha256(script)
    except OSError:
        return "<不可读>"


def _fmt_sec(x: float) -> str:
    return f"{x:.1f} s" if x < 120 else f"{x / 60.0:.2f} min"


def resolve_steps(only: list[str] | None, start: str | None) -> list[tuple[str, str, float, Path]]:
    picked = []
    for num, name, est in STEPS:
        if only and num not in only:
            continue
        if start and num < start:
            continue
        cands = sorted(C.CODE_DIR.glob(f"{num}_*.py"))
        if not cands:
            raise FileNotFoundError(f"未找到步骤 {num} 对应的脚本（在 {C.CODE_DIR}）")
        picked.append((num, name, est, cands[0]))
    return picked


def run_step(num: str, name: str, script: Path, log_lines: list[str],
             source: str | None = None) -> tuple[bool, float]:
    t0 = time.time()
    log_lines.append(f"[{num}] 开始 {name}  ← {script.name}")
    print(f"\n{'=' * 78}\n[{num}] {name}  （{script.name}）\n{'=' * 78}", flush=True)

    def _launch() -> tuple[int, str]:
        env = None
        if source:
            env = {**os.environ, "Q3_Q2_SOURCE": source}
        p = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(C.PROJECT_DIR),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        chunks: list[str] = []
        assert p.stdout is not None
        for ln in p.stdout:
            chunks.append(ln)
            print(ln, end="", flush=True)
        p.wait()
        return int(p.returncode or 0), "".join(chunks)

    rc, combined = _launch()

    _FATAL_INIT = ("init_sys_streams", "Bad file descriptor",
                   "can't initialize sys standard streams")
    if rc != 0 and len(combined.strip()) < 200 and any(
            t in combined for t in _FATAL_INIT):
        warn = (f"  ⚠ [{num}] 子进程在**解释器启动阶段**崩溃（fd 故障，"
                f"非脚本逻辑错误），自动重试 1 次…")
        print(warn, flush=True)
        log_lines.append(warn)
        rc, combined = _launch()

    dt = time.time() - t0
    (C.LOG_DIR / f"第三问_12_步骤{num}日志.txt").write_text(combined, encoding="utf-8")
    tail = "\n".join(combined.rstrip().splitlines()[-14:])
    ok = rc == 0
    log_lines.append(f"[{num}] {'成功' if ok else f'失败(EXIT={rc})'}  用时 {_fmt_sec(dt)}")
    if not ok:
        log_lines.append(f"[{num}] 失败输出尾部：\n{tail}")
        print(f"\n✗ [{num}] {name} 失败，EXIT={rc}", flush=True)
        if any(t in combined for t in _FATAL_INIT):
            hint = ("  ⚠ 这是**解释器启动阶段**的 fd 故障，不是脚本逻辑错误。\n"
                    "    子进程已用 stdin=DEVNULL 启动且已重试过 1 次；若仍失败，\n"
                    "    说明父进程的 stdout 管道被上层回收。请改为前台运行，或在\n"
                    "    screen / tmux 会话内后台运行。")
            print(hint, flush=True)
            log_lines.append(hint)
        elif len(combined.strip()) < 200:
            hint = (f"  ⚠ 输出不足 200 字符（{len(combined)} 字符），"
                    f"子进程可能在导入阶段即失败；请单独运行 "
                    f"`{script.name}` 查看完整回溯。")
            print(hint, flush=True)
            log_lines.append(hint)
    else:
        print(f"\n✓ [{num}] {name} 完成，用时 {_fmt_sec(dt)}", flush=True)
    return ok, dt


def read_anchors(out_root: Path) -> dict[str, str]:
    import csv
    out: dict[str, str] = {}
    path = out_root / "模型结果" / C.YEARLY_CSV.name
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].strip():
                out[row[0].strip()] = row[1].strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="第三问端到端流水线编排器")
    ap.add_argument("--from", dest="start", default=None,
                    help="从指定序号开始（例如 05）")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定序号（例如 08 11）")
    ap.add_argument("--source", choices=("real", "stub", "off"), default=None,
                    help="第二问数据来源（real 真实 / stub 合成占位骨架自检 / off 禁用）；"
                         "默认沿用环境变量 Q3_Q2_SOURCE")
    ap.add_argument("--dry-run", action="store_true", help="只打印执行计划")
    args = ap.parse_args()

    def _pad(x: str) -> str:
        x = str(x).strip()
        return x.zfill(2) if x.isdigit() else x
    args.only = [_pad(x) for x in args.only] if args.only else None
    args.start = _pad(args.start) if args.start else None

    source = args.source or (os.environ.get("Q3_Q2_SOURCE", "real") or "real").strip().lower()
    if args.source:
        os.environ["Q3_Q2_SOURCE"] = source
    deliverables = get_deliverables(source)
    out_root = C.out_root(source)

    C.ensure_dirs()
    steps = resolve_steps(args.only, args.start)

    print("第三问完整求解流水线")
    print(f"  项目目录：{C.PROJECT_DIR}")
    print(f"  解释器　：{sys.executable}")
    print(f"  数据源　：{source}" + ("  ⚠ 合成占位数据，仅骨架自检" if source == "stub" else ""))
    print(f"  输出目录：{out_root}")
    print(f"  步骤数　：{len(steps)}")
    for num, name, est, script in steps:
        print(f"    {num}  {name:<24s} 预计 {_fmt_sec(est):>9s}  {script.name}")
    if args.dry_run:
        return 0

    frozen_paths: dict[str, Path] = {}
    frozen: dict[str, str] = {}
    for num, _name, _est, script in steps:
        frozen_paths[num] = script
        frozen[num] = _script_sha(script)
    for lib in LIB_FILES:
        p = _HERE / lib
        if p.is_file():
            frozen_paths[f"lib:{lib}"] = p
            frozen[f"lib:{lib}"] = _script_sha(p)
    n_lib = sum(1 for k in frozen if k.startswith("lib:"))
    print(f"  冻结指纹：{len(steps)} 个步骤脚本 + {n_lib} 个共享模块 sha256 已记录")
    print("  守门判据：运行期间任一被改写 ⇒ 退出码 2，产物不得作为定稿")

    def _drift_now() -> list[tuple[str, str, str, str]]:
        bad: list[tuple[str, str, str, str]] = []
        for k, p in frozen_paths.items():
            now = _script_sha(p)
            if now != frozen[k]:
                bad.append((k, p.name, frozen[k], now))
        return bad

    log_lines: list[str] = [
        "第三问完整求解日志",
        f"解释器：{sys.executable}",
        f"第二问数据源：{source}" + ("（合成占位，骨架自检）" if source == "stub" else ""),
        f"输出根目录：{out_root}",
        f"开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    log_lines.append("── 源码冻结指纹 ──")
    for k, p in frozen_paths.items():
        log_lines.append(f"  {k:>18s}  {p.name:<34s} {frozen[k]}")
    log_lines.append("")

    t_all = time.time()
    timings: list[tuple[str, str, float, bool]] = []
    failed_step: str | None = None
    drift: dict[str, tuple[str, str]] = {}
    _told: set[str] = set()
    for num, name, _est, script in steps:
        for k, fn, b4, now in _drift_now():
            drift[k] = (b4, now)
            if k in _told:
                continue
            _told.add(k)
            msg = (f"⚠⚠ ★版本漂移：{fn} 在本次流水线运行期间被改写"
                   f"（{b4[:8]} → {now[:8]}）"
                   + (f"；[{k}] 这一步将加载**新**代码" if not k.startswith("lib:")
                      else ";此后所有步骤都加载**新**代码")
                   + "。此前已完成的步骤产物来自**旧**代码 ⇒ 本次全部产物"
                     "不能作为定稿；改码后必须重跑受影响的步骤及其全部下游"
                     "（本脚本退出码将为 2）。")
            print("\n" + msg + "\n", flush=True)
            log_lines.append(msg)
        ok, dt = run_step(num, name, script, log_lines, source=source)
        timings.append((num, name, dt, ok))
        if not ok:
            failed_step = num
            break

    for k, _fn, b4, now in _drift_now():
        drift.setdefault(k, (b4, now))

    total = time.time() - t_all

    log_lines.append("")
    log_lines.append("── 交付物检查 ──")
    print("\n" + "=" * 78 + "\n交付物检查\n" + "=" * 78, flush=True)
    missing: list[str] = []
    for p in deliverables:
        if p.exists():
            size = p.stat().st_size
            unit = f"{size / 1024:.1f} KiB" if size < 1024 ** 2 else f"{size / 1024 ** 2:.2f} MiB"
            line = f"  ✓ {p.relative_to(C.PROJECT_DIR)}  ({unit})"
        else:
            line = f"  ✗ 缺失：{p.relative_to(C.PROJECT_DIR)}"
            missing.append(str(p.relative_to(C.PROJECT_DIR)))
        print(line, flush=True)
        log_lines.append(line)

    anchors = read_anchors(out_root)
    if anchors:
        log_lines.append("")
        log_lines.append("── 关键数字锚点（来源：模型结果/第三问_年度汇总.csv）──")
        print("\n── 关键数字锚点 ──", flush=True)
        for k in ANCHOR_KEYS:
            if k in anchors:
                line = f"  {k:24s} = {anchors[k]}"
                print(line, flush=True)
                log_lines.append(line)

    log_lines.append("")
    log_lines.append("── 各步骤耗时 ──")
    for num, name, dt, ok in timings:
        log_lines.append(f"  {num}  {name:<24s} {_fmt_sec(dt):>10s}  {'成功' if ok else '失败'}")
    log_lines.append(f"  合计 {_fmt_sec(total)}")
    log_lines.append(f"结束时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    C.write_text_utf8(C.SOLVE_LOG_DIR / "第三问_完整求解日志.txt", "\n".join(log_lines))

    rep: list[str] = ["# 第三问最终版 端到端求解报告\n"]
    rep.append(f"- 解释器：`{sys.executable}`")
    rep.append(f"- 第二问数据源：`{source}`"
               + ("　⚠ **合成占位数据，本报告数值不可提交**" if source == "stub" else ""))
    rep.append(f"- 输出根目录：`{out_root}`")
    rep.append(f"- 执行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    rep.append(f"- 总耗时：**{_fmt_sec(total)}**")
    rep.append(f"- 结论：{'全部步骤成功 ✔' if failed_step is None else f'步骤 {failed_step} 失败 ✗'}\n")
    rep.append("## 1. 流水线\n")
    rep.append("| 序号 | 用途 | 脚本 | 耗时 | 状态 |")
    rep.append("|---|---|---|---|---|")
    for num, name, dt, ok in timings:
        script = dict((s[0], s[3]) for s in steps)[num].name
        rep.append(f"| {num} | {name} | `{script}` | {_fmt_sec(dt)} | {'✔' if ok else '✗'} |")
    rep.append("")
    rep.append("## 2. 交付物\n")
    rep.append("| 文件 | 状态 |")
    rep.append("|---|---|")
    for p in deliverables:
        rep.append(f"| `{p.relative_to(C.PROJECT_DIR)}` | "
                   f"{'✔ 存在' if p.exists() else '✗ 缺失'} |")
    rep.append("")
    if anchors:
        rep.append("## 3. 关键数字（年度汇总）\n")
        rep.append("| 指标 | 数值 |")
        rep.append("|---|---|")
        for k in ANCHOR_KEYS:
            if k in anchors:
                rep.append(f"| {k} | {anchors[k]} |")
        rep.append("")
    rep.append("## 4. 前置步骤与本脚本的关系\n")
    rep.append("本脚本只负责**编排**：所有建模逻辑都在 `01`–`11` 中，"
               "本脚本不修改任何中间结果，只读取 `模型结果/第三问_年度汇总.csv` "
               "与各交付物的存在性。若需换参数重跑，改 `_comm3.py` 常量后重跑本脚本即可。")
    rep.append("")
    rep.append("## 5. 源码版本冻结（★ 版本漂移守门）\n")
    rep.append(f"运行前已记录 {len(steps)} 个步骤脚本 + {n_lib} 个共享模块的 sha256；"
               "子进程在**派生瞬间**从磁盘加载 `.py`，若此刻源码已被改写，"
               "则该步跑新代码、此前步骤产物来自旧代码 ⇒ 最终产物由多版本拼成。\n")
    if drift:
        rep.append("### ✗ 检测到版本漂移：本次产物**不得作为定稿**\n")
        rep.append("| 被改写的脚本 | 运行前 | 运行后 |")
        rep.append("|---|---|---|")
        for k, (b4, now) in sorted(drift.items()):
            rep.append(f"| `{frozen_paths[k].name}`（{k}） | `{b4[:12]}` | `{now[:12]}` |")
        rep.append("")
        rep.append("> 处置：冻结代码后，重跑被改写脚本对应的步骤**及其全部下游**，"
                   "再跑 `14` 复核。本脚本退出码为 **2**。\n")
    else:
        rep.append("### ✔ 无漂移\n")
        rep.append("运行期间所有步骤脚本与共享模块的 sha256 未变化，"
                   "本次产物的每一部分都由**同一版本**代码生成。\n")
    rep.append("")
    rep.append("逐步 stdout/stderr 已落盘至 `日志/第三问_12_步骤NN日志.txt`。")
    C.write_text_utf8(C.REPORT_DIR / "第三题_端到端求解报告.md", "\n".join(rep))

    print(f"\n{'=' * 78}")
    print(f"流水线{'完成' if failed_step is None else '中断'}"
          f"：{'全部成功 ✔' if failed_step is None else f'步骤 {failed_step} 失败 ✗'}")
    print(f"总耗时 {_fmt_sec(total)}")
    print(f"已保存：求解日志/第三问_完整求解日志.txt")
    print(f"已保存：报告/第三题_端到端求解报告.md")
    if missing:
        print(f"⚠ 缺失交付物 {len(missing)} 个：{', '.join(missing)}")
    if drift:
        print(f"⚠⚠ ★版本漂移：{len(drift)} 个脚本/模块在运行期间被改写 ⇒ "
              f"本次产物**不得作为定稿**，请重跑受影响步骤及其全部下游")
        for k, (b4, now) in sorted(drift.items()):
            print(f"     {frozen_paths[k].name}（{k}）  {b4[:8]} → {now[:8]}")
    print(f"{'=' * 78}", flush=True)
    if failed_step is not None:
        return 1
    return 2 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
