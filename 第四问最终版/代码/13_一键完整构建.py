#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


C = _load("_comm4.py", "q4_comm")

P = C.PROJECT_DIR
R = C.RESULT_DIR
RP = C.REPORT_DIR


def step(sid: str, script: str, name: str, outs: list[Path], est: str = "",
         branch: str | None = None, always: bool = False,
         args_with_days: bool = False, args_suffix: bool = False,
         deprecated: bool = False) -> dict:
    return {"id": sid, "script": script, "name": name, "outs": outs, "est": est,
            "branch": branch, "always": always, "days": args_with_days,
            "suffix": args_suffix, "deprecated": deprecated}


STEPS = [
    step("00", "00_冻结来源与接口自检.py", "冻结来源与接口自检",
         [R / "第四问_冻结与接口自检.csv"], "≈2 s", always=True),
    step("01", "01_检查并处理附件四.py", "附件四校验与时间映射",
         [C.PRICE_MATRIX_NPZ, C.A3_FORECAST_NPZ], "≈10 s"),
    step("02", "02_共同热启动与历史预测回放.py", "共同热启动与历史预测回放",
         [C.WARMUP_NPZ], "≈1 s"),
    step("03", "03_价格预测校准与诊断.py", "价格预测校准与诊断",
         [C.PRICE_FORECAST_NPZ, C.PRICE_DIAG_CSV], "≈10 s"),
    step("04", "04_构建联合情景库.py", "构建联合情景库",
         [C.JOINT_SCENARIO_NPZ], "≈8 s"),
    step("05", "05_验证规划与价格DP内核.py", "验证规划与价格 DP 内核",
         [RP / "第四问_内核验证报告.md"], "≈228 s"),
    step("06", "06_运行第四问4-2.py", "第四问 4-2 全年闭环回测",
         [C.BACKTEST_42_NPZ], "≈554 s", branch="42"),
    step("07", "07_运行第四问4-3.py", "第四问 4-3 全年闭环回测",
         [C.BACKTEST_43_NPZ], "≈1160 s", branch="43"),
    step("08", "08_结算与年度汇总.py", "结算与年度/月度汇总",
         [C.YEARLY_CSV, C.MONTHLY_CSV, RP / "第四问_结算与年度汇总报告.md"],
         "≈15 s", branch="both"),
    step("09", "09_对照消融与敏感性.py", "对照消融与敏感性（已作废，默认跳过）",
         [C.ABLATION_CSV, RP / "第四问_对照与敏感性报告.md"],
         "≈1 h（并行 6 路）/ ≈4~5 h（单线程）", branch="both",
         args_with_days=True, args_suffix=True, deprecated=True),
    step("10", "10_指定日期明细与作图.py", "指定日期明细与作图",
         [C.SPEC_SLOTS_CSV, C.SPEC_DAY_CSV, C.SPEC_BATT_CSV, C.SPEC_EMERG_CSV,
          RP / "第四问_指定日期结果报告.md"], "≈40 s", branch="both"),
    step("11", "11_生成第四问结果文件.py", "生成 result4-2/4-3.xlsx 并回读",
         [C.RESULT42_XLSX, C.RESULT43_XLSX, R / "第四问_模板回读校验.csv"],
         "≈30 s", branch="both"),
    step("12", "12_独立验收.py", "独立验收 A01–A16",
         [C.ACCEPT_MD, R / "第四问_独立验收结果.csv"], "≈60 s", always=True),
]

ALL_IDS = [s["id"] for s in STEPS]


def branch_ok(req: str | None, branch: str) -> bool:
    if req is None:
        return True
    if branch == "both":
        return True
    return req == branch


def step_args(s: dict, a) -> list[str]:
    out: list[str] = []
    if s["days"] and a.days is not None:
        out += ["--days", str(a.days)]
    if s["suffix"] and a.suffix:
        out += ["--suffix", a.suffix]
    if s["id"] == "11":
        out += ["--branch", a.branch]
        if a.output_dir:
            out += ["--output-dir", a.output_dir]
    if s["id"] == "09" and not (a.days is not None or a.suffix):
        out += ["--exp", "all"]
    return out


def resume_skip(s: dict) -> tuple[bool, str]:
    if s["always"]:
        return False, "自检/验收步骤始终运行"
    src = _HERE / s["script"]
    smt = src.stat().st_mtime if src.exists() else 0.0
    miss = [o for o in s["outs"] if not o.exists()]
    if miss:
        return False, f"缺产物 {len(miss)} 个（{miss[0].name}）"
    old = [o for o in s["outs"] if o.stat().st_mtime < smt]
    if old:
        return False, f"产物早于脚本（{old[0].name}）"
    return True, f"{len(s['outs'])} 个产物已存在且不早于脚本"


def _experiments_09() -> list[str]:
    p9 = _HERE / "09_对照消融与敏感性.py"
    spec = importlib.util.spec_from_file_location("q4_ablation_list", p9)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["q4_ablation_list"] = mod
    spec.loader.exec_module(mod)
    return [str(e["name"]) for e in mod.make_experiments()]


def run_step_09_parallel(s: dict, a, lg: Path) -> int:
    names = _experiments_09()
    sub = C.LOG_DIR / "09_并行"
    sub.mkdir(parents=True, exist_ok=True)
    extra: list[str] = []
    if a.days is not None:
        extra += ["--days", str(a.days)]
    if a.suffix:
        extra += ["--suffix", a.suffix]
    script = _HERE / s["script"]
    abl = C.RESULT_DIR / (f"09_消融{a.suffix}" if a.suffix else "09_消融")

    def one(nm: str) -> tuple[str, int, float]:
        t1 = time.perf_counter()
        with open(sub / f"{nm}.txt", "wb") as fh:
            rc1 = subprocess.run([sys.executable, str(script), "--exp", nm] + extra,
                                 cwd=str(P), stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL).returncode
        return nm, rc1, time.perf_counter() - t1

    res: list[tuple[str, int, float, bool]] = []
    with ThreadPoolExecutor(max_workers=a.parallel_09) as pool:
        futs = [pool.submit(one, nm) for nm in names]
        for fu in as_completed(futs):
            nm, rc1, dt = fu.result()
            f_npz = abl / f"{nm}.npz"
            ok = f_npz.is_file() and f_npz.stat().st_size > 0
            res.append((nm, rc1, dt, ok))
            print(f"      · {nm:16s} 退出码 {rc1}  {dt:7.1f} s  "
                  f"轨迹{'已落盘' if ok else '缺失'}")

    miss = [nm for nm, _r, _d, ok in res if not ok]
    lg.parent.mkdir(parents=True, exist_ok=True)
    with open(lg, "w", encoding="utf-8") as f:
        f.write("$ 并行跑 09 全部实验，再 `--exp summary` 汇总\n")
        f.write(f"  并发 {a.parallel_09} 路；实验 {len(names)} 个；"
                f"参数 {' '.join(extra) or '（正式全量）'}\n\n")
        f.write("===== 子进程 =====\n")
        for nm, rc1, dt, ok in sorted(res, key=lambda x: -x[2]):
            f.write(f"  {nm:16s} 退出码 {rc1}  {dt:8.1f} s  "
                    f"轨迹{'已落盘' if ok else '缺失'}  日志 日志/09_并行/{nm}.txt\n")
        f.write("\n===== --exp summary =====\n\n")
        f.flush()
        rc2 = subprocess.run([sys.executable, str(script), "--exp", "summary"] + extra,
                             cwd=str(P), stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL).returncode
    if miss:
        print(f"      ✗ 实验轨迹缺失 {len(miss)} 个：{', '.join(miss)}")
        return 1 if rc2 == 0 else rc2
    return rc2


def main() -> int:
    ap = argparse.ArgumentParser(description="第四问 13 一键完整构建")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    ap.add_argument("--from", dest="start", default=None,
                    help=f"起始步骤（{'/'.join(ALL_IDS)}）")
    ap.add_argument("--only", default=None, help="仅运行列出的步骤，逗号分隔")
    ap.add_argument("--branch", choices=["42", "43", "both"], default="both",
                    help="分支选择：06 仅 42、07 仅 43；11 透传")
    ap.add_argument("--output-dir", default="", help="11 的输出目录（默认 提交结果）")
    ap.add_argument("--resume", action="store_true",
                    help="产物已存在且不早于脚本时跳过 01–11")
    ap.add_argument("--days", type=int, default=None, help="联调：限量天数（06/07/09）")
    ap.add_argument("--suffix", default="", help="联调：产物后缀（06/07/09）")
    ap.add_argument("--parallel-09", dest="parallel_09", type=int, default=6,
                    help="09 的并发度（实验粒度，默认 6；0/1 为顺序）")
    ap.add_argument("--continue-on-fail", dest="continue_on_fail",
                    action="store_true",
                    help="已知未通过项时仍继续后续步骤（最终退出码仍非 0；"
                         "仅在项目已披露该未通过项时使用，不得据此判定通过）")
    a = ap.parse_args()

    C.ensure_dirs()

    todo = list(STEPS)
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        bad = [x for x in want if x not in ALL_IDS]
        if bad:
            print(f"未知步骤：{bad}；可选 {ALL_IDS}")
            return 2
        todo = [s for s in STEPS if s["id"] in want]
    elif a.start:
        if a.start not in ALL_IDS:
            print(f"未知起始步骤 {a.start}；可选 {ALL_IDS}")
            return 2
        todo = [s for s in STEPS if ALL_IDS.index(s["id"]) >= ALL_IDS.index(a.start)]
        _dep = [s["id"] for s in todo if s.get("deprecated")]
        if _dep:
            todo = [s for s in todo if not s.get("deprecated")]
            print(f"[提示] 默认跳过已作废步骤 {_dep}（其产物已作废，"
                  "需重跑请显式 --only 09）")
    else:
        _dep = [s["id"] for s in todo if s.get("deprecated")]
        if _dep:
            todo = [s for s in todo if not s.get("deprecated")]
            print(f"[提示] 默认跳过已作废步骤 {_dep}（其产物已作废，"
                  "需重跑请显式 --only 09）")

    log: list[str] = []
    rows: list[list] = []
    runs: list[tuple[str, int, float, str]] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 78)
    p("第四问 13 —— 一键完整构建（§14 顺序）")
    p("=" * 78)
    p(f"  分支 {a.branch}；步骤 {len(todo)} 个；resume={a.resume}；"
      f"days={a.days if a.days is not None else '全年'}；"
      f"suffix={a.suffix or '（无）'}；output-dir={a.output_dir or '提交结果'}；"
      f"09 并发={a.parallel_09}；continue-on-fail={a.continue_on_fail}")
    p(f"  执行 {len(todo)} 个步骤：{', '.join(s['id'] for s in todo)}")
    p("")

    plan: list[tuple[dict, list[str], str]] = []
    for s in todo:
        if not branch_ok(s["branch"], a.branch):
            plan.append((s, [], "跳过：分支不匹配"))
            continue
        if a.resume:
            sk, why = resume_skip(s)
            if sk:
                plan.append((s, [], f"跳过（resume）：{why}"))
                continue
        plan.append((s, step_args(s, a), "运行"))

    if a.dry_run:
        p(f"{'序号':<4} {'脚本':<28} {'状态':<10} {'预估':<9} 命令")
        p("-" * 78)
        for s, argv, st in plan:
            cmd = " ".join([Path(sys.executable).name, s["script"]] + argv)
            p(f"{s['id']:<4} {s['script']:<28} {st:<10} {s['est']:<9} {cmd}")
        p("")
        p("（--dry-run：未执行任何步骤）")
        return 0

    t_all = time.perf_counter()
    for s, argv, st in plan:
        if st != "运行":
            p(f"[{s['id']}] {s['name']}：{st}")
            rows.append([s["id"], s["name"], s["script"], st, 0, "0.0", ""])
            continue
        script = _HERE / s["script"]
        if not script.is_file():
            rows.append([s["id"], s["name"], s["script"], "失败：脚本缺失", 1, "0.0", ""])
            p(f"[{s['id']}] {s['name']}：脚本缺失 {s['script']}")
            break
        lg = C.LOG_DIR / f"13_{s['id']}_{script.stem}.txt"
        cmd = [sys.executable, str(script)] + argv
        p(f"[{s['id']}] {s['name']} —— {script.name} {' '.join(argv)}")
        t0 = time.perf_counter()
        if s["id"] == "09" and a.parallel_09 >= 2:
            p(f"      并行 {a.parallel_09} 路跑实验，末段 `--exp summary` 汇总"
              "（子进程日志 日志/09_并行/*.txt）")
            rc = run_step_09_parallel(s, a, lg)
        else:
            with open(lg, "w", encoding="utf-8") as f:
                f.write("$ " + " ".join(cmd) + "\n\n")
                f.flush()
                rc = subprocess.run(cmd, cwd=str(P), stdout=f,
                                    stderr=subprocess.STDOUT).returncode
        wall = time.perf_counter() - t0
        tail = ""
        if lg.exists():
            lines = [x.rstrip() for x in lg.read_text(encoding="utf-8",
                                                      errors="ignore").splitlines() if x.strip()]
            tail = " / ".join(lines[-2:])[:180]
        runs.append((s["id"], rc, wall, lg.name))
        p(f"      退出码 {rc}；耗时 {wall:.1f} s；日志 日志/{lg.name}")
        if tail:
            p(f"      末尾：{tail}")
        p("")
        ok_out = all(o.exists() for o in s["outs"])
        status = "PASS" if rc == 0 and ok_out else ("FAIL" if rc != 0 else "FAIL：产物缺失")
        rows.append([s["id"], s["name"], s["script"], status, rc, f"{wall:.1f}", lg.name])
        if status != "PASS":
            if a.continue_on_fail:
                p(f"  ⚠ 步骤 {s['id']} 未通过（已记录，不得判定为通过）；"
                  "按 --continue-on-fail 继续后续步骤，最终退出码仍为非 0")
                continue
            p(f"  ✗ 步骤 {s['id']} 未通过，编排中止（修复后可用 --from {s['id']} 续跑；"
              "若该未通过项已披露且需跑完其余步骤，可加 --continue-on-fail）")
            break

    t_wall = time.perf_counter() - t_all
    nrun = len(runs)
    nfail = sum(1 for r in rows if not str(r[3]).startswith("PASS") and
                not str(r[3]).startswith("跳过"))
    p("=" * 78)
    p(f"13 结束：计划 {len(plan)} 步，实跑 {nrun} 步，未通过 {nfail} 步，"
      f"总耗时 {t_wall:.1f} s")
    for sid, rc, wall, lgf in runs:
        p(f"  {sid}: 退出码 {rc}，{wall:.1f} s，日志 日志/{lgf}")
    p("=" * 78)

    C.write_csv_utf8_sig(R / "第四问_构建步骤清单.csv",
                         ("步骤", "名称", "脚本", "结果", "退出码", "耗时/s", "日志"),
                         rows)
    rp = ["# 第四问 一键完整构建报告（13）\n",
          f"> 顺序：{' → '.join(ALL_IDS)}（§14）。分支 `{a.branch}`，"
          f"resume={a.resume}，days={a.days if a.days is not None else '全年'}"
          f"，suffix={a.suffix or '（无）'}。\n",
          f"- 计划步骤：{len(plan)}；实跑：{nrun}；未通过：{nfail}；"
          f"总耗时 {t_wall:.1f} s",
          f"- 步骤明细：`模型结果/第四问_构建步骤清单.csv`\n",
          "| 步骤 | 名称 | 脚本 | 结果 | 退出码 | 耗时/s | 日志 |",
          "|---|---|---|---|---|---|---|"]
    rp += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in rows]
    rp += ["", "## 运行日志\n", "```"] + log + ["```"]
    C.write_text_utf8(RP / "第四问_一键完整构建报告.md", "\n".join(rp))
    C.write_text_utf8(C.LOG_DIR / "13_一键完整构建日志.txt", "\n".join(log) + "\n")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
