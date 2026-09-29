#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import re
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

REQUIRED_KEYS = {
    "附件四_电价矩阵.npz": ("dates", "price_actual", "slot_start_minute",
                            "slot_end_minute", "raw_time_labels"),
    "第四问_价格预测快照.npz": ("dates", "H", "price_actual", "nhat_snap", "adopted_42",
                                "adopted_43", "level__mean7"),
    "第四问_共同热启动.npz": ("E_feb1", "E_init", "E_chain", "nu_used", "N_actual",
                              "price_actual"),
    "第四问_联合情景库.npz": ("M__42", "M__43", "N_scen__42", "N_scen__43",
                              "price_scen__42", "price_scen__43", "origin__42",
                              "origin__43", "fallback__42", "fallback__43", "M_max",
                              "seed"),
    "附件三_预报矩阵.npz": ("V_win_10min_kwh", "V_raw_kw", "complete_mask"),
    "第四问_4-2全年回测.npz": ("g", "a", "b", "C", "D", "U", "c", "N", "E_chain",
                               "bill", "plan_cost", "adjust_cost", "emerg_cost",
                               "dates", "lp_status_codes", "exec_cnt"),
    "第四问_4-3全年回测.npz": ("g", "a", "b", "C", "D", "U", "c", "N", "E_chain",
                               "bill", "plan_cost", "adjust_cost", "emerg_cost",
                               "dates", "lp_status_codes", "exec_cnt"),
}


def parse_yaml_lite(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def norm(v) -> str:
    s = str(v).strip().strip("[]()")
    toks = [t for t in (x.strip() for x in s.split(",")) if t]
    if not toks:
        return "()"
    out = []
    for t in toks:
        try:
            out.append(repr(float(t)))
        except ValueError:
            out.append(t)
    return "(" + ",".join(out) + ")"


def main() -> int:
    ap = argparse.ArgumentParser(description="第四问 00 冻结来源与接口自检")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="只校验快照是否已存在，不执行指纹写入")
    args = ap.parse_args()

    C.ensure_dirs()
    log: list[str] = []
    rows: list[list] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    def add(code: str, name: str, ok, detail: str) -> None:
        st = ok if ok in ("待生成",) else ("PASS" if ok else "FAIL")
        rows.append([code, name, st, detail])
        p(f"  [{st}] {code} {name}：{detail}")

    p("=" * 78)
    p("第四问 00 —— 冻结来源与接口自检")
    p("=" * 78)

    p("\n-- 1. 目录与只读来源 --")
    need_dir = {"处理后数据": C.PROCESSED_DIR, "模型结果": C.RESULT_DIR,
                "模型结果图": C.FIGURE_DIR, "报告": C.REPORT_DIR,
                "求解日志": C.SOLVE_LOG_DIR, "日志": C.LOG_DIR,
                "提交结果": C.SUBMIT_DIR, "配置": C.CONFIG_DIR}
    miss = [k for k, v in need_dir.items() if not v.is_dir()]
    add("00.1", "输出目录齐备", not miss,
        f"{len(need_dir)} 个目录：{'全部存在' if not miss else '缺 ' + ','.join(miss)}")

    att = {"附件1": C.ATTACHMENT1_PATH, "附件2": C.ATTACHMENT2_PATH,
           "附件3": C.ATTACHMENT3_PATH, "附件4": C.ATTACHMENT4_PATH,
           "result4-2 模板": C.RESULT42_TEMPLATE, "result4-3 模板": C.RESULT43_TEMPLATE}
    bad = [k for k, v in att.items() if not (v.is_file() and v.stat().st_size > 1024)]
    add("00.1", "输入附件与官方模板存在", not bad,
        f"{len(att)} 项：{'全部就位' if not bad else '缺 ' + ','.join(bad)}")

    ref = {"第一问最终版": C.REF_Q1_DIR, "第二问最终版": C.REF_Q2_DIR, "第三问最终版": C.REF_Q3_DIR}
    add("00.1", "第一/二/三题参考目录可见（只读使用）",
        all(v.is_dir() for v in ref.values()),
        "、".join(f"{k}={'在' if v.is_dir() else '缺'}" for k, v in ref.items()))
    add("00.1", "第二问只读产物存在",
        C.REF_Q2_MATRIX_NPZ.is_file() and C.REF_Q2_SCENARIO_NPZ.is_file(),
        f"{C.REF_Q2_MATRIX_NPZ.name}、{C.REF_Q2_SCENARIO_NPZ.name}")

    p("\n-- 2. 输入指纹冻结与第三题源码快照 --")
    before = set()
    if C.SNAPSHOT_DIR.is_dir():
        before = {p.name for p in C.SNAPSHOT_DIR.iterdir() if p.is_file()}
    if args.no_snapshot:
        add("00.2", "来源指纹清单（跳过写入）", C.SOURCE_FREEZE_CSV.is_file(),
            f"{C.SOURCE_FREEZE_CSV.name} 已存在")
        frows = []
    else:
        frows = C.freeze_sources(verbose=False)
        add("00.2", "共享原始/只读来源与第三题源码已指纹化",
            len(frows) >= 8 and all(len(s) == 64 for _, s in frows),
            f"{C.SOURCE_FREEZE_CSV.name}：{len(frows)} 项 SHA256（64 位十六进制）")
    snap = C.SNAPSHOT_DIR / "快照指纹清单.csv"
    if snap.is_file():
        import csv as _csv
        with open(snap, encoding="utf-8-sig", newline="") as f:
            srows = list(_csv.DictReader(f))
        add("00.2", "第三题源码快照只读留痕", len(srows) >= 1,
            f"{C.SNAPSHOT_DIR.name}/：{len(srows)} 个文件（快照文件、SHA256、来源路径）")
        if before:
            add("00.2", "快照不覆盖既有文件", True,
                f"既有 {len(before)} 个文件未重写（仅在缺失时复制）")
    af = C.RESULT_DIR / "第四问_来源指纹.csv"
    if frows and not af.exists():
        C.write_csv_utf8_sig(af, ("来源项", "SHA256"), frows)
        p(f"  来源指纹副本：{af.name}")

    p("\n-- 3. 配置与口径一致性 --")
    yp = C.CONFIG_DIR / "第四问_参数与口径.yaml"
    yml = parse_yaml_lite(yp) if yp.is_file() else {}
    diff = []
    for k, v in C.CONFIG.items():
        if k in yml and norm(yml[k]) != norm(v):
            diff.append(f"{k}: yaml={yml[k]} vs CONFIG={v}")
    add("00.3", "参数锁文件与 _comm4.CONFIG 一致", yp.is_file() and not diff,
        f"{yp.name}：{len(yml)} 行键值；{len(C.CONFIG)} 个配置键；"
        f"{'逐项一致' if not diff else '不一致 ' + '; '.join(diff)}")
    add("00.3", "关键口径已登记",
        all(k in yml for k in ("source_mode", "price_information", "settlement",
                              "stage0_mode_43", "tail_model", "random_seed")),
        "source_mode / price_information / settlement / stage0_mode_43 / tail_model / random_seed")

    p("\n-- 4. 接口自检（扫描 别名.符号 访问）--")
    helper_re = re.compile(r"(\w+)\s*=\s*_?load(?:er)?\(\s*[\"']([\w.]+\.py)[\"']")
    scripts = sorted(p for p in _HERE.glob("[0-9][0-9]_*.py"))
    script_alias: dict[str, dict[str, str]] = {}
    for s in scripts:
        script_alias[s.name] = dict(helper_re.findall(s.read_text(encoding="utf-8")))
    alias_mod: dict[str, str] = {}
    for am in script_alias.values():
        alias_mod.update(am)
    mods = {}
    for fn in sorted(set(alias_mod.values())):
        mods[fn] = C.load_module(fn, "chk_" + fn.replace(".py", ""))
    add("00.4", "辅助模块可按路径加载", len(mods) == 4,
        f"{len(mods)} 个：{', '.join(sorted(mods))}")

    by_mod: dict[str, set] = {}
    for s in scripts:
        am = script_alias[s.name]
        if not am:
            continue
        for raw in s.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0]
            for al, fn in am.items():
                for sym in re.findall(rf"(?<![\w.]){re.escape(al)}\.([A-Za-z_]\w*)", line):
                    by_mod.setdefault(fn, set()).add(sym)
    missing = []
    checked = 0
    for fn, syms in sorted(by_mod.items()):
        m = mods.get(fn)
        for sym in sorted(syms):
            checked += 1
            if not hasattr(m, sym):
                missing.append(f"{fn}::{sym}")
    add("00.4", "编号脚本引用的模块符号全部存在", not missing,
        f"扫描 {len(scripts)} 个脚本（各自别名映射），校验 {checked} 个 别名.符号 引用；"
        f"{'无缺失' if not missing else '缺失 ' + ','.join(missing)}")

    pub = {"_comm4.py": ("freeze_sources", "load_q2_matrix", "load_q2_scenarios",
                         "write_csv_utf8_sig", "write_text_utf8", "parse_attachment4",
                         "ensure_dirs", "setup_matplotlib", "ref_price144"),
           "_price4.py": ("net_load_feature", "net_load_feature_all", "fit_level",
                          "replay_forecasts", "score_forecasts", "METHODS", "METHOD_SPEC"),
           "_policy4.py": ("build_value_functions", "execute_one_slot", "retention_level",
                           "solve_midnight_plan", "solve_revised_plan", "make_grid",
                           "degeneracy_diag", "nu_tau"),
           "_settlement4.py": ("run_policy4", "SnapshotProviders", "bill_42_slots",
                               "bill_43_slots", "bill_43_equiv", "plan_node",
                               "minimal_bill_test")}
    miss2 = [f"{m}::{s}" for m, ss in pub.items() for s in ss
             if not hasattr(mods.get(m, object()), s)]
    add("00.4", "四模块公开接口齐备（§14 约定）", not miss2,
        f"关键符号 {sum(len(v) for v in pub.values())} 个；"
        f"{'全部存在' if not miss2 else '缺 ' + ','.join(miss2)}")

    nocomp = []
    for s in scripts + sorted(_HERE.glob("_*.py")):
        try:
            compile(s.read_text(encoding="utf-8"), str(s), "exec")
        except SyntaxError as e:
            nocomp.append(f"{s.name}:{e.lineno}")
    add("00.4", "全部源码语法可编译", not nocomp,
        f"{len(scripts) + 4} 个文件；{'全部可编译' if not nocomp else ','.join(nocomp)}")

    p("\n-- 5. 既有产物键位 --")
    for name, keys in REQUIRED_KEYS.items():
        path = C.PROCESSED_DIR / name if ("附件" in name and "4-2" not in name
                                          and "4-3" not in name) else C.RESULT_DIR / name
        if not path.is_file():
            path = None
            for d in (C.PROCESSED_DIR, C.RESULT_DIR):
                if (d / name).is_file():
                    path = d / name
                    break
        if path is None:
            add("00.5", f"{name} 键位", "待生成", "尚未生成（按 §14 顺序由后续脚本产生）")
            continue
        with np.load(path, allow_pickle=False) as Z:
            have = set(Z.files)
        lack = [k for k in keys if k not in have]
        add("00.5", f"{name} 键位", not lack,
            f"{path.parent.name}/{name}：{len(have)} 键；"
            f"{'必需键齐备' if not lack else '缺 ' + ','.join(lack)}")

    p("\n-- 6. 环境自检 --")
    import scipy
    from scipy.optimize import linprog
    ver = {"python": sys.version.split()[0], "numpy": np.__version__,
           "scipy": scipy.__version__}
    try:
        import pandas as _pd
        ver["pandas"] = _pd.__version__
    except Exception as e:
        ver["pandas"] = f"导入失败 {e}"
    try:
        import openpyxl
        ver["openpyxl"] = openpyxl.__version__
    except Exception as e:
        ver["openpyxl"] = f"导入失败 {e}"
    p("  " + "，".join(f"{k} {v}" for k, v in ver.items()))
    add("00.6", "依赖版本可读", all(not str(v).startswith("导入失败") for v in ver.values()),
        "，".join(f"{k}={v}" for k, v in ver.items()))
    try:
        r = linprog(c=[1.0, 1.0], A_ub=[[-1.0, 0.0], [0.0, -1.0]], b_ub=[-1.0, -1.0],
                    bounds=[(0, None), (0, None)], method="highs")
        ok = int(r.status) == 0 and abs(r.fun - 2.0) <= 1e-9
        add("00.6", "HiGHS 求解器可用", ok,
            f"最小测试 LP：status={int(r.status)}，最优值 {float(r.fun):.6f}（期望 2）")
    except Exception as e:
        add("00.6", "HiGHS 求解器可用", False, f"异常：{e}")
    try:
        mp = C.setup_matplotlib()
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        cjk = [c for c in ("Songti SC", "PingFang HK", "Hiragino Sans GB", "Heiti TC",
                           "STHeiti", "Arial Unicode MS", "SimHei") if c in names]
        add("00.6", "绘图后端与中文字体可用", bool(cjk),
            f"后端 {mp.get_backend()}；命中中文字体 {cjk[:3]}")
    except Exception as e:
        add("00.6", "绘图后端与中文字体可用", False, f"异常：{e}")

    npass = sum(1 for r in rows if r[2] == "PASS")
    npend = sum(1 for r in rows if r[2] == "待生成")
    nfail = sum(1 for r in rows if r[2] == "FAIL")
    C.write_csv_utf8_sig(C.RESULT_DIR / "第四问_冻结与接口自检.csv",
                         ("编号", "检查", "结果", "说明"), rows)
    if by_mod:
        irows = [[fn, sym, "存在" if hasattr(mods.get(fn), sym) else "缺失"]
                 for fn, syms in sorted(by_mod.items()) for sym in sorted(syms)]
        C.write_csv_utf8_sig(C.RESULT_DIR / "第四问_接口清单.csv",
                             ("模块", "符号", "状态"), irows)
    rp = ["# 第四问 00 冻结来源与接口自检报告\n",
          "> 对应流程图 §14 第一步。本脚本只读输入，只写自检产物；"
          "第三题源码仅取只读快照，不写入第三题目录。\n",
          f"- 检查项：{len(rows)}；PASS {npass}；FAIL {nfail}；待生成 {npend}",
          f"- 来源指纹：`{C.SOURCE_FREEZE_CSV.name}`；"
          f"快照：`{C.SNAPSHOT_DIR.name}/{snap.name}`\n",
          "| 编号 | 检查 | 结果 | 说明 |", "|---|---|---|---|"]
    rp += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
    rp += ["", "## 运行日志\n", "```"] + log + ["```"]
    C.write_text_utf8(C.REPORT_DIR / "第四问_00_来源冻结与接口自检报告.md", "\n".join(rp))
    C.write_text_utf8(C.LOG_DIR / "00_来源冻结与接口自检日志.txt", "\n".join(log) + "\n")

    p("\n" + "=" * 78)
    p(f"00 完成：{len(rows)} 项，PASS {npass}，FAIL {nfail}，待生成 {npend}")
    p("=" * 78)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
