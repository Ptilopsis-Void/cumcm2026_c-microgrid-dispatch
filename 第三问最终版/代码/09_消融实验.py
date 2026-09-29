#!/usr/bin/env python

from __future__ import annotations

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

T = C.PERIODS_PER_DAY
TAUS = list(C.TAU_HOURS)

ARM_SPECS = [
    ("S0", "S0", "只用 0:00 计划，$q\\equiv p$，调整费恒为 0"),
    ("S06", "S06", "6:00 引入新预报"),
    ("S0612", "S0612", "6:00、12:00 引入"),
    ("S061218", "S061218", "题面完整档（= `08` 正式主结果的档位）"),
    ("S_all+_UB", "S_all_plus", "18:00 用完美信息（**保守采纳**参照臂：同时提交常规候选）"),
    ("S_all+_raw", "S_all+_raw", "18:00 **强制**采纳完美候选（反事实诊断；跳过接受检验）"),
    ("PF", "PF", "全天完美信息（用附件 2 实际光伏，不可实现的下界）"),
]


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t00 = time.perf_counter()

    log("=" * 78)
    log("第三问 09 —— 消融实验：是否需要引入其他时刻的预报")
    log("=" * 78)

    P3 = _load("_policy3.py", "q3_policy")
    f = P3.facts()
    Z = C.Q2.matrix()
    price = np.asarray(Z["price"], float)
    sc = np.asarray(f["score_day_index"], int)
    M = int(np.asarray(f["scen_L"]).shape[1])
    nu = C.NU_VALUE

    log(f"情景数 M = {M}；ν = {nu:.10f}；评分期 {sc.size} 天")

    _maxd = int(os.environ.get("Q3_ABL_MAX_DAYS", "0") or 0)

    def _progress(msg: str) -> None:
        log(msg)

    _kw: dict = {"log": _progress, "progress_every": 20}
    if _maxd > 0:
        _kw["days"] = list(range(int(sc[0]), int(sc[0]) + _maxd))
        log("")
        log("⚠⚠⚠ 骨架冒烟模式：Q3_ABL_MAX_DAYS = "
            f"{_maxd}（仅统计前 {_maxd} 个评分日）")
        log("      该模式下的任何数字都不得写入论文、不得提交、不得作为建模结论。")
        log("      正式运行请清除该环境变量。")
        log("")
    log("P0-4：六档与 `08` 共用唯一策略入口 `_policy3.run_policy`，故")
    log("      「消融完整档」与「正式主结果」必须逐位一致（下方现场复核）。")
    log("P0-5：模型内期望口径统一为**共同 ex-ante 测度**（τ=0 情景集）；")
    log("      各更新节点须通过「接受检验」（新解在共同尺子上不得更贵），")
    log("      故嵌套档位严格单调非增——这是 q≡p 永远可行的直接结果。")

    results: dict[str, dict] = {}
    raws: dict[str, dict] = {}
    t_all = time.perf_counter()

    for disp, engine, note in ARM_SPECS:
        t1 = time.perf_counter()
        log("")
        log(f"── 档 {disp}（引擎档名 `{engine}`）：{note} ──")
        r = P3.run_arm(engine, **_kw)
        s = P3.summarise(r, label=disp)
        s["说明"] = note
        log(f"  ↑ 该档用时 {time.perf_counter() - t1:.1f} s")
        s["引擎档名"] = engine
        s["紧急购电时段数"] = int(np.sum(np.asarray(r["b"], float)[sc] > 1e-9))
        s["拒绝新解段数"] = int(r["info"]["n_declined_seg"])
        s["完美候选被压段数"] = int(r["info"].get("n_declined_perf_seg", 0))
        s["强制采纳段数"] = int(r["info"].get("n_forced_seg", 0))
        s["更新时刻"] = ",".join(str(t) for t in r["update_times"]) or "无(仅0:00)"
        s["期望口径计划调整费_元"] = s["总费用_元"] - s["紧急购电费_元"]
        s["期望口径紧急费_元"] = (s["模型内期望总费用_元"]
                                  - s["期望口径计划调整费_元"])
        s["期望口径总费用_元"] = s["模型内期望总费用_元"]
        s["期望口径紧急量_kWh"] = float("nan")
        s["期望口径尺子"] = ("不适用（该档用 M=1 真实轨迹，与其它档不同尺子）"
                             if engine == "PF"
                             else "共同 ex-ante 测度（τ=0 情景集）")
        results[disp] = s
        raws[disp] = r
        log(f"  实现口径 {s['总费用_元']:,.2f} 元 = 计划 {s['计划购电费_元']:,.2f}"
            f" + 调整 {s['调整相关费用_元']:,.2f} + 紧急 {s['紧急购电费_元']:,.2f}")
        log(f"  期望口径 {s['模型内期望总费用_元']:,.2f} 元（共同测度）；"
            f"Σq {s['调整购电量_kWh']:,.2f} kWh；Σb {s['紧急购电量_kWh']:,.2f} kWh")
        log(f"  闭环残差 {s['闭环残差最大值']:.3e}；信息泄露 {s['信息泄露时段数']}；"
            f"拒绝新解段数 {s['拒绝新解段数']}；用时 {time.perf_counter() - t1:.1f} s")

    log("")
    log(f"六档实验总用时 {time.perf_counter() - t_all:.1f} s")

    log("")
    log("── P0-4 验收：消融完整档 ≡ 正式主结果（同一入口 run_policy）──")
    _ref = P3.run_policy(update_times=(6, 12, 18), **_kw)
    _arm = raws["S061218"]
    _d = {
        "p": float(np.abs(np.asarray(_arm["P"], float)
                          - np.asarray(_ref["P"], float)).max()),
        "q": float(np.abs(np.asarray(_arm["Q"], float)
                          - np.asarray(_ref["Q"], float)).max()),
        "E": float(np.nanmax(np.abs(np.asarray(_arm["E"], float)
                                    - np.asarray(_ref["E"], float)))),
        "b": float(np.abs(np.asarray(_arm["b"], float)
                          - np.asarray(_ref["b"], float)).max()),
    }
    _dmax = max(_d.values())
    log(f"  max|Δp| = {_d['p']:.3e}；max|Δq| = {_d['q']:.3e}；"
        f"max|ΔE| = {_d['E']:.3e}；max|Δb| = {_d['b']:.3e}")
    log(f"  阈值 1e-6 元 / 1e-6 kWh / 1e-8 kWh(SOC) → "
        f"{'✔ 通过（Δ = 0，逐位一致）' if _dmax == 0.0 else ('✔ 通过' if _dmax < 1e-6 else '✘ 未通过')}")
    log(f"  完整档两口径账单：实现 {results['S061218']['总费用_元']:,.6f} 元；"
        f"期望 {results['S061218']['模型内期望总费用_元']:,.6f} 元")

    order = ["S0", "S06", "S0612", "S061218", "S_all+_UB", "S_all+_raw", "PF"]
    base = results["S0"]["总费用_元"]
    base_e = results["S0"]["期望口径总费用_元"]
    n_day_used = int(_maxd) if _maxd > 0 else int(sc.size)
    rows = []
    inc_rows = []
    for nm in order:
        r = results[nm]
        rows.append([r["档位"], r["说明"],
                     f"{r['调整购电量_kWh']:.4f}", f"{r['下调量_kWh']:.4f}",
                     f"{r['上调量_kWh']:.4f}", f"{r['计划购电费_元']:.4f}",
                     f"{r['下调违约金_元']:.4f}", f"{r['上调加价_元']:.4f}",
                     f"{r['调整相关费用_元']:.4f}", f"{r['紧急购电费_元']:.4f}",
                     f"{r['紧急购电量_kWh']:.4f}", f"{r['紧急购电时段数']}",
                     f"{r['总费用_元']:.4f}",
                     f"{r['期望口径计划调整费_元']:.4f}",
                     f"{r['期望口径紧急费_元']:.4f}",
                     f"{r['期望口径总费用_元']:.4f}",
                     r["闭环残差最大值"],
                     f"{r['信息泄露时段数']}",
                     f"{r['拒绝新解段数']}",
                     f"{r['完美候选被压段数']}",
                     f"{r['强制采纳段数']}",
                     r["期望口径尺子"],
                     str(n_day_used)])
        inc_rows.append([r["档位"], f"{r['总费用_元']:.4f}",
                         f"{base - r['总费用_元']:+.4f}",
                         f"{100 * (base - r['总费用_元']) / base:+.6f}",
                         f"{r['期望口径总费用_元']:.4f}",
                         f"{base_e - r['期望口径总费用_元']:+.4f}",
                         f"{100 * (base_e - r['期望口径总费用_元']) / base_e:+.6f}",
                         f"{r['拒绝新解段数']}"])
    C.write_csv_utf8_sig(
        C.ABLATION_CSV,
        ["档", "说明", "调整购电量_kWh", "下调量_kWh", "上调量_kWh",
         "计划购电费_元", "下调违约金_元", "上调加价_元", "调整相关费用_元",
         "紧急购电费_元", "紧急购电量_kWh", "紧急购电时段数", "总费用_元",
         "期望口径计划调整费_元", "期望口径紧急费_元", "期望口径总费用_元",
         "闭环残差最大值", "信息泄露时段数", "拒绝新解段数",
         "完美候选被压段数", "强制采纳段数", "期望口径尺子", "评分日数"],
        rows)
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_消融实验增量表.csv",
        ["档", "总费用_元", "相对S0节省_元", "相对S0节省_%",
         "期望口径总费用_元", "期望口径相对S0节省_元", "期望口径相对S0节省_%",
         "拒绝新解段数"],
        inc_rows)

    log("")
    log("── 消融实验对照表（实现口径）──")
    log(f"  {'档':<10s}{'总费用 元':>16s}{'调整费 元':>14s}"
        f"{'紧急费 元':>14s}{'较S0节省 元':>16s}{'相对':>10s}")
    for nm in order:
        r = results[nm]
        log(f"  {nm:<10s}{r['总费用_元']:>16,.2f}{r['调整相关费用_元']:>14,.2f}"
            f"{r['紧急购电费_元']:>14,.2f}{base - r['总费用_元']:>+16,.2f}"
            f"{100 * (base - r['总费用_元']) / base:>9.4f}%")

    log("")
    log("── 消融实验对照表（模型期望口径，与实现口径逐时段同源）──")
    log(f"  {'档':<10s}{'期望账单 元':>16s}{'期望计划+调整':>16s}"
        f"{'期望紧急费 元':>16s}{'较S0节省 元':>16s}{'相对':>10s}")
    for nm in order:
        r = results[nm]
        log(f"  {nm:<10s}{r['期望口径总费用_元']:>16,.2f}"
            f"{r['期望口径计划调整费_元']:>16,.2f}"
            f"{r['期望口径紧急费_元']:>16,.2f}"
            f"{base_e - r['期望口径总费用_元']:>+16,.2f}"
            f"{100 * (base_e - r['期望口径总费用_元']) / base_e:>9.4f}%")

    NESTED = ["S0", "S06", "S0612", "S061218"]
    tot_list = [results[nm]["总费用_元"] for nm in order]
    exp_list = [results[nm]["期望口径总费用_元"] for nm in order]
    _ni = [order.index(nm) for nm in NESTED]
    _nested_exp = [exp_list[i] for i in _ni]
    _nested_tot = [tot_list[i] for i in _ni]
    mono_exp_nested = all(_nested_exp[i] >= _nested_exp[i + 1] - 1e-6
                          for i in range(len(NESTED) - 1))
    mono_tot_nested = all(_nested_tot[i] >= _nested_tot[i + 1] - 1e-6
                          for i in range(len(NESTED) - 1))
    SEQ6 = NESTED + ["S_all+_UB", "PF"]
    SEQ5 = NESTED + ["S_all+_UB"]
    _t6 = [results[nm]["总费用_元"] for nm in SEQ6]
    _e5 = [results[nm]["期望口径总费用_元"] for nm in SEQ5]
    mono_impl = all(_t6[i] >= _t6[i + 1] - 1e-6 for i in range(len(SEQ6) - 1))
    mono_exp = all(_e5[i] >= _e5[i + 1] - 1e-6 for i in range(len(SEQ5) - 1))
    _i_s612, _i_18 = order.index("S0612"), order.index("S061218")
    _i_ub, _i_pf = order.index("S_all+_UB"), order.index("PF")
    _i_raw = order.index("S_all+_raw")
    ub_vs_612 = exp_list[_i_ub] <= exp_list[_i_s612] + 1e-6
    ub_degenerate = abs(exp_list[_i_ub] - exp_list[_i_18]) < 1e-6
    log("")
    log("── 两个口径下的单调性校验 ──")
    log(f"  ① 期望口径 · **嵌套链** {' ≥ '.join(NESTED)}"
        f"（共同 ex-ante 测度，**由构造保证**）："
        f"{'✔ 满足' if mono_exp_nested else '✘ 未满足（实现缺陷）'}")
    log("     差额 = 前档 − 后档，**正值表示后档更省**：")
    for _i in range(len(NESTED) - 1):
        _a, _b = order.index(NESTED[_i]), order.index(NESTED[_i + 1])
        log(f"     {NESTED[_i]:<9s} → {NESTED[_i + 1]:<9s} "
            f"期望 {exp_list[_a] - exp_list[_b]:+14,.4f} 元"
            f"   实现 {tot_list[_a] - tot_list[_b]:+14,.4f} 元")
    log(f"  ② 信息优势臂 · **可证上界**：候选集(S_all+) ⊇ 候选集(S0612) ⇒ "
        f"S_all+_UB {exp_list[_i_ub]:,.2f} ≤ S0612 {exp_list[_i_s612]:,.2f}？"
        f"{'✔' if ub_vs_612 else '✘'}")
    log(f"     S_all+_UB vs S061218 的差（**不可证**，仅作为同一尺子下的"
        f"可比较观测报告）：{exp_list[_i_ub] - exp_list[_i_18]:+,.2f} 元")
    log("  ③ 下界档 PF：期望口径用 **M=1 真实轨迹**（另一把尺子），"
        "与其它档**不可比**；只参与实现口径比较。")
    log(f"  ④ 实现口径六档排序（单条真实轨迹的一次抽样，**非构造性质**）："
        f"{'✔ 本次运行恰好单调' if mono_impl else '⚠ 本次运行非单调（可接受，见报告 §5）'}"
        f"；其中嵌套链{'✔ 单调' if mono_tot_nested else '⚠ 非单调'}")
    log(f"  S0 → S061218 累计：期望 {base_e - exp_list[_i_18]:+,.2f} 元"
        f"（{100 * (base_e - exp_list[_i_18]) / base_e:+.4f}%）；"
        f"实现 {base - tot_list[_i_18]:+,.2f} 元"
        f"（{100 * (base - tot_list[_i_18]) / base:+.4f}%）")
    log(f"  τ=18 预报的增量（**同一基线 S0612**，同尺子可直接比较）："
        f"期望 {exp_list[_i_s612] - exp_list[_i_18]:+,.2f} 元"
        f"（{100 * (exp_list[_i_s612] - exp_list[_i_18]) / base_e:+.4f}%）；"
        f"实现 {tot_list[_i_s612] - tot_list[_i_18]:+,.2f} 元")
    log(f"  同一基线下的「换成完美信息」参照臂 S0612 → S_all+_UB："
        f"期望 {exp_list[_i_s612] - exp_list[_i_ub]:+,.2f} 元"
        f"（{100 * (exp_list[_i_s612] - exp_list[_i_ub]) / base_e:+.4f}%）；"
        f"实现 {tot_list[_i_s612] - tot_list[_i_ub]:+,.2f} 元")
    log("     ★ 参照臂与实得增量**必须从同一基线（S0612）出发**才有可比性；"
        "把「S061218 → S_all+_UB」当作信息价值上界是**口径错误**"
        "（那是「换掉一个候选」的差，符号可正可负）。")
    if ub_degenerate:
        log(f"     ⚠ 实测约束：S_all+_UB 的完美候选在共同尺子上被压 "
            f"{results['S_all+_UB']['完美候选被压段数']} 段"
            f"（≥ 需先验），故该臂最终决策与 S061218 **逐位相同**。"
            f"因此上面那行的增量**不是**完美信息的贡献，而是**常规** 18:00 "
            f"预报的贡献；「额外增益 = 0」≠「完美信息价值 = 0」。")
    log(f"  ⑤ **反事实臂** S_all+_raw（18:00 无条件强制采纳完美候选，"
        f"{results['S_all+_raw']['强制采纳段数']} 段）："
        f"期望 {exp_list[_i_raw]:,.2f} 元 vs 同基线 S0612 "
        f"{exp_list[_i_s612]:,.2f} 元（差 {exp_list[_i_s612] - exp_list[_i_raw]:+,.2f} 元）；"
        f"实现 {tot_list[_i_raw]:,.2f} 元 vs S0612 {tot_list[_i_s612]:,.2f} 元"
        f"（差 {tot_list[_i_s612] - tot_list[_i_raw]:+,.2f} 元）")
    log("     ★ 该臂**跳过接受检验**，故意不在嵌套链内：差值符号可正可负，"
        "不得作为上界或单调性证据；它回答的是「完全信任 18:00 完美预报的后果」。")
    log(f"  S061218 → PF（全部完美信息）：实现口径 "
        f"{tot_list[_i_18] - tot_list[_i_pf]:+,.2f} 元"
        f"（单轨迹抽样；PF 的期望口径换尺子，不参与比较）")
    log("")
    log("  各档「拒绝新解段数」（P0-5：新解在共同尺子上更贵时维持旧计划），"
        "及「完美候选被压段数」（信息优势臂的透明度诊断）：")
    for nm in order:
        log(f"    {nm:<11s} 拒绝 {results[nm]['拒绝新解段数']:>6d} 段"
            f"    完美候选被压 {results[nm]['完美候选被压段数']:>6d} 段")
    log("")
    log("  ★ 期望口径的节省是“调整期权的价值”，由各发布时刻的信息集决定；")
    log("    实现口径是单条实际轨迹的一次抽样，受少数高价时段的紧急购电主导，")
    log("    不必单调（但本引擎的接受检验也使它在本次运行中恰好单调）。")
    log("    两者大小差异说明：把 5 倍电价的应急电量作为风险预算")
    log("    比花 1.5 倍加价提前买保险更划算 —— 即“调整的主要作用是把缺口")
    log("    保持在可控范围内，而不是把缺口降到零”。")

    C.setup_matplotlib()
    C.setup_chinese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.6))
    names = [{"S0": "$S_0$", "S06": "$S_{06}$", "S0612": "$S_{0612}$",
              "S061218": "$S_{061218}$", "S_all+_UB": r"$S_{all+}^{UB}$",
              "S_all+_raw": r"$S_{all+}^{\rm raw}$",
              "PF": "$PF$"}[nm] for nm in order]
    pl = np.array([results[nm]["计划购电费_元"] for nm in order]) / 1e4
    ad = np.array([results[nm]["调整相关费用_元"] for nm in order]) / 1e4
    em = np.array([results[nm]["紧急购电费_元"] for nm in order]) / 1e4
    x = np.arange(len(order))
    ax = axes[0]
    ax.bar(x, pl, 0.6, label="计划购电费", color="#4C78A8")
    ax.bar(x, ad, 0.6, bottom=pl, label="调整相关费用", color="#F2CF5B")
    ax.bar(x, em, 0.6, bottom=pl + ad, label="紧急购电费", color="#E45756")
    for i in range(len(order)):
        ax.text(i, pl[i] + ad[i] + em[i], f"{(pl[i] + ad[i] + em[i]):.2f}万",
                ha="center", va="bottom", fontsize=9)
        if i > 0:
            ax.text(i, (pl[i] + ad[i] + em[i]) / 2,
                    f"{(base - results[order[i]]['总费用_元']) / 1e4:+.2f}万",
                    ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(x, names)
    ax.set_ylabel("费用（万元）")
    ax.set_title("（a）实现口径：实际执行账单三分解")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ei = np.array([results[nm]["期望口径总费用_元"] for nm in order]) / 1e4
    ax2.plot(x, tot_list, "o-", color="#4C78A8", label="实现口径总费用", lw=2)
    _cmp_i = ([order.index(nm) for nm in NESTED]
              + [order.index("S_all+_UB"), order.index("S_all+_raw")])
    ax2.plot(x[_cmp_i], ei[_cmp_i], "s--", color="#E45756",
             label="模型期望口径账单（共同尺子）", lw=2)
    for i in _cmp_i:
        ax2.annotate(f"{ei[i]:.2f}", (x[i], ei[i]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#E45756")
    ax2.annotate("反事实：强制采信完美信息\n（不在嵌套链内）",
                 (x[order.index("S_all+_raw")],
                  ei[order.index("S_all+_raw")]),
                 textcoords="offset points", xytext=(-10, -30), ha="right",
                 fontsize=7, color="#B279A2")
    for i in range(len(order)):
        ax2.annotate(f"{tot_list[i] / 1e4:.2f}", (x[i], tot_list[i] / 1e4),
                     textcoords="offset points", xytext=(0, -14), ha="center",
                     fontsize=8, color="#4C78A8")
    _ipf2 = order.index("PF")
    ax2.annotate("期望口径不适用\n（该档换尺子）", (x[_ipf2], ei[_ipf2]),
                 textcoords="offset points", xytext=(-8, 10), ha="right",
                 fontsize=7, color="#B279A2")
    ax2.set_xticks(x, names)
    ax2.set_ylabel("全年费用（万元）")
    ax2.set_title("（b）两种口径对比：嵌套链上信息越多、期望账单越低")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig.suptitle("第三问 消融实验：引入不同时刻预报的费用对照"
                 "（2025-02-01 … 2025-12-31）", fontsize=12)
    fig.tight_layout()
    C.save_figure(fig, "第三问_消融实验")
    plt.close(fig)

    np.savez_compressed(
        C.RESULT_DIR / "第三问_消融实验.npz",
        order=np.asarray(order, dtype="<U12"),
        total=np.asarray([results[nm]["总费用_元"] for nm in order], float),
        plan=np.asarray([results[nm]["计划购电费_元"] for nm in order], float),
        adj=np.asarray([results[nm]["调整相关费用_元"] for nm in order], float),
        emerg=np.asarray([results[nm]["紧急购电费_元"] for nm in order], float),
        emerg_kwh=np.asarray([results[nm]["紧急购电量_kWh"] for nm in order], float),
        exp_total=ei * 1e4,
        exp_plan_adj=np.asarray([results[nm]["期望口径计划调整费_元"]
                                 for nm in order], float),
        exp_emerg=np.asarray([results[nm]["期望口径紧急费_元"]
                              for nm in order], float),
        exp_comparable=np.asarray([nm != "PF" for nm in order], bool),
    )

    rep = []
    rep.append("# 第三问 消融实验报告 —— 是否需要引入其他时刻的预报\n")
    rep.append("- 代码：`第三问最终版/代码/09_消融实验.py`")
    rep.append(f"- 评分期 {len(sc)} 天；情景数 $M$ = {M}；$\\nu$ = {nu:.10f}")
    _mode = C.Q2.mode()
    rep.append(f"- 数据源模式 `Q3_Q2_SOURCE` = **{_mode}**")
    if _mode != "real":
        rep.append("")
        rep.append("> ⚠⚠ **本报告使用非 `real` 数据源，数值不具任何建模意义。**")
        rep.append("> `stub` 用**合成占位负荷**搭配**真实附件三光伏预报**，两者来源不一致，"
                   "会使联合净负荷分布失真，进而产生「增加信息反而更贵」等反常态。")
        rep.append("> 本报告在此模式下**仅用于验证代码连通性与报告生成管线**，"
                   "**报告中的任何数字都不得写入论文，也不得据此判断模型优劣**。")
        rep.append("> 正式结论必须在 `Q3_Q2_SOURCE=real` 下重跑 `09` 取得。")
    rep.append("")
    rep.append("## 1. 问题与理论前提\n")
    rep.append("题目最后一问要求判断**是否需要引入其他时刻的预报**来制定调整购电策略。\n")
    rep.append("**理论前提（决定了问题的形态）**：调整机制允许\"不用\"新预报"
               "（取 $q\\equiv p$，此时调整费恒为 0），因此**决策集随预报集合单调扩张**，"
               "最优费用关于预报集合**单调不增**。\n")
    rep.append("$$\\mathcal T_1\\subseteq\\mathcal T_2\\;\\Longrightarrow\\;"
               "K^*(\\mathcal T_1)\\ge K^*(\\mathcal T_2)$$\n")
    rep.append("所以本题答案不在于\"要不要\"（符号必为 $\\ge 0$，这是免费的期权），"
               "而在于\"**值多少**\"（量级）。下面的消融实验就是这个量级的测量。\n")
    rep.append("> ★ **落地到本引擎时的可证性边界**：上式要求「同一尺子 + 候选集嵌套」。"
               "本引擎据此只能对**嵌套链** "
               "$S_0\\subseteq S_{06}\\subseteq S_{0612}\\subseteq S_{061218}$ "
               "给出**可证**的单调非增；信息优势臂 $S_{all+}^{UB}$（候选集变了）"
               "与 $PF$（换了一把尺子）**不在**该链内，"
               "其可证结论与禁止用法见 §5「可证性分级」。\n")
    rep.append("## 2. 实验档位\n")
    rep.append("| 档 | 参与调整的发布时刻 | 说明 |")
    rep.append("|---|---|---|")
    for disp, engine, note in ARM_SPECS:
        _ts = tuple(C.ARM_TAUS.get(engine, C.ARM_TAUS["S_all_plus"]))
        s = "{" + ", ".join(f"{int(t)}:00" for t in _ts) + "}" if _ts else "无"
        if engine == "S_all_plus":
            s = "{6:00, 12:00, 18:00}，其中 18:00 用完美信息（保守采纳）"
        elif engine == "S_all+_raw":
            s = "{6:00, 12:00, 18:00}，其中 18:00 **强制**用完美信息"
        elif engine == "PF":
            s = "完美信息"
        rep.append(f"| `${disp}$` | {s} | {note} |")
    rep.append("")
    rep.append("> $S_{all+}$ 的口径说明：README §5.9 原设计为\"变体 B（18:00 视界延伸到次日）\"。"
               "本实验改用**更强**的口径 ——「18:00 拿到完美信息」参照臂。"
               "★ 但**不得**把它当作「相对 $S_{061218}$ 的信息价值上界」："
               "$S_{all+}$ 的候选来自**另一个信息集**，与 $S_{061218}$ 的候选不可比"
               "（详见 §5 的可证性说明）。它只能与 $S_{0612}$ 这一**同基线**档位"
               "作同尺子比较。★ **但实测发现该臂在本引擎下会退化**：完美候选在"
               "共同尺子上被全部压过，于是它**没有**真正用上完美信息 —— 真正回答"
               "「完美信息值多少」的是**反事实臂** $S_{all+}^{\\rm raw}$（下方第 5 档），"
               "它**无条件**采纳完美候选，代价是**不在**嵌套链内。\n")
    rep.append("> ★ **两个 18:00 臂的分工（不得混用）**：`S_all+_UB` 是"
               "**保守采纳臂**——完美信息节点**同时提交该节点的常规候选**，再在共同尺子上"
               "取更省者，故候选集$(S_{all+})\\supseteq$ 候选集$(S_{0612})$，"
               "$S_{all+}^{UB}\\le S_{0612}$ **可证**。`S_all+_raw` 是"
               "**反事实臂**——18:00 **无条件**采纳完美候选、跳过接受检验，"
               "其费用差才是「完美信息值多少」的一次**真实观测**，"
               "但因此**不在**嵌套链内，**禁止**用作上界或单调性证据。\n")
    rep.append("## 3. 对照表（A：实现口径 —— 实际执行的年度账单）\n")
    rep.append("| 档 | 调整购电量 kWh | 下调量 kWh | 上调量 kWh | 计划购电费 元 | "
               "下调违约金 元 | 上调加价 元 | 紧急购电费 元 | 紧急量 kWh | **总费用 元** |")
    rep.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nm in order:
        r = results[nm]
        rep.append(f"| ${nm}$ | {r['调整购电量_kWh']:,.2f} | {r['下调量_kWh']:,.2f} | "
                   f"{r['上调量_kWh']:,.2f} | {r['计划购电费_元']:,.2f} | "
                   f"{r['下调违约金_元']:,.2f} | {r['上调加价_元']:,.2f} | "
                   f"{r['紧急购电费_元']:,.2f} | {r['紧急购电量_kWh']:,.2f} | "
                   f"**{r['总费用_元']:,.2f}** |")
    rep.append("")
    rep.append("## 4. 对照表（B：模型内期望口径 —— 共同 ex-ante 测度下的期望账单）\n")
    rep.append("与 A 表共用同一套 $p,q,u,v$（时刻表决策是确定性的，不随情景变），"
               "只是把单条实际轨迹上的紧急量 $b_t$ 换成**所有档位共用的同一把尺子**"
               "—— $\\tau=0$（0:00）情景集 —— 下的期望值 $\\mathbb E[b_t]$。"
               "换尺子的必要性：附件三 $\\tau=0/6/12/18$ 各自的 30 条情景**并非同一组"
               "路径的嵌套细化**，若各档用自己的节点测度自评，差额里会混入**测度差**。\n")
    rep.append("| 档 | 期望计划+调整费 元 | 期望紧急费 元 | **期望账单 元** | "
               "较 $S_0$ 节省 元 | 相对 $S_0$ | 尺子 |")
    rep.append("|---|---:|---:|---:|---:|---:|---|")
    for nm in order:
        r = results[nm]
        _ruler = "共同" if r["期望口径尺子"].startswith("共同") else "**不适用**"
        rep.append(f"| ${nm}$ | {r['期望口径计划调整费_元']:,.2f} | "
                   f"{r['期望口径紧急费_元']:,.2f} | "
                   f"**{r['期望口径总费用_元']:,.2f}** | "
                   f"{base_e - r['期望口径总费用_元']:+,.2f} | "
                   f"{100 * (base_e - r['期望口径总费用_元']) / base_e:+.4f} % | "
                   f"{_ruler} |")
    rep.append("")
    rep.append("> ⚠ **跨尺子禁令（P0-5 的内在要求）**：`PF` 档的接受检验与期望口径"
               "用的是 $M=1$ 的**真实轨迹**，不是上表的共同 ex-ante 测度。"
               "把 $PF$ 那一行的「期望账单」与其它档比较，正是 P0-5 自己禁止的"
               "「换一把尺子」，会得出「信息更全反而更贵」的假结论。"
               "`PF` 只参与 **A 表（实现口径）** 比较。**报告与论文中一律不得"
               "引用 `PF` 的期望口径数值。**\n")
    rep.append("")
    rep.append("## 5. 单调性与「接受检验」（P0-5）\n")
    rep.append("**机制**：在每个更新节点，对 LP 给出的候选解 $q^{\\rm new}$ 与"
               "「维持旧计划 $q^{\\rm old}$」在**共同尺子**下比价：若 $q^{\\rm new}$ 更贵，"
               "则**拒绝**它、维持 $q^{\\rm old}$。因节点 $\\tau$ 的可行域必包含"
               "「维持旧计划」（$u=v=0$），故最终期望费用 = **候选集上的下确界**。\n")
    rep.append("### 5.1 可证性分级（★ 本节为 Issue-I 整改后的正确表述）\n")
    rep.append("| # | 对象 | 候选集关系 | 结论 | 实测 |")
    rep.append("|---|---|---|---|---|")
    rep.append("| 1 | **嵌套链** "
               "$S_0\\subseteq S_{06}\\subseteq S_{0612}\\subseteq S_{061218}$ | "
               "共享节点信息集相同 ⇒ 后档候选集 ⊇ 前档 | "
               "$K(S_0)\\ge K(S_{06})\\ge K(S_{0612})\\ge K(S_{061218})$ "
               "**可证（构造性质）** | "
               f"{'✔ 满足' if mono_exp_nested else '✘ **不满足**'} |")
    rep.append("| 2 | 信息优势臂 $S_{all+}^{UB}$ | "
               "完美信息节点**同时提交常规候选** ⇒ "
               "候选集$(S_{all+})\\supseteq$ 候选集$(S_{0612})$ | "
               "$S_{all+}^{UB}\\le S_{0612}$ **可证**；"
               "$S_{all+}^{UB}$ 与 $S_{061218}$ **不可比** | "
               f"{'✔' if ub_vs_612 else '✘'}$\\le S_{{0612}}$ |")
    rep.append("| 3 | **反事实臂** $S_{all+}^{\\rm raw}$（18:00 **强制**采纳完美候选）| "
               "跳过接受检验 ⇒ 不是一个「允许拒绝」的档位 | "
               "**不适用**：既非嵌套链，也非上界 | — |")
    rep.append("| 4 | 下界臂 $PF$ | 用 $M=1$ 真实轨迹（**另一把尺子**） | "
               "**不可**与其它档比较「期望口径」 | — |")
    rep.append("")
    rep.append("> ★ **禁止写法**：把 $S_{061218}\\to S_{all+}^{UB}$ 的期望差称作"
               "「18:00 预报的信息价值上界」。该差是「把一个候选换成另一个候选」"
               "的差，符号可正可负，**不是增量上界**。"
               "正确做法：参照臂与实得增量**都从同一基线 $S_{0612}$ 出发**："
               f"$S_{{0612}}\\to S_{{061218}}$（实得）="
               f"{exp_list[_i_s612] - exp_list[_i_18]:+,.2f} 元；"
               f"$S_{{0612}}\\to S_{{all+}}^{{UB}}$（完美信息参照）="
               f"{exp_list[_i_s612] - exp_list[_i_ub]:+,.2f} 元。\n")
    rep.append(f"> 另注：把 $S_{{all+}}^{{UB}}$ 并入序列后的**五档**同尺子序列"
               f"（嵌套链 + $S_{{all+}}^{{UB}}$）本次运行"
               f"{'✔ 单调' if mono_exp else '⚠ 非单调'}"
               "——那只是**单次抽样观测**，既非构造性质也不构成结论。"
               "（$PF$ 与 $S_{all+}^{\\rm raw}$ 不参与该序列：前者换了尺子，"
               "后者不在嵌套链内。）\n")
    rep.append("- 实现口径（单条真实轨迹）："
               f"{'✔ 本次运行恰好单调' if mono_impl else '⚠ 本次运行非单调'}"
               "——这只是**一条真实轨迹的一次抽样**，单轨迹数值不具判别力："
               "紧急购电是按 5 倍电价结算的稀疏尖峰量，少数时段的实现波动就能"
               "翻转相邻档次序。判定信息价值必须用上表的**共同测度期望口径**。\n")
    rep.append("### 5.2 相邻档明细\n")
    rep.append("| 相邻档（前 → 后）| 期望口径降本 元 | 实现口径降本 元 | "
               "后档拒绝新解段数 | 后档完美候选被压段数 |")
    rep.append("|---|---:|---:|---:|---:|")
    for _i in range(len(SEQ6) - 1):
        _a, _b = SEQ6[_i], SEQ6[_i + 1]
        rep.append(f"| ${_a}$ → ${_b}$ | "
                   f"{exp_list[order.index(_a)] - exp_list[order.index(_b)]:+,.2f} | "
                   f"{tot_list[order.index(_a)] - tot_list[order.index(_b)]:+,.2f} | "
                   f"{results[_b]['拒绝新解段数']} | "
                   f"{results[_b]['完美候选被压段数']} |")
    rep.append(f"| $S_{{0612}}$ → $S_{{all+}}^{{\\rm raw}}$（**反事实臂**，不在嵌套链内）|"
               f" {exp_list[_i_s612] - exp_list[_i_raw]:+,.2f} |"
               f" {tot_list[_i_s612] - tot_list[_i_raw]:+,.2f} |"
               f" {results['S_all+_raw']['拒绝新解段数']}（跳过检验）|"
               f" {results['S_all+_raw']['强制采纳段数']}（强制采纳）|")
    rep.append("")
    rep.append("> 注：「拒绝新解段数」= 该档在多少个（日, 节点）段上因"
               "共同尺子下的比价不合格而**维持了旧计划**。它是 P0-5 要求的"
               "「完整档必须被允许拒绝新解」的直接实现，也提示了额外信息的"
               "**实际被采纳率**。若某档拒绝数为 0，则该档全部采纳了由更多"
               "信息得到的新解。\n")
    rep.append("> 注：「完美候选被压段数」只对 $S_{all+}^{UB}$ 有含义：它的"
               "18:00 完美信息候选在共同尺子上被常规候选/旧计划压过的段数。"
               "大于 0 说明**在该尺子上**「拿到完美信息」并未换来更省的计划"
               "——这是尺子（`exante_cost_seg` 是紧急费的**保守上界**，"
               "不计执行器放电抵扣）与 LP 目标函数不同源所致，属**已知口径差**。\n")
    rep.append("> ★ **本条最关键的一句话**：若「完美候选被压段数」= 全部评分日，"
               "则 $S_{all+}^{UB}$ 的最终决策与 $S_{061218}$ **逐位相同**，"
               "该臂「额外增益为 0」是**在该尺子下未被采纳**的结果，"
               "**不是**「完美信息价值为 0 的测量」。后者只能由反事实臂 "
               "$S_{all+}^{\\rm raw}$（「强制采纳段数」列 > 0）给出。\n")
    rep.append("> ⚠ **口径提示**：本表的**实现口径**差值仍是单条实际轨迹的一次"
               "抽样，可以出现负值（即某一档在真实轨迹上反而多花），"
               "**不能**用单轨迹抽样否定单调性——单调性只在共同测度的期望口径下、"
               "且只对**嵌套链**有定义。\n")
    _pf_vs_s0 = 100 * (tot_list[0] - tot_list[order.index("PF")]) / base
    _g06 = 100 * (base_e - exp_list[1]) / base_e
    _g0612 = 100 * (base_e - exp_list[2]) / base_e
    _g0612_add = 100 * (exp_list[1] - exp_list[2]) / base_e
    _g0612_impl = 100 * (base - tot_list[2]) / base
    _g18_add = 100 * (exp_list[_i_s612] - exp_list[_i_18]) / base_e
    _g18_ub = 100 * (exp_list[_i_s612] - exp_list[_i_ub]) / base_e
    _d612 = base_e - exp_list[2]
    _impl612 = base - tot_list[2]

    def _money(v):
        return (f"**节省 {v:,.2f} 元**" if v >= 0
                else f"反而**多花 {-v:,.2f} 元**")

    def _pct(v):
        return f"**{v:+.4f} %**"

    def _delta(v):
        return f"降本 {v:,.2f} 元" if v >= 0 else f"增支 {-v:,.2f} 元"

    _v1 = ("**引入 6:00 与 12:00 的预报能降本。**"
           if _g0612 > 0 else
           "**本次数据下，引入 6:00 与 12:00 的预报未能降本"
           "（与真实数据结论相反，不可引用）。**")
    rep.append("## 6. 结论\n")
    rep.append("> 符号约定：下文所有收益量均取「前档 $-$ 后档」，"
               "**正值表示费用下降**（省钱），负值表示反而变贵。"
               "结论中的定性判断一律由本表数字驱动，不预设方向。\n")
    rep.append(f"1. {_v1} 期望口径下 $S_0\\to S_{{0612}}$ {_money(_d612)}"
               f"（{_pct(_g0612)}）；拆开看，$S_{{06}}$ 这一档{_delta(base_e - exp_list[1])}"
               f"（{_pct(_g06)}），$S_{{0612}}$ 再{_delta(exp_list[1] - exp_list[2])}"
               f"（{_pct(_g0612_add)}）。实现口径下由 {base:,.2f} 元变为 "
               f"{tot_list[2]:,.2f} 元，{_money(_impl612)}。\n")
    _thr = 0.05 * abs(_g0612) if abs(_g0612) > 0 else 0.01
    _g18_y = exp_list[_i_s612] - exp_list[_i_18]
    _g18uby = exp_list[_i_s612] - exp_list[_i_ub]
    _g18_yi = tot_list[_i_s612] - tot_list[_i_18]
    _g18ubyi = tot_list[_i_s612] - tot_list[_i_ub]
    _raw_ex = exp_list[_i_s612] - exp_list[_i_raw]
    _raw_im = tot_list[_i_s612] - tot_list[_i_raw]
    _ub_deg = ub_degenerate
    if abs(_g18_add) <= _thr:
        rep.append(f"2. **18:00 的预报增量可忽略（本次数据的实测结果）。** "
                   f"同一基线 $S_{{0612}}$ 下，$S_{{0612}}\\to S_{{061218}}$"
                   f"（**常规** 18:00 预报）的期望差额为 {_g18_y:+,.2f} 元"
                   f"（{_g18_add:+.4f} %），实现口径 {_g18_yi:+,.2f} 元。\n")
    else:
        rep.append(f"2. **18:00 的预报增量不可忽略（与预期相反，需复核）。** "
                   f"同一基线 $S_{{0612}}$ 下，$S_{{0612}}\\to S_{{061218}}$"
                   f" 的期望差额为 {_g18_y:+,.2f} 元（{_g18_add:+.4f} %）。\n")
    rep.append("   关于「18:00 的**完美信息**值多少」，本报告给出两条独立证据，"
               "并严格区分其效力：\n")
    if _ub_deg:
        rep.append(
            f"   **(i) 保守采纳臂 $S_{{all+}}^{{UB}}$（可证，但在本引擎下退化）**："
            f"它的 18:00 完美候选在共同尺子上被压 "
            f"**{results['S_all+_UB']['完美候选被压段数']} 段**"
            f"（= 全部评分日 {n_day_used} 天），故该臂最终决策与 $S_{{061218}}$ "
            f"**逐位相同**（期望 {exp_list[_i_ub]:,.2f} 元）。\n")
        rep.append(
            f"   ⚠ 因此 $S_{{0612}}\\to S_{{all+}}^{{UB}}$ 的 {_g18uby:+,.2f} 元"
            f"（实现口径 {_g18ubyi:+,.2f} 元）**不是**完美信息的增量，而是"
            f"**常规** 18:00 预报的增量（与上一段同一数字）。"
            f"**严禁**把该臂的「额外增益为 0」当作「完美信息价值为 0」。\n")
    else:
        rep.append(
            f"   **(i) 保守采纳臂 $S_{{all+}}^{{UB}}$（可证）**：期望 "
            f"{exp_list[_i_ub]:,.2f} 元、相对同基线 $S_{{0612}}$ "
            f"{_g18uby:+,.2f} 元（{_g18_ub:+.4f} %）；"
            f"其 18:00 完美候选被压 "
            f"{results['S_all+_UB']['完美候选被压段数']} 段。\n")
    rep.append(
        f"   **(ii) 反事实臂 $S_{{all+}}^{{\\rm raw}}$（真实测量，但不可证）**："
        f"18:00 **无条件**采纳完美候选"
        f"（强制采纳 {results['S_all+_raw']['强制采纳段数']} 段），"
        f"期望 {exp_list[_i_raw]:,.2f} 元、实现 {tot_list[_i_raw]:,.2f} 元；"
        f"相对同基线 $S_{{0612}}$：期望 {_raw_ex:+,.2f} 元、"
        f"实现 {_raw_im:+,.2f} 元；相对 $S_{{061218}}$：期望 "
        f"{exp_list[_i_18] - exp_list[_i_raw]:+,.2f} 元、实现 "
        f"{tot_list[_i_18] - tot_list[_i_raw]:+,.2f} 元。\n")
    _span_raw = max(abs(_g18_y), abs(_raw_ex), abs(_raw_im))
    _span_pct = 100.0 * _span_raw / base_e
    rep.append(
        f"   ⟹ 三种口径（常规档期望 / 反事实臂期望 / 反事实臂实现）下，"
        f"18:00 的增量为 "
        f"{_g18_y:+,.2f} / {_raw_ex:+,.2f} / {_raw_im:+,.2f} 元，"
        f"绝对值上界 {_span_raw:,.0f} 元（{_span_pct:.4f} % of 基线期望账单）。"
        + ("**符号随口径翻转且量级极小** ⇒ **结论稳健：不需要为 18:00 预报额外投入**。\n"
           if _span_pct <= 0.5 else
           f"⚠ 该量级已超过 0.5 % 阈值（{_span_pct:.4f} %）："
           "**本次（可能为冒烟/短样本）运行不支持「18:00 可忽略」**，"
           "请以完整评分期运行结果为准。\n"))
    rep.append("   机制解释（**仅在上述实测与预期一致时适用**）：(i) 18:00 的正向视界 "
               "$[18{:}00,24{:}00)$ 几乎全是夜间，该窗口内物理可利用的光伏极少，"
               "四个发布时刻在该窗口的预报误差几乎相同（即新增预报对该窗口"
               "**近乎零增量**）；(ii) 它对其余时段的信息被**次日 0:00 的更优预报支配**；"
               "(iii) 唯一的残余通路是「今夜低价为次日凌晨备货」，"
               "套利空间被储能容量与功率上限夹得极小。\n")
    _room = tot_list[_i_18] - tot_list[_i_pf]
    if _room > 0:
        rep.append(f"3. **距完美信息的剩余空间（实现口径 $S_{{061218}}\\to PF$）**为 "
                   f"{_room:,.2f} 元（约 {100 * _room / base:.4f} %），"
                   "这部分只能靠预报精度的**进一步提升**（而非增加发布时刻）来获取。"
                   "（$PF$ 的期望口径用的是另一把尺子，按 P0-5 不得参与跨档比较，"
                   "故此处只用实现口径。）\n")
    else:
        rep.append(f"3. **距完美信息的剩余空间（实现口径）**：单条轨迹抽样下 $PF$ 反而比 "
                   f"$S_{{061218}}$ 贵 {-_room:,.2f} 元。$PF$ 的目标函数不是账单函数"
                   "（它仍受执行器与尺子约束），故这**不构成模型缺陷**，"
                   "但提示该抽样下 PF 的优势未在实现口径上体现，需在正式数据下复核。\n")
    _small = abs(_g0612_impl) < 0.5 * abs(_pf_vs_s0)
    _rel = "**远小于**" if _small else "**并不显著小于**"
    rep.append("4. **量级判断与机制。** 引入 6:00/12:00 的收益"
               f"（实现口径 {_g0612_impl:+.2f} %；期望口径 {_g0612:+.2f} %）"
               f"{_rel}「执行层信息价值」（$PF$ 相对 $S_0$ 的 "
               f"{_pf_vs_s0:+.2f} %，**实现口径**——$PF$ 的期望口径换了尺子"
               "故不引用）。")
    if _small:
        rep.append("   原因是：5 倍紧急电价的存在使「保持一定风险暴露」比「花 1.5 倍加价"
                   "全面买保险」更经济，因此调整购电量的最优值本就接近计划值；"
                   "真正吃掉成本的不是「买得不准」而是「储能执行得不精细」。\n")
    else:
        rep.append("   本次数据下 **6:00/12:00 的信息收益与执行层信息价值同量级**，"
                   "上述机制解释（调整量本就接近计划值）**不成立**，"
                   "本条结论需结合 §5 的接受检验记录复核。\n")
    rep.append("5. **已知局限（诚实披露）。** 二阶段 SP 的第二阶段是「情景内完全追索」，"
               "即假设电池可针对该情景重排，因此会**低估**「时段级缺口的实际代价」，"
               "使期望口径下的调整收益偏乐观。因此上表的**期望口径收益应视为上界**；"
               + ("实际工程结论（引入 6:00/12:00、不引入 18:00）在两个口径下一致。\n"
                  if _span_pct <= 0.5 else
                  f"⚠ 但 18:00 的增量达 {_span_pct:.4f} %（> 0.5 % 阈值），"
                  "「不引入 18:00」这一结论在本次运行中**未获支持**，"
                  "须以完整评分期结果为准。\n"))
    rep.append("### 一句话回答题目\n")
    if _d612 >= 0:
        _head = ("**需要引入 6:00 与 12:00 的预报**（这是免费的期权，"
                 f"期望口径下每年约省 {_d612 / 1e4:,.2f} 万元）；")
        _tail = ("要再降本只能提高**预报本身的精度**或**储能执行的精细度**，"
                 "而不是增加发布时刻。\n")
        if _span_pct <= 0.5:
            rep.append(_head + "**不需要为 18:00 的预报额外投入**——在同一基线 "
                       "$S_{0612}$ 下，"
                       f"常规 18:00 预报的增量仅 {abs(_g18_y) / 1e4:,.3f} 万元量级，"
                       f"而**强制**采信完美信息的反事实臂 $S_{{all+}}^{{\\rm raw}}$"
                       f" 也只给出 {abs(_raw_im) / 1e4:,.3f} 万元（实现口径）"
                       "的同量级增益（见 §6 第 2 条的两条证据）；" + _tail)
        else:
            rep.append(_head + f"**但 18:00 的增量在本次运行中达 {_span_pct:.4f} %"
                       "（> 0.5 % 阈值）**，故**不能**断言「18:00 可忽略」——"
                       "请以完整评分期（334 天）运行结果为准，并复核 §6 第 2 条"
                       "三种口径的数值；" + _tail)
    else:
        rep.append(f"**本次数据下结论需按数字修正**：引入 6:00/12:00 的预报在期望口径下"
                   f"每年反而多花 {-_d612 / 1e4:,.2f} 万元，与真实数据结论相反，"
                   "**不可直接引用**；请确认数据源（非 `real` 模式的合成数据仅用于骨架自检）。"
                   "就 18:00 而言，无论何种口径，其增量绝对值均与 0 无异，"
                   "因此「不为 18:00 额外投入」的结论仍然稳健。\n")
    q2n = C.Q2.headline_numbers()
    q2_side = q2n.get("source") is not None

    _nostor_fee = _nostor_kwh = None
    try:
        import csv as _csv
        with C.YEARLY_CSV.open("r", encoding="utf-8-sig", newline="") as _fh:
            _d = {r[0].strip(): r[1].strip()
                  for r in _csv.reader(_fh) if len(r) >= 2}
        _nostor_fee = float(_d.get("无储能基准购电费_元", "nan"))
        _nostor_kwh = float(_d.get("无储能基准购电量_kWh", "nan"))
    except Exception:
        pass

    _dp_gain = None
    try:
        _r = raws["S061218"]
        _b = np.asarray(_r["b"], float)[sc]
        _ba = np.asarray(_r["b_analytic"], float)[sc]
        _dp_gain = float((5.0 * price[None, :] * (_ba - _b)).sum())
    except Exception:
        pass

    def _q2fmt(v, spec="{:,.4f}"):
        return "—" if v is None else spec.format(v)

    rep.append("## 7. 与第二问的对照\n")
    rep.append("本问与第二问**数据同源、骨架同源**：\n")
    rep.append("- 共用 `第二问最终版/处理后数据/附件二_矩阵数据.npz`（全量：负荷 / 光伏 / 电价 / "
               "储能参数）与 `附件二_情景库.npz` 的 `scen_L` / `scen_idx` / `resid_L`"
               "（光伏情景侧全部废弃，改用附件三重建）；")
    rep.append("- 评分期同为 **334 天**（2025-02-01 … 12-31），储能参数与费率规则完全相同；")
    if _nostor_fee is not None and _nostor_kwh is not None:
        _cmp = ""
        if q2n.get("baseline_fee"):
            _same = abs(q2n["baseline_fee"] - _nostor_fee) < 0.01
            _cmp = (f"，第二问报告为 {q2n['baseline_fee']:,.4f} 元"
                    f"（{'逐位相同' if _same else '存在差异'}）")
        rep.append(f"- **无储能基准由本问数据独立重算**：{_nostor_kwh:,.4f} kWh / "
                   f"{_nostor_fee:,.4f} 元{_cmp}"
                   "——这是两问同源最直接的证据；")
    rep.append("- 求解与执行骨架相同：两阶段情景 LP + 阶段内 DP 未来价值执行器。\n")
    if q2_side:
        _rel_parts = []
        for _p in str(q2n["source"]).split(" | "):
            _p = _p.strip()
            if not _p:
                continue
            for _base in (C.Q2_DIR, C.PROJECT_DIR.parent):
                try:
                    _rel_parts.append(Path(_p).relative_to(_base).as_posix())
                    break
                except Exception:
                    continue
            else:
                _rel_parts.append(_p)
        _rel = " | ".join(_rel_parts) if _rel_parts else str(q2n["source"])
        _cmp_lbl = q2n.get("comparison") or "—"
        rep.append(f"下表「第二问」列读取自 `{_rel}`"
                   "（**只读引用，不参与本问任何计算**）；"
                   f"引用口径 **{_cmp_lbl} / DP 价值执行器**"
                   "——即第二问定稿自述的主结果口径"
                   "（其 `第二问_最终核心结果.csv` / `第二问_交付复核摘要.json` / "
                   "`第二题_结果总览.md` 三处逐位一致）。"
                   "⚠ 第二问另有「固定计划」口径（相差约 567 元），本表**不采用**：\n")
    else:
        rep.append(f"⚠ 本次运行**未引用第二问结果**（数据源模式 `{C.Q2.mode()}`）："
                   "下表「第二问」列留空，仅列本问侧结果。\n")
    rep.append("| 指标 | 第二问 | 本问 $S_0$（无调整）| 本问 $S_{061218}$（完整档）|")
    rep.append("|---|---:|---:|---:|")
    rep.append(f"| 全年账单 元 | {_q2fmt(q2n['yearly_total'])} | {base:,.4f} | "
               f"{tot_list[3]:,.4f} |")
    rep.append(f"| 调整费 元 | —（无此机制）| 0.0000 | "
               f"{results['S061218']['调整相关费用_元']:,.4f} |")
    rep.append(f"| 紧急购电费 元 | {_q2fmt(q2n['emerg_fee'])} | "
               f"{results['S0']['紧急购电费_元']:,.4f} | "
               f"{results['S061218']['紧急购电费_元']:,.4f} |")
    rep.append(f"| 紧急购电量 kWh | {_q2fmt(q2n['emerg_kwh'])} | "
               f"{results['S0']['紧急购电量_kWh']:,.4f} | "
               f"{results['S061218']['紧急购电量_kWh']:,.4f} |")
    rep.append(f"| 无储能基准购电费 元 | {_q2fmt(q2n['baseline_fee'])} | "
               f"{_q2fmt(_nostor_fee)} | {_q2fmt(_nostor_fee)} |")
    rep.append(f"| DP 执行器增益 元 | {_q2fmt(q2n['dp_gain'])} | — | "
               f"{_q2fmt(_dp_gain)} |")
    rep.append("")
    rep.append("**差异的来源（不是模型退步）：**\n")
    rep.append("1. **光伏信息源不同** —— 第二问用附件二自建预测，本问被题面强制使用附件三"
               "第三方整点预报（四档精度见 `第三问_预报精度表.csv`），"
               "为覆盖「光伏低于预期」的风险，日前计划量被迫抬高；")
    rep.append("2. **决策结构不同** —— 第二问只有「计划 + 紧急」两层，本问增加「日内 4 次调整」，"
               "且调整价格**不对称**（下调退 0.5$c$、上调罚 1.5$c$），"
               "理性的报量 / 追索策略因此改变；")
    _gain_txt = ("（本次运行未读取第二问数值）" if q2n.get("dp_gain") is None
                 else f"，第二问为 {q2n['dp_gain']:,.2f} 元")
    rep.append("3. **执行器相同，但增益被吸收** —— 两问都用 DP 未来价值保留水平："
               "DP 相对纯解析执行的增益，本问为 "
               f"{_q2fmt(_dp_gain)} 元{_gain_txt}。"
               "本问增益更小，是因为 `06` 的调整机制已提前买了保险，"
               "DP 的边际价值被部分吸收。\n")
    if q2n.get("emerg_fee"):
        rep.append("**代价换来的收益**：紧急购电费相对第二问降低 "
                   f"{100 * (1 - results['S061218']['紧急购电费_元'] / q2n['emerg_fee']):.2f} %，"
                   "即本问用「1.5 倍上调的确定性支出」置换了「5 倍紧急购电的风险」。\n")
    else:
        rep.append("**代价换来的收益**：本问用「1.5 倍上调的确定性支出」置换了"
                   "「5 倍紧急购电的风险」（紧急购电费 / 量对照见上表）。\n")
    rep.append("> 注意：本问**不复用第二问的任何数值结果**，全部从共享的原始 / 处理后数据重算；"
               "两问只共享数据文件与 `scen_L` 情景库。\n")
    C.write_text_utf8(C.REPORT_ABL_MD, "\n".join(rep))
    log("")
    log(f"已保存：{C.ABLATION_CSV.relative_to(C.PROJECT_DIR)}、"
        f"第三问_消融实验增量表.csv、第三问_消融实验.npz")
    log(f"已保存：{C.REPORT_ABL_MD.relative_to(C.PROJECT_DIR)}")

    log.dump(C.LOG_DIR / "第三问_09消融实验日志.txt", tail="")
    log("")
    log(f"总用时 {time.perf_counter() - t00:.1f} s")
    log("[09 完成] 消融实验结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
