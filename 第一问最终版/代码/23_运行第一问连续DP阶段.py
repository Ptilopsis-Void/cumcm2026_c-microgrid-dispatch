from pathlib import Path
import sys
import importlib.util
from datetime import datetime

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parent
RESULT_DIR = PROJECT / "模型结果"
FIG_DIR = PROJECT / "模型结果图"
REPORT_DIR = PROJECT / "报告"
LOG_DIR = PROJECT / "求解日志"
sys.path.insert(0, str(CODE_DIR))

J_LOCK = 35126.9485892896
Q_LOCK = 59482.6989983539


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, str(CODE_DIR / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def setup_font():
    try:
        import matplotlib
        from matplotlib import font_manager
        avail = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((c for c in ["Songti SC", "PingFang HK", "Hiragino Sans GB",
                                   "Heiti TC", "STHeiti", "Arial Unicode MS"] if c in avail), None)
        if chosen:
            import matplotlib
            matplotlib.rcParams["font.family"] = [chosen]
        import matplotlib
        matplotlib.rcParams["axes.unicode_minus"] = False
        return chosen
    except Exception:
        return None


def make_diagnostic_fig(ctx, fig_pdf, fig_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()

    cpl_cache = ctx["cpl_cache"]
    REP_TIMES = ctx["REP_TIMES"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=False)
    for ax, (t, label) in zip(axes, REP_TIMES):
        knots = cpl_cache[t]
        e = [p[0] for p in knots]
        F = [p[1] for p in knots]
        ax.plot(e, F, marker="o", markersize=3, linewidth=1.4)
        lo, hi = ctx["reach1"](t)
        ax.set_xlim(lo, hi)
        ax.set_title(f"t={t}  ({label})", fontsize=10)
        ax.set_xlabel("储能电量 e (kWh)")
        ax.set_ylabel("F_t(e) (元)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.3)
    fig.suptitle("第一问 连续状态动态规划价值函数 F_t(e)（诊断图，非正式论文图）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for p in (fig_pdf, fig_png):
        fig.savefig(p, dpi=160)
    plt.close(fig)
    return {"pdf": str(fig_pdf), "png": str(fig_png)}


def write_report(m21, ctx, r, q21, q22, new_files):
    A = []
    ap = A.append
    ap("# 第一问 连续状态动态规划验证报告（第 3.3 节）\n")
    ap("> 本报告基于后续时域 LP 构造最终连续 LP **同一模型**的")
    ap("> 价值函数重构与 Bellman 一致性检验（而非并行的“第二套模型”）。不修改 LP/MILP/图2—图4/result1.xlsx，")
    ap("> F_1 与轨迹首步仍由 LP 求得，不是独立 DP 求解器。\n")

    ap("## 1. 连续 DP 数值实现方法\n")
    ap("采用“**后续时域 LP 价值查询 + 自适应分段线性重构**”")
    ap("（不采用粗糙固定 SOC 网格）：")
    ap("1. 给定 (t,e)，构造 t..T 的后续时域 LP，固定进入状态 E_{t-1}=e，精确施加终端 E_T=6000 kWh；")
    ap("2. 返回最小费用 F_t(e) 作为价值函数查询；")
    ap("3. 在有效状态域 I_t 内用中点细分自适应增加采样点，当相邻区间斜率变化或线性插值")
    ap("   误差超过容差（1e-6 元）时继续细分，直到最小子区间宽度 1e-4 kWh；")
    ap("4. 保存折点、线段斜率与值函数数据。该数值工具仅用于计算与验证 Bellman 递推（式10—14），")
    ap("   论文理论主体仍是式（10）—（14）的 Bellman 递推。\n")

    ap("## 2. 是否使用固定网格\n")
    ap("否。采用自适应分段线性重构，最小子区间宽度 1e-4 kWh，线性残差容差 1e-6 元。")
    ap("等价于在“斜率变化位置”附近自动加密、在纯线性区自动保持稀疏。\n")

    ap("## 3. 价值函数查询次数\n")
    ap(f"- 21 构造价值函数：{q21} 次 `tail_lp_value` 查询")
    ap(f"- 22 校验（LP费用一致性 + Bellman 中尾 LP 查询）：累计 {q22} 次")
    ap(f"- 轨迹恢复另用 `tail_lp_solution`（逐时首步决策）{m21.T} 次。\n")

    ap("## 4. 整日 LP 与后续时域 LP 费用一致性\n")
    ap(f"F_1(6000) = {r['F1']:.10f} 元；锁定 LP 最优值 J_1 = {J_LOCK} 元；")
    ap(f"绝对误差 = {r['cost_err']:.3e} 元 ≤ 1e-5，满足本文设置的数值容差。\n")
    ap(f"后续 LP 首步恢复轨迹总购电量 = {r['g_sum']:.10f} kWh；锁定 LP 总购电量 = {Q_LOCK} kWh；")
    ap(f"误差 = {r['q_err']:.3e} kWh；轨迹费用 Σc_t g_t = {r['traj_cost']:.8f} 元。\n")

    ap("## 5. Bellman 递推验证\n")
    ap("对代表时刻 0:00、6:00、12:00、18:00、临近24:00，从有效域选取左端点、中间状态、右端点")
    ap("与自适应折点，验证 Bellman 等式最大残差：")
    ap(f"- 最大残差 = {r['max_bell']:.3e} 元（发生于 {r['max_bell_info']}），容差 1e-3。\n")

    ap("## 6. 凸性与边际价值单调性\n")
    ap("对相邻采样状态计算割线斜率 s_i，验证 s_{i+1}≥s_i−ε：")
    ap(f"- 最大凸性违例 = {r['max_conv']:.3e} 元/kWh（发生于 {r['max_conv_info']}），容差 1e-3；")
    ap(f"- 边际价值 λ_t(e)=−∂F_t(e)，λ 非增 ⟺ 割线斜率不减，结论同源成立。\n")

    ap("## 7. 终端可达性与可达状态域\n")
    ap(f"- 终端条件精确执行：E_144 = {r['E_end']:.10f} kWh（误差 {r['E_term_err']:.3e} kWh），严格等于 6000；")
    ap("- 可达状态域 I_t 始终在 [1200,10800] 内，且随临近终点单调收缩；")
    ap(f"- I_{{T+1}} = [{r['I_Tp1'][0]:.6f}, {r['I_Tp1'][1]:.6f}] = {{6000}}（精确单点）。\n")

    ap("## 8. 代表时刻价值函数采样统计\n")
    ap("统计的是采样折线中斜率跳变超过1e-4元/kWh的位置数，不等同于真实折点数。最后时段单独采用解析折点6000及6637.9122037 kWh，内部真实折点为2个。")
    ap("")
    ap("| 时刻 | 时段 t | 采样点数 | 采样斜率变化点数 |")
    ap("|---|---|---|---|")
    for t, label in ctx["REP_TIMES"]:
        ap(f"| {label} | {t} | {r['knot_count'][t]} | {r['bp_count'][t]} |")
    ap("")

    ap("## 9. 命题 3.1 简化口径\n")
    ap("后续时域LP右端关于初始状态e仿射，最优值是仿射函数族的上确界，因此在有效域上凸且分段线性。凸函数的次梯度单调不减，故负次梯度表示的边际价值单调不增。数值采样用于一致性检查：")
    ap("1. F_t(e) 在 I_t 上是凸分段线性函数（割线斜率单调不减，最大违例 %.3e）；" % r["max_conv"])
    ap("2. 边际价值 λ_t(e)∈−∂F_t(e) 随 e 增加单调不增；")
    ap("3. 即储能量越充足，再增加一单位储能所节约的未来购电费用越低。\n")

    ap("## 10. 验收结果\n")
    ap(f"{r['n_pass']}/{r['n_total']} 项全部通过（详见 `模型结果/连续DP_性质验收.csv`）。\n")

    ap("## 11. 新增文件清单\n")
    for f in new_files:
        ap(f"- {f}")
    ap("")

    ap("## 12. 异常与待确认\n")
    if r["all_pass"]:
        ap("本次预期可达测试点均有限。LP 可能存在多组等价最优轨迹，后续 LP 首步恢复轨迹与之费用、总购电量及全部约束一致；")
        ap("论文图2—图4仍使用已锁定的 LP 结果，不被本阶段等价轨迹覆盖。")
    else:
        ap("存在未通过项，需按第 3.3 节清单排查索引/终端/效率方向/S 单位/净负荷/正部函数/可达域/时间映射。")
    ap("")

    out = REPORT_DIR / "第一问连续状态动态规划验证报告.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(A), encoding="utf-8")
    return out


def write_log(m21, ctx, r, q21, q22, fig, report_file, input_shas):
    L = []
    A = L.append
    A("=" * 72)
    A("第一问 连续状态动态规划日志（第 3.3 节）")
    A(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    A("=" * 72)
    A("")
    A("[输入文件 SHA-256]")
    for name, h in input_shas.items():
        A(f"  {name}: {h}")
    A("")
    A("[数值实现]")
    A("  后续时域LP价值查询 + 自适应分段线性重构（非固定网格）")
    A("  最小子区间宽度 1e-4 kWh，线性残差容差 1e-6 元")
    A(f"  tail_lp_value 查询：21阶段={q21} 次，累计(含22)={q22} 次")
    A(f"  轨迹恢复 tail_lp_solution：{m21.T} 次")
    A("")
    A("[LP-DP 互证]")
    A(f"  F_1(6000) = {r['F1']:.10f} 元")
    A(f"  与 LP 目标误差 = {r['cost_err']:.3e} 元 (容差 1e-5)")
    A(f"  DP轨迹总购电量 = {r['g_sum']:.10f} kWh (LP={59482.6989983539})")
    A(f"  总购电量误差 = {r['q_err']:.3e} kWh")
    A(f"  DP轨迹费用 = {r['traj_cost']:.8f} 元 (误差 {r['traj_cost_err']:.3e})")
    A("")
    A("[Bellman / 凸性 / 单调性]")
    A(f"  最大Bellman残差 = {r['max_bell']:.3e} 元 @ {r['max_bell_info']} (容差 1e-3)")
    A(f"  最大凸性违例 = {r['max_conv']:.3e} 元/kWh @ {r['max_conv_info']} (容差 1e-3)")
    A(f"  边际价值单调性 = 与凸性同源 (λ=-斜率)")
    A("")
    A("[终端与可达域]")
    A(f"  E_144 = {r['E_end']:.10f} kWh (误差 {r['E_term_err']:.3e})")
    A(f"  I_{{T+1}} = [{r['I_Tp1'][0]:.6f},{r['I_Tp1'][1]:.6f}] = {{6000}} 精确={r['I_Tp1_exact']}")
    A(f"  可达域不超设备范围={r['domain_in_range']}, 宽度最大回弹={r['max_width_grow']:.3e}")
    A("")
    A("[轨迹约束校验]")
    A(f"  max供需平衡残差 = {r['balance_resid']:.3e} kWh")
    A(f"  max储能递推残差 = {r['storage_resid']:.3e} kWh")
    A(f"  min g = {r['min_g']:.3e} (≥0, 无售电)")
    A(f"  min U = {r['min_U']:.3e} (≥0, 弃光非负)")
    A(f"  max C = {r['max_C']:.6f}, max D = {r['max_D']:.6f}, S = {m21.S:.6f}")
    A("")
    A("[代表时刻采样斜率变化点数]")
    for t, label in ctx["REP_TIMES"]:
        A(f"  t={t:3d} ({label}) : 采样点 {r['knot_count'][t]} 个，折点 {r['bp_count'][t]} 个")
    A("")
    A("[诊断图文件]")
    A(f"  {fig['pdf']}")
    A(f"  {fig['png']}")
    A("")
    A("[原始 LP 文件是否保持不变]")
    A(f"  最终LP_原始最优解.csv SHA-256 = {r['sol_sha']}")
    A(f"  {'与第三阶段冻结值一致（未改动）' if r['sha_unchanged'] else '不一致（已改动！）'}")
    A("")
    A("[总体验收]")
    A(f"  {r['n_pass']}/{r['n_total']} 项通过")
    A(f"  状态：{'通过' if r['all_pass'] else '未通过'}")
    A("")

    out = LOG_DIR / "第一问连续状态动态规划日志.txt"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main():
    m21 = load_mod("21_构造连续状态价值函数")
    m22 = load_mod("22_校验连续DP与价值函数")

    m21.reset_query_count()
    ctx = m21.main()
    q21 = m21.QUERY_COUNT
    ctx["reach1"] = m21.reachable_domain

    r = m22.main(ctx, m21)
    q22 = m21.QUERY_COUNT
    if not r["all_pass"]:
        raise RuntimeError("Bellman/价值函数验收失败，停止生成正式图表")

    fig_pdf = FIG_DIR / "第一问_连续DP价值函数诊断.pdf"
    fig_png = FIG_DIR / "第一问_连续DP价值函数诊断.png"
    fig = make_diagnostic_fig(ctx, fig_pdf, fig_png)

    new_files = [
        "代码/21_构造连续状态价值函数.py",
        "代码/22_校验连续DP与价值函数.py",
        "代码/23_运行第一问连续DP阶段.py",
        "模型结果/连续DP_价值函数折点.csv",
        "模型结果/连续DP_边际价值.csv",
        "模型结果/连续DP_LP互证.csv",
        "模型结果/连续DP_贝尔曼残差.csv",
        "模型结果/连续DP_性质验收.csv",
        "报告/第一问连续状态动态规划验证报告.md",
        "求解日志/第一问连续状态动态规划日志.txt",
        "模型结果图/第一问_连续DP价值函数诊断.pdf",
        "模型结果图/第一问_连续DP价值函数诊断.png",
    ]

    input_shas = {
        "最终LP_原始最优解.csv": m22.sha256(RESULT_DIR / "最终LP_原始最优解.csv"),
        "最终LP_原始对偶数据.csv": m22.sha256(RESULT_DIR / "最终LP_原始对偶数据.csv"),
        "最终LP_变量边界对偶数据.csv": m22.sha256(RESULT_DIR / "最终LP_变量边界对偶数据.csv"),
    }

    report_file = write_report(m21, ctx, r, q21, q22, new_files)
    log_file = write_log(m21, ctx, r, q21, q22, fig, report_file, input_shas)

    print("=" * 70)
    print("[23] 第五阶段完成（连续状态动态规划）")
    print("=" * 70)
    print("1. 数值实现方法：后续时域LP价值查询 + 自适应分段线性重构（非固定网格）")
    print("2. 固定网格：否；最小子区间宽度 1e-4 kWh，线性残差容差 1e-6 元")
    print(f"3. 价值函数查询次数：21阶段 {q21} 次，累计含校验 {q22} 次（轨迹恢复另 tail_lp_solution {m21.T} 次）")
    print(f"4. F_1(6000) = {r['F1']:.10f} 元")
    print(f"5. 与最终LP目标绝对误差 = {r['cost_err']:.3e} 元")
    print(f"6. DP恢复轨迹总购电量 = {r['g_sum']:.10f} kWh")
    print(f"7. 与最终LP总购电量误差 = {r['q_err']:.3e} kWh")
    print(f"8. 最大Bellman递推残差 = {r['max_bell']:.3e} 元")
    print(f"9. 最大凸性违例 = {r['max_conv']:.3e} 元/kWh")
    print(f"10. 最大边际价值单调性违例 = {r['max_conv']:.3e} 元/kWh（与凸性同源）")
    print(f"11. 终端条件精确执行：E_144={r['E_end']:.10f}（误差 {r['E_term_err']:.3e}）")
    print(f"12. 可达状态域全部验证：I_{{T+1}}={{6000}}={r['I_Tp1_exact']}，不超设备={r['domain_in_range']}")
    print(f"13. 代表时刻采样斜率变化点数（阈值1e-4）：" + ", ".join(f"{t}({label})={r['bp_count'][t]}" for t, label in ctx["REP_TIMES"]))
    print(f"14. 验证 F_t 为凸分段线性：{'是（最大违例 %.2e）' % r['max_conv']}")
    print(f"15. 验证 λ_t(e) 随库存单调不增：{'是（最大违例 %.2e）' % r['max_conv']}")
    print("16. 是否修改 LP/MILP/图2—图4/result1.xlsx：否（本阶段校验前后LP文件SHA-256一致：" +
          ("是" if r["sha_unchanged"] else "否") + "）")
    print(f"17. 新增文件清单：{len(new_files)} 个（见报告第 11 节）")
    print(f"18. 全部验收是否通过：{'通过 %d/%d' % (r['n_pass'], r['n_total']) if r['all_pass'] else '未通过'}")
    print("19. 数值检查通过；LP可能有多组等价最优轨迹，后续LP首步轨迹费用/购电量/约束一致，"
          "论文图2—图4保留锁定 LP 结果。")
    print("=" * 70)
    return {"ctx": ctx, "r": r, "q21": q21, "q22": q22, "fig": fig,
            "report_file": report_file, "log_file": log_file}


if __name__ == "__main__":
    main()
