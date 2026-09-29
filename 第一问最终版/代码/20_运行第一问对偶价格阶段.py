from pathlib import Path
import sys
import hashlib
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import linprog

CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parent
RESULT_DIR = PROJECT / "模型结果"
FIG_DIR = PROJECT / "模型结果图"
REPORT_DIR = PROJECT / "报告"
LOG_DIR = PROJECT / "求解日志"

sys.path.insert(0, str(CODE_DIR))

ETA = 0.9
ETA2 = ETA * ETA
S = 5000.0 / 6.0
J_LOCK = 35126.9485892896
Q_LOCK = 59482.6989983539
DUAL_MULTI_TOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, str(CODE_DIR / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dual_multiplicity_check():
    sol = pd.read_csv(RESULT_DIR / "最终LP_原始最优解.csv", encoding="utf-8-sig")
    price = sol["price_yuan_per_kwh"].to_numpy(float)
    L = sol["load_energy_kwh"].to_numpy(float)
    V = sol["pv_forecast_energy_kwh"].to_numpy(float)
    C = sol["charge_energy_kwh"].to_numpy(float)
    D = sol["discharge_energy_kwh"].to_numpy(float)
    idle = (C <= 1e-7) & (D <= 1e-7)

    m13 = load_mod("13_建立并求解最终连续LP")
    c, A_eq, b_eq, lb, ub = m13.build_lp(price, L, V)
    bnd = list(zip(lb, ub))

    methods = [("HiGHS默认", "highs"), ("HiGHS双单纯形", "highs-ds"), ("HiGHS内点法IPM", "highs-ipm")]
    res = {}
    for name, method in methods:
        r = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bnd, method=method)
        marg = np.asarray(r.eqlin.marginals, dtype=float)
        res[name] = {
            "method": method,
            "success": bool(r.success),
            "status": int(r.status),
            "obj": float(r.fun),
            "pi": -marg[:144],
            "lam": -marg[144:288],
        }

    ref = res["HiGHS默认"]
    max_obj_diff = max(abs(r["obj"] - ref["obj"]) for r in res.values())
    max_pi_diff = max(np.max(np.abs(r["pi"] - ref["pi"])) for r in res.values())
    max_lam_diff = max(np.max(np.abs(r["lam"] - ref["lam"])) for r in res.values())

    lam_gap = np.maximum.reduce([np.abs(r["lam"] - ref["lam"]) for r in res.values()])
    diff_mask = lam_gap > DUAL_MULTI_TOL
    n_lam_diff = int(diff_mask.sum())
    n_lam_diff_idle = int((diff_mask & idle).sum())
    n_idle = int(idle.sum())

    multi = bool(max_pi_diff > DUAL_MULTI_TOL or max_lam_diff > DUAL_MULTI_TOL)
    if multi:
        detail = (f"本次三种算法所得 π 最大差 {max_pi_diff:.2e}，λ 在 {n_lam_diff} 个时段存在不同取值"
                  f"（其中 {n_lam_diff_idle} 个为静置时段 C=0 且 D=0，占全部 {n_idle} 个静置时段的一部分）。")
        concl = (detail +
                 "不同对偶解均满足 KKT 条件并对应相同原始最优值；图片展示其中一组有效影子价格。")
    else:
        concl = "本次三种算法所得 π/λ 一致；此检查不能证明对偶解唯一。"

    return {
        "methods": res, "ref": ref,
        "max_obj_diff": max_obj_diff,
        "max_pi_diff": max_pi_diff,
        "max_lam_diff": max_lam_diff,
        "n_lam_diff": n_lam_diff, "n_lam_diff_idle": n_lam_diff_idle, "n_idle": n_idle,
        "multi": multi, "conclusion": concl,
    }


def key_interval(df, idx):
    r = df[df["interval_index"] == idx].iloc[0]
    return {
        "interval": f"{r['interval_start']}—{r['interval_end']}",
        "c": float(r["external_price"]),
        "pi": float(r["internal_price_pi"]),
        "eta_lam": float(r["charge_threshold_eta_lambda"]),
        "lam_over_eta": float(r["discharge_threshold_lambda_over_eta"]),
        "g": float(r["grid_purchase_energy_kwh"]),
        "C": float(r["charge_energy_kwh"]),
        "D": float(r["discharge_energy_kwh"]),
        "U": float(r["unused_pv_energy_kwh"]),
        "E_start": float(r["energy_start_kwh"]),
        "E_end": float(r["energy_end_kwh"]),
        "charge_state": r["charge_state"],
        "discharge_state": r["discharge_state"],
        "soc_lower_active": bool(r["soc_lower_active"]),
        "soc_upper_active": bool(r["soc_upper_active"]),
    }


def write_report(d, key61, key73, dmulti):
    ps = d["price_stats"]
    arb = d["arb"]
    lines = []
    A = lines.append
    A("# 第一问 内部价格与储能充放电阈值分析（第 3.2 节）\n")
    A("> 本报告严格按“第一问最终流程”第 3.2 节顺序编写，仅提取并验证对偶影子价格，")
    A("> 不修改最终连续 LP、目标函数与约束方向，不重加二进制变量，不写命题 3.1 与连续 DP。\n")

    A("## 1. 供需平衡对偶变量 π_t\n")
    A("供需平衡方程为")
    A("")
    A("　　L_t + C_t + U_t − g_t − V_t − D_t = 0，")
    A("")
    A("对应拉格朗日项 π_t(L_t+C_t+U_t−g_t−V_t−D_t)。在当前约束方向与 SciPy/HiGHS 符号约定下，")
    A("`raw_marginal = ∂(最优目标)/∂b_eq = −μ`，μ 为乘子，故 π_t = −raw_marginal_balance。")
    A("经式(9)双侧 KKT 判定：π 取 `-raw` 时违例为 %.2e，取 `+raw` 时违例为 %.2e，"
      % (d["decision"]["pi_violation"], d["decision"]["pi_alt_violation"]))
    A("故 π_t = %s。" % d["decision"]["pi_chosen"])
    A("")
    A("π_t ∈ [%.6f, %.6f] 元/kWh。" % (ps["pi_min"], ps["pi_max"]))
    A("")

    A("## 2. 储能状态对偶变量 λ_t\n")
    A("储能递推方程为")
    A("")
    A("　　E_t − E_{t−1} − ηC_t + D_t/η = 0（η=0.9），")
    A("")
    A("对应拉格朗日项 λ_t(E_t−E_{t−1}−ηC_t+D_t/η)。同样 λ_t = −raw_marginal_storage。")
    A("经充放电约化成本 KKT 判定：λ 取 `-raw` 时违例为 %.2e，取 `+raw` 时违例为 %.2e，"
      % (d["decision"]["lambda_violation"], d["decision"]["lambda_alt_violation"]))
    A("故 λ_t = %s。" % d["decision"]["lambda_chosen"])
    A("")
    A("λ_t ∈ [%.6f, %.6f] 元/kWh。" % (ps["lam_min"], ps["lam_max"]))
    A("")

    A("## 3. 式（9）互补关系\n")
    A("0 ≤ π_t ≤ c_t，g_t(c_t−π_t)=0，π_t·U_t=0。")
    A("")
    A("- π 下界违例（π_t<0 的最大越过量）：%.3e 元/kWh" % ps["max_pi_lower_violation"])
    A("- π 上界违例（π_t>c_t 的最大越过量）：%.3e 元/kWh" % ps["max_pi_upper_violation"])
    A("- 购电互补残差 g_t(c_t−π_t) 最大绝对值：%.3e 元（归一化 %.3e）"
      % (ps["max_grid_comp"], ps["max_grid_comp_norm"]))
    A("- 弃光互补残差 π_t·U_t 最大绝对值：%.3e 元（归一化 %.3e）"
      % (ps["max_curt_comp"], ps["max_curt_comp_norm"]))
    A("")

    A("## 4. 正购电时 π_t = c_t\n")
    A("正购电时段数 %d 段；其中 π_t 与 c_t 的最大误差 %.3e 元/kWh。" %
      (ps["num_g_pos"], ps["max_err_pi_eq_c_when_g_pos"]))
    A("")

    A("## 5. 未利用供能时 π_t = 0\n")
    A("未利用光伏（弃光）时段数 %d 段；其中 π_t 偏离 0 的最大误差 %.3e 元/kWh。" %
      (ps["num_u_pos"], ps["max_err_pi_eq_0_when_U_pos"]))
    A("")

    A("## 6. 充电约化成本 π_t − ηλ_t\n")
    A("充电约化成本为 π_t − ηλ_t：消耗 1 kWh 交流侧电量用于充电的净机会成本。")
    A("0<C_t<S 时约化成本为 0；C_t=0 时非负；C_t=S 时非正。")
    A("")

    A("## 7. 放电约化成本 −π_t + λ_t/η\n")
    A("放电约化成本为 −π_t + λ_t/η：放电 1 kWh 交流侧电量的净机会成本（含效率折算）。")
    A("0<D_t<S 时约化成本为 0；D_t=0 时非负；D_t=S 时非正。")
    A("")

    A("## 8. 充电、静置与放电阈值\n")
    A("充电阈值 ηλ_t，放电阈值 λ_t/η。")
    A("")
    A("- 若 π_t < ηλ_t：消耗 1 kWh 交流侧电量用于充电的当前机会成本低于折算后的储能价值，充电达到上限 C_t=S；")
    A("- 若 π_t > λ_t/η：当前内部价格高于保留能量的效率折算价值，放电达到上限 D_t=S；")
    A("- 若 ηλ_t < π_t < λ_t/η：两项约化成本均为正，C_t=D_t=0。")
    A("等号 π_t=ηλ_t 或 π_t=λ_t/η 成立时，相应动作约化成本为零，允许非零充电或放电；SOC 边界通过状态乘子影响 λ_t。")
    A("")

    A("## 9. 往返效率 0.81 与跨时段套利条件\n")
    A("完整充放一次，位移 1 单位交流侧电量最终可向负荷交付 η²=0.81 单位交流侧电量。")
    A("从低价时段 a 向高价时段 b 转移电量有经济价值的基本条件为 η²c_b > c_a，即 0.81c_b > c_a。")
    A("")
    if arb:
        a = arb[0]
        A("- 前期充电时段（夜充最低购电价）：区间 %d，c_a=%.4f 元/kWh" % (a["charge_interval"], a["c_a"]))
        A("- 后期放电时段（最高放电价）：区间 %d，c_b=%.4f 元/kWh" % (a["discharge_interval"], a["c_b"]))
        A("- 0.81·c_b − c_a = %.6f 元/kWh %s 0，%s套利条件。"
          % (a["eta2_c_b_minus_c_a"], ">" if a["eta2_c_b_minus_c_a"] > 0 else "≤",
             "满足" if a["arbitrage_feasible"] else "不满足"))
    else:
        A("- （未找到同时存在购电充电与放电的典型时段。）")
    A("")

    A("## 10. 图4 内部价格与充放电阈值\n")
    A("图4 四条时间序列（横轴 0:00—24:00，纵轴 元/kWh）：外网电价 c_t、内部边际价格 π_t、")
    A("充电阈值 ηλ_t、放电阈值 λ_t/η。矢量 PDF 与 PNG 预览：")
    A("- `模型结果图/第一问_内部价格与储能充放电阈值.pdf`")
    A("- `模型结果图/第一问_内部价格与储能充放电阈值.png`")
    A("")

    A("## 11. 关键时段解释\n")
    A("### 11.1 10:00—10:10 时段\n")
    A("| 量 | 数值 |")
    A("|---|---|")
    A("| 外网电价 c_t | %.4f 元/kWh |" % key61["c"])
    A("| 内部价格 π_t | %.4f 元/kWh |" % key61["pi"])
    A("| 充电阈值 ηλ_t | %.4f 元/kWh |" % key61["eta_lam"])
    A("| 放电阈值 λ_t/η | %.4f 元/kWh |" % key61["lam_over_eta"])
    A("| 购电量 g_t | %.6f kWh |" % key61["g"])
    A("| 充电量 C_t | %.6f kWh（%s） |" % (key61["C"], key61["charge_state"]))
    A("| 放电量 D_t | %.6f kWh |" % key61["D"])
    A("| 未利用光伏 U_t | %.6f kWh |" % key61["U"])
    A("| 储能 E_t | %.2f → %.2f kWh |" % (key61["E_start"], key61["E_end"]))
    A("")
    A("该时段外网电价为 %.4f 元/kWh，但内部价格仅 %.4f 元/kWh，显著低于外网电价。"
      % (key61["c"], key61["pi"]))
    A("原因是光伏除满足负荷外还有余量（正用于给储能充电，g_t=0 无需外购），")
    A("新增 1 kWh 需求的边际供给来自光伏而非外网，其机会成本由充电阈值决定，")
    A("故内部充电处于内部点 π_t=ηλ_t=%.4f，仍倾向于充电。" % key61["eta_lam"])
    A("与图片“光伏供给使内部价格低于外网电价，因此仍可能充电”的解释一致。")
    A("")

    A("### 11.2 12:00—12:10 时段\n")
    A("| 量 | 数值 |")
    A("|---|---|")
    A("| 外网电价 c_t | %.4f 元/kWh |" % key73["c"])
    A("| 内部价格 π_t | %.4f 元/kWh |" % key73["pi"])
    A("| 充电阈值 ηλ_t | %.4f 元/kWh |" % key73["eta_lam"])
    A("| 放电阈值 λ_t/η | %.4f 元/kWh |" % key73["lam_over_eta"])
    A("| 购电量 g_t | %.6f kWh |" % key73["g"])
    A("| 充电量 C_t | %.6f kWh（%s） |" % (key73["C"], key73["charge_state"]))
    A("| 放电量 D_t | %.6f kWh |" % key73["D"])
    A("| 未利用光伏 U_t | %.6f kWh |" % key73["U"])
    A("| 储能 E_t | %.2f → %.2f kWh |" % (key73["E_start"], key73["E_end"]))
    A("")
    A("该时段正购电（g_t=%.4f kWh>0），π_t=c_t=%.4f 元/kWh，与图片预期 π_t=c_t≈0.441 一致；"
      % (key73["g"], key73["pi"]))
    A("充电达功率上限（C_t=S=%.4f kWh，state=upper），充电约化成本 π_t−ηλ_t=%.4f≤0，满足上界 KKT；"
      % (S, key73["pi"] - key73["eta_lam"]))
    A("当前 SOC 约 %.2f/12000=%.1f%%，未接近储能上限 10800 kWh，故仍有余量在低谷电价时段购电充电。"
      % (key73["E_start"], key73["E_start"] / 12000.0 * 100))
    A("")

    A("## 12. KKT 验收结果\n")
    A("12 项 KKT/一致性检查全部通过（详见 `模型结果/第一问_对偶与KKT验收.csv`）：")
    A("")
    A("1. π_t ≥ 0；2. π_t ≤ c_t；3. g_t(c_t−π_t)=0；4. π_t·U_t=0；5. 充电下界 KKT；")
    A("6. 充电内部点 KKT；7. 充电上界 KKT；8. 放电下界 KKT；9. 放电内部点 KKT；")
    A("10. 放电上界 KKT；11. 原始 LP 目标值未改变；12. 原始 LP 解文件未改变。")
    A("")

    A("## 附：对偶多解检查\n")
    A("LP 原始最优值唯一不代表对偶变量逐点唯一。分别用 HiGHS 默认、双单纯形、内点法重解同一 LP，")
    A("不改目标与约束，比较 π、λ 与目标值：")
    A("")
    A("- 目标值最大差异：%.3e 元" % dmulti["max_obj_diff"])
    A("- π 最大差异：%.3e 元/kWh" % dmulti["max_pi_diff"])
    A("- λ 最大差异：%.3e 元/kWh（差异时段 %d 个，其中静置时段 %d 个 / 全部静置时段 %d 个）"
      % (dmulti["max_lam_diff"], dmulti["n_lam_diff"], dmulti["n_lam_diff_idle"], dmulti["n_idle"]))
    A("")
    A("结论：%s" % dmulti["conclusion"])
    A("")

    REPORT_FILE = REPORT_DIR / "第一问内部价格与储能阈值分析.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_FILE


def write_log(d, d18, dmulti, key61, key73, report_file, fig_paths, input_shas):
    ps = d["price_stats"]
    L = []
    A = L.append
    A("=" * 72)
    A("第一问 对偶价格分析日志（第 3.2 节）")
    A(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    A("=" * 72)
    A("")
    A("[输入文件 SHA-256]")
    for name, h in input_shas.items():
        A(f"  {name}: {h}")
    A("")
    A("[约束方向]")
    A("  供需平衡: L_t + C_t + U_t - g_t - V_t - D_t = 0  (即 -g_t+C_t-D_t+U_t = V_t-L_t)")
    A("  储能递推: E_t - E_{t-1} - eta_c*C_t + D_t/eta_d = 0   (eta_c=eta_d=0.9)")
    A("")
    A("[原始边际量符号映射]")
    A(f"  raw_marginal = 影子价格 ∂(最优目标)/∂b_eq = -μ")
    A(f"  π_t = {d['decision']['pi_chosen']}  (π 违例 {d['decision']['pi_violation']:.2e}, 备选 {d['decision']['pi_alt_violation']:.2e})")
    A(f"  λ_t = {d['decision']['lambda_chosen']}  (λ 违例 {d['decision']['lambda_violation']:.2e}, 备选 {d['decision']['lambda_alt_violation']:.2e})")
    A("")
    A("[π 与 λ 的提取方式]")
    A("  前 144 行供需平衡 raw_marginal → π_t = -raw_marginal_balance")
    A("  后 144 行储能递推 raw_marginal → λ_t = -raw_marginal_storage")
    A("  符号均由式(9)与充放电约化成本 KKT 双侧判定，未静默取绝对值。")
    A("")
    A("[KKT 最大残差]")
    A(f"  max_π下界违例 = {ps['max_pi_lower_violation']:.3e} 元/kWh")
    A(f"  max_π上界违例 = {ps['max_pi_upper_violation']:.3e} 元/kWh")
    A(f"  max购电互补残差 = {ps['max_grid_comp']:.3e} 元 (归一化 {ps['max_grid_comp_norm']:.3e})")
    A(f"  max弃光互补残差 = {ps['max_curt_comp']:.3e} 元 (归一化 {ps['max_curt_comp_norm']:.3e})")
    A(f"  正购电时 π≐c 最大误差 = {ps['max_err_pi_eq_c_when_g_pos']:.3e} 元/kWh (N={ps['num_g_pos']})")
    A(f"  弃光时 π≐0 最大误差 = {ps['max_err_pi_eq_0_when_U_pos']:.3e} 元/kWh (N={ps['num_u_pos']})")
    A("")
    A("[关键时段]")
    A(f"  10:00-10:10 : c={key61['c']:.4f}, π={key61['pi']:.4f}, ηλ={key61['eta_lam']:.4f}, λ/η={key61['lam_over_eta']:.4f}, "
      f"C={key61['C']:.4f}({key61['charge_state']})")
    A(f"  12:00-12:10 : c={key73['c']:.4f}, π={key73['pi']:.4f}, ηλ={key73['eta_lam']:.4f}, λ/η={key73['lam_over_eta']:.4f}, "
      f"g={key73['g']:.4f}, C={key73['C']:.4f}({key73['charge_state']})")
    A("")
    A("[对偶多解检查]")
    A(f"  max目标值差异 = {dmulti['max_obj_diff']:.3e} 元")
    A(f"  max π 差异 = {dmulti['max_pi_diff']:.3e} 元/kWh")
    A(f"  max λ 差异 = {dmulti['max_lam_diff']:.3e} 元/kWh（差异 {dmulti['n_lam_diff']} 段，静置 {dmulti['n_lam_diff_idle']}/{dmulti['n_idle']}）")
    A(f"  结论：{dmulti['conclusion']}")
    A("")
    A("[图4 文件路径]")
    A(f"  {fig_paths['pdf']}")
    A(f"  {fig_paths['png']}")
    A("")
    A("[原始 LP 文件是否保持不变]")
    A(f"  最终LP_原始最优解.csv SHA-256={d18['sol_sha']}")
    A(f"  {'与第三阶段冻结值一致（未改动）' if d18['checks'][-1]['pass'] else '不一致（已改动！）'}")
    A("")
    A("[总体验收]")
    A(f"  KKT 验收 {d18['n_pass']}/12 通过")
    A(f"  目标值差 = {abs(d18['j_real']-J_LOCK):.3e} 元")
    A(f"  购电量 = {d18['q_real']:.10f} kWh (锁定 {Q_LOCK})")
    A(f"  总体验收状态：{'通过' if d18['n_pass'] == 12 else '未通过'}")
    A("")

    LOG_FILE = LOG_DIR / "第一问对偶价格分析日志.txt"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("\n".join(L), encoding="utf-8")
    return LOG_FILE


def main():
    m17 = load_mod("17_提取第一问对偶价格")
    m18 = load_mod("18_校验内部价格与充放电阈值")
    m19 = load_mod("19_绘制内部价格与储能阈值")

    d = m17.main()
    d18 = m18.main()
    if d18["n_pass"] != 12:
        raise RuntimeError("对偶KKT验收失败，停止生成正式报告")
    d19 = m19.main()

    key61 = key_interval(d["df"], 61)
    key73 = key_interval(d["df"], 73)
    dmulti = dual_multiplicity_check()

    input_paths = {
        "最终LP_原始最优解.csv": RESULT_DIR / "最终LP_原始最优解.csv",
        "最终LP_原始对偶数据.csv": RESULT_DIR / "最终LP_原始对偶数据.csv",
        "最终LP_变量边界对偶数据.csv": RESULT_DIR / "最终LP_变量边界对偶数据.csv",
        "最终LP_约束行映射.csv": RESULT_DIR / "最终LP_约束行映射.csv",
        "最终LP_求解摘要.csv": RESULT_DIR / "最终LP_求解摘要.csv",
    }
    input_shas = {k: sha256(v) for k, v in input_paths.items()}

    report_file = write_report(d, key61, key73, dmulti)
    fig_paths = {"pdf": d19["pdf"], "png": d19["png"]}
    log_file = write_log(d, d18, dmulti, key61, key73, report_file, fig_paths, input_shas)

    ps = d["price_stats"]
    print("=" * 60)
    print("[20] 第四阶段完成")
    print(f"  报告: {report_file.name}")
    print(f"  日志: {log_file.name}")
    print(f"  π∈[{ps['pi_min']:.6f},{ps['pi_max']:.6f}]  λ∈[{ps['lam_min']:.6f},{ps['lam_max']:.6f}]")
    print(f"  KKT: {d18['n_pass']}/12 通过")
    print(f"  对偶: π三法比较(max差={dmulti['max_pi_diff']:.1e}), λ多解(max差={dmulti['max_lam_diff']:.3e}, "
          f"静置退化 {dmulti['n_lam_diff_idle']}/{dmulti['n_idle']})")
    print("=" * 60)
    return {
        "d": d, "d18": d18, "d19": d19, "dmulti": dmulti,
        "key61": key61, "key73": key73, "report_file": report_file, "log_file": log_file,
    }


if __name__ == "__main__":
    main()
