import importlib.util
import sys
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy

PROJECT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT / "报告"
LOG_DIR = PROJECT / "求解日志"

REPORT_MD = REPORT_DIR / "第一问最终连续LP求解报告.md"
LOG_TXT = LOG_DIR / "第一问最终连续LP求解日志.txt"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    m13 = load("lp13", "13_建立并求解最终连续LP.py")
    m14 = load("lp14", "14_校验最终连续LP结果.py")
    r13 = m13.main()
    r14 = m14.main()

    cross_failed = r14["benchmark_available"] and \
        (r14["obj_abs"] > 1e-5 or r14["q_abs"] > 1e-4)
    if r14["hard_pass"] != r14["hard_total"] or cross_failed:
        raise RuntimeError("最终LP验收失败")
    t = r13["totals"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    md = []
    md.append("# 第一问最终连续LP求解报告")
    md.append("")
    md.append(f"- 生成时间：{now}")
    md.append("")
    md.append("## 1. 运行环境")
    md.append(f"- Python {pyver}；{platform.system()} {platform.release()}")
    md.append(f"- scipy {scipy.__version__}；numpy {np.__version__}；pandas {pd.__version__}")
    md.append(f"- 求解器：HiGHS {r13['highs_version']}（scipy.optimize.linprog, method=highs）")
    md.append("")
    md.append("## 2. 输入文件及SHA-256")
    md.append("- `处理后数据/附件一_第一问基础数据.csv`")
    md.append(f"- SHA-256：`{r13['csv_sha']}`（与第一阶段冻结值一致）")
    md.append("")
    md.append("## 3. 模型变量数量")
    md.append(f"- 连续变量 {r13['n_var']} 个：g(144) + C(144) + D(144) + U(144) + E(145)")
    md.append("")
    md.append("## 4. 等式约束数量")
    md.append(f"- 等式 {r13['n_eq']} 条：供需平衡 144 + 储能递推 144")
    md.append("")
    md.append("## 5. 变量边界")
    md.append("- g_t ≥ 0；U_t ≥ 0")
    md.append("- 0 ≤ C_t ≤ S = 5000×1/6 = 833.3333333333334 kWh")
    md.append("- 0 ≤ D_t ≤ S = 833.3333333333334 kWh")
    md.append("- 1200 ≤ E_t ≤ 10800（t=0..144）；E_0 = E_144 = 6000")
    md.append("")
    md.append("## 6. 目标函数")
    md.append("- min J_1 = Σ_{t=1}^{144} c_t · g_t（g_t 已是电量 kWh，不乘 Δt）")
    md.append("")
    md.append("## 7. 供需平衡方向")
    md.append("- L_t + C_t + U_t − g_t − V_t − D_t = 0（即 −g_t + C_t − D_t + U_t = V_t − L_t）")
    md.append("")
    md.append("## 8. SOC递推方向")
    md.append("- E_t − E_{t−1} − η_c·C_t + D_t/η_d = 0，η_c = η_d = 0.9")
    md.append("")
    md.append("## 9. 求解器状态")
    md.append(f"- 状态码 {r13['status_code']}：{r13['message']}")
    md.append(f"- 求解耗时 {r13['elapsed']:.5f} s；迭代次数 {r13['nit']}")
    md.append("")
    md.append("## 10. 最优费用")
    md.append(f"- J_1 = {r13['objective']:.10f} 元")
    md.append("")
    md.append("## 11. 总购电量")
    md.append(f"- Σ g_t = {t['grid']:.10f} kWh")
    md.append("")
    md.append("## 12. 总充电量")
    md.append(f"- Σ C_t = {t['charge']:.10f} kWh")
    md.append("")
    md.append("## 13. 总放电量")
    md.append(f"- Σ D_t = {t['discharge']:.10f} kWh")
    md.append("")
    md.append("## 14. 总未利用光伏量")
    md.append(f"- Σ U_t = {t['unused_pv']:.10f} kWh")
    md.append("")
    md.append("## 15. 储电量最小值和最大值")
    md.append(f"- min E_t = {t['E_min']:.6f} kWh；max E_t = {t['E_max']:.6f} kWh")
    md.append("")
    md.append("## 16. 最大充放电量及对应功率")
    md.append(f"- 最大充电量 {t['max_charge_energy']:.6f} kWh（{t['max_charge_power']:.6f} kW）")
    md.append(f"- 最大放电量 {t['max_discharge_energy']:.6f} kWh（{t['max_discharge_power']:.6f} kW）")
    md.append("")
    md.append("## 17. 最大约束残差")
    md.append(f"- 供需平衡 max|r| = {r14['max_balance_resid']:.3e} kWh")
    md.append(f"- 储能递推 max|r| = {r14['max_energy_resid']:.3e} kWh")
    md.append("")
    md.append("## 18. 同时充放电诊断")
    md.append(f"- 阈值 1e-7：{r14['n_sim_1e7']} 个；1e-5：{r14['n_sim_1e5']} 个；1e-3：{r14['n_sim_1e3']} 个")
    md.append(f"- 明细：{r14['sim_detail']}")
    md.append("")
    md.append("## 19. 与旧MILP费用误差")
    if r14["benchmark_available"]:
        md.append(f"- objective_abs_error = {r14['obj_abs']:.6e} 元（rel {r14['obj_rel']:.6e}）")
        md.append(f"- 验收阈值 1e-5 元：{'通过' if r14['obj_abs'] <= 1e-5 else '未通过'}")
    else:
        md.append("- 未提供开发期历史MILP基准，本项跳过，不影响连续LP复现。")
    md.append("")
    md.append("## 20. 与旧MILP购电量误差")
    if r14["benchmark_available"]:
        md.append(f"- purchase_abs_error = {r14['q_abs']:.6e} kWh")
        md.append(f"- 验收阈值 1e-4 kWh：{'通过' if r14['q_abs'] <= 1e-4 else '未通过'}")
    else:
        md.append("- 未提供开发期历史MILP基准，本项跳过。")
    md.append("")
    md.append("## 21. 后续分析")
    md.append("- 同时充放电消去验证（效率消去）")
    md.append("- 对偶边际价格分析（本阶段已原样保存原始对偶/边界边际数据，不解释）")
    md.append("- 后续时域LP价值函数与Bellman一致性分析")
    md.append("- 不在本报告展开：内部电价含义、对偶阈值、命题3.1、连续DP、论文结论")
    md.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    log = []
    log.append(f"[{now}] 第一问最终连续LP阶段")
    log.append(f"运行环境：Python {pyver} | scipy {scipy.__version__} | HiGHS {r13['highs_version']}")
    log.append(f"输入：处理后数据/附件一_第一问基础数据.csv  SHA-256={r13['csv_sha']}")
    log.append(f"模型：连续LP | 变量 {r13['n_var']} | 等式 {r13['n_eq']}")
    log.append(f"求解状态：{r13['message']}（status={r13['status_code']}）")
    log.append(f"求解耗时：{r13['elapsed']:.5f} s | 迭代 {r13['nit']}")
    log.append(f"最优费用 J_1 = {r13['objective']:.10f} 元")
    log.append(f"总购电量 = {t['grid']:.10f} kWh")
    log.append(f"总充电量 = {t['charge']:.10f} kWh")
    log.append(f"总放电量 = {t['discharge']:.10f} kWh")
    log.append(f"总未利用光伏 = {t['unused_pv']:.10f} kWh")
    log.append(f"储电量 min={t['E_min']:.6f} max={t['E_max']:.6f} kWh")
    log.append(f"最大充电量 {t['max_charge_energy']:.6f} kWh（{t['max_charge_power']:.6f} kW）")
    log.append(f"最大放电量 {t['max_discharge_energy']:.6f} kWh（{t['max_discharge_power']:.6f} kW）")
    log.append(f"约束校验：硬约束 {r14['hard_pass']}/{r14['hard_total']} 通过")
    log.append(f"供需平衡 max|r|={r14['max_balance_resid']:.3e} kWh；储能递推 max|r|={r14['max_energy_resid']:.3e} kWh")
    log.append(f"首末储电量 E_0={r14['E0']:.6f} E_144={r14['E144']:.6f} kWh")
    log.append(f"同时充放电：1e-7:{r14['n_sim_1e7']} 1e-5:{r14['n_sim_1e5']} 1e-3:{r14['n_sim_1e3']}")
    if r14["benchmark_available"]:
        log.append(f"与旧MILP互证：费用误差 {r14['obj_abs']:.6e} 元；购电量误差 {r14['q_abs']:.6e} kWh")
    else:
        log.append("与旧MILP互证：未提供开发期历史基准，已跳过")
    log.append(f"原始对偶数据：已保存（最终LP_原始对偶数据.csv / 最终LP_变量边界对偶数据.csv）")
    log.append("")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_TXT.write_text("\n".join(log), encoding="utf-8")

    print(f"[15] 已生成报告：{REPORT_MD.name}")
    print(f"[15] 已生成日志：{LOG_TXT.name}")
    return {"r13": r13, "r14": r14}


if __name__ == "__main__":
    main()
