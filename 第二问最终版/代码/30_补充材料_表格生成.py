from __future__ import annotations

import importlib.util
import sys
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
import pandas as pd

OUT = C.PROJECT_DIR / "论文补充材料"
RES = C.RESULT_DIR
ARCH = C.PROJECT_DIR / "分档留档"

DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
DATE_TAG = {"2025-03-20": "2025.3.20", "2025-06-21": "2025.6.21",
            "2025-09-23": "2025.9.23", "2025-12-21": "2025.12.21"}
SPEC_SLOTS = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
              "16:00-16:10", "18:00-18:10", "20:00-20:10"]
SEGS = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
        "12:00-16:00", "16:00-20:00", "20:00-24:00"]

LOG: list[str] = []


def p(msg: str = "") -> None:
    LOG.append(msg)
    print(msg, flush=True)


def f6(x) -> str:
    return f"{float(x):.6f}"


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8")
    p(f"  [写出] {name}")


def md_table(header: list[str], rows: list[list[str]],
             align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def load_all():
    d = {}
    d["slot"] = pd.read_csv(RES / "第二问_指定时段计划购电量.csv")
    d["day"] = pd.read_csv(RES / "第二问_指定日期全天购电量与费用.csv")
    d["cost"] = pd.read_csv(RES / "第二问_指定日期费用分解.csv")
    d["emerg"] = pd.read_csv(RES / "第二问_指定日期紧急购电事件.csv")
    d["batt"] = pd.read_csv(RES / "第二问_指定日期储能充放电与储电量.csv")
    d["arm"] = pd.read_csv(RES / "第二问_六类方案重新计算对照.csv")
    d["nostore"] = pd.read_csv(RES / "第二问_无储能基准.csv")
    d["reserve"] = pd.read_csv(RES / "第二问_固定保留策略扫描.csv")
    d["grid"] = pd.read_csv(RES / "第二问_DP三档网格年度收敛.csv")
    d["anchor"] = pd.read_csv(ARCH / "锚点对照_分解3日.csv")
    return d


def table_a(d: dict) -> None:
    slot = d["slot"].set_index("时段")
    day = d["day"].set_index("日期")
    cost = d["cost"].set_index("日期")

    body_flat, body_wide = [], []
    for dt in DATES:
        g = [float(slot.loc[s, dt]) for s in SPEC_SLOTS]
        gtot = float(day.loc[dt, "全天计划购电量_kWh"])
        btot = float(day.loc[dt, "全天紧急购电量_kWh"])
        fee = float(day.loc[dt, "全天结算费用_元"])
        pfee = float(cost.loc[dt, "计划费_元"])
        efee = float(cost.loc[dt, "紧急费_元"])
        body_flat.append((dt, g, gtot, btot, fee, pfee, efee))
        body_wide.append([dt] + [f6(v) for v in g] + [f6(gtot), f6(fee)])

    txt = ["# 表A  指定日期购电量（复刻题目「表1」格式）\n",
           "> 单位：kWh（购电量）、元（购电费）。数值取自 `模型结果/第二问_指定时段计划购电量.csv`"
           " 与 `第二问_指定日期全天购电量与费用.csv`（主档：分解预测 + 3 日光伏回望 + DP 价值执行器）。\n"]
    for k, (dt, g, gtot, btot, fee, pfee, efee) in enumerate(body_flat, 1):
        txt.append(f"## 表A-{k}  {DATE_TAG[dt]}\n")
        txt.append(md_table(
            ["时间段", "购电量", "时间段", "购电量", "时间段", "购电量"],
            [[SPEC_SLOTS[0], f6(g[0]), SPEC_SLOTS[1], f6(g[1]), SPEC_SLOTS[2], f6(g[2])],
             [SPEC_SLOTS[3], f6(g[3]), SPEC_SLOTS[4], f6(g[4]), SPEC_SLOTS[5], f6(g[5])],
             ["**全天购电量**", f"**{f6(gtot)}**", "**全天购电费**", f"**{f6(fee)}**", "", ""]],
            ["---", "--:", "---", "--:", "---", "--:"]))
        txt.append(f"其中：六个指定时段计划量合计 = {f6(sum(g))} kWh（不是紧急补购量）；"
                   f"全天购电费 = 计划费 {f6(pfee)} + 紧急费 {f6(efee)}。\n")
        if btot > 0:
            txt.append(f"该日发生紧急购电 **{f6(btot)} kWh**，明细按时段给出"
                       f"（见 `表C`），紧急电价 = 交易时刻电价的 5 倍。\n")
        else:
            txt.append("该日**无**紧急购电（实际净缺口全部由储能与计划购电量覆盖）。\n")

    txt.append("---\n\n## 表A-宽  四指定日期并排（紧凑版式，供版面紧张时替换）\n")
    txt.append(md_table(
        ["日期"] + SPEC_SLOTS + ["全天计划购电量", "全天购电费"],
        body_wide,
        ["---"] + ["--:"] * 6 + ["--:", "--:"]))

    write("表A_指定日期购电量_题目表1格式.md", "\n".join(txt))

    rows = []
    for dt, g, gtot, btot, fee, pfee, efee in body_flat:
        for s, v in zip(SPEC_SLOTS, g):
            rows.append([dt, s, f6(v)])
        rows.append([dt, "六个指定时段计划量合计", f6(sum(g))])
        rows.append([dt, "全天计划购电量", f6(gtot)])
        rows.append([dt, "全天紧急购电量", f6(btot)])
        rows.append([dt, "全天购电费", f6(fee)])
    pd.DataFrame(rows, columns=["日期", "项目", "数值"]).to_csv(
        OUT / "表A_指定日期购电量.csv", index=False, encoding="utf-8-sig")
    p("  [写出] 表A_指定日期购电量.csv")


def table_b(d: dict) -> None:
    batt = d["batt"].set_index("日期")
    txt = ["# 表B  指定日期储能设备充放电量与储电量（复刻题目「表2」格式）\n",
           "> 单位：kWh。充/放电量按**交流侧**统计；储电量按 `η·充 − 放/η` 递推，"
           "**不得**用「充 − 放」直接相减。数据取自 `模型结果/第二问_指定日期储能充放电与储电量.csv`。\n"]
    for k, dt in enumerate(DATES, 1):
        r = batt.loc[dt]
        rows = [[s, f6(r[f"{s}_充电量_kWh"]), f6(r[f"{s}_放电量_kWh"])] for s in SEGS]
        rows.append(["**0:00 储电量**", f"**{f6(r['0:00储电量_kWh'])}**", ""])
        rows.append(["**24:00 储电量**", f"**{f6(r['24:00储电量_kWh'])}**", ""])
        txt.append(f"## 表B-{k}  {DATE_TAG[dt]}\n")
        txt.append(md_table(["时间段", "充电量", "放电量"], rows,
                            ["---", "--:", "--:"]))
        c_tot = sum(float(r[f"{s}_充电量_kWh"]) for s in SEGS)
        d_tot = sum(float(r[f"{s}_放电量_kWh"]) for s in SEGS)
        e0, e1 = float(r["0:00储电量_kWh"]), float(r["24:00储电量_kWh"])
        txt.append(f"合计：充电 {f6(c_tot)} / 放电 {f6(d_tot)} kWh；"
                   f"储电量 {f6(e0)} → {f6(e1)}（Δ = {f6(e1 - e0)}）。\n")

    txt.append("---\n\n> **跨日连续性说明**：24:00 储电量即为次日 0:00 的初始储电量。\n"
               "> 本问**不施加**「0:00 与 24:00 储电量相同」的日循环约束（该约束仅问题 1 要求）。\n"
               "> 指定日期 0:00 储电量："
               + "；".join(f"{DATE_TAG[dt]} {f6(batt.loc[dt, '0:00储电量_kWh'])}" for dt in DATES)
               + " kWh ——\n> 均不等于 6000 kWh，正说明库存沿时间轴连续演进。\n")
    write("表B_指定日期充放电_题目表2格式.md", "\n".join(txt))

    rows = []
    for dt in DATES:
        r = batt.loc[dt]
        for s in SEGS:
            rows.append([dt, s, f6(r[f"{s}_充电量_kWh"]), f6(r[f"{s}_放电量_kWh"]), "", ""])
        rows.append([dt, "0:00 储电量", "", "", f6(r["0:00储电量_kWh"]), ""])
        rows.append([dt, "24:00 储电量", "", "", "", f6(r["24:00储电量_kWh"])])
    pd.DataFrame(rows, columns=["日期", "时间段", "充电量_kWh", "放电量_kWh",
                                "0:00储电量_kWh", "24:00储电量_kWh"]
                 ).to_csv(OUT / "表B_指定日期充放电.csv", index=False, encoding="utf-8-sig")
    p("  [写出] 表B_指定日期充放电.csv")


def table_c(d: dict) -> None:
    em = d["emerg"]
    day = d["day"].set_index("日期")
    by_date: dict[str, list] = {dt: [] for dt in DATES}
    for _, r in em.iterrows():
        by_date[str(r["日期"])].append((str(r["购电时间段"]),
                                       float(r["购电量_kWh"]),
                                       int(r["连续时段数"])))

    txt = ["# 表C  指定日期紧急购电量（复刻题目「表3」格式）\n",
           "> 单位：kWh。紧急购电电价 = **交易时刻电价的 5 倍**。\n"
           "> 事件**按连续发生的 10 min 时段合并**；同日相邻事件累计，**跨日事件分开**。\n"
           "> 数据取自 `模型结果/第二问_指定日期紧急购电事件.csv`（与 `提交结果/result2.xlsx` 的"
           "「紧急购电量」工作表同源）。\n"]

    maxn = max((len(v) for v in by_date.values()), default=0)
    rows = []
    for i in range(maxn):
        row = []
        for dt in DATES:
            lst = by_date[dt]
            if i < len(lst):
                row += [lst[i][0], f6(lst[i][1])]
            else:
                row += ["", ""]
        rows.append(row)
    head = []
    for dt in DATES:
        head += [f"{DATE_TAG[dt]} 时间段", f"{DATE_TAG[dt]} 购电量"]
    txt.append("## 版式一：四日期并排（复刻题目表3）\n")
    txt.append(md_table(head, rows, ["---"] + ["--:"] * (len(head) - 1)))

    rows2 = []
    for dt in DATES:
        lst = by_date[dt]
        if not lst:
            rows2.append([dt, "—", "0.000000", "0", "该日无紧急购电"])
        else:
            for s, v, n in lst:
                rows2.append([dt, s, f6(v), str(n), ""])
    txt.append("\n## 版式二：长表（与 `result2.xlsx`「紧急购电量」工作表同口径）\n")
    txt.append(md_table(["日期", "紧急购电时间段", "紧急购电量", "连续时段数", "备注"],
                        rows2, ["---", "---", "--:", "--:"]))
    txt.append("\n> 题目表4 的填写示例为「同一日期可有多个时间段、每段一行」，"
               "本表版式二与之完全一致。\n")

    rows3 = []
    for dt in DATES:
        lst = by_date[dt]
        tot = sum(v for _, v, _ in lst)
        rows3.append([DATE_TAG[dt], str(len(lst)), f6(tot),
                      f6(float(day.loc[dt, "全天紧急购电量_kWh"]))])
    txt.append("\n## 汇总校验（事件数 / 事件量合计 vs 表A 全天紧急购电量）\n")
    txt.append(md_table(["日期", "事件数", "事件量合计", "表A 全天紧急购电量"],
                        rows3, ["---", "--:", "--:", "--:"]))
    txt.append("\n> 两列应完全相等——本表即表A 中紧急量的逐时段展开。\n")
    write("表C_指定日期紧急购电_题目表3格式.md", "\n".join(txt))

    out = []
    for dt in DATES:
        for s, v, n in by_date[dt]:
            out.append([dt, s, f6(v), n])
    pd.DataFrame(out, columns=["日期", "紧急购电时间段", "购电量_kWh", "连续时段数"]
                 ).to_csv(OUT / "表C_指定日期紧急购电.csv", index=False, encoding="utf-8-sig")
    p("  [写出] 表C_指定日期紧急购电.csv")


def table_d(d: dict) -> None:
    ns = float(d["nostore"].loc[d["nostore"]["项目"] == "无储能购电费", "数值"].iloc[0])
    arm = d["arm"].set_index(["比较方式", "执行器"])

    def tot(grp, ex):
        return float(arm.loc[(grp, ex), "总费用_元"])

    def ekh(grp, ex):
        return float(arm.loc[(grp, ex), "紧急量_kWh"])

    keep = tot("各自重订", "DP价值执行器")
    rows = [
        ["L0  无储能、事后按实购电", f6(ns), "—", "—", "购电量 = 实际净负荷正部；事后基准"],
        ["L1  有储能 + 解析响应（因果）", f6(tot("各自重订", "解析响应")), "", f6(ekh("各自重订", "解析响应")),
         "闭式规则：缺额即放电、富余即充电"],
        ["L2  有储能 + MPC 滚动优化（因果）", f6(tot("各自重订", "MPC")), "", f6(ekh("各自重订", "MPC")),
         "完整剩余时域 + 误差修正滚动"],
        ["L3  有储能 + 价值函数引导执行器（因果）", f6(tot("各自重订", "DP价值执行器")), "", f6(ekh("各自重订", "DP价值执行器")),
         "**本问主结果**（跨日连续 + 各自重订）"],
    ]
    for r in rows[1:]:
        r[2] = f6(ns - float(r[1]))

    txt = ["# 表D  成本阶梯：从「不调控」到「价值函数引导」\n",
           f"> 比较基准：2025 年评分期（2025-02-01 … 12-31，334 天）。"
           f"L0 取自 `模型结果/第二问_无储能基准.csv`，L1–L3 取自"
           f"`模型结果/第二问_六类方案重新计算对照.csv`（各自重订组）。\n"]
    txt.append(md_table(["档位", "年度购电费（元）", "较 L0 节省（元）",
                         "年度紧急购电量（kWh）", "说明"],
                        rows, ["---", "--:", "--:", "--:", "---"]))

    save = ns - keep
    pct = save / ns * 100.0
    txt.append(f"\n## 比较口径\n\nL0 是事后按实购电基准，L1–L3 是因果调度。L0 与 L3 的账单差为 {f6(save)} 元（{pct:.4f}%），同时涉及信息、采购方法和储能配置，不能全部归为储能贡献。L1 与 L3 是各自重订的整条策略链比较，不是同一固定计划下的纯执行器贡献。\n")
    r0 = d["reserve"]
    txt.append("\n## 反向对照：固定保留比例全部增本（L1 的规则族加固）\n")
    rr = [[f"{float(x['α百分比'].strip('%')):.3f}%",
           f6(x["1月紧急量_kWh"]), f6(x["2-12月总费用_元"]), f6(x["较α=0增加_元"])]
          for _, x in r0.iterrows()]
    txt.append(md_table(["备用比例 α", "1 月紧急量（kWh）", "2–12 月总费用（元）",
                         "较 α=0 增加（元）"], rr, ["--:", "--:", "--:", "--:"]))
    txt.append("\n> 五档结果仅支持本数据集和该保留规则族的观察。1月紧急量是固定保留扫描中的独立诊断，不代表主方案在预热期调度；实际评分从2月初6000 kWh启动。\n")
    write("表D_成本阶梯与储能价值.md", "\n".join(txt))

    pd.DataFrame(rows, columns=["档位", "年度购电费_元", "较L0节省_元",
                                "年度紧急购电量_kWh", "说明"]).to_csv(
        OUT / "表D_成本阶梯与储能价值.csv", index=False, encoding="utf-8-sig")
    p("  [写出] 表D_成本阶梯与储能价值.csv")
    return ns, keep


def table_e(d: dict, ns: float, keep: float) -> None:
    old = pd.read_csv(OUT / "表E_储能与信息_2x2.csv")
    causal_nostore = float(old.loc[(old["储能配置"] == "无储能") & (old["信息结构"] == "因果信息"), "年度购电费_元"].iloc[0])
    txt = ["# 表E  储能与采购方案对照（尚非受控因子分解）\n",
           "> 保留既有数字；两组因果方案的采购方法不同，不作为纯信息价值、纯储能价值或交互效应的估计。\n"]
    rows = [["无储能", f6(ns), f6(causal_nostore)], ["有储能", "未计算，本轮不追加实验", f6(keep)]]
    txt.append(md_table(["配置", "完全信息/事后基准", "因果方案"], rows))
    txt.append(f"\n无储能因果方案采用点预测直接采购 g=max(Nhat,0)，已有费用合计 {f6(causal_nostore)} 元。有储能方案采用情景 LP 优化采购再由 DP 执行。\n")
    txt.append(f"\n两方案账单差 {f6(causal_nostore-keep)} 元混合了储能与采购策略差异；无储能点预测方案相对按实采购的差额 {f6(causal_nostore-ns)} 元仅描述该策略的回测差距。完全信息有储能未计算，同情景优化无储能亦未计算，不能认定正交分解完整、交互项方向或投资优劣。\n")
    write("表E_储能与信息_2x2因子.md", "\n".join(txt))

    pd.DataFrame([
        ["无储能", "完全信息", f6(ns)],
        ["无储能", "因果信息", f6(causal_nostore)],
        ["有储能", "完全信息", ""],
        ["有储能", "因果信息", f6(keep)],
    ], columns=["储能配置", "信息结构", "年度购电费_元"]).to_csv(
        OUT / "表E_储能与信息_2x2.csv", index=False, encoding="utf-8-sig")
    p("  [写出] 表E_储能与信息_2x2.csv")
    return causal_nostore


def table_f(d: dict, ns: float, keep: float) -> None:
    arm = d["arm"].set_index(["比较方式", "执行器"])
    an = d["anchor"].set_index("指标")
    r0 = d["reserve"]

    a_resp = float(arm.loc[("各自重订", "解析响应"), "总费用_元"])
    a_mpc = float(arm.loc[("各自重订", "MPC"), "总费用_元"])
    e_resp = float(arm.loc[("各自重订", "解析响应"), "紧急量_kWh"])
    e_mpc = float(arm.loc[("各自重订", "MPC"), "紧急量_kWh"])

    mae7 = float(an.loc["净负荷MAE_kW", "分解+七日光伏"])
    mae3 = float(an.loc["净负荷MAE_kW", "分解+三日光伏"])
    a7 = pd.read_csv(ARCH / "分解7日" / "第二问_执行器对照表.csv").set_index(["比较方式", "执行器"])
    cost7 = float(a7.loc[("各自重订", "DP价值执行器"), "总费用_元"])
    cost3 = keep

    a75 = float(r0.loc[r0["α"] == 0.75, "较α=0增加_元"].iloc[0])
    a75p = float(r0.loc[r0["α"] == 0.75, "相对增幅"].iloc[0].strip("%"))

    rows = [
        ["① 更复杂的 MPC 反而更贵",
         f"MPC {f6(a_mpc)} vs 解析响应 {f6(a_resp)}",
         f"**+{f6(a_mpc - a_resp)} 元（+{(a_mpc / a_resp - 1) * 100:.3f}%）**；"
         f"紧急量 {f6(e_resp)} → {f6(e_mpc)} kWh（+{(e_mpc / e_resp - 1) * 100:.2f}%）",
         "当前 MPC 实现费用较高，具体原因需要针对性对照"],
        ["② 预测精度更高反而更贵",
         f"净负荷 MAE {mae3} vs {mae7} kW",
         f"MAE 更优的 7 日档费用 {f6(cost7)} 元，"
         f"**比 3 日档的 {f6(cost3)} 元更贵 {f6(cost7 - cost3)} 元**",
         "MAE 不是结算费用的代理指标"],
        ["③ 固定保留比例全部增本",
         f"α: 0 → 75%（5 档）",
         f"费用**单调上升**，α=75% 时多花 **{f6(a75)} 元（+{a75p:.2f}%）**",
         "仅针对已检验的保留规则与数据集"],
        ["④ 日前模型存在系统性乐观偏差",
         "30 情景内诊断",
         f"平均绝对差 **{pd.read_csv(RES / '第二问_日前乐观偏差诊断.csv')['平均绝对差_元'].mean():.4f} 元/天**；统计范围为现有30情景",
         "两种情景内评估都使用未来轨迹；差异还包含动作限制与网格近似"],
    ]
    txt = ["# 表F  四条反直觉发现\n",
           "> 记录本数据集上的四项观测，分别注明解释边界；不预设统一因果机制。\n"]
    txt.append(md_table(["#", "反直觉现象", "数值证据", "机理解释"], rows,
                        ["---", "---", "---", "---"]))
    txt.append("\n## 解释边界\n\nMPC 是状态反馈下的滚动优化，不是静态规则。DP 的价值函数可用于解释经济保留，但这些对照不证明其为唯一或结构上必需的方法。预测精度与费用不必同向，仍需结合误差时段、电价和库存分析；本轮不追加机制实验。逐情景诊断不等于实际因果策略的期望费用。\n")
    write("表F_四条反直觉发现.md", "\n".join(txt))

    pd.DataFrame(rows, columns=["#", "反直觉现象", "数值证据", "机理解释"]
                 ).to_csv(OUT / "表F_四条反直觉发现.csv", index=False,
                          encoding="utf-8-sig")
    p("  [写出] 表F_四条反直觉发现.csv")


def table_h(d: dict) -> None:
    g = d["grid"]
    rows = [[x["网格档"], f6(x["年计划购电量_kWh"]), f6(x["年紧急购电量_kWh"]),
             f6(x["年总费用_元"]), f6(x["年末库存_kWh"]), f"{float(x['运行时间_s']):.1f}"]
            for _, x in g.iterrows()]
    base = float(g["年总费用_元"].iloc[0])
    last = float(g["年总费用_元"].iloc[-1])
    txt = ["# 表H  离散化收敛性\n",
           "> 价值函数在储电量轴上以 δ 为步长离散 + 线性插值。三档网格下重新求解全年。\n"]
    txt.append("## H-1 价值函数网格收敛（δ）\n")
    txt.append(md_table(["网格步长 δ", "年计划购电量（kWh）", "年紧急购电量（kWh）",
                         "年总费用（元）", "年末库存（kWh）", "运行时间（s）"],
                        rows, ["---", "--:", "--:", "--:", "--:", "--:"]))
    txt.append(f"\n> 三档极差仅 **{f6(abs(base - last))} 元"
               f"（相对 {abs(base - last) / base * 100:.4f}%）**，"
               f"年末库存按现有输出精度一致；三档变化较小支持当前工程配置，但未证明连续极限或给出误差上界。\n")
    detail = pd.read_csv(RES / "第二问_DP三档网格指定日期收敛.csv")
    txt.append("\n## H-2 指定日期网格敏感性（既有结果，不追加实验）\n")
    txt.append(md_table(list(detail.columns), [[str(v) for v in row] for row in detail.itertuples(index=False, name=None)]))
    write("表H_收敛性.md", "\n".join(txt))

    pd.DataFrame(rows, columns=["网格步长δ", "年计划购电量_kWh", "年紧急购电量_kWh",
                                "年总费用_元", "年末库存_kWh", "运行时间_s"]
                 ).to_csv(OUT / "表H_收敛性.csv", index=False, encoding="utf-8-sig")
    p("  [写出] 表H_收敛性.csv")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p("=" * 74)
    p("第二问 30 —— 论文补充材料生成（严格增量：只读既有结果，只写 论文补充材料/）")
    p("=" * 74)
    d = load_all()
    p("[已读入 10 个既有结果文件]")
    p("")
    table_a(d)
    table_b(d)
    table_c(d)
    ns, keep = table_d(d)
    causal_nostore = table_e(d, ns, keep)
    table_f(d, ns, keep)
    table_h(d)
    from _paper_guidance import build_guidance
    draft, guide = build_guidance(d, causal_nostore)
    write("段落草稿.md", draft)
    write("README.md", guide)
    p("")
    p(f"关键补充数字：(因果, 无储能) = {f6(causal_nostore)} 元")
    p(f"                 点预测与按实采购费用差  = {f6(causal_nostore - ns)} 元")
    p(f"                 两种因果方案费用差    = {f6(causal_nostore - keep)} 元")
    p("")
    p(f"[30 完成] 产出目录：{OUT}")
    C.write_text_utf8(C.SOLVE_LOG_DIR / "第二问_30补充材料日志.txt",
                      "\n".join(LOG + ["", "[30 完成] 论文补充材料生成结束。"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
