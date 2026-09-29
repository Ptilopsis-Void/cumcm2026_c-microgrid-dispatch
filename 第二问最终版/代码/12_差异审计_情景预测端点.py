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
DP = _load("06_DP价值执行器.py", "q2_dp")

import numpy as np
import pandas as pd

T = C.PERIODS_PER_DAY
M = C.M_SCENARIOS
ETA = C.ETA
S = C.S_PERIOD_KWH
WD_NAME = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def audit_scenario_window(date_strs, score_idx):
    rows = []
    for d in score_idx:
        pool = list(range(max(1, d - C.RESIDUAL_WINDOW_DAYS), d))
        if not pool:
            pool = [0]
        for w in range(M):
            i = pool[w % len(pool)]
            rows.append((date_strs[d], w + 1, date_strs[i], i, len(pool)))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_逐日情景日期索引.csv",
        ("当日日期", "情景序号(1..30)", "使用的历史日期", "历史日期索引", "可用池长度"),
        rows)
    return rows


def scenario_window_report(date_strs):
    d31 = 31
    pool = list(range(max(1, d31 - C.RESIDUAL_WINDOW_DAYS), d31))
    md = [
        "# 第二问 情景窗口边界审计报告",
        "",
        "## 1. 2025-02-01 实际进入情景库的 30 个历史日期",
        "",
        f"评分期首日 2025-02-01（索引 {d31}）的情景 pool = `range(max(1, {d31}-30), {d31})` = `range(1, 31)`。",
        "",
    ]
    tbl = ["| 序号 | 历史日期 | 索引 |", "| --- | --- | --- |"]
    for w in range(M):
        i = pool[w % len(pool)]
        tbl.append(f"| {w + 1} | {date_strs[i]} | {i} |")
    md += tbl
    md += [
        "",
        "## 2. off-by-one 结论",
        "",
        "1. **2025-01-01（索引 0）被排除**：它是全年首个日期，无任何前视历史，其负荷预测回退到"
        "「取当日」、残差恒为 0，不构成有效残差，故不作为历史情景来源是合理的。",
        "2. **评分期首日 2025-02-01 的窗口 = 2025-01-02…2025-01-31 共 30 天**，"
        "恰好等于「此前 30 天」，无缺少、无多算、无重复。",
        "3. 对于所有评分日 d≥31，`max(1, d-30) = d-30`（因 d-30≥1），窗口恒为 `[d-30, d)` 共 30 天，"
        "不存在因 `max(1,·)` 导致的评分期窗口缩短。",
        "4. 最早「可定义无前视预测」的历史日为 2025-01-02（索引 1，用 2025-01-01 单日做预测），"
        "2025-01-01 是唯一被永久排除的日期，且其对评分期首日窗口外（不在此前 30 天）。",
        "",
        "## 3. 结论",
        "",
        "**情景窗口不存在 off-by-one**：`range(max(1, d-30), d)` 对所有评分日给出恰好 30 个、"
        "全部严格早于当日的互异历史日；排除索引 0 符合「无前视预测」原则，不影响任何评分日。",
        "",
    ]
    C.write_text_utf8(C.REPORT_DIR / "第二问_情景窗口边界审计.md", "\n".join(md))


def causal_forecast_variant(load_e, wd, lookback, fallback, min_same=2):
    n_day = load_e.shape[0]
    Lhat = np.zeros_like(load_e)
    src = np.empty(n_day, dtype=object)
    for d in range(n_day):
        cand = [i for i in range(max(0, d - lookback), d) if wd[i] == wd[d]]
        if len(cand) >= min_same:
            Lhat[d] = load_e[cand].mean(0)
            src[d] = f"同星期×{len(cand)}"
        else:
            lo = max(0, d - fallback)
            if d - lo > 0:
                Lhat[d] = load_e[lo:d].mean(0)
                src[d] = f"回退前{d - lo}天"
            else:
                Lhat[d] = load_e[0]
                src[d] = "无历史(取当日)"
    return Lhat, src


def audit_forecast_fallback(load_e, pv_e, wd, dates):
    Lhat_cur, src_cur = causal_forecast_variant(load_e, wd, C.LOAD_LOOKBACK_DAYS,
                                                C.LOAD_FALLBACK_DAYS, min_same=1)
    Lhat_ref, src_ref = causal_forecast_variant(load_e, wd, C.LOAD_LOOKBACK_DAYS,
                                                C.LOAD_FALLBACK_DAYS, min_same=2)
    end = 46
    rows = []
    for d in range(end):
        cand = [i for i in range(max(0, d - C.LOAD_LOOKBACK_DAYS), d) if wd[i] == wd[d]]
        cand_txt = "、".join(dates[i].isoformat() for i in cand)
        trigger_ref = len(cand) < 2
        diff = not np.allclose(Lhat_cur[d], Lhat_ref[d], atol=1e-9)
        rows.append((dates[d].isoformat(), WD_NAME[wd[d]], cand_txt, len(cand),
                     "是" if trigger_ref else "否",
                     "同星期" if len(cand) >= 2 else ("回退" if d > 0 else "取当日"),
                     "是" if any(i == d for i in cand) else "否",
                     "是" if diff else "否"))
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_早期预测来源审计.csv",
        ("日期", "星期", "同星期候选日期", "候选数量", "图片口径是否触发回退(<2)",
         "图片口径实际采用", "预测是否含当前日", "两口径预测是否不同"),
        rows)

    diff_days = [d for d in range(load_e.shape[0])
                 if not np.allclose(Lhat_cur[d], Lhat_ref[d], atol=1e-9)]
    _all_warm = all(d < 14 for d in diff_days)
    if _all_warm:
        _warm_txt = ("**全部落在主档预热段 d < 14 内**，因此该回退规则在主档下仍然生效，"
                     "只作用于预热段；")
    else:
        _n_out = sum(1 for d in diff_days if d >= 14)
        _warm_txt = (f"其中 {len(diff_days) - _n_out} 天落在主档预热段 d < 14 内（该回退规则仍生效），"
                     f"另有 {_n_out} 天在 d ≥ 14，主档已改用分解预测、不再走该回退分支；")
    md = [
        "# 第二问 早期预测来源审计（负荷回退 min_same 口径）",
        "",
        "> **口径提示（必读）**：本报告审计的是**同星期均值预测器**在「同星期历史样本不足」时的回退规则，"
        "它适用于：(i) 朴素档 `Q2_PREDICTOR=naive`（全程使用该规则）；"
        "(ii) **主档（`decomp`）的预热段 d < 14（2025-01-01 ~ 01-14）**。"
        "主档自 2025-01-15 起改用分解预测（式 17a~17c：日电量水平 × 日内形状，"
        "形状取最近 3 个同类型历史日），**不存在该回退分支**。",
        ">",
        f"> 当前 `代码/04_构建情景库.py::_naive_load` 的口径为 `if len(cand) >= 2:`，"
        f"即 **同星期样本 <2 才回退到此前最多 {C.LOAD_FALLBACK_DAYS} 天**，与图片文字一致；"
        f"历史记录里的 `if cand:`（min_same=1）**已不再是当前代码**。",
        "",
        f"同星期回望窗 `LOAD_LOOKBACK_DAYS = {C.LOAD_LOOKBACK_DAYS}` 天，"
        f"回退窗 `LOAD_FALLBACK_DAYS = {C.LOAD_FALLBACK_DAYS}` 天。",
        "",
        "图片口径：同星期样本**不足两天**（<2）时回退到此前最多 7 天；",
        "历史代码（已修）：只要**存在 ≥1 个**同星期样本就不回退。",
        "",
        f"两口径下负荷预测不同的日期共 {len(diff_days)} 个：",
        f"`{[dates[d].isoformat() for d in diff_days]}`",
        "",
        f"这 {len(diff_days)} 天（2025-01-01 ~ 01-14 内）{_warm_txt}其负荷残差进入 "
        "2025-02-01 ~ 2025-02-13 的情景库（pool 含索引 7…13），"
        "其余评分日的情景 pool 不包含这些早期日期，因此该口径差异**只影响 2 月上旬约 13 个评分日**。",
        "",
        "## 判定",
        "",
        "图片文字明确「同星期样本不足两天时回退」，故正确口径是 **<2 回退**（即 `min_same = 2`）。"
        "当前代码已与此一致（`if len(cand) >= 2:`）。此差异不改变评分期 MAE"
        "（评分日同星期样本 ≥5，不会触发回退），但会小幅改变 2 月上旬情景与日前计划。",
        "",
        "| 项 | 权威出处 |",
        "| --- | --- |",
        "| 根因与修复 | `报告/第二问_最终差异审计报告.md`（§2 差异根因：`if cand:` → `if len(cand) >= 2:`） |",
        "| 逐日差异明细 | `模型结果/第二问_预测回退规则影响对照.csv` |",
        "| 主档当前预测器 | `报告/第二问_最终模型口径说明.md` §3.1 |",
        "",
    ]
    C.write_text_utf8(C.REPORT_DIR / "第二问_早期预测来源审计.md", "\n".join(md) + "\n")
    return diff_days


def audit_endpoint_rule(scen_N_day, g_day, price, tag, grid_delta=6.0):
    vf = DP.build_value_functions(scen_N_day, g_day, price, delta=grid_delta)
    Hbar = vf["Hbar"]
    grid = vf["grid"]
    R = vf["R"]
    rows = []
    for t in range(T):
        cc = 5.0 * price[t]
        Gt = cc * ETA * grid + Hbar[t + 1]
        minv = float(Gt.min())
        cand = {tol: np.where(Gt <= minv + tol)[0] for tol in (1e-8, 1e-7, 1e-6, 1e-5)}
        min_ep = float(grid[cand[1e-8].min()])
        max_ep = float(grid[cand[1e-8].max()])
        chosen = float(R[t])
        width = max_ep - min_ep
        mis = 0.0 if abs(chosen - max_ep) < 1e-9 else 1.0
        rows.append((tag, t + 1, f"{minv:.8f}", f"{min_ep:.3f}", f"{max_ep:.3f}",
                     f"{chosen:.3f}", f"{width:.3f}",
                     ";".join(f"{to}:{len(cand[to])}" for to in (1e-8, 1e-7, 1e-6, 1e-5)),
                     str(int(mis))))
    return rows


def main() -> int:
    C.ensure_dirs()
    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    BT = np.load(C.RESULT_DIR / "第二问_执行器回测.npz", allow_pickle=False)
    load_e = np.asarray(Z["load_energy_kwh"], float)
    pv_e = np.asarray(Z["pv_energy_kwh"], float)
    price = np.asarray(Z["price"], float)
    date_strs = [str(s) for s in Z["dates"]]
    dates = [pd.Timestamp(s).date() for s in date_strs]
    wd = np.array([d.weekday() for d in dates])
    score_idx = np.asarray(Z["score_day_index"], int)
    scen_N_all = np.asarray(Zd["scen_L"], float) - np.asarray(Zd["scen_V"], float)
    plan_g = np.asarray(BT["plan_g"], float)
    date_of = {s: i for i, s in enumerate(date_strs)}

    print("=" * 74)
    print("第二问 12 —— 最终差异审计（只读）：情景窗口 / 预测回退 / 端点规则")
    print("=" * 74)

    audit_scenario_window(date_strs, score_idx)
    scenario_window_report(date_strs)
    print("[1] 情景窗口：已产出逐日 30 情景日期索引 + 边界报告")

    diff_days = audit_forecast_fallback(load_e, pv_e, wd, dates)
    print(f"[2] 预测回退：两口径不同的日期 {len(diff_days)} 个："
          f"{[dates[d].isoformat() for d in diff_days]}")
    print("    → 报告/第二问_早期预测来源审计.md（与「情景窗口边界审计」已拆分）")

    ep_rows = []
    for day in ("2025-03-20", "2025-09-23"):
        d = date_of[day]
        ep_rows += audit_endpoint_rule(scen_N_all[d], plan_g[d], price, day)
    d = date_of["2025-03-20"]
    b_day = BT["dp_b"][d]
    emerg_t = np.where(b_day > 1e-6)[0]
    print(f"[3] 03-20 紧急购电时段：{emerg_t}")
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_DP平坦区间端点审计.csv",
        ("日期", "时段序号", "最小目标值_元", "最小端点_kWh", "最大端点_kWh",
         "实际选择动作_kWh", "平坦区间宽度_kWh", "各容差候选点数(1e-8/1e-7/1e-6/1e-5)",
         "是否端点误选"),
        ep_rows)
    flat_cnt = sum(1 for r in ep_rows if float(r[6]) > 1e-6)
    mis_cnt = sum(int(r[8]) for r in ep_rows)
    md = [
        "# 第二问 DP 端点规则审计报告（平坦最优区间取最大端点）",
        "",
        "## 1. 检查对象",
        "",
        "对 2025-03-20、2025-09-23 两日逐时段检查动态保留水平 `R_t = max argmin_{e}(5c_t·η·e + H̄_{t+1}(e))`：",
        "记录最小目标值、最小/最大平坦端点、实际选择动作、平坦区间宽度、不同容差下的候选点数、是否端点误选。",
        "",
        "## 2. 结果摘要",
        "",
        f"- 平坦区间宽度 > 1e-6 kWh 的时段数：**{flat_cnt}** / {len(ep_rows)}；",
        f"- 端点误选（实际选择 ≠ 最大端点）时段数：**{mis_cnt}**；",
        f"- 说明：`max_argmin` 用 `argmin(a[::-1])` 取最大 argmin，对凸分段线性函数在平坦区间正确取右端点。",
        "",
        "## 3. 容差影响",
        "",
        "在 1e-8 / 1e-7 / 1e-6 / 1e-5 元四档容差下统计 argmin 候选点数，评估数值噪声是否会把单一最优"
        "误判为平坦区间。正式结果采用最小合理容差 1e-8 元（不会错误扩张平坦区间）。",
        "",
    ]
    C.write_text_utf8(C.REPORT_DIR / "第二问_DP端点规则审计.md", "\n".join(md))
    print(f"[3] 端点规则：平坦区间时段 {flat_cnt}，端点误选 {mis_cnt}")

    print("=" * 74)
    print("审计完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
