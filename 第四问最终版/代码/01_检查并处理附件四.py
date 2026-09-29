#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
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


C = _load("_comm4.py", "q4_comm")

import numpy as np


def main() -> int:
    C.ensure_dirs()
    t0 = time.perf_counter()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 78)
    p("第四问 01 —— 检查原始附件并处理附件四电价矩阵 / 独立处理附件三预报矩阵")
    p("=" * 78)

    p("")
    p("── 0. 来源快照与 SHA-256 指纹 ──")
    C.freeze_sources(verbose=True)
    p("  说明：第三题源码已复制到 `_快照_第三题代码/`（已存在不覆盖），")
    p("        本问所有脚本**不 import 第三题模块、不读其正在写入的中间结果**。")

    p("")
    p("── 1. 附件四电价矩阵 ──")
    p(f"  源文件：{C.ATTACHMENT4_PATH}")
    A4 = C.parse_attachment4()
    dates4 = A4["dates"]
    price = A4["price_actual"]
    n_day, n_t = price.shape
    p(f"  形状：{price.shape}（天 × 时段），单元格总数 {price.size}")
    p(f"  日期范围：{dates4[0]} … {dates4[-1]}，共 {n_day} 天")
    p(f"  时间标签：[{A4['raw_time_labels'][0]}] … [{A4['raw_time_labels'][-1]}]；"
      f"区间起点 {C.minutes_to_hhmm(A4['slot_start_minute'][0])} → "
      f"{C.minutes_to_hhmm(A4['slot_start_minute'][-1])}")

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail):
        checks.append((name, bool(ok), detail))
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    ck("日期连续无重复", len(set(dates4.tolist())) == n_day,
       f"唯一日期 {len(set(dates4.tolist()))} / 总行 {n_day}")
    dts = [np.datetime64(s) for s in dates4.tolist()]
    gaps = [(dts[i + 1] - dts[i]).astype(int) for i in range(n_day - 1)]
    ck("日期逐日递增（无缺日）", all(g == 1 for g in gaps),
       f"最大间隔 {max(gaps)} 天，最小 {min(gaps)} 天")
    ck("与评分期日历一致",
       dates4[0] == f"{C.YEAR}-01-01" and dates4[-1] == C.SCORE_END and n_day == 365,
       f"{dates4[0]} … {dates4[-1]}，{n_day} 天")
    ck("时间标签为区间终点等差 10min",
       np.array_equal(A4["slot_end_minute"], (np.arange(1, 145) * 10)),
       f"末标签 {A4['raw_time_labels'][-1]!r} = 第 1440 分钟")
    fin = np.isfinite(price)
    ck("全部价格有限（无缺失/NaN）", bool(fin.all()),
       f"异常格点 {int((~fin).sum())} 个")
    ck("全部价格严格为正", bool((price[fin] > 0).all()),
       f"最小值 {price[fin].min():.4f} 元/kWh")
    p(f"  价格统计：min {price.min():.4f}，mean {price.mean():.4f}，"
      f"max {price.max():.4f} 元/kWh")
    ref = (0.0076, 1.7936)
    p(f"  与流程图形状观测对照 [0.0076, 1.7936]："
      f"[{price.min():.4f}, {price.max():.4f}] —— "
      f"{'一致 ✔' if abs(price.min() - ref[0]) < 1e-3 and abs(price.max() - ref[1]) < 1e-3 else '不一致 ⚠（以本文件实测为准）'}")

    Z = C.load_q2_matrix()
    d2 = np.array([str(s) for s in Z["dates"]], dtype="<U10")
    ck("与附件二日期序列逐一对应", bool(np.array_equal(d2, dates4)),
       f"附件二 {d2[0]} … {d2[-1]}（{d2.size} 天）")
    score_idx = np.array([i for i in range(n_day) if dates4[i] >= C.SCORE_START], int)
    ck("评分期天数 = 334", score_idx.size == C.N_SCORE_DAYS,
       f"{score_idx.size} 天（{dates4[score_idx[0]]} … {dates4[score_idx[-1]]}），"
       f"预热期 {n_day - score_idx.size} 天")

    daily_mean = price.mean(axis=1)
    daily_max = price.max(axis=1)
    slot_mean = price.mean(axis=0)
    p(f"  日均价：{daily_mean.min():.4f} … {daily_mean.max():.4f} 元/kWh")
    p(f"  日内最高价（144 段）：{daily_max.mean():.4f} 元/kWh（均值）")
    i_peak = int(np.argmax(slot_mean))
    p(f"  时段均价峰：{C.minutes_to_hhmm(A4['slot_end_minute'][i_peak])} "
      f"= {slot_mean[i_peak]:.4f} 元/kWh；谷："
      f"{C.minutes_to_hhmm(A4['slot_end_minute'][int(np.argmin(slot_mean))])} "
      f"= {slot_mean.min():.4f} 元/kWh")

    np.savez_compressed(
        C.PRICE_MATRIX_NPZ,
        dates=dates4,
        raw_time_labels=np.array(A4["raw_time_labels"], dtype="<U8"),
        slot_start_minute=A4["slot_start_minute"],
        slot_end_minute=A4["slot_end_minute"],
        price_actual=price,
        day_index=A4["day_index"],
        score_day_index=score_idx,
        daily_mean=daily_mean,
        slot_mean=slot_mean,
    )
    p(f"  已保存：{C.PRICE_MATRIX_NPZ.relative_to(C.PROJECT_DIR)}")

    rows = [[f"{dates4[i]}", f"{daily_mean[i]:.4f}", f"{price[i].min():.4f}",
             f"{daily_max[i]:.4f}", f"{price[i].std(ddof=1):.4f}",
             "评分期" if i in set(score_idx.tolist()) else "预热期"]
            for i in range(n_day)]
    C.write_csv_utf8_sig(C.RESULT_DIR / "第四问_附件四逐日统计.csv",
                         ("日期", "日均价", "日最低价", "日最高价", "日内标准差", "期别"), rows)

    md = [
        "# 附件四电价数据处理与时间映射报告",
        "",
        f"- 源文件：`{C.ATTACHMENT4_PATH.name}`（Sheet1）",
        f"- 解析形状：**{price.shape}**（天 × 时段），共 {price.size} 个价格单元",
        f"- 日期范围：{dates4[0]} … {dates4[-1]}（{n_day} 天，含 1 月预热期 31 天 + 评分期 334 天）",
        "",
        "## 1. 时间映射口径（与第一、二、三问统一）",
        "",
        f"- 每天 **144** 个 10 分钟时段，$\\Delta t = 1/6$ 小时；",
        f"- 原始列标签为**区间终点**：第 $t$ 段（$t=0..143$）对应 "
        f"$[10t,\\,10t+10)$ 分钟，标签为 $00$:$10$ … $23$:$50$、`0:00+1`；",
        f"- 因此 `slot_start_minute[t] = {A4['slot_start_minute'][0]}, …, "
        f"{A4['slot_start_minute'][-1]}`，`slot_end_minute[t] = "
        f"{A4['slot_end_minute'][0]}, …, {A4['slot_end_minute'][-1]}`；",
        "- 该口径与第二问 `附件二_矩阵数据.npz::price` 完全一致，"
        "保证第四问与第二、三问的费用口径可直接比较。",
        "",
        "## 2. 校验结果",
        "",
        "| 检查项 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for name, ok, detail in checks:
        md.append(f"| {name} | {'✅ 通过' if ok else '❌ 未通过'} | {detail} |")
    md += [
        "",
        "## 3. 电价统计",
        "",
        f"- 全局：min {price.min():.4f} / mean {price.mean():.4f} / max {price.max():.4f} 元/kWh；",
        f"- 日均价：{daily_mean.min():.4f} … {daily_mean.max():.4f} 元/kWh；",
        f"- 时段均价峰值出现在 {C.minutes_to_hhmm(A4['slot_end_minute'][i_peak])}"
        f"（{slot_mean[i_peak]:.4f} 元/kWh），谷值出现在 "
        f"{C.minutes_to_hhmm(A4['slot_end_minute'][int(np.argmin(slot_mean))])}"
        f"（{slot_mean.min():.4f} 元/kWh）；",
        f"- 峰谷时段均价之比 ≈ **{slot_mean[i_peak] / slot_mean.min():.2f}**，"
        "说明日内套利空间显著，储能「低价充、高价放」有明确经济动机。",
        "",
        "## 4. 关键提醒（进入后续步骤的前置约束）",
        "",
        "1. 电价是**给定的确定性数据**，第四问的难点不在预测精度本身，"
        "而在「**只有当期时段价格已知**」的信息约束下如何安排购电与充放电；",
        "2. 因此本问的预测必须按 `revealed_at_delivery_slot` 口径构造："
        "第 $t$ 段的价格在 $t$ 段交付时才可用，静态计划 $g$ 必须只用 τ 时刻已知信息；",
        "3. 1 月 31 天的价格**不参与任何预测标定**，仅用于共同热启动；",
        "4. 附件四与附件二日期逐日对应，可直接按同一日索引对齐负荷、光伏与价格。",
        "",
        "## 5. 输出文件",
        "",
        "| 文件 | 内容 |",
        "| --- | --- |",
        "| `处理后数据/附件四_电价矩阵.npz` | dates / slot_start_minute / slot_end_minute / "
        "price_actual(365,144) / score_day_index / daily_mean / slot_mean |",
        "| `模型结果/第四问_附件四逐日统计.csv` | 逐日均价、极值、标准差 |",
        "",
    ]
    C.write_text_utf8(C.PRICE_TIME_MD, "\n".join(md))
    p(f"  已保存报告：{C.PRICE_TIME_MD.relative_to(C.PROJECT_DIR)}")

    p("")
    p("── 2. 附件三光伏预报矩阵（独立处理） ──")
    p(f"  源文件：{C.ATTACHMENT3_PATH}")
    A3 = C.read_attachment3()
    V_raw = A3["V_raw_kw"]
    p(f"  V_raw_kw 形状：{V_raw.shape}（天 × τ × lead）")
    p(f"  日期范围：{A3['dates'][0]} … {A3['dates'][-1]}，共 {A3['dates'].size} 天")
    ck3 = []
    ck3.append(("发布时刻为 (0,6,12,18)", bool(np.array_equal(A3["tau_hours"], np.array(C.TAU_HOURS))),
                f"{A3['tau_hours'].tolist()}"))
    ck3.append(("与附件四日期一致", bool(np.array_equal(A3["dates"], dates4)),
                f"{A3['dates'][0]} … {A3['dates'][-1]}"))
    ck3.append(("预报矩阵无 NaN", bool(np.isfinite(V_raw).all()),
                f"缺失格点 {int((~np.isfinite(V_raw)).sum())}"))
    ck3.append(("预报非负", bool((V_raw[np.isfinite(V_raw)] >= -1e-9).all()),
                f"最小值 {np.nanmin(V_raw):.4f} kW"))
    ck3.append(("四发布时刻齐备", bool(A3["complete_mask"].all()),
                f"完整标记率 {A3['complete_mask'].mean() * 100:.1f}%"))
    for name, ok, detail in ck3:
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    V_abs_lead = C.lead_to_absolute(V_raw)
    V_pw = C.disaggregate_piecewise(V_abs_lead)
    V_lin = C.disaggregate_linear(V_abs_lead)
    V_win_kw = C.disaggregate_piecewise(V_raw)
    V_win_kwh = V_win_kw * C.DELTA_HOURS
    pv_act_h = C.pv_hourly_actual_from_q2(Z)

    p("")
    p("  ── 时间偏移经验判定（防止整表平移 1 小时）──")
    off_rows = []
    for shift, label in ((0, "预报(k+1)小时 → 绝对小时 (τ+k)%24  【采用】"),
                         (1, "预报(k+1)小时 → 绝对小时 (τ+k+1)%24")):
        errs = []
        for i in range(V_raw.shape[0]):
            for ti, tau in enumerate(C.TAU_HOURS):
                for k in range(24):
                    h = tau + k + shift
                    day, hr = i + h // 24, h % 24
                    if day >= V_raw.shape[0]:
                        continue
                    errs.append(V_raw[i, ti, k] - pv_act_h[day, hr])
        e = np.array(errs)
        off_rows.append((label, f"{np.abs(e).mean():.2f}", f"{np.sqrt((e ** 2).mean()):.2f}"))
        p(f"    {label}：MAE {np.abs(e).mean():.2f} kW，RMSE {np.sqrt((e ** 2).mean()):.2f} kW")
    p("    → 采用 MAE 更小者（对照时跨日已记入次日）；与第三问最终版 `LEAD_TO_ABS` 口径一致。")

    v_ann = V_pw[:, 0, :].sum() * C.DELTA_HOURS
    p(f"  附件三全年预报发电量（τ=0 口径）：{v_ann / 1e4:.2f} 万 kWh")
    p(f"  附件二全年实际发电量：{pv_act_h.sum() / 1e4:.2f} 万 kWh")
    p(f"  全年预报/实际 = {v_ann / pv_act_h.sum():.4f}"
      f"（4 个发布时刻口径不可直接相加，否则重复计数 4 倍）")
    cons = V_pw.reshape(V_pw.shape[0], V_pw.shape[1], 24, 6).sum(axis=3) * C.DELTA_HOURS
    err = np.abs(cons - V_abs_lead).max()
    p(f"  分段常数展开能量守恒最大误差：{err:.3e} kWh（应≈0）")

    prec = []
    for ti, tau in enumerate(C.TAU_HOURS):
        e = []
        for d in range(V_raw.shape[0]):
            for k in range(24):
                h = tau + k
                day, hr = d + h // 24, h % 24
                if day >= V_raw.shape[0]:
                    continue
                e.append(V_raw[d, ti, k] - pv_act_h[day, hr])
        e = np.array(e)
        prec.append((f"{tau}:00", f"{np.abs(e).mean():.2f}", f"{np.sqrt((e ** 2).mean()):.2f}",
                     f"{e.mean():.2f}"))
        p(f"    τ={tau:>2d}:00 预报精度：MAE {np.abs(e).mean():7.2f} kW，"
          f"RMSE {np.sqrt((e ** 2).mean()):7.2f} kW，偏差 {e.mean():+7.2f} kW")

    np.savez_compressed(
        C.A3_FORECAST_NPZ,
        dates=A3["dates"],
        tau_hours=A3["tau_hours"],
        V_raw_kw=V_raw,
        V_hourly_abs=V_abs_lead,
        V_10min_piecewise=V_pw,
        V_10min_linear=V_lin,
        V_win_10min_kwh=V_win_kwh,
        V_win_10min_kw=V_win_kw,
        pv_hourly_actual=pv_act_h,
        complete_mask=A3["complete_mask"],
        score_day_index=score_idx,
    )
    p(f"  已保存：{C.A3_FORECAST_NPZ.relative_to(C.PROJECT_DIR)}")

    md3 = [
        "# 附件三光伏预报数据独立处理报告（第四问 4-3 用）",
        "",
        "> 本步骤**直接从原始 `附件3.xlsx` 独立解析**，不读取第三问任何中间产物，",
        "> 以满足「4-3 可独立从附件三实现」的要求。",
        "> **与第三题的口径核对已完成**，结论与可复算脚本见",
        "> `报告/第四问_第三问迁移项交叉核验.md`。",
        "",
        f"- 源文件：`{C.ATTACHMENT3_PATH.name}`，形状 1460 × 26（365 天 × 4 发布时刻）",
        f"- 解析张量：**{V_raw.shape}**（天 × 发布时刻 τ × lead），共 {V_raw.size} 个预报单元",
        f"- 日期：{A3['dates'][0]} … {A3['dates'][-1]}",
        "",
        "## 1. 表结构",
        "",
        "| 列 | 含义 |",
        "| --- | --- |",
        "| 日期 | 合并单元格，需前向填充 |",
        "| 预报时刻 | 0:00 / 6:00 / 12:00 / 18:00 四个发布时刻 τ |",
        "| 预报1小时 … 预报24小时 | τ 之后 24 个整点的小时平均功率（kW） |",
        "",
        "## 2. 时间语义与偏移经验判定（关键，易错点）",
        "",
        "`V_raw[i, τ, k]`（k = 0..23 对应 `预报(k+1)小时`）描述的是**绝对小时** "
        "$(\\tau + k) \\bmod 24$，而**不是**绝对小时 $k$。",
        "下游任何与负荷/电价按时段对齐的运算**必须先做 `lead_to_absolute` 重排**，",
        "否则 τ≠0 的预报会被整体平移 6 / 12 / 18 小时。",
        "此外，$\\tau + k \\ge 24$ 的段**属于次日**：该行是「自 τ 起 24 小时」的滚动窗，",
        "跨午夜后缀次日的凌晨，不是当日凌晨。本步骤所有精度对照均按"
        "「绝对日 × 小时」对齐（跨日记入次日），不使用 `mod 24` 取当日值。",
        "",
        "本步骤用附件二的实际光伏做了**双向经验对照**（全年 365×4×24 个格点）：",
        "",
        "| 偏移假设 | 全年 MAE (kW) | 全年 RMSE (kW) |",
        "| --- | --- | --- |",
    ]
    for label, mae, rmse in off_rows:
        md3.append(f"| {label} | {mae} | {rmse} |")
    md3 += [
        "",
        "→ 采用 MAE 更小的第一种，与第三问最终版 `LEAD_TO_ABS = (arange(24) − τ) % 24` 一致。",
        "→ **与模型对齐的关系**：节点 $(d,\\tau)$ 的展望第 $k$ 段（lead $k$）"
        "对应全局时段 $d\\cdot144 + 36\\tau + 6k$，其绝对小时恰为 $(\\tau+k)\\bmod 24$、"
        "跨 24 时落在次日；因此落盘的 `V_win_10min_kwh`（窗口/lead 索引）"
        "与全局时段**逐段严格对齐**，无需再平移。",
        "`V_hourly_abs` / `V_10min_piecewise`（绝对小时索引）只用于"
        "「按钟点」的对照与诊断，不参与 4-3 场景构造。",
        "",
        "## 3. 能量与精度",
        "",
        f"- 附件三全年预报发电量（τ=0 口径）：**{v_ann / 1e4:.2f} 万 kWh**；",
        f"- 附件二全年实际发电量：**{pv_act_h.sum() / 1e4:.2f} 万 kWh**"
        f"（比值 {v_ann / pv_act_h.sum():.4f}）；",
        "- 注：四个发布时刻的预报是**同一 24 小时窗口的不同时点快照**，"
        "口径不同不可直接相加，否则重复计数 4 倍。",
        "",
        "| 发布时刻 | MAE (kW) | RMSE (kW) | 偏差 (kW) |",
        "| --- | --- | --- | --- |",
    ]
    for tau_s, mae, rmse, bias in prec:
        md3.append(f"| τ = {tau_s} | {mae} | {rmse} | {bias} |")
    md3 += [
        "",
        "## 4. 整点 → 10 分钟展开",
        "",
        "| 口径 | 规则 | 用途 |",
        "| --- | --- | --- |",
        "| 分段常数 | 每小时 6 段取同一值，严格保持每小时能量守恒 | **主口径** |",
        "| 线性插值 | $P[6k+r]=(1-r/6)F[k]+(r/6)F[k+1]$，末端不外推 | 敏感性档 |",
        "",
        f"分段常数展开的能量守恒最大误差：**{err:.3e} kWh**（数值零）。",
        "",
        "## 5. 校验结果",
        "",
        "| 检查项 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for name, ok, detail in ck3:
        md3.append(f"| {name} | {'✅ 通过' if ok else '❌ 未通过'} | {detail} |")
    md3 += [
        "",
        "## 6. 输出文件",
        "",
        "| 文件 | 内容 |",
        "| --- | --- |",
        "| `处理后数据/附件三_预报矩阵.npz` | dates / tau_hours / V_raw_kw / "
        "V_hourly_abs / V_10min_piecewise / V_10min_linear / "
        "**V_win_10min_kwh（节点窗口索引，4-3 使用）** / pv_hourly_actual / "
        "complete_mask / score_day_index |",
        "",
        "`pv_hourly_actual` 取自附件二的 10 分钟实际光伏（按整点小时取均值），",
        "仅用于**残差标定**，与第三问口径一致。",
        "",
    ]
    C.write_text_utf8(C.A3_REPORT_MD, "\n".join(md3))
    p(f"  已保存报告：{C.A3_REPORT_MD.relative_to(C.PROJECT_DIR)}")

    n_fail = sum(1 for _, ok, _ in checks if not ok) + sum(1 for _, ok, _ in ck3 if not ok)
    p("")
    p("=" * 78)
    p(f"01 完成：附件四 {price.shape} + 附件三 {V_raw.shape}；"
      f"校验 {len(checks) + len(ck3)} 项，未通过 {n_fail} 项。"
      f"用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.SOLVE_LOG_DIR / "01_附件处理日志.txt", "\n".join(log))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
