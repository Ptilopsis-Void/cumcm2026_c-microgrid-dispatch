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


C = _load("_comm3.py", "q3_comm")

import numpy as np


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t0 = time.perf_counter()

    log("=" * 78)
    log("第三问 02 —— 处理附件 3 基础数据")
    log("=" * 78)

    A3 = C.read_attachment3()
    dates = A3["dates"]
    V_raw = np.asarray(A3["V_raw_kw"], float)
    seen = np.asarray(A3["complete_mask"], bool)
    log(f"附件 3：{A3['source'].name} → 张量 {V_raw.shape}（天数 × τ 数 × 整点数）")
    log(f"日期：{dates[0]} … {dates[-1]}，共 {dates.size} 天")

    log("")
    log("── 完整性校验 ──")
    n_missing = int((~seen).sum())
    n_nan = int(np.sum(~np.isfinite(V_raw)))
    log(f"缺失槽位（(日期,τ) 组合未出现）：{n_missing} / {seen.size}")
    log(f"NaN/Inf 个数：{n_nan}")
    assert n_missing == 0, "附件 3 存在缺失的 (日期, 预报时刻) 组合"
    assert n_nan == 0, "附件 3 张量存在非有限值"

    Z = C.Q2.matrix()
    q2_dates = np.array([str(x) for x in Z["dates"]], dtype="<U10")
    same_dates = (q2_dates.size == dates.size) and bool(np.all(q2_dates == dates))
    log(f"与附件二矩阵日期逐日对齐（{q2_dates.size} 天）："
        f"{'✔ 通过' if same_dates else '✘ 不一致'}")
    assert same_dates, "附件 3 与附件 2 的日期序列不一致"

    month_first = [d for d in dates.tolist() if d.endswith("-01")]
    log(f"每月 1 日出现 {len(month_first)} 次（预期 12）："
        f"{'✔' if len(month_first) == 12 else '✘'}")

    log("")
    log("── 10 min 展开口径 ──")
    V_pw = C.disaggregate(V_raw, C.DISAGG_PIECEWISE)
    V_li = C.disaggregate(V_raw, C.DISAGG_LINEAR)
    log(f"piecewise_constant → {V_pw.shape}；linear_interp → {V_li.shape}")

    w_cons_pw = float(np.max(np.abs(V_pw.reshape(V_pw.shape[:-1] + (24, 6)).mean(-1)
                                    - V_raw)))
    w_cons_li = float(np.max(np.abs(V_li.reshape(V_li.shape[:-1] + (24, 6)).mean(-1)
                                    - V_raw)))
    log(f"分段常数：每小时 6 段均值 − 原始整点值 的最大偏差 = {w_cons_pw:.3e} "
        f"{'✔' if w_cons_pw < 1e-9 else '✘'}")
    log(f"线性插值：每小时 6 段均值 − 原始整点值 的最大偏差 = {w_cons_li:.3e} "
        f"（线性插值把能量在小时之间搬移，均值不等是预期行为）")
    log(f"线性插值展开非负性检查：min = {V_li.min():.4f} kW "
        f"{'✔ 非负' if V_li.min() >= 0 else '✘ 出现负值'}")
    log(f"线性插值全日能量偏差（相对）："
        f"{abs(V_li.sum(axis=-1).sum() - V_raw.sum() / 6 * 6) / max(V_raw.sum(), 1e-9):.3e}")

    log("")
    log("── 由附件 2 回算「小时平均实际出力」（供残差库使用）──")
    pv_kw = np.asarray(Z["pv_kw"], float)
    pv_hourly_actual = pv_kw.reshape(pv_kw.shape[0], 24, 6).mean(-1)
    log(f"pv_hourly_actual：{pv_hourly_actual.shape}，"
        f"均值 {pv_hourly_actual.mean():.2f} kW，最大 {pv_hourly_actual.max():.2f} kW")
    log("说明：附件 2 的 10 min 实际光伏 → 按整点小时取均值，与附件 3 的整点口径对齐。")

    pv_energy = float(pv_kw.sum() * C.DELTA_HOURS)
    log(f"附件 2 实际光伏全年发电量（10 min 口径）：{pv_energy / 1e4:.2f} 万 kWh")

    wd = np.array([int(d[5:7]) for d in dates.tolist()])
    warmup_idx = np.where(wd == 1)[0]
    score_idx = np.where(wd != 1)[0]
    log("")
    log("── 评分期索引 ──")
    log(f"预热期（1 月）{warmup_idx.size} 天（{dates[warmup_idx[0]]} … "
        f"{dates[warmup_idx[-1]]}），预期 31 → {'✔' if warmup_idx.size == 31 else '✘'}")
    log(f"评分期 {score_idx.size} 天（{dates[score_idx[0]]} … "
        f"{dates[score_idx[-1]]}），预期 334 → {'✔' if score_idx.size == 334 else '✘'}")

    price = np.asarray(Z["price"], float)
    nu = C.compute_nu()
    log("")
    log("── 共享常量 ──")
    log(f"电价（附件 1，144 时段）：min {price.min():.4f} / max {price.max():.4f} "
        f"/ 均值 {price.mean():.4f} 元/kWh")
    log(f"ν = {nu!r}")
    log(f"与 _comm3.NU_VALUE = {C.NU_VALUE!r} 逐位一致："
        f"{'✔' if abs(nu - C.NU_VALUE) < 1e-15 else '✘'}")
    assert abs(nu - C.NU_VALUE) < 1e-15

    np.savez_compressed(
        C.V_FORECAST_NPZ,
        dates=dates,
        tau_hours=A3["tau_hours"],
        V_raw_kw=V_raw,
        V_10min_piecewise=V_pw,
        V_10min_linear=V_li,
        pv_hourly_actual=pv_hourly_actual,
        price=price,
        warmup_day_index=warmup_idx,
        score_day_index=score_idx,
        delta_hours=np.array([C.DELTA_HOURS]),
        s_period_kwh=np.array([C.S_PERIOD_KWH]),
        nu=np.array([nu]),
        disagg_default=np.array([C.DISAGG_PIECEWISE]),
    )
    log("")
    log(f"已保存：{C.V_FORECAST_NPZ.relative_to(C.PROJECT_DIR)}"
        f"（{C.V_FORECAST_NPZ.stat().st_size / 1024:.1f} KiB）")

    checks = [
        ("附件 3 行数 = 1460", True, "365 天 × 4 个预报时刻（由 01 脚本核验）"),
        ("附件 3 列数 = 26", True, "日期 + 预报时刻 + 预报1..24小时"),
        ("(日期,τ) 组合无缺失", n_missing == 0, f"缺失 {n_missing} 个"),
        ("预报数值无 NaN/Inf", n_nan == 0, f"非有限值 {n_nan} 个"),
        ("与附件 2 日期逐日对齐", same_dates, f"共 {dates.size} 天"),
        ("piecewise 能量守恒", w_cons_pw < 1e-9, f"最大偏差 {w_cons_pw:.3e}"),
        ("linear 展开非负", bool(V_li.min() >= 0), f"min {V_li.min():.4f} kW"),
        ("预热期 = 31 天", warmup_idx.size == 31, f"实际 {warmup_idx.size}"),
        ("评分期 = 334 天", score_idx.size == 334, f"实际 {score_idx.size}"),
        ("ν 逐位一致", abs(nu - C.NU_VALUE) < 1e-15, f"{nu!r}"),
        ("npz 可回读", False, "见下方回读校验"),
    ]
    Zc = np.load(C.V_FORECAST_NPZ, allow_pickle=False)
    rb_ok = (np.allclose(Zc["V_raw_kw"], V_raw)
             and np.allclose(Zc["V_10min_piecewise"], V_pw)
             and list(Zc["dates"]) == list(dates))
    checks[-1] = ("npz 可回读且逐值一致", rb_ok, "V_raw / piecewise / dates 全部一致")
    log("")
    log("── 验收表 ──")
    for name, ok, detail in checks:
        log(f"  {'✔' if ok else '✘'} {name}：{detail}")

    C.write_csv_utf8_sig(
        C.RAW_INFO_DIR / "附件三处理后验收表.csv",
        ["验收项", "是否通过", "说明"],
        [[n, "通过" if ok else "不通过", d] for n, ok, d in checks])

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    log("")
    log(f"验收：{len(checks) - n_fail}/{len(checks)} 通过")
    log(f"总用时 {time.perf_counter() - t0:.1f} s")
    log("")
    log("[02 完成] 附件 3 已转成模型可直接调用的矩阵。")

    log.dump(C.LOG_DIR / "第三问_02数据处理日志.txt",
             "[02 完成] 附件 3 已转成模型可直接调用的矩阵。")

    C.write_text_utf8(
        C.REPORT_PROC_MD,
        "\n".join(["# 附件 3 数据处理报告", "",
                   "> 脚本：`第三问最终版/代码/02_处理附件三基础数据.py`；"
                   "产出：`处理后数据/附件三_预报矩阵.npz`", ""] + log.lines + [""]))

    C.write_text_utf8(
        C.REPORT_ACCEPT_MD,
        "\n".join(["# 附件 3 数据处理验收表", "",
                   "| 验收项 | 结果 | 说明 |", "|---|---|---|"]
                  + [f"| {n} | {'通过' if ok else '不通过'} | {d} |"
                     for n, ok, d in checks] + [""]))

    C.write_text_utf8(C.REPORT_TIME_MD, f"""# 附件三时间标签映射说明

> 本题所有时间标签的约定（**口径冻结**）。与附件 2 完全一致，见
> `第二问最终版/报告/附件二时间标签映射说明.md`。

## 1. 时间段划分

* 全天 24 h 等分为 **144 个 10 分钟时段**，Δ t = 1/6 h；
  第 $t$ 个时段（$t=1,\\dots,144$）对应 $[ (t-1)\\Delta t,\\ t\\Delta t )$；
* **流量变量（购电量、充放电量、紧急购电量）的单位一律是「该时段内的电量」kWh**，
  不是功率；
* 功率上限换算：$S = P_{{\\max}}\\cdot\\Delta t = 5000\\times \\frac{{1}}{{6}} = {C.S_PERIOD_KWH:.6f}$ kWh/时段。

## 2. 附件 3 的整点口径

附件 3 的 24 个预报列名为 `预报1小时 … 预报24小时`。本方案采用的口径是
**元素偏移 0**：

$$\\text{{预报 }}k\\text{{ 小时}} \\;=\\; \\text{{整点区间 }}[\\tau + k - 1,\\; \\tau + k)\\ \\text{{的平均出力}}$$

即发布时刻 $\\tau$ 的 24 个元素覆盖 $[\\tau,\\ \\tau+24)$ 的 24 个整点小时。

**支持该口径的四条证据**（`报告/第三问_附件三口径核验报告.md` 有完整数据）：

1. 列名逐字为 `预报1小时…预报24小时`，字面即「1 小时后」；
2. 若把元素 $k$ 映射到**日内小时** $hh=(\\tau+k)\\bmod 24$，四个 $\\tau$ 的
   「$hh$ → 非零天数」剖面完全重合（非零窗口一律为 $[4:00,\\ 19:00)$），
   → 元素偏移为 0 时四条曲线共享同一昼夜结构；
3. 平移扫描（相对 MAE）在**位移 0** 处取到唯一全局最小；
4. 滚动增益（评价窗口 $d=1..363$，与核验脚本 `03` 全表同窗口）：$[6:00,24:00)$
   窗口 443.18 → 388.83 kW，$[12:00,24:00)$ 351.44 → 305.75 → 272.43 kW，
   说明晚发布的预报确实更新了后续时段；逐点数值见
   `模型结果/第三问_滚动增益表.csv`。

## 3. 两种整点 → 10 min 展开口径

| 口径 | 公式 | 整点小时 MAE | RMSE | 采用 |
|---|---|---|---|---|
| `piecewise_constant` | 每小时复制 6 份 | **171.05 kW** | **295.40 kW** | ✔ 默认 |
| `linear_interp` | $P[6k+r] = (1-r/6)F[k] + (r/6)F[k+1]$ | 276.02 kW | 424.19 kW | 敏感档 |

分段常数保**每小时能量守恒**（展开后 6 段均值 = 原始整点值，误差 < 1e-9 kW）；
线性插值会把能量在相邻小时之间搬移，且在日出/日落边界产生虚假的爬坡。
因此主结果用分段常数。

## 4. 跨日与列序

* 第 $\\tau=0:00$ 行的 24 个元素覆盖当日 $[0:00,\\ 24:00)$；
  $\\tau=6/12/18$ 行的元素会**跨到次日**（例如 $\\tau=18:00$ 的元素 7 覆盖次日 $[0:00,1:00)$）；
* `提交结果/result3.xlsx` 的表头列标签为 `0:10-0:20 … 0:00-0:10+1`，
  比模型实际区间**整体晚一个时段**（官方模板的既成约定）。
  本方案按 **「列序 = 时间序」** 填写：第 $N$ 个数据列（$N=1..144$，Excel 列
  B..EO）对应模型第 $N$ 个时段 $[(N-1)\\Delta t,\\ N\\Delta t)$；
  **表头文字一律不改写**。

## 5. 与第二问的衔接

附件 2 的 10 min 实际值（`load_kw` / `pv_kw`）与附件 3 的整点预报在
**同一时间轴**上（均为区间终点标签、均为 [0:00,24:00) 覆盖），
因此残差 `附件3预报 − 附件2实际` 可以直接按 $k$ 对齐，不需要额外平移。
""")

    log("已保存：报告/附件三数据处理报告.md、附件三数据处理验收表.md、"
        "附件三时间标签映射说明.md、原始数据说明/附件三处理后验收表.csv")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
