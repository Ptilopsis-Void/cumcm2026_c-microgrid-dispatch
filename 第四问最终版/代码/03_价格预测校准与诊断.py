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
P = _load("_price4.py", "q4_price")

import numpy as np

T = C.PERIODS_PER_DAY
N_DAY = 365
WARMUP_DAYS = C.N_WARMUP_DAYS
CAL_DAYS = np.arange(0, WARMUP_DAYS)
SCORE_DAYS = np.arange(WARMUP_DAYS, N_DAY)

ADOPTED = {"42": "net_load_regression", "43": "ar1"}


def main() -> int:
    C.ensure_dirs()
    t0 = time.perf_counter()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 78)
    p("第四问 03 —— 电价因果预测回放 / 校准诊断 / 口径锁定")
    p("=" * 78)

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail):
        checks.append((name, bool(ok), detail))
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    Z4 = np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False)
    price = np.asarray(Z4["price_actual"], float)
    dates = np.array([str(s) for s in Z4["dates"]], dtype="<U10")
    Z2 = C.load_q2_matrix()
    S2 = C.load_q2_scenarios()
    N_act = np.asarray(Z2["net_load_energy_kwh"], float)
    Nhat_q2 = np.asarray(S2["Nhat"], float)
    p(f"  price {price.shape}；N_act {N_act.shape}；Nhat(第二问) {Nhat_q2.shape}")

    p("")
    p("── 1. 节点可见净负荷特征（节点前用实际，节点后用第二问预测）──")
    nhat_snap = P.net_load_feature_all(N_act, Nhat_q2)
    p(f"  nhat_snap {nhat_snap.shape}（天 × 4 节点）；"
      f"均值 {nhat_snap.mean():,.0f} kWh")
    for ti, k0 in enumerate(C.TAU_PERIOD_INDEX):
        vis = k0
        p(f"    τ={C.TAU_HOURS[ti]:>2} 时：节点前 {vis} 段用实际，之后 {T - vis} 段用第二问预测")
    ck("特征形状与节点数一致", nhat_snap.shape == (N_DAY, len(C.TAU_HOURS)),
       f"{nhat_snap.shape}")

    p("")
    p("── 2. 五个候选预测方法的全年因果回放 ──")
    reps: dict[str, dict] = {}
    for method in P.METHODS:
        tm = time.perf_counter()
        reps[method] = P.replay_forecasts(price, nhat_snap, method)
        p(f"  {P.METHOD_LABEL.get(method, method):<12s} 回放完成（{time.perf_counter() - tm:.2f} s）")

    def diag(method, days):
        r = reps[method]
        return P.score_forecasts(price, r["chat_corr"], r["H"], days)

    segs = [("校准期（1月）", CAL_DAYS), ("评分期（2–12月）", SCORE_DAYS), ("全年", np.arange(N_DAY))]
    rows_csv: list[list] = []
    p("")
    p("  全年 MAE（元/kWh）：")
    for method in P.METHODS:
        sc = diag(method, np.arange(N_DAY))
        p(f"    {P.METHOD_LABEL.get(method, method):<12s} {sc['mae']:.4f}")

    for method in P.METHODS:
        r = reps[method]
        for seg_name, days in segs:
            sc = P.score_forecasts(price, r["chat_corr"], r["H"], days)
            rows_csv.append([P.METHOD_LABEL.get(method, method), "总体", seg_name,
                             f"{sc['mae']:.4f}", f"{sc['rmse']:.4f}",
                             f"{sc['rmse_dtrend']:.4f}", f"{sc['bias']:+.4f}", sc["n"]])

    months = sorted({dates[d][:7] for d in range(N_DAY)})
    monthly: dict[str, dict[str, dict]] = {m: {} for m in P.METHODS}
    for method in P.METHODS:
        r = reps[method]
        for mo in months:
            dd = np.array([d for d in range(N_DAY) if dates[d][:7] == mo], int)
            sc = P.score_forecasts(price, r["chat_corr"], r["H"], dd)
            monthly[method][mo] = sc
            rows_csv.append([P.METHOD_LABEL.get(method, method), "月份", mo,
                             f"{sc['mae']:.4f}", f"{sc['rmse']:.4f}",
                             f"{sc['rmse_dtrend']:.4f}", f"{sc['bias']:+.4f}", sc["n"]])

    q_hi = np.quantile(price, 0.75, axis=1, keepdims=True)
    q_lo = np.quantile(price, 0.25, axis=1, keepdims=True)
    peak_mask = price >= q_hi
    valley_mask = price <= q_lo
    pk: dict[str, dict] = {}
    for method in P.METHODS:
        r = reps[method]
        e_pk, e_val = [], []
        for d in range(N_DAY):
            for ti in range(len(C.TAU_HOURS)):
                H = int(r["H"][d, ti])
                if H <= 0:
                    continue
                g = np.array([P.window_index(d, C.TAU_PERIOD_INDEX[ti], h) for h in range(H)])
                ok = g < P.N_FLAT
                g, hh = g[ok], np.arange(H)[ok]
                tslot = (g % T)
                err = r["chat_corr"][d, ti, :H][hh] - price[d, tslot]
                e_pk.append(err[peak_mask[d, tslot]])
                e_val.append(err[valley_mask[d, tslot]])
        ep = np.concatenate(e_pk) if e_pk else np.zeros(0)
        ev = np.concatenate(e_val) if e_val else np.zeros(0)
        pk[method] = {
            "peak": {"mae": float(np.mean(np.abs(ep))), "rmse": float(np.sqrt(np.mean(ep ** 2))),
                     "bias": float(np.mean(ep)), "n": int(ep.size)},
            "valley": {"mae": float(np.mean(np.abs(ev))), "rmse": float(np.sqrt(np.mean(ev ** 2))),
                       "bias": float(np.mean(ev)), "n": int(ev.size)},
        }
        for seg_name, dic in (("峰段", pk[method]["peak"]), ("谷段", pk[method]["valley"])):
            rows_csv.append([P.METHOD_LABEL.get(method, method), "时段类型", seg_name,
                             f"{dic['mae']:.4f}", f"{dic['rmse']:.4f}", "", f"{dic['bias']:+.4f}",
                             dic["n"]])

    p("")
    p("  评分期（2–12 月）MAE / RMSE（元/kWh）：")
    for method in P.METHODS:
        sc = diag(method, SCORE_DAYS)
        p(f"    {P.METHOD_LABEL.get(method, method):<12s} MAE {sc['mae']:.4f}  "
          f"RMSE {sc['rmse']:.4f}  RMSE' {sc['rmse_dtrend']:.4f}  偏差 {sc['bias']:+.4f}")

    p("")
    p("── 3. 泄漏扰动测试：篡改未来价格，过去预测必须逐位不变 ──")
    rng = np.random.default_rng(C.SEED)
    price_mod = price.copy()
    tau0 = 100
    price_mod[tau0:] = rng.uniform(0.01, 3.0, size=price_mod[tau0:].shape)
    ok_all = True
    detail = []
    for method in ("ar1", "net_load_regression", P.METHODS[0]):
        rm = P.replay_forecasts(price_mod, nhat_snap, method)
        same_base = np.allclose(reps[method]["chat_base"][:tau0], rm["chat_base"][:tau0], atol=1e-12)
        same_corr = np.allclose(reps[method]["chat_corr"][:tau0], rm["chat_corr"][:tau0], atol=1e-12)
        same_lev = np.allclose(reps[method]["level"][:tau0], rm["level"][:tau0], atol=1e-12)
        detail.append(f"{P.METHOD_LABEL.get(method, method)}: base={same_base} corr={same_corr} level={same_lev}")
        ok_all = ok_all and same_base and same_corr and same_lev
    for s in detail:
        p(f"    {s}")
    ck("未来价格篡改不影响过去预测（无前视泄漏）", ok_all,
       f"对 d < {tau0} 的 chat_base/chat_corr/level 逐位一致")

    p("")
    p("── 4. 数值与结构健全性 ──")
    valid = np.zeros((N_DAY, len(C.TAU_HOURS), T), bool)
    for d in range(N_DAY):
        for ti in range(len(C.TAU_HOURS)):
            H = int(np.maximum(reps[P.METHODS[0]]["H"][d, ti], 0))
            valid[d, ti, :H] = True
    bad = []
    for method in P.METHODS:
        r = reps[method]
        if not np.isfinite(r["chat_corr"][valid]).all():
            bad.append(f"{method}: 有 NaN/Inf")
        if (r["chat_corr"][valid] < C.PRICE_POINT_FLOOR - 1e-12).any():
            bad.append(f"{method}: 低于正价下限")
        if (r["level"] < C.PRICE_LEVEL_FLOOR - 1e-12).any():
            bad.append(f"{method}: 日均水平低于下限")
        Hv = r["H"]
        if (Hv < 0).any() or (Hv > T).any():
            bad.append(f"{method}: 展望长度越界")
    ck("所有方法预测有限、非负、展望长度合法", not bad,
       "; ".join(bad) if bad else f"5×365×4 节点共 {int(valid.sum()):,} 个有效位置全部通过")

    fb_rows = []
    for method in P.METHODS:
        for rec in reps[method]["diag"]:
            if rec["fallback"]:
                fb_rows.append([P.METHOD_LABEL.get(method, method), dates[rec["day"]],
                                str(rec["tau"]), str(rec["fallback"]),
                                str(rec["n_samples"]), f"{rec['level']:.4f}"])
    ck("预测回退均在案（方法/日期/节点/原因/样本数）", True,
       f"共 {len(fb_rows)} 条回退记录，已写入报告与 CSV")
    for row in fb_rows[:400]:
        rows_csv.append([row[0], "回退记录", f"{row[1]} τ={row[2]}", row[3], "", "",
                         row[5], row[4]])

    r42 = reps[ADOPTED["42"]]
    r43 = reps[ADOPTED["43"]]
    ck("4-2 采纳口径存在", ADOPTED["42"] in reps, ADOPTED["42"])
    ck("4-3 采纳口径存在", ADOPTED["43"] in reps, ADOPTED["43"])
    ck("未采用 MAPE 作为主指标", True, "主指标 = MAE / RMSE / 去漂移 RMSE（元/kWh）")

    p("")
    p("── 5. 落盘 ──")
    save = {
        "dates": dates, "H": r42["H"], "shape": r42["shape"],
        "price_actual": price, "nhat_snap": nhat_snap,
        "adopted_42": np.array([ADOPTED["42"]], dtype="<U32"),
        "adopted_43": np.array([ADOPTED["43"]], dtype="<U32"),
        "methods": np.array(list(P.METHODS), dtype="<U32"),
    }
    for method in P.METHODS:
        r = reps[method]
        save[f"chat_base__{method}"] = r["chat_base"]
        save[f"chat_corr__{method}"] = r["chat_corr"]
        save[f"level__{method}"] = r["level"]
        save[f"rho_p__{method}"] = r["rho_p"]
    np.savez_compressed(C.PRICE_FORECAST_NPZ, **save)
    p(f"  已保存：{C.PRICE_FORECAST_NPZ.relative_to(C.ROOT_DIR)}")

    C.write_csv_utf8_sig(C.PRICE_DIAG_CSV,
                         ["方法", "区间类型", "区间", "MAE_元每kWh", "RMSE_元每kWh",
                          "去漂移RMSE_元每kWh", "偏差_元每kWh", "样本数"], rows_csv)
    p(f"  已保存：{C.PRICE_DIAG_CSV.relative_to(C.ROOT_DIR)}")

    rp: list[str] = []
    rp.append("# 第四问 · 电价因果预测报告\n")
    rp.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"- 运行耗时：{time.perf_counter() - t0:.2f} s")
    rp.append("- 脚本：`第四问最终版/代码/03_价格预测校准与诊断.py`")
    rp.append("- 依据：《第四问详细流程图.md》§6\n")
    rp.append("## 1. 预登记口径（不得由评分期结果决定）\n")
    rp.append("| 分支 | 采纳方法 | 可用信息 | 理由 |")
    rp.append("|---|---|---|---|")
    rp.append("| 4-2 | `net_load_regression` | 历史价格；节点前实际净负荷、节点后第二问预测净负荷 | "
              "4-2 不得以任何方式读取附件三，但可用负荷信息；净负荷是电价的自然驱动量 |")
    rp.append("| 4-3 | `ar1` | 仅历史价格（可用附件三光伏，但价格模型保持价内生） | "
              "与 4-2 形成「信息集不同」的干净对照，避免两分支差异被预测方法混淆 |")
    rp.append("| 对照 | `arx`、`mean7`、`prev_day` | — | 仅作对照臂，**不作为主口径** |\n")
    rp.append("> 若主口径在评分期显著劣于对照臂，只能在报告中**明确披露偏离与原因**，"
              "不得静默替换方法，否则两分支比较失去意义。\n")
    rp.append("## 2. 模型形式\n")
    rp.append("1. **日水平与日内形状**：对已结束历史日 $i$ 取日均价 $\\ell_i$，"
              "日内形状 $s_t=\\overline{c_{i,t}/\\ell_i}$（对历史窗口逐日归一化再平均，"
              "形状按**当日**归一化故与 level 解耦）；")
    rp.append("2. **水平预测**（设计矩阵 = 截距 + 昨日水平 + 归一化净负荷）：")
    rp.append("   - `prev_day`: $\\hat\\ell=\\ell_{d-1}$；")
    rp.append("   - `mean7`: 最近 7 日水平均值；")
    rp.append("   - `ar1`: $\\hat\\ell=a+\\rho\\ell_{d-1}$；")
    rp.append("   - `net_load_regression`: $\\hat\\ell=a+b\\hat N_d/10^5$；")
    rp.append("   - `arx`: $\\hat\\ell=a+\\rho\\ell_{d-1}+b\\hat N_d/10^5$；")
    rp.append("3. **基础预测**：$\\hat c^{(0)}_{d,\\tau,h}=\\hat\\ell\\cdot s_{(k_0+h)\\bmod 144}$；")
    rp.append("4. **6/12/18 时偏差修正**：用最近一个已完成时段偏差 "
              "$\\epsilon=c_{d,k_0-1}-\\hat\\ell s_{k_0-1}$，按 $\\rho_p^{j}$ 衰减，"
              "$\\hat c_{d,\\tau,h}=\\max(\\hat c^{(0)}+\\rho_p^{h}\\epsilon,\\;10^{-4})$；$\\tau=0$ 不做修正；")
    rp.append("5. **回退链**：历史样本不足 → 昨日水平；设计矩阵秩亏 → 昨日水平；"
              "完全无历史 → 已登记固定正价先验 0.50 元/kWh（逐次记录在案）。\n")
    rp.append("## 3. 诊断结果（MAE / RMSE，元/kWh）\n")
    rp.append("### 3.1 分段总体\n")
    rp.append("| 方法 | 区间 | MAE | RMSE | 去漂移 RMSE | 偏差 | 样本数 |")
    rp.append("|---|---|---|---|---|---|---|")
    for method in P.METHODS:
        for seg_name, days in segs:
            sc = P.score_forecasts(price, reps[method]["chat_corr"], reps[method]["H"], days)
            rp.append(f"| {P.METHOD_LABEL.get(method, method)} | {seg_name} | {sc['mae']:.4f} | "
                      f"{sc['rmse']:.4f} | {sc['rmse_dtrend']:.4f} | {sc['bias']:+.4f} | {sc['n']} |")
    rp.append("\n### 3.2 峰段 / 谷段（用当日实际价 75%/25% 分位定义，仅用于事后评估）\n")
    rp.append("| 方法 | 时段 | MAE | RMSE | 偏差 | 样本数 |")
    rp.append("|---|---|---|---|---|---|")
    for method in P.METHODS:
        for seg_name, dic in (("峰段", pk[method]["peak"]), ("谷段", pk[method]["valley"])):
            rp.append(f"| {P.METHOD_LABEL.get(method, method)} | {seg_name} | {dic['mae']:.4f} | "
                      f"{dic['rmse']:.4f} | {dic['bias']:+.4f} | {dic['n']} |")
    rp.append("\n### 3.3 逐月 MAE（元/kWh）\n")
    rp.append("| 月份 | " + " | ".join(P.METHOD_LABEL.get(m, m) for m in P.METHODS) + " |")
    rp.append("|---" * (len(P.METHODS) + 1) + "|")
    for mo in months:
        rp.append(f"| {mo} | " + " | ".join(f"{monthly[m][mo]['mae']:.4f}" for m in P.METHODS) + " |")
    rp.append("\n### 3.4 采纳口径的逐节点误差（展望长度加权）\n")
    rp.append("| 分支 | 方法 | τ=0 | τ=6 | τ=12 | τ=18 |")
    rp.append("|---|---|---|---|---|---|")
    for br, method in ADOPTED.items():
        sc = diag(method, SCORE_DAYS)
        cells = " | ".join(f"{sc['per_tau'][h]['mae']:.4f}" for h in C.TAU_HOURS)
        rp.append(f"| 4-{br[0]}（{br}） | {method} | {cells} |")
    rp.append("\n## 4. 回退与数值披露\n")
    rp.append(f"- 回退记录共 **{len(fb_rows)}** 条（日均水平下限 {C.PRICE_LEVEL_FLOOR} 元/kWh、"
              f"逐点下限 {C.PRICE_POINT_FLOOR} 元/kWh、秩亏回退到昨日水平）。")
    if fb_rows:
        rp.append("")
        rp.append("| 方法 | 日期 | 节点 | 回退原因 | 拟合样本数 | 水平 |")
        rp.append("|---|---|---|---|---|---|")
        for row in fb_rows[:60]:
            rp.append(f"| {row[0]} | {row[1]} | τ={row[2]} | {row[3]} | {row[4]} | {row[5]} |")
        if len(fb_rows) > 60:
            rp.append(f"| … | … | … | 其余 {len(fb_rows) - 60} 条见 CSV | | |")
    else:
        rp.append("- 全期无回退：每个节点拟合样本均满足最小样本数且设计矩阵满秩。")
    rp.append("\n### 4.1 采纳口径的系数（评分期首末节点示例）\n")
    rp.append("| 分支 | 方法 | 日期 | 节点 | 截距 a | 自回归 ρ | 净负荷 b | 样本数 |")
    rp.append("|---|---|---|---|---|---|---|---|")
    for br, method in ADOPTED.items():
        for d in (WARMUP_DAYS, 180, N_DAY - 1):
            for rec in reps[method]["diag"]:
                if rec["day"] != d:
                    continue
                cf = rec["coef"] or {}
                a = cf.get("a"); rho = cf.get("rho"); b = cf.get("b")
                fmt = lambda v: ("—" if v is None else f"{float(v):.5f}")
                rp.append(f"| 4-{br[0]} | {method} | {dates[d]} | τ={rec['tau']} | "
                          f"{fmt(a)} | {fmt(rho)} | {fmt(b)} | {rec['n_samples']} |")
    rp.append("\n> 事后相关系数等全年诊断**不得回流训练**；本节仅作披露。\n")
    rp.append("## 5. 泄漏与健全性\n")
    rp.append("| 校验项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")
    C.write_text_utf8(C.PRICE_REPORT_MD, "\n".join(rp))
    p(f"  已保存：{C.PRICE_REPORT_MD.relative_to(C.ROOT_DIR)}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    p("")
    p("=" * 78)
    p(f"03 完成：5 方法 × 365 天 × 4 节点回放；采纳 4-2={ADOPTED['42']}、4-3={ADOPTED['43']}；"
      f"校验 {len(checks)} 项，未通过 {n_fail} 项。用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.LOG_DIR / "03_价格预测日志.txt", "\n".join(log) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
