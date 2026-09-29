from __future__ import annotations

import importlib.util
import os
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

ACC_CSV = C.RESULT_DIR / "第二问_预测精度表.csv"


PREDICTOR = os.environ.get("Q2_PREDICTOR", "decomp").strip().lower()
_PV_WINDOW_ENV = os.environ.get("Q2_PV_WINDOW", "").strip()

TYPE_LOCK_DAY = 15
CLASSIFY_DAYS = 14
SHAPE_DAYS = 3
BETA_WINDOW_DAYS = 35
LOW_TYPE, HIGH_TYPE = 0, 1


def pv_window_days() -> int:
    return int(_PV_WINDOW_ENV) if _PV_WINDOW_ENV else int(C.PV_LOOKBACK_DAYS)


def _naive_load(load_e: np.ndarray, wd: np.ndarray, d: int, tag: str = ""):
    cand = [i for i in range(max(0, d - C.LOAD_LOOKBACK_DAYS), d) if wd[i] == wd[d]]
    if len(cand) >= 2:
        return load_e[cand].mean(0), f"{tag}同星期×{len(cand)}"
    lo = max(0, d - C.LOAD_FALLBACK_DAYS)
    if d - lo > 0:
        return load_e[lo:d].mean(0), f"{tag}回退前{d - lo}天"
    return load_e[0], f"{tag}无历史(取当日)"


def identify_low_weekdays(load_e: np.ndarray, wd: np.ndarray, n_classify: int = CLASSIFY_DAYS):
    B = load_e.sum(1)
    per: dict[int, float] = {}
    for w in range(7):
        sel = [i for i in range(min(n_classify, len(B))) if wd[i] == w]
        if sel:
            per[w] = float(B[sel].mean())
    order = sorted(per, key=lambda w: per[w])
    return set(order[:2]), per


def _beta(k: np.ndarray, B: np.ndarray, d: int, window: int = BETA_WINDOW_DAYS) -> float:
    ratios = []
    for i in range(max(1, d - window), d):
        if k[i] == k[i - 1] or B[i - 1] <= 0:
            continue
        ratios.append(float(np.log(B[i] / B[i - 1])) / float(k[i] - k[i - 1]))
    return float(np.median(ratios)) if ratios else 0.0


def _decomp_load(load_e, B, k, wd, d, shape_days: int = SHAPE_DAYS):
    same = [i for i in range(d - 1, -1, -1) if k[i] == k[d]][:shape_days]
    denom = float(B[same].sum()) if same else 0.0
    if denom > 0:
        shape = load_e[same].sum(0) / denom
        beta = _beta(k, B, d)
        bhat = float(B[d - 1]) * float(np.exp(beta * float(k[d] - k[d - 1])))
        return bhat * shape, f"分解{len(same)}日(β={beta:+.4f})"
    return _naive_load(load_e, wd, d, tag="兜底")


def causal_forecast(load_e: np.ndarray, pv_e: np.ndarray, dates: list,
                    predictor: str | None = None, pv_window: int | None = None):
    n_day, T = load_e.shape
    pred = (predictor or PREDICTOR).strip().lower()
    if pred not in ("decomp", "naive"):
        raise ValueError(f"未知预测器 Q2_PREDICTOR={pred!r}，应为 'decomp' 或 'naive'")
    pw = pv_window_days() if pv_window is None else int(pv_window)

    wd = np.array([d.weekday() for d in dates], dtype=int)
    B = load_e.sum(1)
    low_set, per_wd = identify_low_weekdays(load_e, wd, CLASSIFY_DAYS)
    k = np.array([LOW_TYPE if w in low_set else HIGH_TYPE for w in wd], dtype=int)
    lock0 = TYPE_LOCK_DAY - 1

    Lhat = np.zeros((n_day, T))
    Vhat = np.zeros((n_day, T))
    src = np.empty(n_day, dtype=object)

    for d in range(n_day):
        if pred == "naive":
            Lhat[d], src[d] = _naive_load(load_e, wd, d)
        elif d < lock0:
            Lhat[d], src[d] = _naive_load(load_e, wd, d, tag="预热")
        else:
            Lhat[d], src[d] = _decomp_load(load_e, B, k, wd, d)
        m = min(pw, d)
        Vhat[d] = pv_e[d - m:d].mean(0) if m > 0 else pv_e[0]

    diag = {"predictor": pred, "pv_window": pw, "k": k, "B": B,
            "low_weekdays": sorted(low_set), "per_weekday_energy": per_wd}
    return Lhat, Vhat, src, diag


def build_scenarios(Lhat, Vhat, resid_L, resid_V):
    n_day, T = Lhat.shape
    M = C.M_SCENARIOS
    scen_L = np.zeros((n_day, M, T))
    scen_V = np.zeros((n_day, M, T))
    scen_idx = np.zeros((n_day, M), dtype=int)
    for d in range(n_day):
        pool = list(range(max(1, d - C.RESIDUAL_WINDOW_DAYS), d))
        if not pool:
            pool = [0]
        for w in range(M):
            i = pool[w % len(pool)]
            scen_idx[d, w] = i
            scen_L[d, w] = np.maximum(0.0, Lhat[d] + resid_L[i])
            scen_V[d, w] = np.maximum(0.0, Vhat[d] + resid_V[i])
    return scen_L, scen_V, scen_idx


def mp_rho(resid_N, d):
    lo = max(1, d - C.MPC_RHO_WINDOW_DAYS)
    if d - lo < 2:
        return 0.0
    E = resid_N[lo:d]
    a = E[:, :-1].reshape(-1)
    b = E[:, 1:].reshape(-1)
    den = float(a @ a)
    if den <= 1e-12:
        return 0.0
    rho = float(a @ b) / den
    return float(np.clip(rho, *C.MPC_RHO_CLIP))


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    p("=" * 74)
    p("第二问 04 —— 因果预测与联合情景库")
    p("=" * 74)

    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    load_e, pv_e = Z["load_energy_kwh"], Z["pv_energy_kwh"]
    net_e = Z["net_load_energy_kwh"]
    date_strs = [str(s) for s in Z["dates"]]
    dates = [pd.Timestamp(s).date() for s in date_strs]
    score_idx = Z["score_day_index"]
    n_day, T = load_e.shape

    Lhat, Vhat, src, diag = causal_forecast(load_e, pv_e, dates)
    Nhat = Lhat - Vhat
    resid_L = load_e - Lhat
    resid_V = pv_e - Vhat
    resid_N = net_e - Nhat
    B_act = diag["B"]

    p("")
    p(f"[式 17] 因果预测已生成（预测器 = {diag['predictor']}，光伏回望窗 = {diag['pv_window']} 天）")
    p("  第 d 天仅使用 d 之前的数据；残差按各历史日当时可得信息重新生成")
    _wk = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    if diag["predictor"] == "decomp":
        _per = diag["per_weekday_energy"]
        p("  日类型识别（用 1/1—1/14 平均日电量，取最低两类）：")
        for w in sorted(_per, key=lambda q: _per[q]):
            _mark = "← 低负载类 k=0" if w in diag["low_weekdays"] else ""
            p(f"    {_wk[w]}  平均日电量 {_per[w]:12.3f} kWh  {_mark}")
        p(f"  自 1/15 起锁定分类，低负载星期 = {[_wk[w] for w in diag['low_weekdays']]}")
    p(f"  负载来源分布：{pd.Series([src[d] for d in range(n_day)]).value_counts().to_dict()}")

    scen_L, scen_V, scen_idx = build_scenarios(Lhat, Vhat, resid_L, resid_V)
    scen_N = scen_L - scen_V
    p("")
    p(f"[式 19] 情景库已生成：形状 {scen_L.shape}（天 × 情景 × 时段），等权 1/{C.M_SCENARIOS}")
    p(f"[式 19] 仅做非负截断 [·]^+，不含任何光伏历史最大值截断（符合图片基准）")

    rho = np.array([mp_rho(resid_N, d) for d in range(n_day)])
    p(f"[式 28] MPC 误差修正系数 ρ：评分期 min={rho[score_idx].min():.6f} "
      f"max={rho[score_idx].max():.6f} mean={rho[score_idx].mean():.6f}")

    dt = C.DELTA_HOURS
    mae_L = np.abs(load_e - Lhat).mean(1) / dt
    mae_V = np.abs(pv_e - Vhat).mean(1) / dt
    mae_N = np.abs(net_e - Nhat).mean(1) / dt
    month = np.array([d.strftime("%Y-%m") for d in dates])
    rows = []
    for m in sorted(set(month)):
        sel = np.where(month == m)[0]
        rows.append((m, len(sel), f"{mae_L[sel].mean():.6f}", f"{mae_V[sel].mean():.6f}",
                     f"{mae_N[sel].mean():.6f}"))
    rows.append(("全年(评分期)", len(score_idx),
                 f"{mae_L[score_idx].mean():.6f}", f"{mae_V[score_idx].mean():.6f}",
                 f"{mae_N[score_idx].mean():.6f}"))
    C.write_csv_utf8_sig(ACC_CSV,
                         ("月份", "天数", "负载MAE_kW", "光伏MAE_kW", "净负荷MAE_kW"), rows)
    p("")
    p("[预测精度] 评分期平均 MAE：")
    p(f"    负载 {mae_L[score_idx].mean():.4f} kW")
    p(f"    光伏 {mae_V[score_idx].mean():.4f} kW")
    p(f"    净负荷 {mae_N[score_idx].mean():.4f} kW")
    p(f"    → 净负荷 MAE 是否小于负载与光伏 MAE 之和："
      f"{'✔ 是' if mae_N[score_idx].mean() < mae_L[score_idx].mean() + mae_V[score_idx].mean() else '✘ 否'}")

    Bhat = Lhat.sum(1)
    mape_L = float(100.0 * (np.abs(load_e - Lhat) / np.maximum(load_e, 1e-9))[score_idx].mean())
    mape_B = float(100.0 * (np.abs(B_act - Bhat) / np.maximum(B_act, 1e-9))[score_idx].mean())
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_预测精度锚点对照.csv",
        ("指标", "本次运行", "朴素基线", "分解+七日光伏", "分解+三日光伏"),
        [("负载MAPE_%", f"{mape_L:.6f}", "4.717", "3.191", "3.191"),
         ("日电量MAPE_%", f"{mape_B:.6f}", "3.402", "0.897", "0.897"),
         ("负载MAE_kW", f"{mae_L[score_idx].mean():.6f}", "202.720", "135.210", "135.210"),
         ("光伏MAE_kW", f"{mae_V[score_idx].mean():.6f}", "153.400", "153.400", "154.459"),
         ("净负荷MAE_kW", f"{mae_N[score_idx].mean():.6f}", "296.209", "241.285", "244.431")])
    p("")
    p("[口径锚点] 评分期（2—12 月）与 PDF 表 6 对照：")
    p(f"    负载 MAPE   {mape_L:8.3f} %   （朴素 4.717 / 分解 3.191）")
    p(f"    日电量 MAPE {mape_B:8.3f} %   （朴素 3.402 / 分解 0.897）")
    p(f"    负载 MAE    {mae_L[score_idx].mean():8.3f} kW  （朴素 202.720 / 分解 135.210）")
    p(f"    光伏 MAE    {mae_V[score_idx].mean():8.3f} kW  （7日 153.400 / 3日 154.459）")
    p(f"    净负荷 MAE  {mae_N[score_idx].mean():8.3f} kW  （7日 241.285 / 3日 244.431）")

    bad = int((scen_V < 0).sum() + (scen_L < 0).sum())
    p("")
    p(f"[情景边界] 负值个数 = {bad}（应为 0）")
    p(f"  情景净负荷 min={scen_N.min():.4f} max={scen_N.max():.4f} kWh（物理允许为负=光伏富余）")

    np.savez_compressed(
        C.SCENARIO_NPZ,
        dates=Z["dates"], Lhat=Lhat, Vhat=Vhat, Nhat=Nhat,
        resid_L=resid_L, resid_V=resid_V, resid_N=resid_N,
        scen_L=scen_L, scen_V=scen_V, scen_N=scen_N,
        scen_idx=scen_idx, rho=rho,
        score_day_index=score_idx, warmup_day_index=Z["warmup_day_index"],
        price=Z["price"],
    )
    p(f"情景库已写出：{C.SCENARIO_NPZ.name}")

    _wkname = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    _low_names = "、".join(_wkname[w] for w in diag["low_weekdays"])
    if diag["predictor"] == "decomp":
        _rule_load = ("**日电量水平 × 日内形状**分解（式 17a~17c）："
                      "ŝ = Σ_{i∈I_d}L_{i,t}/Σ_{i∈I_d}B_i，"
                      "B̂_d = B_{d−1}·exp[β_d·(k_d − k_{d−1})]，L̂_{d,t} = B̂_d·ŝ_{d,t}")
        _par_load = (f"形状取最近 {SHAPE_DAYS} 个同类型历史日；"
                     f"β 取此前 {BETA_WINDOW_DAYS} 天类型切换日对数比例的中位数")
        _note_type = (f"日类型用 1/1—1/14 的平均日电量识别（低负载星期 = {_low_names}），"
                      "自 1/15 起锁定分类；1/15 前日类型未锁定，负载沿用同星期均值。")
    else:
        _rule_load = "距今 L_d 天内同星期样本的逐时段均值（**朴素基线**）"
        _par_load = f"L_d = {C.LOAD_LOOKBACK_DAYS} 天，回退窗口 {C.LOAD_FALLBACK_DAYS} 天"
        _note_type = "本档为朴素基线（全程同星期均值），用于与分解预测对照。"

    md = [
        "# 第二问预测精度与情景构造报告",
        "",
        "## 1. 因果预测（式 17）",
        "",
        f"**预测器 = `{diag['predictor']}`；光伏回望窗 = {diag['pv_window']} 天。**",
        "",
        "| 量 | 规则 | 参数 |",
        "| --- | --- | --- |",
        f"| 负载 L̂ | {_rule_load} | {_par_load} |",
        f"| 光伏 V̂ | 前 m_d 天的逐时段均值 | m_d = min({diag['pv_window']}, d−1) |",
        "",
        _note_type,
        "",
        "第 d 天的预测只使用第 d 天 0:00 之前的数据，逐日滚动生成，"
        "**不含任何前视信息**，也不使用附件一（附件一为附件二的全年均值典型日）。",
        "更换预测器后，历史残差 e_i = 实际_i − 当日预测_i 按**各历史日当时可得信息**"
        "重新生成，不复用旧预测器的残差。",
        "",
        "## 2. 预测精度（kW 口径 MAE）",
        "",
        "| 月份 | 天数 | 负载 MAE | 光伏 MAE | 净负荷 MAE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {float(r[2]):.4f} | {float(r[3]):.4f} | {float(r[4]):.4f} |")
    md += [
        "",
        f"净负荷 MAE（{mae_N[score_idx].mean():.4f} kW）小于负载 MAE 与光伏 MAE 之和"
        f"（{mae_L[score_idx].mean() + mae_V[score_idx].mean():.4f} kW），"
        "说明负载与光伏的预测误差存在部分相互抵消，直接用净负荷预测调度的思想是合理的。",
        "",
        "## 3. 联合情景（式 18~19）",
        "",
        f"- 情景数 M = {C.M_SCENARIOS}，等权 1/M；",
        f"- 残差库 = 最近 {C.RESIDUAL_WINDOW_DAYS} 个历史日的残差（该历史日 0:00 即可获得，无前视）；",
        "- **同一历史日的负载残差与光伏残差配对使用**，保留两者联合相关性；",
        "- 情景均值与预测值的关系：E_ω[L_{t,ω}] = L̂_t + mean_ω(e^L) ≈ L̂_t（残差近零均值）；",
        "- 情景仅做非负截断 $[·]^+$，**不含**光伏历史最大值截断（式 19 基准，图片外截断已删除）。",
        "",
        "## 4. MPC 误差修正系数（式 28）",
        "",
        f"- ρ 由最近 {C.MPC_RHO_WINDOW_DAYS} 天净负荷残差的一阶自回归拟合，截断到 {C.MPC_RHO_CLIP}；",
        f"- 评分期 ρ ∈ [{rho[score_idx].min():.6f}, {rho[score_idx].max():.6f}]，"
        f"均值 {rho[score_idx].mean():.6f}。",
        "",
        "## 5. 输出文件",
        "",
        f"- `处理后数据/{C.SCENARIO_NPZ.name}`",
        f"- `模型结果/{ACC_CSV.name}`",
        "",
    ]
    C.write_text_utf8(C.SCENARIO_MD, "\n".join(md))
    p(f"报告已写出：{C.SCENARIO_MD.name}")

    C.write_text_utf8(C.SOLVE_LOG_TXT, "\n".join(log + ["", "[04 完成] 情景库构建结束。"]))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
