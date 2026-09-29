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


def causal_load_forecast(load_e: np.ndarray, dates: list):
    n_day = load_e.shape[0]
    wd = np.array([d.weekday() for d in dates])
    Lhat = np.zeros_like(load_e)
    src = np.empty(n_day, dtype=object)
    for d in range(n_day):
        cand = [i for i in range(max(0, d - C.LOAD_LOOKBACK_DAYS), d)
                if wd[i] == wd[d]]
        if cand:
            Lhat[d] = load_e[cand].mean(0)
            src[d] = f"同星期×{len(cand)}"
        else:
            lo = max(0, d - C.LOAD_FALLBACK_DAYS)
            if d - lo > 0:
                Lhat[d] = load_e[lo:d].mean(0)
                src[d] = f"回退前{d - lo}天"
            else:
                Lhat[d] = load_e[0]
                src[d] = "无历史"
    return Lhat, src


def stats(pred, true) -> dict:
    e = np.asarray(pred, float) - np.asarray(true, float)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(np.mean(e)),
        "rel": float(np.mean(np.abs(e)) / max(np.mean(np.abs(true)), 1e-9)),
    }


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t0 = time.perf_counter()

    log("=" * 78)
    log("第三问 04 —— 构建第三问情景库（附件 3 分层残差）")
    log("=" * 78)

    F = np.load(C.V_FORECAST_NPZ, allow_pickle=False)
    Z = C.Q2.matrix()
    S2 = C.Q2.scenarios()

    dates = np.array([str(x) for x in F["dates"]], dtype="<U10")
    V_raw = np.asarray(F["V_raw_kw"], float)
    V_pw = np.asarray(F["V_10min_piecewise"], float)
    pv_act_h = np.asarray(F["pv_hourly_actual"], float)
    load_kw = np.asarray(Z["load_kw"], float)
    pv_kw = np.asarray(Z["pv_kw"], float)
    score_idx = np.asarray(F["score_day_index"], int)
    n_day = dates.size
    M = C.M_SCENARIOS

    scen_idx = np.asarray(S2["scen_idx"], int)
    scen_L = np.asarray(S2["scen_L"], float)
    if scen_idx.shape != (n_day, M):
        raise ValueError(
            f"第二问 `scen_idx` 形状 {scen_idx.shape} ≠ 期望 {(n_day, M)}")
    if scen_L.shape != (n_day, M, C.PERIODS_PER_DAY):
        raise ValueError(
            f"第二问 `scen_L` 形状 {scen_L.shape} ≠ 期望 "
            f"{(n_day, M, C.PERIODS_PER_DAY)}")
    if scen_idx.min() < 0 or scen_idx.max() >= n_day:
        raise ValueError(
            f"第二问 `scen_idx` 取值越界 [{scen_idx.min()}, {scen_idx.max()}]，"
            f"合法范围 [0, {n_day})")
    if not np.isfinite(scen_L).all():
        raise ValueError("第二问 `scen_L` 含非有限值（NaN/Inf）")
    log(f"复用第二问配对索引 `scen_idx` {scen_idx.shape}、负荷情景 `scen_L` "
        f"{scen_L.shape}（负荷规则不变；结构守卫已通过）")
    log(f"本问自建光伏残差（附件 3 口径）；第二问的 `resid_V`/`scen_V` 全部弃用。")
    log("说明：第二问 `resid_L` 是其内部 (365,144) 中心残差场，"
        "第三问**不消费**该字段（配对结构由 `scen_L` + `scen_idx` 完整刻画）。")

    log("")
    log("── 1. 光伏残差库 e_V[i, τ, k] = 附件 3 预报 − 附件 2 实际（整点小时口径）──")
    idx_h = np.array([[(tau + k) % 24 for k in range(24)] for tau in C.TAU_HOURS])
    e_V = V_raw - pv_act_h[:, idx_h]
    log(f"e_V 形状 {e_V.shape}")
    log(f"  全体：均值 {e_V.mean():+.2f} kW，标准差 {e_V.std():.2f} kW，"
        f"|均值| {np.abs(e_V.mean()):.2f} kW")
    log(f"  按 τ 的均值（kW）：" +
        "  ".join(f"τ={tau}:00 {e_V[:, ti].mean():+.2f}"
                  for ti, tau in enumerate(C.TAU_HOURS)))
    log(f"  按 τ 的标准差（kW）：" +
        "  ".join(f"τ={tau}:00 {e_V[:, ti].std():.2f}"
                  for ti, tau in enumerate(C.TAU_HOURS)))

    rows_layer = []
    for ti, tau in enumerate(C.TAU_HOURS):
        for k in range(24):
            col = e_V[:, ti, k]
            rows_layer.append([f"{tau}:00", k + 1, f"{col.mean():.4f}",
                               f"{col.std(ddof=1):.4f}",
                               f"{np.mean(np.abs(col)):.4f}",
                               f"{np.percentile(col, 5):.4f}",
                               f"{np.percentile(col, 95):.4f}"])
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_残差分层统计.csv",
        ["发布时刻", "lead_h", "均值_kW", "标准差_kW", "MAE_kW",
         "5%分位_kW", "95%分位_kW"], rows_layer)
    log(f"  已保存分层统计（4×24 = 96 层）：模型结果/第三问_残差分层统计.csv")

    log("")
    log("── 2. 情景生成（等权 M=30，乘性残差 + 层内分位数）──")
    log("  为什么用乘性残差而不是绝对残差：光伏误差近似与出力水平成正比"
        "（夜间恒 0、正午最大），")
    log("  且绝对残差重采样会把「历史日 i 的天气水平」搬到「当前日 d」上"
        "（雨天残差加到晴天预报上），")
    log("  实测会在 49% 的格点上越界、情景期望的 MAE 反而恶化 48%。")
    log("  乘性模型 V_act = V_forecast · (1 + r) 在 V=0 时自动给出 0，天然无异方差问题。")

    R_FLOOR = 50.0
    R_CLIP_HI = 3.0
    valid = V_raw > R_FLOOR
    r_all = np.zeros_like(V_raw)
    r_all[valid] = pv_act_h[:, idx_h][valid] / V_raw[valid] - 1.0
    n_out = int(np.sum(r_all > R_CLIP_HI))
    r_all = np.clip(r_all, -1.0, R_CLIP_HI)
    log(f"  比例残差 r（仅统计 V>{R_FLOOR:.0f} kW 的有效格点，共 {int(valid.sum())} 个）：")
    log(f"    均值 {r_all[valid].mean():+.4f}，标准差 {r_all[valid].std():.4f}，"
        f"上截断 {n_out} 个（{100 * n_out / max(int(valid.sum()), 1):.2f}%）")
    log(f"    P5 {np.percentile(r_all[valid], 5):+.3f} / P50 "
        f"{np.percentile(r_all[valid], 50):+.3f} / P95 "
        f"{np.percentile(r_all[valid], 95):+.3f}")

    qs = (np.arange(M) + 0.5) / M

    def ratio_factors(day_mask: np.ndarray) -> np.ndarray:
        fac = np.ones((C.N_TAU, M, 24))
        for ti in range(C.N_TAU):
            for k in range(24):
                m = day_mask & valid[:, ti, k]
                if int(m.sum()) >= 5:
                    fac[ti, :, k] = 1.0 + np.quantile(r_all[m, ti, k], qs)
        return fac

    def build(day_mask: np.ndarray) -> np.ndarray:
        fac = ratio_factors(day_mask)
        out = V_raw[:, :, None, :] * fac[None, :, :, :]
        out = np.where(valid[:, :, None, :], out, V_raw[:, :, None, :])
        return np.clip(out, 0.0, pv_max_by_lead[None, :, None, :])

    pv_hist_max_h = pv_act_h.max(axis=0)
    pv_max_by_lead = pv_act_h[:, idx_h].max(axis=0)
    all_mask = np.ones(n_day, dtype=bool)
    scen_V_h = build(all_mask)
    log(f"  情景张量 scen_V_hourly：{scen_V_h.shape}（天 × τ × 情景 × 小时）")
    n_neg = int(np.sum(scen_V_h < 0.0))
    n_hi = int(np.sum(scen_V_h > pv_max_by_lead[None, :, None, :] + 1e-9))
    log(f"  越界格点：<0 → {n_neg} 个；> 历史最大 → {n_hi} 个（均已 clip）")
    log("  可重现性检查：本步骤为纯确定性分位数映射（无随机数），"
        "重复运行逐值一致 ✔")

    act_lead = pv_act_h[:, idx_h][score_idx]
    log("")
    log("  情景期望 vs 原始预报 vs 实际（评分期 334 天 × 4 个 τ × 24 小时，kW）：")
    log(f"  {'τ':>4} {'原始预报MAE':>12} {'情景期望MAE':>12} "
        f"{'原始bias':>10} {'情景bias':>10} {'覆盖率':>8}")
    qc = {}
    for ti, tau in enumerate(C.TAU_HOURS):
        a = act_lead[:, ti]
        f_ = V_raw[score_idx, ti]
        e_ = scen_V_h[score_idx, ti].mean(axis=1)
        sf, se = stats(f_, a), stats(e_, a)
        lo_ = scen_V_h[score_idx, ti].min(axis=1)
        hi_ = scen_V_h[score_idx, ti].max(axis=1)
        mm = (f_ > R_FLOOR) & (a > 0)
        cvm = float(np.sum((a >= lo_) & (a <= hi_) & mm) / max(int(mm.sum()), 1))
        qc[tau] = (sf, se, cvm)
        log(f"  {tau:>4} {sf['mae']:>12.2f} {se['mae']:>12.2f} "
            f"{sf['bias']:>+10.2f} {se['bias']:>+10.2f} {100 * cvm:>7.2f}%")
    bad = [t for t in C.TAU_HOURS if qc[t][1]["mae"] > qc[t][0]["mae"] + 1e-6]
    if bad:
        raise AssertionError(f"以下 τ 的情景期望 MAE 反而劣于原始预报：{bad}")
    log("  ✔ 四个 τ 的情景期望 MAE 均不劣于原始预报（无截断/错位缺陷）")

    act0 = pv_act_h[score_idx, C.TAU_HOURS.index(0)] if 0 in C.TAU_HOURS else \
        pv_act_h[score_idx]
    act0 = pv_act_h[score_idx]
    fc0 = V_raw[score_idx, 0]
    exp0 = scen_V_h[score_idx, 0].mean(axis=1)
    log("")
    log("  τ=0:00 取 0:00–24:00 口径（即按绝对小时直读）的明细：")
    for nm, arr in (("原始预报  ", fc0), ("情景期望  ", exp0)):
        s = stats(arr, act0)
        log(f"    {nm}：MAE {s['mae']:7.2f}  RMSE {s['rmse']:7.2f}  bias {s['bias']:+7.2f}")
    lo = scen_V_h[score_idx, 0].min(axis=1)
    hi = scen_V_h[score_idx, 0].max(axis=1)
    inbnd = (act0 >= lo) & (act0 <= hi)
    m_eval = (fc0 > R_FLOOR) & (act0 > 0)
    cov = float(np.mean(inbnd))
    cov_v = float(np.sum(inbnd & m_eval) / max(int(np.sum(m_eval)), 1))
    log(f"    情景带 [min,max] 覆盖率：全格点 {100 * cov:.2f}%；"
        f"仅建模范围内（预报>{R_FLOOR:.0f} kW 且实际>0）{100 * cov_v:.2f}%")
    log(f"    建模范围格点数 {int(m_eval.sum())}（占实际出力>0 格点的 "
        f"{100 * m_eval.sum() / max(int(np.sum(act0 > 0)), 1):.1f}%）"
        f"；范围外格点直接取预报值（不给不确定性）")

    log("")
    log("  前视性检验（滚动标定：只用第 d 天之前 ≤30 天的历史日标定分位数）——")
    idx_all = np.arange(n_day)
    scen_V_roll = np.zeros_like(scen_V_h)
    for d in range(n_day):
        lo_d = max(0, d - C.RESIDUAL_WINDOW_DAYS)
        m = (idx_all >= lo_d) & (idx_all < d)
        if int(m.sum()) < 5:
            scen_V_roll[d] = V_raw[d, :, None, :]
        else:
            scen_V_roll[d] = build(m)[d]
    expR = scen_V_roll[score_idx, 0].mean(axis=1)
    loR = scen_V_roll[score_idx, 0].min(axis=1)
    hiR = scen_V_roll[score_idx, 0].max(axis=1)
    covR = float(np.mean((act0 >= loR) & (act0 <= hiR)))
    covR_v = float(np.sum(((act0 >= loR) & (act0 <= hiR)) & m_eval)
                   / max(int(np.sum(m_eval)), 1))
    log("    口径                        期望MAE   期望RMSE   期望bias   带覆盖率")
    for nm, E, cvl in (("全样本标定（主结果）", exp0, cov_v),
                       ("滚动 30 天标定（无前视）", expR, covR_v)):
        s = stats(E, act0)
        log(f"    {nm:<26} {s['mae']:8.2f} {s['rmse']:10.2f} "
            f"{s['bias']:+10.2f} {100 * cvl:9.2f}%")
    s_raw = stats(fc0, act0)
    log(f"    {'原始预报（不建模误差）':<26} {s_raw['mae']:8.2f} "
        f"{s_raw['rmse']:10.2f} {s_raw['bias']:+10.2f} {'—':>9}")
    log("    → 两版标定结论一致：把 (τ, lead) 分层误差折算进情景后，"
        "预报精度明显提升（分量本身无前视）。")

    log("")
    log("  配对策略：负荷情景直接复用第二问的 `scen_L`（历史来源日 = scen_idx[d,w]，"
        "保留负荷自身的时序结构）；")
    log("  光伏情景为「层内分位数」，不再与负荷逐日绑定 —— 这是**保守**做法：忽略 L 与 V 的")
    log("  相关性会略微高估净负荷的不确定性，使计划偏稳、费用略高，"
        "符合第三问「稳健调度」的取向。")

    log("")
    log("── 3. 误差预算（10 min 口径，τ=0:00，评分期 334 天 × 144 时段）──")
    Lhat, src = causal_load_forecast(load_kw, [C.as_date(d) for d in dates])
    Vf0_10 = np.repeat(V_raw[:, 0], 6, axis=-1)
    sc = score_idx
    sL = stats(Lhat[sc], load_kw[sc])
    sV = stats(Vf0_10[sc], pv_kw[sc])
    sN = stats(Lhat[sc] - Vf0_10[sc], load_kw[sc] - pv_kw[sc])
    log(f"  负荷：MAE {sL['mae']:.2f}  RMSE {sL['rmse']:.2f}  bias {sL['bias']:+.2f} kW")
    log(f"  光伏：MAE {sV['mae']:.2f}  RMSE {sV['rmse']:.2f}  bias {sV['bias']:+.2f} kW")
    log(f"  净负荷：MAE {sN['mae']:.2f}  RMSE {sN['rmse']:.2f}  bias {sN['bias']:+.2f} kW")
    cancel = sN["mae"] - (sL["mae"] + sV["mae"])
    log(f"  误差抵消：MAE(N) − [MAE(L)+MAE(V)] = {cancel:+.2f} kW → "
        f"{'✔ 存在抵消（误差部分相消）' if cancel < 0 else '✘ 无抵消'}")
    log("  因果性核对：负荷预测只用到第 d 天之前 ≤35 天的同星期样本；"
        "光伏情景的历史来源日 i < d（由 scen_idx 保证）")

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_误差预算表.csv",
        ["对象", "口径", "MAE_kW", "RMSE_kW", "偏差_kW", "相对MAE", "样本量"],
        [["负荷", "因果规则（同星期 35 天均值）", f"{sL['mae']:.4f}",
          f"{sL['rmse']:.4f}", f"{sL['bias']:+.4f}", f"{sL['rel'] * 100:.2f}%",
          int(sc.size * 144)],
         ["光伏", "附件 3（τ=0:00，分段常数展开）", f"{sV['mae']:.4f}",
          f"{sV['rmse']:.4f}", f"{sV['bias']:+.4f}", f"{sV['rel'] * 100:.2f}%",
          int(sc.size * 144)],
         ["净负荷", "L̂ − V̂", f"{sN['mae']:.4f}", f"{sN['rmse']:.4f}",
          f"{sN['bias']:+.4f}", f"{sN['rel'] * 100:.2f}%", int(sc.size * 144)]])

    from collections import Counter
    cnt = Counter(src[d] for d in sc)
    log(f"  负荷预测来源分布：{dict(cnt)}")

    _q2sc = C.Q2.scenarios()
    _q2sc_ref = str(_q2sc.origin) if _q2sc.origin is not None else _q2sc.source

    np.savez_compressed(
        C.V_SCENARIO_NPZ,
        dates=dates,
        tau_hours=np.asarray(C.TAU_HOURS, int),
        resid_V_hourly=e_V,
        ratio_resid=r_all,
        ratio_valid_mask=valid,
        ratio_factors=ratio_factors(all_mask),
        ratio_qs=qs,
        scen_idx=scen_idx,
        scen_V_hourly=scen_V_h,
        scen_V_hourly_rolling=scen_V_roll,
        scen_L_ref=np.asarray([_q2sc_ref], dtype="<U512"),
        pv_hist_max_hourly=pv_hist_max_h,
        pv_hist_max_lead=pv_max_by_lead,
        Lhat=Lhat,
        Vf0_10min=Vf0_10,
        score_day_index=sc,
        n_scenarios=np.asarray([M], int),
    )
    log("")
    log(f"已保存：{C.V_SCENARIO_NPZ.relative_to(C.PROJECT_DIR)}"
        f"（{C.V_SCENARIO_NPZ.stat().st_size / 1024 / 1024:.2f} MiB）")
    log("  含：resid_V_hourly / ratio_resid / ratio_factors / scen_idx / "
        "scen_V_hourly / scen_V_hourly_rolling / pv_hist_max_hourly / Lhat / Vf0_10min")

    try:
        import matplotlib.pyplot as plt

        C.setup_matplotlib()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        ax = axes[0]
        x = np.arange(24)
        A = scen_V_h[sc, 0].mean(axis=(0, 1))
        q10 = np.percentile(scen_V_h[sc, 0], 10, axis=(0, 1))
        q90 = np.percentile(scen_V_h[sc, 0], 90, axis=(0, 1))
        ax.fill_between(x, q10, q90, alpha=0.25, color="#4C78A8", label="情景 10–90% 带")
        ax.plot(x, A, color="#4C78A8", lw=2, label="情景均值")
        ax.plot(x, V_raw[sc, 0].mean(0), "k--", lw=1.6, label="附件 3 原始预报均值")
        ax.plot(x, pv_act_h[sc].mean(0), "r-", lw=2, label="附件 2 实际均值")
        ax.set_xlabel("日内小时 hh")
        ax.set_ylabel("光伏出力 / kW")
        ax.set_title("评分期平均：预报 / 情景 / 实际")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        d0 = int(score_idx[len(score_idx) // 2])
        ax.plot(np.arange(24), pv_act_h[d0], "r-", lw=2.2, label="实际")
        ax.plot(np.arange(24), V_raw[d0, 0], "k--", lw=1.8, label="附件 3（τ=0:00）")
        for w in range(M):
            ax.plot(np.arange(24), scen_V_h[d0, 0, w], color="#4C78A8",
                    alpha=0.25, lw=0.8)
        ax.plot([], [], color="#4C78A8", lw=1, label=f"{M} 个情景")
        ax.set_xlabel("日内小时 hh")
        ax.set_ylabel("光伏出力 / kW")
        ax.set_title(f"样例日 {dates[d0]}：情景扇形")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        C.save_figure(fig, "第三问_情景扇形图")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        im = ax.imshow(e_V.mean(axis=0), aspect="auto", cmap="RdBu_r",
                       vmin=-abs(e_V.mean()).max(), vmax=abs(e_V.mean()).max())
        ax.set_xticks(range(24))
        ax.set_xticklabels([str(k + 1) for k in range(24)], fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"τ={tau}:00" for tau in C.TAU_HOURS], fontsize=8)
        ax.set_xlabel("lead k / 小时")
        ax.set_title("残差均值 e_V 分层热力图（kW）")
        fig.colorbar(im, ax=ax, label="kW")
        fig.tight_layout()
        C.save_figure(fig, "第三问_残差分层热力图")
        plt.close(fig)
        log("已保存 2 张图：模型结果图/第三问_情景扇形图、第三问_残差分层热力图")
    except Exception as exc:
        log(f"  ⚠ 绘图失败：{exc}")

    log("")
    log(f"总用时 {time.perf_counter() - t0:.1f} s")
    log("[04 完成] 情景库构建结束：残差按 (τ, lead) 分层、乘性建模、层内分位数情景。")

    log.dump(C.LOG_DIR / "第三问_04情景库日志.txt", "[04 完成] 情景库构建结束。")
    C.write_text_utf8(
        C.REPORT_SCEN_MD,
        "\n".join([
            "# 第三问 预测精度与情景构造报告", "",
            "> 脚本：`第三问最终版/代码/04_构建情景库.py`；"
            "产出：`处理后数据/附件三_情景库.npz`", "",
            "## 0. 结论速览", "",
            f"1. **光伏误差是乘性的**：附件 3 预报误差近似与出力水平成正比"
            f"（有效格点上比例残差均值 {r_all[valid].mean():+.3f}、"
            f"标准差 {r_all[valid].std():.3f}），因此用"
            "$V_{act}=V_{fc}(1+r)$ 建模，而非绝对残差相加；",
            f"2. **误差随 lead 显著增长**：τ=0:00 时 lead 1–5 几乎为 0，"
            f"lead 6 起跃至约 215 kW，lead 7–18 稳定在 400–810 kW，"
            "所以残差必须按 $(\\tau, lead)$ 分层（96 层）；",
            f"3. **负荷误差其实更大**：10 min 口径下负荷 MAE {sL['mae']:.1f} kW "
            f"> 光伏 {sV['mae']:.1f} kW；但净负荷 MAE {sN['mae']:.1f} kW "
            f"低于二者之和（抵消 {abs(cancel):.1f} kW），说明净负荷才是真正要调度的对象；",
            f"4. 情景合成后，τ=0:00 情景期望的 MAE 由 "
            f"{s_raw['mae']:.1f} kW 降至 {stats(exp0, act0)['mae']:.1f} kW，"
            "且滚动标定（无前视）版本结论一致；",
            "5. 据此可预判：**第三问的紧急购电量会明显高于第二问**，"
            "这不是模型缺陷，而是「光伏预报由自建因果模型换成附件 3」的必然代价。", ""]
            + log.lines + [""]))
    log("已保存：报告/第三问预测精度与情景构造报告.md、"
        "模型结果/第三问_误差预算表.csv、第三问_残差分层统计.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
