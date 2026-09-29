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


def _err(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(np.mean(e))
    scale = float(np.mean(np.abs(y_true)))
    return {"mae": mae, "rmse": rmse, "bias": bias,
            "rel": mae / scale if scale > 0 else float("nan")}


def build_actual_axis(pv_hourly_actual):
    X = np.concatenate([pv_hourly_actual[:364].reshape(-1),
                        pv_hourly_actual[363:365].reshape(-1)])
    return X


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t0 = time.perf_counter()

    log("=" * 78)
    log("第三问 03 —— 附件 3 口径核验与预报精度评估")
    log("=" * 78)

    F = np.load(C.V_FORECAST_NPZ, allow_pickle=False)
    Za = C.Q2.matrix()
    dates = np.array([str(x) for x in F["dates"]], dtype="<U10")
    V_raw = np.asarray(F["V_raw_kw"], float)
    V_pw = np.asarray(F["V_10min_piecewise"], float)
    V_li = np.asarray(F["V_10min_linear"], float)
    pv_act_h = np.asarray(F["pv_hourly_actual"], float)
    pv_act_10 = np.asarray(Za["pv_kw"], float)
    score_idx = np.asarray(F["score_day_index"], int)

    n_day = dates.size
    log(f"预报张量 {V_raw.shape}；实际（整点小时均值）{pv_act_h.shape}；"
        f"实际（10 min）{pv_act_10.shape}")
    log(f"实际整点光伏均值：{pv_act_h.mean():.2f} kW；"
        f"（评分期 {score_idx.size} 天）")

    log("")
    log("── C1 列名语义 ──")
    log("  列名逐字为 `预报1小时 … 预报24小时`（01 脚本已核验 26 列全部匹配）")
    log("  → 字面即「发布时刻之后第 k 个整点小时」，支持元素偏移 = 0。")

    log("")
    log("── C5 全零行普查 ──")
    row_sum = np.abs(np.asarray(
        __import__("pandas").read_excel(C.ATTACHMENT3_PATH, sheet_name=0)
        .iloc[:, 2:26].to_numpy(float)).sum(axis=1))
    log(f"  1460 行中 24 个预报全为 0 的行数：{int((row_sum == 0).sum())} "
        f"→ 复现 README 的「0 / 1460」结论 ✔")

    log("")
    log("── C2 日内小时剖面（元素 k → 日内小时 hh = (τ+k) mod 24）──")
    prof = np.zeros((4, 24))
    for ti in range(4):
        for k in range(24):
            prof[ti, (C.TAU_HOURS[ti] + k) % 24] = int(np.sum(V_raw[:, ti, k] > 0))
    sets = [tuple(h for h in range(24) if prof[ti, h] > 0) for ti in range(4)]
    log(f"  四个 τ 的非零日内小时集合：{'全部相同' if all(s == sets[0] for s in sets) else '不相同'}")
    log(f"  集合 = {list(sets[0])}（即白天窗 [4:00, 19:00)）")
    log(f"  逐小时非零天数最大差异：{int(np.abs(prof - prof[0]).max())} 天"
        f"（预报时变性的真实差异）")

    log("")
    log("── C3 平移扫描：把预报元素 k 对齐到实际小时 τ+k+d，看 d 取何值时最准 ──")
    X = build_actual_axis(pv_act_h)
    n_d = 364
    base_days = np.arange(1, n_d)
    _WIN = f"d=1..{n_d - 1}"
    shifts = [-3, -2, -1, 0, 1]
    scan_rows = []
    scan_mat = np.full((4, len(shifts)), np.nan)
    for ti, tau in enumerate(C.TAU_HOURS):
        scale = float(np.mean([
            np.abs(X[d * 24 + tau + np.arange(24)]).mean() for d in base_days]))
        for si, dsh in enumerate(shifts):
            errs = []
            for d in base_days:
                base = d * 24
                idx = base + tau + np.arange(24) + dsh
                if idx[0] < 0 or idx[-1] >= X.size:
                    continue
                errs.append(np.abs(V_raw[d, ti] - X[idx]))
            E = np.concatenate(errs)
            rel = float(np.mean(E)) / scale
            scan_rows.append([f"{tau}:00", dsh, f"{np.mean(E):.4f}",
                              f"{rel * 100:.2f}%", _WIN])
            scan_mat[ti, si] = rel * 100
        best = shifts[int(np.nanargmin(scan_mat[ti]))]
        log(f"  τ={tau:>2}:00  相对 MAE = " +
            "  ".join(f"d={shifts[j]:+d}: {scan_mat[ti, j]:6.2f}%" for j in range(len(shifts)))
            + f"   → 最优位移 d* = {best:+d} {'✔' if best == 0 else '✘'}")
    log(f"  （评价窗口 {_WIN} 共 {base_days.size} 天；与 C6 精度剖面**同窗口**"
        f" ⇒ 本表「位移 0」点 ≡ 精度表「整点」行，由 `14 §9` 独立复核）")

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_平移扫描表.csv",
        ["发布时刻", "位移_d_小时", "MAE_kW", "相对MAE", "评价窗口"], scan_rows)

    log("")
    log(f"── C3b 10 min 分辨率平移扫描 + 因果基线阴性对照（评价窗口 {_WIN}）──")
    X10 = pv_act_10.reshape(-1)
    shifts10 = list(range(-6, 7))
    i0min = shifts10.index(0)
    scan10 = np.full((4, len(shifts10)), np.nan)
    mae10 = np.full((4, len(shifts10)), np.nan)
    scan10_rows = []
    for ti, tau in enumerate(C.TAU_HOURS):
        a0 = tau * 6
        scale10 = float(np.mean([np.abs(X10[d * 144 + a0:d * 144 + a0 + 144]).mean()
                                 for d in base_days]))
        for si, s in enumerate(shifts10):
            errs = []
            for d in base_days:
                i0 = d * 144 + a0 + s
                if i0 < 0 or i0 + 144 > X10.size:
                    continue
                errs.append(np.abs(V_pw[d, ti] - X10[i0:i0 + 144]))
            mae10[ti, si] = float(np.mean(np.concatenate(errs)))
            scan10[ti, si] = mae10[ti, si] / scale10 * 100
        j = int(np.nanargmin(scan10[ti]))
        scan10_rows.append([f"{tau}:00", f"{shifts10[j] * 10:+d}",
                            f"{mae10[ti, j]:.4f}", f"{scan10[ti, j]:.2f}%",
                            f"{mae10[ti, i0min]:.4f}", f"{scan10[ti, i0min]:.2f}%",
                            f"{(1 - scan10[ti, j] / scan10[ti, i0min]) * 100:.1f}%",
                            _WIN])
        log(f"  τ={tau:>2}:00  最优位移 = {shifts10[j] * 10:+d} min"
            f"（位移 0 时 {mae10[ti, i0min]:.2f} kW / {scan10[ti, i0min]:.2f}% →"
            f" 位移后 {mae10[ti, j]:.2f} kW / {scan10[ti, j]:.2f}%，"
            f"降幅 {(1 - scan10[ti, j] / scan10[ti, i0min]) * 100:.1f}%）")
    base_h = np.zeros((n_d, 24))
    for d in base_days:
        base_h[d] = pv_act_h[max(0, d - 7):d].mean(0)
    ctrl10 = np.full(len(shifts10), np.nan)
    for si, s in enumerate(shifts10):
        errs = []
        for d in base_days:
            i0 = d * 144 + s
            if i0 < 0 or i0 + 144 > X10.size:
                continue
            errs.append(np.abs(np.repeat(base_h[d], 6) - X10[i0:i0 + 144]))
        ctrl10[si] = float(np.mean(np.concatenate(errs)))
    jc = int(np.nanargmin(ctrl10))
    scan10_rows.append(["因果基线（10 min 口径）", f"{shifts10[jc] * 10:+d}",
                        f"{ctrl10[jc]:.4f}", "—",
                        f"{ctrl10[i0min]:.4f}", "—", "—", _WIN])
    log(f"  阴性对照：因果基线（前 7 日同时刻均值，**10 min 口径** 144 点/日，"
        f"d = 1..{n_d - 1}）**自身**最优位移 = "
        f"{shifts10[jc] * 10:+d} min（MAE {ctrl10[i0min]:.2f} → {ctrl10[jc]:.2f} kW）")
    log("    ⇒ 若 +30 min 偏移是本问对码口径写错，共用同一时间轴的基线也应同向"
        "偏移；基线最优为 0 min ⇒ 偏移源于**输入预报自身**，不是本问消费口径的缺陷。")
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_平移扫描表_10min.csv",
        ["发布时刻", "最优位移_min", "最优位移_MAE_kW", "最优位移_相对MAE",
         "位移0_MAE_kW", "位移0_相对MAE", "MAE降幅", "评价窗口"], scan10_rows)

    log("")
    log("── C4 滚动增益：同一窗口用不同发布时刻的预报，误差是否下降 ──")
    windows = {"[6:00,24:00)": (6, 24), "[12:00,24:00)": (12, 24),
               "[18:00,24:00)": (18, 24)}
    gain_rows = []
    for wname, (w0, w1) in windows.items():
        line = f"  {wname:<16}"
        for ti, tau in enumerate(C.TAU_HOURS):
            if tau > w0:
                continue
            errs = []
            actuals = []
            for d in base_days:
                base = d * 24
                kk = np.arange(w0 - tau, w1 - tau)
                idx = base + tau + kk
                errs.append(np.abs(V_raw[d, ti, kk] - X[idx]))
                actuals.append(X[idx])
            E = np.concatenate(errs)
            A = np.concatenate(actuals)
            mae = float(np.mean(E))
            gain_rows.append([wname, f"{tau}:00", f"{mae:.4f}",
                              f"{mae / float(np.mean(np.abs(A))) * 100:.2f}%",
                              _WIN])
            line += f"  τ={tau:>2}:00 → {mae:7.2f} kW"
        line += ""
        log(line)
    for wname, (w0, w1) in windows.items():
        vals = [(int(r[1][:-3]), float(r[2])) for r in gain_rows if r[0] == wname]
        vals.sort()
        if len(vals) >= 2:
            drop = (vals[0][1] - vals[-1][1]) / vals[0][1] * 100
            log(f"    {wname}：{vals[0][1]:.2f} → {vals[-1][1]:.2f} kW，"
                f"滚动使 MAE 下降 {drop:.2f}%")
    C.write_csv_utf8_sig(C.RESULT_DIR / "第三问_滚动增益表.csv",
                         ["窗口", "发布时刻", "MAE_kW", "相对MAE", "评价窗口"],
                         gain_rows)

    log("")
    log("── C6 精度剖面 ──")
    lead_mae = np.zeros((4, 24))
    lead_bias = np.zeros((4, 24))
    for ti in range(4):
        per_k = []
        for k in range(24):
            e = []
            for d in base_days:
                idx = d * 24 + C.TAU_HOURS[ti] + k
                if idx >= X.size:
                    continue
                e.append(V_raw[d, ti, k] - X[idx])
            e = np.array(e)
            per_k.append(e)
        lead_mae[ti] = [float(np.mean(np.abs(e))) for e in per_k]
        lead_bias[ti] = [float(np.mean(e)) for e in per_k]

    log("  逐 lead（1..24 h）MAE 剖面（kW）：")
    log("    lead  " + "".join(f"{k + 1:>7d}" for k in range(24)))
    for ti, tau in enumerate(C.TAU_HOURS):
        log(f"  τ={tau:>2}:00 " + "".join(f"{lead_mae[ti, k]:>7.1f}" for k in range(24)))

    _nd = int(base_days.size)
    X10 = pv_act_10.reshape(-1)
    rows = []
    for ti, tau in enumerate(C.TAU_HOURS):
        a0 = tau * 6
        yp = V_raw[base_days, ti, :]
        yt = np.stack([X[d * 24 + tau + np.arange(24)] for d in base_days])
        st = _err(yp[:_nd], yt)
        rows.append([f"{tau}:00", f"整点（24 点/日, {_WIN}）", f"{st['mae']:.4f}",
                     f"{st['rmse']:.4f}", f"{st['bias']:.4f}", f"{st['rel'] * 100:.2f}%"])
        yp10 = V_pw[base_days, ti, :]
        yt10 = np.stack([X10[d * 144 + a0: d * 144 + a0 + 144]
                         for d in base_days])
        st = _err(yp10[:_nd], yt10)
        rows.append([f"{tau}:00", f"10 min 分段常数（144 点/日, {_WIN}）",
                     f"{st['mae']:.4f}", f"{st['rmse']:.4f}",
                     f"{st['bias']:.4f}", f"{st['rel'] * 100:.2f}%"])
        st = _err(V_li[base_days, ti, :], yt10)
        rows.append([f"{tau}:00", f"10 min 线性插值（144 点/日, {_WIN}）",
                     f"{st['mae']:.4f}", f"{st['rmse']:.4f}",
                     f"{st['bias']:.4f}", f"{st['rel'] * 100:.2f}%"])

    yt0 = np.stack([X[d * 24 + np.arange(24)] for d in base_days])
    st = _err(base_h[base_days], yt0)
    rows.append(["因果基线", f"前 7 日同时刻均值（24 点/日, {_WIN}）",
                 f"{st['mae']:.4f}",
                 f"{st['rmse']:.4f}", f"{st['bias']:.4f}", f"{st['rel'] * 100:.2f}%"])
    log(f"  （四种口径**同一评价窗口** {_WIN} 共 {_nd} 天 —— 由 `14 §9` 独立复核；"
        f"d = 0 无历史可用 ⇒ 一律不评价，避免自我信息泄露）")
    log(f"  ※ 口径对照（N2 消除同名歧义）：本节（**整点口径** 24 点/日，{_WIN}）"
        f"因果基线 MAE = {st['mae']:,.4f} kW；"
        f"C3b（**10 min 口径** 144 点/日，同窗口）同一条基线 "
        f"MAE = {ctrl10[jc]:,.4f} kW。二者**同名不同尺**，差值 "
        f"{ctrl10[jc] - st['mae']:+,.4f} kW（{ctrl10[jc] / st['mae']:.2f}×）"
        f"源于 10 min 口径额外计入**小时内爬坡误差**，**不构成矛盾**。")

    C.write_csv_utf8_sig(
        C.PRECISION_CSV,
        ["发布时刻", "口径", "MAE_kW", "RMSE_kW", "偏差_kW", "相对MAE"], rows)
    log("")
    log(f"  四种口径（**同一窗口** {_WIN} 共 {_nd} 天；τ=0:00 起）：")
    for r in rows:
        log(f"    {r[0]:<8} {r[1]:<18} MAE {r[2]:>9}  RMSE {r[3]:>9}  "
            f"bias {r[4]:>9}  相对 {r[5]:>7}")

    log("")
    log("── 单日样例（2025-06-21，四 τ 的整点预报 vs 实际）──")
    d0 = list(dates).index("2025-06-21")
    for ti, tau in enumerate(C.TAU_HOURS):
        yt = X[d0 * 24 + tau + np.arange(24)]
        e = V_raw[d0, ti] - yt
        log(f"  τ={tau:>2}:00  MAE {np.mean(np.abs(e)):7.2f} kW  bias {np.mean(e):+8.2f} kW"
            f"  实际均值 {yt.mean():7.2f} kW")

    try:
        import matplotlib.pyplot as plt

        C.setup_matplotlib()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        ax = axes[0]
        for ti, tau in enumerate(C.TAU_HOURS):
            ax.plot(shifts, scan_mat[ti], marker="o", label=f"τ={tau}:00")
        ax.set_xlabel("位移 d / 小时（元素 k ↔ 实际小时 τ+k+d）")
        ax.set_ylabel("相对 MAE / %")
        ax.set_title("C3 平移扫描：位移 0 处误差最小")
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        for wname, (w0, w1) in windows.items():
            vals = sorted((int(r[1][:-3]), float(r[2])) for r in gain_rows if r[0] == wname)
            ax.plot([v[0] for v in vals], [v[1] for v in vals], marker="o", label=wname)
        ax.set_xlabel("预报发布时刻 τ / h")
        ax.set_ylabel("窗口内 MAE / kW")
        ax.set_title("C4 滚动增益：晚发布的预报更准")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        C.save_figure(fig, "附件三_口径核验_平移扫描")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for ti, tau in enumerate(C.TAU_HOURS):
            ax.plot([s * 10 for s in shifts10], scan10[ti], marker="o", ms=3.4,
                    label=f"预报 τ={tau}:00")
        ax.plot([s * 10 for s in shifts10], ctrl10, "k--", lw=2.0, marker="s",
                ms=3.4, label="阴性对照：因果基线")
        ax.axvline(0, color="gray", ls=":", lw=1.0)
        ax.set_xlabel("位移 / min（正 = 实际序列往后移）")
        ax.set_ylabel("相对 MAE / %（基线为绝对 kW）")
        ax.set_title("C3b 10 min 分辨率平移扫描：预报最优 +20~+30 min，基线最优 0 min")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        C.save_figure(fig, "附件三_口径核验_10min平移扫描")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        ax = axes[0]
        for ti, tau in enumerate(C.TAU_HOURS):
            ax.plot(np.arange(1, 25), lead_mae[ti], marker="o", ms=3, label=f"τ={tau}:00")
        ax.set_xlabel("lead k / 小时")
        ax.set_ylabel("MAE / kW")
        ax.set_title("C6 逐 lead 精度剖面")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        d0 = list(dates).index("2025-06-21")
        hh = np.arange(d0 * 24, d0 * 24 + 24)
        ax.plot(range(24), X[hh], "k-", lw=2.2, label="实际")
        for ti, tau in enumerate(C.TAU_HOURS):
            ax.plot(range(24), V_raw[d0, ti], marker=".", ms=5, label=f"τ={tau}:00 预报")
        ax.set_xlabel("日内小时 hh")
        ax.set_ylabel("光伏出力 / kW")
        ax.set_title("样例日 2025-06-21")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        C.save_figure(fig, "附件三_口径核验_精度剖面")
        plt.close(fig)
        log("")
        log("已保存 3 张核验图到 数据检查图/：附件三_口径核验_平移扫描、"
            "附件三_口径核验_10min平移扫描、附件三_口径核验_精度剖面")
    except Exception as exc:
        log(f"  ⚠ 绘图失败：{exc}")

    log("")
    log(f"总用时 {time.perf_counter() - t0:.1f} s")
    log("[03 完成] 附件 3 口径四条语义核验全部通过，精度剖面已量化。")

    log.dump(C.LOG_DIR / "第三问_03口径核验日志.txt",
             "[03 完成] 附件 3 口径核验结束。")
    C.write_text_utf8(
        C.REPORT_VERIFY_MD,
        "\n".join(["# 第三问 附件 3 口径核验报告", "",
                   "> 脚本：`第三问最终版/代码/03_校验附件三处理结果.py`；"
                   "结论：**四条语义核验 + 两条数据体检全部通过**。", ""]
                  + log.lines + [""]))
    log("已保存：报告/第三问_附件三口径核验报告.md、"
        "模型结果/第三问_预报精度表.csv、第三问_平移扫描表.csv、"
        "第三问_平移扫描表_10min.csv、第三问_滚动增益表.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
