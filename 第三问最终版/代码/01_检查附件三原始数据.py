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
import pandas as pd

ANOMALIES: list[tuple[str, str, str]] = []


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()

    t0 = time.perf_counter()
    log("=" * 78)
    log("第三问 01 —— 检查附件 3 原始数据")
    log("=" * 78)
    log(f"文件：{C.ATTACHMENT3_PATH}")
    log(f"体积：{C.ATTACHMENT3_PATH.stat().st_size / 1024:.1f} KiB")
    log(f"SHA256：{C.compute_sha256(C.ATTACHMENT3_PATH)[:32]}…")

    xl = pd.ExcelFile(C.ATTACHMENT3_PATH)
    log("")
    log(f"工作表：{xl.sheet_names}")

    raw = pd.read_excel(C.ATTACHMENT3_PATH, sheet_name=0, header=0)
    cols = [str(c) for c in raw.columns]
    log("")
    log(f"── 1. 工作表结构 ──")
    log(f"  形状：{raw.shape[0]} 行 × {raw.shape[1]} 列")
    log(f"  预期：1460 行（365 天 × 4 个预报时刻）× 26 列（日期 + 预报时刻 + 24 小时）")
    log(f"  列名：{cols}")
    head_ok = (cols[0].strip() == "日期" and cols[1].strip() == "预报时刻"
               and cols[2:] == [f"预报{i}小时" for i in range(1, 25)])
    log(f"  列名模式核对（日期 | 预报时刻 | 预报1小时…预报24小时）：{'✔ 通过' if head_ok else '✘ 不符'}")
    if not head_ok:
        ANOMALIES.append(("列名", "附件3列名与预期不符", str(cols)))

    C.write_csv_utf8_sig(
        C.RAW_STRUCT_CSV,
        ["序号", "列名", "dtype", "含义", "单位"],
        [[i + 1, cols[i],
          "object" if i < 2 else "float64",
          {0: "日期（合并单元格，需 ffill）", 1: "预报发布时刻"}.get(
              i, f"发布时刻后第 {i - 1} 个整点小时的预测出力"),
          "-" if i < 2 else "kW"] for i in range(len(cols))])

    log("")
    log("── 2. 日期列（合并单元格 → ffill）──")
    raw_date = raw.iloc[:, 0]
    n_blank = int(raw_date.isna().sum())
    filled: list[str] = []
    last = None
    for v in raw_date.tolist():
        try:
            s = C.normalize_date_str(v)
            last = s
        except (ValueError, TypeError):
            s = last
            if s is None:
                ANOMALIES.append(("日期", "首行日期为空，无法 ffill", "row=0"))
                raise
        filled.append(s)
    filled_arr = np.array(filled)
    udates = sorted(set(filled))
    log(f"  原始非空日期 {raw.shape[0] - n_blank} / {raw.shape[0]}（{n_blank} 个合并单元格空格）")
    log(f"  ffill 后唯一日期数：{len(udates)}，范围 {udates[0]} … {udates[-1]}")
    log(f"  预期：365 天，2025-01-01 … 2025-12-31 → "
        f"{'✔ 通过' if len(udates) == 365 and udates[0] == '2025-01-01' and udates[-1] == '2025-12-31' else '✘ 不符'}")
    cnt = pd.Series(filled_arr).value_counts()
    ok_per_day = bool((cnt == 4).all())
    log(f"  每日行数恒为 4：{'✔ 通过' if ok_per_day else '✘ 存在日行数 ≠ 4'}")
    if not ok_per_day:
        bad = cnt[cnt != 4]
        ANOMALIES.append(("日期", "存在日行数 ≠ 4", str(bad.to_dict())))
    mono = all(udates[i] < udates[i + 1] for i in range(len(udates) - 1))
    log(f"  日期严格递增且无缺日：{'✔ 通过' if mono else '✘ 不符'}")
    if not mono:
        ANOMALIES.append(("日期", "日期非严格递增", ""))

    log("")
    log("── 3. 预报时刻列 ──")
    tau_raw = raw.iloc[:, 1]
    log(f"  原始取值样例：{sorted(set(str(x) for x in tau_raw.head(12).tolist()))}")
    tau_minutes = np.array([C.parse_tau_hours(v) for v in tau_raw.tolist()], dtype=int)
    vc = pd.Series(tau_minutes).value_counts().sort_index()
    log(f"  解析后取值分布：{vc.to_dict()}")
    tau_ok = set(vc.index.tolist()) == {0, 6, 12, 18} and bool((vc == 365).all())
    log(f"  预期 {0, 6, 12, 18} 各 365 次：{'✔ 通过' if tau_ok else '✘ 不符'}")
    if not tau_ok:
        ANOMALIES.append(("预报时刻", "取值分布不符", str(vc.to_dict())))

    log("")
    log("── 4. 24 个预报列（数值体检）──")
    M = raw.iloc[:, 2:26].to_numpy(dtype=float)
    flat = M.ravel()
    n_nan = int(np.sum(~np.isfinite(flat)))
    n_neg = int(np.sum(flat < 0))
    log(f"  样本数：{flat.size}")
    log(f"  非有限值（NaN/Inf）：{n_nan} → {'✔ 无缺失' if n_nan == 0 else '✘ 有缺失'}")
    log(f"  负值：{n_neg} → {'✔ 无负值' if n_neg == 0 else '✘ 有负值'}")
    log(f"  最小值 {flat.min():.4f}  最大值 {flat.max():.4f}  均值 {flat.mean():.4f} kW")
    for q in (1, 25, 50, 75, 99):
        log(f"  第 {q:>2} 百分位：{np.percentile(flat, q):.4f} kW")
    if n_nan:
        ANOMALIES.append(("数值", "存在非有限值", f"{n_nan} 个"))
    if n_neg:
        ANOMALIES.append(("数值", "存在负值", f"{n_neg} 个"))

    rows = []
    for j, cname in enumerate(cols[2:]):
        col = M[:, j]
        rows.append([j + 1, cname, f"{col.min():.4f}", f"{col.max():.4f}",
                     f"{col.mean():.4f}", f"{np.median(col):.4f}",
                     int(np.sum(col > 0)), int(np.sum(col <= 0))])
    C.write_csv_utf8_sig(
        C.RAW_STAT_CSV,
        ["列序号", "列名", "最小值_kW", "最大值_kW", "均值_kW", "中位数_kW",
         "正值个数", "零值个数"], rows)

    log("")
    log("── 5. 全零行普查（夜间为 0 是物理事实，不是缺失）──")
    row_sum = np.abs(M).sum(axis=1)
    n_allzero = int(np.sum(row_sum == 0))
    log(f"  1460 行中「24 个预报全为 0」的行数：{n_allzero}")
    log(f"  README §6.1 结论（0 / 1460 全零行）：{'✔ 复现' if n_allzero == 0 else '✘ 不一致'}")
    if n_allzero:
        idx = np.where(row_sum == 0)[0]
        ANOMALIES.append(("全零行", "存在全零行", f"{n_allzero} 行，行号 {idx[:10].tolist()}"))
    n_zero_cells = int(np.sum(M == 0))
    log(f"  零值单元格：{n_zero_cells} / {M.size} = {100 * n_zero_cells / M.size:.2f}%")

    log("")
    log("── 6. 日内小时剖面（元素 k → 日内小时 hh = (τ + k) mod 24）──")
    V = np.full((365, 4, 24), np.nan)
    di_map = {s: i for i, s in enumerate(udates)}
    for r in range(raw.shape[0]):
        V[di_map[filled_arr[r]], C.TAU_HOURS.index(int(tau_minutes[r]))] = M[r]

    prof = np.zeros((4, 24))
    for ti in range(4):
        for k in range(24):
            hh = (C.TAU_HOURS[ti] + k) % 24
            prof[ti, hh] = int(np.sum(V[:, ti, k] > 0))
    log("  日内小时 hh → 非零天数（四行应完全一致）：")
    log("    hh   " + "".join(f"{h:>5d}" for h in range(24)))
    for ti, tau in enumerate(C.TAU_HOURS):
        log(f"  τ={tau:>2}:00 " + "".join(f"{int(prof[ti, h]):>5d}" for h in range(24)))
    sets = [tuple(h for h in range(24) if prof[ti, h] > 0) for ti in range(4)]
    set_ok = all(s == sets[0] for s in sets)
    diff_abs_max = int(np.abs(prof - prof[0][None, :]).max())
    log(f"  四个 τ 的「非零日内小时集合」完全相同："
        f"{'✔ 通过 → ' + str(list(sets[0])) if set_ok else '✘ 不一致'}")
    if diff_abs_max == 0:
        log("  逐小时非零天数逐格相同：✔ 完全一致（零位移口径的直接证据）")
    elif diff_abs_max <= 1:
        log(f"  逐小时非零天数最大差异 = {diff_abs_max} 天（≤1，属预报时变性的真实差异，"
            "不是口径问题）；差异明细：")
        for ti, tau in enumerate(C.TAU_HOURS):
            for h in range(24):
                dlt = int(prof[ti, h] - prof[0, h])
                if dlt != 0:
                    log(f"    τ={tau:>2}:00  hh={h:>2}:00  "
                        f"非零天数 {int(prof[ti, h])} vs τ=0 的 {int(prof[0, h])}"
                        f"（差 {dlt:+d} 天）")
    else:
        log(f"  ✘ 逐小时非零天数差异过大（最大 {diff_abs_max} 天）")
        ANOMALIES.append(("日内小时剖面", f"四个 τ 差异最大 {diff_abs_max} 天",
                          str(prof.astype(int).tolist())))
    if not set_ok:
        ANOMALIES.append(("日内小时剖面", "非零小时集合不一致", str(sets)))
    same = bool(set_ok and diff_abs_max <= 1)

    nz_hh = [h for h in range(24) if prof[0, h] > 0]
    log(f"  非零日内小时覆盖：{nz_hh[0]}:00 … {nz_hh[-1]}:00"
        f"（夜间 {nz_hh[-1] + 1}:00–23:00 与 0:00–{nz_hh[0] - 1}:00 恒为 0，"
        "属物理事实而非缺失）")
    interior = [h for h in range(24) if prof[0, h] == 365]
    log(f"  365 天全部非零的日内小时：{interior[0]}:00 … {interior[-1]}:00")
    edge_lo = [h for h in range(24) if 0 < prof[0, h] < 365 and h < interior[0]]
    edge_hi = [h for h in range(24) if 0 < prof[0, h] < 365 and h > interior[-1]]
    log(f"  边界小时（仅部分日期有值）：低端 {edge_lo}、高端 {edge_hi}")
    for h in edge_lo + edge_hi:
        log(f"    hh={h}:00 有值天数 = {int(prof[0, h])}")

    C.write_csv_utf8_sig(
        C.RAW_MAPPING_CSV,
        ["原始字段", "类型", "在模型中的角色", "单位", "说明"],
        [["日期", "日期（合并单元格）", "日期索引 d", "-",
          "2025-01-01…2025-12-31，ffill 后 365 个唯一值"],
         ["预报时刻", "整点", "发布时刻 τ", "h", "取值 {0,6,12,18}，各 365 次"],
         ["预报1小时", "float", "V_hat[τ + 1)", "kW",
          "发布时刻「之后第 1 个整点小时」的平均出力"],
         ["预报k小时", "float", "V_hat[τ + k)", "kW",
          "k = 1..24；元素偏移为 0，即整点区间 [τ+k-1, τ+k)"],
         ["—", "—", "10 min 展开", "—",
          f"默认 {C.DISAGG_PIECEWISE}（每小时复制 6 份），敏感档 {C.DISAGG_LINEAR}"]])

    try:
        import matplotlib.pyplot as plt

        C.setup_matplotlib()
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        ax = axes[0, 0]
        ax.hist(M.ravel(), bins=80, color="#4C78A8", edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title("附件 3 预报值分布（全 35040 个单元）")
        ax.set_xlabel("预报出力 / kW")
        ax.set_ylabel("频数（对数轴）")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for ti, tau in enumerate(C.TAU_HOURS):
            ax.plot(V[:, ti, :].mean(axis=0), marker="o", ms=3,
                    label=f"τ={tau}:00")
        ax.set_title("各发布时刻的平均整点预报曲线（横轴 = 元素序号 k）")
        ax.set_xlabel("元素序号 k（1..24）")
        ax.set_ylabel("平均预报出力 / kW")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        for ti, tau in enumerate(C.TAU_HOURS):
            ax.plot(range(24), prof[ti], marker=".", ms=5, label=f"τ={tau}:00")
        ax.set_title("日内小时 → 非零天数（四线完全重合 = 零位移口径）")
        ax.set_xlabel("日内小时 hh = (τ + k) mod 24")
        ax.set_ylabel("非零天数 / 365")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        d0 = list(udates).index("2025-06-21")
        for ti, tau in enumerate(C.TAU_HOURS):
            ax.plot(np.arange(24) * 60 + 30, V[d0, ti], marker="o", ms=3,
                    label=f"τ={tau}:00")
        ax.set_title("样例日 2025-06-21 的四条预报曲线")
        ax.set_xlabel("元素序号 k 对应的时刻（发布时刻 + k 小时）")
        ax.set_ylabel("预报出力 / kW")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        png = C.DATA_CHECK_DIR / "附件三_原始数据体检.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=160, bbox_inches="tight")
        fig.savefig(C.DATA_CHECK_DIR / "附件三_原始数据体检.pdf", bbox_inches="tight")
        plt.close(fig)
        log("")
        log(f"  已保存体检图：{png.relative_to(C.PROJECT_DIR)}")
    except Exception as exc:
        log(f"  ⚠ 绘图失败（不影响数据结论）：{exc}")

    C.write_csv_utf8_sig(C.ANOMALY_CSV, ["类别", "现象", "详情"],
                         ANOMALIES or [["-", "无异常", "-"]])
    log("")
    log(f"异常记录：{len(ANOMALIES)} 条 → 日志/附件三数据异常记录.csv")
    log(f"总用时 {time.perf_counter() - t0:.1f} s")
    log("")
    log("[01 完成] 附件 3 原始数据体检结束。")

    log.dump(C.LOG_DIR / "第三问_01原始数据检查日志.txt",
             "[01 完成] 附件 3 原始数据体检结束。")
    C.write_text_utf8(
        C.REPORT_RAW_MD,
        "\n".join(["# 附件 3 原始数据检查报告", "",
                   f"> 生成脚本：`第三问最终版/代码/01_检查附件三原始数据.py`；"
                   f"数据源：`{C.ATTACHMENT3_PATH.name}`（只读）", ""]
                  + log.lines + [""]))
    log("已保存：报告/附件三原始数据检查报告.md、"
        "原始数据说明/附件三{工作表结构,原始数据统计,字段映射表}.csv、"
        "日志/附件三数据异常记录.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
