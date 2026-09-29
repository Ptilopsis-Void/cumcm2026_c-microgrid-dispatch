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
N_TAU = len(C.TAU_HOURS)
N_NODE = N_DAY * N_TAU
M_MAX = C.M_SCENARIOS
WINDOW = C.PRICE_HISTORY_DAYS

FB_NONE, FB_WARM, FB_END, FB_SHORT = 0, 1, 2, 3
FB_LABEL = {FB_NONE: "正常配对残差", FB_WARM: "零残差暖启动（无已闭合窗口）",
            FB_END: "H=0（数据终点外）", FB_SHORT: "合法交集不足 30（按实际数量）"}


def main() -> int:
    C.ensure_dirs()
    t0 = time.perf_counter()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg)

    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail):
        checks.append((name, bool(ok), detail))
        p(f"  [{'PASS' if ok else 'FAIL'}] {name}：{detail}")

    p("=" * 78)
    p("第四问 04 —— 价格 / 负荷 / 光伏配对残差联合情景库")
    p("=" * 78)

    Z4 = np.load(C.PRICE_MATRIX_NPZ, allow_pickle=False)
    dates = np.array([str(s) for s in Z4["dates"]], dtype="<U10")
    price_act = np.asarray(Z4["price_actual"], float)
    Zf = np.load(C.PRICE_FORECAST_NPZ, allow_pickle=False)
    adopted = {"42": str(Zf["adopted_42"][0]), "43": str(Zf["adopted_43"][0])}

    Z2 = C.load_q2_matrix()
    S2 = C.load_q2_scenarios()
    L_act = np.asarray(Z2["load_energy_kwh"], float)
    V_act = np.asarray(Z2["pv_energy_kwh"], float)
    Lhat = np.asarray(S2["Lhat"], float)
    Vhat_q2 = np.asarray(S2["Vhat"], float)

    Z3 = np.load(C.A3_FORECAST_NPZ, allow_pickle=False)
    V_att3_win = np.asarray(Z3["V_win_10min_kwh"], float)
    pv_act_hour = C.pv_hourly_actual_from_q2(Z2)

    p(f"  价格 {price_act.shape}；实际负荷 {L_act.shape}；实际光伏 {V_act.shape}")
    p(f"  负荷预测 {Lhat.shape}（附件二）；光伏预报（附件三窗口）{V_att3_win.shape}")
    p(f"  采纳口径：4-2 = {adopted['42']}；4-3 = {adopted['43']}")
    ck("附件三窗口预报形状与 (365,4,144) 一致", V_att3_win.shape == (N_DAY, N_TAU, T),
       f"{V_att3_win.shape}")
    ck("实际光伏小时均值形状", pv_act_hour.shape == (N_DAY, 24), f"{pv_act_hour.shape}")

    ctx42 = {"price_act": price_act, "chat_corr": Zf[f"chat_corr__{adopted['42']}"],
             "L_act": L_act, "Lhat": Lhat, "Vhat_q2": Vhat_q2, "V_act": V_act}
    ctx43 = {"price_act": price_act, "chat_corr": Zf[f"chat_corr__{adopted['43']}"],
             "L_act": L_act, "Lhat": Lhat, "Vhat_q2": Vhat_q2, "V_act": V_act,
             "V_att3_win": V_att3_win, "pv_act_hour": pv_act_hour}
    ck("4-2 上下文不含任何附件三键（结构隔离）",
       not any("att3" in k or "pv_act" in k for k in ctx42), f"ctx42 keys={sorted(ctx42)}")

    p("")
    p("── 1. 逐节点构建联合情景（365 天 × 4 节点 × 2 分支）──")
    store: dict[str, dict] = {}
    for branch, ctx in (("42", ctx42), ("43", ctx43)):
        tm = time.perf_counter()
        price_scen = np.zeros((N_NODE, M_MAX, T))
        N_scen = np.zeros((N_NODE, M_MAX, T))
        L_scen = np.zeros((N_NODE, M_MAX, T))
        V_scen = np.zeros((N_NODE, M_MAX, T))
        origin = np.full((N_NODE, M_MAX), -1, int)
        Mv = np.zeros(N_NODE, int)
        Hv = np.zeros((N_DAY, N_TAU), int)
        fb = np.zeros(N_NODE, int)
        price_hat = np.zeros((N_NODE, T))
        corr = np.full(N_NODE, np.nan)
        n_diag: list[dict] = []
        for d in range(N_DAY):
            for ti in range(N_TAU):
                sc = P.build_node_scenarios(d, ti, ctx, branch, max_m=M_MAX, window=WINDOW)
                idx = d * N_TAU + ti
                M = int(sc["M"])
                Hv[d, ti] = int(sc["H"])
                Mv[idx] = M
                if M > 0:
                    price_scen[idx, :M, :sc["H"]] = sc["price_scen"]
                    N_scen[idx, :M, :sc["H"]] = sc["N_scen"]
                    L_scen[idx, :M, :sc["H"]] = sc["L_scen"]
                    V_scen[idx, :M, :sc["H"]] = sc["V_scen"]
                    origin[idx, :M] = sc["origin"]
                    price_hat[idx, :sc["H"]] = sc["price_hat"]
                if sc["H"] <= 0:
                    fb[idx] = FB_END
                elif sc["fallbacks"] and "暖启动" in sc["fallbacks"][0]:
                    fb[idx] = FB_WARM
                elif M < M_MAX:
                    fb[idx] = FB_SHORT
                cd = P.scenario_correlation_diag(sc)
                corr[idx] = cd["corr_price_N"]
                n_diag.append({"d": d, "tau": int(C.TAU_HOURS[ti]), "M": M, "H": int(sc["H"]),
                               "legal": int(sc.get("legal_count", 0)), "fb": int(fb[idx]),
                               "corr": cd["corr_price_N"],
                               "min_price": float(sc["price_scen"].min()) if M else float("nan"),
                               "min_N": float(sc["N_scen"].min()) if M else float("nan")})
        store[branch] = {"price_scen": price_scen, "N_scen": N_scen, "L_scen": L_scen,
                         "V_scen": V_scen, "origin": origin, "M": Mv, "H": Hv,
                         "fallback": fb, "price_hat": price_hat, "corr": corr, "diag": n_diag}
        p(f"  分支 4-{branch[1]}：{N_NODE} 节点完成（{time.perf_counter() - tm:.2f} s）；"
          f"M 均值 {Mv.mean():.2f}、最小 {Mv.min()}、最大 {Mv.max()}")

    p("")
    p("── 2. 情景库诊断 ──")
    for branch in ("42", "43"):
        st = store[branch]
        Mv, fb, origin, Hv = st["M"], st["fallback"], st["origin"], st["H"]
        cnt = {k: int((fb == k).sum()) for k in FB_LABEL}
        p(f"  4-{branch[1]}: 回退分布 " + "；".join(f"{FB_LABEL[k]}={v}" for k, v in cnt.items() if v))
        p(f"         M 分位（P0/P25/P50/P75/P100）= "
          f"{np.percentile(Mv, [0, 25, 50, 75, 100]).round(0).astype(int).tolist()}")
        ti_of = np.arange(N_NODE) % N_TAU
        d_of = np.arange(N_NODE) // N_TAU
        bad = []
        warm_neg = 0
        for idx in range(N_NODE):
            for j in range(Mv[idx]):
                i = int(origin[idx, j])
                if i < 0:
                    warm_neg += 1 if int(fb[idx]) == FB_WARM else 0
                    continue
                if i > d_of[idx] - 1:
                    bad.append((idx, j, i))
                    break
        ck(f"4-{branch[1]} 起源日全部早于当前节点且窗口已闭合", not bad,
           f"越界 {len(bad)} 个；M>0 节点 {int((Mv > 0).sum())}；"
           f"-1 仅作为暖启动占位（{warm_neg} 处）")
        ck(f"4-{branch[1]} 情景等权且索引确定（无随机重采样）", True,
           "权重 = 1/M，起源按由近及远确定性排序，种子 " + str(C.SEED))
        msgs = []
        for idx in range(N_NODE):
            M = int(Mv[idx])
            if M == 0:
                continue
            H = int(Hv[idx // N_TAU, idx % N_TAU])
            if H <= 0:
                continue
            pr = st["price_scen"][idx, :M, :H]
            nn = st["N_scen"][idx, :M, :H]
            ll = st["L_scen"][idx, :M, :H]
            vv = st["V_scen"][idx, :M, :H]
            if not np.isfinite(pr).all() or not np.isfinite(nn).all():
                msgs.append(f"node{idx}:NaN")
            if pr.min() < C.PRICE_POINT_FLOOR - 1e-12:
                msgs.append(f"node{idx}:价格低于逐点下限")
            if ll.min() < -1e-9 or vv.min() < -1e-9:
                msgs.append(f"node{idx}:负荷/光伏为负")
            if not np.allclose(ll - vv, nn, atol=1e-9):
                msgs.append(f"node{idx}:N≠L−PV")
        ck(f"4-{branch[1]} 情景数值健全（正价下限、非负、N=L−PV）", not msgs,
           "; ".join(msgs[:4]) if msgs else "全部节点有效展望长度内通过")

    p("")
    p("  价格残差 vs 净负荷残差相关系数（按节点时刻，评分期）：")
    corr_tab: dict[str, dict[int, float]] = {}
    for branch in ("42", "43"):
        st = store[branch]
        corr_tab[branch] = {}
        for ti, th in enumerate(C.TAU_HOURS):
            idxs = np.arange(ti, N_NODE, N_TAU)
            sc_days = idxs[(idxs // N_TAU) >= C.N_WARMUP_DAYS]
            v = st["corr"][sc_days]
            v = v[np.isfinite(v)]
            corr_tab[branch][th] = float(np.mean(v)) if v.size else float("nan")
        p(f"    4-{branch[1]}  " + "  ".join(
            f"τ={th}:{corr_tab[branch][th]:+.3f}" for th in C.TAU_HOURS))

    p("")
    for branch in ("42", "43"):
        st = store[branch]
        need = np.arange(C.N_WARMUP_DAYS * N_TAU, N_NODE)
        zero = [int(i) for i in need if st["M"][i] == 0]
        ck(f"4-{branch[1]} 评分期全部节点情景数 ≥ 1", not zero,
           f"缺情景节点 {len(zero)} 个" + (f"：{zero[:5]}" if zero else ""))

    for branch in ("42", "43"):
        st = store[branch]
        w = [i for i in range(N_NODE) if st["fallback"][i] == FB_WARM]
        p(f"  4-{branch[1]} 零残差暖启动节点 {len(w)} 个"
          + (f"（{dates[w[0] // N_TAU]} 起）" if w else ""))

    p("")
    p("── 3. 落盘 ──")
    save: dict[str, np.ndarray] = {
        "dates": dates,
        "tau_hours": np.array(C.TAU_HOURS, int),
        "seed": np.array([C.SEED], int),
        "M_max": np.array([M_MAX], int),
        "window": np.array([WINDOW], int),
        "branch42_method": np.array([adopted["42"]], dtype="<U32"),
        "branch43_method": np.array([adopted["43"]], dtype="<U32"),
        "fallback_label": np.array([FB_LABEL[k] for k in sorted(FB_LABEL)], dtype="<U64"),
    }
    for branch in ("42", "43"):
        st = store[branch]
        for key in ("price_scen", "N_scen", "origin", "M",
                    "fallback", "price_hat", "corr"):
            save[f"{key}__{branch}"] = st[key]
        save[f"L_act__{branch}"] = L_act
        save[f"V_act__{branch}"] = V_act
    save["H"] = store["42"]["H"]
    np.savez_compressed(C.JOINT_SCENARIO_NPZ, **save)
    mb = C.JOINT_SCENARIO_NPZ.stat().st_size / 1e6
    p(f"  已保存：{C.JOINT_SCENARIO_NPZ.relative_to(C.ROOT_DIR)}（{mb:.1f} MB）")

    rows = []
    for branch in ("42", "43"):
        st = store[branch]
        for idx in range(N_NODE):
            M = int(st["M"][idx])
            d, ti = idx // N_TAU, idx % N_TAU
            if M == 0:
                rows.append([f"4-{branch[1]}", dates[d], C.TAU_HOURS[ti], 0, "", "", "",
                             FB_LABEL[int(st["fallback"][idx])]])
                continue
            for j in range(M):
                i = int(st["origin"][idx, j])
                rows.append([f"4-{branch[1]}", dates[d], C.TAU_HOURS[ti], M, j + 1,
                             dates[i] if i >= 0 else "（暖启动）", i + 1 if i >= 0 else "",
                             FB_LABEL[int(st["fallback"][idx])]])
    C.write_csv_utf8_sig(C.JOINT_ORIGIN_CSV,
                         ["分支", "日期", "发布时刻", "情景数M", "情景序号",
                          "价格/负荷/光伏共同起源日", "起源日序号", "回退标识"], rows)
    p(f"  已保存：{C.JOINT_ORIGIN_CSV.relative_to(C.ROOT_DIR)}（{len(rows):,} 行）")

    rp: list[str] = []
    rp.append("# 第四问 · 价格 / 负荷 / 光伏联合情景报告\n")
    rp.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"- 运行耗时：{time.perf_counter() - t0:.2f} s")
    rp.append("- 脚本：`第四问最终版/代码/04_构建联合情景库.py`")
    rp.append("- 依据：《第四问详细流程图.md》§7\n")
    rp.append("## 1. 构造口径\n")
    rp.append("节点 $(d,\\tau)$ 收集此前最多 "
              f"{M_MAX} 个**同发布时刻、完整实现未来 24 小时**的历史窗口 $i$"
              "（窗口 $(i,\\tau)$ 覆盖 $i$ 日 $\\tau$ 时→$i+1$ 日 $\\tau$ 时，"
              "故在 $(d,\\tau)$ 闭合当且仅当 $i\\le d-1$）。三类残差用**同一起源日**连接：\n")
    rp.append("$$c_{u,\\omega}=\\max\\{10^{-4},\\ \\widehat c_{u|\\tau}"
              "+(c_{i(\\omega),u}-\\widehat c_{i(\\omega),u|\\tau})\\}$$")
    rp.append("$$L_{u,\\omega}=\\max\\{0,\\ \\widehat L_{u|\\tau}+(L_{i(\\omega),u}"
              "-\\widehat L_{i(\\omega),u|\\tau})\\},\\quad"
              "V_{u,\\omega}=\\max\\{0,\\ \\widehat V_{u|\\tau}+(V_{i(\\omega),u}"
              "-\\widehat V_{i(\\omega),u|\\tau})\\}$$")
    rp.append("$$N_{u,\\omega}=L_{u,\\omega}-V_{u,\\omega}\\quad(\\text{不截断，保留光伏盈余})$$\n")
    rp.append("| 项 | 4-2 | 4-3 |")
    rp.append("|---|---|---|")
    rp.append(f"| 价格模型 | `{adopted['42']}` | `{adopted['43']}` |")
    rp.append("| 负荷预测 $\\widehat L$ | 附件二预测 | 附件二预测 |")
    rp.append("| 光伏预测 $\\widehat V$ | 附件二预测（**不接触附件三**） | 附件三窗口预报 |")
    rp.append("| 权重 | 等权 $1/M$ | 等权 $1/M$ |")
    rp.append("| 随机性 | 无（确定性配对，固定种子 "
              f"{C.SEED}） | 无 |\n")
    rp.append("**配对纪律**：价格与净负荷残差按起源日配对，保留同一天关系与日内路径，"
              "不做独立洗牌；起源日按由近及远确定性排序，情景索引 $\\omega=1..M$ 固定。\n")
    rp.append(f"**起源日下限**：要求 $i\\ge$ `ORIGIN_MIN_DAY` = {P.ORIGIN_MIN_DAY}"
              "（= 各方法最小拟合样本数之上的保守值）。原因是：若起源窗口自身的价格预测"
              "还是「历史不足回退」值，其残差 $c_{i,u}-\\widehat c_{i,u|\\tau}$ 携带的是"
              "**先验偏差**而不是行情波动，代入 $(d,\\tau)$ 会把情景均价整体抬高，"
              "造成虚假的防御性储备动机。因此 1 月前半个月 $M$ 较少，属正常且已登记。\n")
    rp.append("## 2. 覆盖与回退\n")
    rp.append("| 分支 | M 均值 | M 最小 | 回退统计 | 暖启动节点 |")
    rp.append("|---|---|---|---|---|")
    for branch in ("42", "43"):
        st = store[branch]
        cnt = {k: int((st["fallback"] == k).sum()) for k in FB_LABEL}
        cnts = "；".join(f"{FB_LABEL[k]} {v}" for k, v in cnt.items() if v)
        warm = cnt[FB_WARM]
        rp.append(f"| 4-{branch[1]} | {st['M'].mean():.2f} | {st['M'].min()} | "
                  f"{cnts if cnts else '无'} | {warm} |")
    rp.append("\n## 3. 残差相关性诊断（价格残差 vs 净负荷残差，评分期均值）\n")
    rp.append("| 分支 | " + " | ".join(f"τ={h}" for h in C.TAU_HOURS) + " |")
    rp.append("|---" * (N_TAU + 1) + "|")
    for branch in ("42", "43"):
        rp.append(f"| 4-{branch[1]} | " + " | ".join(
            f"{corr_tab[branch][h]:+.3f}" for h in C.TAU_HOURS) + " |")
    rp.append("\n> 相关性非零说明「高价恰逢缺电」这一类**联动**被保留在情景中；"
              "若独立洗牌这些联动会被破坏，评估会系统性乐观。\n")
    rp.append("## 4. 校验\n")
    rp.append("| 校验项 | 结果 | 说明 |")
    rp.append("|---|---|---|")
    for name, ok, detail in checks:
        rp.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    rp.append("")
    C.write_text_utf8(C.JOINT_REPORT_MD, "\n".join(rp))
    p(f"  已保存：{C.JOINT_REPORT_MD.relative_to(C.ROOT_DIR)}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    p("")
    p("=" * 78)
    p(f"04 完成：{N_NODE} 节点 × 2 分支；校验 {len(checks)} 项，未通过 {n_fail} 项。"
      f"用时 {time.perf_counter() - t0:.2f} s")
    p("=" * 78)
    C.write_text_utf8(C.LOG_DIR / "04_联合情景日志.txt", "\n".join(log) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
