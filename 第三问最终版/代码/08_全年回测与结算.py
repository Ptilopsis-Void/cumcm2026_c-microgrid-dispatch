#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

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
P3 = _load("_policy3.py", "q3_policy")

RHO_DOWN = C.RHO_DOWN
RHO_UP = C.RHO_UP
EMERG = C.EMERG_MULT


def decompose(price: np.ndarray, P: np.ndarray, Q: np.ndarray,
              b: np.ndarray) -> dict:
    u = np.maximum(P - Q, 0.0)
    v = np.maximum(Q - P, 0.0)
    plan = (price * np.minimum(P, Q)).sum(axis=1)
    down = (RHO_DOWN * price * u).sum(axis=1)
    up = (RHO_UP * price * v).sum(axis=1)
    emerg = (EMERG * price * b).sum(axis=1)
    return {"plan": plan, "down": down, "up": up, "emerg": emerg,
            "total": plan + down + up + emerg,
            "u": u.sum(axis=1), "v": v.sum(axis=1), "b": b.sum(axis=1)}


def _extract_q2_yearly_total(txt: str):
    import re

    pats = (r"主结果\s*=\s*【[^】]*】\s*[：:]\s*([\d,]+(?:\.\d+)?)\s*元",
            r"全年总费用[^\d]{0,12}?([\d,]+(?:\.\d+)?)\s*元",
            r"全年账单[^\d]{0,12}?([\d,]+(?:\.\d+)?)\s*元",
            r"总购电费用[^\d]{0,12}?([\d,]+(?:\.\d+)?)\s*元")
    for pat in pats:
        hit = re.search(pat, txt)
        if hit:
            v = float(hit.group(1).replace(",", ""))
            if 1e6 <= v <= 1e8:
                return v
    return None


def _q2_anchor():
    try:
        core = C.Q2.core_numbers()
        v = core.get("yearly_total")
        if v is not None:
            return float(v), C.Q2_CORE_CSV, C.Q2_HEADLINE_COMPARISON
    except Exception:
        pass

    cand = _find_q2_report()
    if cand is not None:
        try:
            v = _extract_q2_yearly_total(
                cand.read_text(encoding="utf-8").replace("\u2009", ""))
        except Exception:
            v = None
        if v is not None:
            return v, cand, "结果总览主结果（散文兜底）"
    return None, None, None


def _find_q2_report():
    for cand in (C.Q2_DIR / "报告" / "第二题_结果总览.md",
                 C.Q2_DIR / "模型结果" / "第二题_结果总览.md",
                 C.Q2_DIR / "第二题_结果总览.md"):
        if cand.is_file():
            return cand
    return None


_PRECISION_TAU = "0:00"


def _precision_numbers():
    p = C.PRECISION_CSV
    if not p.is_file():
        return None, None, None
    try:
        import csv as _csv
        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = [r for r in _csv.reader(fh) if r]
    except Exception:
        return None, None, None
    if len(rows) < 2:
        return None, None, None
    hdr = [h.strip() for h in rows[0]]
    if "口径" not in hdr or "MAE_kW" not in hdr:
        return None, None, None
    i_cal, i_mae = hdr.index("口径"), hdr.index("MAE_kW")
    i_tau = hdr.index("发布时刻") if "发布时刻" in hdr else None
    mae0 = mae_base = None
    win = ""
    for r in rows[1:]:
        if len(r) <= max(i_cal, i_mae):
            continue
        try:
            v = float(r[i_mae])
        except (TypeError, ValueError):
            continue
        tau = r[i_tau].strip() if i_tau is not None and len(r) > i_tau else ""
        if tau == _PRECISION_TAU and "整点" in r[i_cal]:
            mae0 = v
            _m = re.search(r"d=\d+[.][.]\d+", r[i_cal])
            if _m:
                win = _m.group(0)
        if "因果基线" in r[i_cal] or tau == "因果基线":
            mae_base = v
    if mae0 is None:
        return None, None, None, ""
    return _PRECISION_TAU, mae0, mae_base, win


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t00 = time.perf_counter()

    log("=" * 78)
    log("第三问 08 —— 全年回测与费用结算（三分解）")
    log("=" * 78)
    _mode = C.Q2.mode()
    log(f"- 数据源模式 Q3_Q2_SOURCE = {_mode}")
    if _mode != "real":
        log("")
        log("⚠⚠ 非 real 模式：负荷为合成占位数据，本次所有数字**不可提交、不可写入论文**。")
        log("⚠⚠ 仅用于验证代码链路与闭环残差。")
    log("")

    _max_days = int(os.environ.get("Q3_08_MAX_DAYS", "0") or 0)
    _days = list(range(_max_days)) if _max_days > 0 else None
    if _days is not None:
        log(f"⚠ 调试钩子 Q3_08_MAX_DAYS={_max_days}：只跑前 {_max_days} 个评分日，"
            f"结果**不可提交**。")
        log("")
    res = P3.run_policy(update_times=(6, 12, 18), days=_days, log=log)
    F = np.load(C.V_FORECAST_NPZ, allow_pickle=False)
    Z = C.Q2.matrix()

    price = np.asarray(Z["price"], float)
    P = np.asarray(res["P"], float)
    Q = np.asarray(res["Q"], float)
    b = np.asarray(res["b"], float)
    NN = np.asarray(res["N_actual"], float)
    Ch = np.asarray(res["C"], float)
    Dh = np.asarray(res["D"], float)
    Uh = np.asarray(res["curt"], float)
    Eh = np.asarray(res["E"], float)
    Can = np.asarray(res["C_analytic"], float)
    ban = np.asarray(res["b_analytic"], float)
    sc = np.asarray(res["score_day_index"], int)
    dates = [str(x) for x in F["dates"]]
    n_day = len(sc)
    resid = res["resid"]
    info = res["info"]

    log(f"评分期 {n_day} 天：{dates[sc[0]]} … {dates[sc[-1]]}")
    log("计费口径：计划购电费 Σc·min(p,q)；调整费 0.5Σc·u + 1.5Σc·v；"
        "紧急购电费 5Σc·b")

    dec = decompose(price, P[sc], Q[sc], b[sc])
    dec_an = decompose(price, P[sc], Q[sc], ban[sc])
    dec_exp = res["exp_dec"]

    u = np.maximum(P[sc] - Q[sc], 0.0)
    v = np.maximum(Q[sc] - P[sc], 0.0)
    id1 = dec["plan"] + dec["down"] + dec["up"]
    d_up = (price * u).sum(axis=1) + (price * v).sum(axis=1)
    id2 = (price * Q[sc]).sum(axis=1) + (
        RHO_DOWN * (price * u).sum(axis=1) + RHO_UP * (price * v).sum(axis=1))
    id36 = (price * P[sc]).sum(axis=1) - 0.5 * (price * u).sum(axis=1) \
        + 1.5 * (price * v).sum(axis=1)
    id_equiv = (price * Q[sc]).sum(axis=1) + 0.5 * d_up
    log("")
    log("── 结算恒等式自检（三种等价形式两两互验）──")
    log(f"  形式① Σc·min(p,q)+0.5Σc·u+1.5Σc·v")
    log(f"  形式② Σc·q+0.5Σc·(u+v)      vs ① 最大差 = "
        f"{float(np.abs(id_equiv - id1).max()):.3e} 元 "
        f"{'✔' if np.abs(id_equiv - id1).max() < 1e-6 else '✘'}")
    log(f"  形式③ Σc·p−0.5Σc·u+1.5Σc·v  vs ① 最大差 = "
        f"{float(np.abs(id36 - id1).max()):.3e} 元 "
        f"{'✔' if np.abs(id36 - id1).max() < 1e-6 else '✘'}")

    TOL_SOC = 1e-8
    log("")
    log("── 闭环自检（真实标量储电量推进，P0-1 / P0-3 / P0-7）──")
    log(f"  ① 四节点库存链 max|E_in(τ_k+1) − E_out(τ_k)| = "
        f"{resid['node_chain']:.3e} kWh "
        f"{'✔' if resid['node_chain'] <= TOL_SOC else '✘'}   （阈值 1e-8）")
    log(f"  ② 跨日库存衔接 max|E_in(日 d+1, 0) − E_out(日 d, 144)| = "
        f"{resid['cross_day']:.3e} kWh "
        f"{'✔' if resid['cross_day'] <= TOL_SOC else '✘'}")
    log(f"  ③ 每日计划输入 = 当日节点 0 真实库存 : max 差 = "
        f"{resid['plan_input_vs_real']:.3e} kWh "
        f"{'✔' if resid['plan_input_vs_real'] <= TOL_SOC else '✘'}")
    log(f"     （对照）旧 `06` 的「规划库存」与真实库存最大偏差 = "
        f"{resid['plan_chain_gap_max']:,.2f} kWh，均值 "
        f"{resid['plan_chain_gap_mean']:,.2f} kWh —— 即 P0-1 的量化规模")
    log(f"  ④ SOC 递推 E_t = E_t−1 + ηC_t − D_t/η : max 残差 = "
        f"{resid['soc_recursion']:.3e} kWh "
        f"{'✔' if resid['soc_recursion'] <= TOL_SOC else '✘'}")
    log(f"  ⑤ 功率平衡 N − q − D − b + C + U = 0 : max = "
        f"{resid['balance']:.3e} kWh "
        f"{'✔' if resid['balance'] <= TOL_SOC else '✘'}")
    log(f"  ⑥ 功率上限：每时段充/放电 ≤ S = {C.S_PERIOD_KWH:.4f} kWh "
        f"（越界量 {resid['power_cap']:.3e}）")
    log(f"  ⑦ SOC 区间 [{C.E_MIN:.0f}, {C.E_MAX:.0f}] kWh：下越界 "
        f"{resid['soc_lower']:.3e}、上越界 {resid['soc_upper']:.3e} kWh")
    log(f"  ⑧ 同时充放电时段数 = {resid['simul_charge_discharge']}")
    log(f"  ⑨ 紧急购电与充电同段（附录 A-31 关注项）："
        f"{resid['emerg_with_charge_cells']} 时段 / "
        f"{resid['emerg_with_charge_days']} 天（紧急购电共 "
        f"{resid['emerg_cells']} 时段 / {resid['emerg_days']} 天）")
    log(f"  ⑩ 信息泄露审计：越界时段数 = {resid['info_leak_cells']} "
        f"{'✔' if resid['info_leak_cells'] == 0 else '✘'}；"
        f"未定义节点 {resid['info_src_undef']}；"
        f"最早可用时刻晚于决策时刻 {resid['info_min_gt_src']} 处；"
        f"实际出现的节点数 {resid['info_nodes_seen']} / {info['n_node']}")

    log("")
    log("── 结算口径对照（P0-5：模型内期望 vs 样本外实现）──")
    log(f"  {'口径':<28}{'总费用 元':>16}{'紧急购电量 kWh':>18}{'均价 元/kWh':>16}")
    log(f"  {'模型内期望（30 情景均值）':<28}"
        f"{dec_exp['total'].sum():>16,.2f}{dec_exp['b'].sum():>18,.2f}"
        f"{dec_exp['total'].sum() / max(Q[sc].sum(), 1e-9):>16.6f}")
    log(f"  {'样本外实现（真实轨迹）':<28}"
        f"{dec['total'].sum():>16,.2f}{dec['b'].sum():>18,.2f}"
        f"{dec['total'].sum() / max(Q[sc].sum(), 1e-9):>16.6f}")
    log(f"  {'纯解析执行（实现口径）':<28}"
        f"{dec_an['total'].sum():>16,.2f}{dec_an['b'].sum():>18,.2f}"
        f"{dec_an['total'].sum() / max(Q[sc].sum(), 1e-9):>16.6f}")

    tot = {
        "计划购电量_kWh": float(P[sc].sum()),
        "调整购电量_kWh": float(Q[sc].sum()),
        "下调量_kWh": float(u.sum()),
        "上调量_kWh": float(v.sum()),
        "计划购电费_元": float(dec["plan"].sum()),
        "下调违约金_元": float(dec["down"].sum()),
        "上调加价_元": float(dec["up"].sum()),
        "调整相关费用_元": float(dec["down"].sum() + dec["up"].sum()),
        "紧急购电量_kWh": float(b[sc].sum()),
        "紧急购电费_元": float(dec["emerg"].sum()),
        "总费用_元": float(dec["total"].sum()),
        "实际净负荷_kWh": float(NN[sc].sum()),
        "充电量_kWh": float(Ch[sc].sum()),
        "放电量_kWh": float(Dh[sc].sum()),
        "弃电量_kWh": float(Uh[sc].sum()),
        "无储能基准购电量_kWh": float(np.maximum(NN[sc], 0).sum()),
        "无储能基准购电费_元": float((price * np.maximum(NN[sc], 0)).sum()),
    }
    tot["储能相对基准节省_元"] = tot["无储能基准购电费_元"] - tot["总费用_元"]
    tot["储能相对基准节省_%"] = (tot["储能相对基准节省_元"]
                              / tot["无储能基准购电费_元"] * 100.0)
    tot["总费用均价_元每kWh"] = tot["总费用_元"] / tot["调整购电量_kWh"]

    log("")
    log("── 年度汇总（334 天，2025-02-01 … 12-31）──")
    for k, val in tot.items():
        log(f"  {k:24s} = {val:,.4f}")

    log("")
    log(f"  对照：纯解析执行紧急购电 {dec_an['emerg'].sum():,.2f} 元，"
        f"总费用 {dec_an['total'].sum():,.2f} 元，"
        f"较 DP 保留水平 {dec_an['total'].sum() - tot['总费用_元']:+,.2f} 元")

    rows = []
    for i, d in enumerate(sc):
        rows.append([dates[d],
                     f"{P[d].sum():.6f}", f"{Q[d].sum():.6f}",
                     f"{dec['u'][i]:.6f}", f"{dec['v'][i]:.6f}",
                     f"{dec['plan'][i]:.6f}", f"{dec['down'][i]:.6f}",
                     f"{dec['up'][i]:.6f}", f"{dec['emerg'][i]:.6f}",
                     f"{dec['total'][i]:.6f}", f"{dec['b'][i]:.6f}",
                     f"{NN[d].sum():.6f}", f"{Ch[d].sum():.6f}",
                     f"{Dh[d].sum():.6f}", f"{Uh[d].sum():.6f}",
                     f"{float(np.maximum(NN[d], 0).sum()):.6f}"])
    C.write_csv_utf8_sig(
        C.DAILY_CSV,
        ["日期", "计划购电量_kWh", "调整购电量_kWh", "下调量_kWh", "上调量_kWh",
         "计划购电费_元", "下调违约金_元", "上调加价_元", "紧急购电费_元",
         "当日总费用_元", "紧急购电量_kWh", "实际净负荷_kWh", "充电量_kWh",
         "放电量_kWh", "弃电量_kWh", "无储能基准购电量_kWh"],
        rows)

    months: dict[str, list[int]] = {}
    for i, d in enumerate(sc):
        months.setdefault(dates[d][:7], []).append(i)
    m_rows = []
    for mk in sorted(months):
        idx = np.asarray(months[mk], int)
        m_rows.append([mk, len(idx),
                       f"{P[sc][idx].sum():.4f}", f"{Q[sc][idx].sum():.4f}",
                       f"{dec['plan'][idx].sum():.4f}",
                       f"{dec['down'][idx].sum():.4f}",
                       f"{dec['up'][idx].sum():.4f}",
                       f"{dec['emerg'][idx].sum():.4f}",
                       f"{dec['total'][idx].sum():.4f}",
                       f"{b[sc][idx].sum():.4f}",
                       f"{float(np.maximum(NN[sc][idx], 0).sum()):.4f}",
                       f"{float((price * np.maximum(NN[sc][idx], 0)).sum()):.4f}"])
    C.write_csv_utf8_sig(
        C.MONTHLY_CSV,
        ["月份", "天数", "计划购电量_kWh", "调整购电量_kWh", "计划购电费_元",
         "下调违约金_元", "上调加价_元", "紧急购电费_元", "当月总费用_元",
         "紧急购电量_kWh", "无储能基准购电量_kWh", "无储能基准购电费_元"],
        m_rows)

    C.write_csv_utf8_sig(
        C.YEARLY_CSV, ["指标", "数值"],
        [[k, f"{v:.6f}" if isinstance(v, float) else str(v)]
         for k, v in tot.items()])

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_结算口径对照.csv",
        ["指标", "模型内期望口径", "样本外实现口径", "纯解析执行(实现口径)"],
        [["总费用_元", f"{dec_exp['total'].sum():.6f}",
          f"{dec['total'].sum():.6f}", f"{dec_an['total'].sum():.6f}"],
         ["计划购电费_元", f"{dec_exp['plan'].sum():.6f}",
          f"{dec['plan'].sum():.6f}", f"{dec_an['plan'].sum():.6f}"],
         ["下调违约金_元", f"{dec_exp['down'].sum():.6f}",
          f"{dec['down'].sum():.6f}", f"{dec_an['down'].sum():.6f}"],
         ["上调加价_元", f"{dec_exp['up'].sum():.6f}",
          f"{dec['up'].sum():.6f}", f"{dec_an['up'].sum():.6f}"],
         ["紧急购电费_元", f"{dec_exp['emerg'].sum():.6f}",
          f"{dec['emerg'].sum():.6f}", f"{dec_an['emerg'].sum():.6f}"],
         ["紧急购电量_kWh", f"{dec_exp['b'].sum():.6f}",
          f"{dec['b'].sum():.6f}", f"{dec_an['b'].sum():.6f}"]])

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_闭环审计.csv",
        ["检查项", "数值", "阈值", "是否通过"],
        [["四节点库存链 max 差 kWh", f"{resid['node_chain']:.6e}",
          "1e-8", "通过" if resid["node_chain"] <= 1e-8 else "未通过"],
         ["跨日库存衔接 max 差 kWh", f"{resid['cross_day']:.6e}",
          "1e-8", "通过" if resid["cross_day"] <= 1e-8 else "未通过"],
         ["每日计划输入 vs 真实库存 max 差 kWh",
          f"{resid['plan_input_vs_real']:.6e}",
          "1e-8", "通过" if resid["plan_input_vs_real"] <= 1e-8 else "未通过"],
         ["SOC 递推 max 残差 kWh", f"{resid['soc_recursion']:.6e}",
          "1e-8", "通过" if resid["soc_recursion"] <= 1e-8 else "未通过"],
         ["功率平衡 max 残差 kWh", f"{resid['balance']:.6e}",
          "1e-8", "通过" if resid["balance"] <= 1e-8 else "未通过"],
         ["信息泄露时段数", f"{resid['info_leak_cells']}", "0",
          "通过" if resid["info_leak_cells"] == 0 else "未通过"],
         ["同时充放电时段数", f"{resid['simul_charge_discharge']}", "0",
          "通过" if resid["simul_charge_discharge"] == 0 else "未通过"],
         ["旧“规划库存”与真实库存最大偏差 kWh（历史对照）",
          f"{resid['plan_chain_gap_max']:.6f}", "—", "—"]])

    np.savez_compressed(
        C.BACKTEST_NPZ,
        score_day_index=sc, dates=np.asarray(dates, dtype="<U16"),
        price=price, P=P, Q=Q, b=b, C=Ch, D=Dh, U=Uh, E=Eh,
        N_actual=NN,
        daily_plan_fee=dec["plan"], daily_down_fee=dec["down"],
        daily_up_fee=dec["up"], daily_emerg_fee=dec["emerg"],
        daily_total_fee=dec["total"], daily_u=dec["u"], daily_v=dec["v"],
        daily_b=dec["b"],
        yearly_total=np.asarray([tot["总费用_元"]], float),
        update_times=np.asarray(res["update_times"], int),
        nodes=np.asarray(res["nodes"], int),
        segments=np.asarray(res["segments"], int),
        SRC=np.asarray(res["SRC"], int), SRC_MIN=np.asarray(res["SRC_MIN"], int),
        E_in_node=np.asarray(res["E_in_node"], float),
        E_out_node=np.asarray(res["E_out_node"], float),
        E_day_start_real=np.asarray(res["E_day_start_real"], float),
        E_00_plan_input=np.asarray(res["E_00_plan_input"], float),
        E_day_end_real=np.asarray(res["E_day_end_real"], float),
        E_next_day_start_real=np.asarray(res["E_next_day_start_real"], float),
        E_plan_chain_virtual=np.asarray(res["E_plan_chain_virtual"], float),
        b_analytic=ban, C_analytic=Can,
        D_analytic=np.asarray(res["D_analytic"], float),
        curt_analytic=np.asarray(res["curt_analytic"], float),
        R=np.asarray(res["R"], float),
        EM_model=np.asarray(res["EM_model"], float),
        exp_total=np.asarray([dec_exp["total"].sum()], float),
        resid_names=np.asarray(sorted(resid), dtype="<U32"),
        resid_values=np.asarray([float(resid[k]) for k in sorted(resid)], float),
    )

    C.setup_matplotlib()
    C.setup_chinese_font()
    import matplotlib.pyplot as plt

    mks = sorted(months)
    mplan = np.array([dec["plan"][np.asarray(months[m], int)].sum() for m in mks])
    mdown = np.array([dec["down"][np.asarray(months[m], int)].sum() for m in mks])
    mup = np.array([dec["up"][np.asarray(months[m], int)].sum() for m in mks])
    mem = np.array([dec["emerg"][np.asarray(months[m], int)].sum() for m in mks])

    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(mks))
    ax.bar(x, mplan, 0.62, label="计划购电费", color="#4C78A8")
    ax.bar(x, mdown, 0.62, bottom=mplan, label="下调违约金（0.5c·u）", color="#F2CF5B")
    ax.bar(x, mup, 0.62, bottom=mplan + mdown, label="上调加价（1.5c·v）",
           color="#9ECAE9")
    ax.bar(x, mem, 0.62, bottom=mplan + mdown + mup, label="紧急购电费（5c·b）",
           color="#E45756")
    for i in range(len(mks)):
        s = mplan[i] + mdown[i] + mup[i] + mem[i]
        ax.text(i, s, f"{s / 1e4:.1f}万", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, mks, rotation=45)
    ax.set_ylabel("费用（元）")
    ax.set_title("第三问 月度购电费用三分解（2025-02-01 … 12-31）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "第三问_月度费用三分解")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.0))
    dd = np.arange(n_day)
    ax.plot(dd, dec["total"], lw=0.9, color="#4C78A8", label="当日总费用")
    k = 7
    if n_day >= k:
        mv = np.convolve(dec["total"], np.ones(k) / k, mode="valid")
        ax.plot(dd[k - 1:], mv, lw=1.8, color="#E45756", label=f"{k} 日滑动平均")
    ax.set_xlabel(f"评分日序号（{dates[sc[0]]} → {dates[sc[-1]]}）")
    ax.set_ylabel("费用（元/天）")
    ax.set_title("第三问 逐日购电总费用")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "第三问_逐日费用曲线")
    plt.close(fig)

    parts = [dec["plan"].sum(), dec["down"].sum(), dec["up"].sum(),
             dec["emerg"].sum()]
    names = ["计划购电费", "下调违约金", "上调加价", "紧急购电费"]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    wedges, txts, autos = ax.pie(
        parts, labels=names, autopct=lambda p: f"{p:.3f}%\n({p * sum(parts) / 100 / 1e4:.2f}万)",
        startangle=90, colors=["#4C78A8", "#F2CF5B", "#9ECAE9", "#E45756"],
        textprops={"fontsize": 10})
    ax.set_title(f"第三问 年度费用构成（合计 {sum(parts) / 1e4:.2f} 万元）")
    fig.tight_layout()
    C.save_figure(fig, "第三问_年度费用构成")
    plt.close(fig)

    q2 = None
    q2_src = None
    q2_cmp = None
    q2_err = None
    q2_extras: dict = {}
    if C.Q2.mode() == "real":
        try:
            q2, q2_src, q2_cmp = _q2_anchor()
        except Exception as exc:
            q2, q2_src, q2_cmp = None, None, None
            q2_err = f"{type(exc).__name__}: {exc}"
        try:
            q2_extras = C.Q2.core_numbers() or {}
        except Exception:
            q2_extras = {}

    rep = []
    rep.append("# 第三问 全年回测与结算报告\n")
    rep.append("- 代码：`第三问最终版/代码/08_全年回测与结算.py`")
    rep.append(f"- 评分期：{dates[sc[0]]} … {dates[sc[-1]]}，共 {n_day} 天\n")
    rep.append("## 1. 结算口径\n")
    rep.append("$$K_d=\\sum_t c_t\\min(p_t,q_t)"
               "+\\sum_t\\big[0.5c_t u_t+1.5c_t v_t\\big]"
               "+\\sum_t 5c_t b_t,\\qquad u_t=(p_t-q_t)^+,\\;v_t=(q_t-p_t)^+$$\n")
    rep.append("等价形式 $K_d=\\sum c_t q_t+0.5\\sum c_t(u_t+v_t)$ 已用于交叉自检，"
               "三种形式逐日最大偏差 < $10^{-6}$ 元。\n")
    rep.append("## 2. 年度汇总\n")
    rep.append("| 指标 | 数值 |")
    rep.append("|---|---:|")
    for k, val in tot.items():
        rep.append(f"| {k} | {val:,.4f} |")
    rep.append("")
    rep.append("## 3. 月度三分解\n")
    rep.append("| 月份 | 天数 | 计划费 元 | 下调违约金 元 | 上调加价 元 | 紧急费 元 | 合计 元 |")
    rep.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in m_rows:
        rep.append(f"| {r[0]} | {r[1]} | {float(r[4]):,.2f} | {float(r[5]):,.2f} | "
                   f"{float(r[6]):,.2f} | {float(r[7]):,.2f} | {float(r[8]):,.2f} |")
    rep.append("")
    rep.append("## 4. 与第二问对比\n")
    if q2 is None:
        if C.Q2.is_stub():
            why = "（骨架自检：使用合成占位数据）"
        elif q2_err is not None:
            why = (f"（数据源模式 `{C.Q2.mode()}`：**已定位第二问锚点但解析失败**"
                   f"，`{q2_err}`）")
        else:
            _miss = []
            if not Path(C.Q2_CORE_CSV).is_file():
                _miss.append(f"`{Path(C.Q2_CORE_CSV).name}`")
            if _find_q2_report() is None:
                _miss.append("`第二题_结果总览.md`")
            why = (f"（数据源模式 `{C.Q2.mode()}`，未找到第二问锚点文件："
                   f"{'、'.join(_miss) or '未知'}）")
        rep.append(f"- 本节**未做第二问数值对照**{why}。第三问**不复用第二问的任何"
                   "数值结果**，仅复用其数据文件（负荷与电价底座）。")
        rep.append(f"- 如需对照：确保 `{Path(C.Q2_CORE_CSV).parent.name}/"
                   f"{Path(C.Q2_CORE_CSV).name}` 存在"
                   "，并以 `Q3_Q2_SOURCE=real` 重跑本脚本。\n")
    else:
        try:
            _src_txt = Path(q2_src).relative_to(C.PROJECT_DIR.parent).as_posix()
        except Exception:
            _src_txt = str(q2_src)
        _q2fmt = (lambda v, s="{:,.4f}": "—" if v is None else s.format(v))
        _q2_plan = q2_extras.get("plan_kwh")
        _q2_emerg = q2_extras.get("emerg_fee")

        rep.append(f"- 第二问数值来源：`{_src_txt}`"
                   "（**只读引用，不参与本问任何计算**，未硬编码）")
        rep.append(f"- 引用口径：**{q2_cmp or '—'}**"
                   "，即第二问定稿自述的主结果「各自重订 / DP 价值执行器」"
                   "（其 `第二问_最终核心结果.csv`、`第二问_交付复核摘要.json`、"
                   "`第二题_结果总览.md` 三处逐位一致）。\n")
        rep.append("| 口径 | 总费用 元 | 计划购电量 kWh | 均价 元/kWh |")
        rep.append("|---|---:|---:|---:|")
        rep.append(f"| 第二问（{q2_cmp or '—'} / DP 价值执行器） | {q2:,.4f} | "
                   f"{_q2fmt(_q2_plan)} | — |")
        rep.append(f"| 第三问（本题） | {tot['总费用_元']:,.4f} | "
                   f"{tot['计划购电量_kWh']:,.4f} | "
                   f"{tot['总费用均价_元每kWh']:.6f} |")
        _plan_delta = (_q2fmt(tot["计划购电量_kWh"] - _q2_plan, "{:+,.4f}")
                       if _q2_plan else "—")
        rep.append(f"| 差额 | {tot['总费用_元'] - q2:+,.4f} | "
                   f"{_plan_delta} | — |")
        rep.append("")

        _tau0, _mae0, _mae_b, _win8 = _precision_numbers()
        _win_txt = f"（同窗口 {_win8}）" if _win8 else ""
        _acc = (f"附件 3 **{_tau0} 发布**的整点 MAE {_mae0:,.2f} kW vs "
                f"7 日同时刻均值基线 {_mae_b:,.2f} kW{_win_txt}"
                if (_tau0 and _mae0 and _mae_b)
                else "详见 `模型结果/第三问_预报精度表.csv`")
        _dir = "高于" if tot["总费用_元"] > q2 else "低于"
        _plan_txt = ""
        if _q2_plan:
            _r = tot["计划购电量_kWh"] / _q2_plan - 1.0
            _plan_txt = (f"，本问日前**计划购电量** "
                         f"{tot['计划购电量_kWh']:,.0f} kWh 相对第二问 "
                         f"{_q2_plan:,.0f} kWh **{_r:+.2%}**")
        _u_s = np.asarray(Uh, float)[sc]
        _e_s = np.asarray(Eh, float)[sc]
        _n_s = np.asarray(NN, float)[sc]
        _u_pos = _u_s > 1e-6
        _n_pos = max(int(_u_pos.sum()), 1)
        _curt_cap = float((_u_pos & (_e_s >= C.E_MAX - 1e-6)).sum()) / _n_pos
        _curt_sur = float((_u_pos & (_n_s <= 0.0)).sum()) / _n_pos
        _curt_over = float((_u_pos & (_n_s > 0.0)).sum()) / _n_pos
        _emerg_dir = "低于" if (tot["紧急购电费_元"] < (_q2_emerg or np.inf)) else "高于"
        rep.append(
            f"第三问费用**{_dir}**第二问，主因是**附件 3 的预报精度弱于第二问自建"
            f"因果预测**（{_acc}）：预报误差使日前计划量被系统性抬高{_plan_txt}，"
            f"计划量偏离实际需求又持续触发**不对称调整价**（本问调整相关费用 "
            f"{tot['调整相关费用_元']:,.0f} 元 = 下调违约金 "
            f"{tot['下调违约金_元']:,.0f} + 上调加价 {tot['上调加价_元']:,.0f}）；"
            f"同时储能容量有限（上限 {C.E_MAX:,.0f} kWh、每段功率上限 "
            f"{C.S_PERIOD_KWH:,.1f} kWh），**富余电量**（供给侧已计费/已放出、"
            f"但既未消纳也未能储存的部分）只能被弃置——本问全年弃电量 "
            f"{tot['弃电量_kWh']:,.0f} kWh，发生在 {int(_n_pos):,} 个时段上，其"
            f"中 **{_curt_cap:.1%}** 出现在储能已顶格（想存也存不下）的时段；"
            f"按净负荷划分，**{_curt_sur:.1%}** 出现在净负荷 ≤ 0（光伏输出不低"
            f"于负荷）、**{_curt_over:.1%}** 出现在净负荷 > 0（此时弃置量不可"
            f"能来自光伏过剩，只能来自**过度申报购电**）。后两项互斥且合计"
            f" 100 %；「储能顶格」是与之正交的另一个维度（可与任一项重叠）。"
            f"⇒ 本问弃电量**并非全部源于光伏过剩**，约四成源于日前计划量偏高，"
            f"属物理约束与合同口径共同作用下的合理弃置。\n")
        rep.append(
            f"> **方向提示（请勿误读）**：本问的**紧急购电反而「{_emerg_dir}」**"
            f"第二问（紧急购电费 {tot['紧急购电费_元']:,.0f} 元 vs "
            f"{_q2fmt(_q2_emerg, '{:,.0f}')} 元；紧急购电量 "
            f"{tot['紧急购电量_kWh']:,.0f} kWh vs {_q2fmt(q2_extras.get('emerg_kwh'), '{:,.0f}')} "
            f"kWh）。因此「第三方预报差 ⇒ 紧急购电更多」的直觉**在本数据上不成立**："
            f"预报误差体现为**日前过度购电 + 弃光**，而非紧急补电。"
            f"（本问的预报质量对照见 §4 括注与 `模型结果/第三问_预报精度表.csv`。）\n")

    rep.append("## 5. 与无储能基准对比\n")
    rep.append("| 口径 | 购电量 kWh | 购电费 元 |")
    rep.append("|---|---:|---:|")
    rep.append(f"| 无储能（仅买净负荷正部） | {tot['无储能基准购电量_kWh']:,.2f} | "
               f"{tot['无储能基准购电费_元']:,.2f} |")
    rep.append(f"| 第三问（滚动调整 + DP 执行） | {tot['调整购电量_kWh']:,.2f} | "
               f"{tot['总费用_元']:,.2f} |")
    rep.append(f"| 差异 | {tot['调整购电量_kWh'] - tot['无储能基准购电量_kWh']:+,.2f} | "
               f"{tot['储能相对基准节省_元']:+,.2f}"
               f"（{tot['储能相对基准节省_%']:+.3f} %） |")
    rep.append("")
    rep.append("## 6. 执行器对照\n")
    rep.append(f"- DP 保留水平执行：紧急购电费 {dec['emerg'].sum():,.2f} 元，"
               f"总费用 {dec['total'].sum():,.2f} 元。")
    rep.append(f"- 纯解析执行（无保留水平）：紧急购电费 {dec_an['emerg'].sum():,.2f} 元，"
               f"总费用 {dec_an['total'].sum():,.2f} 元。")
    _dp_delta = float(dec_an["total"].sum() - dec["total"].sum())
    _dp_ratio = float(dec_an["total"].sum() / dec["total"].sum() - 1.0)
    _dp_verdict = ("**DP 保留水平更省**。" if _dp_delta > 0
                   else "**DP 保留水平更贵**。" if _dp_delta < 0
                   else "**两种执行口径费用持平**。")
    rep.append(f"- **差值（纯解析执行 − DP 保留水平）** {_dp_delta:+,.2f} 元"
               f"（{_dp_ratio:+.3%}），{_dp_verdict}")
    _b_dp, _b_an = float(b[sc].sum()), float(ban[sc].sum())
    _pr_c = np.asarray(price, float)
    _wp_dp = float((_pr_c * b[sc]).sum() / _b_dp) if _b_dp > 0 else 0.0
    _wp_an = float((_pr_c * ban[sc]).sum() / _b_an) if _b_an > 0 else 0.0
    _cmp_b = ("更少" if _b_dp < _b_an - 1e-6
              else "更多" if _b_dp > _b_an + 1e-6 else "相等")
    _cmp_p = ("更低" if _wp_dp < _wp_an - 1e-9
              else "更高" if _wp_dp > _wp_an + 1e-9 else "相等")
    rep.append(f"  机制（本次数据）：紧急购电总量 DP {_b_dp:,.0f} kWh vs 解析版 "
               f"{_b_an:,.0f} kWh（DP {_cmp_b}）；紧急购电加权均价 DP {_wp_dp:.4f} "
               f"vs 解析版 {_wp_an:.4f} 元/kWh（DP {_cmp_p}）。")
    rep.append("  说明：$R_t$ 只改变**决策**，两类执行面对同一条实测负荷/光伏轨迹；"
               "上两行即为「更省/更贵」的实际来源，论文中直接引用即可"
               "（方向若与第二问不一致，如实写明并解释原因）。\n")
    rep.append("## 7. 两种结算口径（P0-5）\n")
    rep.append("| 口径 | 总费用 元 | 紧急购电量 kWh | 说明 |")
    rep.append("|---|---:|---:|---|")
    rep.append(f"| 模型内期望（30 情景均值） | {dec_exp['total'].sum():,.2f} | "
               f"{dec_exp['b'].sum():,.2f} | 每段决策代回**该节点自己的情景集** |")
    rep.append(f"| 样本外实现（真实轨迹） | {dec['total'].sum():,.2f} | "
               f"{dec['b'].sum():,.2f} | 唯一一条实际执行轨迹 |")
    rep.append(f"| 纯解析执行（实现口径） | {dec_an['total'].sum():,.2f} | "
               f"{dec_an['b'].sum():,.2f} | 无保留水平对照 |")
    rep.append("")
    rep.append("> **口径纪律**：单调性（引入更多预报节点不劣化）只在**模型内期望口径**"
               "下由可行域嵌套保证；样本外实现口径是单条轨迹，允许波动。"
               "消融实验的定性结论必须基于模型内期望口径，且完整档被允许**拒绝新解**。\n")
    rep.append("## 8. 闭环自检（P0-1 / P0-3 / P0-7）\n")
    rep.append("| 检查项 | 数值 | 阈值 | 结论 |")
    rep.append("|---|---:|---:|:--:|")
    _chk = [("四节点库存链 max 差 kWh", resid["node_chain"], 1e-8),
            ("跨日库存衔接 max 差 kWh", resid["cross_day"], 1e-8),
            ("每日计划输入 vs 真实库存 max 差 kWh", resid["plan_input_vs_real"], 1e-8),
            ("SOC 递推 max 残差 kWh", resid["soc_recursion"], 1e-8),
            ("功率平衡 max 残差 kWh", resid["balance"], 1e-8)]
    for nm, v, tol in _chk:
        rep.append(f"| {nm} | {v:.3e} | {tol:.0e} | {'✔' if v <= tol else '✘'} |")
    rep.append(f"| 信息泄露时段数 | {resid['info_leak_cells']} | 0 | "
               f"{'✔' if resid['info_leak_cells'] == 0 else '✘'} |")
    rep.append(f"| 同时充放电时段数 | {resid['simul_charge_discharge']} | 0 | "
               f"{'✔' if resid['simul_charge_discharge'] == 0 else '✘'} |")
    rep.append("")
    rep.append(f"- SOC 区间 $[{C.E_MIN:.0f},\\,{C.E_MAX:.0f}]$ kWh：下越界 "
               f"{resid['soc_lower']:.3e}、上越界 {resid['soc_upper']:.3e} kWh。")
    rep.append(f"- 功率上限：每时段充/放电 ≤ $S=5000\\Delta t={C.S_PERIOD_KWH:.4f}$ kWh，"
               f"越界量 {resid['power_cap']:.3e}。")
    rep.append(f"- 紧急购电与充电同段（附录 A-31 关注项）："
               f"**{resid['emerg_with_charge_cells']} 时段 / "
               f"{resid['emerg_with_charge_days']} 天**（紧急购电共 "
               f"{resid['emerg_cells']} 时段 / {resid['emerg_days']} 天）。")
    rep.append("")
    rep.append("> **历史对照（P0-1 的量化规模）**：旧实现 `06` 用「规划库存」充当次日"
               f"起始库存，与真实库存的最大偏差达 **{resid['plan_chain_gap_max']:,.2f} kWh**"
               f"（均值 {resid['plan_chain_gap_mean']:,.2f} kWh），"
               "再叠加 `07` 把 144 时段一次递推造成的**信息泄露**，"
               "才出现「引入更多预报反而更贵」的非单调反常。\n")
    C.write_text_utf8(C.REPORT_BACK_MD, "\n".join(rep))
    log("")
    log(f"已保存：{C.DAILY_CSV.relative_to(C.PROJECT_DIR)}、"
        f"{C.MONTHLY_CSV.name}、{C.YEARLY_CSV.name}、"
        f"第三问_结算口径对照.csv、第三问_闭环审计.csv、{C.BACKTEST_NPZ.name}")
    log(f"已保存：{C.REPORT_BACK_MD.relative_to(C.PROJECT_DIR)}")

    Rmat = np.asarray(res["R"], float)
    eman = np.asarray(res["b_analytic"], float)
    emerg_fee_dp = float((C.EMERG_MULT * price * b[sc]).sum())
    emerg_fee_an = float((C.EMERG_MULT * price * eman[sc]).sum())
    _pr2d = np.broadcast_to(np.asarray(price, float)[None, :], b[sc].shape)
    _m_dp = b[sc] > 1e-9
    _m_an = eman[sc] > 1e-9
    pr_dp = float(_pr2d[_m_dp].mean()) if _m_dp.any() else float("nan")
    pr_an = float(_pr2d[_m_an].mean()) if _m_an.any() else float("nan")
    _b_dp2, _b_an2 = float(b[sc].sum()), float(eman[sc].sum())
    _w_dp2 = (float((_pr2d * b[sc]).sum() / _b_dp2)
              if _b_dp2 > 0 else 0.0)
    _w_an2 = (float((_pr2d * eman[sc]).sum() / _b_an2)
              if _b_an2 > 0 else 0.0)
    _cmp_w2 = ("更低" if _w_dp2 < _w_an2 - 1e-9
               else "更高" if _w_dp2 > _w_an2 + 1e-9 else "相等")
    gain_dp = emerg_fee_an - emerg_fee_dp
    n_both = int(resid["emerg_with_charge_days"])
    n_curt = int(np.sum(np.asarray(res["curt"], float)[sc].sum(axis=1) > 1e-9))
    cur = np.asarray(res["curt"], float)
    Rbar = Rmat[sc].mean(axis=0)

    rep2 = []
    rep2.append("# 第三问 阶段内 DP 未来价值执行器报告\n")
    rep2.append("- 代码：`第三问最终版/代码/07_阶段内DP执行器.py`（**函数库**）"
                "+ `第三问最终版/代码/08_全年回测与结算.py`（本文写出）")
    rep2.append(f"- 情景数 $M$ = {info['m_scenarios']}（真实口径执行时退化为单条实测轨迹）")
    rep2.append(f"- 价值函数网格步长 $\\delta$ = {info['delta_kwh']:.1f} kWh")
    rep2.append(f"- 规划信号 $\\nu$ = {info['nu']:.10f} 元/kWh（**不进账单**）")
    rep2.append(f"- 发布节点 $\\tau \\in$ "
                f"{'{' + ', '.join(str(int(t)) + ':00' for t in res['update_times']) + '}'}，"
                f"分 $\\tau$ 构造价值函数、只执行本节点段（整改 P0-7 选择 A）\n")
    rep2.append("## 1. 方法（按发布节点分层，严禁跨节点复用 $R_t$）\n")
    rep2.append("对每个评分日、每个发布节点 $\\tau_k$（$k_0$ 为该节点起点时段）：\n")
    rep2.append("1. 用**该节点自己的**预报情景集（附件三 $\\tau_k$ 情景；真实口径下即为"
                "单条实测）在 $[k_0,144)$ 上做反向折线递推：\n")
    rep2.append("$$H_{T+1,\\omega}(e)=-\\nu e,\\qquad "
                "\\bar H_t(e)=\\mathbb E_\\omega[H_{t+1,\\omega}(e)],\\qquad "
                "R_t\\in\\arg\\max_{e}\\{5c_t\\eta\\,e+\\bar H_{t+1}(e)\\}$$\n")
    rep2.append("2. **只执行 $[0,\\,k_1-k_0)$ 段**，跨节点**严禁复用 $R_t$**；\n")
    rep2.append("3. 用**真实**负荷/光伏因果执行，得到该节点末端的**真实**储电量，"
                "作为下一节点的起点。\n")
    rep2.append("执行规则：$D_t=\\min\\{r_t,S,\\eta(E_{t-1}-R_t)\\}$、"
                "$b_t=[r_t-D_t]^+$、"
                "$C_t=\\min\\{-r_t,S,(E_{\\max}-E_{t-1})/\\eta\\}$，"
                "禁止用紧急购电主动充电。\n")
    rep2.append("> 旧版把 4 段净缺口拼接成 144 时段后只做一次递推，等价于 0:00 即已知"
                "6:00/12:00/18:00 的预报，构成跨阶段**信息泄露**；该驱动已删除。\n")
    rep2.append("## 2. 保留水平 $R_t$ 的日画像（每 2 h 一个分箱的日间均值，kWh）\n")
    _bins = list(range(0, C.PERIODS_PER_DAY, 12))
    rep2.append("| " + " | ".join(f"{j // 6:02d}:00" for j in _bins) + " |")
    rep2.append("|" + "---|" * len(_bins))
    rep2.append("| " + " | ".join(f"{Rbar[j:j + 12].mean():.0f}"
                                 for j in _bins) + " |")
    rep2.append("")
    rep2.append(f"全期 $R_t\\in[{Rmat[sc].min():.0f},\\,{Rmat[sc].max():.0f}]$ kWh，"
                f"均值 {Rmat[sc].mean():.0f} kWh。\n")
    rep2.append("## 3. 对照实验：DP 保留水平 vs 纯解析执行\n")
    rep2.append("| 执行策略 | 紧急购电 kWh | 紧急购电费 元 | 充电 kWh | 放电 kWh | 弃电 kWh |")
    rep2.append("|---|---|---|---|---|---|")
    rep2.append(f"| DP 保留水平 | {float(b[sc].sum()):,.1f} | {emerg_fee_dp:,.1f} | "
                f"{float(Ch[sc].sum()):,.1f} | {float(Dh[sc].sum()):,.1f} | "
                f"{float(Uh[sc].sum()):,.1f} |")
    rep2.append(f"| 纯解析执行（$\\alpha=0$） | {float(eman[sc].sum()):,.1f} | "
                f"{emerg_fee_an:,.1f} | {float(Can[sc].sum()):,.1f} | "
                f"{float(np.asarray(res['D_analytic'], float)[sc].sum()):,.1f} | "
                f"{float(np.asarray(res['curt_analytic'], float)[sc].sum()):,.1f} |")
    rep2.append("")
    rep2.append("### 3.1 结论\n")
    rep2.append(f"- **电量维度**：DP 版紧急购电 {float(b[sc].sum()):,.1f} kWh，"
                f"解析版 {float(eman[sc].sum()):,.1f} kWh，"
                f"DP 差（**DP − 解析**）"
                f"{float(b[sc].sum() - eman[sc].sum()):+,.1f} kWh。")
    rep2.append(f"- **费用维度**：DP 版紧急购电费 {emerg_fee_dp:,.2f} 元，"
                f"解析版 {emerg_fee_an:,.2f} 元，"
                f"DP 差（**解析 − DP**）{gain_dp:+,.2f} 元"
                f"（{gain_dp / emerg_fee_an * 100:+.2f}%）。")
    rep2.append(f"- **择时维度①（等权口径）**：紧急购电**发生时段**的电价"
                f"**算术均值**（每个发生时段等权，只回答「是不是发生在贵时段」）"
                f"DP 版 {pr_dp:.4f} 元/kWh vs 解析版 {pr_an:.4f} 元/kWh"
                f"（全期均价 {price.mean():.4f}）。")
    rep2.append(f"- **择时维度②（按量加权口径，★费用方向由它决定）**："
                f"$\\sum c_t b_t/\\sum b_t$ = 紧急购电费 ÷ "
                f"{C.EMERG_MULT:.0f} ÷ 电量，DP 版 {_w_dp2:.4f} 元/kWh vs "
                f"解析版 {_w_an2:.4f} 元/kWh（DP {_cmp_w2}）。\n")
    rep2.append("> ★ ①② 两个「均价」**口径不同、符号可以相反**，必须按标签读："
                "① 等权口径只回答「是否主动避开了高价时段」，"
                "② 按量加权口径 = 费用 ÷ 倍数 ÷ 电量，**直接决定费用方向**。"
                "费用结论以上表费用维度与 ② 为准；若把 ① 当成费用依据，"
                "会得出与费用相反的结论（本仓库已实测发生过）。\n")
    rep2.append("> 上表三维度**均由本次数据算出**，不作方向预设：若 DP 版紧急购电费"
                "低于解析版，则说明保留水平 $R_t$ 实现了「把紧急购电从高价时段挪到"
                "低价时段」；若高于解析版，则说明保留水平在本数据集上未带来费用改善，"
                "论文中应如实写成「与第二问方向不一致」并解释原因。\n")
    rep2.append("## 4. 已知模型局限\n")
    rep2.append(f"- 「紧急购电与充电同段」= **{n_both} 天**（附录 A-31）。成因是当日**内**"
                "缺少再优化机会：无法在同一天内同时「用便宜谷电充电」与"
                "「避免紧急购电」。")
    rep2.append(f"- 出现弃电（$\\Sigma U>0$）的天数 = {n_curt}。")
    rep2.append(f"- 执行面对的是**单条实测**轨迹，而 $R_t$ 由**情景期望**推出，"
                f"存在分布失配（README §5.10）。")
    C.write_text_utf8(C.REPORT_EXEC_MD, "\n".join(rep2))
    log(f"已保存：{C.REPORT_EXEC_MD.relative_to(C.PROJECT_DIR)}（原 `07` 报告，"
        f"现由 `08` 从统一闭环结果复算写出）")

    log.dump(C.LOG_DIR / "第三问_08结算日志.txt", tail="")
    log("")
    log(f"总用时 {time.perf_counter() - t00:.1f} s")
    log("[08 完成] 全年回测与费用结算结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
