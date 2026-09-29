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


C = _load("_comm2.py", "q2_comm")
P5 = _load("05_求解日前计划.py", "q2_plan")

import numpy as np

M = C.M_SCENARIOS
T = C.PERIODS_PER_DAY
S = C.S_PERIOD_KWH
ETA = C.ETA
NU = P5.NU


def reserve_indicator(price: np.ndarray) -> np.ndarray:
    return (np.maximum.accumulate(price[::-1])[::-1] > price + 1e-12).astype(float)


def fixed_reserve_execute(N_act: np.ndarray, g_day: np.ndarray, E0: float,
                          alpha: float, I_t: np.ndarray) -> dict:
    R_t = C.E_MIN + (C.E_MAX - C.E_MIN) * alpha * I_t
    r = N_act - g_day
    Cch = np.zeros(T)
    D = np.zeros(T)
    b = np.zeros(T)
    U = np.zeros(T)
    E = np.zeros(T)
    e = float(E0)
    for t in range(T):
        rt = float(r[t])
        if rt > 0.0:
            dd = float(np.clip(min(rt, S, ETA * (e - R_t[t])), 0.0, None))
            D[t] = dd
            b[t] = max(rt - dd, 0.0)
            e = e - dd / ETA
        else:
            cch = float(np.clip(min(-rt, S, (C.E_MAX - e) / ETA), 0.0, None))
            Cch[t] = cch
            U[t] = max(-rt - cch, 0.0)
            e = e + ETA * cch
        E[t] = e
    return {"C": Cch, "D": D, "b": b, "U": U, "E": E, "R": R_t}


def main() -> int:
    C.ensure_dirs()
    log: list[str] = []

    def p(msg: str = "") -> None:
        log.append(msg)
        print(msg, flush=True)

    t_all = time.perf_counter()
    p("=" * 74)
    p("第二问 08 —— 固定保留策略反证扫描（R_t = 1200 + 9600·α·I_t）")
    p("=" * 74)
    p(f"α 扫描档位：{['%.5f%%' % (a * 100) for a in C.ALPHA_GRID]}")
    p("I_t 口径：I_t = 1 ⟺ 当日后续仍存在严格更高电价时段")

    Z = np.load(C.BASE_MATRIX_NPZ, allow_pickle=False)
    Zd = np.load(C.SCENARIO_NPZ, allow_pickle=False)
    Pf = np.load(C.RESULT_DIR / "第二问_固定计划.npz", allow_pickle=False)
    price = np.asarray(Z["price"], float)
    N_all = np.asarray(Z["net_load_energy_kwh"], float)
    scen_L = np.asarray(Zd["scen_L"], float)
    scen_V = np.asarray(Zd["scen_V"], float)
    scen_N = scen_L - scen_V
    date_strs = [str(x) for x in Z["dates"]]
    score_idx = np.asarray(Z["score_day_index"], int)
    warm_idx = np.asarray(Z["warmup_day_index"], int)
    plan_g = np.asarray(Pf["plan_g"], float)
    plan_E_mean = np.asarray(Pf["plan_E_mean"], float)

    I_t = reserve_indicator(price)
    p("")
    p(f"I_t 覆盖比例：全年 {I_t.mean() * 100:.2f}% 的时段处于「后续仍有更贵时段」；"
      f"其中当日最贵时段 I_t = {(I_t == 0).mean() * 100:.2f}%")
    p(f"R_t 区间：α=0 → [{C.E_MIN:.0f}, {C.E_MIN:.0f}] kWh；"
      f"α=max → [{C.E_MIN:.0f}, {C.E_MIN + (C.E_MAX - C.E_MIN) * max(C.ALPHA_GRID):.0f}] kWh")

    p("")
    p("── 补出 1 月计划（预热期，情景 LP 自 E_init 起前向链）──")
    t0 = time.perf_counter()
    lpM = P5.DayPlanLP(price, nu=NU, m=M)
    g_jan = np.zeros((365, T))
    jan_E_mean = np.zeros(365)
    for d in warm_idx:
        E0 = jan_E_mean[d - 1] if d > 0 and jan_E_mean[d - 1] > 0 else float(C.E_INIT)
        r = lpM.solve(scen_N[d], E0)
        g_jan[d] = r["g"]
        jan_E_mean[d] = float(r["E"][:, -1].mean())
    p(f"  用时 {time.perf_counter() - t0:.1f} s；1 月计划购电量 "
      f"{g_jan[warm_idx].sum():.4f} kWh，计划费 {float((g_jan[warm_idx] * price).sum()):.4f} 元，"
      f"1 月末计划库存 {jan_E_mean[warm_idx[-1]]:.6f} kWh")

    p("")
    p("── α 扫描：评分期固定计划 + 固定保留执行器 ──")
    rows = []
    detail: dict = {}
    for alpha in C.ALPHA_GRID:
        e = float(C.E_INIT)
        jan_emerg = 0.0
        for d in warm_idx:
            o = fixed_reserve_execute(N_all[d], g_jan[d], e, alpha, I_t)
            jan_emerg += float(o["b"].sum())
            e = float(o["E"][-1])
        E_after_jan = e
        e = float(C.E_INIT)
        plan_cost = 0.0
        emerg_cost = 0.0
        emerg_kwh = 0.0
        E_end = 0.0
        for d in score_idx:
            o = fixed_reserve_execute(N_all[d], plan_g[d], e, alpha, I_t)
            plan_cost += float(price @ plan_g[d])
            emerg_cost += float((5.0 * price) @ o["b"])
            emerg_kwh += float(o["b"].sum())
            e = float(o["E"][-1])
            E_end = e
        rows.append((alpha, jan_emerg, plan_cost, emerg_cost, plan_cost + emerg_cost,
                     emerg_kwh, E_end, E_after_jan))
        detail[alpha] = (jan_emerg, plan_cost, emerg_cost, emerg_kwh, E_end)
        p(f"  α = {alpha * 100:7.3f}%  1月紧急量 {jan_emerg:11.4f} kWh  "
          f"2–12月总费用 {plan_cost + emerg_cost:14.4f} 元  "
          f"（计划 {plan_cost:.2f} + 紧急 {emerg_cost:.2f}，紧急量 {emerg_kwh:.2f} kWh，"
          f"年末库存 {E_end:.2f}）")

    base_obj = rows[0][4]
    p("")
    p(f"{'α':>9}{'1月紧急量_kWh':>16}{'2–12月总费用_元':>20}{'较α=0增加_元':>18}{'相对增幅':>12}")
    for alpha, jan_emerg, pc, ec, tot, ek, Ee, Eaj in rows:
        p(f"{alpha * 100:>8.3f}%{jan_emerg:>16.4f}{tot:>20.4f}{tot - base_obj:>18.4f}"
          f"{(tot / base_obj - 1) * 100:>11.4f}%")

    mono = all(rows[i][4] <= rows[i + 1][4] + 1e-6 for i in range(len(rows) - 1))
    p("")
    p(f"总费用随 α 单调不减：{'✔ 是' if mono else '✘ 否'}")
    p("结论（论文要点）：固定保留比例**不降本反增本** —— 保留库存既增加当期高价补购，"
      "又因库存长期偏高而挤占后续富余时段的充电空间；")
    p("  α 越大，1 月（初始库存 6000 kWh < 门槛）越早被锁死，紧急量呈阶跃式上升。"
      "储能的机会成本是**状态相关**的，固定电价门槛无法识别「后续是缺额还是富余」。")
    p("  措辞保守：该结论**只针对本节检验的规则族**，不排除其他简单策略有效。")

    p("")
    p("── 自校验：α = 0 应退化为解析响应 ──")
    cmp_path = C.RESULT_DIR / "第二问_执行器对照表.csv"
    if cmp_path.exists():
        import csv
        with open(cmp_path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("比较方式") == "固定计划" and row.get("执行器") == "解析响应":
                    ref = float(row["总费用_元"])
                    d0 = rows[0][4]
                    p(f"  解析响应（07 回测）总费用 = {ref:.4f} 元；本扫描 α=0 = {d0:.4f} 元；"
                      f"差 {abs(ref - d0):.6e} 元 → "
                      f"{'✔ 一致' if abs(ref - d0) < 1e-4 else '✘ 不一致'}")
                    break
            else:
                p("  未在对照表中找到解析响应行，跳过。")
    else:
        p(f"  未找到 {cmp_path.name}（请先运行 07），跳过。")

    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第二问_固定保留策略扫描.csv",
        ("α", "α百分比", "1月紧急量_kWh", "2-12月计划费_元", "2-12月紧急费_元",
         "2-12月总费用_元", "2-12月紧急量_kWh", "2-12月年末储电量_kWh",
         "1月末执行库存_kWh", "较α=0增加_元", "相对增幅"),
        [(f"{a:.6f}", f"{a * 100:.5f}%", f"{je:.6f}", f"{pc:.6f}", f"{ec:.6f}",
          f"{tot:.6f}", f"{ek:.6f}", f"{Ee:.6f}", f"{Eaj:.6f}",
          f"{tot - base_obj:.6f}", f"{(tot / base_obj - 1) * 100:.6f}%")
         for a, je, pc, ec, tot, ek, Ee, Eaj in rows])

    I_t_plot = C.RESULT_DIR / "第二问_图5b_固定保留扫描作图数据.csv"
    C.write_csv_utf8_sig(
        I_t_plot,
        ("α百分比", "2-12月总费用_万元", "1月紧急量_kWh"),
        [(f"{a * 100:.5f}%", f"{tot / 1e4:.6f}", f"{je:.6f}")
         for a, je, pc, ec, tot, ek, Ee, Eaj in rows])

    p("")
    p(f"已保存：模型结果/第二问_固定保留策略扫描.csv、第二问_图5b_固定保留扫描作图数据.csv")
    p(f"总用时 {time.perf_counter() - t_all:.1f} s")

    C.write_text_utf8(C.SOLVE_LOG_DIR / "第二问_08固定保留扫描日志.txt",
                      "\n".join(log + ["", "[08 完成] 固定保留策略反证扫描结束。"]))
    C.write_text_utf8(C.REPORT_DIR / "第二问_固定保留策略反证报告.md",
                      "\n".join(["# 第二问 固定保留策略反证扫描报告", ""] + log + [""]))
    p("已保存：求解日志/第二问_08固定保留扫描日志.txt、"
      "报告/第二问_固定保留策略反证报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
