#!/usr/bin/env python
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


def main() -> int:
    C.ensure_dirs()
    log = C.Tee()
    t00 = time.perf_counter()

    log("=" * 78)
    log("第三问 00 —— 接口契约自检 / 骨架连通性自检")
    log("=" * 78)

    mode = C.Q2.mode()
    log(f"数据来源模式 Q3_Q2_SOURCE = {mode}")
    log(f"输出根目录 OUT_ROOT = {C.OUT_ROOT}")
    if mode == "stub":
        log("")
        log("⚠⚠ 沙箱模式（stub）：使用**合成占位数据**，数值无建模意义；")
        log("   所有产物写入 `_骨架自检/`，不会触碰正式结果。")
        log("   正式运行请去掉 Q3_Q2_SOURCE=stub（或设为 real）。")
    elif mode == "off":
        log("")
        log("ℹ 沙箱模式（off）：**不加载第二问数据**，仅验证与其无关的骨架项；")
        log("   所有产物写入 `_骨架自检/`，不会触碰正式结果。")
        log("   需要第二问数据的步骤（02、04–11）会以 Q2Unavailable 干净失败。")

    log("")
    log("── 1. 第二问数据接口契约校验 ──")
    rep = C.Q2.validate(strict=False)
    contract_ok = bool(rep.get("ok"))

    if mode == "off":
        log("  已跳过（模式 off）")
    else:
        for k, v in sorted(rep.get("present", {}).items()):
            log(f"  ✔ {k:28s} shape={v}")
        for k in rep.get("absent", []):
            log(f"  ○ {k:28s} 缺失（可选键，不影响运行）")
        for w in rep.get("warnings", []):
            log(f"  ⚠ {w}")
        for e in rep.get("errors", []):
            log(f"  ✘ {e}")
        log("")
        log(f"  契约结论：{'✔ 满足第三问全部必需依赖' if contract_ok else '✘ 不满足，需第二问配合补齐'}")

    rows = C.Q2.contract_table()
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_接口契约.csv",
        ["数据文件", "键名", "形状", "单位", "必需性", "第三问使用者", "口径备注"],
        rows)

    md = ["# 第三问 ← 第二问 接口契约\n"]
    md.append("> 由 `第三问/代码/00_接口契约自检.py` 生成。")
    md.append("> **第三问对第二问的全部依赖都在此表内**；取数统一经 "
              "`第三问最终版/代码/_q2_adapter.py` 的 `C.Q2.matrix()` / `C.Q2.scenarios()`。\n")
    md.append("## 1. 必需依赖\n")
    md.append("| 数据文件 | 键名 | 形状 | 单位 | 必需性 | 第三问使用者 | 口径备注 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")
    md.append("## 2. 单位口径（必须一致，否则第三问全线错位）\n")
    md.append("- **功率**（`*_kw`）：kW，整点值 = 该小时平均功率；")
    md.append("- **流量/电量**（`*_energy_kwh`、`scen_L`）：kWh / 单个 10 min 时段；")
    md.append("- 换算：`电量 = 功率 × Δt`，`Δt = 1/6 h`；")
    md.append("- 储能功率上限 `S = 5000 kW × Δt = 833.3333333333334 kWh/时段`；")
    md.append("- 时段标签采用**区间右端点**（`0:00-0:10` 标签为 `0:10`）。\n")
    md.append("## 3. 第三问明确弃用的第二问键\n")
    md.append("第三问只复用**负荷侧**；光伏侧改由附件 3 重建（README §5.10）：\n")
    md.append("- " + "、".join(f"`{k}`" for k in C.SCENARIO_VOID))
    md.append("")
    md.append("## 4. 第二问若调整数据结构\n")
    md.append("只需修改 `第三问最终版/代码/_q2_adapter.py`：\n")
    md.append("1. 键名变化 → 在 `KEY_ALIASES` 中登记 `契约名 → 实际键名`；")
    md.append("2. 形状变化 → 更新对应 `CONTRACT_*` 条目的 `shape`；")
    md.append("3. 单位变化 → 更新 `unit` 并同步 `validate()` 中的**量纲哨兵**。\n")
    md.append("`02`–`11` **无需任何改动**。\n")
    C.write_text_utf8(C.REPORT_DIR / "第三问_第二问接口契约.md", "\n".join(md))
    log(f"  已保存：{C.RESULT_DIR.relative_to(C.PROJECT_DIR)}/第三问_接口契约.csv、"
        f"{C.REPORT_DIR.relative_to(C.PROJECT_DIR)}/第三问_第二问接口契约.md")

    log("")
    log("── 2. 骨架连通性自检（横切口径，与第二问数值无关）──")
    import numpy as np

    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        log(f"  {'✔' if ok else '✘'} {name}" + (f"　{detail}" if detail else ""))

    chk("K1 Δt = 1/6 h", abs(C.DELTA_HOURS - 1 / 6) < 1e-15,
        f"Δt = {C.DELTA_HOURS:.10f}")
    chk("K2 每日 144 时段", C.PERIODS_PER_DAY == 144,
        f"{C.PERIODS_PER_DAY}")
    chk("K3 功率上限 S = 5000·Δt",
        abs(C.S_PERIOD_KWH - C.P_MAX_KW * C.DELTA_HOURS) < 1e-9,
        f"S = {C.S_PERIOD_KWH:.10f} kWh/时段")
    chk("K4 储能参数自洽",
        C.E_MIN < C.E_INIT < C.E_MAX and 0 < C.ETA < 1,
        f"SOC∈[{C.E_MIN:.0f},{C.E_MAX:.0f}]，E_init={C.E_INIT:.0f}，η={C.ETA}")
    chk("K5 评分期 334 天", C.N_SCORE_DAYS == 334 and C.N_WARMUP_DAYS == 31,
        f"预热 {C.N_WARMUP_DAYS} + 评分 {C.N_SCORE_DAYS}")

    k0s = list(C.TAU_PERIOD_INDEX)
    chk("K6 τ → 时段索引 k0 = 6τ",
        k0s == [int(t / C.DELTA_HOURS) for t in C.TAU_HOURS],
        f"τ={C.TAU_HOURS} → k0={k0s}")
    chk("K7 四个发布时刻连续覆盖全天",
        all(k0s[i + 1] - k0s[i] == 36 for i in range(3)) and k0s[0] == 0,
        f"分段长度 {[k0s[i + 1] - k0s[i] for i in range(3)]} + 末段 36")

    Vd = np.tile(np.arange(C.LEAD_HOURS, dtype=float),
                 (2, C.N_TAU, C.M_SCENARIOS, 1))
    Va = np.asarray(C.lead_to_absolute(Vd), float)
    ok8 = True
    det8 = ""
    for ti, tau in enumerate(C.TAU_HOURS):
        for ha in range(24):
            want = (ha - tau) % 24
            if not np.allclose(Va[:, ti, :, ha], want):
                ok8 = False
                det8 = f"τ={tau} ha={ha} 期望 {want} 实得 {Va[0, ti, 0, ha]}"
                break
        if not ok8:
            break
    chk("K8 lead→绝对小时重排正确", ok8,
        det8 or "8 个 τ × 24 个绝对小时逐点验证通过")

    Vraw = np.arange(2 * C.N_TAU * 24, dtype=float).reshape(2, C.N_TAU, 24) + 1.0
    Vpw = C.disaggregate(Vraw, C.DISAGG_PIECEWISE)
    dev = float(np.max(np.abs(
        Vpw.reshape(Vraw.shape[:-1] + (24, 6)).mean(-1) - Vraw)))
    chk("K9 分段常数展开守恒", dev < 1e-9, f"最大偏差 {dev:.3e}")

    chk("K10 光伏情景能量换算含 Δt",
        abs(C.scenario_pv_energy(np.ones((1, C.N_TAU, 1, 24)) * 600.0, 0, 0).max()
            - 600.0 * C.DELTA_HOURS) < 1e-9,
        f"600 kW → {600.0 * C.DELTA_HOURS:.6f} kWh/时段")

    pr = np.full(144, 0.8)
    p = np.full(144, 100.0)
    q = p.copy()
    q[10] = 60.0
    q[20] = 150.0
    u = np.maximum(p - q, 0.0)
    v = np.maximum(q - p, 0.0)
    b = np.zeros(144)
    b[30] = 5.0
    k1 = float((pr * np.minimum(p, q)).sum() + 0.5 * (pr * u).sum()
               + 1.5 * (pr * v).sum() + 5.0 * (pr * b).sum())
    k2 = float((pr * q).sum() + 0.5 * (pr * (u + v)).sum() + 5.0 * (pr * b).sum())
    k3 = float((pr * p).sum() - 0.5 * (pr * u).sum()
               + 1.5 * (pr * v).sum() + 5.0 * (pr * b).sum())
    kf = float(C.adjust_cost(pr, p, q, u, v, b)["total"])
    chk("K11 结算三形式等价", max(abs(k1 - k2), abs(k2 - k3), abs(k3 - kf)) < 1e-9,
        f"① {k1:.6f}　② {k2:.6f}　③ {k3:.6f}　④ {kf:.6f}")

    chk("K12 时间标签口径",
        C.interval_label(1) == "00:00-00:10"
        and C.interval_label(36) == "05:50-06:00"
        and C.parse_time_label_to_minutes("24:00") == 1440,
        f"t=1 → {C.interval_label(1)}；t=36 → {C.interval_label(36)}；"
        f"'24:00' → {C.parse_time_label_to_minutes('24:00')}")

    chk("K13 指定时刻 → 时段索引", C.slot_to_period_index("12:00") == 73,
        f"'12:00' → {C.slot_to_period_index('12:00')}（即区间 11:50-12:00）")

    _forbidden = ("Q2_MATRIX_NPZ", "Q2_SCENARIO_NPZ", "附件二_矩阵数据",
                  "附件二_情景库")
    _offenders: list[str] = []
    for _p in sorted(C.CODE_DIR.glob("*.py")):
        if _p.name in ("_q2_adapter.py", "_comm3.py", "00_接口契约自检.py"):
            continue
        try:
            _src = _p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for _ln, _line in enumerate(_src.splitlines(), 1):
            _code = _line.split("#", 1)[0]
            if "np.load" in _code or "load(" in _code:
                for _pat in _forbidden:
                    if _pat in _code:
                        _offenders.append(f"{_p.name}:{_ln}")
    chk("K14 无脚本绕过接入层直连第二问 .npz", not _offenders,
        ("违规 " + "、".join(_offenders)) if _offenders
        else "全部经 C.Q2.matrix() / C.Q2.scenarios() 访问")

    _iso_ok = True
    if C.Q2.mode() in ("stub", "off"):
        _iso_ok = (C.OUT_ROOT.resolve()
                   == (C.PROJECT_DIR / "_骨架自检").resolve()
                   and C.SUBMIT_DIR.resolve().is_relative_to(C.OUT_ROOT.resolve()))
    chk("K15 非 real 模式输出隔离于 _骨架自检/", _iso_ok,
        f"OUT_ROOT={C.OUT_ROOT.name or C.OUT_ROOT}")

    _m_ok = C.Q2.mode() in ("real", "stub", "off")
    chk("K16 接入层模式取值合法", _m_ok, f"Q3_Q2_SOURCE={C.Q2.mode()}")

    _p3 = _load("_policy3.py", "q3_policy_check")
    _p3_ok = all(hasattr(_p3, _n) for _n in
                 ("run_policy", "run_arm", "summarise", "decompose",
                  "exante_cost_seg", "_residuals"))
    chk("K17 `_policy3` 暴露统一策略入口与结算/测度/残差工具", _p3_ok,
        "run_policy / run_arm / summarise / decompose / exante_cost_seg"
        " / _residuals")

    _bypass: list[str] = []
    for _nm in ("08_全年回测与结算.py", "09_消融实验.py"):
        _f = C.CODE_DIR / _nm
        if not _f.exists():
            continue
        _src = _f.read_text(encoding="utf-8", errors="replace")
        _c = "\n".join(l.split("#", 1)[0] for l in _src.splitlines())
        if "_policy3" not in _c:
            _bypass.append(f"{_nm} 未加载 _policy3")
        elif not ("run_policy" in _c or "run_arm" in _c):
            _bypass.append(f"{_nm} 未调用 run_policy/run_arm")
        if "TwoStageSP" in _c:
            _bypass.append(f"{_nm} 仍在自行建 LP")
    chk("K18 P0-4：08/09 均改用 `_policy3` 统一入口，不再自行建 LP",
        not _bypass, "、".join(_bypass) if _bypass else "✔ 唯一入口")

    _dep_bad = []
    _src3 = (C.CODE_DIR / "_policy3.py").read_text(encoding="utf-8",
                                                   errors="replace")
    for _nm in ("06_滚动调整规划.py", "08_全年回测与结算.py",
                "09_消融实验.py"):
        if f'_load("{_nm}' in _src3 or f"_load('{_nm}" in _src3:
            _dep_bad.append(_nm)
    chk("K19 `_policy3` 不加载 06/08/09（单向依赖，无重复实现）",
        not _dep_bad, "、".join(_dep_bad) if _dep_bad else "✔ 单向")

    _q2d = str(C.Q2_DIR)
    _no_hist = not any(_t in _q2d for _t in (
        "历史模型", "归档", "_备份", "备份", "历史版本"))
    _src_desc = {"env": "环境变量 Q3_Q2_DIR 显式指定",
                 "probe": "按候选名自动探测",
                 "none": "候选名均未命中"}.get(C.Q2_DIR_SOURCE, C.Q2_DIR_SOURCE)
    chk("K20 第二问目录解析未落入历史模型/归档（防静默取错版本）",
        _no_hist, f"Q2_DIR={C.Q2_DIR.name}（来源：{_src_desc}）")
    if C.Q2.mode() == "real":
        _found = C.Q2_MATRIX_NPZ.is_file() and C.Q2_SCENARIO_NPZ.is_file()
        chk("K20b real 模式下第二问两个 npz 均存在", _found,
            f"矩阵={C.Q2_MATRIX_NPZ.is_file()} 情景={C.Q2_SCENARIO_NPZ.is_file()}")

    _spec = [
        ("额定容量 kWh", C.BATTERY.get("rated_capacity_kwh"), 12000.0),
        ("SOC 下限 kWh", C.E_MIN, 1200.0),
        ("SOC 上限 kWh", C.E_MAX, 10800.0),
        ("初始 SOC kWh", C.E_INIT, 6000.0),
        ("最大充电功率 kW", C.BATTERY.get("max_charge_power_kw"), 5000.0),
        ("最大放电功率 kW", C.BATTERY.get("max_discharge_power_kw"), 5000.0),
        ("充电效率 η_c", C.ETA_C, 0.9),
        ("放电效率 η_d", C.ETA_D, 0.9),
        ("下调倍率 ρ_down", C.RHO_DOWN, 0.5),
        ("上调倍率 ρ_up", C.RHO_UP, 1.5),
        ("紧急购电倍率 κ_em", C.EMERG_MULT, 5.0),
        ("情景数 M", C.M_SCENARIOS, 30),
        ("每日时段数", C.PERIODS_PER_DAY, 144),
    ]
    _drift = []
    for _n, _g, _v in _spec:
        try:
            _gv = float(_g)
        except (TypeError, ValueError):
            _drift.append(f"{_n}: 取不到数值（{_g!r}）≠ 附录 {_v!r}")
            continue
        if abs(_gv - float(_v)) > 1e-9:
            _drift.append(f"{_n}: {_gv!r} ≠ 附录 {_v!r}")
    chk("K21 附录 1 物理常数逐项与文档一致（防共享输入漂移）",
        not _drift, f"{len(_spec)} 项全对" if not _drift else "；".join(_drift))
    try:
        _nu_calc = float(C.compute_nu())
        _nu_ok, _nu_det = (abs(float(C.NU_VALUE) - _nu_calc) < 1e-12,
                           f"配置 {C.NU_VALUE!r} vs 复算 {_nu_calc!r}")
    except Exception as _exc:
        _nu_ok, _nu_det = False, f"复算失败 {type(_exc).__name__}: {_exc}"
    chk("K21b 终端价值 ν = 附件 1 前 30 时段均价 / η（逐位复算）",
        _nu_ok, _nu_det)

    log("")
    log("── 3. 数据通路自检（形状 / 量纲，不含业务数值）──")
    if mode == "off":
        log("  已跳过（模式 off）")
    else:
        Z = C.Q2.matrix()
        n_day = int(np.asarray(Z["load_kw"]).shape[0])
        price = np.asarray(Z["price"], float)
        load_kw = np.asarray(Z["load_kw"], float)
        pv_kw = np.asarray(Z["pv_kw"], float)
        chk("D1 price 形状 (144,) 且全正",
            price.shape == (144,) and bool((price > 0).all()),
            f"shape={price.shape}，范围 [{price.min():.4f}, {price.max():.4f}]")
        chk("D2 load_kw / pv_kw 形状一致且非负",
            load_kw.shape == pv_kw.shape == (n_day, 144)
            and load_kw.min() >= 0 and pv_kw.min() >= 0,
            f"shape={load_kw.shape}，load 均值 {load_kw.mean():.1f} kW，"
            f"pv 均值 {pv_kw.mean():.1f} kW")
        ratio = float(load_kw.mean() / max(float(
            np.asarray(Z["load_energy_kwh"], float).mean()), 1e-9))
        chk("D3 量纲哨兵 load_kw / load_energy_kwh ≈ 6", 4.0 < ratio < 8.0,
            f"比值 {ratio:.4f}（预期 ≈ 1/Δt = 6）")
        uoff = 31 if n_day >= 365 else 0
        chk("D4 天数轴 = 365（31 预热 + 334 评分）", n_day == 365,
            f"n_day = {n_day}" + (f"，首个评分日下标 {uoff}" if n_day == 365 else ""))

        S2 = C.Q2.scenarios()
        scen_L = np.asarray(S2["scen_L"], float)
        M = C.M_SCENARIOS
        chk("D5 scen_L 形状与情景数一致",
            scen_L.shape == (n_day, M, 144) and float(scen_L.min()) >= 0,
            f"shape={scen_L.shape}，均值 {scen_L.mean():.2f} kWh/时段")
        chk("D6 scen_L 与 load_energy 量级一致（同口径）",
            abs(float(scen_L.mean()) / max(float(
                np.asarray(Z["load_energy_kwh"], float).mean()), 1e-9) - 1.0) < 0.15,
            f"scen_L 均值 / load_energy 均值 = "
            f"{float(scen_L.mean()) / max(float(np.asarray(Z['load_energy_kwh'], float).mean()), 1e-9):.4f}")

        F = np.load(C.V_FORECAST_NPZ, allow_pickle=False) \
            if C.V_FORECAST_NPZ.exists() else None
        if F is not None:
            sc_idx = np.asarray(F["score_day_index"], int)
            d = int(sc_idx[0])
            chk("D7 附件 3 与第二问日期数一致",
                int(np.asarray(F["dates"]).size) == n_day,
                f"附件3 {int(np.asarray(F['dates']).size)} 天 vs 第二问 {n_day} 天")
            if C.V_SCENARIO_NPZ.exists():
                Zd = np.load(C.V_SCENARIO_NPZ, allow_pickle=False)
                scen_V_abs = C.lead_to_absolute(
                    np.asarray(Zd["scen_V_hourly"], float))
                LV = C.scenario_pv_energy(scen_V_abs, d, 0)
                net = scen_L[d] - LV
                chk("D8 情景净负荷相减通路（scen_L − V·Δt）",
                    net.shape == (M, 144),
                    f"shape={net.shape}，第 {d} 天均值 {net.mean():.2f} kWh/时段")
        else:
            log("  ○ D7/D8 已跳过：尚未生成 `处理后数据/附件三_预报矩阵.npz`"
                "（先跑 01–04）")

        if C.V_FORECAST_NPZ.exists() and C.V_SCENARIO_NPZ.exists():
            try:
                _d0 = int(np.asarray(
                    np.load(C.V_FORECAST_NPZ, allow_pickle=False)[
                        "score_day_index"], int)[0])
                _probe = _p3.run_policy(update_times=(6, 12, 18),
                                        days=[_d0, _d0 + 1],
                                        progress_every=0)
                _res = _probe["resid"]
                chk("D9 闭环残差（2 天冒烟）达到机器精度",
                    float(_res["max_abs"]) < 1e-6
                    and int(_res["info_leak_cells"]) == 0,
                    f"max_abs={float(_res['max_abs']):.3e}，"
                    f"信息泄露 {int(_res['info_leak_cells'])} 时段")
                _s0 = _p3.run_policy(update_times=(), days=[_d0, _d0 + 1],
                                     progress_every=0)
                _e0 = float(_s0["exp_dec"]["total"].sum())
                _e1 = float(_probe["exp_dec"]["total"].sum())
                chk("D10 P0-5：共同测度下嵌套档期望费用单调非增（2 天冒烟）",
                    _e0 >= _e1 - 1e-9,
                    f"S0 {_e0:,.4f} 元 ≥ S061218 {_e1:,.4f} 元"
                    f"（差 {_e0 - _e1:+,.4f} 元）")
            except Exception as _exc:
                chk("D9/D10 闭环引擎小样冒烟", False,
                    f"{type(_exc).__name__}: {_exc}")
        else:
            log("  ○ D9/D10 已跳过：需要 `附件三_预报矩阵.npz` 与 "
                "`附件三_情景库.npz`（先跑 01–04）")

    log("")
    log("── 4. 流水线骨架（脚本依赖顺序）──")
    pipeline = [
        ("01", "检查附件三原始数据", "附件 3", "—"),
        ("02", "处理附件三基础数据", "附件 3 + 第二问 dates", "附件三_预报矩阵.npz"),
        ("03", "校验附件三处理结果", "附件 3 + 第二问 pv_kw", "—"),
        ("04", "构建情景库", "附件三预报 + 第二问 scen_L/scen_idx",
         "附件三_情景库.npz"),
        ("05", "求解阶段 0 计划（LP 内核，5 种模式）",
         "第二问 price + 附件三情景", "—（被 `_policy3` 调用）"),
        ("06", "滚动调整规划（⚠ 历史实现，见下注）",
         "第二问 price/scen_L + 阶段0", "—（不参与正式结果）"),
        ("07", "阶段内 DP 执行器（**函数库**，P0-7 选择 A）",
         "第二问 load_kw/pv_kw/scen_L", "—（被 `_policy3` 调用；独立驱动已退役）"),
        ("_policy3", "★ 统一闭环策略引擎（唯一策略入口）",
         "05 + 07 + 第二问 + 附件三",
         "内存结果（由 08/09 调用）"),
        ("08", "★ 全年回测与结算（唯一主结果入口）",
         "`_policy3.run_policy` + 第二问 price",
         "第三问_全年回测.npz + 5 CSV + 报告×2（含执行器报告）"),
        ("09", "消融实验", "`_policy3.run_arm`（六档共用入口）",
         "第三问_消融实验对照表.csv + npz + 报告"),
        ("10", "指定日期明细与作图", "08 + 第二问 price", "4 CSV + 3 图"),
        ("11", "生成并校验 result3.xlsx", "08 + 第二问 load/pv 电量",
         "提交结果/result3.xlsx"),
        ("12", "端到端编排", "00–14", "日志/报告汇总"),
        ("13", "敏感性与收敛性检验（P1-2/P1-3/P1-4）",
         "`_policy3.run_policy(m/nu/delta)`",
         "3 CSV + npz + 图 + 报告（非交付主结果）"),
        ("14", "★ 最终一致性校验（只读交叉复核）",
         "08 + 09 + 13 产物 + 第二问原始电量",
         "1 CSV + 1 报告（不改任何主结果）"),
    ]
    C.write_csv_utf8_sig(
        C.RESULT_DIR / "第三问_流水线骨架.csv",
        ["序号", "用途", "第二问依赖", "主要产物"],
        [[a, b, c, d] for a, b, c, d in pipeline])
    for num, name, dep, out in pipeline:
        log(f"  {num:<8s} {name:<26s} 依赖：{dep:<40s} → {out}")
    log("")
    log("  注：`06` 为历史实现（以「规划 SOC」为滚动起点、阶段 0 未内生调整费，")
    log("      且 4 段净缺口拼接成 144 时段做一次递推 ⇒ 跨阶段信息泄露）。")
    log("      正式结果与消融**全部**经 `_policy3` 的「真实标量 SOC 闭环」路径，")
    log("      不使用 `06` 的产物；其文件保留仅为追溯。")
    log("      注：`07` 现为**函数库**（P0-7 选择 A），独立驱动已删除；")
    log("      其「DP 保留水平 vs 纯解析执行」结论由 `08` 从统一闭环结果复算写出。")

    fig = ["```mermaid", "flowchart LR",
           "  A3[附件3 预报] --> S01[01 检查]",
           "  S01 --> S02[02 处理] --> S03[03 校验]",
           "  S02 --> S04[04 情景库]",
           "  Q2M[(第二问 矩阵数据)] -.-> S02",
           "  Q2M -.-> S03",
           "  Q2S[(第二问 情景库 scen_L)] -.-> S04",
           "  S04 --> S05[05 阶段0 LP 内核]",
           "  Q2M -.-> S05",
           "  POL{{_policy3 统一闭环策略引擎}}",
           "  S05 -.内核.-> POL",
           "  S07[07 DP 执行器 函数库] -.内核.-> POL",
           "  Q2M -.-> POL",
           "  Q2S -.-> POL",
           "  POL --> S08[08 回测结算 唯一主结果入口]",
           "  POL --> S09[09 消融实验 六档]",
           "  S08 --> S10[10 指定日期明细]",
           "  S08 --> S11[11 result3.xlsx]",
           "  POL --> S13[13 敏感性与收敛性]",
           "  S09 --> S14[14 最终一致性校验]",
           "  S13 --> S14",
           "  S08 --> S14",
           "  Q2M -.-> S14",
           "  Q2M -.-> S08",
           "  Q2M -.-> S07",
           "  Q2S -.-> S07",
           "  S06[06 滚动调整 历史实现] -.保留追溯.-> ARCH[(不参与正式结果)]",
           "  Q2S -.-> S06",
           "```"]

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    log("")
    log("── 5. 自检汇总 ──")
    log(f"  骨架连通性：{len(checks) - n_fail}/{len(checks)} 通过"
        f"{'，全部通过 ✔' if n_fail == 0 else f'，{n_fail} 项未通过 ✘'}")
    log(f"  接口契约　：{'✔ 满足' if (contract_ok or mode == 'off') else '✘ 不满足'}")
    log(f"  输出根目录：{C.OUT_ROOT}")
    log(f"  总用时 {time.perf_counter() - t00:.1f} s")

    rep_md = ["# 第三问 骨架自检报告\n"]
    rep_md.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    rep_md.append(f"- 数据来源模式：`{mode}`"
                  + ("（**沙箱 / 合成占位数据**）" if C.SANDBOX else ""))
    rep_md.append(f"- 输出根目录：`{C.OUT_ROOT}`")
    rep_md.append(f"- 结论："
                  + (f"骨架 {len(checks) - n_fail}/{len(checks)} 通过，"
                     if checks else "")
                  + ("接口契约满足 ✔" if (contract_ok or mode == "off")
                     else "接口契约不满足 ✘") + "\n")
    rep_md.append("## 1. 骨架连通性检查\n")
    rep_md.append("| 检查项 | 结果 | 细节 |")
    rep_md.append("|---|---|---|")
    for name, ok, detail in checks:
        rep_md.append(f"| {name} | {'✔' if ok else '✘'} | {detail} |")
    rep_md.append("")
    rep_md.append("## 2. 接口契约校验\n")
    if mode == "off":
        rep_md.append("已跳过（模式 off）。\n")
    else:
        for k, v in sorted(rep.get("present", {}).items()):
            rep_md.append(f"- ✔ `{k}` shape={v}")
        for k in rep.get("absent", []):
            rep_md.append(f"- ○ `{k}` 缺失（可选）")
        for w in rep.get("warnings", []):
            rep_md.append(f"- ⚠ {w}")
        for e in rep.get("errors", []):
            rep_md.append(f"- ✘ {e}")
        rep_md.append("")
    rep_md.append("## 3. 流水线骨架\n")
    rep_md.append("| 序号 | 用途 | 第二问依赖 | 主要产物 |")
    rep_md.append("|---|---|---|---|")
    for a, b, c, d in pipeline:
        rep_md.append(f"| {a} | {b} | {c} | {d} |")
    rep_md.append("")
    rep_md.extend(fig)
    rep_md.append("")
    rep_md.append("## 4. 骨架已就位、待第二问数据回填的部分\n")
    rep_md.append("以下步骤的**代码结构与校验逻辑已完成**，只差第二问数据定型后正式求解：\n")
    rep_md.append("| 步骤 | 状态 | 阻塞原因 |")
    rep_md.append("|---|---|---|")
    rep_md.append("| 01–04（光伏侧） | 可独立完成 | 不依赖第二问（仅 02 做日期对齐校验） |")
    rep_md.append("| 05–06（LP 规划） | 结构就位 | 需 `price` / `scen_L` |")
    rep_md.append("| 07–08（执行与结算） | 结构就位 | 需 `load_kw` / `pv_kw` / `price` |")
    rep_md.append("| 09（消融） | 结构就位 | 同上 |")
    rep_md.append("| 10–11（明细与提交） | 结构就位 | 同上 |")
    rep_md.append("| 12（编排） | 已验证可跑 | 无需第二问 |")
    rep_md.append("")
    C.write_text_utf8(C.REPORT_DIR / "第三问_骨架自检报告.md", "\n".join(rep_md))
    log(f"  已保存：{C.REPORT_DIR.relative_to(C.PROJECT_DIR)}/第三问_骨架自检报告.md")

    log.dump(C.LOG_DIR / "第三问_00骨架自检日志.txt", tail="")
    log("")
    log("[00 完成] 接口契约自检 / 骨架自检结束。")

    return 0 if (n_fail == 0 and (contract_ok or mode == "off")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
