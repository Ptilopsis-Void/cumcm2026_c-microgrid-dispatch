#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_Q2_DIR_CANDIDATES: tuple[str, ...] = (
    "第二题", "第二问最终版", "第二问", "第二题最终版",
)
_Q2_DIR_FORBIDDEN = ("历史模型", "归档", "_备份", "备份", "_调试备份", "历史版本")


def _has_q2_payload(d: Path) -> bool:
    if not d.is_dir():
        return False
    if any(part in _Q2_DIR_FORBIDDEN for part in d.parts):
        return False
    pd = d / "处理后数据"
    return (pd / "附件二_矩阵数据.npz").is_file() and \
        (pd / "附件二_情景库.npz").is_file()


def resolve_q2_dir() -> tuple[Path, str]:
    env = os.environ.get("Q3_Q2_DIR")
    if env:
        return Path(env), "env"
    root = _PROJECT_DIR.parent
    for name in _Q2_DIR_CANDIDATES:
        d = root / name
        if _has_q2_payload(d):
            return d, "probe"
    return root / _Q2_DIR_CANDIDATES[0], "none"


Q2_DIR, Q2_DIR_SOURCE = resolve_q2_dir()
MATRIX_NPZ = Q2_DIR / "处理后数据" / "附件二_矩阵数据.npz"
SCENARIO_NPZ = Q2_DIR / "处理后数据" / "附件二_情景库.npz"

CONTRACT_MATRIX: dict[str, dict] = {
    "dates": dict(
        shape="(365,)", unit="日期字符串 YYYY-MM-DD", required=True,
        users="02", note="仅用于与附件 3 的日期序列逐日对齐校验"),
    "price": dict(
        shape="(144,)", unit="元/kWh", required=True,
        users="_comm3/05/06/07/08/10", note="附件 1 电价曲线，144 个 10 min 时段"),
    "load_kw": dict(
        shape="(365,144)", unit="kW", required=True,
        users="04/07/09", note="**功率**；转电量须乘 Δt=1/6 h"),
    "pv_kw": dict(
        shape="(365,144)", unit="kW", required=True,
        users="02/03/04/07/09", note="**功率**；光伏实际出力（10 min）"),
    "load_energy_kwh": dict(
        shape="(365,144)", unit="kWh/时段", required=True,
        users="11", note="= load_kw · Δt，电量守恒校验用"),
    "pv_energy_kwh": dict(
        shape="(365,144)", unit="kWh/时段", required=True,
        users="11", note="= pv_kw · Δt，电量守恒校验用"),
    "net_load_energy_kwh": dict(
        shape="(365,144)", unit="kWh/时段", required=False,
        users="—", note="= (load_kw − pv_kw) · Δt，第三问未直接使用"),
    "delta_hours": dict(
        shape="标量", unit="h", required=False, users="—", note="应为 1/6"),
    "s_period_kwh": dict(
        shape="标量", unit="kWh/时段", required=False, users="—",
        note="应为 5000·Δt = 833.333…"),
    "score_day_index": dict(
        shape="(334,)", unit="行下标", required=False,
        users="—", note="第三问改用附件 3 的 score_day_index，此处仅作交叉核对"),
    "warmup_day_index": dict(
        shape="(31,)", unit="行下标", required=False, users="—", note="预热期"),
}

CONTRACT_SCENARIO: dict[str, dict] = {
    "scen_L": dict(
        shape="(365,30,144)", unit="kWh/时段", required=True,
        users="04/05/06/07/09",
        note="负荷情景。**复用**，负荷预测规则不变"),
    "scen_idx": dict(
        shape="(365,30)", unit="历史日下标", required=True,
        users="04", note="**复用**，保证与第二问配对一致"),
    "resid_L": dict(
        shape="(365,144)", unit="kWh/时段", required=False,
        users="04", note="负荷残差库，第三问仅记录来源，不参与换算"),
}

SCENARIO_VOID: tuple[str, ...] = (
    "scen_V", "scen_N", "resid_V", "resid_N", "Vhat", "Nhat", "rho",
    "pv_hist_max", "Lhat",
)

KEY_ALIASES: dict[str, str] = {}


class Q2Unavailable(RuntimeError):
    pass


class Q2ContractError(RuntimeError):
    pass


MODES = ("real", "stub", "off")


def resolve_mode() -> str:
    m = (os.environ.get("Q3_Q2_SOURCE", "real") or "real").strip().lower()
    if m not in MODES:
        raise ValueError(f"Q3_Q2_SOURCE 必须是 {MODES} 之一，收到 {m!r}")
    return m


def _canon(key: str) -> str:
    return KEY_ALIASES.get(key, key)


class Bundle:

    def __init__(self, data: dict, source: str, is_stub: bool = False,
                 origin: Path | None = None):
        self._data = data
        self.source = source
        self.is_stub = is_stub
        self.origin = origin

    def __getitem__(self, key: str):
        k = _canon(key)
        if k not in self._data:
            raise KeyError(
                f"[Q2 {self.source}] 契约键 {key!r} 不存在。"
                f"可用键：{sorted(self._data)}")
        return self._data[k]

    def __contains__(self, key: str) -> bool:
        return _canon(key) in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def get(self, key: str, default=None):
        k = _canon(key)
        return self._data.get(k, default)

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        tag = "stub" if self.is_stub else "real"
        return f"<Q2Bundle {self.source} [{tag}] keys={len(self._data)}>"


def _load_real(path: Path, contract: dict, source: str) -> Bundle:
    import numpy as np

    if not path.is_file():
        _src = {"env": "环境变量 Q3_Q2_DIR 显式指定", "probe": "按候选名自动探测",
                "none": "候选名均未命中"}[Q2_DIR_SOURCE]
        raise Q2Unavailable(
            f"第二问数据不存在：{path}\n"
            f"  已解析第二问目录 = {Q2_DIR}（来源：{_src}）\n"
            f"  候选目录名：{', '.join(_Q2_DIR_CANDIDATES)}\n"
            f"  如第二问尚未产出或正在整改，可设 Q3_Q2_SOURCE=stub 跑骨架自检；"
            f"或设 Q3_Q2_DIR=<第二题目录> 显式指向。\n"
            f"  ★ 显式指定时不做静默回退（防止误用历史模型产物）。")

    z = np.load(path, allow_pickle=False)
    missing = [k for k, v in contract.items() if v["required"] and k not in z.files]
    if missing:
        raise Q2ContractError(
            f"第二问 {path.name} 缺少必需键：{missing}\n"
            f"  实际键：{sorted(z.files)}\n"
            f"  若第二问改了键名，请在 `_q2_adapter.KEY_ALIASES` 中登记映射。")
    return Bundle({k: z[k] for k in z.files}, source=source, origin=path)


def _stub_price(n: int = 144) -> "object":
    import numpy as np

    t = np.arange(n) / n * 24.0
    base = 0.62
    peak = 0.55 * (np.exp(-((t - 9.5) / 1.6) ** 2)
                   + 1.15 * np.exp(-((t - 19.0) / 2.0) ** 2))
    valley = -0.22 * (np.exp(-((t - 3.0) / 2.6) ** 2)
                      + 0.8 * np.exp(-((t - 14.0) / 2.2) ** 2))
    return np.clip(base + peak + valley, 0.37, 1.40).astype(float)


def _stub_matrix(n_day: int = 365, n_t: int = 144) -> dict:
    import numpy as np

    rng = np.random.default_rng(20260911)
    dt = 1.0 / 6.0
    hod = (np.arange(n_t) + 0.5) * dt

    shape_d = (1.9 * np.exp(-((hod - 11.0) / 3.0) ** 2)
               + 2.1 * np.exp(-((hod - 19.5) / 3.2) ** 2)
               + 0.6)
    shape_d = shape_d / shape_d.mean()
    doy = np.arange(n_day) / 365.0 * 2 * np.pi
    season = 1.0 + 0.12 * np.cos(doy - 0.6)
    load_kw = 4225.0 * shape_d[None, :] * season[:, None]
    load_kw *= 1.0 + 0.04 * rng.standard_normal((n_day, n_t))
    load_kw = np.clip(load_kw, 300.0, None)

    bell = np.exp(-((hod - 12.5) / 3.1) ** 2)
    bell[bell < 1e-3] = 0.0
    bell = bell / bell.mean()
    pv_kw = 2177.0 * bell[None, :] * (1.0 + 0.30 * np.cos(doy - 2.6))[:, None]
    pv_kw *= 1.0 + 0.18 * rng.standard_normal((n_day, n_t))
    pv_kw = np.clip(pv_kw, 0.0, 9995.8875)
    pv_kw[pv_kw < 1.0] = 0.0

    M = 30
    load_e = load_kw * dt
    scen_L = load_e[:, None, :] * (
        1.0 + 0.05 * rng.standard_normal((n_day, M, n_t)))
    scen_L = np.clip(scen_L, 0.0, None)
    scen_L[:, 0, :] = load_e

    dates = np.array(
        [(np.datetime64("2025-01-01") + np.timedelta64(i, "D")).astype(str)
         for i in range(n_day)], dtype="<U10")

    return {
        "dates": dates,
        "price": _stub_price(n_t),
        "load_kw": load_kw,
        "pv_kw": pv_kw,
        "load_energy_kwh": load_e,
        "pv_energy_kwh": pv_kw * dt,
        "net_load_energy_kwh": (load_kw - pv_kw) * dt,
        "delta_hours": np.asarray(dt),
        "s_period_kwh": np.asarray(5000.0 * dt),
        "score_day_index": np.arange(31, 365),
        "warmup_day_index": np.arange(31),
    }


def _stub_scenarios(matrix: dict) -> dict:
    import numpy as np

    n_day = matrix["load_kw"].shape[0]
    M = 30
    load_e = np.asarray(matrix["load_energy_kwh"], float)
    rng = np.random.default_rng(20260912)
    scen_L = load_e[:, None, :] * (
        1.0 + 0.05 * rng.standard_normal((n_day, M, load_e.shape[1])))
    scen_L = np.clip(scen_L, 0.0, None)
    scen_L[:, 0, :] = load_e
    return {
        "scen_L": scen_L,
        "scen_idx": np.tile(np.arange(M), (n_day, 1)),
        "resid_L": scen_L[:, 0, :] - load_e,
    }


def _check_shape(key: str, spec: str, arr) -> str | None:
    import numpy as np

    try:
        shape = tuple(np.shape(arr))
    except Exception:
        return f"{key}: 无法取得形状"
    if spec.strip() in ("标量", "scalar"):
        if shape not in ((), (1,)):
            return f"{key}: 期望标量，实得 {shape}"
        return None
    want = tuple(int(x) for x in spec.strip("() ").split(",") if x.strip())
    if len(want) != len(shape):
        return f"{key}: 维度不符，期望 {len(want)} 维 {want}，实得 {shape}"
    for w, s in zip(want, shape):
        if w in (365,) and s not in (365,):
            return f"{key}: 天数轴不符，期望 {w}，实得 {s}"
        if w not in (365, 334, 31, 30, 144, 24) and w != s:
            return f"{key}: 轴长不符，期望 {want}，实得 {shape}"
    return None


def validate(mode: str | None = None, strict: bool = False) -> dict:
    import numpy as np

    mode = mode or resolve_mode()
    rep: dict = {"mode": mode, "ok": True, "errors": [], "warnings": [],
                 "present": {}, "absent": []}
    if mode == "off":
        rep["warnings"].append("模式 off：不加载第二问数据，跳过契约校验")
        return rep

    if mode == "stub":
        mtx = _stub_matrix()
        scn = _stub_scenarios(mtx)
        rep["warnings"].append("模式 stub：使用合成占位数据，数值无建模意义")
    else:
        try:
            mtx = dict(_load_real(MATRIX_NPZ, CONTRACT_MATRIX, "矩阵数据")._data)
            scn = dict(_load_real(SCENARIO_NPZ, CONTRACT_SCENARIO, "情景库")._data)
        except (Q2Unavailable, Q2ContractError) as e:
            rep["ok"] = False
            rep["errors"].append(str(e))
            return rep

    for name, data, contract in (("矩阵数据", mtx, CONTRACT_MATRIX),
                                 ("情景库", scn, CONTRACT_SCENARIO)):
        for key, spec in contract.items():
            if key not in data:
                if spec["required"]:
                    rep["ok"] = False
                    rep["errors"].append(
                        f"{name}.{key} 缺失（必需，使用者：{spec['users']}）")
                else:
                    rep["absent"].append(f"{name}.{key}")
                continue
            arr = data[key]
            if key == "price":
                ok_pos = bool(np.all(np.asarray(arr, float) > 0))
                if not ok_pos:
                    rep["errors"].append(f"{name}.price 出现非正值")
                    rep["ok"] = False
            err = _check_shape(key, spec["shape"], arr)
            if err:
                rep["ok"] = False
                rep["errors"].append(f"{name}.{err}")
            else:
                rep["present"][f"{name}.{key}"] = str(tuple(np.shape(arr)))

    if "load_kw" in mtx and "load_energy_kwh" in mtx:
        r = float(np.mean(mtx["load_kw"])) / max(float(np.mean(mtx["load_energy_kwh"])), 1e-9)
        if not (4.0 < r < 8.0):
            msg = (f"量纲哨兵：mean(load_kw)/mean(load_energy_kwh) = {r:.3f}，"
                   f"预期 ≈ 6（= 1/Δt）。疑似单位口径变化（kW ↔ kWh）")
            if strict:
                rep["errors"].append(msg)
                rep["ok"] = False
            else:
                rep["warnings"].append(msg)

    if strict and rep["absent"]:
        rep["warnings"].append(f"可选键缺失 {len(rep['absent'])} 个（不影响运行）")
    return rep


_cache: dict[str, Bundle] = {}


def matrix() -> Bundle:
    mode = resolve_mode()
    if mode == "off":
        raise Q2Unavailable("Q3_Q2_SOURCE=off：禁止访问第二问数据")
    if "matrix" in _cache:
        return _cache["matrix"]
    if mode == "stub":
        b = Bundle(_stub_matrix(), source="stub:矩阵数据", is_stub=True)
    else:
        b = _load_real(MATRIX_NPZ, CONTRACT_MATRIX, "矩阵数据")
    _cache["matrix"] = b
    return b


def scenarios() -> Bundle:
    mode = resolve_mode()
    if mode == "off":
        raise Q2Unavailable("Q3_Q2_SOURCE=off：禁止访问第二问数据")
    if "scenarios" in _cache:
        return _cache["scenarios"]
    if mode == "stub":
        if "matrix" not in _cache:
            _cache["matrix"] = Bundle(_stub_matrix(), source="stub:矩阵数据",
                                      is_stub=True)
        b = Bundle(_stub_scenarios(_cache["matrix"]._data),
                   source="stub:情景库", is_stub=True)
    else:
        b = _load_real(SCENARIO_NPZ, CONTRACT_SCENARIO, "情景库")
    _cache["scenarios"] = b
    return b


def is_stub() -> bool:
    return resolve_mode() == "stub"


def reset_cache() -> None:
    _cache.clear()


def contract_table() -> list[list[str]]:
    rows: list[list[str]] = []
    for name, contract in (("矩阵数据", CONTRACT_MATRIX), ("情景库", CONTRACT_SCENARIO)):
        for key, spec in contract.items():
            rows.append([name, key, spec["shape"], spec["unit"],
                         "必需" if spec["required"] else "可选",
                         spec["users"], spec["note"]])
    for k in SCENARIO_VOID:
        rows.append(["情景库", k, "—", "—", "**弃用**", "—",
                     "第三问改用附件 3 重建光伏侧"])
    return rows


Q2_RESULT_DIR = Q2_DIR / "模型结果"
Q2_CORE_CSV = Q2_RESULT_DIR / "第二问_最终核心结果.csv"
Q2_EXEC_CSV = Q2_RESULT_DIR / "第二问_执行器对照表.csv"
Q2_NOSTOR_CSV = Q2_RESULT_DIR / "第二问_无储能基准.csv"
Q2_ACCURACY_CSV = Q2_RESULT_DIR / "第二问_预测精度表.csv"

Q2_HEADLINE_COMPARISON = "各自重订"
Q2_HEADLINE_EXECUTOR = "DP"
Q2_CORE_KEYS = {
    "yearly_total": "全年总费用_元",
    "emerg_fee": "全年紧急购电费_元",
    "emerg_kwh": "全年紧急购电量_kWh",
    "plan_kwh": "全年计划购电量_kWh",
    "dp_gain": "相对解析响应节省额_元",
}


def q2_core_numbers() -> dict:
    out = {k: None for k in Q2_CORE_KEYS}
    rows = _csv_rows(Q2_CORE_CSV)
    if not rows:
        return out
    by_key = {d.get("指标", "").strip(): d for d in rows}
    for name, key in Q2_CORE_KEYS.items():
        d = by_key.get(key)
        if d is not None:
            out[name] = _num(d, "数值")
    return out


def q2_report_candidates() -> list[Path]:
    return [
        Q2_DIR / "报告" / "第二题_结果总览.md",
        Q2_DIR / "模型结果" / "第二题_结果总览.md",
        Q2_DIR / "第二题_结果总览.md",
    ]


def _csv_rows(path: Path) -> list[dict]:
    import csv

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
    except Exception:
        return []
    if len(rows) < 2:
        return []
    head = [h.strip() for h in rows[0]]
    out: list[dict] = []
    for r in rows[1:]:
        if not any(x.strip() for x in r):
            continue
        out.append({head[i]: (r[i].strip() if i < len(r) else "")
                    for i in range(len(head))})
    return out


def _num(d: dict, *names: str):
    for nm in names:
        if nm in d and d[nm] not in ("", "nan", "NaN", "-"):
            try:
                return float(d[nm])
            except ValueError:
                continue
    return None


def _degroup(line: str) -> str:
    import re

    return re.sub(r"(?<=\d)[,\u2009\u00a0 ](?=\d)", "", line)


def _grab_after(txt: str, key: str, lo: float, hi: float):
    import re

    for line in txt.splitlines():
        pos = line.find(key)
        if pos < 0:
            continue
        tail = _degroup(line[pos + len(key):]).replace("*", " ")
        for s in re.findall(r"\d+(?:\.\d+)?", tail):
            v = float(s)
            if lo <= v <= hi:
                return v
    return None


def _warmup_month_labels() -> set:
    import numpy as _np

    try:
        Z = matrix()
        dates = [str(x) for x in _np.asarray(Z["dates"]).ravel()]
        wi = _np.asarray(Z.get("warmup_day_index", []), dtype=int).ravel()
    except Exception:
        return set()
    if wi.size == 0 or not dates:
        return set()
    warm = {dates[i] for i in wi if 0 <= i < len(dates)}
    bym: dict = {}
    for d in dates:
        bym.setdefault(d[:7], []).append(d)
    return {m for m, ds in bym.items() if ds and all(d in warm for d in ds)}


def q2_headline_numbers() -> dict:
    out: dict = {"source": None, "comparison": None, "yearly_total": None,
                 "emerg_fee": None, "emerg_kwh": None, "baseline_fee": None,
                 "pv_mae_kw": None, "dp_gain": None}
    if resolve_mode() != "real":
        return out

    sources: list[str] = []

    core = q2_core_numbers()
    if any(v is not None for v in core.values()):
        sources.append(str(Q2_CORE_CSV))
        for k in ("yearly_total", "emerg_fee", "emerg_kwh", "dp_gain"):
            if core.get(k) is not None:
                out[k] = core[k]
        out["comparison"] = Q2_HEADLINE_COMPARISON

    rows = _csv_rows(Q2_EXEC_CSV)
    if rows:
        dp_rows = [d for d in rows if Q2_HEADLINE_EXECUTOR in d.get("执行器", "")]
        if dp_rows:
            pref = [d for d in dp_rows
                    if Q2_HEADLINE_COMPARISON in d.get("比较方式", "")]
            dp = (pref or dp_rows)[0]
            used_cmp = dp.get("比较方式", "")
            if any(out[k] is None
                   for k in ("yearly_total", "emerg_fee", "emerg_kwh")):
                sources.append(str(Q2_EXEC_CSV))
            if out["yearly_total"] is None:
                out["yearly_total"] = _num(dp, "总费用_元", "总费用")
            if out["emerg_fee"] is None:
                out["emerg_fee"] = _num(dp, "紧急费_元", "紧急费")
            if out["emerg_kwh"] is None:
                out["emerg_kwh"] = _num(dp, "紧急量_kWh", "紧急购电量_kWh", "紧急量")
            if out["comparison"] is None:
                out["comparison"] = used_cmp
            if out["dp_gain"] is None and out["yearly_total"] is not None:
                same = [d for d in rows if d.get("比较方式") == used_cmp]
                an = next((d for d in same if "解析" in d.get("执行器", "")), None)
                t_an = _num(an, "总费用_元") if an else None
                if t_an is not None:
                    out["dp_gain"] = t_an - out["yearly_total"]

    rows = _csv_rows(Q2_NOSTOR_CSV)
    if rows:
        sources.append(str(Q2_NOSTOR_CSV))
        hit = next((d for d in rows if "无储能购电费" in d.get("项目", "")), None)
        if hit is not None:
            out["baseline_fee"] = _num(hit, "数值")

    rows = _csv_rows(Q2_ACCURACY_CSV)
    if rows:
        sources.append(str(Q2_ACCURACY_CSV))
        warm_m = _warmup_month_labels()
        sel = [d for d in rows if d.get("月份", "") not in warm_m] or rows
        tot_w = tot_v = 0.0
        for d in sel:
            v = _num(d, "光伏MAE_kW", "光伏_MAE_kW")
            w = _num(d, "天数") or 1.0
            if v is not None:
                tot_v += v * w
                tot_w += w
        if tot_w > 0:
            out["pv_mae_kw"] = tot_v / tot_w

    if out["yearly_total"] is None:
        for p in q2_report_candidates():
            if not p.is_file():
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            v = (_grab_after(txt, "全年总费用", 1e6, 1e8)
                 or _grab_after(txt, "全年账单", 1e6, 1e8))
            if v is not None:
                out["yearly_total"] = v
                sources.append(str(p))
            break

    out["source"] = " | ".join(sources) if sources else None
    return out
