#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import shutil
import sys
from datetime import date as _date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "代码"
RAW_INFO_DIR = PROJECT_DIR / "原始数据说明"
PROCESSED_DIR = PROJECT_DIR / "处理后数据"
RESULT_DIR = PROJECT_DIR / "模型结果"
FIGURE_DIR = PROJECT_DIR / "模型结果图"
REPORT_DIR = PROJECT_DIR / "报告"
LOG_DIR = PROJECT_DIR / "日志"
CONFIG_DIR = PROJECT_DIR / "配置"
SUBMIT_DIR = PROJECT_DIR / "提交结果"
SOLVE_LOG_DIR = PROJECT_DIR / "求解日志"
DIAG_DIR = PROJECT_DIR / "内部诊断" / "非论文图"
SNAPSHOT_DIR = PROJECT_DIR / "_快照_第三题代码"

ALL_DIRS = [
    CODE_DIR, RAW_INFO_DIR, PROCESSED_DIR, RESULT_DIR, FIGURE_DIR,
    REPORT_DIR, LOG_DIR, CONFIG_DIR, SUBMIT_DIR, SOLVE_LOG_DIR, DIAG_DIR,
]

ROOT_DIR = PROJECT_DIR.parent
REF_Q1_DIR = ROOT_DIR / "第一问最终版"
REF_Q2_DIR = ROOT_DIR / "第二问最终版"
REF_Q3_DIR = ROOT_DIR / "第三问最终版"


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def _find_attachment(filename, base: Path | None = None) -> Path:
    filename = Path(filename)
    env = os.environ.get("CUMCM_ATTACHMENT_DIR", "").strip()
    if env and (Path(env) / filename).is_file():
        return Path(env) / filename
    rel = Path("CUMCM2026Problems") / "C题" / "附件" / filename
    start = base or PROJECT_DIR
    for p in [start, *start.parents]:
        c = p / rel
        if c.is_file():
            return c
    raise FileNotFoundError(f"未找到原始附件：{filename}（可设 CUMCM_ATTACHMENT_DIR）")


ATTACHMENT_DIR = _find_attachment("附件1.xlsx").parent
ATTACHMENT1_PATH = ATTACHMENT_DIR / "附件1.xlsx"
ATTACHMENT2_PATH = ATTACHMENT_DIR / "附件2.xlsx"
ATTACHMENT3_PATH = ATTACHMENT_DIR / "附件3.xlsx"
ATTACHMENT4_PATH = ATTACHMENT_DIR / "附件4.xlsx"
RESULT42_TEMPLATE = ATTACHMENT_DIR / "附件5" / "result4-2.xlsx"
RESULT43_TEMPLATE = ATTACHMENT_DIR / "附件5" / "result4-3.xlsx"

REF_Q2_MATRIX_NPZ = REF_Q2_DIR / "处理后数据" / "附件二_矩阵数据.npz"
REF_Q2_SCENARIO_NPZ = REF_Q2_DIR / "处理后数据" / "附件二_情景库.npz"

SNAPSHOT_SOURCES = [
    "代码/_comm3.py",
    "代码/_policy3.py",
    "代码/05_求解阶段0计划.py",
    "代码/07_阶段内DP执行器.py",
    "代码/02_处理附件三基础数据.py",
]

DELTA_HOURS = 1.0 / 6.0
INTERVAL_MINUTES = 10
PERIODS_PER_DAY = 144
T = PERIODS_PER_DAY
TIMESTAMP_INTERPRETATION = "区间终点（标签表示该10分钟区间的结束时刻）"

YEAR = 2025
SCORE_START = "2025-02-01"
SCORE_END = "2025-12-31"
N_WARMUP_DAYS = 31
N_SCORE_DAYS = 334

TAU_HOURS = (0, 6, 12, 18)
TAU_PERIOD_INDEX = (0, 36, 72, 108)

LOOKAHEAD = PERIODS_PER_DAY

BATTERY = {
    "rated_capacity_kwh": 12000.0,
    "min_energy_kwh": 1200.0,
    "max_energy_kwh": 10800.0,
    "initial_energy_kwh": 6000.0,
    "max_charge_power_kw": 5000.0,
    "max_discharge_power_kw": 5000.0,
    "charge_efficiency": 0.9,
    "discharge_efficiency": 0.9,
}
E_MIN = BATTERY["min_energy_kwh"]
E_MAX = BATTERY["max_energy_kwh"]
E_INIT = BATTERY["initial_energy_kwh"]
ETA_C = BATTERY["charge_efficiency"]
ETA_D = BATTERY["discharge_efficiency"]
ETA = ETA_C
P_MAX_KW = BATTERY["max_charge_power_kw"]
S_PERIOD_KWH = P_MAX_KW * DELTA_HOURS

RHO_DOWN = 0.5
RHO_UP = 1.5
EMERG_MULT = 5.0

NU_FIXED = 0.4815481481481481
NU_PRICE_POINTS = 30

CONFIG = {
    "source_mode": "real",
    "price_information": "revealed_at_delivery_slot",
    "settlement": "B_final_relative_to_midnight_plan",
    "stage0_mode_43": "plan",
    "price_history_days": 35,
    "scenario_max_count": 30,
    "soc_grid_kwh": 6.0,
    "price_level_floor": 0.01,
    "price_point_floor": 0.0001,
    "ar_rho_bounds": (0.0, 0.99),
    "price_method_42": "net_load_regression",
    "price_method_43": "ar1",
    "purchase_update_hours_42": (),
    "purchase_update_hours_43": (6, 12, 18),
    "value_update_hours": (0, 6, 12, 18),
    "lookahead_hours": 24,
    "tail_model": "scenario_recourse_approximation",
    "warmup_policy": "common_causal_policy",
    "random_seed": 20260912,
}
M_SCENARIOS = CONFIG["scenario_max_count"]
PRICE_HISTORY_DAYS = CONFIG["price_history_days"]
PRICE_LEVEL_FLOOR = CONFIG["price_level_floor"]
PRICE_POINT_FLOOR = CONFIG["price_point_floor"]
SOC_GRID_KWH = CONFIG["soc_grid_kwh"]
AR_RHO_BOUNDS = CONFIG["ar_rho_bounds"]
SEED = int(CONFIG["random_seed"])

PRICE_MATRIX_NPZ = PROCESSED_DIR / "附件四_电价矩阵.npz"
PRICE_TIME_MD = REPORT_DIR / "附件四_数据处理与时间映射报告.md"

A3_FORECAST_NPZ = PROCESSED_DIR / "附件三_预报矩阵.npz"
A3_SCENARIO_NPZ = PROCESSED_DIR / "附件三_情景库.npz"
A3_REPORT_MD = REPORT_DIR / "附件三_独立处理报告.md"

PRICE_FORECAST_NPZ = PROCESSED_DIR / "第四问_价格预测快照.npz"
PRICE_DIAG_CSV = RESULT_DIR / "第四问_价格预测精度.csv"
PRICE_REPORT_MD = REPORT_DIR / "第四问_电价预测报告.md"

JOINT_SCENARIO_NPZ = PROCESSED_DIR / "第四问_联合情景库.npz"
JOINT_ORIGIN_CSV = RESULT_DIR / "第四问_情景来源索引.csv"
JOINT_REPORT_MD = REPORT_DIR / "第四问_联合情景报告.md"

WARMUP_NPZ = PROCESSED_DIR / "第四问_共同热启动.npz"
WARMUP_REPORT_MD = REPORT_DIR / "第四问_共同热启动报告.md"

BACKTEST_42_NPZ = RESULT_DIR / "第四问_4-2全年回测.npz"
BACKTEST_43_NPZ = RESULT_DIR / "第四问_4-3全年回测.npz"
YEARLY_CSV = RESULT_DIR / "第四问_年度汇总.csv"
MONTHLY_CSV = RESULT_DIR / "第四问_月度费用分解.csv"
ABLATION_CSV = RESULT_DIR / "第四问_对照与敏感性.csv"
SPEC_SLOTS_CSV = RESULT_DIR / "第四问_指定时段购电量.csv"
SPEC_DAY_CSV = RESULT_DIR / "第四问_指定日期费用.csv"
SPEC_BATT_CSV = RESULT_DIR / "第四问_指定日期充放电.csv"
SPEC_EMERG_CSV = RESULT_DIR / "第四问_指定日期紧急购电事件.csv"

RESULT42_XLSX = SUBMIT_DIR / "result4-2.xlsx"
RESULT43_XLSX = SUBMIT_DIR / "result4-3.xlsx"

ACCEPT_MD = REPORT_DIR / "第四问_独立验收报告.md"
SOURCE_FREEZE_CSV = RAW_INFO_DIR / "第四问_来源指纹.csv"

SPEC_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
SPEC_SLOTS = [
    ("10:00", "10:10"), ("12:00", "12:10"), ("14:00", "14:10"),
    ("16:00", "16:10"), ("18:00", "18:10"), ("20:00", "20:10"),
]
BATT_BLOCKS = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
               "12:00-16:00", "16:00-20:00", "20:00-24:00"]


def compute_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv_utf8_sig(path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def write_text_utf8(path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fmt_num(v, nd: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv == int(fv) and abs(fv) < 1e15:
        return str(int(fv))
    return f"{fv:.{nd}f}".rstrip("0").rstrip(".")


def parse_time_label_to_minutes(value) -> int:
    if isinstance(value, dt_time):
        return value.hour * 60 + value.minute
    if isinstance(value, str):
        s = value.strip()
        if "+1" in s:
            base = s.split("+")[0].strip()
            h, m = (int(x) for x in base.split(":")[:2])
            return 24 * 60 + h * 60 + m
        parts = s.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    if isinstance(value, (int, float)):
        return int(round(float(value) * 24 * 60))
    raise TypeError(f"unexpected time label: {type(value).__name__} = {value!r}")


def minutes_to_hhmm(mins, use_24: bool = False) -> str:
    mins = int(round(mins))
    if use_24 and mins == 1440:
        return "24:00"
    return f"{(mins // 60) % 24:02d}:{mins % 60:02d}"


def slot_to_period_index(slot: str) -> int:
    return parse_time_label_to_minutes(slot) // INTERVAL_MINUTES


def load_q2_matrix() -> dict:
    Z = np_load(REF_Q2_MATRIX_NPZ)
    return {k: Z[k] for k in Z.files}


def load_q2_scenarios() -> dict:
    Z = np_load(REF_Q2_SCENARIO_NPZ)
    return {k: Z[k] for k in Z.files}


def np_load(path):
    import numpy as np
    return np.load(path, allow_pickle=False)


def ref_price144() -> "object":
    import numpy as np
    import pandas as pd

    df = pd.read_excel(ATTACHMENT1_PATH, sheet_name=0, header=0)
    cols = [str(c).strip() for c in df.columns]
    if "电价" not in cols:
        raise ValueError(f"附件一缺少「电价」列：{cols}")
    if df.shape[0] != PERIODS_PER_DAY:
        raise ValueError(f"附件一电价行数应为 {PERIODS_PER_DAY}，实际 {df.shape[0]}")
    arr = np.asarray(df["电价"].to_numpy(float), float).reshape(-1)

    reuse = np.asarray(load_q2_matrix()["price"], float).reshape(-1)
    if reuse.size != arr.size:
        raise ValueError(f"复用字段长度 {reuse.size} 与附件一 {arr.size} 不一致")
    dev = float(np.abs(arr - reuse).max())
    if dev > 1e-12:
        raise ValueError(f"附件一电价与第二问复用字段不一致，最大偏差 {dev:.3e} 元/kWh")
    return arr


def freeze_sources(verbose: bool = True) -> list[tuple[str, str]]:
    items: list[tuple[Path, str]] = []
    for p in [ATTACHMENT1_PATH, ATTACHMENT2_PATH, ATTACHMENT3_PATH, ATTACHMENT4_PATH,
              RESULT42_TEMPLATE, RESULT43_TEMPLATE,
              REF_Q2_MATRIX_NPZ, REF_Q2_SCENARIO_NPZ]:
        if p.is_file():
            items.append((p, "共享原始/只读"))
    for rel in SNAPSHOT_SOURCES:
        p = REF_Q3_DIR / rel
        if p.is_file():
            items.append((p, "第三题源码"))

    rows: list[tuple[str, str]] = []
    for p, kind in items:
        sha = compute_sha256(p)
        try:
            rel = str(p.relative_to(ROOT_DIR))
        except ValueError:
            rel = str(p)
        rows.append((f"{kind}|{rel}", sha))

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_rows = []
    for rel in SNAPSHOT_SOURCES:
        src = REF_Q3_DIR / rel
        if not src.is_file():
            continue
        dst = SNAPSHOT_DIR / Path(rel).name
        sha = compute_sha256(src)
        if not dst.exists():
            shutil.copy2(src, dst)
        snap_rows.append((Path(rel).name, sha, str(src.relative_to(ROOT_DIR))))
    write_csv_utf8_sig(SNAPSHOT_DIR / "快照指纹清单.csv",
                       ("快照文件", "SHA256", "来源路径"), snap_rows)

    write_csv_utf8_sig(SOURCE_FREEZE_CSV, ("来源项", "SHA256"), rows)
    if verbose:
        print(f"  来源指纹已写：{SOURCE_FREEZE_CSV.relative_to(PROJECT_DIR)}"
              f"（{len(rows)} 项）")
        print(f"  第三题源码快照：{SNAPSHOT_DIR.name}/（{len(snap_rows)} 个文件，"
              f"已存在则不覆盖）")
    return rows


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in ["Songti SC", "PingFang HK", "Hiragino Sans GB", "Heiti TC",
              "STHeiti", "Arial Unicode MS"]:
        if c in available:
            matplotlib.rcParams["font.family"] = [c]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib


def load_module(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, CODE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_attachment4(path=None) -> dict:
    import numpy as np
    import pandas as pd

    path = Path(path or ATTACHMENT4_PATH)
    df = pd.read_excel(path, sheet_name=0, header=0)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = df.columns[0]
    time_labels = list(df.columns[1:])
    if len(time_labels) != PERIODS_PER_DAY:
        raise ValueError(f"附件四时间列数应为 {PERIODS_PER_DAY}，实际 {len(time_labels)}")

    raw_dates = pd.to_datetime(df[date_col], errors="coerce")
    raw_dates = raw_dates.ffill()
    if raw_dates.isna().any():
        raise ValueError("附件四日期列存在无法解析且无法前向填充的单元格")
    dates = np.array([d.strftime("%Y-%m-%d") for d in raw_dates], dtype="<U10")

    price = df.iloc[:, 1:].to_numpy(float)

    ends = np.array([parse_time_label_to_minutes(x) for x in time_labels], int)
    starts = ends - INTERVAL_MINUTES
    if not np.array_equal(ends, np.arange(1, PERIODS_PER_DAY + 1) * INTERVAL_MINUTES):
        raise ValueError(f"附件四时间标签非「区间终点」等差序列：{time_labels[:3]}…{time_labels[-1]}")

    day_index = np.array([i for i in range(dates.size) if dates[i] >= SCORE_START], int)

    return {
        "dates": dates,
        "raw_time_labels": np.array(time_labels, dtype=object),
        "slot_start_minute": starts,
        "slot_end_minute": ends,
        "price_actual": price,
        "day_index": np.arange(dates.size, dtype=int),
        "score_day_index": day_index,
        "source": str(path),
    }


def read_attachment3(path=None) -> dict:
    import numpy as np
    import pandas as pd

    path = Path(path or ATTACHMENT3_PATH)
    df = pd.read_excel(path, sheet_name=0, header=0)
    if df.shape[1] != 26:
        raise ValueError(f"附件三列数异常：{df.shape[1]}（预期 26）")

    raw = pd.to_datetime(df.iloc[:, 0], errors="coerce").ffill()
    if raw.isna().any():
        raise ValueError("附件三日期列存在无法前向填充的空洞")
    dates_all = np.array([x.strftime("%Y-%m-%d") for x in raw], dtype="<U10")
    tau_all = df.iloc[:, 1].map(parse_time_label_to_minutes).to_numpy(int) // 60
    vals = df.iloc[:, 2:26].to_numpy(float)

    udates = np.array(sorted(set(dates_all.tolist())), dtype="<U10")
    d_index = {s: i for i, s in enumerate(udates)}
    t_index = {h: j for j, h in enumerate(TAU_HOURS)}
    N_TAU, LEAD = len(TAU_HOURS), 24
    V = np.full((udates.size, N_TAU, LEAD), np.nan)
    seen = np.zeros((udates.size, N_TAU), dtype=bool)
    for r in range(dates_all.size):
        ti = t_index.get(int(tau_all[r]))
        if ti is None:
            raise ValueError(f"附件三出现非 {TAU_HOURS} 的发布时刻：{tau_all[r]}")
        i = d_index[dates_all[r]]
        if seen[i, ti]:
            raise ValueError(f"附件三重复行：{dates_all[r]} τ={tau_all[r]}")
        V[i, ti] = vals[r]
        seen[i, ti] = True

    return {
        "dates": udates,
        "tau_hours": np.array(TAU_HOURS, dtype=int),
        "V_raw_kw": V,
        "complete_mask": seen,
        "source": str(path),
    }


def _lead_to_abs_map():
    import numpy as np
    return np.stack([(np.arange(24) - tau) % 24 for tau in TAU_HOURS])


LEAD_TO_ABS = _lead_to_abs_map()


def lead_to_absolute(V_hourly):
    import numpy as np
    a = np.asarray(V_hourly, float)
    if a.shape[-1] != 24:
        raise ValueError(f"末维应为 24，实际 {a.shape}")
    if a.ndim == 3:
        idx = LEAD_TO_ABS[None, :, :]
    elif a.ndim == 4:
        idx = LEAD_TO_ABS[None, :, None, :]
    else:
        raise ValueError(f"维度应为 3 或 4，实际 {a.ndim}")
    return np.take_along_axis(a, np.broadcast_to(idx, a.shape), axis=-1)


def disaggregate_piecewise(V_hourly):
    import numpy as np
    a = np.asarray(V_hourly, float)
    reps = int(round(1.0 / DELTA_HOURS))
    return np.repeat(a, reps, axis=-1)[..., :PERIODS_PER_DAY]


def disaggregate_linear(V_hourly):
    import numpy as np
    a = np.asarray(V_hourly, float)
    n_h = a.shape[-1]
    f = np.concatenate([a, a[..., -1:]], axis=-1)
    reps = int(round(1.0 / DELTA_HOURS))
    r = np.arange(reps, dtype=float) / reps
    left = np.repeat(a, reps, axis=-1)
    right = np.repeat(f[..., 1:], reps, axis=-1)
    w = np.tile(r, n_h)
    return ((1.0 - w) * left + w * right)[..., :PERIODS_PER_DAY]


DISAGG_PIECEWISE = "piecewise_constant"
DISAGG_LINEAR = "linear_interp"


def disaggregate(V_hourly, mode=DISAGG_PIECEWISE):
    if mode == DISAGG_PIECEWISE:
        return disaggregate_piecewise(V_hourly)
    if mode == DISAGG_LINEAR:
        return disaggregate_linear(V_hourly)
    raise ValueError(f"未知展开口径：{mode}")


def pv_hourly_actual_from_q2(Z: dict):
    import numpy as np
    pv = np.asarray(Z["pv_kw"], float)
    reps = int(round(1.0 / DELTA_HOURS))
    return pv.reshape(pv.shape[0], 24, reps).mean(axis=2)
