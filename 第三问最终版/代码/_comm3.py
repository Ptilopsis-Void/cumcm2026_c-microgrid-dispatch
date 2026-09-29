from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "代码"
RAW_INFO_DIR = PROJECT_DIR / "原始数据说明"
CONFIG_DIR = PROJECT_DIR / "配置"

Q2_SOURCE = (os.environ.get("Q3_Q2_SOURCE", "real") or "real").strip().lower()
SANDBOX = Q2_SOURCE in ("stub", "off")
OUT_ROOT = PROJECT_DIR / "_骨架自检" if SANDBOX else PROJECT_DIR


def out_root(source: str | None = None) -> Path:
    src = (source or Q2_SOURCE or "real").strip().lower()
    return PROJECT_DIR / "_骨架自检" if src in ("stub", "off") else PROJECT_DIR

PROCESSED_DIR = OUT_ROOT / "处理后数据"
RESULT_DIR = OUT_ROOT / "模型结果"
FIGURE_DIR = OUT_ROOT / "模型结果图"
DATA_CHECK_DIR = OUT_ROOT / "数据检查图"
REPORT_DIR = OUT_ROOT / "报告"
LOG_DIR = OUT_ROOT / "日志"
SUBMIT_DIR = OUT_ROOT / "提交结果"
SOLVE_LOG_DIR = OUT_ROOT / "求解日志"

ALL_DIRS = [
    OUT_ROOT, CODE_DIR, RAW_INFO_DIR, PROCESSED_DIR, RESULT_DIR, FIGURE_DIR,
    DATA_CHECK_DIR, REPORT_DIR, LOG_DIR, CONFIG_DIR, SUBMIT_DIR, SOLVE_LOG_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def _find_attachment(filename) -> Path:
    rel = Path("CUMCM2026Problems") / "C题" / "附件" / Path(filename)
    env = os.environ.get("CUMCM_ATTACHMENT_DIR", "").strip()
    if env and (Path(env) / Path(filename)).is_file():
        return Path(env) / Path(filename)
    for p in [PROJECT_DIR, *PROJECT_DIR.parents]:
        c = p / rel
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"未找到原始附件：{filename}（可设置环境变量 CUMCM_ATTACHMENT_DIR）")


ATTACHMENT1_PATH = _find_attachment("附件1.xlsx")
ATTACHMENT2_PATH = _find_attachment("附件2.xlsx")
ATTACHMENT3_PATH = _find_attachment("附件3.xlsx")
ATTACHMENT5_R3_PATH = _find_attachment(Path("附件5") / "result3.xlsx")
ATTACHMENT_DIR = ATTACHMENT1_PATH.parent

Q2_DIR: Path = Path()
Q2_DIR_SOURCE: str = "none"
Q2_MATRIX_NPZ: Path = Path()
Q2_SCENARIO_NPZ: Path = Path()


def _load_sibling(filename: str, modname: str):
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(modname, CODE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_Q2_ADAPTER = _load_sibling("_q2_adapter.py", "q3_q2_adapter")

Q2_CORE_CSV = _Q2_ADAPTER.Q2_CORE_CSV
Q2_HEADLINE_COMPARISON = _Q2_ADAPTER.Q2_HEADLINE_COMPARISON

Q2_DIR = _Q2_ADAPTER.Q2_DIR
Q2_DIR_SOURCE = _Q2_ADAPTER.Q2_DIR_SOURCE
Q2_MATRIX_NPZ = _Q2_ADAPTER.MATRIX_NPZ
Q2_SCENARIO_NPZ = _Q2_ADAPTER.SCENARIO_NPZ

CONTRACT_MATRIX = _Q2_ADAPTER.CONTRACT_MATRIX
CONTRACT_SCENARIO = _Q2_ADAPTER.CONTRACT_SCENARIO
SCENARIO_VOID = _Q2_ADAPTER.SCENARIO_VOID
Q2Unavailable = _Q2_ADAPTER.Q2Unavailable
Q2ContractError = _Q2_ADAPTER.Q2ContractError


class _Q2Facade:

    @staticmethod
    def matrix():
        return _Q2_ADAPTER.matrix()

    @staticmethod
    def scenarios():
        return _Q2_ADAPTER.scenarios()

    @staticmethod
    def mode() -> str:
        return _Q2_ADAPTER.resolve_mode()

    @staticmethod
    def is_stub() -> bool:
        return _Q2_ADAPTER.is_stub()

    @staticmethod
    def validate(strict: bool = False) -> dict:
        return _Q2_ADAPTER.validate(strict=strict)

    @staticmethod
    def contract_table() -> list:
        return _Q2_ADAPTER.contract_table()

    @staticmethod
    def headline_numbers() -> dict:
        return _Q2_ADAPTER.q2_headline_numbers()

    @staticmethod
    def core_numbers() -> dict:
        return _Q2_ADAPTER.q2_core_numbers()

    @staticmethod
    def reset_cache() -> None:
        _Q2_ADAPTER.reset_cache()


Q2 = _Q2Facade()

DELTA_HOURS = 1.0 / 6.0
INTERVAL_MINUTES = 10
PERIODS_PER_DAY = 144
TIMESTAMP_INTERPRETATION = "区间终点（标签表示该10分钟区间的结束时刻）"

YEAR = 2025
WARMUP_START = "2025-01-01"
WARMUP_END = "2025-01-31"
SCORE_START = "2025-02-01"
SCORE_END = "2025-12-31"
N_WARMUP_DAYS = 31
N_SCORE_DAYS = 334

TAU_HOURS = (0, 6, 12, 18)
TAU_PERIOD_INDEX = (0, 36, 72, 108)
N_TAU = len(TAU_HOURS)
LEAD_HOURS = 24
TAU_LABELS = ("0:00", "6:00", "12:00", "18:00")

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

OPERATION = {
    "allow_grid_charging": True,
    "allow_grid_export": False,
    "allow_pv_curtailment": True,
}

M_SCENARIOS = 30
RESIDUAL_WINDOW_DAYS = 30
LOAD_LOOKBACK_DAYS = 35
LOAD_FALLBACK_DAYS = 7
PV_LOOKBACK_DAYS = 7

RHO_DOWN = 0.5
RHO_UP = 1.5
EMERG_MULT = 5.0

NU_PRICE_POINTS = 30
NU_VALUE = 0.4815481481481481

DISAGG_PIECEWISE = "piecewise_constant"
DISAGG_LINEAR = "linear_interp"

ABLATION_ARMS = ("S0", "S06", "S0612", "S061218", "S_all_plus", "PF")
ARM_TAUS = {
    "S0": (),
    "S06": (6,),
    "S0612": (6, 12),
    "S061218": (6, 12, 18),
    "S_all_plus": (6, 12, 18),
    "PF": (6, 12, 18),
    "S_all+_raw": (6, 12, 18),
}

SPEC_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
SPEC_SLOTS = [
    ("10:00", "10:10"), ("12:00", "12:10"), ("14:00", "14:10"),
    ("16:00", "16:10"), ("18:00", "18:10"), ("20:00", "20:10"),
]

RAW_STRUCT_CSV = RAW_INFO_DIR / "附件三工作表结构.csv"
RAW_STAT_CSV = RAW_INFO_DIR / "附件三原始数据统计.csv"
RAW_MAPPING_CSV = RAW_INFO_DIR / "附件三字段映射表.csv"

V_FORECAST_NPZ = PROCESSED_DIR / "附件三_预报矩阵.npz"
V_SCENARIO_NPZ = PROCESSED_DIR / "附件三_情景库.npz"

STAGE0_NPZ = RESULT_DIR / "第三问_阶段0计划.npz"
ROLLING_NPZ = RESULT_DIR / "第三问_滚动调整.npz"
BACKTEST_NPZ = RESULT_DIR / "第三问_全年回测.npz"
ABLATION_CSV = RESULT_DIR / "第三问_消融实验对照表.csv"
DAILY_CSV = RESULT_DIR / "第三问_逐日费用三分解.csv"
MONTHLY_CSV = RESULT_DIR / "第三问_月度费用三分解.csv"
YEARLY_CSV = RESULT_DIR / "第三问_年度汇总.csv"
PRECISION_CSV = RESULT_DIR / "第三问_预报精度表.csv"
CHECK_CSV = RESULT_DIR / "第三问_约束校验结果.csv"

REPORT_RAW_MD = REPORT_DIR / "附件三原始数据检查报告.md"
REPORT_PROC_MD = REPORT_DIR / "附件三数据处理报告.md"
REPORT_ACCEPT_MD = REPORT_DIR / "附件三数据处理验收表.md"
REPORT_TIME_MD = REPORT_DIR / "附件三时间标签映射说明.md"
REPORT_VERIFY_MD = REPORT_DIR / "第三问_附件三口径核验报告.md"
REPORT_SCEN_MD = REPORT_DIR / "第三问预测精度与情景构造报告.md"
REPORT_STAGE0_MD = REPORT_DIR / "第三问_阶段0计划报告.md"
REPORT_ROLL_MD = REPORT_DIR / "第三问_滚动调整报告.md"
REPORT_EXEC_MD = REPORT_DIR / "第三问_执行器报告.md"
REPORT_BACK_MD = REPORT_DIR / "第三问_全年回测与结算报告.md"
REPORT_ABL_MD = REPORT_DIR / "第三问_消融实验报告.md"
REPORT_SPEC_MD = REPORT_DIR / "第三问_指定日期明细报告.md"
REPORT_RESULT3_MD = REPORT_DIR / "第三问_result3生成与校验报告.md"
REPORT_OVERVIEW_MD = REPORT_DIR / "第三题_结果总览.md"

LOG_TXT = LOG_DIR / "附件三数据处理运行日志.txt"
ANOMALY_CSV = LOG_DIR / "附件三数据异常记录.csv"

RESULT3_XLSX = SUBMIT_DIR / "result3.xlsx"

SHEET_PLAN3 = "计划购电量"
SHEET_ADJUST = "调整购电量"
SHEET_BATT = "充放电量"
SHEET_EMERG = "紧急购电量"

RESULT3_MAPPING_NOTE = (
    "官方模板『计划购电量』/『调整购电量』表头列标签为 0:10-0:20、…、0:00-0:10+1，"
    "比模型实际区间整体晚一个时段；本方案按『列序 = 时间序』填写："
    "第 N 个数据列（N=1..144，Excel 列 B..EO）对应模型第 N 个时段 [ (N-1)Δt, NΔt )。"
    "表头文字一律不改写。"
)
BATT_BLOCKS = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
               "12:00-16:00", "16:00-20:00", "20:00-24:00"]

def compute_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv_utf8_sig(path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    hdr = list(header)
    n = len(hdr)
    body = [list(r) for r in rows]
    for i, r in enumerate(body):
        if len(r) != n:
            raise ValueError(
                f"{Path(path).name}：第 {i + 1} 行有 {len(r)} 列，表头有 {n} 列；"
                "列错位会静默污染结果，故直接失败。")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in body:
            w.writerow(r)


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
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.{nd}f}".rstrip("0").rstrip(".")


fmt = fmt_num


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
    raise TypeError(f"unexpected time label type: {type(value).__name__} = {value!r}")


def minutes_to_hhmm(mins, use_24: bool = False) -> str:
    mins = int(round(mins))
    if use_24 and mins == 1440:
        return "24:00"
    h = (mins // 60) % 24
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def period_bounds(t: int):
    return (t - 1) * INTERVAL_MINUTES, t * INTERVAL_MINUTES


def slot_to_period_index(slot: str) -> int:
    return parse_time_label_to_minutes(slot) // INTERVAL_MINUTES + 1


def interval_label(t: int) -> str:
    a, b = period_bounds(t)
    return f"{minutes_to_hhmm(a)}-{minutes_to_hhmm(b, use_24=True)}"


def parse_tau_hours(value) -> int:
    if isinstance(value, dt_time):
        return int(value.hour)
    if isinstance(value, str):
        s = value.strip().replace("：", ":")
        return int(s.split(":")[0])
    if isinstance(value, (int, float)):
        f = float(value)
        if f <= 1.0 + 1e-9:
            return int(round(f * 24)) % 24
        return int(round(f)) % 24
    raise TypeError(f"unexpected 预报时刻 type: {type(value).__name__} = {value!r}")


def normalize_date_str(value) -> str:
    if isinstance(value, (dt_time,)):
        raise TypeError("日期列不应出现纯时间对象")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none"}:
        raise ValueError("空日期")
    s = s.split(" ")[0].split("T")[0]
    parts = s.replace("/", "-").split("-")
    y, m, d = (int(x) for x in parts[:3])
    return f"{y:04d}-{m:02d}-{d:02d}"


def read_attachment3(path=None) -> dict:
    import numpy as np
    import pandas as pd

    path = Path(path or ATTACHMENT3_PATH)
    df = pd.read_excel(path, sheet_name=0, header=0)
    cols = list(df.columns)
    if len(cols) != 26:
        raise ValueError(f"附件 3 列数异常：{len(cols)}（预期 26）")

    raw_date = df.iloc[:, 0].copy()
    filled = []
    last = None
    for v in raw_date.tolist():
        try:
            s = normalize_date_str(v)
            last = s
            filled.append(s)
        except (ValueError, TypeError):
            if last is None:
                raise ValueError("附件 3 第 1 行日期为空，无法 ffill")
            filled.append(last)
    dates_long = np.array(filled, dtype="<U10")

    tau_long = np.array([parse_tau_hours(v) for v in df.iloc[:, 1].tolist()],
                        dtype=int)

    vals_long = df.iloc[:, 2:26].to_numpy(dtype=float)

    udates = sorted(set(dates_long.tolist()))
    d_index = {s: i for i, s in enumerate(udates)}
    t_index = {h: j for j, h in enumerate(TAU_HOURS)}
    V = np.full((len(udates), N_TAU, LEAD_HOURS), np.nan)
    seen = np.zeros((len(udates), N_TAU), dtype=bool)
    for r in range(dates_long.size):
        di = d_index[dates_long[r]]
        ti = t_index[int(tau_long[r])]
        V[di, ti] = vals_long[r]
        seen[di, ti] = True

    return {
        "dates": np.array(udates, dtype="<U10"),
        "tau_hours": np.array(TAU_HOURS, dtype=int),
        "V_raw_kw": V,
        "complete_mask": seen,
        "long": {"date": dates_long, "tau_hour": tau_long, "values": vals_long},
        "source": path,
    }


def _lead_to_abs_map():
    import numpy as np

    return np.stack([(np.arange(24) - tau) % 24 for tau in TAU_HOURS])


LEAD_TO_ABS = _lead_to_abs_map()


def lead_to_absolute(V_hourly):
    import numpy as np

    a = np.asarray(V_hourly, dtype=float)
    if a.shape[-1] != 24:
        raise ValueError(f"末维应为 24，实际 {a.shape}")
    if a.ndim == 3:
        idx = LEAD_TO_ABS[None, :, :]
    elif a.ndim == 4:
        idx = LEAD_TO_ABS[None, :, None, :]
    else:
        raise ValueError(f"维度应为 3 或 4，实际 {a.ndim}")
    return np.take_along_axis(a, np.broadcast_to(idx, a.shape), axis=-1)


def scenario_pv_energy(V_hourly_abs, d, ti):
    import numpy as np

    a = np.asarray(V_hourly_abs, dtype=float)
    part = a[d, ti] if a.ndim == 4 else a[d]
    if part.ndim == 1:
        part = part[None, :]
    return np.repeat(part, int(round(1.0 / DELTA_HOURS)), axis=-1) \
        [..., :PERIODS_PER_DAY] * DELTA_HOURS


def disaggregate_piecewise(V_hourly):
    import numpy as np

    a = np.asarray(V_hourly, dtype=float)
    reps = int(round(1.0 / DELTA_HOURS))
    return np.repeat(a, reps, axis=-1)[..., :PERIODS_PER_DAY]


def disaggregate_linear(V_hourly):
    import numpy as np

    a = np.asarray(V_hourly, dtype=float)
    n_h = a.shape[-1]
    f = np.concatenate([a, a[..., -1:]], axis=-1)
    reps = int(round(1.0 / DELTA_HOURS))
    r = np.arange(reps, dtype=float) / reps
    left = np.repeat(a, reps, axis=-1)
    right = np.repeat(f[..., 1:], reps, axis=-1)
    w = np.tile(r, n_h)
    out = (1.0 - w) * left + w * right
    return out[..., :PERIODS_PER_DAY]


def disaggregate(V_hourly, mode: str = DISAGG_PIECEWISE):
    if mode == DISAGG_PIECEWISE:
        return disaggregate_piecewise(V_hourly)
    if mode == DISAGG_LINEAR:
        return disaggregate_linear(V_hourly)
    raise ValueError(f"未知展开口径：{mode}")


def adjust_cost(price, p, q=None, u=None, v=None, b=None):
    import numpy as np

    price = np.asarray(price, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if q is not None:
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q[None, :]
        u = np.maximum(p[None, :] - q, 0.0)
        v = np.maximum(q - p[None, :], 0.0)
    else:
        u = np.atleast_2d(np.asarray(u, dtype=float))
        v = np.atleast_2d(np.asarray(v, dtype=float))
        q = p[None, :] - u + v
    if b is None:
        b = np.zeros_like(q)
    else:
        b = np.atleast_2d(np.asarray(b, dtype=float))

    plan_fee = float(np.sum(price[None, :] * np.minimum(p[None, :], q)))
    fee_down = float(np.sum(RHO_DOWN * price[None, :] * u))
    fee_up = float(np.sum(RHO_UP * price[None, :] * v))
    emerg = float(np.sum(EMERG_MULT * price[None, :] * b))
    return {
        "plan_fee": plan_fee,
        "adj_fee_down": fee_down,
        "adj_fee_up": fee_up,
        "adj_fee": fee_down + fee_up,
        "emerg_fee": emerg,
        "total": plan_fee + fee_down + fee_up + emerg,
    }


def compute_nu(path=None) -> float:
    import pandas as pd

    df = pd.read_excel(path or ATTACHMENT1_PATH, sheet_name=0, header=0)
    price = df.iloc[:, 1].to_numpy(float)[:NU_PRICE_POINTS]
    return float(price.mean() / ETA)


def load_price_and_matrix():
    import pandas as pd

    df = pd.read_excel(ATTACHMENT1_PATH, sheet_name=0, header=0)
    price = df.iloc[:, 1].to_numpy(float)
    Z = Q2.matrix()
    return price, Z


def setup_chinese_font():
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Songti SC", "PingFang HK", "Hiragino Sans GB", "Heiti TC",
                  "STHeiti", "Arial Unicode MS"]
    chosen = next((c for c in candidates if c in available), None)
    if chosen is not None:
        matplotlib.rcParams["font.family"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    return setup_chinese_font()


def save_figure(fig, stem: str, also_pdf: bool = True):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{stem}.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    if also_pdf:
        try:
            fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        except Exception:
            pass
    return png


class Tee:

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str = "") -> None:
        self.lines.append(str(msg))
        print(msg, flush=True)

    def dump(self, path, tail: str = "") -> None:
        text = "\n".join(self.lines + ([tail] if tail else [])) + "\n"
        write_text_utf8(path, text)

    def markdown(self, title: str, extra: Sequence[str] = ()) -> str:
        return "\n".join([f"# {title}", "", *extra, *self.lines, ""])


def as_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")
