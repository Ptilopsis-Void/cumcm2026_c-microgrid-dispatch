from __future__ import annotations

import csv
import hashlib
import os
from datetime import time as dt_time
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "代码"
RAW_INFO_DIR = PROJECT_DIR / "原始数据说明"
PROCESSED_DIR = PROJECT_DIR / "处理后数据"
RESULT_DIR = PROJECT_DIR / "模型结果"
FIGURE_DIR = PROJECT_DIR / "模型结果图"
DATA_CHECK_DIR = PROJECT_DIR / "数据检查图"
REPORT_DIR = PROJECT_DIR / "报告"
LOG_DIR = PROJECT_DIR / "日志"
CONFIG_DIR = PROJECT_DIR / "配置"
SUBMIT_DIR = PROJECT_DIR / "提交结果"
SOLVE_LOG_DIR = PROJECT_DIR / "求解日志"
INTERNAL_DIAG_DIR = PROJECT_DIR / "内部诊断" / "非论文图"

ALL_DIRS = [
    CODE_DIR, RAW_INFO_DIR, PROCESSED_DIR, RESULT_DIR, FIGURE_DIR,
    DATA_CHECK_DIR, REPORT_DIR, LOG_DIR, CONFIG_DIR, SUBMIT_DIR, SOLVE_LOG_DIR,
    INTERNAL_DIAG_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def _find_attachment(filename: str) -> Path:
    env = os.environ.get("CUMCM_ATTACHMENT_DIR", "").strip()
    if env and (Path(env) / filename).is_file():
        return Path(env) / filename
    rel = Path("CUMCM2026Problems") / "C题" / "附件" / filename
    for p in [PROJECT_DIR, *PROJECT_DIR.parents]:
        c = p / rel
        if c.is_file():
            return c
    raise FileNotFoundError(f"未找到原始附件：{filename}（可设置环境变量 CUMCM_ATTACHMENT_DIR）")


ATTACHMENT1_PATH = _find_attachment("附件1.xlsx")
ATTACHMENT2_PATH = _find_attachment("附件2.xlsx")
ATTACHMENT5_PATH = _find_attachment(Path("附件5") / "result2.xlsx")
ATTACHMENT_DIR = ATTACHMENT1_PATH.parent

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
PV_LOOKBACK_DAYS = 3

NU_PRICE_POINTS = 30

MPC_RHO_CLIP = (0.0, 0.99)
MPC_RHO_WINDOW_DAYS = 30

ALPHA_GRID = (0.0, 0.09645, 0.25, 0.50, 0.75)

RAW_STRUCT_CSV = RAW_INFO_DIR / "附件二工作表结构.csv"
RAW_STAT_CSV = RAW_INFO_DIR / "附件二原始数据统计.csv"
RAW_MAPPING_CSV = RAW_INFO_DIR / "附件二字段映射表.csv"

BASE_LONG_CSV = PROCESSED_DIR / "附件二_第二问基础数据.csv"
BASE_MATRIX_NPZ = PROCESSED_DIR / "附件二_矩阵数据.npz"
SCENARIO_NPZ = PROCESSED_DIR / "附件二_情景库.npz"

DPVALUE_NPZ = RESULT_DIR / "第二问_未来价值函数.npz"
RESERVE_CSV = RESULT_DIR / "第二问_动态保留水平.csv"
EXEC_COMPARE_CSV = RESULT_DIR / "第二问_执行器对照.csv"
YEARLY_CSV = RESULT_DIR / "第二问_年度汇总.csv"
MONTHLY_CSV = RESULT_DIR / "第二问_月度费用节省.csv"
DAILY_CSV = RESULT_DIR / "第二问_逐日费用对照.csv"
RESERVE_SWEEP_CSV = RESULT_DIR / "第二问_固定保留策略扫描.csv"
SPEC_PLAN_CSV = RESULT_DIR / "第二问_指定时段计划购电量.csv"
SPEC_DAY_CSV = RESULT_DIR / "第二问_指定日期全天购电量与费用.csv"
SPEC_COST_CSV = RESULT_DIR / "第二问_指定日期费用分解.csv"
SPEC_EMERG_CSV = RESULT_DIR / "第二问_指定日期紧急购电事件.csv"
SPEC_BATT_CSV = RESULT_DIR / "第二问_指定日期储能充放电与储电量.csv"
CHECK_CSV = RESULT_DIR / "第二问_约束校验结果.csv"
PLOT_DATA_CSV = RESULT_DIR / "第二问_作图数据.csv"

REPORT_MD = REPORT_DIR / "附件二数据处理报告.md"
ACCEPT_MD = REPORT_DIR / "附件二数据处理验收表.md"
TIME_MD = REPORT_DIR / "附件二时间标签映射说明.md"
MODEL_MD = REPORT_DIR / "第二问模型求解报告.md"
MODEL_ACCEPT_MD = REPORT_DIR / "第二问模型结果验收表.md"
SCENARIO_MD = REPORT_DIR / "第二问预测精度与情景构造报告.md"

LOG_TXT = LOG_DIR / "附件二数据处理运行日志.txt"
ANOMALY_CSV = LOG_DIR / "附件二数据异常记录.csv"
SOLVE_LOG_TXT = SOLVE_LOG_DIR / "第二问求解日志.txt"

RESULT2_XLSX = SUBMIT_DIR / "result2.xlsx"

BASE_LONG_FIELDS = [
    "date", "day_index", "interval_index", "original_time_label",
    "interval_start", "interval_end", "delta_hours",
    "price_yuan_per_kwh", "load_kw", "pv_kw", "net_load_kw",
    "load_energy_kwh", "pv_energy_kwh", "net_load_energy_kwh",
]
FORMAL_INPUT_FIELDS = [
    "interval_index", "price_yuan_per_kwh", "load_energy_kwh",
    "pv_energy_kwh", "net_load_energy_kwh",
]
DECISION_FIELDS = [
    ("g_t", "计划购电量（0:00 锁定，全情景共享）", "kWh/时段"),
    ("b_t", "紧急购电量（按该时刻电价 5 倍结算）", "kWh/时段"),
    ("C_t", "储能充电量（交流侧）", "kWh/时段"),
    ("D_t", "储能放电量（交流侧）", "kWh/时段"),
    ("U_t", "未利用供能量（含弃光与超需计划电量）", "kWh/时段"),
    ("E_t", "储电量（时段末）", "kWh"),
]

SHEET_PLAN = "计划购电量"
SHEET_BATT = "充放电量"
SHEET_EMERG = "紧急购电量"
RESULT2_MAPPING_NOTE = (
    "官方模板『计划购电量』表头列标签为 0:10-0:20、…、0:00-0:10+1，"
    "比模型实际区间整体晚一个时段；由附件一时间标签口径可唯一确定全天 144 个区间为 "
    "0:00-0:10、0:10-0:20、…、23:50-0:00+1。本方案按『列序 = 时间序』填写："
    "第 N 个数据列（N=1..144，Excel 列 B..EO）对应模型第 N 个时段 [ (N-1)Δt, NΔt )。"
)
BATT_BLOCKS = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
SPEC_SLOTS = [
    ("10:00", "10:10"), ("12:00", "12:10"), ("14:00", "14:10"),
    ("16:00", "16:10"), ("18:00", "18:10"), ("20:00", "20:10"),
]
SPEC_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def compute_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
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
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.{nd}f}".rstrip("0").rstrip(".")


def fmt(v, nd: int = 6) -> str:
    return fmt_num(v, nd)


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


def label_to_text(value) -> str:
    if isinstance(value, dt_time):
        return minutes_to_hhmm(value.hour * 60 + value.minute)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return minutes_to_hhmm(parse_time_label_to_minutes(value))
    return str(value)


def period_bounds(t: int) -> tuple:
    return (t - 1) * INTERVAL_MINUTES, t * INTERVAL_MINUTES


def slot_to_period_index(slot: str) -> int:
    mins = parse_time_label_to_minutes(slot)
    return mins // INTERVAL_MINUTES + 1


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
