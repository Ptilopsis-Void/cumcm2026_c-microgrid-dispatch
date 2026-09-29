import hashlib
from pathlib import Path
from datetime import time as dt_time

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

ATTACHMENT_DIR = PROJECT_DIR.parent / "CUMCM2026Problems" / "C题" / "附件"
ATTACHMENT1_PATH = ATTACHMENT_DIR / "附件1.xlsx"

CODE_DIR = PROJECT_DIR / "代码"
RAW_INFO_DIR = PROJECT_DIR / "原始数据说明"
PROCESSED_DIR = PROJECT_DIR / "处理后数据"
REPORT_DIR = PROJECT_DIR / "报告"
FIGURE_DIR = PROJECT_DIR / "数据检查图"
LOG_DIR = PROJECT_DIR / "日志"
CONFIG_DIR = PROJECT_DIR / "配置"

ALL_DIRS = [
    CODE_DIR, RAW_INFO_DIR, PROCESSED_DIR, REPORT_DIR,
    FIGURE_DIR, LOG_DIR, CONFIG_DIR,
]

CSV_BASE_PATH = PROCESSED_DIR / "附件一_第一问基础数据.csv"
CSV_PLOT_PATH = PROCESSED_DIR / "附件一_第一问作图数据.csv"

STRUCT_CSV = RAW_INFO_DIR / "附件一工作表结构.csv"
MAPPING_CSV = RAW_INFO_DIR / "附件一字段映射表.csv"
STAT_CSV = RAW_INFO_DIR / "附件一原始数据统计.csv"

CHECK_MD = REPORT_DIR / "附件一原始数据检查报告.md"
TIME_MD = REPORT_DIR / "附件一时间标签映射说明.md"
PROCESS_MD = REPORT_DIR / "附件一数据处理报告.md"
ACCEPT_MD = REPORT_DIR / "附件一数据处理验收表.md"

FIG_LOAD_PV_PDF = FIGURE_DIR / "附件一_负荷与光伏预测.pdf"
FIG_LOAD_PV_PNG = FIGURE_DIR / "附件一_负荷与光伏预测.png"
FIG_NET_PDF = FIGURE_DIR / "附件一_净需求与光伏剩余.pdf"
FIG_NET_PNG = FIGURE_DIR / "附件一_净需求与光伏剩余.png"
FIG_PRICE_PDF = FIGURE_DIR / "附件一_分时电价.pdf"
FIG_PRICE_PNG = FIGURE_DIR / "附件一_分时电价.png"

LOG_TXT = LOG_DIR / "附件一数据处理运行日志.txt"
ANOMALY_CSV = LOG_DIR / "附件一数据异常记录.csv"
CONFIG_YAML = CONFIG_DIR / "第一问数据处理配置.yaml"

DELTA_HOURS = 1.0 / 6.0
INTERVAL_MINUTES = 10
PERIODS_PER_DAY = 144
TIMESTAMP_INTERPRETATION = "区间终点（标签表示该10分钟区间的结束时刻）"

BATTERY = {
    "rated_capacity_kwh": 12000.0,
    "min_energy_kwh": 1200.0,
    "max_energy_kwh": 10800.0,
    "initial_energy_kwh": 6000.0,
    "terminal_energy_kwh": 6000.0,
    "max_charge_power_kw": 5000.0,
    "max_discharge_power_kw": 5000.0,
    "charge_efficiency": 0.9,
    "discharge_efficiency": 0.9,
}

OPERATION = {
    "allow_grid_charging": True,
    "allow_grid_export": False,
    "allow_pv_curtailment": True,
    "final_model_type": "continuous_linear_programming",
    "binary_charge_state_required": False,
}

FORMAL_INPUT_FIELDS = [
    "interval_index", "original_time_label", "interval_start", "interval_end",
    "delta_hours", "price_yuan_per_kwh", "load_kw", "pv_forecast_kw",
]

DESCRIPTIVE_FIELDS = [
    "net_load_kw", "net_demand_kw", "pv_surplus_kw",
    "load_energy_kwh", "pv_forecast_energy_kwh",
    "net_demand_energy_kwh", "pv_surplus_energy_kwh",
]

DECISION_FIELDS = [
    ("g_t", "外网计划购电量", "kWh"),
    ("C_t", "交流侧充电量", "kWh"),
    ("D_t", "交流侧放电量", "kWh"),
    ("E_t", "储能电量", "kWh"),
    ("U_t", "未利用供能", "kWh"),
]


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time_label_to_minutes(value):
    if isinstance(value, dt_time):
        return value.hour * 60 + value.minute
    if isinstance(value, str):
        s = value.strip()
        if '+1' in s:
            base = s.split('+')[0].strip()
            h, m = (int(x) for x in base.split(':')[:2])
            return 24 * 60 + h * 60 + m
        parts = s.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    raise TypeError(f"unexpected time label type: {type(value).__name__} = {value!r}")


def minutes_to_hhmm(mins, use_24=False):
    mins = int(round(mins))
    if use_24 and mins == 1440:
        return "24:00"
    h = (mins // 60) % 24
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def label_to_text(value):
    if isinstance(value, dt_time):
        return minutes_to_hhmm(value.hour * 60 + value.minute)
    if isinstance(value, str):
        return value.strip()
    return str(value)


def setup_chinese_font():
    import matplotlib
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Songti SC", "PingFang HK", "Hiragino Sans GB", "Heiti TC", "STHeiti", "Arial Unicode MS"]
    chosen = next((c for c in candidates if c in available), None)
    if chosen is not None:
        matplotlib.rcParams["font.family"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


MODEL_RESULT_DIR = PROJECT_DIR / "模型结果"
MODEL_FIG_DIR = PROJECT_DIR / "模型结果图"
SOLVE_LOG_DIR = PROJECT_DIR / "求解日志"
SUBMIT_DIR = PROJECT_DIR / "提交结果"

for _d in (MODEL_RESULT_DIR, MODEL_FIG_DIR, SOLVE_LOG_DIR, SUBMIT_DIR):
    if _d not in ALL_DIRS:
        ALL_DIRS.append(_d)

ANNEX5_DIR = ATTACHMENT_DIR / "附件5"
RESULT1_TEMPLATE = ANNEX5_DIR / "result1.xlsx"
SUBMIT_RESULT1 = SUBMIT_DIR / "result1.xlsx"
SUBMIT_INTERNAL = SUBMIT_DIR / "第一问内部结果表.xlsx"

OPT_SCHEDULE_CSV = MODEL_RESULT_DIR / "第一问_最优调度结果.csv"
COST_SUMMARY_CSV = MODEL_RESULT_DIR / "第一问_费用汇总.csv"
BASELINE_CSV = MODEL_RESULT_DIR / "第一问_无储能基准.csv"
VALIDATION_CSV = MODEL_RESULT_DIR / "第一问_约束校验结果.csv"
PLOT_DATA_CSV = MODEL_RESULT_DIR / "第一问_作图数据.csv"

MODEL_REPORT_MD = REPORT_DIR / "第一问模型求解报告.md"
MODEL_ACCEPT_MD = REPORT_DIR / "第一问模型结果验收表.md"

FIG_PRICE_GRID_PDF = MODEL_FIG_DIR / "第一问_电价与电网购电功率.pdf"
FIG_PRICE_GRID_PNG = MODEL_FIG_DIR / "第一问_电价与电网购电功率.png"
FIG_LOAD_PV_GRID_PDF = MODEL_FIG_DIR / "第一问_负荷光伏与购电功率.pdf"
FIG_LOAD_PV_GRID_PNG = MODEL_FIG_DIR / "第一问_负荷光伏与购电功率.png"
FIG_CD_PDF = MODEL_FIG_DIR / "第一问_储能充放电功率.pdf"
FIG_CD_PNG = MODEL_FIG_DIR / "第一问_储能充放电功率.png"
FIG_E_PDF = MODEL_FIG_DIR / "第一问_储能电量变化.pdf"
FIG_E_PNG = MODEL_FIG_DIR / "第一问_储能电量变化.png"
FIG_BASELINE_PDF = MODEL_FIG_DIR / "第一问_有无储能购电费用对比.pdf"
FIG_BASELINE_PNG = MODEL_FIG_DIR / "第一问_有无储能购电费用对比.png"

SOLVE_LOG_TXT = SOLVE_LOG_DIR / "第一问MILP求解日志.txt"

ETA_CHARGE = BATTERY["charge_efficiency"]
ETA_DISCHARGE = BATTERY["discharge_efficiency"]
P_CHARGE_MAX = BATTERY["max_charge_power_kw"]
P_DISCHARGE_MAX = BATTERY["max_discharge_power_kw"]
E_MIN = BATTERY["min_energy_kwh"]
E_MAX = BATTERY["max_energy_kwh"]
E_INITIAL = BATTERY["initial_energy_kwh"]
E_TERMINAL = BATTERY["terminal_energy_kwh"]

TABLE1_SLOTS = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
                "16:00-16:10", "18:00-18:10", "20:00-20:10"]

WINDOW_HOURS = 4
PERIODS_PER_WINDOW = 24
WINDOW_LABELS = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
                 "12:00-16:00", "16:00-20:00", "20:00-24:00"]


def load_formal_input():
    df = pd.read_csv(CSV_BASE_PATH, encoding="utf-8-sig")
    if len(df) != PERIODS_PER_DAY:
        raise ValueError(f"正式输入数据应为 {PERIODS_PER_DAY} 行，实际 {len(df)} 行")
    missing = [c for c in FORMAL_INPUT_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"正式输入数据缺失字段：{missing}")
    if df[FORMAL_INPUT_FIELDS].isna().any().any():
        raise ValueError("正式输入字段存在缺失值")
    return df
