import hashlib
import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
ATTACH1 = PROJECT.parent / "CUMCM2026Problems" / "C题" / "附件" / "附件1.xlsx"
CSV_PATH = PROJECT / "处理后数据" / "附件一_第一问基础数据.csv"
CONFIG = PROJECT / "配置" / "第一问数据处理配置.yaml"
MANIFEST = PROJECT / "历史模型" / "MILP历史基准文件清单.csv"
OUT = PROJECT / "报告" / "第一问最终模型接口验收表.md"

FORMAL = ["interval_index", "original_time_label", "interval_start", "interval_end",
          "delta_hours", "price_yuan_per_kwh", "load_energy_kwh", "pv_forecast_energy_kwh"]
DELTA = 1.0 / 6.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def read_config():
    d = {}
    section = None
    for raw in open(CONFIG, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip() or line.strip().startswith("#"):
            continue
        s = line.lstrip()
        ind = len(line) - len(s)
        if ind == 0:
            section = s.rstrip(":").strip()
            d[section] = {}
        else:
            k, _, v = s.partition(":")
            d[section][k.strip()] = v.strip().strip('"')
    return d


def main():
    cfg = read_config()
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    checks = []

    n_rows = len(df)
    checks.append(("清洗CSV仍为144行", "144 行", f"行数 = {n_rows}", n_rows == 144))

    att_sha = sha256(ATTACH1)
    base_sha = cfg["input"]["attachment1_sha256"]
    checks.append(("原始附件SHA-256不变", "与配置基准一致", f"{att_sha}", att_sha == base_sha))

    csv_sha = sha256(CSV_PATH)
    manifest_sha = None
    if MANIFEST.is_file():
        with open(MANIFEST, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["原路径"] == "处理后数据/附件一_第一问基础数据.csv":
                    manifest_sha = r["SHA-256"]
        checks.append(("清洗CSV数值未被修改", "当前SHA与冻结清单一致", f"{csv_sha}",
                       manifest_sha is not None and csv_sha == manifest_sha))
    else:
        checks.append(("清洗CSV已生成SHA-256", "历史清单缺失时仅记录，不阻断复现",
                       f"{csv_sha}（已跳过开发期历史冻结比对）", True))

    err4 = float(np.max(np.abs(df["load_energy_kwh"].to_numpy() - df["load_kw"].to_numpy() * DELTA)))
    checks.append(("load_energy_kwh 恒等于 load_kw/6", "max 偏差 ≈ 0", f"max|L_t - load_kw/6| = {err4:.3e}", err4 < 1e-8))

    err5 = float(np.max(np.abs(df["pv_forecast_energy_kwh"].to_numpy() - df["pv_forecast_kw"].to_numpy() * DELTA)))
    checks.append(("pv_forecast_energy_kwh 恒等于 pv_forecast_kw/6", "max 偏差 ≈ 0", f"max|V_t - pv_kw/6| = {err5:.3e}", err5 < 1e-8))

    miss = int(df[FORMAL].isna().sum().sum())
    checks.append(("正式模型输入不存在缺失", "8 字段全无缺失", f"缺失计数 = {miss}", miss == 0))

    lim_c = float(cfg["battery"]["interval_charge_limit_kwh"])
    lim_d = float(cfg["battery"]["interval_discharge_limit_kwh"])
    target = 5000.0 * DELTA
    checks.append(("单时段充放电量上限为5000/6", f"{target!r}",
                   f"charge={lim_c!r}, discharge={lim_d!r}",
                   abs(lim_c - target) < 1e-9 and abs(lim_d - target) < 1e-9))

    checks.append(("最终配置允许电网充电", "allow_grid_charging = true",
                   f"= {cfg['operation']['allow_grid_charging']}",
                   cfg["operation"]["allow_grid_charging"] == "true"))

    checks.append(("最终配置禁止电网售电", "allow_grid_export = false",
                   f"= {cfg['operation']['allow_grid_export']}",
                   cfg["operation"]["allow_grid_export"] == "false"))

    checks.append(("最终配置允许弃光", "allow_pv_curtailment = true",
                   f"= {cfg['operation']['allow_pv_curtailment']}",
                   cfg["operation"]["allow_pv_curtailment"] == "true"))

    bin_false = cfg["operation"]["binary_charge_state_required"] == "false"
    ftype = cfg["operation"]["final_model_type"]
    checks.append(("最终配置不需要二进制变量", "binary_charge_state_required = false",
                   f"binary={cfg['operation']['binary_charge_state_required']}, final_model_type={ftype}",
                   bin_false and ftype == "continuous_linear_programming"))

    if MANIFEST.is_file():
        registered = 0
        mismatch = []
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8-sig")):
            rel = r["原路径"]
            registered += 1
            if rel == "配置/第一问数据处理配置.yaml":
                continue
            archived = PROJECT / "历史模型" / "修订前基准" / rel
            if not archived.is_file() or sha256(archived) != r["SHA-256"]:
                mismatch.append(rel)
        checks.append(("旧MILP代码和结果已登记且未被覆盖", "历史材料存在时核验全部指纹",
                       f"登记={registered}，不一致={len(mismatch)}"
                       f"{(' ' + '、'.join(mismatch)) if mismatch else ''}",
                       len(mismatch) == 0))
    else:
        checks.append(("开发期历史MILP快照", "不属于正式复现的必需输入",
                       "未随支撑材料提供，已跳过", True))

    passed = sum(1 for c in checks if c[3])
    lines = [
        "# 第一问最终模型接口验收表",
        "",
        "- 验收对象：最终连续 LP 的数据—模型接口",
        f"- 验收结论：{'✅ 全部通过' if passed == len(checks) else '⚠️ 存在未通过项'}（{passed} / {len(checks)}）",
        "",
        "| 序号 | 验收项 | 要求 | 实测结果 | 结果 |",
        "|---|---|---|---|---|",
    ]
    for i, (name, req, val, ok) in enumerate(checks, 1):
        lines.append(f"| {i} | {name} | {req} | {val} | {'通过' if ok else '未通过'} |")
    lines.append("")
    lines.append(f"- 汇总：{passed} / {len(checks)} 项通过。")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    for i, (name, req, val, ok) in enumerate(checks, 1):
        print(f"[{i:2d}] {'PASS' if ok else 'FAIL'}  {name}")
    print(f"结果：{passed} / {len(checks)} 项通过")
    if passed != len(checks):
        raise RuntimeError("数据接口或历史快照验收失败")


if __name__ == "__main__":
    main()
