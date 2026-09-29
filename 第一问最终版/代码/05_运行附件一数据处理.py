from pathlib import Path
import sys
import subprocess
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comm import (
    PROJECT_DIR, ATTACHMENT1_PATH, ALL_DIRS,
    DELTA_HOURS, INTERVAL_MINUTES, PERIODS_PER_DAY, TIMESTAMP_INTERPRETATION,
    BATTERY, OPERATION, compute_sha256, CONFIG_YAML, LOG_TXT,
)

SCRIPT_ORDER = [
    "01_检查附件一原始数据.py",
    "02_处理附件一基础数据.py",
    "03_校验附件一处理结果.py",
    "04_绘制附件一检查图.py",
]


def _rel(p):
    try:
        return Path(p).relative_to(PROJECT_DIR.parent).as_posix()
    except ValueError:
        return Path(p).as_posix()


def write_config_yaml(attachment1_sha256):
    content = (
        "# 第一问数据处理配置（仅记录统一口径，本阶段不运行优化）\n"
        "input:\n"
        f'  attachment1_path: "{_rel(ATTACHMENT1_PATH)}"\n'
        f'  attachment1_sha256: "{attachment1_sha256}"\n'
        "\n"
        "time:\n"
        f"  interval_minutes: {INTERVAL_MINUTES}\n"
        f"  delta_hours: {DELTA_HOURS}\n"
        f"  periods_per_day: {PERIODS_PER_DAY}\n"
        f'  timestamp_interpretation: "{TIMESTAMP_INTERPRETATION}"\n'
        "\n"
        "battery:\n"
        f"  rated_capacity_kwh: {BATTERY['rated_capacity_kwh']:g}\n"
        f"  min_energy_kwh: {BATTERY['min_energy_kwh']:g}\n"
        f"  max_energy_kwh: {BATTERY['max_energy_kwh']:g}\n"
        f"  initial_energy_kwh: {BATTERY['initial_energy_kwh']:g}\n"
        f"  terminal_energy_kwh: {BATTERY['terminal_energy_kwh']:g}\n"
        f"  max_charge_power_kw: {BATTERY['max_charge_power_kw']:g}\n"
        f"  max_discharge_power_kw: {BATTERY['max_discharge_power_kw']:g}\n"
        f"  charge_efficiency: {BATTERY['charge_efficiency']:g}\n"
        f"  discharge_efficiency: {BATTERY['discharge_efficiency']:g}\n"
        f"  interval_charge_limit_kwh: {BATTERY['max_charge_power_kw'] * DELTA_HOURS}\n"
        f"  interval_discharge_limit_kwh: {BATTERY['max_discharge_power_kw'] * DELTA_HOURS}\n"
        "\n"
        "operation:\n"
        f"  allow_grid_charging: {'true' if OPERATION['allow_grid_charging'] else 'false'}\n"
        f"  allow_grid_export: {'true' if OPERATION['allow_grid_export'] else 'false'}\n"
        f"  allow_pv_curtailment: {'true' if OPERATION['allow_pv_curtailment'] else 'false'}\n"
        "  final_model_type: continuous_linear_programming\n"
        "  binary_charge_state_required: false\n"
    )
    CONFIG_YAML.write_text(content, encoding="utf-8")


def render_tree(root: Path):
    lines = [f"{root.name}/"]
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            if sub.name not in {"__pycache__", ".DS_Store"}:
                lines.append(f"├── {sub.name}")
            continue
        lines.append(f"├── {sub.name}/")
        files = [f for f in sorted(sub.iterdir()) if f.is_file() and f.name not in {".DS_Store"}]
        for i, f in enumerate(files):
            prefix = "└──" if i == len(files) - 1 else "├──"
            lines.append(f"│   {prefix} {f.name}")
    return "\n".join(lines)


def main():
    t0 = datetime.datetime.now()
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    write_config_yaml(compute_sha256(ATTACHMENT1_PATH))

    log = [f"第一问数据处理运行日志", f"起始时间：{t0:%Y-%m-%d %H:%M:%S}",
           f"项目目录：{_rel(PROJECT_DIR)}", f"附件1：{_rel(ATTACHMENT1_PATH)}", ""]

    code_dir = PROJECT_DIR / "代码"
    all_ok = True
    for name in SCRIPT_ORDER:
        script = code_dir / name
        r = subprocess.run([sys.executable, str(script)],
                           capture_output=True, text=True, cwd=str(code_dir))
        log.append(f"==== 运行 {name} ==== 返回码 {r.returncode}")
        if r.stdout.strip():
            log.append(r.stdout.strip())
        if r.stderr.strip():
            log.append("[stderr] " + r.stderr.strip())
        log.append("")
        if r.returncode != 0:
            all_ok = False
            log.append(f"[错误] {name} 运行失败，返回码 {r.returncode}")
            LOG_TXT.write_text("\n".join(log), encoding="utf-8")
            raise RuntimeError(f"数据处理失败：{name}\n{r.stderr}")

    t1 = datetime.datetime.now()
    log.append(f"结束时间：{t1:%Y-%m-%d %H:%M:%S}")
    log.append(f"总耗时：{(t1 - t0).total_seconds():.2f} 秒")
    log.append(f"整体状态：{'全部成功' if all_ok else '存在失败步骤'}")
    log.append("")
    log.append("== 生成目录树 ==")
    log.append(render_tree(PROJECT_DIR))

    LOG_TXT.write_text("\n".join(log), encoding="utf-8")

    print("=" * 60)
    print(f"[05] 第一问数据处理流程结束，整体状态：{'全部成功' if all_ok else '存在失败步骤'}")
    print("=" * 60)
    print(render_tree(PROJECT_DIR))

    return {"all_ok": all_ok}


if __name__ == "__main__":
    main()
