import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent

def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    names = ["05_运行附件一数据处理.py", "12_接口验收.py", "15_运行最终连续LP阶段.py",
             "20_运行第一问对偶价格阶段.py", "23_运行第一问连续DP阶段.py",
             "25_生成最终提交与报告.py", "09_绘制第一问结果图.py",
             "26_最终版独立验收.py"]
    historical = CODE.parent / "历史模型" / "MILP历史基准校验摘要.csv"
    if historical.is_file():
        names.insert(3, "16_验证同时充放电消去.py")
    else:
        print("[SKIP] 未提供开发期历史MILP基准；跳过16_验证同时充放电消去.py。")
    for name in names:
        print(f"\n运行 {name}", flush=True)
        subprocess.run([sys.executable, "-B", str(CODE/name)], env=env, check=True)
    print("最终连续LP完整流程及独立验收通过。")

if __name__ == "__main__":
    main()
