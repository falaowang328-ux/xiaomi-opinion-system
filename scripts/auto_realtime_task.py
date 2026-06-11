# -*- coding: utf-8 -*-

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

import schedule
import time
import subprocess
from datetime import datetime
import os


# =========================
# 1. 脚本路径
# =========================

crawl_path = r"C:\Users\FALAOWANG\Desktop\xiaomi-opinion-system\scripts\realtime_crawl.py"
analysis_path = r"C:\Users\FALAOWANG\Desktop\xiaomi-opinion-system\scripts\realtime_analysis.py"


# =========================
# 2. 实时运行子脚本函数
# =========================

def run_script_realtime(path, name):
    print("-" * 60)
    print(f"准备运行：{name}")
    print("脚本路径：", path)

    if not os.path.exists(path):
        print(f"{name} 不存在，请检查路径")
        return False

    process = subprocess.Popen(
        [sys.executable, "-u", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8"
        }
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    print(f"\n{name} 返回码：", process.returncode)

    if process.returncode == 0:
        print(f"{name} 运行成功")
        return True
    else:
        print(f"{name} 运行失败")
        return False


# =========================
# 3. 定时任务函数
# =========================

def job():
    print("=" * 60)
    print("开始执行实时舆情采集：", datetime.now())

    ok1 = run_script_realtime(
        crawl_path,
        "实时采集脚本"
    )

    if ok1:
        run_script_realtime(
            analysis_path,
            "实时分析脚本"
        )
    else:
        print("采集脚本失败，暂不执行分析脚本")

    print("本轮实时采集与分析结束：", datetime.now())
    print("=" * 60)


# =========================
# 4. 主程序入口
# =========================

if __name__ == "__main__":

    # 程序启动后先立即执行一次
    job()

    # 然后每 10 分钟自动执行一次
    schedule.every(10).minutes.do(job)

    print("定时任务已启动：每 10 分钟自动采集并分析一次。")

    while True:
        schedule.run_pending()
        time.sleep(1)