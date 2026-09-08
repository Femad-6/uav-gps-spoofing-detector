"""一键全流程：下载数据 → 数据管线（特征/标签/划分）→ 训练与评测 → 图表。

用法:
  python scripts/run_all.py                # 全量（需 2.6GB 数据）
  python scripts/run_all.py --tiny         # 冒烟：仅用前 3 个航班（快速验证流程）
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 允许 import src

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="小数据冒烟（前 3 个航班）")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        print("[1/3] 下载数据 ...")
        subprocess.run([sys.executable, "scripts/download_data.py"], cwd=ROOT)

    if args.tiny:
        tiny = ROOT / "data" / "tiny"
        tiny.mkdir(parents=True, exist_ok=True)
        csvs = sorted((ROOT / "data" / "raw").rglob("*.csv"))[:3]
        if not csvs:
            print("[error] --tiny 需要先下载数据（或手动放置 CSV 到 data/raw）")
            return 1
        for c in csvs:
            shutil.copy(c, tiny / c.name)
        os.environ["SPOOFING_DATA_DIR"] = str(tiny)
        print(f"      tiny 数据: {[c.name for c in csvs]}")

    print("[2/3] 数据管线（特征/标签/划分）...")
    r = subprocess.run([sys.executable, "scripts/make_dataset.py"], cwd=ROOT)
    if r.returncode != 0:
        return r.returncode

    if not args.skip_train:
        print("[3/3] 训练 + 评测 ...")
        models = "m0,m1,m2,m4" if args.tiny else "m0,m1,m2,m3,m4"
        r = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--models", models], cwd=ROOT)
        if r.returncode != 0:
            return r.returncode
    print("[done] 一键流程完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
