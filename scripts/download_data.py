"""下载 UAV-GPS-Spoofing-Dataset 到 data/raw/。

数据文件托管在 Google Drive（仓库 README 的 "Data Links" 节）：
- 优先用 gdown 下载 PX4_Simulation_Data.zip 并解压；
- 同时 git clone 仓库本体以备参考（README/说明），失败可忽略。

用法: python scripts/download_data.py
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = "https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset.git"
DRIVE_FILE_ID = "1jypbdKFP-4vN7HMwbnm4z4i-b0OhoPtS"  # 见仓库 README Data Links
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&export=download"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ZIP_PATH = RAW_DIR / "PX4_Simulation_Data.zip"
EXTRACTED_MARKER = RAW_DIR / "data.ready"


def clone_readme() -> None:
    """克隆仓库说明（非必须，失败仅提示）。"""
    if any(RAW_DIR.iterdir()):
        return
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO, str(RAW_DIR)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        print("[ok] 仓库说明已克隆:", RAW_DIR)
    except Exception as e:
        print(f"[warn] git clone 失败（可忽略）: {e}")


def download_zip() -> bool:
    if ZIP_PATH.exists():
        print(f"[skip] 已存在 {ZIP_PATH}，跳过下载")
        return True
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    print("[info] 从 Google Drive 下载数据（可能较大，请耐心等待）...")
    gdown.download(DRIVE_URL, str(ZIP_PATH), quiet=False)
    return ZIP_PATH.exists()


def extract_zip() -> bool:
    if EXTRACTED_MARKER.exists():
        return True
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
        EXTRACTED_MARKER.write_text("ok", encoding="utf-8")
        print("[ok] 数据解压完成")
        return True
    except Exception as e:
        print(f"[error] 解压失败: {e}")
        return False


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clone_readme()
    if not download_zip():
        print("[error] 数据下载失败。手动方案：浏览器打开")
        print(f"      {DRIVE_URL.split('&')[0]}")
        print(f"      下载 PX4_Simulation_Data.zip 到 {RAW_DIR} 后重跑。")
        return 1
    if not extract_zip():
        return 1
    print("[done] 数据就绪:", RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
