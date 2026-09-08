"""项目全局配置：常量、路径、随机种子与环境探测。

后续所有模块从这里导入，保证单点修改。
"""
from __future__ import annotations

from pathlib import Path

# ---- 全局常量（固定种子，保证可复现）----
SEED = 42
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
NOMINAL_HZ = 20.0

# ---- 路径 ----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"


def require_env(verbose: bool = True) -> dict:
    """探测运行环境：Python/torch/CUDA/GPU。返回 dict 供报告引用。"""
    info = {"python": __import__("sys").version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if info["cuda_available"] else "无"
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False
        info["gpu"] = "无（未安装 torch）"
    if verbose:
        print(info)
    return info
