"""标签构建（由 attack 列提取攻击区间）与按航班训练/测试划分。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import PROCESSED_DIR, SEED, SPLIT_CONFIG
from src.features import attack_intervals

def build_labels(all_flights: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[float, float]]]:
    """各航班 attack 列 → [(start_s, end_s), ...]；无攻击航班为 []。"""
    return {fid: attack_intervals(df) for fid, df in all_flights.items()}


def assign_split(flights: List[str], test_frac: float = 0.2, seed: int = SEED) -> Dict[str, str]:
    """按航班划分 train/test（固定 seed 可复现，航班不重叠）。"""
    rng = random.Random(seed)
    flights = sorted(flights)
    n_test = max(1, int(round(len(flights) * test_frac)))
    test = set(rng.sample(flights, n_test))
    return {f: ("test" if f in test else "train") for f in flights}


def load_split() -> Dict[str, str]:
    if SPLIT_CONFIG.exists():
        return json.loads(SPLIT_CONFIG.read_text(encoding="utf-8"))
    return {}


def save_split(split: Dict[str, str]) -> None:
    SPLIT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_CONFIG.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")


def load_dataset(path=None) -> pd.DataFrame:
    """加载窗级特征数据集（含 split 列）。"""
    path = path or (PROCESSED_DIR / "features_windowed.parquet")
    return pd.read_parquet(path)
