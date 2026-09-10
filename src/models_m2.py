"""M2: 改进模型 — HistGradientBoosting on L1(L2)L3 全特征。

GBDT 对 NaN 原生支持，无需插补；与 M1 的唯一差别在特征集（M1 仅 f_*，M2 全量）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import SEED
from src.models import _feature_columns


def _fit_histgb(X_tr: pd.DataFrame, y_tr, prefixes, seed: int):
    feats = _feature_columns(X_tr, prefixes)
    if not feats:
        raise ValueError("未找到特征列")
    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    model.fit(X_tr[feats], np.asarray(y_tr))
    model.feat_names_ = feats
    model.feature_family_ = "enhanced" if any(c.startswith("e_") for c in feats) else "legacy"
    return model


def fit_m2(X_tr: pd.DataFrame, y_tr, seed: int = SEED) -> HistGradientBoostingClassifier:
    """M2 legacy：固定使用原有 f_* 与 c_*，不静默吸收新增特征。"""
    return _fit_histgb(X_tr, y_tr, ("f_", "c_"), seed)


def fit_m2e(X_tr: pd.DataFrame, y_tr, seed: int = SEED) -> HistGradientBoostingClassifier:
    """M2e 消融：与 M2 同参数，仅增加论文启发的 e_* 特征。"""
    return _fit_histgb(X_tr, y_tr, ("f_", "c_", "e_"), seed)
