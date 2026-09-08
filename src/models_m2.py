"""M2: 改进模型 — HistGradientBoosting on L1(L2)L3 全特征。

GBDT 对 NaN 原生支持，无需插补；与 M1 的唯一差别在特征集（M1 仅 f_*，M2 全量）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import SEED
from src.models import _feature_columns


def fit_m2(X_tr: pd.DataFrame, y_tr, seed: int = SEED) -> HistGradientBoostingClassifier:
    feats = _feature_columns(X_tr, None)  # 所有特征（f_* 与 c_*）
    if not feats:
        raise ValueError("未找到特征列")
    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    model.fit(X_tr[feats], np.asarray(y_tr))
    model.feat_names_ = feats
    return model
