"""M4: 无监督异常检测 — IsolationForest（仅用正常窗拟合）。

贴合现实：攻击标签稀缺，正常巡航数据容易获得。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from src.config import SEED
from src.models import _feature_columns


def fit_m4(X_norm: pd.DataFrame, seed: int = SEED) -> IsolationForest:
    feats = _feature_columns(X_norm, None)
    imp = SimpleImputer(strategy="median")
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(imp.fit_transform(X_norm[feats]))
    model.feat_names_ = feats
    model.imputer_ = imp
    # 分数标定仅由训练正常数据确定，避免评测时借用测试集范围。
    train_scores = -model.decision_function(imp.transform(X_norm[feats]))
    model.score_min_ = float(train_scores.min())
    model.score_max_ = float(train_scores.max())
    return model


def score_m4(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """归一化异常分数（越大越异常）。"""
    Xm = model.imputer_.transform(X[model.feat_names_])
    s = -model.decision_function(Xm)          # 取反：大值 = 异常
    lo = getattr(model, "score_min_", float(s.min()))
    hi = getattr(model, "score_max_", float(s.max()))
    return (s - lo) / (hi - lo + 1e-12)
