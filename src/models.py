"""M0 阈值规则 + M1 学习类 baseline。

- M0 ThresholdDetector: 训练集一致性特征 mean+k*std 阈值，任一特征越限即报警。
- M1 fit_m1: 仅 L1 原始统计特征（f_* 列）的随机森林/逻辑回归分类器。
- predict_proba_all: 统一输出 flight_id/window_start_s/label/prob_1/pred。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.config import SEED

L2_COLUMNS = ["c_imu_gps_acc", "c_gps_baro", "c_heading_vel", "c_jump", "c_smoothness"]
META_COLUMNS = ["flight_id", "window_start_s", "label", "split"]


class ThresholdDetector:
    """M0: 多一致性特征阈值规则（对称 mean±k*σ）。"""

    def __init__(self, k: float = 3.0):
        self.k = k
        self.stats_: Dict[str, tuple] = {}
        self.cols_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "ThresholdDetector":
        self.cols_ = [c for c in L2_COLUMNS if c in X.columns]
        self.stats_ = {}
        for c in self.cols_:
            y = X[c].to_numpy(dtype=float)
            ok = ~np.isnan(y)
            if ok.sum() < 3:
                continue
            mean = float(np.mean(y[ok]))
            std = float(np.std(y[ok]))
            self.stats_[c] = (mean, std if std > 1e-9 else 1.0)
        return self

    def predict(self, window: pd.Series) -> bool:
        for c, (mean, std) in self.stats_.items():
            v = window.get(c)
            if v is None or np.isnan(v):
                continue
            if abs(float(v) - mean) > self.k * std:
                return True
        return False

    def score(self, window: pd.Series) -> float:
        """连续异常分数（用于 ROC）：z 最大偏离量归一化，1σ→0，3σ→1。"""
        zs = []
        for c, (mean, std) in self.stats_.items():
            v = window.get(c)
            if v is None or np.isnan(v):
                continue
            zs.append(abs(float(v) - mean) / std)
        if not zs:
            return 0.0
        mz = max(zs)
        if mz <= 1.0:
            return 0.0
        return float(min(1.0, mz / 3.0))


def _feature_columns(X: pd.DataFrame,
                     prefix: str | Tuple[str, ...] | None) -> List[str]:
    """取特征列；可按一个或多个前缀过滤，None 表示全部特征。"""
    feats = [c for c in X.columns if c not in META_COLUMNS]
    if prefix is not None:
        prefixes = (prefix,) if isinstance(prefix, str) else prefix
        feats = [c for c in feats if isinstance(c, str) and c.startswith(prefixes)]
    return feats


def fit_m1(X_tr: pd.DataFrame, y_tr, model_type: str = "random_forest",
           seed: int = SEED, features: str | Tuple[str, ...] | None = "f_"):
    """监督分类器（默认仅 L1 的 f_* 列）。模型对象带 feat_names_/imputer_ 属性。"""
    feats = _feature_columns(X_tr, features)
    if not feats:
        raise ValueError("未找到 f_* 特征列")
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif model_type == "logistic":
        model = LogisticRegression(random_state=seed, max_iter=1000)
    else:
        raise ValueError(f"未知 model_type: {model_type}")
    imp = SimpleImputer(strategy="median")
    model.fit(imp.fit_transform(X_tr[feats]), np.asarray(y_tr))
    # 在线流式推理是小批量调用：多线程线程池重建开销大于收益
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    model.feat_names_ = feats
    model.feature_family_ = ("all" if features is None else
                             "legacy" if not any(c.startswith("e_") for c in feats)
                             else "enhanced")
    model.imputer_ = imp
    return model


def predict_proba_all(model, X: pd.DataFrame) -> pd.DataFrame:
    """统一预测输出: flight_id, window_start_s, label, prob_1, pred(阈0.5)。"""
    Xd = X.copy()
    has_meta = {"flight_id", "window_start_s", "label"} <= set(Xd.columns)
    feats = getattr(model, "feat_names_", [c for c in Xd.columns if c not in META_COLUMNS])
    Xm = Xd[feats]
    imp = getattr(model, "imputer_", None)
    if imp is not None:
        Xm = imp.transform(Xm)
    p = model.predict_proba(Xm)[:, 1]
    out = pd.DataFrame({
        "flight_id": Xd.get("flight_id", pd.Series(["", ] * len(Xd))),
        "window_start_s": Xd.get("window_start_s", pd.Series(np.nan, index=Xd.index)),
        "label": Xd.get("label", pd.Series(np.nan, index=Xd.index)),
        "split": Xd.get("split", pd.Series("", index=Xd.index)),
        "prob_1": p,
    })
    if not has_meta:
        out["flight_id"] = ""
    out["pred"] = (out["prob_1"] >= 0.5).astype(int)
    return out.reset_index(drop=True)
