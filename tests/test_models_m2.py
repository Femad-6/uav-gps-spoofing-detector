import numpy as np
import pandas as pd

from src.models import predict_proba_all
from src.models_m2 import fit_m2, fit_m2e


def test_fit_m2_handles_nan_and_predicts_probabilities():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 8)),
                     columns=[f"f_{i}" for i in range(8)])
    X.iloc[::3, 1] = np.nan
    m = fit_m2(X, pd.Series(np.random.default_rng(1).integers(0, 2, 100)))
    c = predict_proba_all(m, X)
    assert len(c) == 100
    assert c["prob_1"].between(0, 1).all()


def test_fit_m2_deterministic():
    X = pd.DataFrame(np.random.default_rng(3).normal(size=(80, 6)),
                     columns=[f"f_{i}" for i in range(6)])
    y = pd.Series(np.random.default_rng(4).integers(0, 2, 80))
    c1 = predict_proba_all(fit_m2(X, y, seed=42), X)
    c2 = predict_proba_all(fit_m2(X, y, seed=42), X)
    assert c1["prob_1"].tolist() == c2["prob_1"].tolist()


def test_m2e_adds_enhanced_columns_without_changing_m2():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"f_a": rng.normal(size=40),
                      "c_a": rng.normal(size=40),
                      "e_a": rng.normal(size=40)})
    y = pd.Series([0, 1] * 20)
    legacy = fit_m2(X, y)
    enhanced = fit_m2e(X, y)
    assert legacy.feat_names_ == ["f_a", "c_a"]
    assert enhanced.feat_names_ == ["f_a", "c_a", "e_a"]
