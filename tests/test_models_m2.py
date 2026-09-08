import numpy as np
import pandas as pd

from src.models import predict_proba_all
from src.models_m2 import fit_m2


def test_fit_m2_handles_nan_and_predicts_probabilities():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 8)))
    X.iloc[::3, 1] = np.nan
    m = fit_m2(X, pd.Series(np.random.default_rng(1).integers(0, 2, 100)))
    c = predict_proba_all(m, X)
    assert len(c) == 100
    assert c["prob_1"].between(0, 1).all()


def test_fit_m2_deterministic():
    X = pd.DataFrame(np.random.default_rng(3).normal(size=(80, 6)))
    y = pd.Series(np.random.default_rng(4).integers(0, 2, 80))
    c1 = predict_proba_all(fit_m2(X, y, seed=42), X)
    c2 = predict_proba_all(fit_m2(X, y, seed=42), X)
    assert c1["prob_1"].tolist() == c2["prob_1"].tolist()
