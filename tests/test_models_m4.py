import numpy as np
import pandas as pd

from src.models_m4 import fit_m4, score_m4


def test_m4_scores_attacks_higher():
    norm = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 6)))
    X = pd.concat([norm, norm * 5])
    m = fit_m4(norm)
    s = score_m4(m, X)
    assert s[:200].mean() < s[200:].mean()
