import numpy as np
import pandas as pd

from src.models_m4 import fit_m4, score_m4


def test_m4_scores_attacks_higher():
    norm = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 6)))
    X = pd.concat([norm, norm * 5])
    m = fit_m4(norm)
    s = score_m4(m, X)
    assert s[:200].mean() < s[200:].mean()


def test_m4_test_score_does_not_depend_on_other_test_rows():
    rng = np.random.default_rng(7)
    normal = pd.DataFrame(rng.normal(size=(100, 4)))
    held_out = pd.DataFrame(rng.normal(loc=3, size=(20, 4)))
    model = fit_m4(normal)
    alone = score_m4(model, held_out)
    combined = score_m4(model, pd.concat([held_out, normal * 20], ignore_index=True))[:len(held_out)]
    assert np.allclose(alone, combined)
