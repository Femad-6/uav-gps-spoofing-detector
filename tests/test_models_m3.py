import numpy as np
import pytest

from src.models_m3 import fit_m3, predict_m3


def test_m3_train_predict_smoke():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 40, 9))   # 50 窗 × 40 步 × 9 原始特征
    y = rng.integers(0, 2, 50).astype("float32")
    m = fit_m3(X, y, seq_len=40, seed=42, device="cpu", epochs=3)
    p = predict_m3(m, X, device="cpu")
    assert p.shape == (50,)
    assert np.all((p >= 0) & (p <= 1))
