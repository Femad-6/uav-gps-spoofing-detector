import numpy as np
import pandas as pd

from src.models import ThresholdDetector, fit_m1, predict_proba_all


def test_threshold_detector_detects_divergent_window():
    detector = ThresholdDetector()
    X = pd.DataFrame({"c_gps_baro": [0.1, 0.2, 0.15, 0.22]})
    detector.fit(X[["c_gps_baro"]])
    assert detector.predict(pd.Series({"c_gps_baro": 5.9})) is True
    assert detector.predict(pd.Series({"c_gps_baro": 0.16})) is False


def test_fit_m1_is_deterministic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 8)), columns=[f"f_{i}" for i in range(8)])
    y = pd.Series(rng.integers(0, 2, 100))
    m1 = fit_m1(X, y, seed=42)
    m2 = fit_m1(X, y, seed=42)
    c1 = predict_proba_all(m1, X)
    c2 = predict_proba_all(m2, X)
    assert c1["prob_1"].tolist() == c2["prob_1"].tolist()
    assert set(c1.columns) >= {"prob_1", "pred", "flight_id", "window_start_s", "label"}


def test_predict_proba_all_preserves_split_metadata():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=[f"f_{i}" for i in range(4)])
    X["split"] = ["train"] * 10 + ["val"] * 10 + ["test"] * 10
    y = pd.Series([0, 1] * 15)
    model = fit_m1(X, y)
    out = predict_proba_all(model, X)
    assert out["split"].tolist() == X["split"].tolist()
