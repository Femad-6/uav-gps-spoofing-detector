import numpy as np
import pandas as pd

from src.evaluate import evaluate_all, best_threshold


def test_evaluate_metrics_basic():
    preds = pd.DataFrame({
        "prob_1": [0.9, 0.8, 0.1, 0.2],
        "pred": [1, 1, 0, 0],
        "label": [1, 0, 0, 1],
        "flight_id": ["f1"] * 4,
        "window_start_s": [0.0, 1.0, 2.0, 3.0],
    })
    m = evaluate_all({"m": preds})
    assert 0.5 < m["m"]["AUROC"] < 1.0
    assert m["m"]["AUPRC"] >= 0
    assert "delay_mean_s" in m["m"] and "delay_median_s" in m["m"]
    assert m["m"]["delay_mean_s"] == 0.0


def test_evaluate_uses_validation_split_for_threshold():
    preds = pd.DataFrame({
        "prob_1": [0.2, 0.1, 0.8, 0.7, 0.9, 0.2],
        "pred": [1, 0, 1, 1, 1, 0],
        "label": [1, 0, 1, 0, 1, 0],
        "split": ["train", "train", "val", "val", "test", "test"],
        "flight_id": ["tr"] * 2 + ["va"] * 2 + ["te"] * 2,
        "window_start_s": [0.0, 1.0] * 3,
    })
    m = evaluate_all({"m": preds})["m"]
    assert m["threshold_source"] == "val"
    assert np.isclose(m["threshold"], 0.7)
