import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_all, best_threshold, select_operating_threshold


def test_evaluate_metrics_basic():
    preds = pd.DataFrame({
        "prob_1": [0.9, 0.1, 0.8, 0.2, 0.9, 0.8, 0.1, 0.2],
        "pred": [1, 0, 1, 0, 1, 1, 0, 0],
        "label": [1, 0, 1, 0, 1, 0, 0, 1],
        "split": ["train"] * 2 + ["val"] * 2 + ["test"] * 4,
        "flight_id": ["tr"] * 2 + ["va"] * 2 + ["te"] * 4,
        "window_start_s": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0],
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
    assert np.isclose(m["threshold"], 0.8)
    assert m["threshold_policy"] == "val_far<=0.10_max_recall"
    assert m["val_false_alarm_rate"] == 0.0
    assert "f1_threshold" in m


def test_operating_threshold_respects_validation_false_alarm_cap():
    val = pd.DataFrame({"label": [0, 0, 0, 0, 1, 1],
                        "prob_1": [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]})
    threshold, far = select_operating_threshold(val, max_false_alarm_rate=0.25)
    assert np.isclose(threshold, 0.7)
    assert far <= 0.25


def test_operating_threshold_has_no_alarm_candidate():
    val = pd.DataFrame({"label": [0, 0, 1], "prob_1": [1.0, 1.0, 1.0]})
    threshold, far = select_operating_threshold(val, max_false_alarm_rate=0.0)
    assert threshold > 1.0
    assert far == 0.0


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_operating_threshold_rejects_single_class_validation(labels):
    with pytest.raises(ValueError, match="normal and attacked"):
        select_operating_threshold(
            pd.DataFrame({"label": labels, "prob_1": [0.1, 0.9]}))


def test_confirmed_metrics_suppress_isolated_test_spikes():
    rows = []
    for split, fid, labels, scores in [
        ("train", "tr", [0, 1], [0.1, 0.9]),
        ("val", "va", [0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9]),
        ("test", "normal", [0, 0, 0, 0, 0], [0.1, 0.85, 0.1, 0.1, 0.1]),
        ("test", "attacked", [1, 1, 1, 1, 1], [0.9, 0.9, 0.9, 0.1, 0.1]),
    ]:
        for index, (label, score) in enumerate(zip(labels, scores)):
            rows.append({"split": split, "flight_id": fid,
                         "window_start_s": index * 0.5, "label": label,
                         "prob_1": score, "pred": int(score >= 0.5)})
    metrics = evaluate_all({"m": pd.DataFrame(rows)})["m"]
    assert metrics["false_alarm_rate"] > 0.0
    assert metrics["confirmed_false_alarm_rate"] == 0.0
    assert metrics["confirmed_recall"] > 0.0
    assert metrics["confirmed_missed_flights_rate"] == 0.0
    assert metrics["confirmed_delay_mean_s"] == 1.0
