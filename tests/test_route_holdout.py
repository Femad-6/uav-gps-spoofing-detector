import numpy as np
import pandas as pd
import pytest

from scripts.run_route_holdout import fit_predict_model


def _dataset():
    rng = np.random.default_rng(21)
    rows = []
    for split, count in (("train", 40), ("val", 12), ("test", 16)):
        for index in range(count):
            label = index % 2
            rows.append({
                "flight_id": f"{split}_{index // 4}",
                "window_start_s": float(index % 4) * 0.5,
                "label": label,
                "split": split,
                "f_signal": float(label) + rng.normal(scale=0.1),
                "c_gps_baro": float(label) + rng.normal(scale=0.1),
                "e_signal": float(label) + rng.normal(scale=0.1),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", ["m0", "m1", "m1b", "m2", "m2e", "m4"])
def test_route_holdout_dispatch_preserves_split(name):
    data = _dataset()
    predictions = fit_predict_model(name, data)
    assert len(predictions) == len(data)
    assert predictions["split"].tolist() == data["split"].tolist()
    assert predictions["prob_1"].notna().all()


def test_route_holdout_dispatch_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown route-holdout model"):
        fit_predict_model("unknown", _dataset())
