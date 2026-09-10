import numpy as np
import pandas as pd

from fastapi.testclient import TestClient

from src.models import fit_m1, predict_proba_all
from src.server import create_app


def _synthetic_df():
    t = np.arange(0, 10.0, 0.05)
    return pd.DataFrame({"time_s": t, "lat": 34.0 + 1e-5 * t, "lon": 108.9,
                         "alt_gps": 20 + 0.5 * t, "alt_baro": 20 + 0.0 * t,
                         "vel_e": 0.1 * np.ones_like(t), "vel_n": np.zeros_like(t),
                         "vel_u": np.ones_like(t), "ax": np.zeros_like(t),
                         "ay": np.zeros_like(t), "az": 9.81 * np.ones_like(t),
                         "gx": np.zeros_like(t), "gy": np.zeros_like(t), "gz": np.zeros_like(t),
                         "roll": np.zeros_like(t), "pitch": np.zeros_like(t), "yaw": np.zeros_like(t),
                         "mx": np.zeros_like(t), "my": np.zeros_like(t), "mz": np.zeros_like(t)})


def _train_model(path):
    from src.features import window_extract
    X = window_extract(_synthetic_df())
    y = X["label"].copy()
    y.iloc[:3] = 1                      # 让分类器见过两类
    m = fit_m1(X.drop(columns=["label"]), y)
    import joblib
    joblib.dump(m, path)
    return m


def _rows(n=200):
    df = _synthetic_df().head(n)
    return df.to_dict("records")


def test_healthz():
    app = create_app(model_path="Nonexistent.joblib")
    assert TestClient(app).get("/healthz").json() == {"status": "ok"}


def test_predict_endpoint_schema(tmp_path):
    app = create_app(model_path=str(tmp_path / "m.joblib"))
    _train_model(tmp_path / "m.joblib")
    client = TestClient(app)
    r = client.post("/predict", json={"flight_id": "test", "rows": _rows()})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"spoofing", "confidence", "alarm_intervals", "n_windows"}
    assert body["n_windows"] > 0


def test_predict_endpoint_sorts_and_resamples_rows(tmp_path):
    app = create_app(model_path=str(tmp_path / "m.joblib"))
    _train_model(tmp_path / "m.joblib")
    rows = list(reversed(_rows()))
    r = TestClient(app).post("/predict", json={"flight_id": "test", "rows": rows})
    assert r.status_code == 200
    assert r.json()["n_windows"] > 0


def test_predict_without_model_returns_503():
    app = create_app(model_path="definitely_missing.joblib")
    client = TestClient(app)
    r = client.post("/predict", json={"flight_id": "t", "rows": _rows()})
    assert r.status_code == 503


def test_predict_file_rejects_oversized_upload(tmp_path):
    app = create_app(model_path=str(tmp_path / "m.joblib"), max_upload_bytes=32)
    _train_model(tmp_path / "m.joblib")
    r = TestClient(app).post(
        "/predict_file", files={"file": ("large.csv", b"x" * 33, "text/csv")})
    assert r.status_code == 413


def test_api_uses_threshold_stored_in_model(tmp_path):
    path = tmp_path / "m.joblib"
    model = _train_model(path)
    model.decision_threshold_ = 1.1
    import joblib
    joblib.dump(model, path)
    body = TestClient(create_app(model_path=str(path))).post(
        "/predict", json={"flight_id": "test", "rows": _rows()}).json()
    assert body["spoofing"] is False
    assert body["threshold"] == 1.1
