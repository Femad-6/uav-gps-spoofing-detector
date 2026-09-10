import numpy as np
import pandas as pd

from src.features import window_extract, make_features, WINDOW_SIZE, STRIDE


def _df():
    t = np.arange(0, 10.0, 0.05)  # 200 samples
    return pd.DataFrame({"time_s": t, "lat": 34.0 + 1e-5 * t, "lon": 108.9,
                         "alt_gps": 20 + 0.5 * t, "alt_baro": 20 + 0.0 * t,
                         "vel_e": 0.1 * np.ones_like(t), "vel_n": np.zeros_like(t), "vel_u": np.ones_like(t),
                         "ax": np.zeros_like(t), "ay": np.zeros_like(t), "az": 9.81 * np.ones_like(t),
                         "gx": np.zeros_like(t), "gy": np.zeros_like(t), "gz": np.zeros_like(t),
                         "roll": np.zeros_like(t), "pitch": np.zeros_like(t), "yaw": np.zeros_like(t),
                         "mx": np.zeros_like(t), "my": np.zeros_like(t), "mz": np.zeros_like(t)})


def test_window_shapes():
    out = window_extract(_df())
    assert {"flight_id", "window_start_s", "label"} <= set(out.columns)
    assert out.shape[0] == 17   # (200-40)/10 + 1 = 17 窗
    assert "c_gps_baro" in out.columns
    assert "f_acc_norm_mean" in out.columns


def test_consistency_feature_catches_gps_baro_divergence():
    out = window_extract(_df())
    assert out["c_gps_baro"].mean() > 1.0   # GPS 高度平均高出 2m 以上


def test_make_features_labels_windows():
    df = _df()
    flights = {"f1": df}
    labels = {"f1": [(3.0, 7.0)]}
    X = make_features(flights, labels)
    assert set(X["label"]) <= {0, 1}
    assert X.loc[X["window_start_s"].between(4, 6), "label"].eq(1).all()


def test_enhanced_position_features_use_relative_motion():
    df = _df()
    out = window_extract(df).iloc[0]
    assert out["e_gps_step_mean"] > 0.0
    assert out["e_gps_step_max"] >= out["e_gps_step_mean"]
    assert out["e_gps_speed_resid_mean"] > 0.0


def test_track_yaw_wrap_does_not_create_full_circle_error():
    df = _df()
    df["lat"] = 34.0 - np.arange(len(df)) * 1e-7
    df["lon"] = 108.9
    df["vel_e"] = 0.0
    df["vel_n"] = -1.0
    df["yaw"] = -np.pi + 0.01
    out = window_extract(df).iloc[0]
    assert out["e_track_yaw_mean_abs"] < 0.1


def test_enhanced_magnetometer_and_vibration_features_are_finite():
    df = _df()
    df["mx"] = 1.0 + 0.1 * np.sin(df["time_s"])
    df["my"] = 0.2 * np.cos(df["time_s"])
    df["mz"] = 0.5
    df["ax"] = np.sin(10 * df["time_s"])
    out = window_extract(df).iloc[0]
    assert np.isfinite(out["e_mag_norm_mean"])
    assert out["e_mag_norm_std"] > 0.0
    assert out["e_vibration_rms"] > 0.0


def test_missing_magnetometer_yields_nan_enhanced_heading_features():
    df = _df()
    df[["mx", "my", "mz"]] = np.nan
    out = window_extract(df).iloc[0]
    assert np.isnan(out["e_yaw_mag_change_corr"])
    assert np.isnan(out["e_yaw_mag_resid_std"])


def test_level_body_acceleration_matches_navigation_horizontal_acceleration():
    df = _df()
    df["vel_e"] = df["time_s"]
    df["vel_n"] = 0.0
    df["ax"] = 1.0
    df["ay"] = 0.0
    df["az"] = 9.81
    out = window_extract(df).iloc[0]
    assert out["e_imu_gps_acc_h_mean"] < 0.05
