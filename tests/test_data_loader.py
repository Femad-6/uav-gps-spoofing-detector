import pandas as pd

from src.data_loader import load_flight, infer_schema


def test_load_flight_normalized_units():
    df = load_flight("tests/data/load_fixture.csv")
    needed = {"time_s", "lat", "lon", "alt_gps", "alt_baro",
              "vel_e", "vel_n", "vel_u", "ax", "ay", "az",
              "gx", "gy", "gz", "roll", "pitch", "yaw", "attack"}
    assert needed <= set(df.columns)
    assert df["time_s"].is_monotonic_increasing
    # 向量字符串被拆分
    assert df["vel_e"].iloc[0] == 0.5
    assert df["az"].iloc[0] == 9.81


def test_load_flight_resamples_to_20hz():
    df = load_flight("tests/data/load_fixture.csv")
    assert df["time_s"].diff().dropna().between(0.049, 0.051).all()


def test_infer_schema_maps_common_names(tmp_path):
    schema = infer_schema(["timestamp", "lat", "lon", "alt", "vx", "vy", "vz", "baro_alt"])
    assert schema["time_s"] == "timestamp"
    assert schema["alt_baro"] == "baro_alt"  # baro 优先级高于 alt
    assert schema["alt_gps"] == "alt"
    assert schema["vel_e"] == "vx"
