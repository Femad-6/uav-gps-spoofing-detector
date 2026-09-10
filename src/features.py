"""滑窗特征工程：L1 原始统计 / L2 物理一致性 / L3 时序聚合。

输入：规范化单航班 DataFrame（20Hz，列见 data_loader.NORMALIZED_COLUMNS）。
输出：一行一窗的 DataFrame，含 flight_id / window_start_s / label 与
      f_<名>_<聚合>（L1+L3）、c_<名>（L2）特征列。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import NOMINAL_HZ, STRIDE_SECONDS, WINDOW_SECONDS
from src.data_loader import NORMALIZED_COLUMNS

WINDOW_SIZE = int(round(WINDOW_SECONDS * NOMINAL_HZ))    # 40 样本
STRIDE = int(round(STRIDE_SECONDS * NOMINAL_HZ))         # 10 样本
EPS = 1e-6
AGGS = ("mean", "std", "min", "max", "slope", "kurt")

# L1 统计对象（每窗对其计算 6 种聚合）
L1_QUANTITIES = ["alt_gps", "alt_baro", "d_alt", "speed_horiz", "vel_u",
                 "acc_norm", "gyro_norm", "roll", "pitch", "yaw", "jerk"]
# L2 一致性特征
L2_NAMES = ["c_imu_gps_acc", "c_gps_baro", "c_heading_vel", "c_jump", "c_smoothness"]
ENHANCED_NAMES = [
    "e_gps_step_mean", "e_gps_step_std", "e_gps_step_max",
    "e_gps_speed_resid_mean", "e_gps_speed_resid_max",
    "e_track_yaw_mean_abs", "e_track_yaw_change_corr", "e_track_yaw_slope_diff",
    "e_mag_norm_mean", "e_mag_norm_std", "e_mag_norm_cv", "e_mag_axis_change_max",
    "e_yaw_mag_change_corr", "e_yaw_mag_resid_std",
    "e_imu_gps_acc_h_mean", "e_imu_gps_acc_h_max", "e_imu_gps_acc_dir_mean",
    "e_vibration_rms", "e_vibration_peak",
    "e_gps_baro_abs_mean", "e_gps_baro_abs_max",
]


def _slope(y: np.ndarray) -> float:
    if len(y) >= 2 and np.isfinite(y).sum() >= 2:
        return float(np.polyfit(np.arange(len(y)), y, 1)[0])
    return np.nan


def _kurt(y: np.ndarray) -> float:
    if len(y) >= 4 and np.isfinite(y).sum() >= 4:
        return float(pd.Series(y).kurt())
    return np.nan


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    """有限、非退化序列的 Pearson 相关系数。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < EPS or np.std(b[ok]) < EPS:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _local_enu(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """小范围经纬度转相对 east/north 米坐标，不保留绝对位置。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        empty = np.full(len(lat), np.nan)
        return empty.copy(), empty
    first = np.flatnonzero(ok)[0]
    lat0, lon0 = lat[first], lon[first]
    north = (lat - lat0) * 111320.0
    east = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    return east, north


def _safe_stats(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    return float(np.mean(values)), float(np.std(values)), float(np.max(values))


def _enhanced_feats(cols: Dict[str, np.ndarray]) -> Dict[str, float]:
    """论文启发的相对轨迹、磁航向、振动和导航系一致性特征。"""
    out = {name: np.nan for name in ENHANCED_NAMES}

    # 相对位置与航迹：避免将绝对经纬度/试验地点泄漏给模型。
    east, north = _local_enu(cols["lat"], cols["lon"])
    de, dn = np.diff(east), np.diff(north)
    step = np.hypot(de, dn)
    step_mean, step_std, step_max = _safe_stats(step)
    out.update(e_gps_step_mean=step_mean, e_gps_step_std=step_std,
               e_gps_step_max=step_max)
    gps_speed = step * NOMINAL_HZ
    reported_speed = np.hypot(cols["vel_e"][1:], cols["vel_n"][1:])
    speed_resid = np.abs(gps_speed - reported_speed)
    resid_mean, _, resid_max = _safe_stats(speed_resid)
    out.update(e_gps_speed_resid_mean=resid_mean, e_gps_speed_resid_max=resid_max)

    track_heading = np.arctan2(de, dn)
    yaw = np.asarray(cols["yaw"][1:], dtype=float)
    moving = np.isfinite(track_heading) & np.isfinite(yaw) & np.isfinite(reported_speed)
    moving &= reported_speed >= 0.2
    if moving.sum() >= 3:
        track_valid = np.unwrap(track_heading[moving])
        yaw_valid = np.unwrap(yaw[moving])
        out["e_track_yaw_mean_abs"] = float(
            np.mean(np.abs(_wrap_angle(track_heading[moving] - yaw[moving]))))
        if len(track_valid) >= 4:
            out["e_track_yaw_change_corr"] = _finite_corr(
                np.diff(track_valid), np.diff(yaw_valid))
        out["e_track_yaw_slope_diff"] = float(abs(
            _slope(track_valid) - _slope(yaw_valid)))

    # 磁力计：使用模长稳定性和倾斜补偿后的相对航向变化。
    mx, my, mz = (np.asarray(cols[c], dtype=float) for c in ("mx", "my", "mz"))
    mag_norm = np.sqrt(mx * mx + my * my + mz * mz)
    mag_mean, mag_std, _ = _safe_stats(mag_norm)
    out["e_mag_norm_mean"] = mag_mean
    out["e_mag_norm_std"] = mag_std
    out["e_mag_norm_cv"] = (mag_std / (abs(mag_mean) + EPS)
                            if np.isfinite(mag_mean) and np.isfinite(mag_std) else np.nan)
    mag_delta = np.sqrt(np.diff(mx) ** 2 + np.diff(my) ** 2 + np.diff(mz) ** 2)
    _, _, out["e_mag_axis_change_max"] = _safe_stats(mag_delta)
    roll = np.asarray(cols["roll"], dtype=float)
    pitch = np.asarray(cols["pitch"], dtype=float)
    yaw_full = np.asarray(cols["yaw"], dtype=float)
    mx_h = mx * np.cos(pitch) + mz * np.sin(pitch)
    my_h = (mx * np.sin(roll) * np.sin(pitch) + my * np.cos(roll)
            - mz * np.sin(roll) * np.cos(pitch))
    mag_heading = np.arctan2(-my_h, mx_h)
    mag_ok = np.isfinite(mag_heading) & np.isfinite(yaw_full)
    if mag_ok.sum() >= 4:
        mh = np.unwrap(mag_heading[mag_ok])
        yh = np.unwrap(yaw_full[mag_ok])
        out["e_yaw_mag_change_corr"] = _finite_corr(np.diff(mh), np.diff(yh))
        out["e_yaw_mag_resid_std"] = float(
            np.std(_wrap_angle(mag_heading[mag_ok] - yaw_full[mag_ok])))

    # 机体系加速度旋转到导航系，仅比较水平分量以避开重力符号歧义。
    ax, ay, az = (np.asarray(cols[c], dtype=float) for c in ("ax", "ay", "az"))
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw_full), np.cos(yaw_full)
    acc_e = cy * cp * ax + (cy * sp * sr - sy * cr) * ay + (cy * sp * cr + sy * sr) * az
    acc_n = sy * cp * ax + (sy * sp * sr + cy * cr) * ay + (sy * sp * cr - cy * sr) * az
    gps_acc_e = np.gradient(cols["vel_e"], 1.0 / NOMINAL_HZ)
    gps_acc_n = np.gradient(cols["vel_n"], 1.0 / NOMINAL_HZ)
    acc_resid = np.hypot(acc_e - gps_acc_e, acc_n - gps_acc_n)
    acc_mean, _, acc_max = _safe_stats(acc_resid)
    out.update(e_imu_gps_acc_h_mean=acc_mean, e_imu_gps_acc_h_max=acc_max)
    acc_body_heading = np.arctan2(acc_e, acc_n)
    acc_gps_heading = np.arctan2(gps_acc_e, gps_acc_n)
    acc_moving = np.hypot(acc_e, acc_n) >= 0.1
    acc_moving &= np.hypot(gps_acc_e, gps_acc_n) >= 0.1
    acc_moving &= np.isfinite(acc_body_heading) & np.isfinite(acc_gps_heading)
    if acc_moving.any():
        out["e_imu_gps_acc_dir_mean"] = float(np.mean(np.abs(
            _wrap_angle(acc_body_heading[acc_moving] - acc_gps_heading[acc_moving]))))

    acc_norm = np.sqrt(ax * ax + ay * ay + az * az)
    finite_acc = acc_norm[np.isfinite(acc_norm)]
    if len(finite_acc):
        centered = finite_acc - np.median(finite_acc)
        out["e_vibration_rms"] = float(np.sqrt(np.mean(centered * centered)))
        out["e_vibration_peak"] = float(np.max(np.abs(centered)))
    d_alt_abs = np.abs(np.asarray(cols["alt_gps"]) - np.asarray(cols["alt_baro"]))
    alt_mean, _, alt_max = _safe_stats(d_alt_abs)
    out.update(e_gps_baro_abs_mean=alt_mean, e_gps_baro_abs_max=alt_max)
    return out


def _window_feats(w: np.ndarray) -> Dict[str, float]:
    """w: 40×N 的规范化数值矩阵（列序与 NORMALIZED_COLUMNS 一致，不含 attack）。"""
    cols = {c: w[:, i] for i, c in enumerate([c for c in NORMALIZED_COLUMNS if c != "attack"])}
    out: Dict[str, float] = {}

    # ---- 派生量（每样本）----
    speed_horiz = np.hypot(cols["vel_e"], cols["vel_n"])
    acc_norm = np.sqrt(cols["ax"] ** 2 + cols["ay"] ** 2 + cols["az"] ** 2)
    gyro_norm = np.sqrt(cols["gx"] ** 2 + cols["gy"] ** 2 + cols["gz"] ** 2)
    d_alt = cols["alt_gps"] - cols["alt_baro"]
    jerk = np.abs(np.diff(acc_norm)) if len(acc_norm) >= 2 else np.array([np.nan])
    deriv = {k: v for k, v in {
        "alt_gps": cols["alt_gps"], "alt_baro": cols["alt_baro"], "d_alt": d_alt,
        "speed_horiz": speed_horiz, "vel_u": cols["vel_u"],
        "acc_norm": acc_norm, "gyro_norm": gyro_norm,
        "roll": cols["roll"], "pitch": cols["pitch"], "yaw": cols["yaw"],
        "jerk": jerk,
    }.items()}
    for name, y in deriv.items():
        y = np.asarray(y, dtype=float)
        if np.isfinite(y).sum() == 0:
            for agg in AGGS:
                out[f"f_{name}_{agg}"] = np.nan
            continue
        out[f"f_{name}_mean"] = float(np.nanmean(y))
        out[f"f_{name}_std"] = float(np.nanstd(y))
        out[f"f_{name}_min"] = float(np.nanmin(y))
        out[f"f_{name}_max"] = float(np.nanmax(y))
        out[f"f_{name}_slope"] = _slope(y)
        out[f"f_{name}_kurt"] = _kurt(y)

    # ---- L2 物理一致性 ----
    # 1) GPS 速度变化率 vs IMU 加速度（身体系，去重力偏置）
    if np.isfinite(cols["vel_e"]).sum() >= 3 and np.isfinite(acc_norm).sum() >= 3:
        vel = np.vstack([cols["vel_e"], cols["vel_n"], cols["vel_u"]])
        gps_acc = np.hypot(np.hypot(np.gradient(vel[0], 1 / NOMINAL_HZ),
                                    np.gradient(vel[1], 1 / NOMINAL_HZ)),
                           np.gradient(vel[2], 1 / NOMINAL_HZ))
        imu_nav = acc_norm - 9.81  # 身体系 z-up 加速度含重力偏置，近似去除
        res = np.abs(imu_nav - gps_acc)
        out["c_imu_gps_acc"] = float(np.nanmean(res))
    else:
        out["c_imu_gps_acc"] = np.nan
    # 2) GPS 高度 vs 气压计高度
    out["c_gps_baro"] = float(np.nanmean(d_alt)) if np.isfinite(d_alt).sum() else np.nan
    # 3) 速度矢向 vs 航向（yaw）
    if np.isfinite(cols["yaw"]).sum() >= 4 and np.isfinite(cols["vel_e"]).sum() >= 4:
        vel_heading = np.arctan2(cols["vel_e"] + 1e-9, cols["vel_n"] + 1e-12)
        out["c_heading_vel"] = float(np.nanmean(np.abs(_wrap_angle(vel_heading - cols["yaw"]))))
    else:
        out["c_heading_vel"] = np.nan
    # 4) 位置跳跃（水平位移样本间突变的 99 分位）
    if np.isfinite(cols["lat"]).sum() >= 2 and np.isfinite(cols["lon"]).sum() >= 2:
        lat_rad = np.radians(np.nanmean(cols["lat"]))
        dx = (cols["lat"][1:] - cols["lat"][:-1]) * 111320.0
        dy = (cols["lon"][1:] - cols["lon"][:-1]) * 111320.0 * np.cos(lat_rad)
        step = np.hypot(dx, dy)
        out["c_jump"] = float(np.nanpercentile(step, 99))
    else:
        out["c_jump"] = np.nan
    # 5) 平滑度（加速度二阶差分）
    if len(acc_norm) >= 3 and np.isfinite(acc_norm).sum() >= 3:
        out["c_smoothness"] = float(1.0 / (np.nanmean(np.abs(np.diff(acc_norm, 2))) + EPS))
    else:
        out["c_smoothness"] = np.nan
    out.update(_enhanced_feats(cols))
    return out


def attack_intervals(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """从 attack 列提取 [(start_s, end_s)]。"""
    if "attack" not in df.columns or df["attack"].isna().all():
        return []
    a = df["attack"].to_numpy(dtype=float)
    t = df["time_s"].to_numpy(dtype=float)
    s = a > 0.5
    if not s.any():
        return []
    idx = np.where(s)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return [(float(t[i]), float(t[j])) for i, j in zip(starts, ends)]


def window_extract(df: pd.DataFrame, intervals_override: Optional[List[Tuple[float, float]]] = None):
    """单航班窗口化。intervals_override=None 时尝试 df['attack'] 列。"""
    num_cols = [c for c in NORMALIZED_COLUMNS if c != "attack"]
    base = df.reindex(columns=NORMALIZED_COLUMNS)
    arr = base[num_cols].to_numpy(dtype=float)
    t = base["time_s"].to_numpy(dtype=float)
    n = len(arr)
    if n < WINDOW_SIZE:
        return pd.DataFrame(columns=["flight_id", "window_start_s", "label"])
    intervals = intervals_override if intervals_override is not None else attack_intervals(base)
    rows = []
    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        win = arr[start:start + WINDOW_SIZE]
        ws = float(t[start])
        feats = _window_feats(win)
        feats["flight_id"] = base.attrs.get("flight_id", "")
        feats["window_start_s"] = ws
        # 窗区间 [ws, ws+WINDOW_SECONDS) 与攻击区间的重叠比率
        overlap = 0.0
        for (as_, ae) in intervals:
            ov = min(ws + WINDOW_SECONDS, ae) - max(ws, as_)
            overlap += max(0.0, ov)
        feats["label"] = int(overlap / WINDOW_SECONDS >= 0.5)
        rows.append(feats)
    if not rows:
        return pd.DataFrame(columns=["flight_id", "window_start_s", "label"])
    return pd.DataFrame(rows)


def make_features(all_flights: Dict[str, pd.DataFrame],
                  labels: Optional[Dict[str, List[Tuple[float, float]]]] = None):
    """全航班窗口化；labels={flight_id: [(s,e),...]} 覆盖默认 attack 列标签。"""
    labels = labels or {}
    parts = []
    for fid, df in all_flights.items():
        df = df.assign(**{})
        d = df.copy()
        d.attrs["flight_id"] = fid
        w = window_extract(d, intervals_override=labels.get(fid))
        if len(w) and w["flight_id"].eq("").all():
            w["flight_id"] = fid
        parts.append(w)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "flight_id" not in out.columns:
        out["flight_id"] = ""
    return out.reset_index(drop=True)
