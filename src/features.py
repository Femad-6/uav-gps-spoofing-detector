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
