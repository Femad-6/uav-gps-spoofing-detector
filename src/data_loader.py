"""规范化数据加载器。

将任意飞行日志 CSV（PX4/Gazebo 风格列名；可能为 'a;b;c' 向量字符串或
x/y/z 拆分列）转换为统一的规范化 DataFrame（20Hz 均匀网格）。

规范化列名一览：
    time_s, lat, lon, alt_gps, alt_baro,
    vel_e, vel_n, vel_u,          # ENU, m/s
    ax, ay, az,                   # 加速度 m/s^2
    gx, gy, gz,                   # 角速度 rad/s
    roll, pitch, yaw,             # rad
    mx, my, mz,                   # 磁场（原单位）
    attack                        # 0/1 诱骗使能标志
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import NOMINAL_HZ

NORMALIZED_COLUMNS = [
    "time_s", "lat", "lon", "alt_gps", "alt_baro",
    "vel_e", "vel_n", "vel_u",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "roll", "pitch", "yaw",
    "mx", "my", "mz",
    "attack",
]


def _norm_name(col: str) -> str:
    """小写、下划线/连字符转空格，便于正则匹配。"""
    return unicodedata.normalize("NFKD", col).lower().replace("_", " ").replace("-", " ")


# 向量分量正则: 规范化名 -> (canonical, 轴)。按顺序先匹配先采纳。
COMP_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"velocity[ _]east|(^|_)vx$"), "vel_e", ""),
    (re.compile(r"velocity[ _]north|(^|_)vy$"), "vel_n", ""),
    (re.compile(r"velocity[ _]up|(^|_)vz$"), "vel_u", ""),
    (re.compile(r"linear acceleration (x|y|z)$|accel?[ _]?(x|y|z)$"), "acc", "axis"),
    (re.compile(r"angular velocity (x|y|z)$|gyro[ _]?(x|y|z)$"), "gyro", "axis"),
    (re.compile(r"magnetic field (x|y|z)$|mag(netic)?[ _]?(x|y|z)$"), "mag", "axis"),
    (re.compile(r"orientation[ _](x|y|z|w)$|quat[ _](x|y|z|w)$"), "quat", "axis"),
]


def infer_schema(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    """由列名推断 规范化名 -> 原始列名；未映射为 None。"""
    schema: Dict[str, Optional[str]] = {c: None for c in NORMALIZED_COLUMNS}
    # 额外追踪: 向量字符串列与四元数拆分列
    extras: Dict[str, Optional[str]] = {"_vel": None, "_acc": None, "_gyro": None,
                                        "_mag": None, "_quat": None}
    quat_axes: Dict[str, Optional[str]] = {f"_q{a}": None for a in "xyzw"}
    for col in columns:
        n = _norm_name(str(col))
        if not n:
            continue
        matched = False
        for pat, canon, group in COMP_PATTERNS:
            m = pat.search(n)
            if m:
                if group == "axis":
                    ax = m.group(1)
                    if canon == "acc":
                        if schema[f"a{ax}"] is None:
                            schema[f"a{ax}"] = col
                    elif canon == "gyro":
                        if schema[f"g{ax}"] is None:
                            schema[f"g{ax}"] = col
                    elif canon == "mag":
                        if schema[f"m{ax}"] is None:
                            schema[f"m{ax}"] = col
                    elif canon == "quat":
                        if quat_axes[f"_q{ax}"] is None:
                            quat_axes[f"_q{ax}"] = col
                else:
                    if schema[canon] is None:
                        schema[canon] = col
                matched = True
                break
        if matched:
            continue
        if re.search(r"pressure alt|baro", n):
            if schema["alt_baro"] is None:
                schema["alt_baro"] = col
        elif re.search(r"^time( usec| us| s)?$|^timestamp$", n):
            if schema["time_s"] is None:
                schema["time_s"] = col
        elif re.search(r"latitude", n):
            if schema["lat"] is None:
                schema["lat"] = col
        elif re.search(r"longitude", n):
            if schema["lon"] is None:
                schema["lon"] = col
        elif re.search(r"altitude$|(^|_)alt$", n):
            if schema["alt_gps"] is None:
                schema["alt_gps"] = col
        elif re.search(r"(^|_)velocity$", n):        # 向量字符串或标量（含分量时忽略）
            if extras["_vel"] is None:
                extras["_vel"] = col
        elif re.search(r"(^|_)linear acceleration$", n):
            if extras["_acc"] is None:
                extras["_acc"] = col
        elif re.search(r"(^|_)angular velocity$|(^|_)gyro$", n):
            if extras["_gyro"] is None:
                extras["_gyro"] = col
        elif re.search(r"(^|_)magnetic field$|(^|_)mag$", n):
            if extras["_mag"] is None:
                extras["_mag"] = col
        elif re.search(r"(^|_)orientation$|(^|_)quat$", n):
            if extras["_quat"] is None:
                extras["_quat"] = col
        elif re.search(r"(^|_)roll$|(^|_)pitch$|(^|_)yaw$", n):
            axis = re.search(r"(roll|pitch|yaw)", n).group(1)
            if schema[axis] is None:
                schema[axis] = col
        elif re.search(r"attack|spoof", n):
            if schema["attack"] is None:
                schema["attack"] = col
    for k, v in extras.items():
        schema[k] = v
    for k, v in quat_axes.items():
        schema[k] = v
    return schema


def suffixed_usec(col: str) -> bool:
    return "_usec" in col or "_us" in col


def _to_vec(series: pd.Series) -> List[List[float]]:
    """'a;b;c' 向量字符串或标量 → 元素为数值列表的列表。"""
    out = []
    for v in series.astype(object):
        if isinstance(v, str) and ";" in v:
            parts = [float(x) for x in v.split(";")]
        else:
            try:
                parts = [float(v)]
            except (TypeError, ValueError):
                parts = []
        out.append(parts)
    return out


def _vec_to_columns(mat: List[List[float]], names: List[str]) -> Dict[str, np.ndarray]:
    vals = {n: np.full(len(mat), np.nan, dtype=float) for n in names}
    for i, parts in enumerate(mat):
        if len(parts) == len(names):
            for n, p in zip(names, parts):
                vals[n][i] = p
    return vals


def _quat_to_rpy(comp: Dict[str, Optional[str]], raw: pd.DataFrame) -> Dict[str, np.ndarray]:
    """四元数 → roll/pitch/yaw(rad)。优先级: x/y/z/w 拆分列 > 向量字符串(按 x;y;z;w)。"""
    n = len(raw)
    res = {k: np.full(n, np.nan) for k in ("roll", "pitch", "yaw")}
    if all(comp.get(f"_q{a}") for a in "xyzw"):
        qx = pd.to_numeric(raw[comp["_qx"]], errors="coerce").to_numpy(dtype=float)
        qy = pd.to_numeric(raw[comp["_qy"]], errors="coerce").to_numpy(dtype=float)
        qz = pd.to_numeric(raw[comp["_qz"]], errors="coerce").to_numpy(dtype=float)
        qw = pd.to_numeric(raw[comp["_qw"]], errors="coerce").to_numpy(dtype=float)
    elif comp.get("_quat"):
        mat = _to_vec(raw[comp["_quat"]])
        if mat and len(mat[0]) == 4:
            arr = np.asarray(mat, dtype=float)
            qx, qy, qz, qw = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        elif mat and len(mat[0]) == 3:   # 3 元视为欧拉角 rad
            return _vec_to_columns(mat, ["roll", "pitch", "yaw"])
        else:
            return res
    else:
        return res
    ok = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(qz) & np.isfinite(qw)
    with np.errstate(invalid="ignore"):
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    res["roll"][ok] = roll[ok]
    res["pitch"][ok] = pitch[ok]
    res["yaw"][ok] = yaw[ok]
    return res


def load_flight(path: str | Path, resample_hz: float = NOMINAL_HZ) -> pd.DataFrame:
    """读取单个 CSV → 规范化 DataFrame（20Hz 均匀网格）。

    - '_usec' 时间列按微秒转秒（并排序）；
    - 支持拆分列 / ';' 向量字符串两种形态；
    - 缺失列以 NaN 空列补齐。
    """
    raw = pd.read_csv(path)
    raw = raw.rename(columns=lambda c: str(c).strip())
    schema = infer_schema(raw.columns)
    n = len(raw)

    df = pd.DataFrame()
    tcol = schema["time_s"]
    if tcol:
        t = pd.to_numeric(raw[tcol], errors="coerce").to_numpy(dtype=float)
        if suffixed_usec(tcol):
            t = t / 1e6
        order = np.argsort(t)
        raw = raw.iloc[order].reset_index(drop=True)
        t = t[order]
    else:
        t = np.arange(n) * (1.0 / resample_hz)

    # 标量列
    for canon in ("lat", "lon", "alt_gps", "alt_baro"):
        col = schema.get(canon)
        if col:
            df[canon] = pd.to_numeric(raw[col], errors="coerce")
    # 速度/加速度/角速度/磁场：分量列优先，缺省时向量字符串兜底
    for canon, extra in (("vel", "_vel"), ("acc", "_acc"),
                         ("gyro", "_gyro"), ("mag", "_mag")):
        names = {"vel": ["vel_e", "vel_n", "vel_u"], "acc": ["ax", "ay", "az"],
                 "gyro": ["gx", "gy", "gz"], "mag": ["mx", "my", "mz"]}[canon]
        cols = [schema.get(x) for x in names]
        if cols and all(cols):
            for x, c in zip(names, cols):
                df[x] = pd.to_numeric(raw[c], errors="coerce")
        elif schema.get(extra):
            vec = _vec_to_columns(_to_vec(raw[schema[extra]]), names)
            for x in names:
                df[x] = vec[x]
    # 姿态：拆分列/向量字符串，独立的 roll/pitch/yaw 列最后覆盖
    rpy = _quat_to_rpy(schema, raw)
    for k in ("roll", "pitch", "yaw"):
        df[k] = rpy[k]
    for k in ("roll", "pitch", "yaw"):
        col = schema.get(k)
        if col:
            df[k] = pd.to_numeric(raw[col], errors="coerce")
    # 攻击标签
    if schema.get("attack"):
        a = pd.to_numeric(raw[schema["attack"]].astype(str)
                          .str.replace("True", "1").str.replace("False", "0"), errors="coerce")
        df["attack"] = a.fillna(0.0)
    else:
        df["attack"] = 0.0

    # 统一重采样
    df["time_s"] = t
    df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    if len(df) < 3 or df["time_s"].nunique() < 3:
        return df.reindex(columns=NORMALIZED_COLUMNS)
    grid = np.arange(df["time_s"].iloc[0], df["time_s"].iloc[-1], 1.0 / resample_hz)
    num_cols = [c for c in NORMALIZED_COLUMNS if c != "attack"]
    resampled = pd.DataFrame({"time_s": grid})
    for c in num_cols:
        y = df[c].to_numpy(dtype=float)
        if np.isnan(y).all():
            resampled[c] = np.nan
        else:
            ok = ~np.isnan(y)
            resampled[c] = np.interp(grid, df["time_s"][ok], y[ok])
    y = df["attack"].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(df["time_s"], grid) - 1, 0, len(y) - 1)
    resampled["attack"] = y[idx] if len(y) else 0.0
    return resampled.reindex(columns=NORMALIZED_COLUMNS)


def load_all(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """加载 merged 日志（log_*.csv）目录下全部航班。flight_id 由相对路径生成。"""
    data_dir = Path(data_dir)
    flights: Dict[str, pd.DataFrame] = {}
    csvs = sorted(p for p in data_dir.rglob("log_*.csv"))
    for p in csvs:
        try:
            df = load_flight(p)
        except Exception as e:
            print(f"[warn] 跳过 {p}: {e}")
            continue
        if len(df) < 3:
            continue
        rel = p.relative_to(data_dir).with_suffix("").as_posix().replace("/", "_")
        flights[rel] = df
    return flights
