"""规范化数据加载器。

将任意飞行日志 CSV（PX4/Gazebo 风格的列名、可能含 ';' 分隔的向量字符串）
转换为统一的规范化 DataFrame（列见 NORMALIZED_COLUMNS，20Hz 均匀网格）。

规范化列名一览：
    time_s, lat, lon, alt_gps, alt_baro,
    vel_e, vel_n, vel_u,          # ENU, m/s
    ax, ay, az,                   # 加速度 m/s^2
    gx, gy, gz,                   # 角速度 rad/s
    roll, pitch, yaw,             # rad
    mx, my, mz,
    attack                       # 0/1 诱骗使能标志（无该列则为空列）
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

# 列名模式 -> 规范化名。按列表顺序**先匹配先采纳**。
# 注意：alt_baro 相关模式必须先于 alt_gps。
PATTERNS: List[Tuple[str, str]] = [
    (r"pressure\s*alt|baro|barometric", "alt_baro"),
    (r"timestamp|_usec|_us$|^time$|time_s|^time\b", "time_s"),
    (r"latitude|^lat$|^lat\b", "lat"),
    (r"longitude|^lon$|^lon\b", "lon"),
    (r"altitude|^alt$|^alt\b", "alt_gps"),
    (r"velocity", "velocity"),          # 向量或标量，暂存后展开
    (r"vx|vel[_-]?e|vel_e", "vel_e"),
    (r"vy|vel[_-]?n|vel_n", "vel_n"),
    (r"vz|vel[_-]?u|vel_u", "vel_u"),
    (r"linear\s*acc|accel|^ax$|^ay$|^az$", "accels"),  # 向量或分量
    (r"angular\s*vel|gyro", "gyros"),
    (r"orientation|quat", "orientation"),
    (r"mag(netic)?|^mx$|^my$|^mz$", "mags"),
    (r"attack|spoof", "attack"),
    (r"(^|[_-])(roll|pitch|yaw)", "rpy"),
]


def _norm_name(col: str) -> str:
    """小写化并把下划线转空格，便于模式匹配。"""
    return (
        unicodedata.normalize("NFKD", col)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def infer_schema(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    """由列名列表推断 规范化名 -> 原始列名。未映射的规范化名值为 None。"""
    schema: Dict[str, Optional[str]] = {c: None for c in NORMALIZED_COLUMNS}
    used: set = set()
    for col in columns:
        norm = _norm_name(str(col))
        for pat, target in PATTERNS:
            if re.search(pat, norm):
                if target in ("velocity", "accels", "gyros", "orientation", "mags", "rpy"):
                    # 暂存到临时键，由 resolve 阶段拆解
                    schema.setdefault(f"_{target}", col)
                else:
                    if schema.get(target) is None:
                        schema[target] = col
                used.add(col)
                break
    # 向量类列名单独记录（含分量列的情况在解析列时处理）
    for target in ("velocity", "accels", "gyros", "orientation", "mags", "rpy"):
        col = schema.get(f"_{target}") if f"_{target}" in schema else None
        # 上面 setdefault 已写入；这里仅清理临时键与目标非必须
    for k in list(schema.keys()):
        if k.startswith("_"):
            if k[1:] in ("velocity", "accels", "gyros", "orientation", "mags") and schema.get(k[1:]) is None:
                tmp = schema.pop(k)
                schema[f"_{k[1:]}"] = tmp if k.startswith("_") else None
    return schema


def _to_vec(series: pd.Series) -> List[np.ndarray]:
    """把列中的 '1;2;3' 或纯数值转换为 N x 3 矩阵；失败返回空列表。"""
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


def _vec_to_columns(mat: List[List[float]], names: List[str], df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """将向量列表拆成命名列；len(部分)!=3 时返回全 NaN。"""
    vals = {n: np.full(len(mat), np.nan, dtype=float) for n in names}
    for i, parts in enumerate(mat):
        if len(parts) == 3:
            for n, p in zip(names, parts):
                vals[n][i] = p
    return vals


def _quat_to_rpy(mat: List[List[float]]) -> Dict[str, np.ndarray]:
    """四元数 (w;x;y;z) -> roll/pitch/yaw (rad)。若为 3 元则视为欧拉角(rad)。"""
    if mat and len(mat[0]) == 3:
        return _vec_to_columns(mat, ["roll", "pitch", "yaw"], None)  # 3 元素直接当欧拉角
    n = len(mat)
    res = {k: np.full(n, np.nan) for k in ("roll", "pitch", "yaw")}
    if not mat or len(mat[0]) != 4:
        return res
    for i, (w, x, y, z) in enumerate(mat):
        res["roll"][i] = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        res["pitch"][i] = np.arcsin(min(1.0, max(-1.0, 2 * (w * y - z * x))))
        res["yaw"][i] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return res


def load_flight(path: str | Path, resample_hz: float = NOMINAL_HZ) -> pd.DataFrame:
    """读取单个 CSV 并返回规范化 DataFrame（20Hz 均匀网格）。

    - 时间列 '_usec' 结尾的按微秒转秒；
    - 向量字符串 'a;b;c' 拆成 3 列（Velocity/Linear Acceleration/Angular
      Velocity/Magnetic Field/四元数 Orientation）；
    - 缺失的规范化列以全 NaN 空列补齐。
    """
    raw = pd.read_csv(path)
    raw = raw.rename(columns=lambda c: str(c).strip())
    schema = infer_schema(raw.columns)

    df = pd.DataFrame()
    # 1) 时间
    tcol = schema["time_s"]
    t = pd.to_numeric(raw[tcol], errors="coerce") if tcol else pd.Series(np.arange(len(raw)) * (1.0 / resample_hz))
    if tcol and suffixed_usec(tcol):
        t = t / 1e6
    order = t.argsort()
    raw = raw.iloc[order].reset_index(drop=True)
    t = pd.to_numeric(raw[schema["time_s"]], errors="coerce") if schema["time_s"] else t.iloc[order]

    # 2) 标量列
    scalar_map = {"lat": None, "lon": None, "alt_gps": None, "alt_baro": None,
                  "vel_e": None, "vel_n": None, "vel_u": None, "ax": None, "ay": None, "az": None,
                  "gx": None, "gy": None, "gz": None, "roll": None, "pitch": None, "yaw": None,
                  "mx": None, "my": None, "mz": None, "attack": None}
    for canon in scalar_map:
        col = schema.get(canon)
        if col:
            df[canon] = pd.to_numeric(raw[col].astype(str).str.extract(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")[0]
                                      if raw[col].dtype == object and raw[col].astype(str).str.contains(";").any()
                                      else raw[col], errors="coerce")
    # 向量列展开
    vec_cols = {}
    for canon, names in [("velocity", ["vel_e", "vel_n", "vel_u"]),
                         ("accels", ["ax", "ay", "az"]),
                         ("gyros", ["gx", "gy", "gz"]),
                         ("mags", ["mx", "my", "mz"])]:
        col = schema.get(f"_{canon}")
        if col is None:
            col = schema.get(canon)
        if col:
            vec_cols.update(_vec_to_columns(_to_vec(raw[col]), names, raw))
    if schema.get("_orientation") or schema.get("orientation"):
        col = schema.get("_orientation") or schema.get("orientation")
        vec_cols.update(_quat_to_rpy(_to_vec(raw[col])))
    for k, v in vec_cols.items():
        df[k] = v
    # 若分量列直接存在（如 vx/vy/vz），覆盖
    for canon in ("vel_e", "vel_n", "vel_u", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "roll", "pitch", "yaw"):
        col = schema.get(canon)
        if col and canon not in df:
            df[canon] = pd.to_numeric(raw[col], errors="coerce")

    # 3) 归一化残差：attack 若为 'Attack: True/False' 式字符串先转换
    if schema.get("attack") and schema["attack"] not in df:
        a = raw[schema["attack"]]
        df["attack"] = pd.to_numeric(a.astype(str).str.replace("True", "1").str.replace("False", "0"), errors="coerce").fillna(0)

    # 4) 统一时间与重采样
    df["time_s"] = t.to_numpy(dtype=float)
    df = df.dropna(subset=["time_s"])
    df = df.sort_values("time_s").reset_index(drop=True)
    if df["time_s"].nunique() < 3:
        return df.reindex(columns=NORMALIZED_COLUMNS)
    grid = np.arange(df["time_s"].iloc[0], df["time_s"].iloc[-1], 1.0 / resample_hz)
    num_cols = [c for c in df.columns if c in NORMALIZED_COLUMNS and c != "attack"]
    resampled = pd.DataFrame({"time_s": grid})
    for c in num_cols:
        y = df[c].to_numpy(dtype=float)
        if np.isnan(y).all():
            resampled[c] = np.nan
        else:
            ok = ~np.isnan(y)
            resampled[c] = np.interp(grid, df["time_s"][ok], y[ok])
    if "attack" in df.columns:
        y = df["attack"].to_numpy(dtype=float)
        idx = np.clip(np.searchsorted(df["time_s"], grid) - 1, 0, len(y) - 1)
        resampled["attack"] = y[idx] if len(y) > 0 else 0.0
    else:
        resampled["attack"] = 0.0
    return resampled.reindex(columns=NORMALIZED_COLUMNS)


def suffixed_usec(col: str) -> bool:
    return "_usec" in col or "_us" in col


def load_all(data_dir: str | Path, only_merged: bool = True) -> Dict[str, pd.DataFrame]:
    """加载目录下全部日志。默认仅取 merged 目录（时间同步版）。

    flight_id 由相对路径生成（如 'Merged/Curved/Attacked/log_...'）。
    """
    data_dir = Path(data_dir)
    flights: Dict[str, pd.DataFrame] = {}
    csvs = sorted(p for p in data_dir.rglob("*.csv") if "raw" not in str(p).lower() or not only_merged)
    if only_merged:
        csvs = [p for p in data_dir.rglob("*.csv") if "merged" in str(p).lower() or (not only_merged)]
    # 若没有名为 merged 的目录，退化为所有 log_*.csv
    if not csvs:
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
