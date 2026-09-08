"""因果流式检测：逐行喂入规范化数据，满窗（40 行）即产出推理。

只使用当前与历史数据（buffer 累积），不偷看未来，符合在线部署语义。
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.config import STRIDE_SECONDS, WINDOW_SECONDS
from src.features import WINDOW_SIZE, STRIDE, window_extract

DEFAULT_THRESHOLD = 0.5


def model_predictor(model) -> Callable[[pd.Series], float]:
    """把 sklearn 模型包装成 特征行 -> prob_1 的调用。"""
    from src.models import predict_proba_all

    def _p(row: pd.Series) -> float:
        df = row.to_frame().T
        return float(predict_proba_all(model, df)["prob_1"].iloc[0])

    return _p


class CausalStreamer:
    """因果流式检测器。

    feed(row: dict 规范化行) -> 满窗后每 STRIDE 行返回窗口预测 dict 或 None。
    只保留最近 WINDOW_SIZE 行（定长环形缓冲），逐窗推理，不偷看未来。
    """

    def __init__(self, predictor: Callable[[pd.Series], float],
                 feature_extractor: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
                 threshold: float = DEFAULT_THRESHOLD):
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.buf: deque = deque(maxlen=WINDOW_SIZE)
        self.total = 0
        self.alerts: List[dict] = []

    def _features(self, buf_df: pd.DataFrame) -> pd.Series:
        if self.feature_extractor is not None:
            return self.feature_extractor(buf_df)
        w = window_extract(buf_df)
        if len(w) == 0:
            raise ValueError("buffer 样本不足")
        return w.iloc[0].drop(labels=["flight_id", "window_start_s", "label"])

    def feed(self, row: dict) -> Optional[dict]:
        self.buf.append(row)
        self.total += 1
        if self.total < WINDOW_SIZE or (self.total - WINDOW_SIZE) % STRIDE != 0:
            return None
        buf_df = pd.DataFrame(self.buf)
        feats = self._features(buf_df)
        prob = float(self.predictor(feats))
        alert = prob >= self.threshold
        ws = float(self.buf[0]["time_s"])
        out = {"window_start_s": ws, "prob": prob, "alert": alert}
        if alert:
            self.alerts.append(out)
        return out

    def alarm_intervals(self) -> List[Tuple[float, float]]:
        """合并相邻报警窗（间隔 <= 2*STRIDE 视为连续）为 [start, end]。"""
        if not self.alerts:
            return []
        ws = [a["window_start_s"] for a in self.alerts]
        intervals: List[List[float]] = [[ws[0], ws[0] + WINDOW_SECONDS]]
        for s in ws[1:]:
            if s - intervals[-1][1] <= 2 * STRIDE * 0.05:
                intervals[-1][1] = max(intervals[-1][1], s + WINDOW_SECONDS)
            else:
                intervals.append([s, s + WINDOW_SECONDS])
        return [(float(a), float(b)) for a, b in intervals]
