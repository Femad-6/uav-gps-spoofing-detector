"""与模型无关的连续窗口报警确认和迟滞清除策略。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AlarmDecision:
    raw_alert: bool
    confirmed_alert: bool


class TemporalAlarmPolicy:
    """3-of-5 确认，确认后连续 3 个低分窗才清除。"""

    def __init__(self, threshold: float, confirm_count: int = 3,
                 confirm_window: int = 5, clear_count: int = 3,
                 clear_ratio: float = 0.8):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if not 1 <= confirm_count <= confirm_window:
            raise ValueError("confirm_count must be within confirm_window")
        if clear_count < 1 or not 0.0 < clear_ratio < 1.0:
            raise ValueError("invalid alarm clearing policy")
        self.threshold = float(threshold)
        self.confirm_count = int(confirm_count)
        self.clear_count = int(clear_count)
        self.clear_threshold = float(clear_ratio * threshold)
        self._recent: deque[bool] = deque(maxlen=int(confirm_window))
        self._confirmed = False
        self._below_clear = 0

    def update(self, probability: float) -> AlarmDecision:
        probability = float(probability)
        if not math.isfinite(probability):
            raise ValueError("probability must be finite")
        raw_alert = probability >= self.threshold
        self._recent.append(raw_alert)
        if not self._confirmed and sum(self._recent) >= self.confirm_count:
            self._confirmed = True
            self._below_clear = 0
        elif self._confirmed:
            if probability < self.clear_threshold:
                self._below_clear += 1
                if self._below_clear >= self.clear_count:
                    self._confirmed = False
                    self._below_clear = 0
                    self._recent.clear()
            else:
                self._below_clear = 0
        return AlarmDecision(raw_alert=raw_alert,
                             confirmed_alert=self._confirmed)

    def reset(self) -> None:
        self._recent.clear()
        self._confirmed = False
        self._below_clear = 0
