import numpy as np

from src.stream import CausalStreamer
from src.features import WINDOW_SIZE, STRIDE


def _rows(n=200):
    t = np.arange(0, n) * 0.05
    return [{"time_s": float(x), "lat": 34.0, "lon": 108.9, "alt_gps": 20.0,
             "alt_baro": 20.0, "vel_e": 0.1, "vel_n": 0.0, "vel_u": 1.0,
             "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "mx": 0.0, "my": 0.0, "mz": 0.0,
             "attack": 0.0} for x in t]


def test_causal_stream_consistent_probability():
    s = CausalStreamer(lambda row: 0.9)
    results = [r for r in (s.feed(r) for r in _rows()) if r]
    assert len(results) > 0
    assert all(r["prob"] == 0.9 for r in results)
    assert all(r["alert"] is True for r in results)


def test_causal_stream_emits_after_full_window():
    s = CausalStreamer(lambda row: 0.1)
    emitted = sum(1 for r in _rows(WINDOW_SIZE) if s.feed(r) is not None)
    assert emitted == 1                     # 第 40 行到齐即产出第一窗
    emitted2 = sum(1 for r in _rows(2 * STRIDE) if s.feed(r) is not None)
    assert emitted2 == 2                    # 其后每 STRIDE 行产出一次


def test_alarm_intervals_merging():
    s = CausalStreamer(lambda row: 0.9)
    for r in _rows(200):
        if s.feed(r) is not None:
            pass
    iv = s.alarm_intervals()
    assert len(iv) == 1
    assert iv[0][0] == 2 * STRIDE * 0.05 and iv[0][1] > iv[0][0]


def test_single_high_window_is_not_confirmed_spoofing():
    scores = iter([0.9, 0.1, 0.1, 0.1, 0.1])
    streamer = CausalStreamer(lambda row: next(scores), threshold=0.5)
    results = [result for result in
               (streamer.feed(row) for row in _rows(WINDOW_SIZE + 4 * STRIDE))
               if result]
    assert results[0]["alert"] is True
    assert not any(result["confirmed_alert"] for result in results)
    assert streamer.alarm_intervals() == []


def test_three_of_five_high_windows_confirm_stream_alarm():
    scores = iter([0.9, 0.1, 0.9, 0.1, 0.9])
    streamer = CausalStreamer(lambda row: next(scores), threshold=0.5)
    results = [result for result in
               (streamer.feed(row) for row in _rows(WINDOW_SIZE + 4 * STRIDE))
               if result]
    assert results[-1]["confirmed_alert"] is True
    assert len(streamer.alarm_intervals()) == 1
