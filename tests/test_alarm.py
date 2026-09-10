import pytest

from src.alarm import TemporalAlarmPolicy


def test_temporal_policy_requires_three_of_five_scores():
    policy = TemporalAlarmPolicy(threshold=0.7)
    states = [policy.update(p).confirmed_alert
              for p in [0.8, 0.1, 0.9, 0.2, 0.95]]
    assert states == [False, False, False, False, True]


def test_temporal_policy_uses_hysteresis_to_clear():
    policy = TemporalAlarmPolicy(threshold=0.5)
    for probability in [0.9, 0.9, 0.9]:
        policy.update(probability)
    assert policy.update(0.45).confirmed_alert is True
    assert policy.update(0.39).confirmed_alert is True
    assert policy.update(0.39).confirmed_alert is True
    assert policy.update(0.39).confirmed_alert is False


def test_temporal_policy_reset_clears_flight_history():
    policy = TemporalAlarmPolicy(threshold=0.5)
    for probability in [0.9, 0.9, 0.9]:
        policy.update(probability)
    policy.reset()
    assert policy.update(0.1).confirmed_alert is False


@pytest.mark.parametrize("probability", [float("nan"), float("inf")])
def test_temporal_policy_rejects_non_finite_scores(probability):
    policy = TemporalAlarmPolicy(threshold=0.5)
    with pytest.raises(ValueError, match="finite"):
        policy.update(probability)
