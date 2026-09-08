from src.config import SEED, WINDOW_SECONDS, STRIDE_SECONDS, NOMINAL_HZ


def test_constants():
    assert SEED == 42
    assert WINDOW_SECONDS == 2.0
    assert STRIDE_SECONDS == 0.5
    assert NOMINAL_HZ == 20.0
