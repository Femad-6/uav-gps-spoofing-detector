from src.labels import assign_split


def test_assign_split_disjoint_and_reproducible():
    flights = [f"f{i}" for i in range(20)]
    s1 = assign_split(flights)
    assert set(s1.values()) <= {"train", "test"} and "test" in s1.values()
    assert assign_split(flights) == s1
    s2 = assign_split(flights, seed=7)
    assert s2 != s1
