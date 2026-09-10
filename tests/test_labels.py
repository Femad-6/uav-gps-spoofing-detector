from src.labels import assign_route_holdout, assign_split


def test_assign_split_disjoint_and_reproducible():
    flights = [f"f{i}" for i in range(20)]
    s1 = assign_split(flights)
    assert set(s1.values()) == {"train", "val", "test"}
    assert assign_split(flights) == s1
    s2 = assign_split(flights, seed=7)
    assert s2 != s1


def test_assign_route_holdout_keeps_route_out_of_train_and_val():
    flights = [f"data_Merged_{route}_Normal_f{i}" for route in ("Straight", "Curved", "Random")
               for i in range(5)]
    split = assign_route_holdout(flights, "Straight")
    assert all(split[f] == "test" for f in flights if "_Straight_" in f)
    assert {split[f] for f in flights if "_Straight_" not in f} == {"train", "val"}
