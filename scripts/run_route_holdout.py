"""跨航线泛化实验：每次留出一种航线，仅评测 M1 随机森林。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.config import OUTPUT_DIR, SEED
from src.evaluate import evaluate_all
from src.labels import assign_route_holdout, load_dataset
from src.models import fit_m1, predict_proba_all

ROUTES = ("Straight", "Curved", "Random")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    args = ap.parse_args()

    X = load_dataset()
    results = {}
    flights = sorted(X["flight_id"].unique())
    for route in args.routes:
        data = X.copy()
        data["split"] = data["flight_id"].map(assign_route_holdout(flights, route, seed=SEED))
        train = data[data["split"] == "train"]
        model = fit_m1(train, train["label"])
        metrics = evaluate_all({"m1": predict_proba_all(model, data)})["m1"]
        results[f"m1_holdout_{route.lower()}"] = metrics
        print(f"[{route}] AUROC={metrics['AUROC']:.3f}, F1={metrics['F1']:.3f}, "
              f"FAR={metrics['false_alarm_rate']:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "metrics_route_holdout.csv"
    pd.DataFrame(results).T.to_csv(out)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
