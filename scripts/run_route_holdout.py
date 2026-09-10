"""跨航线泛化实验：每次留出一种航线，比较 legacy 与增强模型。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np

from src.config import OUTPUT_DIR, SEED
from src.evaluate import evaluate_all
from src.labels import assign_route_holdout, load_dataset
from src.models import ThresholdDetector, fit_m1, predict_proba_all
from src.models_m2 import fit_m2, fit_m2e
from src.models_m4 import fit_m4, score_m4

ROUTES = ("Straight", "Curved", "Random")
MODELS = ("m0", "m1", "m1b", "m2", "m2e", "m4")


def fit_predict_model(name: str, data: pd.DataFrame) -> pd.DataFrame:
    """仅在 train 拟合指定模型，返回保留 split 的全数据预测。"""
    train = data[data["split"] == "train"]
    if name == "m0":
        model = ThresholdDetector().fit(train[train["label"] == 0])
        probabilities = np.array([model.score(row) for _, row in data.iterrows()])
        frame = data[["flight_id", "window_start_s", "label", "split"]].copy()
        frame["prob_1"] = probabilities
        frame["pred"] = (probabilities >= 0.5).astype(int)
        return frame
    if name == "m1":
        model = fit_m1(train, train["label"])
        return predict_proba_all(model, data)
    if name == "m1b":
        model = fit_m1(train, train["label"], features=("f_", "c_"))
        return predict_proba_all(model, data)
    if name == "m2":
        return predict_proba_all(fit_m2(train, train["label"]), data)
    if name == "m2e":
        return predict_proba_all(fit_m2e(train, train["label"]), data)
    if name == "m4":
        model = fit_m4(train[train["label"] == 0])
        probabilities = score_m4(model, data)
        frame = data[["flight_id", "window_start_s", "label", "split"]].copy()
        frame["prob_1"] = probabilities
        frame["pred"] = (probabilities >= 0.5).astype(int)
        return frame
    raise ValueError(f"unknown route-holdout model: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = ap.parse_args()

    X = load_dataset()
    results = {}
    flights = sorted(X["flight_id"].unique())
    for route in args.routes:
        data = X.copy()
        data["split"] = data["flight_id"].map(assign_route_holdout(flights, route, seed=SEED))
        for name in args.models:
            metrics = evaluate_all({name: fit_predict_model(name, data)})[name]
            results[f"{name}_holdout_{route.lower()}"] = metrics
            print(f"[{route}/{name}] AUROC={metrics['AUROC']:.3f}, "
                  f"F1={metrics['F1']:.3f}, FAR={metrics['false_alarm_rate']:.3f}, "
                  f"confirmed_FAR={metrics['confirmed_false_alarm_rate']:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "metrics_route_holdout.csv"
    pd.DataFrame(results).T.to_csv(out)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
