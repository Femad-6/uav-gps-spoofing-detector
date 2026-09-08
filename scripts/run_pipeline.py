"""全链路：训练 M0-M4 → 保存模型 → 评测 → 指标表 + 图表。

用法: python scripts/run_pipeline.py [--models m0,m1,m2,m3,m4] [--skip-m3]
前置: data_processed/features_windowed.parquet（由 scripts/make_dataset.py 生成）
输出: outputs/models/<name>.joblib, outputs/metrics.csv, outputs/figs/*.png
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, SEED, WINDOW_SECONDS
from src.data_loader import load_all
from src.evaluate import evaluate_all, plot_all, save_metrics
from src.features import WINDOW_SIZE, STRIDE
from src.labels import load_dataset
from src.models import ThresholdDetector, fit_m1, predict_proba_all
from src.models_m2 import fit_m2
from src.models_m3 import RAW_COLUMNS, fit_m3, predict_m3
from src.models_m4 import fit_m4, score_m4

ALL_MODELS = ["m0", "m1", "m2", "m3", "m4"]


def _m0_preds(detector: ThresholdDetector, X: pd.DataFrame) -> pd.DataFrame:
    probs = detector.score(X) if hasattr(detector, "score") else None
    # detector.score 接受 Series；逐行计算
    probs = np.array([detector.score(r) for _, r in X.iterrows()])
    out = X[["flight_id", "window_start_s", "label", "split"]].copy()
    out["prob_1"] = probs
    out["pred"] = (probs >= 0.5).astype(int)
    return out


def _seq_inputs_update() -> tuple:
    """为 M3 重建原始序列输入（与 X 的窗对齐）。返回 (X, seq_dict by (fid,ws))。"""
    flights = load_all(DATA_DIR)
    seqs: dict = {}
    for fid, df in flights.items():
        num = df[[c for c in RAW_COLUMNS if c in df.columns]]
        arr = num.to_numpy(dtype=float)
        t = df["time_s"].to_numpy(dtype=float)
        n = len(arr)
        if n < WINDOW_SIZE:
            continue
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            win = arr[start:start + WINDOW_SIZE]
            ws = float(t[start])
            if np.isfinite(win).all():
                seqs[(fid, round(ws, 3))] = win
    return seqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--skip-m3", action="store_true")
    args = ap.parse_args()
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.skip_m3:
        wanted = [m for m in wanted if m != "m3"]

    X = load_dataset()
    if X.empty:
        print("[error] 未找到窗级数据集，请先运行 scripts/make_dataset.py")
        return 1
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tr, te = X[X["split"] == "train"], X[X["split"] == "test"]
    print(f"[data] 窗数 {len(X)} (train {len(tr)} / test {len(te)})")

    preds, models = {}, {}

    def _finalize(name: str, frame: pd.DataFrame, model) -> None:
        frame = frame.reset_index(drop=True)
        preds[name] = frame
        models[name] = model
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"[ok] {name} 训练并保存 -> {MODEL_DIR / (name + '.joblib')}")

    if "m0" in wanted:
        t0 = time.time()
        det = ThresholdDetector().fit(tr)
        _finalize("m0", _m0_preds(det, X), det)
        print(f"     训练耗时 {time.time() - t0:.1f}s")

    if "m1" in wanted:
        t0 = time.time()
        m1 = fit_m1(tr, tr["label"])
        _finalize("m1", predict_proba_all(m1, X), m1)
        print(f"     训练耗时 {time.time() - t0:.1f}s")

    if "m2" in wanted:
        t0 = time.time()
        m2 = fit_m2(tr, tr["label"])
        _finalize("m2", predict_proba_all(m2, X), m2)
        print(f"     训练耗时 {time.time() - t0:.1f}s")

    if "m4" in wanted:
        t0 = time.time()
        m4 = fit_m4(tr[tr["label"] == 0])
        scores = score_m4(m4, X)
        frame = X[["flight_id", "window_start_s", "label", "split"]].copy()
        frame["prob_1"] = scores
        frame["pred"] = (scores >= 0.5).astype(int)
        _finalize("m4", frame, m4)
        print(f"     训练耗时 {time.time() - t0:.1f}s")

    if "m3" in wanted and not args.skip_m3:
        t0 = time.time()
        try:
            import torch  # noqa: F401
            seqs = _seq_inputs_update()
            keys = list(seqs.keys())
            X_seq = np.stack([seqs[k] for k in keys])
            # 与窗级数据集按 (flight_id, window_start_s) 对齐标签
            lookup = {(fid, round(float(ws), 3)): int(lb)
                      for fid, lb, ws in zip(X["flight_id"], X["label"], X["window_start_s"])}
            y_seq = np.array([lookup.get(k, 0) for k in keys], dtype="float32")
            m3 = fit_m3(X_seq, y_seq, seq_len=WINDOW_SIZE, seed=SEED)
            p3 = predict_m3(m3, X_seq)
            frame = pd.DataFrame({"flight_id": [k[0] for k in keys],
                                  "window_start_s": [k[1] for k in keys],
                                  "label": y_seq.astype(int)})
            frame = frame.merge(X[["flight_id", "window_start_s", "split"]], on=["flight_id", "window_start_s"], how="left")
            frame["prob_1"] = p3
            frame["pred"] = (p3 >= 0.5).astype(int)
            _finalize("m3", frame, m3)
            print(f"     训练耗时 {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[warn] M3 跳过: {e}")

    if not preds:
        print("[error] 没有可用的模型")
        return 1

    metrics = evaluate_all(preds)
    save_metrics(metrics, OUTPUT_DIR / "metrics.csv")
    print("\n=== 评测结果 (test) ===")
    print(pd.DataFrame(metrics).T.round(3).to_string())
    plot_all(OUTPUT_DIR, preds, models)
    print(f"[done] metrics.csv 与图表已写入 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
