"""全链路：训练 M0-M4 → 保存模型 → 评测 → 指标表 + 图表。

用法: python scripts/run_pipeline.py [--models m0,m1,m2,m3,m4] [--skip-m3]
前置: data_processed/features_windowed.parquet（由 scripts/make_dataset.py 生成）
输出: outputs/models/<name>.joblib, outputs/metrics.csv, outputs/figs/*.png
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 允许 import src
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

ALL_MODELS = ["m0", "m1", "m1b", "m2", "m3", "m4"]


def _m0_preds(detector: ThresholdDetector, X: pd.DataFrame) -> pd.DataFrame:
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
    tr = X[X["split"] == "train"]
    val = X[X["split"] == "val"]
    te = X[X["split"] == "test"]
    print(f"[data] 窗数 {len(X)} (train {len(tr)} / val {len(val)} / test {len(te)})")

    preds, models = {}, {}

    def _finalize(name: str, frame: pd.DataFrame, model) -> None:
        frame = frame.reset_index(drop=True)
        preds[name] = frame
        models[name] = model
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"[ok] {name} 训练并保存 -> {MODEL_DIR / (name + '.joblib')}")

    if "m0" in wanted:
        t0 = time.time()
        det = ThresholdDetector().fit(tr[tr["label"] == 0])   # 阈值从正常段统计得到
        _finalize("m0", _m0_preds(det, X), det)
        print(f"     训练耗时 {time.time() - t0:.1f}s")

    if "m1" in wanted:
        t0 = time.time()
        m1 = fit_m1(tr, tr["label"])
        _finalize("m1", predict_proba_all(m1, X), m1)
        print(f"     训练耗时 {time.time() - t0:.1f}s")

    if "m1b" in wanted:   # 消融: RF 全特征（检验 L2 特征对 RF 的影响）
        t0 = time.time()
        m1b = fit_m1(tr, tr["label"], features=None)
        _finalize("m1b", predict_proba_all(m1b, X), m1b)
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
            # 与窗级数据集按 (flight_id, window_start_s) 对齐标签和航班划分。
            lookup = {(fid, round(float(ws), 3)): int(lb)
                      for fid, lb, ws in zip(X["flight_id"], X["label"], X["window_start_s"])}
            split_lookup = {(fid, round(float(ws), 3)): sp
                            for fid, sp, ws in zip(X["flight_id"], X["split"], X["window_start_s"])}
            keep = [i for i, key in enumerate(keys) if key in lookup and key in split_lookup]
            X_seq = X_seq[keep]
            keys = [keys[i] for i in keep]
            y_seq = np.array([lookup[k] for k in keys], dtype="float32")
            split_seq = np.array([split_lookup[k] for k in keys])
            train_idx = np.where(split_seq == "train")[0]
            val_idx = np.where(split_seq == "val")[0]
            test_idx = np.where(split_seq == "test")[0]
            if not len(train_idx) or not len(val_idx) or not len(test_idx):
                raise ValueError("M3 需要同时存在训练、验证与测试航班")
            # 仅在训练集内分层下采样以控制 GPU 显存；测试集绝不参与拟合。
            if len(train_idx) > 20000:
                rng = np.random.default_rng(SEED)
                train_labels = y_seq[train_idx]
                train_idx = np.concatenate([
                    rng.choice(train_idx[train_labels == 1], min(10000, (train_labels == 1).sum()), replace=False),
                    rng.choice(train_idx[train_labels == 0], min(10000, (train_labels == 0).sum()), replace=False),
                ])
                rng.shuffle(train_idx)
            m3 = fit_m3(X_seq[train_idx].astype("float32"), y_seq[train_idx],
                        seq_len=WINDOW_SIZE, seed=SEED)
            # 验证窗用于阈值选择，测试窗仅在模型冻结后预测。
            eval_idx = np.concatenate([train_idx, val_idx, test_idx])
            p3 = predict_m3(m3, X_seq[eval_idx].astype("float32"))
            frame = pd.DataFrame({"flight_id": [keys[i][0] for i in eval_idx],
                                  "window_start_s": [keys[i][1] for i in eval_idx],
                                  "label": y_seq[eval_idx].astype(int),
                                  "split": split_seq[eval_idx],
                                  "prob_1": p3})
            frame["pred"] = (p3 >= 0.5).astype(int)
            _finalize("m3", frame, m3)
            print(f"     训练耗时 {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[warn] M3 跳过: {e}")

    if not preds:
        print("[error] 没有可用的模型")
        return 1

    metrics = evaluate_all(preds)
    for name, model in models.items():
        model.decision_threshold_ = float(metrics[name]["threshold"])
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    save_metrics(metrics, OUTPUT_DIR / "metrics.csv")
    print("\n=== 评测结果 (test) ===")
    print(pd.DataFrame(metrics).T.round(3).to_string())
    plot_all(OUTPUT_DIR, preds, models)
    print(f"[done] metrics.csv 与图表已写入 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
