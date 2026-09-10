"""离线评测：AUROC/AUPRC/F1/漏检率/误报率/检测延迟 + 图表输出。"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.alarm import TemporalAlarmPolicy

# from src.config import OUTPUT_DIR


def best_threshold(df: pd.DataFrame) -> float:
    """在验证预测上滑阈值搜索最大化 F1 的 T。"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.05):
        pred = (df["prob_1"] >= t).astype(int)
        f1 = f1_score(df["label"], pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def select_operating_threshold(df: pd.DataFrame,
                               max_false_alarm_rate: float = 0.10) -> tuple[float, float]:
    """在验证集误报率约束下最大化召回，并用 F1/更高阈值依次打破平局。"""
    if not 0.0 <= max_false_alarm_rate <= 1.0:
        raise ValueError("max_false_alarm_rate must be between 0 and 1")
    labels = df["label"].to_numpy(dtype=int)
    scores = df["prob_1"].to_numpy(dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("validation data must contain normal and attacked samples")
    candidates = np.unique(scores)
    candidates = np.r_[candidates, np.nextafter(float(scores.max()), np.inf)]
    feasible = []
    normal = labels == 0
    for threshold in candidates:
        pred = scores >= threshold
        far = float(np.mean(pred[normal]))
        if far <= max_false_alarm_rate + 1e-12:
            recall = float(recall_score(labels, pred, zero_division=0))
            f1 = float(f1_score(labels, pred, zero_division=0))
            feasible.append((recall, f1, float(threshold), far))
    _, _, threshold, far = max(feasible, key=lambda row: row[:3])
    return threshold, far


def evaluate_all(preds: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """preds: {model: 含 split/label/prob_1/pred/flight_id/window_start_s 列的预测帧}。

    返回 {model: {AUROC, AUPRC, F1, precision, recall, false_alarm_rate,
                 missed_flights_rate, delay_mean_s, delay_median_s, threshold}}
    """
    results = {}
    for name, p in preds.items():
        if "split" not in p.columns:
            raise ValueError(f"{name} predictions must preserve train/val/test split")
        val = p[p["split"] == "val"]
        te = p[p["split"] == "test"]
        if val.empty or te.empty:
            raise ValueError(f"{name} predictions require non-empty val and test splits")
        f1_threshold = best_threshold(val)
        T, val_far = select_operating_threshold(val)
        y, s = te["label"].to_numpy(), te["prob_1"].to_numpy()
        pred = (s >= T).astype(int)

        res = {"threshold": T, "threshold_source": "val",
               "threshold_policy": "val_far<=0.10_max_recall",
               "f1_threshold": f1_threshold,
               "val_false_alarm_rate": val_far}
        if len(np.unique(y)) > 1:
            res["AUROC"] = float(roc_auc_score(y, s))
            res["AUPRC"] = float(average_precision_score(y, s))
            res["F1"] = float(f1_score(y, pred, zero_division=0))
            res["precision"] = float(precision_score(y, pred, zero_division=0))
            res["recall"] = float(recall_score(y, pred, zero_division=0))
        else:
            res.update(AUROC=np.nan, AUPRC=np.nan, F1=np.nan,
                       precision=float(precision_score(y, pred, zero_division=0))
                       if len(y) else np.nan, recall=np.nan)

        # ---- 航班级：漏检率 / 检测延迟 / 窗口级误报率 ----
        te_df = pd.DataFrame({
            "flight_id": te["flight_id"], "label": te["label"],
            "pred": pred, "window_start_s": te["window_start_s"],
        })
        delays, missed = [], 0
        attacked, fa_num, fa_den = 0, 0, 0
        for fid, g in te_df.groupby("flight_id"):
            attacked_w = g[g["label"] == 1]
            if len(attacked_w):
                attacked += 1
                attack_start = attacked_w["window_start_s"].min()
                alerts = attacked_w[attacked_w["pred"] == 1]["window_start_s"]
                if len(alerts):
                    delays.append(alerts.min() - attack_start)
                else:
                    missed += 1
            fa_num += int(((g["pred"] == 1) & (g["label"] == 0)).sum())
            fa_den += int((g["label"] == 0).sum())
        res["missed_flights_rate"] = float(missed / attacked) if attacked else np.nan
        res["false_alarm_rate"] = float(fa_num / fa_den) if fa_den else np.nan
        res["delay_mean_s"] = float(np.mean(delays)) if delays else np.nan
        res["delay_median_s"] = float(np.median(delays)) if delays else np.nan

        # 与在线服务相同的 3-of-5 + 迟滞策略，且每个航班重置状态。
        confirmed = pd.Series(0, index=te.index, dtype=int)
        for _, group in te.groupby("flight_id"):
            policy = TemporalAlarmPolicy(T)
            ordered = group.sort_values("window_start_s")
            decisions = [int(policy.update(score).confirmed_alert)
                         for score in ordered["prob_1"].to_numpy(dtype=float)]
            confirmed.loc[ordered.index] = decisions
        res["confirmed_F1"] = float(f1_score(y, confirmed.to_numpy(), zero_division=0))
        res["confirmed_precision"] = float(
            precision_score(y, confirmed.to_numpy(), zero_division=0))
        res["confirmed_recall"] = float(
            recall_score(y, confirmed.to_numpy(), zero_division=0))

        confirmed_df = te_df.copy()
        confirmed_df["pred"] = confirmed.to_numpy()
        confirmed_delays, confirmed_missed = [], 0
        confirmed_attacked, confirmed_fa_num, confirmed_fa_den = 0, 0, 0
        for _, group in confirmed_df.groupby("flight_id"):
            attacked_w = group[group["label"] == 1]
            if len(attacked_w):
                confirmed_attacked += 1
                attack_start = attacked_w["window_start_s"].min()
                alerts = attacked_w[attacked_w["pred"] == 1]["window_start_s"]
                if len(alerts):
                    confirmed_delays.append(alerts.min() - attack_start)
                else:
                    confirmed_missed += 1
            confirmed_fa_num += int(
                ((group["pred"] == 1) & (group["label"] == 0)).sum())
            confirmed_fa_den += int((group["label"] == 0).sum())
        res["confirmed_missed_flights_rate"] = (
            float(confirmed_missed / confirmed_attacked)
            if confirmed_attacked else np.nan)
        res["confirmed_false_alarm_rate"] = (
            float(confirmed_fa_num / confirmed_fa_den)
            if confirmed_fa_den else np.nan)
        res["confirmed_delay_mean_s"] = (
            float(np.mean(confirmed_delays)) if confirmed_delays else np.nan)
        res["confirmed_delay_median_s"] = (
            float(np.median(confirmed_delays)) if confirmed_delays else np.nan)
        results[name] = res
    return results


def save_metrics(metrics: Dict[str, dict], path) -> None:
    pd.DataFrame(metrics).T.to_csv(path)


def plot_all(out_dir, preds: Dict[str, pd.DataFrame],
             models: Optional[Dict[str, object]] = None) -> None:
    """输出 ROC/PR/特征重要度/检测时间线图到 out_dir。"""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    # ROC / PR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in preds.items():
        te = p[p["split"] == "test"] if "split" in p.columns else p
        y, s = te["label"], te["prob_1"]
        if y.nunique() > 1:
            fpr, tpr, _ = roc_curve(y, s)
            axes[0].plot(fpr, tpr, label=name)
            pr, rc, _ = precision_recall_curve(y, s)
            axes[1].plot(rc, pr, label=name)
    axes[0].set(title="ROC curve (test)", xlabel="FPR", ylabel="TPR")
    axes[1].set(title="PR curve (test)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr.png", dpi=150)
    plt.close(fig)

    # 特征重要度（若模型提供）
    for name, model in (models or {}).items():
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            continue
        names = getattr(model, "feat_names_", None)
        if names is None:
            continue
        fig, ax = plt.subplots(figsize=(9, 0.3 * len(imp)))
        order = np.argsort(imp)[::-1][:25]
        ax.barh([names[i] for i in order][::-1], imp[order][::-1])
        ax.set(title=f"Feature importance ({name})", xlabel="Gain")
        fig.tight_layout()
        fig.savefig(fig_dir / f"feature_importance_{name}.png", dpi=150)
        plt.close(fig)

    # 时间线叠加图（每个攻击测试航班取 1 例，最多 4 例）
    for name, p in preds.items():
        te = p[p["split"] == "test"] if "split" in p.columns else p
        attacked = (te["label"] == 1).groupby(te["flight_id"]).any()
        fids = attacked[attacked].index.tolist()[:4]
        if not fids:
            continue
        fig, axes = plt.subplots(len(fids), 1, figsize=(11, 2.4 * len(fids)), squeeze=False)
        for ax, fid in zip(axes[:, 0], fids):
            g = te[te["flight_id"] == fid].sort_values("window_start_s")
            ax.plot(g["window_start_s"], g["label"], lw=1.2, label="ground truth")
            ax.plot(g["window_start_s"], g["pred"], lw=1.2, label="detector", alpha=0.8)
            ax.fill_between(g["window_start_s"], 0, g["prob_1"], alpha=0.2, label="prob_1")
            ax.set(title=f"{name} / {fid}", ylabel="label / prob", ylim=(-0.05, 1.05))
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"timeline_{name}.png", dpi=150)
        plt.close(fig)
    print(f"[eval] 图表已保存: {fig_dir}")
