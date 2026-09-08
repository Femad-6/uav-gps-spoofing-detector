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

# from src.config import OUTPUT_DIR


def best_threshold(df: pd.DataFrame) -> float:
    """在（训练）预测上滑阈值搜索最大化 F1 的 T。"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.05):
        pred = (df["prob_1"] >= t).astype(int)
        f1 = f1_score(df["label"], pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def evaluate_all(preds: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """preds: {model: 含 split/label/prob_1/pred/flight_id/window_start_s 列的预测帧}。

    返回 {model: {AUROC, AUPRC, F1, precision, recall, false_alarm_rate,
                 missed_flights_rate, delay_mean_s, delay_median_s, threshold}}
    """
    results = {}
    for name, p in preds.items():
        tr = p[p["split"] == "train"] if "split" in p.columns else p
        te = p[p["split"] == "test"] if "split" in p.columns else p
        T = best_threshold(tr)
        y, s = te["label"].to_numpy(), te["prob_1"].to_numpy()
        pred = (s >= T).astype(int)

        res = {"threshold": T}
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
