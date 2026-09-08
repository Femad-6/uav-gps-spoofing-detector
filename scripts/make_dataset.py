"""一键数据管线：raw CSV → 规范化 → 滑窗特征 → 标签 → 训练/测试划分 → parquet。

用法: python scripts/make_dataset.py [--test-frac 0.2]
输出: data_processed/features_windowed.parquet、configs/split.json
"""
import argparse
import sys

from src.config import DATA_DIR, PROCESSED_DIR
from src.data_loader import load_all
from src.features import make_features
from src.labels import assign_split, build_labels, load_split, save_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    print(f"[1/4] 加载数据: {DATA_DIR}")
    flights = load_all(DATA_DIR)
    if not flights:
        print("[error] 未找到飞行日志，请先运行 scripts/download_data.py")
        return 1
    print(f"      共 {len(flights)} 个航班, 例: {sorted(flights)[:3]}")

    print("[2/4] 构建标签（攻击区间）")
    labels = build_labels(flights)
    n_attacked = sum(1 for v in labels.values() if v)
    print(f"      有攻击区间航班: {n_attacked}/{len(flights)}")

    print("[3/4] 滑窗特征化")
    X = make_features(flights, labels)
    print(f"      窗数: {len(X)}, 特征列: {X.shape[1] - 3}, 正样本率: {X['label'].mean():.3f}")

    print("[4/4] 训练/测试划分")
    split = load_split()
    if not split:
        split = assign_split(sorted(X["flight_id"].unique()), test_frac=args.test_frac)
        save_split(split)
    X["split"] = X["flight_id"].map(split)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features_windowed.parquet"
    X.to_parquet(out, index=False)
    print(f"[done] 保存: {out} (train {len(X[X.split == 'train'])}, test {len(X[X.split == 'test'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
