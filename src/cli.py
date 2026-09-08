"""CLI 在线检测：python -m src.cli predict --input flight.csv [--model M.joblib]

输出 JSON: {flight, spoofing, confidence, alarm_intervals, n_windows, runtime_s}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib

from src.data_loader import load_flight
from src.stream import CausalStreamer, model_predictor


def main() -> int:
    ap = argparse.ArgumentParser(description="UAV GPS 诱骗流式检测 CLI")
    ap.add_argument("--input", required=True, help="飞行日志 CSV")
    ap.add_argument("--model", default="outputs/models/m2.joblib", help="模型文件 (joblib)")
    ap.add_argument("--threshold", type=float, default=0.5, help="报警阈值")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(json.dumps({"error": f"模型文件不存在: {model_path}"}, ensure_ascii=False), file=sys.stderr)
        return 1
    model = joblib.load(model_path)

    t0 = time.time()
    df = load_flight(args.input)
    if len(df) < 20:
        print(json.dumps({"error": "飞行数据过短或解析失败"}, ensure_ascii=False), file=sys.stderr)
        return 1

    streamer = CausalStreamer(model_predictor(model), threshold=args.threshold)
    results = []
    for row in df.to_dict("records"):
        r = streamer.feed(row)
        if r:
            results.append(r)
    runtime_s = time.time() - t0

    probs = [r["prob"] for r in results]
    out = {
        "flight": str(Path(args.input).name),
        "spoofing": bool(any(r["alert"] for r in results)),
        "confidence": float(max(probs)) if probs else 0.0,
        "alarm_intervals": streamer.alarm_intervals(),
        "n_windows": len(results),
        "runtime_s": round(runtime_s, 3),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
