"""FastAPI 在线检测服务。

GET  /healthz              存活探测
POST /predict              body: {flight_id, rows:[规范化行...]} -> 流式推理 JSON
POST /predict_file         multipart CSV -> 流式推理 JSON

启动: uvicorn src.server:app --port 8000
模型路径: 环境变量 SPOOFING_MODEL_PATH（默认 outputs/models/m1.joblib），
模型未就绪时 /predict 返回 503（服务本身可启动）。
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import NOMINAL_HZ, ROOT
from src.data_loader import NORMALIZED_COLUMNS, load_flight
from src.stream import CausalStreamer, model_predictor

DEFAULT_MODEL_PATH = ROOT / "outputs" / "models" / "m1.joblib"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS = 200_000


class PredictRequest(BaseModel):
    flight_id: str = ""
    rows: List[dict]  # 规范化行（来源: load_flight 输出或 20Hz 同名列）


def normalize_rows(rows: List[dict]) -> pd.DataFrame:
    """校验、排序并重采样 API 的规范化行，保证流式窗口始终表示 2 秒。"""
    raw = pd.DataFrame(rows)
    if "time_s" not in raw:
        raise ValueError("rows 必须包含 time_s（秒）")
    df = raw.reindex(columns=NORMALIZED_COLUMNS).copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.dropna(subset=["time_s"]).sort_values("time_s").drop_duplicates("time_s")
    if len(df) < 3:
        raise ValueError("有效 time_s 样本不足 3 个")
    grid = np.arange(df["time_s"].iloc[0], df["time_s"].iloc[-1], 1.0 / NOMINAL_HZ)
    out = pd.DataFrame({"time_s": grid})
    for col in NORMALIZED_COLUMNS:
        if col in ("time_s", "attack"):
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(values)
        out[col] = np.interp(grid, df["time_s"].to_numpy()[ok], values[ok]) if ok.any() else np.nan
    out["attack"] = 0.0
    return out.reindex(columns=NORMALIZED_COLUMNS)


def create_app(model_path: str | os.PathLike | None = None,
               max_upload_bytes: int = MAX_UPLOAD_BYTES) -> FastAPI:
    model_path = Path(model_path or os.environ.get("SPOOFING_MODEL_PATH", DEFAULT_MODEL_PATH))
    app = FastAPI(title="UAV GPS Spoofing Detector", version="1.0")
    app.state.model_path = model_path

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    def _infer(flight_id: str, df: pd.DataFrame) -> dict:
        threshold = float(getattr(app.state.model, "decision_threshold_", 0.5))
        streamer = CausalStreamer(model_predictor(app.state.model), threshold=threshold)
        t0 = time.time()
        results = []
        for row in df.to_dict("records"):
            r = streamer.feed(row)
            if r:
                results.append(r)
        probs = [r["prob"] for r in results]
        return {
            "flight": flight_id,
            "spoofing": bool(any(r["confirmed_alert"] for r in results)),
            "threshold": threshold,
            "confidence": float(max(probs)) if probs else 0.0,
            "alarm_intervals": streamer.alarm_intervals(),
            "raw_alert_windows": sum(int(r["alert"]) for r in results),
            "confirmed_alert_windows": sum(
                int(r["confirmed_alert"]) for r in results),
            "n_windows": len(results),
            "runtime_s": round(time.time() - t0, 3),
        }

    def _require_model():
        model = getattr(app.state, "model", None)
        if model is None:
            if not app.state.model_path.exists():
                raise HTTPException(503, f"模型未就绪: {app.state.model_path}（先运行 run_pipeline.py 训练）")
            app.state.model = joblib.load(app.state.model_path)
        return app.state.model

    @app.post("/predict")
    def predict(req: PredictRequest):
        _require_model()
        if len(req.rows) > MAX_ROWS:
            raise HTTPException(413, f"rows 超过上限 {MAX_ROWS}")
        try:
            df = normalize_rows(req.rows)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if len(df) < 40:
            raise HTTPException(400, "rows 至少 40 个样本（20Hz 下 2 秒）")
        return _infer(req.flight_id, df)

    @app.post("/predict_file")
    async def predict_file(file: UploadFile = File(...)):
        _require_model()
        content = await file.read(max_upload_bytes + 1)
        if len(content) > max_upload_bytes:
            raise HTTPException(413, f"上传文件超过 {max_upload_bytes} 字节上限")
        try:
            df = load_flight(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"CSV 解析失败: {e}")
        return _infer(file.filename or "upload", df)

    return app


app = create_app()
