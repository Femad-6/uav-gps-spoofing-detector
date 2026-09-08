"""FastAPI 在线检测服务。

GET  /healthz              存活探测
POST /predict              body: {flight_id, rows:[规范化行...]} -> 流式推理 JSON
POST /predict_file         multipart CSV -> 流式推理 JSON

启动: uvicorn src.server:app --port 8000
模型路径: 环境变量 SPOOFING_MODEL_PATH（默认 outputs/models/m2.joblib），
模型未就绪时 /predict 返回 503（服务本身可启动）。
"""
from __future__ import annotations

import io
import os
import time
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import ROOT
from src.data_loader import load_flight
from src.stream import CausalStreamer, model_predictor

DEFAULT_MODEL_PATH = ROOT / "outputs" / "models" / "m2.joblib"


class PredictRequest(BaseModel):
    flight_id: str = ""
    rows: List[dict]  # 规范化行（来源: load_flight 输出或 20Hz 同名列）


def create_app(model_path: str | os.PathLike | None = None) -> FastAPI:
    model_path = Path(model_path or os.environ.get("SPOOFING_MODEL_PATH", DEFAULT_MODEL_PATH))
    app = FastAPI(title="UAV GPS Spoofing Detector", version="1.0")
    app.state.model_path = model_path

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    def _infer(flight_id: str, df: pd.DataFrame) -> dict:
        streamer = CausalStreamer(model_predictor(app.state.model))
        t0 = time.time()
        results = []
        for row in df.to_dict("records"):
            r = streamer.feed(row)
            if r:
                results.append(r)
        probs = [r["prob"] for r in results]
        return {
            "flight": flight_id,
            "spoofing": bool(any(r["alert"] for r in results)),
            "confidence": float(max(probs)) if probs else 0.0,
            "alarm_intervals": streamer.alarm_intervals(),
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
        df = pd.DataFrame(req.rows)
        if len(df) < 40:
            raise HTTPException(400, "rows 至少 40 个样本（20Hz 下 2 秒）")
        return _infer(req.flight_id, df)

    @app.post("/predict_file")
    async def predict_file(file: UploadFile = File(...)):
        _require_model()
        content = await file.read()
        try:
            df = load_flight(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"CSV 解析失败: {e}")
        return _infer(file.filename or "upload", df)

    return app


app = create_app()
