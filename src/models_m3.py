"""M3: BiLSTM 时序模型（可选 GPU）。

直接以每窗 40×9 的原始传感器序列（速度三轴/加速度/角速度）为输入，
对比"特征工程 vs 表征学习"。torch 未安装时 fit_m3 抛 RuntimeError，
由上层流程捕获并标记取消（见 run_pipeline）。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import SEED

# 原始序列输入列（自规范化列名）
RAW_COLUMNS = ["vel_e", "vel_n", "vel_u", "ax", "ay", "az", "gx", "gy", "gz"]


class SpoofLSTM(nn.Module):
    """Linear -> BiLSTM -> Linear+sigmoid。"""

    def __init__(self, input_dim: int = 9, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        out, _ = self.lstm(h)
        return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)


def fit_m3(X_seq: np.ndarray, y_seq: np.ndarray, seq_len: int = 40,
           seed: int = SEED, device: str | None = None, epochs: int = 20):
    """X_seq: (n, seq_len, n_raw)；y_seq: (n,)。返回训练好的 SpoofLSTM（含标准化参数）。"""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch unavailable, M3 skipped") from e
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 逐特征标准化（特征量纲差异大，避免 sigmoid 饱和）
    flat = X_seq.reshape(-1, X_seq.shape[-1])
    mean = flat.mean(0).astype("float32")
    std = flat.std(0).astype("float32") + 1e-6
    x_np = ((X_seq - mean) / std).astype("float32")

    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_seq, dtype=torch.float32, device=device)
    n = x.shape[0]
    idx = np.random.permutation(n)
    n_val = max(1, n // 5)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = SpoofLSTM(input_dim=x.shape[-1], hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    best_loss, best_state, patience = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for b in range(0, len(tr_idx), 64):
            b_idx = torch.as_tensor(tr_idx[b:b + 64], device=device)
            opt.zero_grad()
            loss = loss_fn(model(x[b_idx]), y[b_idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = loss_fn(model(x[val_idx]), y[val_idx]).item()
        if v_loss < best_loss - 1e-5:
            best_loss, patience = v_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.norm_mean_ = torch.as_tensor(mean)
    model.norm_std_ = torch.as_tensor(std)
    return model


def predict_m3(model: SpoofLSTM, X_seq: np.ndarray, device: str | None = None) -> np.ndarray:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        mean = model.norm_mean_.to(device)
        std = model.norm_std_.to(device)
        x = torch.as_tensor(X_seq, dtype=torch.float32, device=device)
        x = (x - mean) / std
        return model(x).cpu().numpy()
