# 无人机 GPS 诱骗检测项目 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 3 天内完成一个可复现的无人机 GPS 诱骗检测项目：数据理解 → 特征工程 → 4 类方法对比 → 因果流式在线检测服务 → 一键流程 → 实验报告。

**Architecture:** 通用管线：CSV 飞行日志 → schema 自动检测 + 人工映射表 → 规范化 DataFrame（20Hz 网格）→ L1/L2/L3 滑窗特征 → X/标签/航班划分 → M0-M4 模型训练与离线评测（AUROC/AUPRC/F1/漏检/误报/检测延迟）→ 同一特征管线驱动 CLI 与 FastAPI 因果流式推理。

**Tech Stack:** Python 3.10+；pandas、numpy、scikit-learn、matplotlib/seaborn、pyarrow；torch（有 GPU，备用，M3 用）；fastapi、uvicorn、pytest；pydantic。

**Spec:** docs/superpowers/specs/2026-09-08-drone-gps-spoofing-detection-design.md

## Global Constraints

- 项目根目录：`d:\保研\西工-任务`（已是工作目录）。所有路径以项目根为相对根。
- 固定随机种子：`SEED = 42`（np.random.seed、random.seed、torch.manual_seed、sklearn 各 estimator 的 random_state）。
- 滑窗参数：窗长 2s = 40 样本 @ 数据集重采样 20Hz；步长 0.5s = 10 样本。
- 依赖最少化：不引入 lightgbm/xgboost，M2 默认用 sklearn `HistGradientBoostingClassifier`；torch 仅 Task 8 引入且失败不阻塞（内置 fallback 报告说明）。
- 按航班划分 train/test：test 至少留 20% 航班，`test_flights` 由 config 显式列出，禁止按时间无间隙切窗口。
- 所有下游输出写入 `outputs/`；图表用 matplotlib（中文字体不可用，图表一律英文标注）。
- README/报告用中文；代码注释用中文或英文均可，与所在文件一致。
- 每个 Task 结束必须 commit（信息格式：`feat:`/`chore:`/`docs:`）。
- 提交物不得包含 data/ 原始数据文件于 git（加入 .gitignore），数据下载依赖 `scripts/download_data.py` 复现。

---

### Task 1: 环境检查、项目骨架与 git 初始化

**Files:**
- Create: `requirements.txt`、`README.md`（骨架）、`.gitignore`、`src/__init__.py`、`src/config.py`、`scripts/download_data.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `src/config.py` 暴露 `SEED=42`、`WINDOW_SECONDS=2.0`、`STRIDE_SECONDS=0.5`、`NOMINAL_HZ=20.0`、`DATA_DIR`、`OUTPUT_DIR`、以及 `require_env()` 函数（校验 GPU/依赖）。后续所有任务 import `src.config`。

- [ ] **Step 1: 环境探测**
```bash
python --version && python -c "import sys; print(sys.executable)"
nvidia-smi   # 若失败记录"无 GPU 或驱动"，不影响后续
```
记录结果到 `docs/environment.md`（含操作系统、Python 版本、可用显存），供报告引用。

- [ ] **Step 2: 安装依赖**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow fastapi uvicorn pytest requests
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 失败则降级 pip install torch
```

- [ ] **Step 3: 创建骨架文件**（requirements.txt 固定版本：`pandas>=2.0`、`numpy>=1.24`、`scikit-learn>=1.3`、`matplotlib>=3.7`、`seaborn>=0.13`、`fastapi>=0.104`、`uvicorn>=0.24`、`pyarrow>=14.0`、`pytest>=7.4`、`torch>=2.1`）

- [ ] **Step 4: 写配置测试**
```python
# tests/test_config.py
from src.config import SEED, WINDOW_SECONDS, STRIDE_SECONDS, NOMINAL_HZ

def test_constants():
    assert SEED == 42
    assert WINDOW_SECONDS == 2.0
    assert STRIDE_SECONDS == 0.5
    assert NOMINAL_HZ == 20.0
```
- [ ] **Step 5: 运行测试** `pytest tests/test_config.py -v` → PASS
- [ ] **Step 6: `git init` + 首次 commit**（`chore: init project skeleton`）

---

### Task 2: 数据下载与 Schema 探测（Discovery）

**Files:**
- Create: `scripts/download_data.py`（git clone 或下载 ZIP 到 `data/raw/`，失败时尝试 https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset/archive/refs/heads/main.zip）、`scripts/inspect_data.py`（输出 `outputs/eda/data_inventory.md`：文件数、每文件列名、行数、时间跨度、样本率；并画出正常/攻击各一例的传感波形到 `outputs/eda/`）
- Create: `notebooks/EDA.ipynb`（SKIP——用 **可复现脚本** `scripts/inspect_data.py` 代替 notebook，同样满足"数据理解"且可直接复现）

**Interfaces:**
- Produces: `data/raw/` 目录；`outputs/eda/data_inventory.md` 记录实际字段名与单位；`data/raw --files清单`。**Task 3 的映射表依赖此产物**（实现者先读 data_inventory.md）。

- [ ] **Step 1: 运行下载脚本** `python scripts/download_data.py` → 验证 `data/raw/` 有 CSV/飞行日志文件
- [ ] **Step 2: 运行 inspect_data.py** → 生成 inventory + 波形图；人工阅读并**在 data_inventory.md 中记录：a) 正常/攻击如何区分（文件名 or 目录 or 时间戳标记）b) 攻击区间是否有显式标注 c) 列名→物理量映射**（这是后续所有任务的输入事实）
- [ ] **Step 3: 将发现的标签信息记录** 到 `docs/dataset-notes.md` 并 commit（`docs: record dataset schema and labels`）

---

### Task 3: 规范化数据加载器

**Files:**
- Create: `src/data_loader.py`
- Test: `tests/data/load_fixture.csv`（合成小文件）+ `tests/test_data_loader.py`

**Interfaces:**
- Consumes: Task 2 的 data_inventory.md（列名映射）。
- Produces:
```python
def load_flight(path: str) -> pd.DataFrame  # 返回规范化 DataFrame
# 规范化列名（None=该列数据不存在，按列缺失处理）：
# time_s float,  lat float, lon float, alt_gps float, alt_baro float,
# vel_e/vel_n/vel_u float（米/秒）, acc_body (ax, ay, az) float,
# gyro (gx, gy, gz) rad/s, roll/pitch/yaw float（弧度, 或 None）, mag (mx, my, mz)
# 规范: 重采样到 20Hz 网格（前后填充按序: ffill+bfill），统一单位米/秒。
def load_all(data_dir: str) -> Dict[str, pd.DataFrame]  # {flight_id: df}
def infer_schema(columns) -> Dict[str, str | None]  # 由列名模式自动推断, 未识别列返回 None
```
- 飞行 ID 规则：以文件名（去扩展名）作为 flight_id。

- [ ] **Step 1: 写失败测试**（fixture 含 PX4 常见列名 `timestamp/lat/lon/alt/landed_vx/landed_vy/landed_vz/baro_alt/roll/pitch/yaw` 与记录时间戳）
```python
# tests/test_data_loader.py
import pandas as pd
from src.data_loader import load_flight, infer_schema

def test_load_flight_normalized_units():
    df = load_flight("tests/data/load_fixture.csv")
    assert {"time_s", "lat", "lon", "alt_gps", "vel_e", "vel_n", "vel_u"} <= set(df.columns)
    assert df["time_s"].monotonic_increasing

def test_infer_schema_maps_common_names(tmp_path):
    schema = infer_schema(["timestamp", "lat", "lon", "alt", "vx", "vy", "vz", "baro_alt"])
    assert schema["time_s"] == "timestamp"
    assert schema["alt_baro"] == "baro_alt"
```
- [ ] **Step 2: 运行 → FAIL**（module not defined）
- [ ] **Step 3: 实现**（正则模式：`^t.*time|timestamp` → time_s；`lat`/`lon` 直配；`alt`→alt_gps、`baro|pressure.*alt`→alt_baro；`vx|vel.*e|vel_n`→对应速度；`ax/ay/az|accel`→acc_body；attitude 单位若为度需转弧度——在 mapping 中加 unit 标志）
- [ ] **Step 4: 运行 → PASS**
- [ ] **Step 5: 用真实数据联调** `python -c "from src.data_loader import load_all; dfs=load_all('data/raw'); print({k: v.shape for k,v in list(dfs.items())[:5]})"` → 打印每个航班规范化后的形状与缺失列（若字段没匹配上，回到 data_inventory.md 调整映射表）
- [ ] **Step 6: Commit**（`feat: add normalized flight loader with schema inference`）

---

### Task 4: 特征工程（L1/L2/L3）

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: Task 3 `load_flight` 的规范化 DataFrame。
- Produces:
```python
def window_extract(df: pd.DataFrame) -> pd.DataFrame
# 输入规范化 df（单航班，20Hz）。内部以 row 为中心滑窗：
# 窗长 40 样本、步长 10 样本（参数来自 config），
# 返回一行一窗，列含:
#   flight_id, window_start_s(float), 标签列 label(int, 0/1), 空窗不产出
#   f_<name>_<agg>: L1 物理量统计量（mean/std/min/max/slope/kurtosis）
#   c_<name>: L2 一致性特征（见下）
# L1 统计对象: alt_gps, alt_baro, (alt_gps-alt_baro), speed_horiz, vel_u,
#   acc_norm, gyro_norm, roll, pitch, yaw, |acc|变动率
# L2 一致性特征（窗内计算）:
#   c_imu_gps_acc  = mean(|acc_body| - |d(vel)/dt|)   （GPS 加速度模 vs IMU 加速度模）
#   c_gps_baro     = mean(alt_gps - alt_baro)
#   c_heading_vel  = 速度矢向角与 yaw 的窗内平均角度差（rad）
#   c_jump         = 相邻采样 GPS 位置跳跃量的 99 分位数（诱捕时微小步进/突变）
#   c_smoothness   = 1 / (mean(|加速度二阶差分|) + 1e-6)（平滑度异常，eps 防除零）
# 缺失列一律以 NaN 传播（后续模型按缺失策略处理）
def make_features(all_flights: Dict[str, pd.DataFrame], labels: Dict[str, List[Tuple[float,float]]]) -> pd.DataFrame
# labels: {flight_id: [(attack_start_s, attack_end_s), ...]}；无攻击区间记录则该航班 label=0；
# 一窗 label=1 当且仅当窗口区间与某攻击区间重叠比重 >= 0.5
```
- [ ] **Step 1: 写失败测试**（合成 df：构造一个"诱骗段"，窗内 GPS 高度缓坡偏离气压计）
```python
# tests/test_features.py
import numpy as np, pandas as pd
from src.features import window_extract, make_features

def _df():
    t = np.arange(0, 10.0, 0.05)  # 200 samples
    return pd.DataFrame({"time_s": t, "lat": 34.0 + 1e-5*t, "lon": 108.9,
        "alt_gps": 20 + 0.5*t, "alt_baro": 20 + 0.0*t,   # 诱骗段 GPS 爬升而气压计不动
        "vel_e": 0.1*np.ones_like(t), "vel_n": np.zeros_like(t), "vel_u": np.ones_like(t),
        "ax": np.zeros_like(t), "ay": np.zeros_like(t), "az": 9.81*np.ones_like(t),
        "gx": np.zeros_like(t), "gy": np.zeros_like(t), "gz": np.zeros_like(t),
        "roll": np.zeros_like(t), "pitch": np.zeros_like(t), "yaw": np.zeros_like(t)})

def test_window_shapes():
    df = _df()
    out = window_extract(df)
    assert {"flight_id", "window_start_s", "label"} <= set(out.columns)
    assert out.shape[0] == 17   # (200-40)/10 + 1 = 17 窗
    assert "c_gps_baro" in out.columns

def test_consistency_feature_catches_gps_baro_divergence():
    out = window_extract(_df())
    assert out["c_gps_baro"].mean() > 1.0   # GPS 高度平均高出 2m 以上

def test_make_features_labels_windows():
    df = _df(); flights = {"f1": df}
    labels = {"f1": [(3.0, 7.0)]}
    X = make_features(flights, labels)
    assert set(X["label"]) <= {0, 1}
    assert X.loc[X["window_start_s"].between(4, 6), "label"].eq(1).all()
```
- [ ] **Step 2: 运行 → FAIL**
- [ ] **Step 3: 实现**（注意窗内 label 用窗区间与攻击区间重叠比例；`c_gps_baro` 用 nanmean 处理缺失；输出列名统一 `c_`/`f_` 前缀）
- [ ] **Step 4: 运行 → PASS**
- [ ] **Step 5: 真实数据冒烟**：对 data/raw 全量跑 `make_features`，打印维度；若有 NaN 占比 >30% 的列，检查映射表。
- [ ] **Step 6: Commit**（`feat: add sliding-window features with physical-consistency layer`）

---

### Task 5: 标签与训练/测试划分

**Files:**
- Create: `src/labels.py`、`configs/split.json`、`scripts/make_dataset.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: Task 4 `make_features`；Task 2 的标签事实（航班正常/攻击划分）。
- Produces:
```python
def build_labels(dataset_notes: Dict) -> Dict[str, List[Tuple[float, float]]]
# 若数据有攻击区间标注：直接读取；否则整航班标注: 正常航班 -> []，攻击航班 -> [(0, 99999)]
def assign_split(flights: List[str], test_frac: float = 0.2, seed: int = 42) -> Dict[str, str]
# {flight_id: "train"/"test"}；用固定 seed 的 random.Random
def load_dataset(config_path: str = "configs/split.json") -> pd.DataFrame
# 输出全量窗级 X + 列 split 字段（train/test），且 test 航班不允许同时出现在 train
```
- [ ] **Step 1: 写失败测试**
```python
# tests/test_labels.py
from src.labels import assign_split

def test_assign_split_disjoint_and_reproducible():
    flights = [f"f{i}" for i in range(20)]
    s1 = assign_split(flights)
    assert set(s1.values()) <= {"train", "test"} and "test" in s1.values()
    assert assign_split(flights) == s1
    s2 = assign_split(flights, seed=7)
    assert s2 != s1
```
- [ ] **Step 2: FAIL → Step 3: 实现 → Step 4: PASS**
- [ ] **Step 5: 生成真实数据 X**：`python scripts/make_dataset.py` → 输出 `data_processed/features_windowed.parquet`（列含 flight_id/split/label 及全部特征），提交前在报告中说明行数
- [ ] **Step 6: Commit**（`feat: add label builder and flight-level split`）

---

### Task 6: M0 阈值规则 + M1 学习类 Baseline

**Files:**
- Create: `src/models.py`（含 `THRESHOLD_MODEL`、`fit_m1`）、`tests/test_models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: Task 5 的 `load_dataset`。
- Produces:
```python
class ThresholdDetector:  # M0
    def fit(self, X_train): ...    # 从训练窗 c_* 特征按均值+k*std 定阈值（k=默认3）
    def predict(self, window: pd.Series) -> bool
def fit_m1(X_tr, y_tr, model_type="random_forest", seed=42) -> "sklearn estimator"
def predict_proba_all(model, X) -> pd.DataFrame  # 列: prob_1, pred, flight_id, window_start_s, label
```
- [ ] **Step 1: 写失败测试**（训练/推断 + 确定性 + 形状断言）
```python
# tests/test_models.py
import numpy as np, pandas as pd
from src.models import ThresholdDetector, fit_m1, predict_proba_all

def test_threshold_detector_detects_divergent_window():
    detector = ThresholdDetector()
    X = pd.DataFrame({"c_gps_baro": [0.1, 0.2, 5.0, 6.0]})
    detector.fit(X[["c_gps_baro"]])
    assert detector.predict(pd.Series({"c_gps_baro": 5.9})) is True
    assert detector.predict(pd.Series({"c_gps_baro": 0.0})) is False

def test_fit_m1_is_deterministic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 8)))
    y = pd.Series(rng.integers(0, 2, 100))
    m1 = fit_m1(X, y, seed=42); m2 = fit_m1(X, y, seed=42)
    c1 = predict_proba_all(m1, X); c2 = predict_proba_all(m2, X)
    assert c1["prob_1"].tolist() == c2["prob_1"].tolist()
```
- [ ] **Step 2: FAIL → Step 3: 实现**（M0 阈值用训练窗 c_* 的 mean+3*std，任一特征越过阈值→报警，返回特征级证据；M1：`RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)`）
- [ ] **Step 4: PASS**
- [ ] **Step 5: 全量训练冒烟**
```bash
python -c "from src.models import fit_m1; from src.labels import load_dataset; d=load_dataset(); m=fit_m1(d[d.split=='train'], d[d.split=='train'].label); print('train ok:', m.__class__.__name__)"
```
- [ ] **Step 6: Commit**（`feat: add M0 threshold and M1 random-forest baselines`）

---

### Task 7: M2 改进模型（HistGradientBoosting on L1+L2+L3）

**Files:**
- Create: `src/models_m2.py`（或并入 src/models.py，推荐独立文件）、`tests/test_models_m2.py`

**Interfaces:**
- Consumes: Task 5 数据集、Task 6 的 `predict_proba_all`。
- Produces:
```python
def fit_m2(X_tr, y_tr, seed=42) -> HistGradientBoostingClassifier
# 自动丢弃全部为 NaN 的列，其余 NaN 内填（hist 支持 NaN 自动处理，无需额外插补）
```
- [ ] **Step 1: 写失败测试**（同 fit_m1 确定性 + 在含 NaN 的合成数据上不报错）
```python
def test_fit_m2_handles_nan_and_predicts_probabilities():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 8)))
    X.iloc[::3, 1] = np.nan
    m = fit_m2(X, pd.Series(np.random.default_rng(1).integers(0, 2, 100)))
    p = m.predict_proba(X)
    assert p.shape == (100, 2)
```
- [ ] **Step 2: FAIL → Step 3: 实现**（`HistGradientBoostingClassifier(max_iter=300, random_state=seed)`）
- [ ] **Step 4: PASS → Step 5: 真实数据跑通** `predict_proba_all` 输出前 500 行无异常
- [ ] **Step 6: Commit**（`feat: add M2 histogram-gradient-boosting improved model`）

---

### Task 8: M3 深度学习对比模型（BiLSTM, 可选 GPU）

**Files:**
- Create: `src/models_m3.py`、`tests/test_models_m3.py`
- Create（若 torch 装不上）：无 —— **失败路径**：`models_m3.py` 里抛 `RuntimeError("torch unavailable, M3 skipped")`，pipeline 捕获并记录"跳过 M3"，报告写明原因。此路径即测试对象。

**Interfaces:**
- Produces:
```python
def fit_m3(X_seq: np.ndarray, y_seq: np.ndarray, seq_len: int, seed=42, device="cuda" if available else "cpu") -> "nn.Module"
# 输入为滑动窗原始序列(flight, window, 40, num_raw_feats)，输出 P(attack)。
# 结构：Linear(input_dim→64) → BiLSTM(64×2, hidden=32, 1层) → 全连接 → sigmoid
# 训练：Adam lr=1e-3, batch=64, epochs=15, 早停 patience=3, 训练时固定 seed
def predict_m3(model, X_seq) -> np.ndarray  # 概率
```
- [ ] **Step 1: 写失败测试**（若 torch 未装：跳过运行但必须显式断言 ImportError 信息）
```python
def test_m3_train_predict_smoke():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 40, 9))   # 50 窗 × 40 × 9 原始特征
    y = rng.integers(0, 2, 50).astype("float32")
    m = fit_m3(X, y, seq_len=40, seed=42, device="cpu")
    p = predict_m3(m, X)
    assert p.shape == (50,) and np.all((p >= 0) & (p <= 1))
```
- [ ] **Step 2: FAIL（或无 torch 时确认 ImportError 提示）→ Step 3: 实现 → Step 4: PASS**
- [ ] **Step 5: GPU 训练冒烟**（`fit_m3(..., device="cuda")` 记录训练时长到 docs/environment.md；无 GPU 则 CPU 训练并记录时长，报告对比）
- [ ] **Step 6: Commit**（`feat: add M3 BiLSTM sequence model (optional GPU)`）

---

### Task 9: M4 无监督异常检测

**Files:**
- Create: `src/models_m4.py`、`tests/test_models_m4.py`

**Interfaces:**
- Produces:
```python
def fit_m4(X_tr, seed=42) -> Union[IsolationForest, OneClassSVM]
# IsolationForest(n_estimators=200, random_state=seed): 特征 = L2/L3 列（且训练只用正常窗）
def score_m4(model, X) -> np.ndarray  # 归一化异常分数（越大越异常）
```
- 标签利用：**只能**用训练集中 label==0 的窗拟合（无监督语义），测试评全窗。

- [ ] **Step 1: 写失败测试**（合成"正常块"集中小值、"离群块"大值 → 分数区分）
```python
def test_m4_scores_attacks_higher():
    from src.models_m4 import fit_m4, score_m4
    norm = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 6)))
    X = pd.concat([norm, norm * 5])
    m = fit_m4(norm)
    s = score_m4(m, X)
    assert s[:200].mean() < s[200:].mean()
```
- [ ] **Step 2: FAIL → Step 3: 实现**（score 用 `decision_function` 取反后 min-max 归一）
- [ ] **Step 4: PASS → Step 5: 真实数据全量跑** 输出分数分布图到 outputs/
- [ ] **Step 6: Commit**（`feat: add M4 unsupervised isolation-forest detector`）

---

### Task 10: 离线评测与图表（核心实验输出）

**Files:**
- Create: `src/evaluate.py`、`tests/test_evaluate.py`
- Create: `scripts/run_pipeline.py`（至此加入 M0-M4 全链路，一键训练+评测）

**Interfaces:**
- Produces:
```python
def evaluate_all(models: Dict[str, Callable], preds: Dict[str, pd.DataFrame], test_df: pd.DataFrame) -> Dict[str, dict]
# 每个模型返回: AUROC, AUPRC, F1_best, precision@F1, recall@F1,
# 漏检率(missed fraction of attack windows), 误报率(false alarm fraction of normal windows),
# 检测延迟(对每个攻击窗: 该航班攻击首窗时刻 → 首次报警窗时刻, 单位秒; 均值/中位数)
def plot_all(outputs_dir: str, preds, metrics): ...  # outputs/figs/: roc.png, pr_curve.png, feature_importance.png, timeline_overlay.png(f1), confusion_matrix.png
def save_metrics(metrics, path="outputs/metrics.csv"): ...
```
- 注意：preds 已含 flight_id/window_start_s/label/prob_1/pred。检测延迟只取 test 航班，攻击窗以 max overlap 法定义窗级真值。

- [ ] **Step 1: 写失败测试**（在可控假数据上验算 AUROC 期望值）
```python
# tests/test_evaluate.py
import numpy as np, pandas as pd
from src.evaluate import evaluate_all

def test_evaluate_metrics_basic():
    preds = pd.DataFrame({"prob_1": [0.9, 0.8, 0.1, 0.2],
                          "pred": [1, 1, 0, 0], "label": [1, 0, 0, 1],  # 一假阳一假阴
                          "flight_id": ["f1"]*4, "window_start_s": [0., 1., 2., 3.]})
    m = evaluate_all({}, {"m": preds}, preds)
    assert 0.5 < m["m"]["AUROC"] < 1.0 and m["m"]["AUPRC"] >= 0
    assert "delay_mean_s" in m["m"] and "delay_median_s" in m["m"]
```
- [ ] **Step 2: FAIL → Step 3: 实现**（AUROC/AUPRC 用 `sklearn.metrics`：F1 在训练集阈值网格上取最佳，无缝复用 train 分布；延迟用窗口时间戳差，**不存在攻击窗的航班不参与延迟**）
- [ ] **Step 4: PASS**
- [ ] **Step 5: 跑全量**：`python scripts/run_pipeline.py --stage eval` → 产出 outputs/metrics.csv 与全部图，人工目检图内容（TIMELINE 叠加图必须能肉眼看出攻击段被检测红线覆盖）
- [ ] **Step 6: Commit**（`feat: add offline evaluation, metrics table and figures`）

---

### Task 11: 因果流式 CLI（在线检测模块）

**Files:**
- Create: `src/stream.py`（因果滑窗缓冲器）、`src/cli.py`、`tests/test_stream.py`

**Interfaces:**
- Consumes: Task 6/7 模型（加载 `outputs/models/` 下文 joblib 模型文件）、Task 4 特征函数。
- Produces:
```python
class CausalStreamer:
    def __init__(self, model, feature_extractor): ...
    def feed(self, row: pd.Series) -> Optional[dict]
    # 只维护 past windows 缓冲，行级 feed；每次产出窗口即做特征并推理；
    # 返回 None（未满窗）或 {"window_start_s":.., "prob":.., "alert":bool(prob>T)}
    def alarm_intervals(self) -> List[Tuple[float, float]]  # 合并报警窗为连续区间
# CLI 命令:
#   python -m src.cli predict --input data/flight.csv --model outputs/models/m2.joblib
#   输出 JSON: {"flight": "...", "spoofing": bool, "confidence": float,
#               "alarm_intervals": [[s,e],...], "n_windows": N, "runtime_s": ...}
```
- 阈值：取训练集上 F1 最佳的 T（存于模型打包文件）。

- [ ] **Step 1: 写失败测试**（构造长序列喂入，验证因果性：**后 10 窗的输出与前 10 窗无关**——单行 feed 与整段批处理结果一致）
```python
def test_causal_stream_consistency():
    model = DummyAwareModel()   # 返回 0.9
    s = CausalStreamer(model, identity_extractor)
    results = []
    for _, row in df.iterrows():
        r = s.feed(row)
        if r: results.append(r)
    assert results[0]["prob"] == 0.9
```
- [ ] **Step 2: FAIL → Step 3: 实现 → Step 4: PASS**
- [ ] **Step 5: 真实航班 CLI 冒烟** 输出合法 JSON 且半段飞行（前 60 窗）能在运行时间维上线性积累
- [ ] **Step 6: Commit**（`feat: add causal streaming CLI detector`）

---

### Task 12: FastAPI 在线服务

**Files:**
- Create: `src/server.py`、`tests/test_api.py`

**Interfaces:**
- Consumes: Task 11 `CausalStreamer` + 模型加载。
- Produces:
- `GET /healthz` → `{"status": "ok"}`
- `POST /predict`（body: `{"rows": [...], "flight_id": "..."}`）→ `{"flight": "...", "spoofing": bool, "confidence": float, "alarm_intervals": [[s,e]], "n_windows": N}`
- `POST /predict_file`（multipart 上传 CSV，范化后走同一 streamer）
- 启动：`uvicorn src.server:app --port 8000`

- [ ] **Step 1: 写失败测试**
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from src.server import app

def test_healthz():
    assert TestClient(app).get("/healthz").json() == {"status": "ok"}

def test_predict_endpoint_schema():
    client = TestClient(app)
    rows = [{"time_s": t, "lat": 34.0, "lon": 108.9, "alt_gps": 20.0,
             "alt_baro": 20.0, "vel_e": 0.1, "vel_n": 0.0, "vel_u": 0.0,
             "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0., "gy": 0., "gz": 0.,
             "roll": 0., "pitch": 0., "yaw": 0.} for t in np.arange(0, 5, 0.05)]
    r = client.post("/predict", json={"flight_id": "test", "rows": rows})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"spoofing", "confidence", "alarm_intervals"}
```
- [ ] **Step 2: FAIL → Step 3: 实现**（模型加载失败时启动仍成功，/predict 返回 503 与原因）
- [ ] **Step 4: PASS → Step 5: 本地起服务并 curl 冒烟**
```bash
uvicorn src.server:app --port 8000 & curl http://127.0.0.1:8000/healthz
```
- [ ] **Step 6: Commit**（`feat: add FastAPI online detection service`）

---

### Task 13: 一键流程与最终验收

**Files:**
- Create: `scripts/run_all.py`（依次执行 download → make_dataset → train(M0-M4) → eval → figures；`--skip-download`/`--skip-train` 支持）、`README.md`（完整运行文档：环境、下载、一键命令、预期输出清单、AI 使用声明）
- Create: `tests/test_smoke.py`（调用 run_all 的小数据模式：`--tiny`，只取前 3 个航班，全链路验证 + pytest 在 CI 式一次跑通）

- [ ] **Step 1: 编写 run_all.py**（参数 `--tiny`：用合成/3航班数据快速冒烟；`--full`：全量），输出结构核对：README 所列产物均存在
- [ ] **Step 2: 写 test_smoke.py**：`subprocess.run([sys.executable, "scripts/run_all.py", "--tiny"])` 返回码 0 且 `outputs/metrics.csv` 存在
- [ ] **Step 3: 空环境复现验证**（或另开 venv）：
```bash
pip install -r requirements.txt && python scripts/run_all.py --full
# 预期: 依次打印 数据下载 → 特征行数 → 各模型训练时长 → metrics.csv 生成 → 8 张图生成 → done
```
- [ ] **Step 4: Commit**（`feat: add one-command pipeline and smoke test`）

---

### Task 14: 综合报告与打包

**Files:**
- Create: `report/report.md`（按任务要求 6 节：问题定义/数据介绍与预处理/方法/实验设置、指标与运行环境/实验结果表格与可视化/AI 工具与外部资源声明），所有表格数字**从 outputs/metrics.csv 粘取**，不允许手编。
- Create: `docs/environment.md` 校对（GPU：驱动+显存+训练时长）

- [ ] **Step 1: 生成 metrics 彩蛋表**（`python -c` 读 outputs/metrics.csv 打印成 markdown 表格贴进报告）
- [ ] **Step 2: 撰写报告 6 节**（每节注明对应脚本/图出处；失败与局限节：如实写出 baseline 失败点，如"未标注即整航班打标导致边窗噪声"）
- [ ] **Step 3: README 补全**（运行预期输出列举 + 可选提交项 checklist；明确"AI 辅助编程范围"声明）
- [ ] **Step 4: 打包**：`data_processed`/`outputs/models` 若体积 > 100MB 则仅提交清单；`compress` 生成 `submit.tar.gz`（含 README/report/src/scripts/tests/configs/requirements.txt/outputs/*.csv|png）
- [ ] **Step 5: Commit**（`docs: finalize report and submission package`）

---

## 风险与前置确认点

1. **数据可下载性**：若本机无法访问 GitHub，改用代理或手动下载（Task 2 Step 1 失败即停，先确认）。
2. **字段匹配**：真实列名与预想不同 → 在 `docs/dataset-notes.md` 记录并调整 `infer_schema` 映射（Task 3 Step 5 已有回环验证）。
3. **标签粒度**：无时间级标注 → 按航班标注（Task 5 已内置该路径），报告写明限制。
4. **torch/GPU 失败**：不影响主链路（Task 8 允许跳过），报告如实记录。
5. **时间预算**：Task 1-2（45min）、Task 3-5（1.5h）、Task 6-9（2h）、Task 10-11（1.5h）、Task 12-13（1.5h）、Task 14（1h）——总计约 8-9h 有效工作量，分布到 3 天。
