# 无人机 GPS 诱骗检测与安全分析

2026 推免短期综合任务。基于公开 UAV-GPS-Spoofing-Dataset（PX4/Gazebo 仿真实机日志，
250Hz，含正常与隐蔽 GPS 诱骗飞行），完成多源传感器一致性特征设计、M0–M4 方法对比、
因果流式在线检测服务（CLI + FastAPI）。

当前严格按航班划分 train/val/test：模型只在 train 拟合，val 选择报警阈值，test 仅用于
最终评测。修正后的最佳测试 AUROC 为 0.777；跨航线实验显示泛化仍有限，详见报告。

## 项目结构

```text
├── src/                     # 核心代码
│   ├── config.py            # 全局配置（随机种子 42、窗参数、路径）
│   ├── data_loader.py       # 飞行日志 → 规范化 20Hz DataFrame（自动列名推断）
│   ├── features.py          # L1 原始统计 + L2 物理一致性 + L3 滑窗特征
│   ├── labels.py            # attack 区间标签 + 按航班划分 train/val/test
│   ├── models.py            # M0 阈值规则、M1 随机森林（L1 特征）
│   ├── models_m2.py         # M2 HistGradientBoosting（全特征）
│   ├── models_m3.py         # M3 BiLSTM（原始时序，可 GPU）
│   ├── models_m4.py         # M4 孤立森林（无监督，仅正常窗训练）
│   ├── evaluate.py          # AUROC/AUPRC/F1/漏检/误报/检测延迟 + 图表
│   ├── stream.py            # 因果流式检测器（在线语义）
│   ├── cli.py               # 命令行检测
│   └── server.py            # FastAPI 在线服务
├── scripts/
│   ├── download_data.py     # 数据下载（gdown + 断点续传重试）
│   ├── make_dataset.py      # 原始日志 → 窗级特征 parquet + 划分
│   ├── run_pipeline.py      # M0-M4 训练 + 评测 → metrics.csv + 图表
│   ├── run_route_holdout.py # 留一航线测试 M1 跨航线泛化
│   └── run_all.py           # 一键全流程（--tiny 冒烟）
├── tests/                   # pytest 测试（核心函数自检）
├── data/raw/                # 原始数据（gitignore，脚本下载）
├── data_processed/          # 窗级特征（gitignore，管线生成）
├── outputs/                 # 实验结果：models/metrics.csv/figs/
├── docs/                    # 设计文档、实施计划、数据集笔记、环境说明
└── report/report.md         # 综合实验报告
```

## 运行环境

- Windows/Linux + Python 3.10+（本项目在 Windows 11 / Python 3.12.1 验证）
- GPU 可选（M3 加速；无 GPU 自动回退 CPU）
- 依赖：`pip install -r requirements.txt`（本次实验的顶层依赖版本见 `requirements-lock.txt`）

## 数据下载

```bash
python scripts/download_data.py
```

- 数据源为 Google Drive（2.58GB），脚本支持断点续传与最多 5 次自动重试；
- 下载完成后解压到 `data/raw/PX4_Simulation_Data/`，合计 360 个 merged 航班日志
  （3 种航线 × 正常/攻击 × 各 60 个，含逐样本 `attack_enabled` 标签）。

## 一键运行

```bash
python scripts/run_all.py        # 全量：下载 → 特征 → 训练 M0-M4 → 评测 → 图表
python scripts/run_all.py --tiny # 冒烟：仅前 3 个航班（~5 分钟，产物写入 *_tiny 目录）
```

预期输出（全量）：

```text
[1/3] 下载数据 ...   → [done] 数据就绪
[2/3] 数据管线       → 窗数 N (train X / val V / test Y)，正样本率 P
[3/3] 训练 + 评测    → 每模型训练耗时 + 评测指标表 + [eval] 图表已保存
[ok] m0 ... m4 训练并保存 -> outputs/models/*.joblib
[done] 一键流程完成
```

产物清单：

- `outputs/models/{m0,m1,m1b,m2,m3,m4}.joblib`
- `outputs/metrics.csv`（各模型 AUROC/AUPRC/F1/漏检率/误报率/检测延迟）
- `outputs/metrics_route_holdout.csv`（M1 跨航线泛化结果）
- `outputs/figs/`（roc_pr.png、feature_importance_*.png、timeline_*.png）
- `data_processed/features_windowed.parquet`（可复现的全量特征）

`--tiny` 不复制或改名原始日志，使用独立的航班划分、`data_processed_tiny/` 与
`outputs_tiny/`，因此不会覆盖全量实验的划分、模型和结果。

## 离线训练/评测（分步）

```bash
python scripts/make_dataset.py                 # 特征 + 标签 + 划分
python scripts/run_pipeline.py                 # 训练 M0-M4 + 测评
python scripts/run_pipeline.py --skip-m3       # 无 torch 时跳过 M3
python scripts/run_route_holdout.py             # M1 分别留出直线/曲线/随机航线测试
```

## 在线检测

**CLI（逐行因果流式）：**

```bash
python -m src.cli predict --input data/raw/PX4_Simulation_Data/Merged/Curved/Attacked/log_21_04_2023_07_21_03.csv \
  --model outputs/models/m1.joblib
```

默认读取模型训练时由验证集确定的报警阈值；可用 `--threshold` 手动覆盖。

预期输出（JSON）：

```json
{
  "flight": "log_21_04_2023_07_21_03.csv",
  "spoofing": true,
  "threshold": 0.25,
  "confidence": 0.965,
  "alarm_intervals": [[2938.6, 3163.1]],
  "n_windows": 456,
  "runtime_s": 22.7
}
```

该输出证明在线因果链路可以运行，不代表已达到部署准确度；独立测试仍有较高误报率。

**API 服务：**

```bash
uvicorn src.server:app --port 8000
curl http://127.0.0.1:8000/healthz
curl -X POST http://127.0.0.1:8000/predict_file -F file=@data/raw/.../log_*.csv
```

接口：`GET /healthz`；`POST /predict`（JSON，`rows` 为规范化行的列表，服务会按 `time_s` 排序并重采样至 20Hz，>=40 行）；
`POST /predict_file`（multipart 上传 CSV，默认上限 50 MiB）。模型未训练时 `/predict` 返回 503。

Docker 服务镜像不内置模型，启动时挂载训练产物：

```bash
docker build -t uav-spoofing-api .
docker run --rm -p 8000:8000 -v "${PWD}/outputs/models:/app/outputs/models:ro" uav-spoofing-api
```

## 测试

```bash
python -m pytest tests/            # 28 个用例，覆盖加载器/特征/标签/模型/评测/流式/API
```

所有实验固定随机种子（`src/config.py: SEED = 42`），按航班划分训练/验证/测试（`configs/split.json`）；
模型仅用训练航班拟合，报警阈值由验证航班选择，测试航班只用于最终报告。
硬件环境记录见 `docs/environment.md`。

## AI 工具与开源代码使用声明

- 本项目使用 Claude Code（Claude Agent SDK）与 OpenAI Codex 辅助方案设计、代码编写、调试与报告草拟；
  每一步均由作业者复核并本地验证运行，所有实验数字取自本地运行输出，无编造。
- 使用的开源库：pandas、numpy、scikit-learn、matplotlib/seaborn、fastapi/uvicorn、pyarrow、
  pytest、torch（完整列表见 `requirements.txt`）。
- 数据集：UAV-GPS-Spoofing-Dataset（开源，见 docs/dataset-notes.md）；
  未复制/修改任何第三方检测代码（检测方法为本项目自主设计）。
