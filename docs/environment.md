# 运行环境

- 收集日期：2026-09-08
- 操作系统：Windows 11 Home China (10.0.26200)
- Python：3.12.1（全局环境，D:\Python）
- GPU：NVIDIA GeForce RTX 4050 Laptop（6GB 显存），驱动 566.24，CUDA 12.7（pytorch 检测结果见《报告》硬件节）
- git：2.55.0.windows.2
- pyarrow：21.0.0（仓库内 `.venv` 验证）
- 顶层依赖快照：`requirements-lock.txt`
- Docker：当前机器未安装，`Dockerfile` 已提供但未在本机实际构建
- 数据集：UAV-GPS-Spoofing-Dataset（PX4-SITL + Gazebo-Classic 仿真采集，250Hz），见 docs/dataset-notes.md
- 说明：开发机全局环境装有 hydra-core/omegaconf，其 pytest 插件与新版 antlr4 运行时冲突；本项目 pytest.ini 已 `-p no:hydra_pytest` 屏蔽，干净环境无此问题。
