# 参考文献

本目录用于存放本项目阅读的第三方论文。**只有明确以开放获取（CC BY）许可发布的论文才随仓库分发**；
受出版方版权限制的副本仅保存在本地，不进入版本库，README 与报告中一律以 DOI 链接引用。

## 随仓库分发（开放获取）

### 1. PERDET: Machine-Learning-Based UAV GPS Spoofing Detection Using Perception Data

- 作者：Xiaomin Wei, Yao Wang, Cong Sun（西安电子科技大学）
- 出处：*Remote Sensing*, 2022, 14(19), 4925
- DOI：[10.3390/rs14194925](https://doi.org/10.3390/rs14194925)
- 许可：© 2022 by the authors, Licensee MDPI, Basel, Switzerland.
  本文以 [Creative Commons Attribution (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 许可发布。
- 文件：`2022_Wei_PerDet_ML_UAV_GPS_Spoofing_Perception_Data.pdf`

### 2. Evaluation and telemetry-based detection of GPS spoofing effects on UAV navigation using software-defined radio

- 作者：Raj Hakani, Abhishek Rawat, Manan Shah, Pavan Verma
- 出处：*Scientific Reports*, 2026, 16, 24385
- DOI：[10.1038/s41598-026-53481-9](https://doi.org/10.1038/s41598-026-53481-9)
- 许可：Scientific Reports 开放获取论文，以 CC BY 许可发布。
- 文件：`2026_Telemetry_Based_GPS_Spoofing_Detection_UAV_SDR.pdf`

## 仅本地保存（不随仓库分发）

以下两篇的下载副本带有出版方的版权声明，明确要求"再分发至服务器需事先获得授权"，
因此**不进入版本库**（由 `.gitignore` 排除），仅以 DOI 链接引用：

- **Detecting Stealthy GPS Spoofing Attack Against UAVs Using Onboard Sensors** —
  Anthony Finn, Mengjie Jia, Yanyan Li, Jiawei Yuan；IEEE INFOCOM WKSHPS 2024；
  DOI：[10.1109/INFOCOMWKSHPS61880.2024.10620818](https://doi.org/10.1109/INFOCOMWKSHPS61880.2024.10620818)。
  该副本页脚标注 `©2024 IEEE` 与 "Authorized licensed use limited to: University of
  Massachusetts - Dartmouth ... Restrictions apply."。
  本项目使用的数据集与该论文同源，是本项目最直接的参考。
- **Efficient Drone Hijacking Detection using Onboard Motion Sensors** —
  Zhiwei Feng, Nan Guan, Mingsong Lv, Weichen Liu, Qingxu Deng, Xue Liu, Wang Yi；
  ACM Transactions on Embedded Computing Systems (TECS), 2018；
  作者自存档副本（Uppsala University）。该副本标注 `© 2018 Association for Computing
  Machinery`，并声明 "To copy otherwise, or republish, to post on servers or to redistribute
  to lists, requires prior specific permission and/or a fee."。

## 使用说明

这些论文仅用于本项目的研究背景、传感器选择与实验设计参考。**其中的物理阈值、特征数值与
模型超参数没有被直接移植到本项目**；`src/` 下的检测方法、特征定义、数据划分与评测协议
均为自主实现。详见 `report/report.md` 第 3.3 节与第 6 节。
