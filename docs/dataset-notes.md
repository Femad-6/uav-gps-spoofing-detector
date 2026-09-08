# 数据集笔记（Schema 与标签事实）

来源：UAV-GPS-Spoofing-Dataset（antthony-finn），PX4-SITL + Gazebo-Classic 2023 采集，250Hz。
论文：[Detecting Stealthy GPS Spoofing Attack Against UAVs Using Onboard Sensors](https://www.semanticscholar.org/paper/Detecting-Stealthy-GPS-Spoofing-Attack-Against-UAVs-Finn-Jia/e6039d9f64851c3e742479b4e9bca96937420ecb) (IEEE INFOCOM 2024 WKSHPS)

## 下载

- 数据宿主：Google Drive（仓库 README "Data Links"），`PX4_Simulation_Data.zip` = 2,577,647,220 字节（2.58GB）。
- 本机下载受代理影响单连接 128MB 上限，已用 `scripts/download_data.py` 的断点续传+重试完成。
- `data/raw/` 解压后：`PX4_Simulation_Data/{Merged,Raw}/{Curved,Random,Straight}/{Attacked,Normal}/`。
- **Merged**（时间同步，本项目的输入，`log_*.csv` 共 **360 个**）：
  `time_usec, pressure_altitude, latitude_deg, longitude_deg, altitude, velocity, velocity_east, velocity_north, velocity_up, attack_enabled, orientation_x, orientation_y, orientation_z, orientation_w, angular_velocity_x/y/z, linear_acceleration_x/y/z, magnetic_field_x/y/z`

## 标签粒度（关键事实）

- `attack_enabled` 是**逐样本** 0/1 字段（merged 版自带）→ 攻击区间可从数据直接提取，无需航班级近似。
- 本航班样例（Curved/Attacked）：attack 区间覆盖全程 97%（攻击在 60-120s 随机开始后的持续期较长）。
- 时间戳为绝对 ucsec（2933.5s 起），非从 0 开始；行序未排序 → 加载器统一排序 + 重采样 20Hz。

## 映射表（scripts 依据）

| 规范化列 | merged 原始列 |
|---|---|
| time_s | time_usec（usec→s，排序） |
| alt_baro | pressure_altitude（优先级 > altitude） |
| alt_gps | altitude |
| lat/lon | latitude_deg/longitude_deg |
| vel_e/n/u | velocity_east/north/up（分量列优先；Velocity 标量列被忽略） |
| ax/ay/az | linear_acceleration_x/y/z（body 系 m/s²） |
| gx/gy/gz | angular_velocity_x/y/z（rad/s） |
| roll/pitch/yaw | orientation_x/y/z/w（四元数，w 在最后）转换 |
| mx/my/mz | magnetic_field_x/y/z（原单位） |
| attack | attack_enabled |

注意：Raw 子目录为每传感器独立 CSV（未同步 + 向量字符串格式），本项目未使用。

## 已知限制

- 输入为仿真数据（PX4+Gazebo），与真实射频诱骗存在差异（无真实信号层噪声）；报告撰写时如实说明。
- 攻击时段内 `attack_enabled=1`，但真实"诱骗生效门限"（GPS 与真值偏差开始显著变大）未必与之一致；报告的检测延迟指标据此字段计算。
