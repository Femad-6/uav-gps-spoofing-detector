# 论文驱动的无人机 GPS 诱骗检测优化设计

## 1. 目标

在保留现有 M0-M4 结果可比较性的前提下，将 Finn 等、PerDet、Feng 等及 Hakani 等论文中可由当前遥测数据支持的方法转化为增强特征、受误报约束的阈值选择、连续窗口报警策略和更严格的跨航线实验。完成后必须重新生成实验产物，并以实际结果更新 `report/report.md`；指标改善不是预设结论。

## 2. 当前基线与问题

当前全量数据包含 360 个航班、89,804 个窗口和 71 个特征。航班级 train/val/test 数量为 252/36/72，对应 62,554/8,533/18,717 个窗口。当前最佳测试 AUROC 约为 0.777，但最佳模型 M2 的窗口误报率仍为 0.568；M1 跨 Straight/Curved/Random 航线 AUROC 分别约为 0.627/0.535/0.533。

代码层面的主要缺口：

- 加载器读取了 `mx/my/mz`，特征层却没有磁力计特征；
- 经纬度仅用于 `c_jump`，没有相对 ENU 位移、航迹角及渐进漂移特征；
- `c_imu_gps_acc` 直接用加速度模减去重力，是机体系与导航系之间的粗略近似；
- 阈值以验证集 F1 最大为目标，对当前接近平衡的窗级标签倾向于高召回、高误报；
- 在线服务以任意单窗越阈值作为整段飞行的攻击结论；
- 跨航线脚本只覆盖 M1，无法判断其他模型是否真正泛化。

## 3. 范围

### 3.1 本轮实现

1. 新增论文启发的增强物理特征，统一使用 `e_` 前缀。
2. 冻结 M0-M4 的旧特征选择语义，新增 M2e 增强特征模型用于公平消融。
3. 新增误报率约束阈值，并保留 F1 最优阈值作为诊断字段。
4. 新增纯状态逻辑的连续窗口确认与迟滞清除策略，离线评测和在线服务复用同一实现。
5. 将跨航线评测扩展至 M0、M1、M1b、M2、M2e、M4。
6. 增加回归测试、tiny 冒烟、全量训练和跨航线重跑。
7. 用实际产物更新 README 和 `report/report.md`。

### 3.2 明确不在本轮实现

- 不进行 SDR 射频发射、真实无人机自动降落或返航控制；
- 不伪造当前数据集不存在的 HDOP、卫星数量或 GPS fix quality；
- 不直接移植其他论文的 3.5 m、15 度、25 m/s^2 等平台特定阈值；
- 不引入 XGBoost 新依赖。先隔离验证特征与决策逻辑的收益，再决定是否增加模型复杂度；
- 不用测试集选择特征、窗口长度、模型参数或报警阈值。

## 4. 架构与数据流

```text
CSV -> data_loader -> 20 Hz normalized rows
    -> features: legacy f_/c_ + enhanced e_
    -> flight-disjoint train/val/test
    -> legacy M0-M4 + enhanced M2e
    -> validation threshold selection
    -> raw window score
    -> shared temporal alarm policy
    -> offline metrics / CLI / FastAPI
```

增强特征与决策策略必须是独立模块边界。特征函数只从当前窗口输入计算，不接触 split 或标签；阈值函数只使用传入的验证预测；时序报警策略只接收按时间排序的概率流。

## 5. 增强特征设计

增强特征全部以 `e_` 开头，使旧模型可显式排除它们。缺少必要传感器时输出 NaN，由现有模型插补或原生缺失值处理机制接管。

### 5.1 相对位置与轨迹

在每个窗口内以首个有效经纬度为局部原点，使用小范围地球近似转换为 east/north 米制坐标，不保留绝对经纬度。

- `e_gps_step_mean`、`e_gps_step_std`、`e_gps_step_max`：相邻位置步长；
- `e_gps_speed_resid_mean`、`e_gps_speed_resid_max`：位置差分速度与 `vel_e/vel_n` 报告速度的残差；
- `e_track_yaw_mean_abs`：GPS 航迹角与姿态 yaw 的圆周绝对差；
- `e_track_yaw_change_corr`：解缠后的航迹角变化与 yaw 变化相关系数；
- `e_track_yaw_slope_diff`：两种方向趋势的斜率差。

速度过低时航迹角不稳定，相关特征应忽略低于小速度门限的样本；有效点不足时返回 NaN。

### 5.2 磁力计与航向

- `e_mag_norm_mean`、`e_mag_norm_std`、`e_mag_norm_cv`：磁场模长及相对波动；
- `e_mag_axis_change_max`：三轴相邻变化的最大模长；
- `e_yaw_mag_change_corr`：姿态 yaw 与磁航向变化的相关性；
- `e_yaw_mag_resid_std`：两种航向之间圆周残差的波动。

磁航向采用 roll/pitch 倾斜补偿。由于机体坐标符号可能随平台变化，本轮只依赖窗口内变化和相关性，不将磁航向绝对偏置作为攻击证据。

### 5.3 IMU-GPS 物理一致性与振动

使用 roll/pitch/yaw 构造旋转矩阵，将机体系加速度旋转到导航系。为降低重力符号和垂直轴约定的不确定性，第一版比较导航系水平加速度与 GPS 水平速度微分：

- `e_imu_gps_acc_h_mean`、`e_imu_gps_acc_h_max`：水平加速度矢量残差；
- `e_imu_gps_acc_dir_mean`：有效运动样本上的加速度方向差；
- `e_vibration_rms`：加速度模相对窗口中位数的 RMS；
- `e_vibration_peak`：去中位数后绝对峰值；
- `e_gps_baro_abs_mean`、`e_gps_baro_abs_max`：高度差绝对值统计。

旋转与坐标约定必须由合成姿态单元测试覆盖。若真实数据诊断显示轴约定与假设不符，报告该限制，不通过查看测试标签反向选择符号。

## 6. 模型与消融设计

### 6.1 基线冻结

- M0：仍只使用原 5 个 `c_` 一致性特征；
- M1：仍只使用旧 `f_` 特征；
- M1b、M2、M4：显式使用 `f_` 和 `c_`，不因新增 `e_` 列而静默改变；
- M3：仍使用原始 40x9 序列。

### 6.2 新增 M2e

M2e 使用与 M2 相同的 `HistGradientBoostingClassifier` 参数，但输入 `f_ + c_ + e_`。M2 与 M2e 的差异仅为增强特征，构成可解释的消融实验。模型保存到 `outputs/models/m2e.joblib`，并进入统一指标表。

## 7. 阈值选择

新增两种验证集阈值：

- `f1_threshold`：保留现有最大化 F1 的结果，仅用于诊断对照；
- `threshold`：正式运行阈值，在验证集正常窗误报率不超过 10% 的可行候选中最大化召回；召回相同时最大化 F1，再相同时选择更高阈值。

候选阈值来自概率取值和边界值，并包含一个高于最大概率的“无报警”候选，保证约束总有可行解。若验证集中没有正常窗或攻击窗，返回明确错误，不回退到测试集。

指标表新增 `threshold_policy`、`f1_threshold`、`val_false_alarm_rate`。测试集仍只在阈值冻结后评估一次。

## 8. 连续窗口报警策略

新增 `src/alarm.py`，提供无模型依赖的 `TemporalAlarmPolicy`：

- 最近 5 个窗口中至少 3 个概率达到正式阈值，状态从 `normal` 进入 `confirmed`；
- confirmed 后，连续 3 个窗口低于 `0.8 * threshold` 才退出；
- 每次更新返回原始越阈值结果和确认状态；
- `reset()` 清空状态，禁止跨航班携带历史。

`CausalStreamer.feed()` 保留 `alert` 原字段表示单窗结果，新增 `confirmed_alert`。CLI/API 的最终 `spoofing` 与报警区间改用确认状态。离线评测按 `flight_id`、`window_start_s` 排序后应用相同策略，并新增确认后的误报率、召回、漏检率和延迟；原始窗级指标继续保留。

## 9. 跨航线评测

`scripts/run_route_holdout.py` 增加 `--models` 参数，支持 M0、M1、M1b、M2、M2e、M4。每个留出航线均重新：

1. 构造航线级 test；
2. 从其余航班构造 val；
3. 仅在 train 拟合模型与预处理；
4. 仅在 val 选择阈值；
5. 在留出航线评测一次。

M3 因原始序列重建和训练成本较高，本轮不进入跨航线矩阵，并在报告中明确。

## 10. 测试策略

严格执行测试先行，每个生产行为先观察对应测试失败：

- `tests/test_features.py`：ENU 位移、静止窗口、航迹/yaw wrap、磁力计缺失、姿态旋转、振动；
- `tests/test_models_m2.py`：M2 只读取旧特征，M2e 读取增强特征；
- `tests/test_models.py`、`tests/test_models_m4.py`：新增列不会污染旧基线；
- `tests/test_evaluate.py`：误报约束、无报警候选、异常验证集、确认态指标；
- `tests/test_stream.py`：3-of-5 确认、迟滞退出、航班 reset；
- `tests/test_api.py`：单窗异常不再使整段飞行确认为攻击；
- 新增或扩展跨航线脚本测试，验证模型列表与 split 边界。

验证顺序：目标测试 -> 全部 pytest -> `run_all.py --tiny --skip-download` -> 全量 `make_dataset.py` -> 全量 `run_pipeline.py` -> 全量 `run_route_holdout.py` -> 产物字段与报告数字核对。

## 11. 报告与成功判据

工程完成判据：

- 所有新行为均有先失败后通过的测试；
- 全套测试通过；
- 航班级 train/val/test 互斥和原数量关系得到验证；
- M0-M4 旧特征语义保持，M2e 提供独立消融；
- CLI/API 不再以单个异常窗直接确认攻击；
- 指标、模型和跨航线产物由当前代码重新生成；
- `report/report.md` 只记录实际输出，并清楚说明新旧协议差异。

研究目标而非强制完成判据：验证测试误报率和跨航线 AUROC 是否改善。若 M2e 或确认策略没有改善，保留结果并解释原因，不删除负面实验、不改用测试集调参。

## 12. 风险与回退

- 增强特征可能增加计算量：记录特征提取和在线运行时间；必要时用消融决定是否保留，而不是先优化微小性能。
- 坐标约定可能影响方向特征：用合成数据验证数学实现，用真实正常航班做无标签分布诊断。
- 10% 验证误报约束不保证测试误报也低：报告 val/test 差距，将其视为域偏移证据。
- 时间确认会增加检测延迟：同时报告确认前后误报率、召回和延迟。
- 如果增强模型变差，M0-M4 与旧输出结构仍可作为回退和对照。
