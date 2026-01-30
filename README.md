# 风电机组数据清洗（考虑空气密度）

## 📖 项目简介

### 研究背景

风力发电作为可再生能源的重要组成部分，其发电效率和数据质量直接影响电网调度和运维决策。风机在运行过程中会产生大量的 SCADA（数据采集与监控系统）数据，这些数据包含风速、功率、环境参数等信息。然而，由于传感器故障、通信异常、极端工况等原因，原始数据往往存在大量噪声和异常值。

空气密度是影响风机功率输出的关键环境因素之一，它随海拔、温度、湿度等参数变化而变化。传统的风机数据清洗方法往往忽略空气密度的影响，导致清洗效果不佳，甚至误判正常数据为异常值。

### 研究目的

本项目旨在开发一套考虑空气密度影响的风机数据清洗方法，通过以下方式提升数据质量：

1. **引入空气密度作为输入特征**：在数据清洗和建模过程中考虑空气密度对功率曲线的影响
2. **多轮迭代清洗**：采用多轮清洗策略，逐步剔除异常值，提高数据质量
3. **多种阈值方法**：支持多种异常检测阈值计算方法（KNN、分位数等），适应不同场景
4. **模块化设计**：采用模块化架构，便于扩展和维护

### 研究意义

- **提升数据质量**：通过考虑空气密度，更准确地识别和剔除异常数据
- **改善功率预测**：高质量的训练数据能提升风功率预测模型的准确性
- **优化运维决策**：准确的数据分析为风场运维和故障诊断提供可靠依据
- **推动方法创新**：为风电数据处理领域提供新的研究思路和工具

---

## 📊 数据说明

### 数据来源

本项目使用的数据来自风电场的 SCADA 系统，主要包括：

- **风机运行数据**：包含风速、功率、转速等时序数据
- **环境数据**：空气密度、温度、气压等环境参数
- **数据格式**：CSV 格式的宽表数据，每行代表一个时间戳的多台风机数据

### 主要变量说明

#### 输入变量
- **风速（Wind Speed）**：风机叶轮前的风速，单位通常为 m/s
- **空气密度（Air Density, ρ）**：单位体积空气的质量，单位为 kg/m³
  - 正常范围：1.07 ~ 1.37 kg/m³
  - 影响因素：海拔高度、温度、湿度、气压

#### 输出变量
- **功率（Power）**：风机输出的有功功率，单位为 kW 或 MW
- **额定功率（Rated Power）**：风机设计的最大功率输出

#### 派生变量
- **归一化功率**：功率 / 额定功率
- **风速-功率残差**：实际功率与预测功率的差值
- **Z-score 残差**：标准化后的残差值

### 数据基本结构

```
数据文件示例结构：
- timestamp: 时间戳
- {station}_风机{id}_风速: 风机编号 id 的风速
- {station}_风机{id}_功率: 风机编号 id 的功率
- {station}_空气密度: 该站点的空气密度
```

**数据组织方式**：
- 宽表格式：一行包含某时刻所有风机的数据
- 时间戳索引：按时间顺序排列
- 多风机并行：同一站点的多台风机数据在同一表中

---

## 🔧 数据清洗与预处理流程

### 整体流程

本项目采用**模块化管道**设计，主要包括以下步骤：

```
1. 数据加载与预处理
   ├── 读取原始 CSV 数据
   ├── 加载空气密度数据
   └── 数据格式验证与清洗

2. 数据集划分
   ├── 训练集 / 验证集 / 测试集划分
   ├── 支持随机划分和时间块划分
   └── 可复用已保存的划分结果

3. 数据归一化
   ├── 风速归一化（Min-Max 或 Standard）
   ├── 空气密度归一化
   └── 功率归一化（相对额定功率）

4. 多轮迭代清洗
   └── 每轮清洗包括：
       ├── 深度神经网络训练（MLP）
       ├── 功率预测
       ├── 残差计算
       ├── 阈值计算（异常检测）
       └── 异常样本标记与剔除

5. 结果保存与评估
   ├── 清洗后数据导出
   ├── 异常样本统计
   └── 可视化结果（可选）
```

### 数据清洗关键步骤

#### 1. 数据预处理
- **缺失值处理**：剔除风速或功率缺失的样本
- **空气密度处理**：
  - 如果启用 `force_drop_rho_na`，则剔除空气密度缺失的样本
  - 支持恒定值填充、随机打乱等多种处理模式
- **风速过滤**：仅保留指定风速范围内的数据（默认 0-15 m/s）

#### 2. 异常值检测方法

本项目支持多种阈值计算方法：

**a) KNN 局部阈值法（`knn_local`）**
- 基于 K 近邻的局部异常检测
- 计算每个样本在其邻域内的残差分布
- 适用于功率曲线存在局部变化的场景

**b) 分位数阈值法（`quantile_power` / `quantile_zresid`）**
- 基于分位数的全局阈值
- `quantile_power`：直接对功率残差计算分位数
- `quantile_zresid`：对标准化残差（Z-score）计算分位数

**c) 自定义阈值**
- 可扩展其他阈值方法
- 通过 `ThresholdMethod` 基类实现

#### 3. 多轮清洗策略

```python
# 伪代码示例
for pass_id in range(cleaning_passes):
    # 1. 训练模型
    model = fit_mlp_center(train_data, use_rho=rho_for_clean)
    
    # 2. 预测功率
    pred_power = model.predict(all_data)
    
    # 3. 计算残差
    residuals = actual_power - pred_power
    
    # 4. 计算阈值
    thresholds = compute_thresholds(residuals, method="knn_local")
    
    # 5. 标记异常值
    is_abnormal = (residuals > thresholds.upper) | (residuals < thresholds.lower)
    
    # 6. 剔除异常值（下一轮使用）
    train_data = train_data[~is_abnormal]
```

**多轮清洗的优势**：
- 第一轮：剔除明显异常值
- 后续轮次：基于更干净的数据重新训练，识别更细微的异常
- 逐步提升数据质量

#### 4. 空气密度处理模式

本项目支持多种空气密度使用模式（`rho_input_mode`）：

- **normal**：使用真实空气密度值
- **constant**：使用恒定值（如训练集均值）
- **shuffle**：打乱空气密度值以验证其影响
- **noise**：添加噪声以测试鲁棒性

---

## 💻 代码使用说明

### 运行方式

本项目通过**配置文件驱动**的方式运行，主要步骤如下：

#### 1. 准备配置文件

配置文件为 JSON 格式，包含实验参数。示例：

```json
{
  "defaults": {
    "stage1_root": "风机数据",
    "out_root": "输出结果目录",
    "device": "cuda:0",
    "seed": 42,
    "wind_scope": [0.0, 15.0],
    "cleaning_passes": 2
  },
  "stations": [
    {
      "name": "站点名称",
      "csv": "数据文件路径.csv",
      "turbine_start": 51,
      "turbine_end": 58
    }
  ],
  "runs": [
    {
      "name": "实验名称",
      "out_subdir": "输出子目录",
      "rho_for_clean": true,
      "rho_for_model": true,
      "thresholds": {
        "method": "knn_local"
      }
    }
  ]
}
```

#### 2. 运行主程序

```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

**参数说明**：
- `--config`：指定配置文件路径（必需）

#### 3. 查看结果

运行完成后，结果将保存在配置文件中指定的 `out_root` 目录下，包括：
- 清洗后的数据文件（CSV 格式）
- 异常样本标记
- 训练/验证/测试集划分结果
- 模型评估指标（如 MAE、RMSE 等）

### 主要脚本说明

#### `main.py`
- **功能**：主入口程序，负责读取配置并启动实验流程
- **输入**：JSON 配置文件
- **输出**：清洗后的数据和评估结果

#### `stage2_modular/pipeline/orchestrator.py`
- **功能**：核心编排器，控制整个数据清洗流程
- **主要方法**：
  - `run_single_run()`：执行单个实验配置
  - `run_stage2_for_station()`：处理单个站点的数据

#### `stage2_modular/models/center.py`
- **功能**：MLP 模型的训练和预测
- **模型结构**：多层感知机（可配置隐藏层）
- **损失函数**：支持 MSE、加权 MSE、Huber 损失等

#### `stage2_modular/thresholds/`
- **功能**：异常检测阈值计算模块
- **包含方法**：
  - `knn_local.py`：KNN 局部阈值
  - `quantile_power.py`：功率分位数阈值
  - `quantile_zresid.py`：残差分位数阈值
  - `base.py`：阈值方法基类

#### `stage2_modular/core/`
- **功能**：核心工具模块
- **主要文件**：
  - `device.py`：设备管理（CPU/GPU）
  - `scaler.py`：数据归一化
  - `splits.py`：数据集划分
  - `utils.py`：通用工具函数

### 示例运行方式

```bash
# 1. 使用默认配置运行
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json

# 2. 如果数据文件位置不同，修改配置文件中的路径
# 编辑 JSON 文件中的 "stage1_root" 和 "csv" 字段

# 3. 调整清洗参数
# 在配置文件的 "runs" 部分修改：
# - "cleaning_passes": 清洗轮数
# - "rho_for_clean": 是否在清洗时使用空气密度
# - "thresholds.method": 阈值计算方法
```

---

## 🛠️ 环境与依赖

### Python 版本

- **推荐版本**：Python 3.8 或更高
- **测试版本**：Python 3.8, 3.9, 3.10

### 依赖库

本项目主要依赖以下 Python 库：

#### 核心依赖
```
numpy >= 1.20.0          # 数值计算
pandas >= 1.3.0          # 数据处理
torch >= 1.10.0          # 深度学习框架（支持 CPU 和 GPU）
```

#### 可选依赖
```
matplotlib               # 数据可视化（如需绘图）
scikit-learn            # 机器学习工具（部分功能）
```

### 安装方式

#### 方式 1：使用 pip 安装（推荐）

```bash
# 安装核心依赖
pip install numpy pandas torch

# 如果使用 GPU（CUDA）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 安装可选依赖
pip install matplotlib scikit-learn
```

#### 方式 2：使用 conda 安装

```bash
# 创建虚拟环境
conda create -n wind-turbine python=3.9
conda activate wind-turbine

# 安装依赖
conda install numpy pandas
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib scikit-learn
```

#### 方式 3：从 requirements.txt 安装（如果提供）

```bash
pip install -r requirements.txt
```

### 硬件要求

- **CPU**：支持多核处理器（推荐）
- **内存**：至少 8GB RAM（大数据集建议 16GB 以上）
- **GPU**：可选，支持 NVIDIA GPU 加速（需安装 CUDA）
  - 推荐显存：6GB 以上
  - 配置文件中通过 `device` 参数指定：`"cuda:0"` 或 `"cpu"`

---

## 📁 项目结构说明

```
Wind-turbine-data-cleaning-considering-air-density/
│
├── main.py                          # 主程序入口
│   └── 功能：读取配置文件，启动数据清洗流程
│
├── experiments_compare_*.json        # 实验配置文件示例
│   └── 功能：定义实验参数、数据路径、清洗策略等
│
├── stage2_modular/                   # 核心代码模块
│   ├── __init__.py
│   │
│   ├── pipeline/                     # 流程编排模块
│   │   ├── orchestrator.py          # 主编排器：控制整个清洗流程
│   │   └── orchestrator1.py         # 备用编排器（可能是早期版本）
│   │
│   ├── models/                       # 模型模块
│   │   ├── center.py                # 中心预测模型（MLP）
│   │   └── quantile.py              # 分位数回归模型（可选）
│   │
│   ├── thresholds/                   # 阈值计算模块
│   │   ├── base.py                  # 阈值方法基类
│   │   ├── registry.py              # 阈值方法注册器
│   │   ├── knn_local.py             # KNN 局部阈值方法
│   │   ├── quantile_power.py        # 功率分位数阈值
│   │   └── quantile_zresid.py       # 残差分位数阈值
│   │
│   └── core/                         # 核心工具模块
│       ├── device.py                # 设备管理（CPU/GPU）
│       ├── scaler.py                # 数据归一化
│       ├── splits.py                # 数据集划分
│       ├── dmode.py                 # 密度模式处理
│       └── utils.py                 # 通用工具函数
│
├── stage2_modular.zip               # 代码压缩包（可能是归档版本）
│
├── 风机数据/                        # 数据目录（需自行准备）
│   └── 存放原始风机 CSV 数据文件
│
└── README.md                        # 本文档
```

### 模块功能详解

#### 1. **pipeline 模块**
- 负责整个数据清洗流程的编排和调度
- 处理多个站点、多个风机、多轮清洗的循环逻辑
- 管理数据集划分、模型训练、阈值计算、异常检测的完整流程

#### 2. **models 模块**
- 实现深度学习模型（主要是 MLP）
- 支持多种损失函数（MSE、加权 MSE、Huber 等）
- 提供训练和预测接口

#### 3. **thresholds 模块**
- 实现多种异常检测阈值计算方法
- 支持方法注册和动态调用
- 可扩展自定义阈值方法

#### 4. **core 模块**
- 提供数据处理的基础工具
- 设备管理、数据归一化、数据集划分等功能
- 通用工具函数（如 CSV 读取、时间统计等）

---

## 📝 使用示例

### 快速开始

1. **准备数据**：将风机数据放在 `风机数据/` 目录下
2. **修改配置**：编辑 JSON 配置文件，指定数据路径和参数
3. **运行程序**：
   ```bash
   python main.py --config your_config.json
   ```
4. **查看结果**：在配置文件指定的输出目录查看清洗后的数据

### 配置文件关键参数说明

```json
{
  "defaults": {
    "cleaning_passes": 2,           // 清洗轮数（1-3 轮推荐）
    "rho_for_clean": true,          // 清洗时是否使用空气密度
    "rho_for_model": true,          // 建模时是否使用空气密度
    "wind_scope": [0.0, 15.0],      // 风速范围（m/s）
    "device": "cuda:0",             // 计算设备（cuda:0 或 cpu）
    "seed": 42                      // 随机种子（保证可重复性）
  },
  "runs": [
    {
      "thresholds": {
        "method": "knn_local",      // 阈值方法（knn_local, quantile_power 等）
        "k": 500,                   // KNN 的 K 值
        "tau": [0.01, 0.99]         // 分位数阈值
      }
    }
  ]
}
```

---

## 📚 参考文献与相关资源

### 相关技术
- 风功率曲线建模
- 异常检测与数据清洗
- 深度学习在风电数据中的应用
- 空气密度对风机性能的影响

### 扩展阅读
- 风电 SCADA 数据质量控制方法
- 基于机器学习的风机故障诊断
- 考虑环境因素的风功率预测

---

## 👥 贡献与反馈

如有问题或建议，欢迎通过以下方式反馈：
- 提交 Issue
- 提交 Pull Request
- 联系项目维护者

---

## 📄 许可证

本项目的许可证信息请参考仓库中的 LICENSE 文件（如有）。

---

**最后更新时间**：2026-01-30
