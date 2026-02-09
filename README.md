# 风机数据清洗（考虑空气密度）

基于空气密度修正的风力发电机组数据清洗系统，集成了 KNN 局部阈值方法和多种优化技术。

---

## ✨ 主要特性

- **空气密度修正**: 考虑空气密度对功率曲线的影响
- **KNN 局部阈值**: 基于局部加权分位数的异常检测
- **窗口筛选优化**: 
  - **主要优化方法**：通过风速和空气密度范围预筛选候选点 ⭐
  - 50-80% 候选点缩减，显著减少计算量
  - 简单直观，物理意义明确
  - GPU 和 CPU 都支持
- **GPU 加速**: 
  - **支持 NVIDIA GPU（CUDA）**，大规模数据显著提速 ✅
  - GPU + 窗口筛选组合获得最佳性能
  - 自动检测并使用可用 GPU
- **模块化设计**: 核心功能、模型、阈值方法分离
- **批量处理**: 支持多站点、多风机、多实验方案

---

## 🚀 快速开始

### ⚡ 设备说明

**默认配置已使用 GPU！**

配置文件中已设置 `"device": "cuda:0"`，系统会自动使用 GPU 加速。

- ✅ **有 GPU**: 使用 GPU 模式（推荐，更快）
- ✅ **无 GPU**: 自动降级到 CPU 模式（仍有优化）

### 1. 安装依赖

**GPU 版本**（推荐，CUDA 11.8）:
```bash
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**GPU 版本**（CUDA 12.1+）:
```bash
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CPU 版本**（备用）:
```bash
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. 运行程序

**基本运行**:
```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

**自动化性能测试**（推荐）:
```bash
python benchmark_knn.py
```

### 3. 查看结果

日志中会显示设备和优化信息：
```
[Device] torch.cuda.is_available()=True; GPUs=1; using=cuda:0; name=NVIDIA GeForce RTX 3090
[KNNLocal] Using GPU path | device=cuda:0 | candidates=50000, queries=10000
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70% reduction)
```

---

## 📖 文档

- **[USER_GUIDE.md](USER_GUIDE.md)** - 完整使用指南（⭐推荐阅读）
  - 详细安装步骤
  - 配置参数说明（窗口筛选等）
  - 运行方法和示例
  - 性能对比和调优
  - 故障排查

- **[CODE_REVIEW_ISSUES.md](CODE_REVIEW_ISSUES.md)** - 代码审查结果
  - 已识别的问题
  - 修复建议
  - 开发者参考

---

## 📁 项目结构

```
.
├── main.py                          # 主入口
├── stage2_modular/
│   ├── core/                        # 核心工具
│   │   ├── device.py               # 设备管理
│   │   ├── utils.py                # 工具函数
│   │   ├── scaler.py               # 数据标准化
│   │   ├── splits.py               # 数据切分
│   │   └── dmode.py                # D尺度计算
│   ├── models/                      # 模型
│   │   ├── center.py               # MLP中心模型
│   │   └── quantile.py             # 分位数回归
│   ├── thresholds/                  # 阈值方法
│   │   ├── knn_local.py            # KNN局部阈值（含优化）
│   │   ├── quantile_power.py       # 功率分位阈值
│   │   └── quantile_zresid.py      # Z残差分位阈值
│   └── pipeline/                    # 流程编排
│       └── orchestrator.py         # 主编排器
├── benchmark_knn.py                 # 性能测试脚本
└── experiments_*.json               # 配置文件
```

---

## ⚙️ 配置示例

### GPU 配置（推荐 - 默认）

```json
{
  "defaults": {
    "device": "cuda:0",              // 使用GPU
    "thresholds": {
      "k_nei": 500,
      "use_window_filter": true,     // 窗口筛选（主要优化）
      "window_v": 0.1,               // 风速窗口
      "window_r": 0.2                // 密度窗口
    }
  }
}
```

### CPU 配置（备用）

```json
{
  "defaults": {
    "device": "cpu",                 // 使用CPU
    "thresholds": {
      "k_nei": 500,
      "use_window_filter": true,     // 窗口筛选
      "window_v": 0.1,
      "window_r": 0.2
    }
  }
}
```

### 禁用优化（基线对比）

```json
{
  "defaults": {
    "thresholds": {
      "use_window_filter": false     // 禁用窗口筛选
    }
  }
}
```

详细配置说明请参考 [USER_GUIDE.md](USER_GUIDE.md)。

---

## 📊 性能表现

### 窗口筛选效果（N=50,000）

| 模式 | 耗时 | 提速比 | 候选筛除率 |
|------|------|--------|-----------|
| GPU（原始） | ~2.7秒 | 1.0x | 0% |
| **GPU + 窗口筛选** | **~1.5秒** | **1.8x** ✅ | **70%** |

### 不同规模表现

| 数据规模 | GPU+窗口筛选 | 推荐 |
|---------|-------------|------|
| N < 10K | 0.2秒 | ✅ |
| 10K-50K | 0.5-1.5秒 | ✅ |
| 50K-100K | 1.5-4秒 | **✅ 推荐** |
| N > 100K | 4-20秒+ | **✅ 推荐** |

详细说明请参考 [USER_GUIDE.md](USER_GUIDE.md)。

---

## 🔧 开发

### 代码审查

运行代码审查发现的问题已记录在 [CODE_REVIEW_ISSUES.md](CODE_REVIEW_ISSUES.md)。

### 主要优化

1. **窗口筛选** - 通过风速和空气密度范围预筛选候选点（主要优化方法）⭐
   - 50-80% 候选点缩减
   - 简单直观，物理意义明确
   - GPU 和 CPU 都支持

2. **批处理** - GPU/CPU 分块处理，控制内存

### 测试

```bash
# 测试窗口筛选功能
python test_window_filtering.py

# 测试 KDTree 优化
python test_knn_optimization.py
```

---

## 📝 更新日志

### 2026-02-09

- ✅ 实现候选集窗口筛选（主要优化）
- ✅ 添加配置参数到 JSON
- ✅ 创建自动化性能测试脚本
- ✅ 完善中文注释和文档
- ✅ 精简文档文件（10个→3个）

### 历史提交

- 修复 orchestrator.py 切分保存 bug
- 添加详细中文注释到所有核心模块
- 代码审查，识别19个潜在问题

---

## 📄 许可证

本项目基于原有代码进行优化和文档完善。

---

## 🤝 贡献

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**文档更新时间**: 2026-02-09
