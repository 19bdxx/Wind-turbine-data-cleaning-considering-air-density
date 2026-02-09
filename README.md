# 风机数据清洗（考虑空气密度）

基于空气密度修正的风力发电机组数据清洗系统，集成了 KNN 局部阈值方法和多种优化技术。

---

## ✨ 主要特性

- **空气密度修正**: 考虑空气密度对功率曲线的影响
- **KNN 局部阈值**: 基于局部加权分位数的异常检测
- **GPU 加速**: 
  - **支持 NVIDIA GPU（CUDA）**，大规模数据显著提速 ✅
  - GPU + 窗口筛选组合获得最佳性能
  - 自动检测并使用可用 GPU
- **CPU 优化**: 
  - KDTree 空间索引（2-4倍加速）
  - 候选集窗口筛选（50-80%候选点缩减）
- **模块化设计**: 核心功能、模型、阈值方法分离
- **批量处理**: 支持多站点、多风机、多实验方案

---

## 🚀 快速开始

### ⚡ 设备说明

**默认配置已使用 GPU！**

配置文件中已设置 `"device": "cuda:0"`，系统会自动使用 GPU 加速。

- ✅ **有 GPU**: 使用 GPU 模式（推荐，更快）
- ✅ **无 GPU**: 自动降级到 CPU 模式（仍有优化）

详细对比请参阅：[GPU_VS_CPU_GUIDE.md](GPU_VS_CPU_GUIDE.md)

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

- **[GPU_VS_CPU_GUIDE.md](GPU_VS_CPU_GUIDE.md)** - GPU vs CPU 选择指南（⭐推荐阅读）
  - GPU 和 CPU 性能对比
  - 设备配置方法
  - 根据数据规模选择设备
  - 常见问题排查

- **[USER_GUIDE.md](USER_GUIDE.md)** - 完整使用指南
  - 详细安装步骤
  - 配置参数说明
  - 运行方法和示例
  - 性能对比方法
  - 故障排查

- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - 优化技术详解
  - KDTree 空间索引原理
  - 窗口筛选算法
  - 配置参数调优
  - 性能测试结果

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
      "use_window_filter": true,     // 窗口筛选
      "window_v": 0.1,
      "window_r": 0.2
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
      "use_kdtree": true,            // KDTree优化
      "use_window_filter": true,
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
      "use_kdtree": false,
      "use_window_filter": false
    }
  }
}
```

详细配置说明请参考 [USER_GUIDE.md](USER_GUIDE.md)。

---

## 📊 性能表现

### GPU vs CPU 对比（N=50,000）

| 模式 | 耗时 | 提速比 | 推荐场景 |
|------|------|--------|----------|
| GPU（原始） | ~2.7秒 | 1.0x | 基线 |
| GPU + 窗口筛选 | ~1.5秒 | **1.8x** ✅ | **推荐**（大数据） |
| CPU + KDTree | ~0.7秒 | 3.9x | 中小数据 |

### 不同规模表现

| 数据规模 | GPU+窗口筛选 | CPU+KDTree | 推荐 |
|---------|-------------|-----------|------|
| N < 10K | 0.2秒 | 0.1秒 | 都可 |
| 10K-50K | 0.5-1.5秒 | 0.3-0.7秒 | GPU ✅ |
| 50K-100K | 1.5-4秒 | 0.7-2.5秒 | GPU ✅ |
| N > 100K | 4-20秒+ | 2.5-20秒+ | **GPU** ✅ |

详细性能测试请参考：
- [GPU_VS_CPU_GUIDE.md](GPU_VS_CPU_GUIDE.md) - 设备选择
- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - 优化详解

---

## 🔧 开发

### 代码审查

运行代码审查发现的问题已记录在 [CODE_REVIEW_ISSUES.md](CODE_REVIEW_ISSUES.md)。

### 主要优化

1. **KDTree 空间索引** - 从 O(Q×N) 降至 O(N log N + Q×K×log N)
2. **窗口筛选** - 预筛选候选集，减少距离计算
3. **批处理** - GPU/CPU 分块处理，控制内存

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

- ✅ 实现 KDTree 空间索引优化
- ✅ 实现候选集窗口筛选
- ✅ 添加配置参数到 JSON
- ✅ 创建自动化性能测试脚本
- ✅ 完善中文注释和文档

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
