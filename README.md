# 风机数据清洗（考虑空气密度）

基于空气密度修正的风力发电机组数据清洗系统，集成了 KNN 局部阈值方法和多种优化技术。

---

## ✨ 主要特性

- **空气密度修正**: 考虑空气密度对功率曲线的影响
- **KNN 局部阈值**: 基于局部加权分位数的异常检测
- **性能优化**: 
  - KDTree 空间索引（2-4倍加速）
  - 候选集窗口筛选（50-80%候选点缩减）
- **模块化设计**: 核心功能、模型、阈值方法分离
- **批量处理**: 支持多站点、多风机、多实验方案

---

## 🚀 快速开始

### 1. 安装依赖

**CPU 版本**（推荐用于测试）:
```bash
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本**（CUDA 11.8）:
```bash
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cu118
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

日志中会显示：
```
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70% reduction)
```

---

## 📖 文档

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

### 基本配置

```json
{
  "defaults": {
    "device": "cpu",
    "thresholds": {
      "k_nei": 500,
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

| 数据规模 | 优化方式 | 候选缩减 | 提速比 |
|---------|---------|---------|--------|
| 5K | KDTree | - | 0.77x |
| 20K | KDTree + 窗口筛选 | 54-70% | 1.66x |
| 50K | KDTree + 窗口筛选 | 70%+ | 3.62x |

详细性能测试请参考 [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)。

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
