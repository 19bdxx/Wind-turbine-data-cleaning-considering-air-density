# 用户使用指南

完整的风机数据清洗系统使用指南，包括安装、配置、运行和性能测试。

---

## ⚡ 重要提示：GPU 使用说明

**您的系统已配置为使用 GPU！**

配置文件中已设置：
```json
{
  "device": "cuda:0"
}
```

这意味着系统会自动使用 GPU 进行加速计算，**无需任何修改**。

- ✅ **GPU 模式是推荐配置**（默认）
- ✅ **GPU 对大规模数据更快**（N>50,000）
- ✅ **KDTree 是可选的 CPU 优化**，不影响 GPU 使用
- ✅ **窗口筛选在 GPU 模式同样有效**

详细对比请参阅：[GPU_VS_CPU_GUIDE.md](GPU_VS_CPU_GUIDE.md)

---

## 📦 环境准备

### Python 版本

**要求**: Python 3.8+（推荐 3.9 或 3.10）

### 依赖安装

#### CPU 环境（推荐用于学习和测试）

```bash
# 基础依赖
pip install numpy pandas scikit-learn

# PyTorch CPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### GPU 环境（CUDA 11.8）

```bash
# 基础依赖
pip install numpy pandas scikit-learn

# PyTorch GPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### GPU 环境（CUDA 12.1）

```bash
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 验证安装

```bash
python -c "import numpy, pandas, sklearn, torch; print('✅ 所有依赖安装成功')"
```

---

## 🚀 运行程序

### 方式1: 直接运行（最简单）

使用现有配置，优化已默认启用：

```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

**预期输出**:
```
========== 实验计划 ==========
站点数量: 1
Run 数量: 6
========== Run 1/6: rho_constant_train_mean ==========
[KNNLocal] Using CPU path | device=cpu | candidates=50000, queries=10000
[KNNLocal] Attempting KDTree optimization (d=2, metric=physics)...
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70.0% reduction)
```

### 方式2: 自动化对比测试（推荐）

自动测试多种配置并生成性能报告：

```bash
python benchmark_knn.py
```

**功能**:
- 自动创建4种配置变体（无优化、仅KDTree、窗口筛选等）
- 依次运行各场景并记录时间
- 提取性能指标（总耗时、候选筛除率、提速比）
- 生成对比表格和详细报告

**输出示例**:
```
========================================================================
性能对比总结
========================================================================

场景                      总耗时(秒)   提速比    候选筛除    状态
--------------------------------------------------------------------------------
无优化（基线）            45.23       1.00x     N/A         ✅
仅 KDTree                 28.67       1.58x     N/A         ✅
KDTree + 窗口筛选 (0.1/0.2) 18.34    2.47x     69.5%       ✅
KDTree + 窗口筛选 (0.2/0.3) 22.15    2.04x     54.3%       ✅
```

### 方式3: 自定义配置运行

#### 场景A: 禁用所有优化（基线）

创建 `config_baseline.json`（复制现有配置并修改）:

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

运行:
```bash
python main.py --config config_baseline.json
```

#### 场景B: 仅启用 KDTree

```json
{
  "defaults": {
    "thresholds": {
      "use_kdtree": true,
      "use_window_filter": false
    }
  }
}
```

#### 场景C: 启用所有优化

```json
{
  "defaults": {
    "thresholds": {
      "use_kdtree": true,
      "use_window_filter": true,
      "window_v": 0.1,
      "window_r": 0.2
    }
  }
}
```

---

## ⚙️ 配置说明

### 配置文件结构

JSON 配置文件包含三个主要部分：

```json
{
  "defaults": {
    // 全局默认参数
  },
  "stations": [
    // 风电场站点列表
  ],
  "runs": [
    // 实验方案列表
  ]
}
```

### 关键参数

#### 设备配置

```json
{
  "defaults": {
    "device": "cpu",           // 设备：cpu, cuda:0, cuda:1
    "seed": 42                 // 随机种子
  }
}
```

#### KNN 阈值参数

```json
{
  "defaults": {
    "thresholds": {
      "k_nei": 500,            // K近邻数量
      "tau_hi": 0.98,          // 上分位点
      "tau_lo": 0.98,          // 下分位点
      
      // KDTree 优化
      "use_kdtree": true,      // 启用KDTree（默认true）
      
      // 窗口筛选优化
      "use_window_filter": true,   // 启用窗口筛选（默认true）
      "window_v": 0.1,             // 风速窗口半径（标准化空间）
      "window_r": 0.2,             // 密度窗口半径（标准化空间）
      "min_candidates": 1000       // 最小候选数
    }
  }
}
```

#### 数据标准化

```json
{
  "defaults": {
    "scaler": {
      "method": "minmax",      // 标准化方法：minmax 或 zscore
      "wind_range": [0, 15],   // 风速范围（原始空间）
      "rho_range": [1.07, 1.37] // 密度范围（原始空间）
    }
  }
}
```

### 窗口筛选参数详解

#### window_v（风速窗口半径）

**默认值**: `0.1`（标准化空间）

**MinMax 归一化 [0,1]**（假设风速范围 0-15 m/s）:
- `0.05`: 约对应 ±0.75 m/s（窄窗口，激进筛选）
- `0.1`: 约对应 ±1.5 m/s（**推荐**）
- `0.15`: 约对应 ±2.25 m/s（宽窗口）

**Z-score 归一化**（假设风速标准差 3 m/s）:
- `0.3`: 约对应 0.3σ ≈ ±0.9 m/s
- `0.5`: 约对应 0.5σ ≈ ±1.5 m/s（**推荐**）
- `0.8`: 约对应 0.8σ ≈ ±2.4 m/s

#### window_r（密度窗口半径）

**默认值**: `0.2`（标准化空间）

**MinMax 归一化 [0,1]**（假设密度范围 1.07-1.37 kg/m³）:
- `0.1`: 约对应 ±0.03 kg/m³（窄窗口）
- `0.2`: 约对应 ±0.06 kg/m³（**推荐**）
- `0.3`: 约对应 ±0.09 kg/m³（宽窗口）

**Z-score 归一化**（假设密度标准差 0.05 kg/m³）:
- `0.5`: 约对应 0.5σ ≈ ±0.025 kg/m³
- `1.0`: 约对应 1.0σ ≈ ±0.05 kg/m³（**推荐**）
- `1.5`: 约对应 1.5σ ≈ ±0.075 kg/m³

#### min_candidates（最小候选数）

**默认值**: `1000`（或 `max(K×2, 1000)`）

**作用**:
- 确保筛选后有足够的候选点
- 若不足，自动扩大窗口（1.5倍递增，最多3次）
- 扩展失败则回退到全量搜索

**推荐值**:
- 通常设为 K 的 2 倍
- 对于 `k_nei=500`，推荐 `min_candidates=1000`
- 对于 `k_nei=800`，推荐 `min_candidates=1600`

---

## 📊 性能对比方法

### 方法1: 自动化测试脚本（推荐）

```bash
python benchmark_knn.py
```

**自动完成**:
1. 创建多种配置变体
2. 依次运行各场景
3. 提取性能指标
4. 生成对比报告

### 方法2: 手动对比

#### 步骤1: 准备配置文件

创建不同配置文件：
- `config_baseline.json`: 无优化
- `config_kdtree.json`: 仅KDTree
- `config_full.json`: 完全优化

#### 步骤2: 依次运行并记录时间

```bash
# 基线
time python main.py --config config_baseline.json 2>&1 | tee log_baseline.txt

# 优化版
time python main.py --config config_full.json 2>&1 | tee log_full.txt
```

#### 步骤3: 提取关键指标

```bash
# 查看窗口筛选效果
grep "reduction" log_full.txt

# 查看 KNN 相关信息
grep "KNNLocal" log_full.txt
```

### 关键性能指标

#### 1. 候选筛除率

**日志标记**: `avg candidates X/N (Y% reduction)`

**含义**: 窗口筛选后剩余候选点比例

**示例**:
```
[KNNLocal] Window filtering: avg candidates 15000/50000 (70% reduction)
```
表示筛除了 70% 的候选点

**典型值**: 50%-80%

#### 2. 总运行时间

使用 `time` 命令测量完整流程耗时

#### 3. 提速比

**计算**: `baseline_time / optimized_time`

**示例**: 基线45秒，优化18秒 → 提速 2.5x

#### 4. KNN 计算时间

如果单独打印，最直接反映优化效果

### 预期效果

| 数据规模 | 场景 | 候选缩减 | 预期提速 |
|---------|------|---------|---------|
| N < 10K | KDTree | - | 1.0-1.5x |
| N < 10K | KDTree + 窗口 | 50%-70% | 0.8-1.2x |
| 10K-50K | KDTree | - | 1.5-2.5x |
| 10K-50K | KDTree + 窗口 | 60%-80% | 1.2-2.0x |
| N > 50K | KDTree | - | 2.5-4.0x |
| N > 50K | KDTree + 窗口 | 70%-90% | 2.0-4.0x |

**注意**:
- 小数据集时，窗口筛选的开销可能抵消收益
- 大数据集时，优化效果更明显
- GPU 模式暂不支持窗口筛选

---

## 🔍 日志分析

### 查看优化是否启用

#### KDTree 优化

```
[KNNLocal] Attempting KDTree optimization (d=2, metric=physics)...
[KNNLocal] KDTree search successful! Processing 500 queries with 500 neighbors each.
```

#### 窗口筛选

```
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70.0% reduction)
```

#### 数据规模

```
[KNNLocal] Using CPU path | device=cpu | candidates=50000, queries=10000
```

### 提取性能指标

```bash
# 运行并保存日志
python main.py --config config.json 2>&1 | tee run.log

# 提取候选筛除率
grep "reduction" run.log

# 提取 KNN 相关时间
grep -E "KNN|elapsed|耗时" run.log

# 查看窗口扩展情况
grep "扩展窗口" run.log
```

---

## 🛠️ 故障排查

### 问题1: ModuleNotFoundError

**错误**:
```
ModuleNotFoundError: No module named 'numpy'
```

**解决**:
```bash
pip install numpy pandas scikit-learn torch
```

### 问题2: CUDA 不可用

**错误**:
```
[KNNLocal] device='cuda' but torch.cuda.is_available()==False
```

**解决**:
- 方案1: 安装 GPU 版本 PyTorch
- 方案2: 修改配置使用 CPU
  ```json
  {"defaults": {"device": "cpu"}}
  ```

### 问题3: 数据文件未找到

**错误**:
```
FileNotFoundError: 风机数据/JSMZS_宽表.csv
```

**解决**:
- 检查配置文件中的 `csv` 路径
- 确保数据文件存在
- 使用绝对路径或相对于 main.py 的路径

### 问题4: 窗口筛选效果不明显

**症状**: 日志显示 "reduction: 5%"，筛选率很低

**原因**: 窗口太大

**解决**:
```json
{
  "thresholds": {
    "window_v": 0.05,  // 减小窗口
    "window_r": 0.1
  }
}
```

### 问题5: 频繁窗口扩展

**症状**: 日志显示多次 "扩展窗口"

**原因**: 窗口太小或数据稀疏

**解决**:
```json
{
  "thresholds": {
    "window_v": 0.15,  // 增大初始窗口
    "window_r": 0.25,
    "min_candidates": 500  // 或减小最小候选数
  }
}
```

### 问题6: 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
```json
{
  "defaults": {
    "knn_batch_q": 8192,      // 减小查询批大小
    "knn_train_chunk": 65536, // 减小训练分块
    "gpu_cache_mib": 12288    // 减小 GPU 缓存
  }
}
```

---

## 🎯 调优指南

### 步骤1: 使用默认值开始

```json
{
  "thresholds": {
    "use_window_filter": true,
    "window_v": 0.1,
    "window_r": 0.2,
    "min_candidates": 1000
  }
}
```

### 步骤2: 运行并观察日志

查找:
```
[KNNLocal] Window filtering: avg candidates 15000/50000 (70% reduction)
```

### 步骤3: 根据效果调整

| 观察到的现象 | 调整建议 | 目标 |
|-------------|---------|------|
| 筛除率 < 30% | 减小 window_v, window_r (×0.7) | 提高筛选率 |
| 筛除率 > 90% | 增大 window_v, window_r (×1.5) | 避免过度筛选 |
| 频繁窗口扩展 | 增大初始窗口或 min_candidates | 减少扩展次数 |
| 性能无提升 | 检查数据规模；考虑禁用（小数据集） | - |

### 步骤4: 多次测试

- 在不同数据集上测试
- 对比优化前后的性能
- 确认结果一致性

---

## 📝 典型配置示例

### 配置1: 默认推荐（平衡）

```json
{
  "defaults": {
    "device": "cpu",
    "thresholds": {
      "k_nei": 500,
      "use_window_filter": true,
      "window_v": 0.1,
      "window_r": 0.2,
      "min_candidates": 1000
    }
  }
}
```

**适用**: 大多数场景，平衡筛选效果和稳定性

### 配置2: 激进筛选（追求最大加速）

```json
{
  "defaults": {
    "thresholds": {
      "k_nei": 500,
      "use_window_filter": true,
      "window_v": 0.05,      // 窄窗口
      "window_r": 0.1,       // 窄窗口
      "min_candidates": 500  // 较小的最小候选数
    }
  }
}
```

**适用**: 数据密度高，追求最大性能提升

**风险**: 可能频繁触发窗口扩展

### 配置3: 保守筛选（追求稳定）

```json
{
  "defaults": {
    "thresholds": {
      "k_nei": 500,
      "use_window_filter": true,
      "window_v": 0.2,        // 宽窗口
      "window_r": 0.3,        // 宽窗口
      "min_candidates": 1500  // 较大的最小候选数
    }
  }
}
```

**适用**: 数据分布不均匀，追求稳定性

**特点**: 很少触发扩展，但筛选效果可能有限

### 配置4: 禁用优化（基线对比）

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

**适用**: 用于性能对比，测试优化效果

---

## 🧪 测试

### 功能测试

```bash
# 测试窗口筛选功能
python test_window_filtering.py

# 测试 KDTree 优化
python test_knn_optimization.py
```

### 性能测试

```bash
# 自动化性能测试
python benchmark_knn.py

# 指定基础配置
python benchmark_knn.py --config my_config.json

# 跳过基线测试（加速）
python benchmark_knn.py --skip-baseline
```

---

## 📚 相关文档

- [README.md](README.md) - 项目简介和快速开始
- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - 优化技术详解
- [CODE_REVIEW_ISSUES.md](CODE_REVIEW_ISSUES.md) - 代码审查结果

---

## ✅ 运行前检查清单

必需：
- [ ] Python 3.8+ 已安装
- [ ] 所有依赖已安装（numpy, pandas, sklearn, torch）
- [ ] 配置文件路径正确
- [ ] 数据文件存在且路径正确
- [ ] 设备配置匹配（CPU/CUDA）

可选：
- [ ] 准备多个配置文件用于对比
- [ ] 确认输出目录有写权限
- [ ] 启用日志记录（`tee` 或重定向）

---

**文档更新时间**: 2026-02-09  
**版本**: 1.0
