# KNN 优化运行与性能验证指南

本文档提供完整的步骤，帮助您运行风机数据清洗程序并验证 KNN 优化（KDTree 加速 + 候选集窗口筛选）的性能效果。

---

## 1️⃣ 入口文件与运行方式

### 1.1 主入口文件

**文件**: `main.py`

这是整个数据清洗流程的主入口，负责：
- 读取 JSON 配置文件
- 批量处理多个风电场站点
- 执行多个实验方案（runs）
- 管理数据切分的复用

### 1.2 基本运行命令

```bash
python main.py --config <配置文件路径>
```

**示例**:
```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

### 1.3 配置文件结构

JSON 配置文件包含三个主要部分：

```json
{
  "defaults": {
    // 全局默认参数（模型、阈值、标准化等）
  },
  "stations": [
    // 风电场站点列表
    {
      "name": "站点名称",
      "csv": "数据文件路径.csv",
      "turbine_start": 51,
      "turbine_end": 58
    }
  ],
  "runs": [
    // 实验方案列表，每个可覆盖 defaults
    {
      "name": "实验方案名称",
      "rho_for_clean": "value",
      // ... 其他参数覆盖 ...
    }
  ]
}
```

---

## 2️⃣ KNN 相关配置参数

### 2.1 参数位置

KNN 相关参数位于配置文件的 **`defaults.thresholds`** 部分：

```json
{
  "defaults": {
    "thresholds": {
      // === 基础 KNN 参数 ===
      "k_nei": 500,              // K 近邻数量
      "tau_hi": 0.98,            // 上分位点
      "tau_lo": 0.98,            // 下分位点
      
      // === KDTree 优化参数 ===
      "use_kdtree": true,        // 是否启用 KDTree 加速（默认 true）
      
      // === 窗口筛选参数（新增）===
      "use_window_filter": true, // 是否启用窗口筛选（默认 true）
      "window_v": 0.1,           // 风速窗口半径（标准化空间）
      "window_r": 0.2,           // 密度窗口半径（标准化空间）
      "min_candidates": 1000     // 最小候选数（默认 max(K×2, 1000)）
    }
  }
}
```

### 2.2 参数说明

#### KDTree 优化
- **use_kdtree** (默认 `true`):
  - 启用时使用空间索引加速 K 近邻搜索
  - 复杂度从 O(Q×N) 降至 O(N log N + Q×K×log N)
  - 适用于 CPU 模式 + 低维特征

#### 窗口筛选优化
- **use_window_filter** (默认 `true`):
  - 在计算距离前，根据特征范围预筛选候选集
  - 将搜索空间从 N 降至 M (M << N)
  - 理想情况下可筛除 50%-80% 的候选点

- **window_v** (默认 `0.1`):
  - 风速窗口半径（在标准化空间）
  - MinMax [0,1]: 0.1 约对应原始空间 1.5 m/s（假设范围 15m/s）
  - Z-score: 0.5 约对应 0.5 个标准差

- **window_r** (默认 `0.2`):
  - 空气密度窗口半径（在标准化空间）
  - MinMax [0,1]: 0.2 约对应原始空间 0.06 kg/m³（假设范围 0.3）
  - Z-score: 1.0 约对应 1.0 个标准差

- **min_candidates** (默认 `max(K×2, 1000)`):
  - 筛选后的最小候选数
  - 若不足会自动扩大窗口（1.5倍递增，最多3次）
  - 确保有足够的点进行 KNN

### 2.3 其他相关参数

```json
{
  "defaults": {
    "device": "cuda:0",          // 设备：cuda:0, cuda:1, cpu
    "knn_batch_q": 16384,        // 查询批大小（GPU模式）
    "knn_train_chunk": 131072,   // 训练分块大小（GPU模式）
    
    "scaler": {
      "method": "minmax",        // 标准化方法：minmax 或 zscore
      "wind_range": [0, 15],     // 风速范围（原始空间）
      "rho_range": [1.07, 1.37]  // 密度范围（原始空间）
    }
  }
}
```

---

## 3️⃣ 依赖安装与环境准备

### 3.1 Python 版本

**要求**: Python 3.8+（推荐 3.9 或 3.10）

### 3.2 安装依赖

#### 方法1: 使用 pip（推荐）

```bash
# 基础依赖
pip install numpy pandas scikit-learn

# PyTorch（根据您的环境选择）
# CPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU 版本（CUDA 11.8）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# GPU 版本（CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 方法2: 一键安装（推荐复制粘贴）

**CPU 环境**:
```bash
pip install numpy pandas scikit-learn && \
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU 环境（CUDA 11.8）**:
```bash
pip install numpy pandas scikit-learn && \
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 验证安装

```bash
python -c "import numpy, pandas, sklearn, torch; print('All packages installed successfully')"
```

### 3.4 数据准备

确保配置文件中指定的数据文件存在：
- 检查 `stations[].csv` 路径是否正确
- 数据文件应包含必要的列（风速、功率、空气密度等）

---

## 4️⃣ 运行示例

### 4.1 快速测试（使用现有配置）

```bash
# 使用默认配置运行（启用所有优化）
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

**预期输出**:
```
========== 实验计划 ==========
站点数量: 1
  [1] JSMZS  CSV=风机数据/JSMZS_宽表.csv  turbines=51..58
Run 数量: 6
  [1] rho_constant_train_mean ...
  ...
========== Run 1/6: rho_constant_train_mean ==========
...
[KNNLocal] Using CPU path | device=cpu | candidates=50000, queries=10000
[KNNLocal] Attempting KDTree optimization (d=2, metric=physics)...
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70.0% reduction)
...
```

### 4.2 自定义配置运行

#### 场景1: 禁用所有优化（基线）

创建配置文件 `config_baseline.json`（复制并修改 `defaults.thresholds`）:
```json
{
  "defaults": {
    // ... 其他配置保持不变 ...
    "thresholds": {
      // ... 其他参数保持不变 ...
      "use_kdtree": false,        // ❌ 禁用 KDTree
      "use_window_filter": false  // ❌ 禁用窗口筛选
    }
  }
}
```

运行:
```bash
python main.py --config config_baseline.json
```

#### 场景2: 仅启用 KDTree

```json
{
  "thresholds": {
    "use_kdtree": true,         // ✅ 启用 KDTree
    "use_window_filter": false  // ❌ 禁用窗口筛选
  }
}
```

#### 场景3: 启用所有优化

```json
{
  "thresholds": {
    "use_kdtree": true,         // ✅ 启用 KDTree
    "use_window_filter": true,  // ✅ 启用窗口筛选
    "window_v": 0.1,
    "window_r": 0.2
  }
}
```

### 4.3 调整窗口大小

**测试不同窗口大小的影响**:

```json
{
  "thresholds": {
    // 窄窗口（激进筛选，可能需要扩展）
    "window_v": 0.05,
    "window_r": 0.1,
    
    // 中等窗口（推荐，平衡筛选率和扩展次数）
    "window_v": 0.1,
    "window_r": 0.2,
    
    // 宽窗口（保守筛选，很少扩展）
    "window_v": 0.2,
    "window_r": 0.3
  }
}
```

---

## 5️⃣ 性能对比方法

### 5.1 查看运行日志

程序运行时会打印详细的时间统计和优化信息。

#### 关键日志标记

**1. KNN 方法选择**:
```
[KNNLocal] Using CPU path | device=cpu | candidates=50000, queries=10000
[KNNLocal] Attempting KDTree optimization (d=2, metric=physics)...
```

**2. 窗口筛选效果**:
```
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70.0% reduction)
```
- `avg candidates`: 平均候选数 / 总数
- `reduction`: 筛选百分比

**3. KDTree 搜索成功**:
```
[KNNLocal] KDTree search successful! Processing 500 queries with 500 neighbors each.
[KNNLocal] KDTree path completed successfully.
```

### 5.2 使用内置计时器

程序内部使用 `Stopwatch` 类记录各阶段耗时。查找日志中的时间统计：

```
[Stopwatch] KNN computation: 12.34 seconds
[Stopwatch] Threshold calculation: 5.67 seconds
[Stopwatch] Total run time: 45.23 seconds
```

### 5.3 提取性能指标

#### 方法1: 使用 grep 提取关键指标

```bash
# 运行并保存日志
python main.py --config config.json 2>&1 | tee run.log

# 提取 KNN 相关时间
grep -E "KNN|window filtering|candidates" run.log

# 提取总时间
grep -E "Total|总耗时|Elapsed" run.log
```

#### 方法2: 结构化对比

创建脚本 `extract_metrics.sh`:
```bash
#!/bin/bash

echo "=== 场景1: 无优化 ==="
python main.py --config config_baseline.json 2>&1 | grep -E "candidates|KNN|Total" | tee metrics_baseline.txt

echo "=== 场景2: 仅 KDTree ==="
python main.py --config config_kdtree.json 2>&1 | grep -E "candidates|KNN|Total" | tee metrics_kdtree.txt

echo "=== 场景3: KDTree + 窗口筛选 ==="
python main.py --config config_full.json 2>&1 | grep -E "candidates|KNN|Total" | tee metrics_full.txt
```

运行:
```bash
chmod +x extract_metrics.sh
./extract_metrics.sh
```

### 5.4 Python 计时脚本

创建 `benchmark_knn.py` 用于自动化测试：

```python
#!/usr/bin/env python3
import time
import subprocess
import re

def run_config(config_file, name):
    """运行配置并提取时间"""
    print(f"\n{'='*60}")
    print(f"运行: {name}")
    print(f"配置: {config_file}")
    print(f"{'='*60}\n")
    
    start = time.time()
    result = subprocess.run(
        ['python', 'main.py', '--config', config_file],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    # 提取关键指标
    output = result.stdout + result.stderr
    
    # 查找窗口筛选信息
    window_match = re.search(r'avg candidates (\d+)/(\d+) \((\d+\.\d+)% reduction\)', output)
    if window_match:
        avg_cand, total, reduction = window_match.groups()
        print(f"窗口筛选: {avg_cand}/{total} 候选 ({reduction}% 筛除)")
    
    # 查找 KNN 时间（如果有打印）
    knn_match = re.search(r'KNN.*?(\d+\.\d+)\s*(?:秒|seconds)', output)
    if knn_match:
        knn_time = knn_match.group(1)
        print(f"KNN 耗时: {knn_time}秒")
    
    print(f"总耗时: {elapsed:.2f}秒")
    
    return {
        'name': name,
        'total_time': elapsed,
        'avg_candidates': int(avg_cand) if window_match else None,
        'reduction': float(reduction) if window_match else 0.0
    }

if __name__ == '__main__':
    results = []
    
    # 场景1: 无优化
    results.append(run_config(
        'config_baseline.json',
        '无优化（基线）'
    ))
    
    # 场景2: 仅 KDTree
    results.append(run_config(
        'config_kdtree.json',
        '仅 KDTree'
    ))
    
    # 场景3: KDTree + 窗口筛选
    results.append(run_config(
        'config_full.json',
        'KDTree + 窗口筛选'
    ))
    
    # 输出对比表格
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}\n")
    
    baseline = results[0]['total_time']
    
    print(f"{'场景':<20} {'总耗时(秒)':<15} {'提速比':<10} {'候选筛除'}")
    print("-" * 60)
    
    for r in results:
        speedup = baseline / r['total_time']
        reduction = f"{r['reduction']:.1f}%" if r['reduction'] else "N/A"
        print(f"{r['name']:<20} {r['total_time']:<15.2f} {speedup:<10.2f}x {reduction}")
```

运行:
```bash
python benchmark_knn.py
```

### 5.5 性能指标说明

#### 关键指标

1. **候选集缩减率**:
   - 公式: `(1 - M/N) × 100%`
   - 典型值: 50%-80%
   - 来源: 日志中的 "window filtering: ... reduction"

2. **总运行时间**:
   - 整个数据清洗流程的端到端时间
   - 包含数据加载、模型训练、KNN 计算、结果输出

3. **KNN 计算时间** (如果单独打印):
   - 仅 KNN 局部阈值计算的时间
   - 最能反映优化效果

4. **提速比**:
   - 公式: `baseline_time / optimized_time`
   - 示例: 2.0x 表示快了一倍

#### 预期效果

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

## 6️⃣ 故障排查

### 6.1 常见问题

#### Q1: ModuleNotFoundError

**错误**:
```
ModuleNotFoundError: No module named 'numpy'
```

**解决**:
```bash
pip install numpy pandas scikit-learn torch
```

#### Q2: CUDA 不可用

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

#### Q3: 数据文件未找到

**错误**:
```
FileNotFoundError: 风机数据/JSMZS_宽表.csv
```

**解决**:
- 检查配置文件中的 `csv` 路径
- 确保数据文件存在
- 使用绝对路径或相对于 main.py 的路径

#### Q4: 窗口筛选效果不明显

**症状**: 日志显示 "reduction: 5%"，筛选率很低

**解决**:
```json
{
  "thresholds": {
    // 减小窗口
    "window_v": 0.05,
    "window_r": 0.1
  }
}
```

#### Q5: 频繁窗口扩展

**症状**: 日志显示多次 "扩展窗口"

**解决**:
```json
{
  "thresholds": {
    // 增大初始窗口
    "window_v": 0.15,
    "window_r": 0.25,
    // 或减小最小候选数
    "min_candidates": 500
  }
}
```

### 6.2 调试模式

在配置中启用调试输出：
```json
{
  "defaults": {
    "debug": {
      "dump_knn_diag": true,
      "knn_diag_points": 100
    }
  }
}
```

---

## 7️⃣ 高级用法

### 7.1 仅测试 KNN 性能

如果只想测试 KNN 部分（不运行完整流程），可以使用提供的测试脚本：

```bash
# 测试窗口筛选
python test_window_filtering.py

# 测试 KDTree 优化
python test_knn_optimization.py
```

### 7.2 大规模数据测试

对于大规模数据（N > 100,000），调整批处理参数：

```json
{
  "defaults": {
    "knn_batch_q": 8192,       // 减小查询批大小
    "knn_train_chunk": 65536,  // 减小训练分块
    "gpu_cache_mib": 12288     // 减小 GPU 缓存
  }
}
```

### 7.3 多风机并行测试

修改配置文件的 `stations` 和 `turbine_start/end` 来测试不同数量的风机：

```json
{
  "stations": [
    {
      "name": "JSMZS",
      "csv": "风机数据/JSMZS_宽表.csv",
      "turbine_start": 51,
      "turbine_end": 52  // 仅测试2台风机
    }
  ]
}
```

### 7.4 输出性能报告

将日志保存并生成报告：

```bash
# 运行并保存日志
python main.py --config config.json 2>&1 | tee performance.log

# 提取关键指标
echo "=== KNN 性能指标 ===" > report.txt
grep -E "candidates|reduction|KDTree|window" performance.log >> report.txt
echo "" >> report.txt
echo "=== 总体时间 ===" >> report.txt
grep -E "Total|总耗时|Elapsed" performance.log >> report.txt

# 查看报告
cat report.txt
```

---

## 8️⃣ 快速检查清单

运行前确认：

- [ ] Python 3.8+ 已安装
- [ ] 所有依赖已安装（numpy, pandas, sklearn, torch）
- [ ] 配置文件路径正确
- [ ] 数据文件存在且路径正确
- [ ] 设备配置匹配（CPU/CUDA）

验证优化时：

- [ ] 准备至少2-3个配置文件（无优化、KDTree、完全优化）
- [ ] 启用日志记录（`tee` 或重定向）
- [ ] 记录关键指标（候选数、缩减率、总时间）
- [ ] 对比多次运行结果（消除偶然性）

---

## 📚 参考资料

- `KNN_OPTIMIZATION_REPORT.md` - KDTree 优化详细分析
- `WINDOW_FILTERING_OPTIMIZATION.md` - 窗口筛选优化详细分析
- `test_window_filtering.py` - 窗口筛选功能测试
- `test_knn_optimization.py` - KDTree 优化测试

---

**最后更新**: 2026-02-09  
**版本**: 1.0  
**作者**: GitHub Copilot
