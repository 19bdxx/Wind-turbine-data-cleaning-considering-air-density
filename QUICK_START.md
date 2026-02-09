# KNN 优化验证 - 快速参考

## 🚀 快速回答

### 1) 入口文件

**主入口**: `main.py`

**运行命令**:
```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

---

### 2) 配置文件参数

#### 无需修改即可使用

当前代码已默认启用所有优化：
- KDTree 优化: 默认 `True`
- 窗口筛选: 默认 `True`，窗口大小 `window_v=0.1`, `window_r=0.2`

#### 可选：手动配置

在 JSON 配置文件中添加/修改（位置：`defaults.thresholds`）:

```json
{
  "defaults": {
    "thresholds": {
      "k_nei": 500,
      
      "use_kdtree": true,           // KDTree 优化（默认 true）
      
      "use_window_filter": true,    // 窗口筛选（默认 true）
      "window_v": 0.1,              // 风速窗口（标准化空间）
      "window_r": 0.2,              // 密度窗口（标准化空间）
      "min_candidates": 1000        // 最小候选数（默认 max(K*2, 1000)）
    }
  }
}
```

**禁用优化（用于对比）**:
```json
{
  "thresholds": {
    "use_kdtree": false,
    "use_window_filter": false
  }
}
```

---

### 3) 完整运行命令

#### 步骤1: 安装依赖

**CPU 版本（推荐用于测试）**:
```bash
pip install numpy pandas scikit-learn && \
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本（CUDA 11.8）**:
```bash
pip install numpy pandas scikit-learn && \
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 步骤2: 验证安装

```bash
python -c "import numpy, pandas, sklearn, torch; print('✅ 依赖安装成功')"
```

#### 步骤3: 运行程序

**方式A: 直接运行（使用默认优化）**
```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

**方式B: 自动化基准测试（推荐）**
```bash
python benchmark_knn.py
```

这将自动测试4种场景：
1. 无优化（基线）
2. 仅 KDTree
3. KDTree + 窗口筛选（默认窗口）
4. KDTree + 窗口筛选（宽窗口）

---

### 4) 性能对比方法

#### 方法1: 自动化测试（最简单）

```bash
python benchmark_knn.py
```

**输出示例**:
```
========================================================================
性能对比总结
========================================================================

场景                      总耗时(秒)   提速比    候选筛除    状态
--------------------------------------------------------------------------------
无优化（基线）            45.23       1.00x     N/A         ✅
仅 KDTree                 28.67       1.58x     N/A         ✅
KDTree + 窗口筛选         18.34       2.47x     69.5%       ✅
```

**优势**:
- 自动运行多种配置
- 自动提取性能指标
- 生成对比报告

#### 方法2: 手动对比

**步骤1**: 准备3个配置文件

创建 `config_baseline.json`（无优化）:
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

创建 `config_optimized.json`（完全优化）:
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

**步骤2**: 依次运行并记录时间

```bash
# 基线
time python main.py --config config_baseline.json 2>&1 | tee log_baseline.txt

# 优化版
time python main.py --config config_optimized.json 2>&1 | tee log_optimized.txt
```

**步骤3**: 提取关键指标

```bash
# 从日志中提取候选筛除率
grep "reduction" log_optimized.txt

# 从日志中提取KNN信息
grep "KNNLocal" log_optimized.txt

# 查看总时间（time命令输出）
```

#### 关键性能指标

**1. 候选筛除率**
- 日志标记: `avg candidates X/N (Y% reduction)`
- 含义: 窗口筛选后剩余候选点比例
- 典型值: 30%-80% 筛除
- 示例: `avg candidates 15000/50000 (70% reduction)`

**2. 总运行时间**
- 使用 `time` 命令测量
- 或从日志中查找 "Total time" / "总耗时"
- 对比基线与优化版的差异

**3. 提速比**
- 计算: `baseline_time / optimized_time`
- 示例: 基线45秒，优化18秒 → 提速 2.5x

**4. KNN 计算时间**（如果单独打印）
- 日志标记: "KNN computation" / "KNN 计算"
- 最直接反映优化效果

#### 查看日志的关键点

```bash
# 检查是否启用 KDTree
grep "Attempting KDTree optimization" log.txt

# 检查是否启用窗口筛选
grep "Using window filtering" log.txt

# 查看窗口筛选效果
grep "Window filtering.*reduction" log.txt

# 查看数据规模
grep "candidates=.*queries=" log.txt
```

---

## 📊 预期效果参考

| 数据规模 | 配置 | 候选筛除 | 预期提速 |
|---------|------|---------|---------|
| N < 10K | KDTree | - | 1.0-1.5x |
| N < 10K | KDTree + 窗口 | 50%-70% | 0.8-1.2x |
| 10K-50K | KDTree | - | 1.5-2.5x |
| 10K-50K | KDTree + 窗口 | 60%-80% | 1.2-2.0x |
| N > 50K | KDTree | - | 2.5-4.0x |
| N > 50K | KDTree + 窗口 | 70%-90% | 2.0-4.0x |

**注意**: 
- 小数据集时窗口筛选可能无加速效果
- GPU 模式暂不支持窗口筛选
- 效果取决于数据分布的局部性

---

## 🔧 故障排查

### 问题1: 模块未找到
```
ModuleNotFoundError: No module named 'numpy'
```
**解决**: 
```bash
pip install numpy pandas scikit-learn torch
```

### 问题2: 数据文件未找到
```
FileNotFoundError: 风机数据/JSMZS_宽表.csv
```
**解决**: 
- 检查配置文件中的 `csv` 路径
- 确保数据文件存在

### 问题3: CUDA 不可用
```
device='cuda' but torch.cuda.is_available()==False
```
**解决**: 
- 修改配置: `"device": "cpu"`
- 或安装 GPU 版 PyTorch

### 问题4: 窗口筛选效果不明显
**症状**: `reduction: 5%`

**解决**: 减小窗口
```json
{
  "window_v": 0.05,
  "window_r": 0.1
}
```

---

## 📚 详细文档

- `RUN_BENCHMARK_GUIDE.md` - 完整运行指南（12000字）
- `WINDOW_FILTERING_OPTIMIZATION.md` - 窗口筛选优化详解
- `KNN_OPTIMIZATION_REPORT.md` - KDTree 优化详解

---

**最后更新**: 2026-02-09  
**快速参考版本**: 1.0
