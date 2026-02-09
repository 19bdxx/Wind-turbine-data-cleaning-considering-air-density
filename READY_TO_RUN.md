# 确认：运行 KNN 优化对比的准备工作

## ✅ 简短回答

**是的！** 您只需要使用当前更新的 `knn_local.py` 和现有的 JSON 配置文件，就可以直接运行对比了。

---

## 📋 已完成的准备工作

### 1. knn_local.py 已完全更新 ✅

**位置**: `stage2_modular/thresholds/knn_local.py`

**已集成的优化**:
- ✅ **KDTree 优化**: 默认启用（条件满足时自动使用）
- ✅ **窗口筛选**: 默认启用，参数：
  - `window_v = 0.1` (风速窗口)
  - `window_r = 0.2` (密度窗口)
  - `min_candidates = max(K*2, 1000)` (最小候选数)
- ✅ **自动回退**: 窗口过小时自动扩展，扩展失败则使用全量
- ✅ **边界处理**: 所有边界情况已妥善处理

**结论**: **无需修改 knn_local.py**，直接使用即可。

### 2. JSON 配置文件已包含窗口筛选参数 ✅

**现有配置**: `experiments_compare_不同切向比例_分风机_JSMZS51-58.json`

**状态**: **已包含窗口筛选参数，可以直接使用**

**已包含的参数** (在 `defaults.thresholds` 部分):
```json
{
  "thresholds": {
    // ... 其他参数 ...
    "use_window_filter": true,   // 启用窗口筛选
    "window_v": 0.1,              // 风速窗口半径
    "window_r": 0.2,              // 密度窗口半径
    "min_candidates": 1000        // 最小候选数
  }
}
```

**说明**: 
- 所有窗口筛选参数已添加到配置文件中
- 使用了推荐的默认值
- 可以根据需要修改这些值（详见 `CONFIG_PARAMETERS.md`）

---

## 🚀 如何运行对比

### 方式1: 直接运行（使用默认优化）

```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

**效果**: 
- KDTree 优化自动启用（CPU模式下）
- 窗口筛选自动启用
- 使用默认窗口参数

**日志中会显示**:
```
[KNNLocal] Attempting KDTree optimization (d=2, metric=physics)...
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70% reduction)
```

### 方式2: 自动化对比测试（推荐）

```bash
python benchmark_knn.py
```

**效果**:
- 自动测试4种场景：
  1. 无优化（基线）
  2. 仅 KDTree
  3. KDTree + 窗口筛选（默认窗口）
  4. KDTree + 窗口筛选（宽窗口）
- 自动生成性能对比报告
- 保存完整日志文件

**输出示例**:
```
场景                      总耗时(秒)   提速比    候选筛除    状态
--------------------------------------------------------------------------------
无优化（基线）            45.23       1.00x     N/A         ✅
仅 KDTree                 28.67       1.58x     N/A         ✅
KDTree + 窗口筛选         18.34       2.47x     69.5%       ✅
```

---

## 🎯 可选：自定义配置

### 场景A: 测试无优化版本（基线对比）

如果想测试未优化的基线性能，创建一个新的配置文件：

**config_baseline.json** (复制现有配置并添加):
```json
{
  "defaults": {
    // ... 保持其他配置不变 ...
    "thresholds": {
      // ... 保持其他参数不变 ...
      "use_kdtree": false,         // ❌ 禁用 KDTree
      "use_window_filter": false   // ❌ 禁用窗口筛选
    }
  }
}
```

运行:
```bash
python main.py --config config_baseline.json
```

### 场景B: 自定义窗口大小

如果想调整窗口参数：

**config_custom.json**:
```json
{
  "defaults": {
    "thresholds": {
      "use_window_filter": true,
      "window_v": 0.15,          // 更宽的风速窗口
      "window_r": 0.25,          // 更宽的密度窗口
      "min_candidates": 500      // 更小的最小候选数
    }
  }
}
```

### 场景C: 仅测试 KDTree（不用窗口筛选）

```json
{
  "defaults": {
    "thresholds": {
      "use_kdtree": true,          // ✅ 启用 KDTree
      "use_window_filter": false   // ❌ 禁用窗口筛选
    }
  }
}
```

---

## ✅ 运行前检查清单

### 必需条件
- [x] **knn_local.py 已更新**: 在当前分支（copilot/code-review-and-comments）
- [x] **配置文件存在**: experiments_compare_不同切向比例_分风机_JSMZS51-58.json
- [ ] **依赖已安装**: 
  ```bash
  pip install numpy pandas scikit-learn torch
  ```
- [ ] **数据文件存在**: 检查配置文件中 `stations[].csv` 路径是否正确

### 可选检查
- [ ] 确认设备配置（`device: "cuda:0"` 或 `"cpu"`）
- [ ] 确认输出目录有写权限
- [ ] 准备好对比用的配置文件变体（如果需要）

---

## 📊 验证优化是否生效

### 查看日志关键标记

运行后，在日志中查找以下信息：

**1. KDTree 优化启用**:
```
[KNNLocal] Attempting KDTree optimization (d=2, metric=physics)...
[KNNLocal] KDTree search successful! Processing 500 queries with 500 neighbors each.
```

**2. 窗口筛选启用和效果**:
```
[KNNLocal] Using window filtering (window_v=0.1, window_r=0.2)...
[KNNLocal] Window filtering: avg candidates 15000/50000 (70.0% reduction)
```

**3. 数据规模**:
```
[KNNLocal] Using CPU path | device=cpu | candidates=50000, queries=10000
```

### 提取性能指标

```bash
# 运行并保存日志
python main.py --config <config>.json 2>&1 | tee run.log

# 提取候选筛除率
grep "reduction" run.log

# 提取 KNN 相关时间
grep -E "KNN|elapsed|耗时" run.log
```

---

## 🎉 总结

### 回答您的问题：

> 所以就只需要将现在更新好的knn_local.py文件和json文件更新，就可以运行对比了吗？

**答案**: **是的！**

更准确地说：

1. ✅ **knn_local.py**: 已经在当前分支更新完成，包含所有优化
2. ✅ **JSON 配置文件**: 现有的文件可以直接使用，**无需修改**
3. ✅ **立即可运行**: 执行 `python main.py --config <配置>.json` 即可
4. ✅ **优化自动生效**: 所有优化都有默认值，会自动启用

### 三个运行选项：

**选项1: 最简单** - 直接运行现有配置
```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

**选项2: 推荐** - 自动化对比测试
```bash
python benchmark_knn.py
```

**选项3: 高级** - 创建多个配置变体手动对比（参见上文）

---

## 📚 相关文档

- `QUICK_START.md` - 5分钟快速开始指南
- `RUN_BENCHMARK_GUIDE.md` - 完整运行指南
- `WINDOW_FILTERING_OPTIMIZATION.md` - 窗口筛选技术文档
- `KNN_OPTIMIZATION_REPORT.md` - KDTree 优化技术文档

---

**最后更新**: 2026-02-09  
**状态**: ✅ 已完成，可以运行
