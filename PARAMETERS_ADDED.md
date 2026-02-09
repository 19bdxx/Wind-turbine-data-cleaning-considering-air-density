# ✅ 窗口筛选参数已添加到配置文件

## 问题解决

用户反馈："window_v, window_r, min_candidates 这三个配置参数在 json 文件中没有啊"

**现在已解决** ✅

---

## 📍 参数位置

**文件**: `experiments_compare_不同切向比例_分风机_JSMZS51-58.json`

**位置**: `defaults.thresholds` 部分

**已添加的参数**:
```json
{
  "defaults": {
    "thresholds": {
      // ... 其他参数 ...
      "use_window_filter": true,   // ✅ 新增：启用窗口筛选
      "window_v": 0.1,              // ✅ 新增：风速窗口半径
      "window_r": 0.2,              // ✅ 新增：密度窗口半径
      "min_candidates": 1000        // ✅ 新增：最小候选数
    }
  }
}
```

---

## 📝 参数快速参考

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **use_window_filter** | Boolean | `true` | 是否启用窗口筛选 |
| **window_v** | Float | `0.1` | 风速窗口半径（标准化空间） |
| **window_r** | Float | `0.2` | 密度窗口半径（标准化空间） |
| **min_candidates** | Integer | `1000` | 最小候选数量 |

---

## 🎯 现在可以做什么

### 1. 直接运行（使用默认值）

```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

✅ 参数已设置为推荐值，直接运行即可

### 2. 自定义参数

编辑 JSON 文件，修改这些值：

```json
{
  "thresholds": {
    "window_v": 0.15,     // 调整风速窗口
    "window_r": 0.25,     // 调整密度窗口
    "min_candidates": 1500 // 调整最小候选数
  }
}
```

### 3. 禁用窗口筛选（用于对比测试）

```json
{
  "thresholds": {
    "use_window_filter": false
  }
}
```

---

## 📚 详细文档

想了解更多？查看：

- **CONFIG_PARAMETERS.md** - 参数详细说明
  - 每个参数的含义和作用
  - MinMax 和 Z-score 归一化下的推荐值
  - 典型配置组合
  - 调优指南

- **WINDOW_FILTERING_OPTIMIZATION.md** - 技术原理
  - 窗口筛选算法详解
  - 复杂度分析
  - 性能测试结果

- **RUN_BENCHMARK_GUIDE.md** - 运行指南
  - 完整的运行说明
  - 性能对比方法
  - 故障排查

---

## ✅ 总结

**问题**: 窗口筛选参数在配置文件中缺失

**解决**: 
1. ✅ 已添加 4 个参数到 JSON 配置文件
2. ✅ 使用了推荐的默认值
3. ✅ 创建了详细的参数说明文档
4. ✅ 用户现在可以轻松查看和修改这些参数

**可以直接使用** - 无需额外配置！

---

**更新时间**: 2026-02-09  
**状态**: ✅ 已完成
