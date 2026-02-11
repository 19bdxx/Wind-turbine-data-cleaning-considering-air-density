# 基准测试超时问题修复

## 🔍 问题

用户运行基准测试时所有场景都超时：

```bash
python benchmark_knn.py

❌ 运行超时（1小时）
❌ 运行超时（1小时）
❌ 运行超时（1小时）
❌ 运行超时（1小时）
```

## 📊 原因

配置文件包含 **20个runs**:
- 每个run: 3-5分钟
- 4个场景 × 20个runs = **80次运行**
- 总耗时: **4-5小时**
- 原超时: 1小时 → **必然超时** ❌

## ✅ 解决方案

### 快速测试模式（推荐）

```bash
python benchmark_knn.py --quick-test
```

**特点**:
- ✅ 只运行2个runs
- ✅ 5-10分钟完成
- ✅ 验证优化效果
- ✅ 不会超时

### 完整测试模式

```bash
python benchmark_knn.py
```

**特点**:
- 运行所有20个runs
- 4-5小时完成
- 超时增加到2小时（可能仍需更长时间）

## 📝 修改内容

1. **benchmark_knn.py**
   - 添加 `--quick-test` 参数
   - 超时 1h → 2h
   - 显示runs数量和预计耗时

2. **benchmark_config_quick.json**（新增）
   - 只包含2个runs
   - 用于快速验证

3. **BENCHMARK_GUIDE.md**（新增）
   - 详细使用指南
   - 故障排查
   - 最佳实践

4. **README.md**
   - 更新快速开始
   - 添加快速测试说明

## 🎯 使用建议

### 开发阶段（验证优化）
```bash
python benchmark_knn.py --quick-test
```
**5-10分钟**，快速验证KNN优化效果 ✅

### 详细分析（完整数据）
```bash
python benchmark_knn.py
```
**4-5小时**，全面性能分析

### 生产运行（实际处理）
```bash
python main.py --config your_config.json
```
不使用benchmark，直接运行 ✅

## 📊 性能对比

| 模式 | Runs | 耗时 | 用途 |
|------|------|------|------|
| 快速 | 2 | 5-10分钟 | 验证优化 ⭐ |
| 完整 | 20 | 4-5小时 | 全面分析 |

## 🔗 更多信息

详细信息请查看：
- **BENCHMARK_GUIDE.md** - 完整使用指南
- **README.md** - 快速开始
- **USER_GUIDE.md** - 详细配置说明

---

**建议：使用 `--quick-test` 快速验证，需要详细分析时再运行完整测试。**
