#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
窗口筛选优化验证测试

测试目标：
1. 验证窗口筛选功能正确性
2. 测试边界情况（候选不足、自动扩展）
3. 对比优化前后的性能和结果一致性
4. 评估不同窗口大小的影响
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_modular.thresholds.knn_local import KNNLocal, _filter_candidates_by_window

def test_window_filter_function():
    """测试窗口筛选函数的基本功能"""
    print("="*60)
    print("测试 1: 窗口筛选函数基本功能")
    print("="*60)
    
    # 生成测试数据
    np.random.seed(42)
    N = 1000
    d = 2
    
    # 训练集：在 [0, 1] 范围内均匀分布
    train_X = np.random.rand(N, d).astype(np.float32)
    
    # 查询点：在中心位置
    query_point = np.array([0.5, 0.5], dtype=np.float32)
    
    # 测试不同窗口大小
    window_sizes = [0.05, 0.1, 0.2, 0.3]
    
    for window_v in window_sizes:
        window_r = window_v
        indices = _filter_candidates_by_window(
            train_X, query_point, 
            window_v, window_r, 
            min_candidates=50
        )
        
        # 验证筛选结果
        filtered = train_X[indices]
        
        # 检查是否在窗口内
        in_window = (
            (filtered[:, 0] >= query_point[0] - window_v) & 
            (filtered[:, 0] <= query_point[0] + window_v) &
            (filtered[:, 1] >= query_point[1] - window_r) & 
            (filtered[:, 1] <= query_point[1] + window_r)
        )
        
        ratio = len(indices) / N
        in_window_ratio = np.sum(in_window) / len(in_window) if len(in_window) > 0 else 0
        
        print(f"  窗口大小 {window_v:.2f}:")
        print(f"    筛选候选数: {len(indices)}/{N} ({ratio*100:.1f}%)")
        print(f"    窗口内占比: {in_window_ratio*100:.1f}%")
        
        # 基本正确性检查
        assert len(indices) >= 50, f"候选数不足: {len(indices)} < 50"
        # 允许扩展导致部分候选在窗口外（扩展是为了满足min_candidates）
        # 只要大部分或有扩展的情况即可
        if window_v >= 0.1:  # 对于较大的窗口，检查更严格
            assert in_window_ratio >= 0.3, f"窗口内比例过低: {in_window_ratio:.2f}"
    
    print("✅ 测试通过：窗口筛选函数工作正常\n")

def test_edge_cases():
    """测试边界情况"""
    print("="*60)
    print("测试 2: 边界情况处理")
    print("="*60)
    
    np.random.seed(42)
    
    # 情况1: 窗口过小，需要自动扩展
    print("  情况1: 窗口过小，触发自动扩展...")
    N = 1000
    train_X = np.random.rand(N, 2).astype(np.float32)
    query_point = np.array([0.5, 0.5], dtype=np.float32)
    
    indices = _filter_candidates_by_window(
        train_X, query_point,
        window_v=0.01,  # 非常小的窗口
        window_r=0.01,
        min_candidates=100
    )
    print(f"    结果: 筛选到 {len(indices)} 个候选（要求最少100个）")
    assert len(indices) >= 100, "自动扩展失败"
    
    # 情况2: K > N，应该返回全部
    print("  情况2: 要求的候选数超过总数...")
    train_X_small = np.random.rand(50, 2).astype(np.float32)
    indices = _filter_candidates_by_window(
        train_X_small, query_point,
        window_v=0.1,
        window_r=0.1,
        min_candidates=100  # 要求100个但只有50个
    )
    print(f"    结果: 返回 {len(indices)} 个候选（总共50个）")
    assert len(indices) == 50, "应该返回全部候选"
    
    # 情况3: 1维特征
    print("  情况3: 1维特征（仅风速）...")
    train_X_1d = np.random.rand(1000, 1).astype(np.float32)
    query_1d = np.array([0.5], dtype=np.float32)
    indices = _filter_candidates_by_window(
        train_X_1d, query_1d,
        window_v=0.1,
        window_r=0.0,  # 1维时不使用
        min_candidates=50
    )
    print(f"    结果: 筛选到 {len(indices)} 个候选")
    assert len(indices) >= 50, "1维筛选失败"
    
    print("✅ 测试通过：边界情况处理正确\n")

def test_knn_with_window_filtering():
    """测试KNN方法集成窗口筛选"""
    print("="*60)
    print("测试 3: KNN方法集成测试")
    print("="*60)
    
    np.random.seed(42)
    N = 5000
    Q = 500
    d = 2
    
    # 生成测试数据
    train_X = np.random.randn(N, d).astype(np.float32) * 0.3 + 0.5
    train_zp = np.abs(np.random.randn(N)) * 0.5
    train_zn = np.abs(np.random.randn(N)) * 0.5
    
    query_X = np.random.randn(Q, d).astype(np.float32) * 0.3 + 0.5
    D_all = np.abs(np.random.randn(Q)) + 1.0
    
    idx_train_mask = np.ones(Q, dtype=bool)
    idx_val_mask = np.zeros(Q, dtype=bool)
    idx_val_mask[:Q//5] = True
    
    method = KNNLocal()
    
    # 配置
    cfg_base = {
        'metric': 'physics',
        'k_nei': 100,
        'grad_mode': 'physics',
        'minmax': {'a': [0.0, 0.0], 'b': [1.0, 1.0]},
        'lambda_t': 6.0,
        'physics_relative': True,
        'residuals': np.random.randn(Q),
        'use_kdtree': True
    }
    
    # 测试1: 不使用窗口筛选
    print("  [1/3] 运行 KNN (无窗口筛选)...")
    cfg_no_window = cfg_base.copy()
    cfg_no_window['use_window_filter'] = False
    
    start = time.time()
    result_no_window = method.compute(
        train_X=train_X, train_zp=train_zp, train_zn=train_zn,
        query_X=query_X, D_all=D_all,
        idx_train_mask=idx_train_mask, idx_val_mask=idx_val_mask,
        taus=(0.98, 0.98), cfg=cfg_no_window, device='cpu'
    )
    time_no_window = time.time() - start
    print(f"    耗时: {time_no_window:.3f}秒")
    
    # 测试2: 使用窗口筛选（宽窗口）
    print("  [2/3] 运行 KNN (窗口筛选, window=0.2)...")
    cfg_wide_window = cfg_base.copy()
    cfg_wide_window['use_window_filter'] = True
    cfg_wide_window['window_v'] = 0.2
    cfg_wide_window['window_r'] = 0.2
    
    start = time.time()
    result_wide = method.compute(
        train_X=train_X, train_zp=train_zp, train_zn=train_zn,
        query_X=query_X, D_all=D_all,
        idx_train_mask=idx_train_mask, idx_val_mask=idx_val_mask,
        taus=(0.98, 0.98), cfg=cfg_wide_window, device='cpu'
    )
    time_wide = time.time() - start
    print(f"    耗时: {time_wide:.3f}秒")
    
    # 测试3: 使用窗口筛选（窄窗口）
    print("  [3/3] 运行 KNN (窗口筛选, window=0.1)...")
    cfg_narrow_window = cfg_base.copy()
    cfg_narrow_window['use_window_filter'] = True
    cfg_narrow_window['window_v'] = 0.1
    cfg_narrow_window['window_r'] = 0.1
    
    start = time.time()
    result_narrow = method.compute(
        train_X=train_X, train_zp=train_zp, train_zn=train_zn,
        query_X=query_X, D_all=D_all,
        idx_train_mask=idx_train_mask, idx_val_mask=idx_val_mask,
        taus=(0.98, 0.98), cfg=cfg_narrow_window, device='cpu'
    )
    time_narrow = time.time() - start
    print(f"    耗时: {time_narrow:.3f}秒")
    
    # 比较结果
    print("\n  === 结果对比 ===")
    
    # 阈值差异
    thr_diff_wide = np.abs(result_no_window.thr_pos - result_wide.thr_pos)
    thr_diff_narrow = np.abs(result_no_window.thr_pos - result_narrow.thr_pos)
    
    print(f"  阈值差异 (宽窗口): max={thr_diff_wide.max():.4f}, mean={thr_diff_wide.mean():.4f}")
    print(f"  阈值差异 (窄窗口): max={thr_diff_narrow.max():.4f}, mean={thr_diff_narrow.mean():.4f}")
    
    # 异常标记差异
    abnormal_diff_wide = np.sum(result_no_window.is_abnormal != result_wide.is_abnormal)
    abnormal_diff_narrow = np.sum(result_no_window.is_abnormal != result_narrow.is_abnormal)
    
    print(f"  异常标记差异 (宽窗口): {abnormal_diff_wide}/{Q} ({abnormal_diff_wide/Q*100:.1f}%)")
    print(f"  异常标记差异 (窄窗口): {abnormal_diff_narrow}/{Q} ({abnormal_diff_narrow/Q*100:.1f}%)")
    
    # 性能提升
    speedup_wide = time_no_window / time_wide
    speedup_narrow = time_no_window / time_narrow
    
    print(f"\n  === 性能提升 ===")
    print(f"  无窗口筛选: {time_no_window:.3f}秒")
    print(f"  宽窗口(0.2): {time_wide:.3f}秒 (提速 {speedup_wide:.2f}x)")
    print(f"  窄窗口(0.1): {time_narrow:.3f}秒 (提速 {speedup_narrow:.2f}x)")
    
    # 验证结果基本一致
    assert thr_diff_wide.mean() < 0.5, "宽窗口结果差异过大"
    assert abnormal_diff_wide / Q < 0.15, "宽窗口异常标记差异过大"
    
    print("\n✅ 测试通过：KNN集成窗口筛选工作正常\n")

def main():
    print("\n" + "="*60)
    print("窗口筛选优化验证测试")
    print("="*60 + "\n")
    
    try:
        # 测试1: 窗口筛选函数
        test_window_filter_function()
        
        # 测试2: 边界情况
        test_edge_cases()
        
        # 测试3: KNN集成
        test_knn_with_window_filtering()
        
        print("="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
