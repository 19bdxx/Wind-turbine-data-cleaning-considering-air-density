#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 局部阈值优化验证脚本

测试目标：
1. 验证 KDTree 优化后的结果与原方法一致
2. 对比优化前后的性能提升
3. 测试不同配置下的行为
"""

import time
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_modular.thresholds.knn_local import KNNLocal

def generate_test_data(N=10000, Q=1000, d=2, seed=42):
    """
    生成测试数据
    
    参数:
        N: 训练集大小
        Q: 查询集大小
        d: 特征维度
        seed: 随机种子
    """
    np.random.seed(seed)
    
    # 训练集：正态分布
    train_X = np.random.randn(N, d).astype(np.float32)
    
    # 模拟残差 z-score (正态分布)
    train_zp = np.abs(np.random.randn(N)) * 0.5  # 正残差
    train_zn = np.abs(np.random.randn(N)) * 0.5  # 负残差
    
    # 查询集：稍微偏移的分布
    query_X = np.random.randn(Q, d).astype(np.float32) + 0.1
    
    # D_all: 不确定性估计 (正值)
    D_all = np.abs(np.random.randn(Q)) + 1.0
    
    # mask
    idx_train_mask = np.ones(Q, dtype=bool)
    idx_val_mask = np.zeros(Q, dtype=bool)
    idx_val_mask[:Q//5] = True  # 20% 验证集
    
    return {
        'train_X': train_X,
        'train_zp': train_zp,
        'train_zn': train_zn,
        'query_X': query_X,
        'D_all': D_all,
        'idx_train_mask': idx_train_mask,
        'idx_val_mask': idx_val_mask,
        'taus': (0.98, 0.98)
    }

def run_test(data, use_kdtree=True, use_gpu=False, metric='physics'):
    """
    运行 KNN 测试
    
    参数:
        data: 测试数据字典
        use_kdtree: 是否使用 KDTree 优化
        use_gpu: 是否使用 GPU
        metric: 距离度量类型
    """
    method = KNNLocal()
    
    # 配置
    cfg = {
        'metric': metric,
        'k_nei': 500,
        'knn_batch_q': 2048,
        'knn_train_chunk': 65536,
        'use_kdtree': use_kdtree,
        'grad_mode': 'physics',  # 使用 physics 模式避免需要模型
        'minmax': {
            'a': [0.0, 0.0],
            'b': [1.0, 1.0]
        },
        'lambda_t': 6.0,
        'physics_relative': True,
        'residuals': np.random.randn(len(data['query_X']))  # 模拟残差
    }
    
    device = 'cuda' if use_gpu else 'cpu'
    
    # 运行并计时
    start_time = time.time()
    
    result = method.compute(
        train_X=data['train_X'],
        train_zp=data['train_zp'],
        train_zn=data['train_zn'],
        query_X=data['query_X'],
        D_all=data['D_all'],
        idx_train_mask=data['idx_train_mask'],
        idx_val_mask=data['idx_val_mask'],
        taus=data['taus'],
        cfg=cfg,
        device=device
    )
    
    elapsed_time = time.time() - start_time
    
    return result, elapsed_time

def compare_results(result1, result2, tolerance=1e-3):
    """
    比较两个结果是否一致
    
    参数:
        result1, result2: ThresholdOutputs 对象
        tolerance: 允许的数值误差
    """
    # 比较阈值
    thr_pos_diff = np.abs(result1.thr_pos - result2.thr_pos)
    thr_neg_diff = np.abs(result1.thr_neg - result2.thr_neg)
    
    # 比较异常标记
    abnormal_diff = np.sum(result1.is_abnormal != result2.is_abnormal)
    
    print(f"\n=== 结果一致性检查 ===")
    print(f"阈值差异 (thr_pos): max={thr_pos_diff.max():.6f}, mean={thr_pos_diff.mean():.6f}")
    print(f"阈值差异 (thr_neg): max={thr_neg_diff.max():.6f}, mean={thr_neg_diff.mean():.6f}")
    print(f"异常标记差异: {abnormal_diff}/{len(result1.is_abnormal)} ({abnormal_diff/len(result1.is_abnormal)*100:.2f}%)")
    
    # 判断是否一致
    is_consistent = (
        thr_pos_diff.max() < tolerance and
        thr_neg_diff.max() < tolerance and
        abnormal_diff == 0
    )
    
    return is_consistent

def main():
    print("="*60)
    print("KNN 局部阈值优化验证测试")
    print("="*60)
    
    # 测试配置
    test_configs = [
        {'N': 5000, 'Q': 500, 'd': 2, 'name': '小规模 (5K/500)'},
        {'N': 20000, 'Q': 2000, 'd': 2, 'name': '中规模 (20K/2K)'},
        {'N': 50000, 'Q': 5000, 'd': 2, 'name': '大规模 (50K/5K)'},
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试: {config['name']}")
        print(f"训练集大小: {config['N']}, 查询集大小: {config['Q']}, 特征维度: {config['d']}")
        print(f"{'='*60}")
        
        # 生成测试数据
        data = generate_test_data(N=config['N'], Q=config['Q'], d=config['d'])
        
        # 测试1: 原始方法 (CPU, 不使用 KDTree)
        print("\n[1/2] 原始方法 (use_kdtree=False)...")
        result_original, time_original = run_test(data, use_kdtree=False, use_gpu=False)
        print(f"    耗时: {time_original:.3f}秒")
        
        # 测试2: KDTree 优化 (CPU)
        print("\n[2/2] KDTree 优化方法 (use_kdtree=True)...")
        result_kdtree, time_kdtree = run_test(data, use_kdtree=True, use_gpu=False)
        print(f"    耗时: {time_kdtree:.3f}秒")
        
        # 比较结果
        is_consistent = compare_results(result_original, result_kdtree)
        
        # 性能提升
        speedup = time_original / time_kdtree if time_kdtree > 0 else float('inf')
        print(f"\n=== 性能对比 ===")
        print(f"原始方法: {time_original:.3f}秒")
        print(f"KDTree 优化: {time_kdtree:.3f}秒")
        print(f"提速比: {speedup:.2f}x")
        
        # 总结
        print(f"\n=== 测试结果 ===")
        if is_consistent:
            print("✅ 结果一致性检查: 通过")
        else:
            print("❌ 结果一致性检查: 失败")
        
        if speedup > 1.0:
            print(f"✅ 性能提升: {speedup:.2f}倍")
        else:
            print(f"⚠️  性能提升: {speedup:.2f}倍 (可能数据量太小)")
    
    print(f"\n{'='*60}")
    print("所有测试完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
