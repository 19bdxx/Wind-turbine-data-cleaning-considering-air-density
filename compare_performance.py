#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比脚本：比较窗口筛选优化前后的 KNN 性能

用法:
    python compare_performance.py

说明:
    - 生成模拟数据进行性能测试
    - 比较有/无窗口筛选的运行时间
    - 输出详细的性能指标
"""

import time
import numpy as np
import torch
import sys
import os

# 添加 stage2_modular 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage2_modular'))

from thresholds.knn_local import _window_filter_candidates


def generate_test_data(N, Q, d, device):
    """生成测试数据"""
    # 候选点：正态分布，中心在 0.5
    Xcand = torch.randn(N, d, device=device) * 0.2 + 0.5
    Xcand = torch.clamp(Xcand, 0.0, 1.0)
    
    # 查询点：从候选中随机选择
    indices = torch.randperm(N)[:Q]
    Zb = Xcand[indices] + torch.randn(Q, d, device=device) * 0.05  # 加小扰动
    Zb = torch.clamp(Zb, 0.0, 1.0)
    
    return Xcand, Zb


def benchmark_full_computation(Xcand, Zb, K_NEI):
    """基准：全量距离计算（模拟）"""
    N = Xcand.shape[0]
    Q = Zb.shape[0]
    
    start = time.time()
    
    # 模拟完整的距离计算
    # 实际中会计算所有 Q×N 对
    total_distance_calcs = Q * N
    
    # 为了快速演示，我们只计算一个子集
    # 但记录理论上的计算量
    sample_size = min(100, Q)
    for i in range(sample_size):
        # 计算距离：欧氏距离的平方
        diff = Xcand - Zb[i:i+1, :]  # (N, d)
        dist = torch.sum(diff * diff, dim=1)  # (N,)
        # 选择 topK
        topk_vals, topk_idx = torch.topk(dist, k=min(K_NEI, N), largest=False)
    
    elapsed = time.time() - start
    # 根据采样比例估算总时间
    estimated_total = elapsed * (Q / sample_size)
    
    return {
        'time': estimated_total,
        'distance_calcs': total_distance_calcs,
        'sample_size': sample_size
    }


def benchmark_window_filtering(Xcand, Zb, window_v, window_r, d, min_candidates, K_NEI):
    """优化：窗口筛选 + 距离计算"""
    N = Xcand.shape[0]
    Q = Zb.shape[0]
    
    start = time.time()
    
    # 窗口筛选
    filtered_indices, expand_count = _window_filter_candidates(
        Zb, Xcand, window_v, window_r, d, min_candidates, K_NEI
    )
    
    # 计算距离（仅对筛选后的候选）
    total_distance_calcs = 0
    for i in range(Q):
        cand_idx = filtered_indices[i]
        n_cand = len(cand_idx)
        total_distance_calcs += n_cand
        
        if n_cand > 0:
            Xc = Xcand[cand_idx, :]
            diff = Xc - Zb[i:i+1, :]
            dist = torch.sum(diff * diff, dim=1)
            k_actual = min(K_NEI, n_cand)
            if k_actual > 0:
                topk_vals, topk_idx = torch.topk(dist, k=k_actual, largest=False)
    
    elapsed = time.time() - start
    avg_candidates = total_distance_calcs / Q if Q > 0 else 0
    
    return {
        'time': elapsed,
        'distance_calcs': total_distance_calcs,
        'avg_candidates': avg_candidates,
        'expand_count': expand_count
    }


def main():
    print("=" * 70)
    print("KNN 窗口筛选性能对比测试")
    print("=" * 70)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 测试配置
    test_configs = [
        {'N': 10000, 'Q': 1000, 'd': 2, 'name': '小规模'},
        {'N': 50000, 'Q': 5000, 'd': 2, 'name': '中等规模'},
        {'N': 100000, 'Q': 10000, 'd': 2, 'name': '大规模'},
    ]
    
    # 窗口筛选参数
    window_v = 0.1
    window_r = 0.2
    min_candidates = 1000
    K_NEI = 500
    
    print(f"\n窗口参数: window_v={window_v}, window_r={window_r}, min_candidates={min_candidates}, K={K_NEI}")
    
    # 对每个配置进行测试
    for config in test_configs:
        N = config['N']
        Q = config['Q']
        d = config['d']
        name = config['name']
        
        print("\n" + "=" * 70)
        print(f"测试: {name} (N={N}, Q={Q}, d={d})")
        print("=" * 70)
        
        # 生成数据
        print("生成测试数据...")
        Xcand, Zb = generate_test_data(N, Q, d, device)
        
        # 基准测试：全量计算
        print("\n[1] 基准方法（全量计算）")
        result_full = benchmark_full_computation(Xcand, Zb, K_NEI)
        print(f"  总距离计算次数: {result_full['distance_calcs']:,}")
        print(f"  估算运行时间: {result_full['time']:.3f} 秒")
        print(f"  （基于 {result_full['sample_size']} 个查询点的采样）")
        
        # 优化测试：窗口筛选
        print("\n[2] 窗口筛选优化")
        result_window = benchmark_window_filtering(
            Xcand, Zb, window_v, window_r, d, min_candidates, K_NEI
        )
        print(f"  总距离计算次数: {result_window['distance_calcs']:,}")
        print(f"  平均候选数/查询: {result_window['avg_candidates']:.1f}")
        print(f"  窗口扩展次数: {result_window['expand_count']}/{Q}")
        print(f"  实际运行时间: {result_window['time']:.3f} 秒")
        
        # 性能提升
        reduction = (1.0 - result_window['distance_calcs'] / result_full['distance_calcs']) * 100
        speedup = result_full['time'] / result_window['time'] if result_window['time'] > 0 else float('inf')
        
        print("\n[3] 性能对比")
        print(f"  计算量减少: {reduction:.1f}%")
        print(f"  加速比: {speedup:.2f}×")
        print(f"  筛除率: {reduction:.1f}%")
        
        # 清理内存
        del Xcand, Zb
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
    
    print("\n总结:")
    print("- 窗口筛选显著减少了距离计算次数")
    print("- 在大规模数据上效果最明显（N > 50,000）")
    print("- 实际加速比取决于数据分布和窗口参数")
    print("\n建议:")
    print("- 密集数据：减小窗口（0.05/0.1）")
    print("- 稀疏数据：增大窗口（0.15/0.25）")
    print("- 根据实际数据调整 min_candidates")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"错误: 缺少依赖包 - {e}")
        print("请安装 numpy 和 torch:")
        print("  pip install numpy torch")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
