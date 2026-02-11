# -*- coding: utf-8 -*-
"""
测试窗口筛选功能的正确性
"""
import numpy as np
import torch
import sys
import os

# 添加 stage2_modular 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage2_modular'))

from thresholds.knn_local import _window_filter_candidates


def test_window_filtering():
    """测试窗口筛选函数"""
    print("=" * 60)
    print("测试窗口筛选功能")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建测试数据 (1D case: 只有风速)
    print("\n测试 1: 一维数据（仅风速）")
    N = 1000  # 候选点数
    Q = 10    # 查询点数
    
    # 候选点: 均匀分布在 [0, 1]
    Xcand_1d = torch.rand(N, 1, device=device)
    
    # 查询点: 选择几个特定位置
    Zb_1d = torch.tensor([[0.5], [0.1], [0.9]], device=device)
    Q_actual = Zb_1d.shape[0]
    
    # 测试参数
    window_v = 0.1
    window_r = 0.2  # 不使用（1D）
    min_candidates = 50
    K_NEI = 100
    
    # 执行窗口筛选
    filtered_indices, expand_count = _window_filter_candidates(
        Zb_1d, Xcand_1d, window_v, window_r, 1, min_candidates, K_NEI
    )
    
    # 验证结果
    print(f"查询点数: {Q_actual}")
    print(f"候选点数: {N}")
    print(f"窗口半径: {window_v}")
    print(f"扩展次数: {expand_count}")
    
    for i, (query, indices) in enumerate(zip(Zb_1d, filtered_indices)):
        n_filtered = len(indices)
        if n_filtered > 0:
            candidates = Xcand_1d[indices, 0]
            min_val = candidates.min().item()
            max_val = candidates.max().item()
            query_val = query[0].item()
            
            print(f"\n  查询点 {i}: {query_val:.3f}")
            print(f"    筛选候选数: {n_filtered}/{N} ({100*n_filtered/N:.1f}%)")
            print(f"    候选范围: [{min_val:.3f}, {max_val:.3f}]")
            print(f"    期望范围: [{query_val-window_v:.3f}, {query_val+window_v:.3f}]")
            
            # 验证所有候选点都在窗口内
            in_window = ((candidates >= query_val - window_v * 1.6) & 
                        (candidates <= query_val + window_v * 1.6)).all()
            print(f"    所有候选在窗口内: {in_window.item()}")
    
    # 测试 2D case
    print("\n" + "=" * 60)
    print("测试 2: 二维数据（风速 + 空气密度）")
    print("=" * 60)
    
    # 候选点: 均匀分布在 [0, 1] x [0, 1]
    Xcand_2d = torch.rand(N, 2, device=device)
    
    # 查询点
    Zb_2d = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]], device=device)
    Q_actual = Zb_2d.shape[0]
    
    # 执行窗口筛选
    filtered_indices, expand_count = _window_filter_candidates(
        Zb_2d, Xcand_2d, window_v, window_r, 2, min_candidates, K_NEI
    )
    
    print(f"查询点数: {Q_actual}")
    print(f"候选点数: {N}")
    print(f"窗口半径: v={window_v}, rho={window_r}")
    print(f"扩展次数: {expand_count}")
    
    for i, (query, indices) in enumerate(zip(Zb_2d, filtered_indices)):
        n_filtered = len(indices)
        if n_filtered > 0:
            candidates = Xcand_2d[indices]
            ws_min, ws_max = candidates[:, 0].min().item(), candidates[:, 0].max().item()
            rho_min, rho_max = candidates[:, 1].min().item(), candidates[:, 1].max().item()
            query_ws, query_rho = query[0].item(), query[1].item()
            
            print(f"\n  查询点 {i}: ws={query_ws:.3f}, rho={query_rho:.3f}")
            print(f"    筛选候选数: {n_filtered}/{N} ({100*n_filtered/N:.1f}%)")
            print(f"    风速范围: [{ws_min:.3f}, {ws_max:.3f}]")
            print(f"    期望风速: [{query_ws-window_v:.3f}, {query_ws+window_v:.3f}]")
            print(f"    密度范围: [{rho_min:.3f}, {rho_max:.3f}]")
            print(f"    期望密度: [{query_rho-window_r:.3f}, {query_rho+window_r:.3f}]")
            
            # 验证所有候选点都在窗口内（考虑扩展）
            max_expand = 1.5 ** 3  # 最多扩展3次
            ws_in = ((candidates[:, 0] >= query_ws - window_v * max_expand) & 
                    (candidates[:, 0] <= query_ws + window_v * max_expand)).all()
            rho_in = ((candidates[:, 1] >= query_rho - window_r * max_expand) & 
                     (candidates[:, 1] <= query_rho + window_r * max_expand)).all()
            print(f"    风速在窗口内: {ws_in.item()}, 密度在窗口内: {rho_in.item()}")
    
    # 测试边缘情况：候选不足时的自动扩展
    print("\n" + "=" * 60)
    print("测试 3: 候选不足时的自动扩展")
    print("=" * 60)
    
    # 创建稀疏数据：大部分点集中在 [0.4, 0.6]
    Xcand_sparse = torch.cat([
        torch.rand(50, 1, device=device) * 0.2 + 0.4,  # [0.4, 0.6]
        torch.rand(10, 1, device=device)                 # [0, 1]
    ])
    
    # 查询点在稀疏区域外
    Zb_sparse = torch.tensor([[0.1]], device=device)  # 远离密集区
    
    # 使用较小的窗口和较高的 min_candidates
    small_window = 0.05
    high_min = 100
    
    filtered_indices, expand_count = _window_filter_candidates(
        Zb_sparse, Xcand_sparse, small_window, window_r, 1, high_min, K_NEI
    )
    
    n_filtered = len(filtered_indices[0])
    print(f"稀疏数据测试:")
    print(f"  候选总数: {len(Xcand_sparse)}")
    print(f"  查询点: {Zb_sparse[0, 0].item():.3f}")
    print(f"  初始窗口: {small_window}")
    print(f"  最小候选要求: {high_min}")
    print(f"  实际筛选数: {n_filtered}")
    print(f"  扩展次数: {expand_count}")
    print(f"  是否使用全量: {n_filtered == len(Xcand_sparse)}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_window_filtering()
