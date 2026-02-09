#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 优化自动化基准测试脚本

功能:
    自动运行多种 KNN 优化配置组合，提取性能指标并生成对比报告

使用方法:
    python benchmark_knn.py [--config <基础配置文件>] [--output <报告文件>]

输出:
    - 控制台打印性能对比表格
    - 生成详细的性能报告文件（可选）
    - 保存各场景的完整日志
"""

import time
import subprocess
import re
import json
import argparse
import sys
from pathlib import Path

def create_config_variant(base_config_path, variant_name, modifications):
    """
    基于基础配置创建变体配置文件
    
    参数:
        base_config_path: 基础配置文件路径
        variant_name: 变体名称（用于文件名）
        modifications: 要修改的参数字典
    
    返回:
        新配置文件路径
    """
    # 读取基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 应用修改
    for key_path, value in modifications.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    # 保存新配置
    output_path = f"config_benchmark_{variant_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return output_path

def run_config(config_file, name, save_log=True):
    """
    运行配置并提取性能指标
    
    参数:
        config_file: 配置文件路径
        name: 场景名称
        save_log: 是否保存完整日志
    
    返回:
        性能指标字典
    """
    print(f"\n{'='*70}")
    print(f"运行场景: {name}")
    print(f"配置文件: {config_file}")
    print(f"{'='*70}\n")
    
    # 运行程序并记录时间
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', 'main.py', '--config', config_file],
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("❌ 运行超时（1小时）")
        return None
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return None
    
    elapsed_time = time.time() - start_time
    
    # 保存日志
    if save_log:
        log_file = f"log_benchmark_{name.replace(' ', '_')}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"📝 日志已保存到: {log_file}")
    
    # 提取性能指标
    metrics = {
        'name': name,
        'config': config_file,
        'success': success,
        'total_time': elapsed_time
    }
    
    # 查找窗口筛选信息
    window_pattern = r'avg candidates (\d+)/(\d+) \((\d+\.?\d*)% reduction\)'
    window_match = re.search(window_pattern, output)
    if window_match:
        avg_cand, total_cand, reduction = window_match.groups()
        metrics['avg_candidates'] = int(avg_cand)
        metrics['total_candidates'] = int(total_cand)
        metrics['reduction_percent'] = float(reduction)
        print(f"✅ 窗口筛选: {avg_cand}/{total_cand} 候选 ({reduction}% 筛除)")
    else:
        metrics['avg_candidates'] = None
        metrics['total_candidates'] = None
        metrics['reduction_percent'] = 0.0
    
    # 查找 KDTree 使用信息
    if 'Attempting KDTree optimization' in output:
        metrics['used_kdtree'] = True
        print("✅ 使用 KDTree 优化")
    else:
        metrics['used_kdtree'] = False
        print("⚪ 未使用 KDTree")
    
    # 查找窗口筛选启用信息
    if 'Using window filtering' in output:
        metrics['used_window_filter'] = True
        print("✅ 使用窗口筛选")
    else:
        metrics['used_window_filter'] = False
        print("⚪ 未使用窗口筛选")
    
    # 查找候选数量
    candidates_pattern = r'candidates=(\d+), queries=(\d+)'
    candidates_match = re.search(candidates_pattern, output)
    if candidates_match:
        n_candidates, n_queries = candidates_match.groups()
        metrics['n_candidates'] = int(n_candidates)
        metrics['n_queries'] = int(n_queries)
        print(f"📊 数据规模: N={n_candidates}, Q={n_queries}")
    
    print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
    
    if not success:
        print(f"⚠️  运行返回非零状态码: {result.returncode}")
    
    return metrics

def print_comparison_table(results):
    """打印性能对比表格"""
    print(f"\n{'='*80}")
    print("性能对比总结")
    print(f"{'='*80}\n")
    
    # 找到基线（无优化）
    baseline = None
    for r in results:
        if r and not r.get('used_kdtree', False) and not r.get('used_window_filter', False):
            baseline = r
            break
    
    if not baseline:
        baseline = results[0]  # 使用第一个作为基线
    
    baseline_time = baseline['total_time']
    
    # 表头
    header = f"{'场景':<25} {'总耗时(秒)':<12} {'提速比':<10} {'候选筛除':<12} {'状态'}"
    print(header)
    print("-" * 80)
    
    # 数据行
    for r in results:
        if not r:
            continue
        
        name = r['name']
        time_str = f"{r['total_time']:.2f}"
        speedup = baseline_time / r['total_time']
        speedup_str = f"{speedup:.2f}x"
        
        if r['reduction_percent'] > 0:
            reduction_str = f"{r['reduction_percent']:.1f}%"
        else:
            reduction_str = "N/A"
        
        status = "✅" if r['success'] else "❌"
        
        print(f"{name:<25} {time_str:<12} {speedup_str:<10} {reduction_str:<12} {status}")
    
    print()

def generate_report(results, output_file):
    """生成详细的性能报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# KNN 优化性能测试报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 测试场景\n\n")
        for i, r in enumerate(results, 1):
            if not r:
                continue
            f.write(f"### 场景 {i}: {r['name']}\n\n")
            f.write(f"- 配置文件: `{r['config']}`\n")
            f.write(f"- KDTree: {'✅ 启用' if r.get('used_kdtree') else '❌ 禁用'}\n")
            f.write(f"- 窗口筛选: {'✅ 启用' if r.get('used_window_filter') else '❌ 禁用'}\n")
            f.write(f"- 总耗时: {r['total_time']:.2f} 秒\n")
            
            if r.get('n_candidates'):
                f.write(f"- 数据规模: N={r['n_candidates']}, Q={r['n_queries']}\n")
            
            if r['reduction_percent'] > 0:
                f.write(f"- 候选筛除: {r['avg_candidates']}/{r['total_candidates']} ({r['reduction_percent']:.1f}%)\n")
            
            f.write("\n")
        
        f.write("## 性能对比\n\n")
        
        # 找基线
        baseline = results[0]
        baseline_time = baseline['total_time']
        
        f.write("| 场景 | 总耗时(秒) | 提速比 | 候选筛除 |\n")
        f.write("|------|-----------|--------|----------|\n")
        
        for r in results:
            if not r:
                continue
            speedup = baseline_time / r['total_time']
            reduction = f"{r['reduction_percent']:.1f}%" if r['reduction_percent'] > 0 else "N/A"
            f.write(f"| {r['name']} | {r['total_time']:.2f} | {speedup:.2f}x | {reduction} |\n")
        
        f.write("\n")
    
    print(f"📄 详细报告已保存到: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="KNN 优化自动化基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        default='experiments_compare_不同切向比例_分风机_JSMZS51-58.json',
        help='基础配置文件路径（默认: experiments_compare_不同切向比例_分风机_JSMZS51-58.json）'
    )
    parser.add_argument(
        '--output',
        default='benchmark_report.md',
        help='输出报告文件路径（默认: benchmark_report.md）'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='跳过基线测试（无优化）'
    )
    
    args = parser.parse_args()
    
    # 检查基础配置文件是否存在
    if not Path(args.config).exists():
        print(f"❌ 配置文件不存在: {args.config}")
        print("提示: 请确保配置文件路径正确，或使用 --config 指定")
        return 1
    
    print("🚀 KNN 优化自动化基准测试")
    print(f"基础配置: {args.config}")
    print(f"输出报告: {args.output}")
    print()
    
    results = []
    
    # 场景1: 无优化（基线）
    if not args.skip_baseline:
        config1 = create_config_variant(
            args.config,
            'baseline',
            {
                'defaults.thresholds.use_kdtree': False,
                'defaults.thresholds.use_window_filter': False
            }
        )
        result1 = run_config(config1, '无优化（基线）', save_log=True)
        if result1:
            results.append(result1)
    
    # 场景2: 仅 KDTree
    config2 = create_config_variant(
        args.config,
        'kdtree_only',
        {
            'defaults.thresholds.use_kdtree': True,
            'defaults.thresholds.use_window_filter': False
        }
    )
    result2 = run_config(config2, '仅 KDTree', save_log=True)
    if result2:
        results.append(result2)
    
    # 场景3: KDTree + 窗口筛选（默认窗口）
    config3 = create_config_variant(
        args.config,
        'kdtree_window_default',
        {
            'defaults.thresholds.use_kdtree': True,
            'defaults.thresholds.use_window_filter': True,
            'defaults.thresholds.window_v': 0.1,
            'defaults.thresholds.window_r': 0.2
        }
    )
    result3 = run_config(config3, 'KDTree + 窗口筛选 (0.1/0.2)', save_log=True)
    if result3:
        results.append(result3)
    
    # 场景4: KDTree + 窗口筛选（宽窗口）
    config4 = create_config_variant(
        args.config,
        'kdtree_window_wide',
        {
            'defaults.thresholds.use_kdtree': True,
            'defaults.thresholds.use_window_filter': True,
            'defaults.thresholds.window_v': 0.2,
            'defaults.thresholds.window_r': 0.3
        }
    )
    result4 = run_config(config4, 'KDTree + 窗口筛选 (0.2/0.3)', save_log=True)
    if result4:
        results.append(result4)
    
    # 打印对比表格
    if results:
        print_comparison_table(results)
        
        # 生成报告
        generate_report(results, args.output)
    else:
        print("❌ 没有成功的测试结果")
        return 1
    
    print(f"\n✅ 基准测试完成！")
    return 0

if __name__ == '__main__':
    sys.exit(main())
