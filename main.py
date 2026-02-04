# -*- coding: utf-8 -*-
"""
主入口文件 - 风机数据清洗实验批处理程序

功能说明：
    本文件是整个数据清洗流程的入口，负责：
    1. 读取 JSON 格式的实验配置文件
    2. 解析多站点、多风机、多实验方案的配置
    3. 按顺序执行每个实验方案（run）
    4. 支持跨 run 复用数据切分（train/val/test）
    
输入：
    - JSON 配置文件（通过命令行参数 --config 指定）
      配置文件结构：
      {
        "defaults": {...},      # 全局默认配置
        "stations": [...],      # 风电场站点列表
        "runs": [...]          # 实验方案列表
      }
      
输出：
    - 每个站点、每台风机的清洗结果 CSV 文件
    - 输出路径由配置文件中的 out_root 和 out_subdir 指定
    
依赖模块：
    - stage2_modular.pipeline.orchestrator: 实验编排器，执行单个实验方案

使用示例：
    python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json

作者：项目团队
日期：2024
"""

import os, json, argparse
from stage2_modular.pipeline.orchestrator import run_single_run

def main():
    """
    主函数：解析配置文件并批量执行实验方案
    
    工作流程：
        1. 解析命令行参数，获取配置文件路径
        2. 加载 JSON 配置文件
        3. 打印实验计划摘要（站点信息、实验方案概览）
        4. 按顺序执行每个实验方案（run）
        5. 支持跨 run 的数据切分复用（通过 reuse_split_from）
    
    配置文件关键字段：
        - defaults: 包含所有默认参数（如模型配置、阈值方法等）
        - stations: 站点列表，每个站点包含：
            - name: 站点名称
            - csv: 宽表数据路径（包含空气密度等环境数据）
            - turbine_start/turbine_end: 风机编号范围
        - runs: 实验方案列表，每个 run 可覆盖 defaults 中的参数
    
    数据切分复用机制：
        - 每个 run 执行后会保存其数据切分（train/val/test 划分）
        - 后续 run 可通过 reuse_split_from 字段指定复用之前 run 的切分
        - 这样可以保证不同实验在相同数据划分上比较，消除随机性影响
    """
    # ========== 1. 解析命令行参数 ==========
    # 构建命令行解析器并声明一个必填参数 --config
    ap = argparse.ArgumentParser(
        description="风机数据清洗批量实验程序",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="实验配置 JSON 文件路径")
    args = ap.parse_args()

    # ========== 2. 加载配置文件 ==========
    # 读取 JSON 配置文件，使用 utf-8 编码以支持中文
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 提取三个关键部分：
    # - defaults: 全局默认配置（模型参数、阈值方法、标准化方法等）
    # - stations: 站点列表（风电场信息、风机范围）
    # - runs: 实验方案列表（每个方案可覆盖 defaults 的部分参数）
    defaults = cfg["defaults"]
    stations = cfg["stations"]
    runs     = cfg["runs"]

    # ========== 3. 打印实验计划摘要 ==========
    # 向用户展示即将执行的实验计划，便于确认配置正确
    print("========== 实验计划 ==========")
    print(f"站点数量: {len(stations)}")
    
    # 打印每个站点的基本信息
    for i,st in enumerate(stations,1):
        print(f"  [{i}] {st['name']}  CSV={st['csv']}  turbines={st['turbine_start']}..{st['turbine_end']}")
    
    print(f"Run 数量: {len(runs)}")
    
    # 打印每个实验方案的关键配置
    # 重点展示影响数据清洗效果的核心参数
    for i,r in enumerate(runs,1):
        extra=[]
        # 提取关键参数：空气密度使用方式、数据清洗轮次
        for k in ["rho_for_clean","rho_for_model","rho_input_mode","cleaning_passes"]:
            if r.get(k, None) is not None: 
                extra.append(f"{k}={r[k]}")
        
        # 提取阈值计算方法（knn / quantile_power / quantile_zresid）
        meth = r.get("thresholds",{}).get("method","knn")
        extra.append(f"thr_method={meth}")
        
        # 打印 run 的名称、输出目录和关键参数
        print(f"  ({i}) name={r.get('name','unnamed')}  out_subdir={r.get('out_subdir',r.get('name','run'))}  {' '.join(extra)}")
    
    print("================================\n")

    # ========== 4. 批量执行实验方案 ==========
    # split_cache_by_run: 存储每个 run 的数据切分信息
    # key: run 的 name，value: {(station, label): (idx_train, idx_val, idx_test)}
    # 用于支持后续 run 复用之前 run 的数据切分
    split_cache_by_run = {}
    
    # 按顺序执行每个实验方案
    for ridx, run in enumerate(runs,1):
        print(f"\n===== RUN {ridx}/{len(runs)}: {run.get('name','unnamed')} =====")
        
        # ===== 处理数据切分复用 =====
        reuse_dict = None
        if run.get("reuse_split_from"):
            # 当前 run 希望复用之前某个 run 的数据切分
            base = run["reuse_split_from"]
            
            if base not in split_cache_by_run:
                # 警告：被依赖的 run 还未执行，无法复用
                # 这通常是配置错误，应该调整 run 的执行顺序
                print(f"[Warn] 需先执行 base='{base}' 才能复用切分")
            else:
                # 从缓存中获取之前 run 的切分信息
                reuse_dict = split_cache_by_run[base]
        
        # ===== 执行当前实验方案 =====
        # run_single_run 会：
        # 1. 合并 defaults 和 run 的配置
        # 2. 遍历所有站点和风机
        # 3. 对每台风机执行数据清洗流程
        # 4. 返回每台风机的数据切分信息
        splits_saved = run_single_run(defaults, stations, run, reuse_split_dict=reuse_dict)
        
        # ===== 缓存切分信息供后续复用 =====
        split_cache_by_run[run.get("name","unnamed")] = splits_saved
    
    # ========== 5. 完成提示 ==========
    print("\n全部 RUN 执行完毕。")

if __name__ == "__main__":
    main()