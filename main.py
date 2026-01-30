# -*- coding: utf-8 -*-
import os, json, argparse
from stage2_modular.pipeline.orchestrator import run_single_run

def main():
    # 构建命令行解析器并声明一个必填参数
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="实验 JSON")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    defaults = cfg["defaults"]
    stations = cfg["stations"]
    runs     = cfg["runs"]

    print("========== 实验计划 ==========")
    print(f"站点数量: {len(stations)}")
    for i,st in enumerate(stations,1):
        print(f"  [{i}] {st['name']}  CSV={st['csv']}  turbines={st['turbine_start']}..{st['turbine_end']}")
    print(f"Run 数量: {len(runs)}")
    # 输出 run 的数量，并逐个 run 打印关键信息
    for i,r in enumerate(runs,1):
        extra=[]
        for k in ["rho_for_clean","rho_for_model","rho_input_mode","cleaning_passes"]:
            if r.get(k, None) is not None: extra.append(f"{k}={r[k]}")
        meth = r.get("thresholds",{}).get("method","knn")
        extra.append(f"thr_method={meth}")
        print(f"  ({i}) name={r.get('name','unnamed')}  out_subdir={r.get('out_subdir',r.get('name','run'))}  {' '.join(extra)}")
    print("================================\n")

    split_cache_by_run = {}
    for ridx, run in enumerate(runs,1):
        print(f"\n===== RUN {ridx}/{len(runs)}: {run.get('name','unnamed')} =====")
        reuse_dict = None
        if run.get("reuse_split_from"):
            base = run["reuse_split_from"]
            if base not in split_cache_by_run:
                print(f"[Warn] 需先执行 base='{base}' 才能复用切分")
            else:
                reuse_dict = split_cache_by_run[base]
        splits_saved = run_single_run(defaults, stations, run, reuse_split_dict=reuse_dict)
        split_cache_by_run[run.get("name","unnamed")] = splits_saved
    print("\n全部 RUN 执行完毕。")

if __name__ == "__main__":
    main()