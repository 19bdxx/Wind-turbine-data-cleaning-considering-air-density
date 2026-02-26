# -*- coding: utf-8 -*-
"""
Analyze Stage2 outputs by turbine — pipeline-aligned (reuse stage2_modular)
- 设备解析: stage2_modular.core.device.resolve_device
- ρ 读取:   stage2_modular.core.utils.load_rho_table (stations[].csv -> ['timestamp','rho'])
- 缩放器:   stage2_modular.core.scaler.Scaler (固定区间/方法来自 experiments.json)
- 切分复用: stage2_modular.core.splits.make_row_keys / load_split_csv（可选）
- 中心模型: stage2_modular.models.center.fit_mlp_center / predict_mlp_center

训练与统计口径（与你的要求一致）：
1) S_scope 为 TRUE 表示规则1-5均为FALSE，属于方法判别作用域；
2) split 仅在 S_scope==TRUE 的样本上划分；
3) S_scope==FALSE 时，Pass1/Pass2 自动 TRUE 只是占位；方法异常率只在 S_scope==TRUE 内统计；
4) 训练/验证 MLP 只用 S_scope==TRUE 且 Pass1/Pass2 均为 FALSE 的 train/val 子集。
"""

import os, re, glob, json, argparse, time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

# === 统一走主流程模块 ===
from stage2_modular.core.device import resolve_device
from stage2_modular.core.scaler import Scaler
from stage2_modular.core.utils import load_rho_table, estimate_prated_from_series
from stage2_modular.core.splits import make_row_keys, load_split_csv
from stage2_modular.models.center import fit_mlp_center, predict_mlp_center


# -------------------------
# 轻量计时/日志
# -------------------------
class Timer:
    def __init__(self, name: str = "", enabled: bool = True):
        self.name = name; self.enabled = enabled
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, et, ev, tb):
        if self.enabled:
            print(f"[Time] {self.name}: {time.perf_counter()-self.t0:.3f}s")

def log(msg: str, debug: bool):
    if debug: print(msg)


# -------------------------
# 配置 & run 发现
# -------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def discover_run_dirs(cfg: dict) -> List[Tuple[str, str]]:
    root = cfg["defaults"]["out_root"]
    return [(r.get("name","unnamed"), os.path.join(root, r.get("out_subdir", r.get("name","run"))))
            for r in cfg["runs"]]

def station_csv_map(cfg: dict) -> Dict[str, str]:
    return {s["name"]: s["csv"] for s in cfg.get("stations", [])}


# -------------------------
# 文件名解析 & 匹配
# -------------------------
def parse_station_tid(base: str) -> Tuple[Optional[str], Optional[int]]:
    m = re.match(r'^([A-Za-z0-9\u4e00-\u9fa5]+)[_-](\d{1,3})号机', base)
    if m: return m.group(1), int(m.group(2))
    m = re.match(r'^([A-Za-z0-9\u4e00-\u9fa5]+)[_-](\d{1,3})', base)
    if m: return m.group(1), int(m.group(2))
    return None, None

def iter_turbine_files(run_dir: str, pattern: str,
                       only_station: Optional[str], tid_range: Optional[Tuple[int,int]]) -> List[str]:
    files = glob.glob(os.path.join(run_dir, pattern), recursive=True)
    out = []
    for fp in files:
        st, tid = parse_station_tid(os.path.basename(fp))
        if only_station and st != only_station: continue
        if tid_range and tid is not None:
            lo, hi = tid_range
            if not (lo <= tid <= hi): continue
        out.append(fp)
    return sorted(out)


# -------------------------
# 指标（按你的口径）
# -------------------------
def compute_ratios(df: pd.DataFrame) -> Dict[str, float]:
    res = dict(abn_rules=0.0, abn_method_scope=0.0, abn_increment=0.0, abn_union=0.0,
               n_total=int(len(df)), n_scope=0)
    if df.empty: return res
    S = df.get("S_scope", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    P1 = df.get("Pass1_异常", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    P2 = df.get("Pass2_异常", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    n_total = len(df); n_scope = int(S.sum()); res["n_scope"] = n_scope
    # 只在 S 内统计方法异常，避免把 S==FALSE 的占位 TRUE 算进去
    M_abn = (P1 | P2) & S
    res["abn_rules"]         = float((~S).mean()) if n_total>0 else 0.0
    res["abn_method_scope"]  = float((M_abn[S]).mean()) if n_scope>0 else 0.0
    res["abn_increment"]     = float(M_abn.mean()) if n_total>0 else 0.0     # 占总样本
    res["abn_union"]         = float(((~S) | M_abn).mean()) if n_total>0 else 0.0
    return res


# -------------------------
# 切分（CSV 或 split_repo）
# -------------------------
def masks_from_csv(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "split" not in df.columns:
        n = len(df); z = np.zeros(n, bool); return z, z, z
    s = df["split"].astype(str).str.lower().to_numpy()
    return (s=="train"), (s=="val"), (s=="test")

def masks_from_split_repo(df: pd.DataFrame, split_dir: str, split_key: str,
                          run_name: str, file_path: str, debug: bool=False) -> Optional[Tuple[np.ndarray,np.ndarray,np.ndarray]]:
    if not (split_dir and os.path.isdir(split_dir) and split_key): return None
    stem = os.path.splitext(os.path.basename(file_path))[0]
    cands = [os.path.join(split_dir, run_name, f"{stem}__{split_key}.csv"),
             os.path.join(split_dir, f"{stem}__{split_key}.csv")]
    sp = next((p for p in cands if os.path.exists(p)), None)
    if not sp:
        log(f"[split_repo] not found for {stem}", debug); return None
    with Timer("load split_repo", debug):
        keys = make_row_keys(df)
        idx_tr, idx_va, idx_te = load_split_csv(sp, keys)
    n = len(df)
    tr = np.zeros(n, bool); tr[idx_tr]=True
    va = np.zeros(n, bool); va[idx_va]=True
    te = np.zeros(n, bool); te[idx_te]=True
    log(f"[split_repo] train={tr.sum()} val={va.sum()} test={te.sum()}", debug)
    return tr, va, te


# -------------------------
# Scaler 统一调用（兼容不同API）
# -------------------------
def apply_scaler(sc: Scaler, v: np.ndarray, r: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        # 常见API1：transform(v, r, use_rho=...)
        v_std, r_std = sc.transform(v, r, use_rho=(r is not None))
        return v_std.astype(np.float32), (r_std.astype(np.float32) if r_std is not None else None)
    except Exception:
        # 常见API2：transform_wind / transform_rho
        v_std = sc.transform_wind(v).astype(np.float32)
        r_std = sc.transform_rho(r).astype(np.float32) if r is not None else None
        return v_std, r_std


# -------------------------
# 训练/推理 统一封装（兼容两类签名）
# -------------------------
def fit_center_any(Xv_tr, Xr_tr, y_tr, Xv_va, Xr_va, y_va, use_rho: bool,
                   mlp_cfg: dict, device: str, thr_cfg: dict, prated_used: float, debug=False):
    try:
        # 常见签名A（分片式）
        return fit_mlp_center(Xv_tr, Xr_tr, y_tr, Xv_va, Xr_va, y_va,
                              use_rho=use_rho, mlp_cfg=mlp_cfg, device=device,
                              thresholds_cfg=thr_cfg, prated_used=prated_used)
    except TypeError:
        # 备选签名B（向量+掩码式）
        n = len(np.concatenate([y_tr, y_va]))
        # 拼接 & 掩码（这里仅为适配；实际项目建议统一签名）
        Xv = np.concatenate([Xv_tr, Xv_va], axis=0)
        Xr = (np.concatenate([Xr_tr, Xr_va], axis=0) if use_rho else None)
        y  = np.concatenate([y_tr, y_va], axis=0)
        tr_mask = np.zeros(n, bool); tr_mask[:len(y_tr)] = True
        va_mask = ~tr_mask
        return fit_mlp_center(Xv, Xr, y, tr_mask, va_mask,
                              mlp_cfg=mlp_cfg, device=device, thresholds_cfg=thr_cfg,
                              prated_used=prated_used, use_rho=use_rho)

def predict_center_any(model, Xv, Xr, prated: float, use_rho: bool):
    try:
        return predict_mlp_center(model, Xv, Xr, prated=prated, use_rho=use_rho)
    except TypeError:
        # 备选签名：有的实现可能不需要 prated 或参数顺序不同
        return predict_mlp_center(model, Xv, Xr)


# -------------------------
# 单风机分析
# -------------------------
def analyze_one_turbine(run_name: str, fpath: str, cfg: dict, device: str,
                        rho_map: Dict[str,str], debug=False, fast_epochs: Optional[int]=None,
                        use_split_repo=True) -> dict:
    base = os.path.basename(fpath)
    station, tid = parse_station_tid(base)
    print(f"  -> {base}  [station={station}, tid={tid}]")

    with Timer("read turbine csv", debug):
        df = pd.read_csv(fpath, encoding="utf-8")

    # —— 统计口径（只依赖CSV本身）——
    ratios = compute_ratios(df)
    log(f"     ratios: rules={ratios['abn_rules']:.4f}, method(scope)={ratios['abn_method_scope']:.4f}, "
        f"increment={ratios['abn_increment']:.4f}, union={ratios['abn_union']:.4f}, "
        f"n_total={ratios['n_total']}, n_scope={ratios['n_scope']}", debug)

    # —— 构造“洁净”训练/验证掩码（你的4点规则）——
    S  = df.get("S_scope", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    P1 = df.get("Pass1_异常", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    P2 = df.get("Pass2_异常", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    split = df.get("split", pd.Series("", index=df.index)).astype(str).str.lower().to_numpy()

    CLEAN = S & (~P1) & (~P2)
    tr_csv = CLEAN & (split=="train")
    va_csv = CLEAN & (split=="val")

    # 可选：与主流程完全一致的 split_repo（若找到，则覆盖 CSV 的 split）
    tr_mask, va_mask, te_mask = tr_csv, va_csv, CLEAN & (split=="test")
    if use_split_repo:
        sp = cfg["defaults"].get("split_repo", {})
        sp_dir, sp_key = sp.get("dir"), sp.get("key")
        maybe = masks_from_split_repo(df, sp_dir, sp_key, run_name, fpath, debug=debug)
        if maybe is not None:
            tr0, va0, te0 = maybe
            tr_mask = CLEAN & tr0
            va_mask = CLEAN & va0
            te_mask = CLEAN & te0

    # —— 基础列检测 —— 
    v_col = next((c for c in ["wind","V","v","Wind","风速"] if c in df.columns), None)
    p_col = next((c for c in ["power","P","p","Power","功率"] if c in df.columns), None)
    if v_col is None or p_col is None:
        return {"run": run_name, "station": station, "turbine": tid, "error": "no_wind_or_power"}

    v = df[v_col].to_numpy(np.float32)
    y = df[p_col].to_numpy(np.float32)

    # —— ρ 获取（优先直接用 CSV 中的 rho 列；无则再尝试左连接）——
    rho_missing_ratio_scope = 1.0
    has_rho = "rho" in df.columns
    r = None
    
    if has_rho:
        # 直接用文件里的 rho；先做数值化（防止字符串/空值）
        df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
        scope = df.get("S_scope", pd.Series(False, index=df.index)).astype(bool).to_numpy()
        denom = int(scope.sum())
        miss = int(df.loc[scope, "rho"].isna().sum()) if denom > 0 else 0
        rho_missing_ratio_scope = float(miss/denom) if denom > 0 else 1.0
        r = df["rho"].to_numpy(np.float32)
        log(f"     rho(in-file): has={has_rho} miss_scope={rho_missing_ratio_scope:.2%}", debug)
    else:
        # 没有 rho 列，再按站点 CSV 做左连接
        if station and station in rho_map:
            with Timer("load rho table & merge", debug):
                rho_tab = load_rho_table(rho_map[station], station)  # -> ['timestamp','rho']
                # 兜底：统一列名
                if "timestamp" not in rho_tab.columns:
                    for cand in ["ts","time","datetime"]:
                        if cand in rho_tab.columns:
                            rho_tab = rho_tab.rename(columns={cand:"timestamp"})
                            break
                if "rho" not in rho_tab.columns:
                    for cand in ["density","air_density","rho_kg_m3"]:
                        if cand in rho_tab.columns:
                            rho_tab = rho_tab.rename(columns={cand:"rho"})
                            break
                # 精确合并；若仍有对齐问题，可换 merge_asof 最近邻
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                rho_tab["timestamp"] = pd.to_datetime(rho_tab["timestamp"], errors="coerce")
                df = df.merge(rho_tab[["timestamp","rho"]], on="timestamp", how="left")
            has_rho = "rho" in df.columns
            if has_rho:
                df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
                scope = df.get("S_scope", pd.Series(False, index=df.index)).astype(bool).to_numpy()
                denom = int(scope.sum()); miss = int(df.loc[scope, "rho"].isna().sum()) if denom>0 else 0
                rho_missing_ratio_scope = float(miss/denom) if denom>0 else 1.0
                r = df["rho"].to_numpy(np.float32)
            log(f"     rho(merged): has={has_rho} miss_scope={rho_missing_ratio_scope:.2%}", debug)

    # —— 缩放（主流程 Scaler；按 JSON 固定区间/方法）——
    sc_cfg = cfg["defaults"].get("scaler", {})
    sc = Scaler(method=sc_cfg.get("method","minmax"),
                fixed=bool(sc_cfg.get("fixed", True)),
                wind_range=tuple(sc_cfg.get("wind_range", [0,15])),
                rho_range=tuple(sc_cfg.get("rho_range", [1.07,1.37])))
    v_std, r_std = apply_scaler(sc, v, r if has_rho else None)

    # —— 设备/MLP 配置 —— 
    mlp_cfg = dict(cfg["defaults"].get("mlp", {}))
    if fast_epochs is not None and fast_epochs>0:
        mlp_cfg["epochs"] = min(int(mlp_cfg.get("epochs", 300)), int(fast_epochs))  # 仅本次分析限速
    device = str(resolve_device(cfg["defaults"].get("device", "cuda:0")))
    thr_cfg = cfg["defaults"].get("thresholds", {})
    prated_used = float(estimate_prated_from_series(df[p_col]))

    # —— A) v->P —— 
    tr_mse_v = va_mse_v = np.nan
    if tr_mask.any() and va_mask.any():
        Xv_tr, y_tr = v_std[tr_mask].reshape(-1,1), y[tr_mask]
        Xv_va, y_va = v_std[va_mask].reshape(-1,1), y[va_mask]
        model_v, _ = fit_center_any(Xv_tr, None, y_tr, Xv_va, None, y_va,
                                    use_rho=False, mlp_cfg=mlp_cfg, device=device,
                                    thr_cfg=thr_cfg, prated_used=prated_used, debug=debug)
        with torch.no_grad():
            yhat_tr = predict_center_any(model_v, Xv_tr, None, prated_used, use_rho=False)
            yhat_va = predict_center_any(model_v, Xv_va, None, prated_used, use_rho=False)
        tr_mse_v = float(np.mean((yhat_tr - y_tr)**2))
        va_mse_v = float(np.mean((yhat_va - y_va)**2))
        log(f"     MLP(v->P): train={tr_mse_v:.2f}  val={va_mse_v:.2f}", debug)
    else:
        log("     [Info] v->P: train/val empty", debug)

    # —— B) v+ρ->P —— 
    tr_mse_vr = va_mse_vr = np.nan
    if has_rho:
        ok = ~np.isnan(df["rho"].to_numpy())
        TR_vr = tr_mask & ok
        VA_vr = va_mask & ok
        if TR_vr.any() and VA_vr.any():
            Xv_tr, Xr_tr, y_tr = v_std[TR_vr], r_std[TR_vr], y[TR_vr]
            Xv_va, Xr_va, y_va = v_std[VA_vr], r_std[VA_vr], y[VA_vr]
            Xv_tr = Xv_tr.reshape(-1,1); Xv_va = Xv_va.reshape(-1,1)
            model_vr, _ = fit_center_any(Xv_tr, Xr_tr, y_tr, Xv_va, Xr_va, y_va,
                                         use_rho=True, mlp_cfg=mlp_cfg, device=device,
                                         thr_cfg=thr_cfg, prated_used=prated_used, debug=debug)
            with torch.no_grad():
                yhat_tr = predict_center_any(model_vr, Xv_tr, Xr_tr, prated_used, use_rho=True)
                yhat_va = predict_center_any(model_vr, Xv_va, Xr_va, prated_used, use_rho=True)
            tr_mse_vr = float(np.mean((yhat_tr - y_tr)**2))
            va_mse_vr = float(np.mean((yhat_va - y_va)**2))
            log(f"     MLP(v,rho->P): train={tr_mse_vr:.2f}  val={va_mse_vr:.2f}", debug)
        else:
            log("     [Info] v,rho->P: train/val empty (rho NA)", debug)

    return {
        "run": run_name,
        "station": station,
        "turbine": tid,
        "__file__": fpath,
        "n_total": ratios["n_total"],
        "n_scope": ratios["n_scope"],
        "abn_rules": ratios["abn_rules"],
        "abn_method_scope": ratios["abn_method_scope"],
        "abn_increment": ratios["abn_increment"],
        "abn_union": ratios["abn_union"],
        "mlp_v_train_mse": tr_mse_v,
        "mlp_v_val_mse": va_mse_v,
        "mlp_vr_train_mse": tr_mse_vr,
        "mlp_vr_val_mse": va_mse_vr,
        "has_rho_joined": bool(has_rho),
        "rho_missing_ratio_scope": float(rho_missing_ratio_scope),
    }


# -------------------------
# run 级加权汇总（按 n_scope 加权）
# -------------------------
def aggregate_runs(df_turb: pd.DataFrame) -> pd.DataFrame:
    if df_turb.empty: return pd.DataFrame()
    w = df_turb["n_scope"].fillna(0).astype(float)
    cols = ["abn_rules","abn_method_scope","abn_increment","abn_union",
            "mlp_v_train_mse","mlp_v_val_mse","mlp_vr_train_mse","mlp_vr_val_mse",
            "rho_missing_ratio_scope"]
    def wavg(s): 
        ww=w.loc[s.index]; den=ww.sum(); 
        return np.nan if den<=0 else float(np.nansum(s*ww)/den)
    out = (df_turb.groupby("run", as_index=False)
           .agg(**{c:(c, wavg) for c in cols},
                n_total=("n_total","sum"),
                n_scope=("n_scope","sum"),
                turbines=("turbine","nunique")))
    return out


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pattern", default="**/*_stage2_mlp.csv")
    ap.add_argument("--station", default=None)
    ap.add_argument("--turbine_range", nargs=2, type=int, default=None)
    ap.add_argument("--out_turbines", default="turbine_results.csv")
    ap.add_argument("--out_runs", default="run_summary.csv")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fast_epochs", type=int, default=None, help="仅本次分析限制 epochs（不改 JSON）")
    ap.add_argument("--no_split_repo", action="store_true", help="禁用 split_repo，对齐 CSV 的 split 列")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dirs = discover_run_dirs(cfg)
    rho_map = station_csv_map(cfg)

    # 设备信息
    dev = str(resolve_device(cfg["defaults"].get("device","cuda:0")))
    print(f"[Env] torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try: print(f"[Env] GPU: {torch.cuda.get_device_name(0)}  device={dev}")
        except Exception: print(f"[Env] device={dev}")

    tid_rng = (int(args.turbine_range[0]), int(args.turbine_range[1])) if args.turbine_range else None

    rows = []
    for run_name, run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"[Warn] run 目录不存在: {run_dir}"); continue
        files = iter_turbine_files(run_dir, args.pattern, args.station, tid_rng)
        print(f"\n=== Analyze run: {run_name}  (files={len(files)}) ===")
        if not files:
            print(f"[Info] 未发现文件: {run_dir}/{args.pattern}"); continue
        for i, fp in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {fp}")
            row = analyze_one_turbine(run_name, fp, cfg, dev, rho_map,
                                      debug=args.debug, fast_epochs=args.fast_epochs,
                                      use_split_repo=(not args.no_split_repo))
            rows.append(row)

    df_turb = pd.DataFrame(rows)
    order = ["run","station","turbine","__file__",
             "n_total","n_scope","abn_rules","abn_method_scope","abn_increment","abn_union",
             "mlp_v_train_mse","mlp_v_val_mse","mlp_vr_train_mse","mlp_vr_val_mse",
             "has_rho_joined","rho_missing_ratio_scope"]
    cols = [c for c in order if c in df_turb.columns] + [c for c in df_turb.columns if c not in order]
    df_turb = df_turb[cols]
    print("\n=== Turbine-level results (head) ===")
    with pd.option_context('display.max_rows', 40, 'display.max_columns', None, 'display.width', 200):
        print(df_turb.head(20))
    df_turb.to_csv(args.out_turbines, index=False, encoding="utf-8")
    print(f"\n已保存逐风机结果到：{args.out_turbines}")

    df_runs = aggregate_runs(df_turb)
    if not df_runs.empty:
        print("\n=== Run-level weighted summary ===")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
            print(df_runs)
        df_runs.to_csv(args.out_runs, index=False, encoding="utf-8")
        print(f"\n已保存 run 聚合结果到：{args.out_runs}")
    else:
        print("\n[Info] run 聚合结果为空（可能输入为空或全部错误行）")


if __name__ == "__main__":
    main()
