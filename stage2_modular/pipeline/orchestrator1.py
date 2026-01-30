# -*- coding: utf-8 -*-
import os, sys, math, json, gc
import numpy as np
import pandas as pd

from ..core.utils import Stopwatch, read_csv_any, load_rho_table, estimate_prated_from_series
from ..core.splits import make_row_keys, save_split_csv, load_split_csv
from ..core.scaler import Scaler
from ..core.dmode import build_D_np
from ..models.center import fit_mlp_center, predict_mlp_center
from ..thresholds.registry import get_method

# --- Added by patch: align boolean masks to subset indices ---
def _align_mask_to_index(mask_full, full_index, target_index, fill=False):
    """
    Align a boolean mask defined on the full index to a subset index.
    - mask_full: array-like or pd.Series of booleans for full_index
    - full_index: pd.Index of the full frame
    - target_index: array-like of labels or an Index to align to
    Returns a numpy bool array with length == len(target_index).
    """
    import pandas as _pd
    import numpy as _np
    if isinstance(target_index, _pd.Index):
        tgt_idx = target_index
    else:
        tgt_idx = _pd.Index(target_index)
    if isinstance(full_index, _pd.Index):
        full_idx = full_index
    else:
        full_idx = _pd.Index(full_index)

    if isinstance(mask_full, (_np.ndarray, list, tuple)):
        s = _pd.Series(mask_full, index=full_idx, dtype="boolean")
    elif isinstance(mask_full, _pd.Series):
        s = mask_full.astype("boolean")
        # Ensure it covers full index
        if not s.index.equals(full_idx):
            s = s.reindex(full_idx)
    else:
        raise TypeError(f"Unsupported mask type: {type(mask_full)}")

    s_sub = s.reindex(tgt_idx, fill_value=bool(fill)).astype("boolean")
    return s_sub.to_numpy(dtype=bool)


def run_stage2_for_station(st_cfg, global_cfg, run_cfg, reuse_split=None):
    station = st_cfg["name"]; wide_csv = st_cfg["csv"]
    print(f"\n====== Stage2-modular: {station} (run={run_cfg.get('name','unnamed')}) ======")
    T=Stopwatch()

    stage1_root = run_cfg["stage1_root"]
    out_root    = run_cfg["out_root"]
    run_tag     = run_cfg.get("out_subdir", run_cfg.get("name", "run"))
    run_dir     = os.path.join(out_root, run_tag); os.makedirs(run_dir, exist_ok=True)

    cleaning_passes = int(run_cfg.get("cleaning_passes", 2))
    rho_for_clean   = bool(run_cfg.get("rho_for_clean", False))
    rho_for_model   = bool(run_cfg.get("rho_for_model", False))
    rho_input_mode  = run_cfg.get("rho_input_mode", "normal")
    rho_shuffle_cfg = run_cfg.get("rho_shuffle", {"seed":42,"granularity":"split"})
    rho_const_value = run_cfg.get("rho_constant_value", "train_mean")
    emit_clean = bool(run_cfg.get("emit_clean_artifacts", True))

    device = run_cfg.get("device","auto")
    split_cfg  = run_cfg["split"]; seed = int(run_cfg.get("seed", 42)); np.random.seed(seed)
    wind_scope = tuple(run_cfg.get("wind_scope", [0.0, 15.0]))
    min_train_samples = int(run_cfg.get("min_train_samples", 1000))
    gpu_cache_bytes = int(run_cfg.get("gpu_cache_mib", 20480)) * 1024**2
    # persistent split config
    split_repo = run_cfg.get("split_repo", {})
    split_dir  = split_repo.get("dir", os.path.join(out_root, "_splits"))
    split_key  = split_repo.get("key", run_cfg.get("reuse_split_from", run_cfg.get("name","default")))
    split_load = bool(split_repo.get("load", True))
    split_save = bool(split_repo.get("save", True))

    scfg = run_cfg.get("scaler", {"method":"minmax","fixed":True,"wind_range":[0,15],"rho_range":[1.07,1.37]})
    scaler = Scaler(method=scfg.get("method","minmax"), fixed=bool(scfg.get("fixed",True)),
                    wind_range=tuple(scfg.get("wind_range",[0,15])), rho_range=tuple(scfg.get("rho_range",[1.07,1.37])))

    th = run_cfg["thresholds"]
    method_name = th.get("method","knn")
    BW_V = float(th.get("bw_v", 0.10)); BW_R = float(th.get("bw_r", 0.10))
    TAU_HI=float(th.get("tau_hi",0.98)); TAU_LO=float(th.get("tau_lo",0.98))
    EPS_RATIO=float(th.get("eps_ratio",0.05)); DELTA_POWER=float(th.get("delta_power",50.0))
    D_MODE = th.get("D_mode","pred_or_both")
    bw_mode = th.get("bw_mode","equiv_power")

    mlp_cfg = run_cfg["mlp"]

    rho_tab=None
    if (rho_for_clean or rho_for_model) or (rho_input_mode in ["shuffle","constant"]):
        rho_tab=load_rho_table(wide_csv, station)

    stage1_dir = os.path.join(stage1_root, station)
    station_out = os.path.join(run_dir, f"{station}_mlp"); os.makedirs(station_out, exist_ok=True)

    def split_indices_by_ratio(df_sorted_index, split=(0.8,0.2,0.0)):
        tr,va,te = split; n=len(df_sorted_index); i_tr=int(round(tr*n)); i_va=int(round((tr+va)*n))
        idx_train=df_sorted_index[:i_tr]; idx_val=df_sorted_index[i_tr:i_va]; idx_test=df_sorted_index[i_va:] if te>1e-12 else np.array([],dtype=df_sorted_index.dtype)
        return idx_train, idx_val, idx_test

    splits_saved={}
    for tid in range(int(st_cfg["turbine_start"]), int(st_cfg["turbine_end"])+1):
        print(f">>> 处理 {station} {tid}号机 ..."); sw=Stopwatch()
        label=f"{tid}号机"; in_masks=os.path.join(stage1_dir, f"{station}_{label}_masks.csv")
        if not os.path.exists(in_masks):
            print(f"   ⚠ 未找到 {os.path.basename(in_masks)}，跳过"); continue

        df = read_csv_any(in_masks); sw.lap("read turbine masks csv")
        df["timestamp"]=pd.to_datetime(df["timestamp"], errors="coerce")
        df=df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if rho_tab is not None:
            df=df.merge(rho_tab, on="timestamp", how="left"); sw.lap("merge rho")

        ws_min,ws_max=wind_scope
        in_scope=(df["wind"].astype(float)>=ws_min) & (df["wind"].astype(float)<=ws_max)
        rule_cols=[c for c in ["规则1","规则2","规则3","规则4","规则5"] if c in df.columns]
        R=pd.Series(False,index=df.index); 
        for c in rule_cols: R|=df[c].fillna(False).astype(bool)
        S_scope = (~R) & in_scope
        df["S_scope"]=S_scope; sw.lap("scope filter")

        prated_raw = estimate_prated_from_series(df.loc[S_scope,"power"]); sw.lap("estimate P_rated")

        S = df.loc[S_scope].copy()

        # ---- Persistent splits ----
        split_path = os.path.join(split_dir, station, f"{tid:03d}_{split_key}.csv")
        idx_train = idx_val = idx_test = None
        did_load_persist = False
        try:
            if split_load and os.path.exists(split_path):
                row_keys = make_row_keys(S)
                idx_train, idx_val, idx_test = load_split_csv(split_path, row_keys)
                if len(idx_train) + len(idx_val) + len(idx_test) > 0:
                    did_load_persist = True
                    print(f"[Split] loaded persisted split: {split_path}  (train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)})")
        except Exception as e:
            print(f"[Split] failed to load persisted split ({split_path}): {e}")

        if rho_for_clean or rho_for_model:
            S["rho"]=pd.to_numeric(S["rho"], errors="coerce")
        else:
            S["rho"]=np.nan

        def arr(ix,col): return S.loc[ix,col].to_numpy(dtype=float)

        if reuse_split is not None and (station,label) in reuse_split:
            idx_train, idx_val, idx_test = reuse_split[(station,label)]
            print(f"[Split] reuse from '{run_cfg.get('reuse_split_from','?')}' for {station}-{tid}")
        else:
            strategy = split_cfg.get("strategy","shuffle")
            ratio    = tuple(split_cfg.get("ratio",[0.8,0.2,0.0]))
            if strategy=="time":
                idx_ordered=S.sort_values("timestamp").index.values
            elif strategy=="shuffle":
                idx_ordered=S.sample(frac=1.0, random_state=seed).index.values
            elif strategy=="block_shuffle":
                freq=split_cfg.get("block_freq","D")
                ts=pd.to_datetime(S["timestamp"], errors="coerce")
                block_id=ts.dt.floor(freq); S2=S.copy(); S2["_block"]=block_id
                S2=S2.sort_values(["_block","timestamp"]).reset_index()
                blocks=S2["_block"].dropna().unique()
                if len(blocks)==0:
                    idx_ordered=S.sample(frac=1.0, random_state=seed).index.values
                else:
                    rng=np.random.default_rng(seed); rng.shuffle(blocks)
                    out_idx=[S2.loc[S2["_block"]==b,"index"].to_numpy() for b in blocks]
                    idx_ordered=np.concatenate(out_idx, axis=0)
            else:
                raise ValueError(f"未知 SPLIT_STRATEGY={strategy}")
            idx_train,idx_val,idx_test=split_indices_by_ratio(idx_ordered, ratio)
            try:
                if split_save:
                    row_keys = make_row_keys(S)
                    os.makedirs(os.path.dirname(split_path), exist_ok=True)
                    save_split_csv(split_path, row_keys, idx_train, idx_val, idx_test)
                    print(f"[Split] persisted to: {split_path}")
            except Exception as e:
                print(f"[Split] failed to save persisted split ({split_path}): {e}")
            splits_saved[(station,label)]=(idx_train,idx_val,idx_test)
        sw.lap("split index")

        S["_split"] = pd.Series(pd.NA, index=S.index, dtype="string")
        S.loc[idx_train, "_split"] = "train"
        S.loc[idx_val,   "_split"] = "val"
        if len(idx_test) > 0:
            S.loc[idx_test,  "_split"] = "test"

        if len(S) < int(run_cfg.get("min_train_samples",1000)):
            print(f"   ⚠ 样本不足：{len(S)}<{run_cfg.get('min_train_samples',1000)}，跳过")
            continue

        scaler.fit_from_train(S.loc[idx_train,"wind"], S.loc[idx_train,"rho"] if (rho_for_clean or rho_for_model) else None, (rho_for_clean or rho_for_model))

        Xv_tr_raw=arr(idx_train,"wind"); Xv_va_raw=arr(idx_val,"wind"); Xv_S_raw=S["wind"].to_numpy(float)

        def make_rho_model_array(base_series, ix, split_tag:str):
            rng = np.random.default_rng(rho_shuffle_cfg.get("seed", 42) + hash((station,tid,split_tag))%100000)
            vals = base_series.loc[ix].to_numpy(float)
            if rho_input_mode=="normal":
                return vals
            if rho_input_mode=="constant":
                if isinstance(rho_const_value, (int,float)):
                    const=float(rho_const_value)
                elif str(rho_const_value)=="train_mean":
                    const=float(np.nanmean(base_series.loc[idx_train].to_numpy(float))) if len(idx_train)>0 else 1.225
                else:
                    const=1.225
                return np.full_like(vals, const, dtype=float)
            if rho_input_mode=="shuffle":
                perm = vals.copy(); rng.shuffle(perm); return perm
            return vals

        if rho_for_model:
            rho_series=S["rho"]
            Xr_tr_raw_model = make_rho_model_array(rho_series, idx_train, "train")
            Xr_va_raw_model = make_rho_model_array(rho_series, idx_val,   "val")
            Xr_S_raw_model  = make_rho_model_array(rho_series, S.index.values, "all")
        else:
            Xr_tr_raw_model = Xr_va_raw_model = Xr_S_raw_model = None

        if rho_for_clean:
            Xr_tr_raw_clean = arr(idx_train,"rho"); Xr_va_raw_clean = arr(idx_val,"rho"); Xr_S_raw_clean = S["rho"].to_numpy(float)
        else:
            Xr_tr_raw_clean = Xr_va_raw_clean = Xr_S_raw_clean = np.zeros(len(Xv_S_raw), dtype=float)

        y_tr=arr(idx_train,"power"); y_va=arr(idx_val,"power")

        Xv_tr_m, Xr_tr_m = scaler.transform(Xv_tr_raw, Xr_tr_raw_model, rho_for_model)
        Xv_va_m, Xr_va_m = scaler.transform(Xv_va_raw, Xr_va_raw_model, rho_for_model)
        Xv_tr_c, Xr_tr_c = scaler.transform(Xv_tr_raw, Xr_tr_raw_clean, rho_for_clean)
        Xv_va_c, Xr_va_c = scaler.transform(Xv_va_raw, Xr_va_raw_clean, rho_for_clean)
        sw.lap("standardize & arrays ready")

        mdl_p1, train_mode1 = fit_mlp_center(
            Xv_tr=Xv_tr_m, Xr_tr=(Xr_tr_m if rho_for_model else None), y_tr=y_tr,
            Xv_va=Xv_va_m, Xr_va=(Xr_va_m if rho_for_model else None), y_va=y_va,
            use_rho=rho_for_model, mlp_cfg=mlp_cfg, device=device, verbose=1, gpu_cache_limit_bytes=gpu_cache_bytes,
            thresholds_cfg=th, prated_used=prated_raw)
        sw.lap("fit center (Pass1)")

        Sv_s_m, Sr_s_m = scaler.transform(Xv_S_raw, Xr_S_raw_model if rho_for_model else None, rho_for_model)
        pred_p1 = predict_mlp_center(mdl_p1, Sv_s_m, (Sr_s_m if rho_for_model else None), prated_raw, rho_for_model)
        res_p1 = S["power"].to_numpy(float) - pred_p1

        if cleaning_passes == 0:
            df_out = df.copy()
            for c in ["pred_center_p1","pred_center","residual","D","thr1_pos","thr1_neg","thr_pos","thr_neg","exceed_ratio","Pass1_异常","Pass2_异常"]:
                if c not in df_out.columns: df_out[c]=np.nan
            df_out.loc[S.index,"pred_center_p1"]=pred_p1
            df_out.loc[S.index,"pred_center"]=pred_p1
            df_out.loc[S.index,"residual"]=res_p1
            df_out.loc[S.index,"D"]=np.nan
            df_out.loc[S.index,"thr1_pos"]=np.nan; df_out.loc[S.index,"thr1_neg"]=np.nan
            df_out.loc[S.index,"thr_pos"]=np.nan;  df_out.loc[S.index,"thr_neg"]=np.nan
            df_out.loc[S.index,"exceed_ratio"]=np.nan
            # initialize boolean columns to avoid dtype warnings
            if "Pass1_异常" not in df_out.columns:
                df_out["Pass1_异常"] = pd.Series(False, index=df_out.index, dtype="boolean")
            if "Pass2_异常" not in df_out.columns:
                df_out["Pass2_异常"] = pd.Series(False, index=df_out.index, dtype="boolean")
            df_out.loc[S.index,"Pass1_异常"]=pd.Series(False, index=S.index, dtype="boolean")
            df_out.loc[S.index,"Pass2_异常"]=pd.Series(False, index=S.index, dtype="boolean")
            if "split" not in df_out.columns:
                df_out["split"] = pd.Series(pd.array([pd.NA]*len(df_out), dtype="string"), index=df_out.index)
            df_out.loc[S.index,"split"] = S["_split"].astype("string")
            turb_out=os.path.join(station_out, f"{tid}号机"); os.makedirs(turb_out, exist_ok=True)
            out_csv=os.path.join(turb_out, f"{station}_{label}_stage2_mlp.csv")
            df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
            continue

        pr_used = prated_raw if math.isfinite(prated_raw) else float(np.nanmax(S["power"]))
        if not math.isfinite(pr_used): pr_used = float(np.nanmedian(S["power"])) if np.isfinite(np.nanmedian(S["power"])) else 1000.0
        D_all = build_D_np(pred_p1, pr_used, D_MODE, EPS_RATIO, DELTA_POWER)
        zpos=np.clip(res_p1,0.0,None)/D_all; zneg=np.clip(-res_p1,0.0,None)/D_all

        Sv_s_c, Sr_s_c = scaler.transform(Xv_S_raw, Xr_S_raw_clean, rho_for_clean)
        train_X = np.c_[Xv_tr_c, (Xr_tr_c if rho_for_clean else np.zeros_like(Xv_tr_c))]
        pos_in_S = S.index.get_indexer(idx_train)
        train_zp = zpos[pos_in_S]; train_zn = zneg[pos_in_S]
        query_X  = np.c_[Sv_s_c, (Sr_s_c if rho_for_clean else np.zeros_like(Sv_s_c))]

        if th.get("bw_mode","equiv_power")=="fixed":
            bw_v_vec=float(BW_V); bw_r_vec=float(BW_R)
        else:
            scale_ratio=scaler.scale_ratio_r_over_v()
            rho_clean_raw = Xr_S_raw_clean if rho_for_clean else np.ones_like(Xv_S_raw)
            rho_safe=np.where(rho_clean_raw>1e-8, rho_clean_raw, 1.0)
            ratio = (scale_ratio * (Xv_S_raw / (3.0 * rho_safe)))
            bw_v_vec=np.clip(BW_R * ratio, float(th.get("bw_v_floor",1e-4)), float(th.get("bw_v_cap",10.0)))
            bw_r_vec=float(BW_R)

        method = get_method(method_name)
        outs1 = method.compute(
            train_X=train_X, train_zp=train_zp, train_zn=train_zn,
            query_X=query_X, D_all=D_all,
            idx_train_mask=S.index.isin(idx_train), idx_val_mask=S.index.isin(idx_val),
            taus=(TAU_LO, TAU_HI),
            cfg={**th,
                 "bw_v_vec":bw_v_vec, "bw_r_vec":bw_r_vec, "residuals":res_p1,
                 "wind_std_for_model": Sv_s_m, "rho_std_for_model": (Sr_s_m if rho_for_model else None),
                 "wind_std_for_clean": Sv_s_c, "rho_std_for_clean": (Sr_s_c if rho_for_clean else None),
                 "y": S["power"].to_numpy(float), "y_hat": pred_p1},
            device=device
        )
        is_p1 = outs1.is_abnormal
        thr1_pos, thr1_neg = outs1.thr_pos, outs1.thr_neg

        is_p1_train = _align_mask_to_index(is_p1, S.index, idx_train, fill=False)
        keep_idx_tr = idx_train[~is_p1_train]
        if len(keep_idx_tr) >= min_train_samples:
            Xv_tr2_raw = S.loc[keep_idx_tr,"wind"].to_numpy(float)
            if rho_for_model:
                rho_series=S["rho"]; Xr_tr2_raw_model = rho_series.loc[keep_idx_tr].to_numpy(float)
            else:
                Xr_tr2_raw_model=None
            Xv_tr2_m, Xr_tr2_m = scaler.transform(Xv_tr2_raw, Xr_tr2_raw_model, rho_for_model)
            y_tr2 = S.loc[keep_idx_tr,"power"].to_numpy(float)
            mdl_center, train_mode2 = fit_mlp_center(Xv_tr2_m, (Xr_tr2_m if rho_for_model else None), y_tr2,
                                                     Xv_va_m, (Xr_va_m if rho_for_model else None), y_va,
                                                     use_rho=rho_for_model, mlp_cfg=mlp_cfg, device=device, verbose=0, gpu_cache_limit_bytes=gpu_cache_bytes,
                                                     thresholds_cfg=th, prated_used=pr_used)
        else:
            mdl_center=mdl_p1

        pred_c = predict_mlp_center(mdl_center, Sv_s_m, (Sr_s_m if rho_for_model else None), pr_used, rho_for_model)
        res_c = S["power"].to_numpy(float) - pred_c
        D_all2 = build_D_np(pred_c, pr_used, D_MODE, EPS_RATIO, DELTA_POWER)
        zpos2=np.clip(res_c,0.0,None)/D_all2; zneg2=np.clip(-res_c,0.0,None)/D_all2

        keep_train_mask = S.index.isin(keep_idx_tr)
        v_keep = S.loc[keep_train_mask,"wind"].to_numpy(float)
        r_keep = S.loc[keep_train_mask,"rho"].to_numpy(float) if rho_for_clean else None
        v_keep_s, r_keep_s = scaler.transform(v_keep, r_keep, rho_for_clean)
        train2_X = np.c_[v_keep_s, (r_keep_s if rho_for_clean else np.zeros_like(v_keep_s))]
        train2_zp = zpos2[keep_train_mask]; train2_zn = zneg2[keep_train_mask]

        outs2 = method.compute(
            train_X=train2_X, train_zp=train2_zp, train_zn=train2_zn,
            query_X=query_X, D_all=D_all2,
            idx_train_mask=keep_train_mask, idx_val_mask=(S.index.isin(idx_val) & (~is_p1)),
            taus=(TAU_LO, TAU_HI),
            cfg={**th,
                 "bw_v_vec":bw_v_vec, "bw_r_vec":bw_r_vec, "residuals":res_c,
                 "wind_std_for_model": Sv_s_m, "rho_std_for_model": (Sr_s_m if rho_for_model else None),
                 "wind_std_for_clean": Sv_s_c, "rho_std_for_clean": (Sr_s_c if rho_for_clean else None),
                 "y": S["power"].to_numpy(float), "y_hat": pred_c},
            device=device
        )
        is_p2 = np.zeros_like(is_p1, dtype=bool); rem_mask=(~is_p1)
        is_p2[rem_mask] = outs2.is_abnormal[rem_mask]

        for c in ["pred_center_p1","pred_center","residual","D","thr1_pos","thr1_neg","thr_pos","thr_neg","exceed_ratio","Pass1_异常","Pass2_异常"]:
            if c not in df.columns: df[c]=np.nan
        df.loc[S.index,"pred_center_p1"]=pred_p1
        df.loc[S.index,"pred_center"]=pred_c
        df.loc[S.index,"residual"]=res_c
        df.loc[S.index,"D"]=D_all2
        df.loc[S.index,"thr1_pos"]=thr1_pos; df.loc[S.index,"thr1_neg"]=thr1_neg
        df.loc[S.index,"thr_pos"]=outs2.thr_pos;  df.loc[S.index,"thr_neg"]=outs2.thr_neg
        df.loc[S.index,"Pass1_异常"]=pd.Series(is_p1, index=S.index, dtype="boolean")
        df.loc[S.index,"Pass2_异常"]=pd.Series(is_p2, index=S.index, dtype="boolean")

        Tpos=df.loc[S.index,"thr_pos"].to_numpy(float); Tneg=np.abs(df.loc[S.index,"thr_neg"].to_numpy(float))
        r = df.loc[S.index,"residual"].to_numpy(float)
        ex_ratio = np.maximum((r - Tpos)/np.maximum(Tpos,1e-6), (-r - Tneg)/np.maximum(Tneg,1e-6))
        ex_ratio=np.clip(ex_ratio,0.0,None); df.loc[S.index,"exceed_ratio"]=ex_ratio

        if "split" not in df.columns:
            df["split"] = pd.Series(pd.array([pd.NA]*len(df), dtype="string"), index=df.index)
        df.loc[S.index,"split"] = S["_split"].astype("string")

        turb_out=os.path.join(station_out, f"{tid}号机"); os.makedirs(turb_out, exist_ok=True)
        out_csv=os.path.join(turb_out, f"{station}_{label}_stage2_mlp.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        sw.lap("write csv")

    T.total(f"{station} station total")
    return splits_saved

def run_single_run(defaults, stations, run_cfg, reuse_split_dict=None):
    merged={}; merged.update(defaults); merged.update(run_cfg)
    for k in ["split","mlp","thresholds","scaler"]:
        if k in defaults or k in run_cfg:
            merged[k]={**defaults.get(k,{}), **run_cfg.get(k,{})}
    all_splits={}
    for st in stations:
        try:
            reuse=None
            if reuse_split_dict is not None and run_cfg.get("reuse_split_from") is not None:
                reuse = reuse_split_dict.get(st["name"], {})
            splits_saved = run_stage2_for_station(st, defaults, merged, reuse_split=reuse)
            all_splits.setdefault(st["name"],{}).update(splits_saved)
        except Exception as e:
            print(f"❌ 处理失败 [{st['name']}] @ run={run_cfg.get('name','unnamed')}: {e}", file=sys.stderr)
    return all_splits
