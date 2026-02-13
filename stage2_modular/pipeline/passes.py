# stage2_modular/pipeline/passes.py
# -*- coding: utf-8 -*-
"""
Pass execution logic for Stage 2 pipeline.
Handles Pass 1 and Pass 2 threshold computation and model training.
"""
import numpy as np
from ..core.dmode import build_D_np
from ..models.center import fit_mlp_center, predict_mlp_center
from ..thresholds.registry import get_method


def execute_pass1(S, idx_train, idx_val, Xv_tr_m, Xr_tr_m, y_tr, Xv_va_m, Xr_va_m, y_va,
                  Sv_s_m, Sr_s_m, Xv_tr_c, Xr_tr_c, Sv_s_c, Sr_s_c,
                  rho_for_model, rho_for_clean, prated_raw, mlp_cfg, th, device,
                  gpu_cache_bytes, scaler, predict_fn_builder):
    """
    Execute Pass 1: Train center model and compute initial thresholds.
    
    Parameters
    ----------
    S : pd.DataFrame
        Scope data
    idx_train, idx_val : array-like
        Training and validation indices
    Xv_tr_m, Xr_tr_m, y_tr : array-like
        Training data (model space)
    Xv_va_m, Xr_va_m, y_va : array-like
        Validation data (model space)
    Sv_s_m, Sr_s_m : array-like
        Full data (model space)
    Xv_tr_c, Xr_tr_c : array-like
        Training data (clean space)
    Sv_s_c, Sr_s_c : array-like
        Full data (clean space)
    rho_for_model : bool
        Whether model uses density
    rho_for_clean : bool
        Whether cleaning uses density
    prated_raw : float
        Rated power estimate
    mlp_cfg : dict
        MLP configuration
    th : dict
        Threshold configuration
    device : str
        Device for computation
    gpu_cache_bytes : int
        GPU cache limit
    scaler : Scaler
        Feature scaler
    predict_fn_builder : callable
        Function to build prediction functions
        
    Returns
    -------
    dict
        Dictionary containing:
        - model: trained model
        - predictions: predictions on full data
        - residuals: residuals on full data
        - D_all: normalization scales
        - zpos, zneg: z-scores
        - thresholds: threshold outputs
        - thr_pos, thr_neg: threshold values
        - is_abnormal: boolean mask
    """
    from ..core.utils import estimate_prated_from_series
    
    # Train Pass1 center model
    mdl_p1, train_mode1 = fit_mlp_center(
        Xv_tr=Xv_tr_m, Xr_tr=(Xr_tr_m if rho_for_model else None), y_tr=y_tr,
        Xv_va=Xv_va_m, Xr_va=(Xr_va_m if rho_for_model else None), y_va=y_va,
        use_rho=rho_for_model, mlp_cfg=mlp_cfg, device=device, verbose=1,
        gpu_cache_limit_bytes=gpu_cache_bytes,
        thresholds_cfg=th,
        prated_used=estimate_prated_from_series(S.loc[idx_train, "power"])
    )
    
    # Predict on full data
    pred_p1 = predict_mlp_center(
        mdl_p1, Sv_s_m, (Sr_s_m if rho_for_model else None),
        prated_raw, rho_for_model
    )
    res_p1 = S["power"].to_numpy(float) - pred_p1
    
    # Compute D scales
    import math
    pr_used = prated_raw if math.isfinite(prated_raw) else float(np.nanmax(S["power"]))
    if not math.isfinite(pr_used):
        pr_used = (float(np.nanmedian(S["power"]))
                  if np.isfinite(np.nanmedian(S["power"])) else 1000.0)
    
    D_MODE = th.get("D_mode", "pred_or_both")
    EPS_RATIO = float(th.get("eps_ratio", 0.05))
    DELTA_POWER = float(th.get("delta_power", 50.0))
    D_all = build_D_np(pred_p1, pr_used, D_MODE, EPS_RATIO, DELTA_POWER)
    
    zpos = np.clip(res_p1, 0.0, None) / D_all
    zneg = np.clip(-res_p1, 0.0, None) / D_all
    
    # Prepare KNN inputs
    train_X = np.c_[Xv_tr_c, (Xr_tr_c if rho_for_clean else np.zeros_like(Xv_tr_c))]
    pos_in_S = S.index.get_indexer(idx_train)
    train_zp = zpos[pos_in_S]
    train_zn = zneg[pos_in_S]
    query_X = np.c_[Sv_s_c, (Sr_s_c if rho_for_clean else np.zeros_like(Sv_s_c))]
    
    # Build minmax parameters for physics
    if scaler.method == "minmax":
        a = np.array([scaler.V_min, scaler.R_min], dtype=float)
        b = np.array([scaler.V_max - scaler.V_min,
                     scaler.R_max - scaler.R_min], dtype=float)
    else:
        a = np.array([0.0, 0.0], dtype=float)
        b = np.array([1.0, 1.0], dtype=float)
    
    # Build prediction functions
    predict_fn_z, predict_torch_z = predict_fn_builder(
        mdl_p1, scaler, rho_for_model, rho_for_clean, pr_used
    )
    
    # Determine grad_mode
    METRIC = th.get("metric", "tanorm")
    LAMBDA_T = float(th.get("lambda_t", 6.0))
    GRAD_MODE = th.get("grad_mode", "auto")
    GRAD_EPS = th.get("grad_eps", 0.1)
    PHYS_REL = bool(th.get("physics_relative", True))
    
    grad_mode_for_pass1 = (
        GRAD_MODE if (((rho_for_clean and rho_for_model) or
                      ((not rho_for_clean) and (not rho_for_model))))
        else "physics"
    )
    predict_torch_for_pass1 = (predict_torch_z if grad_mode_for_pass1 == "auto" else None)
    predict_fn_for_pass1 = (predict_fn_z if grad_mode_for_pass1 == "auto" else None)
    
    # Compute thresholds
    TAU_HI = float(th.get("tau_hi", 0.98))
    TAU_LO = float(th.get("tau_lo", 0.98))
    
    method = get_method(th.get("method", "knn"))
    outs1 = method.compute(
        train_X=train_X, train_zp=train_zp, train_zn=train_zn,
        query_X=query_X, D_all=D_all,
        idx_train_mask=S.index.isin(idx_train),
        idx_val_mask=S.index.isin(idx_val),
        taus=(TAU_LO, TAU_HI),
        cfg={
            **th,
            "metric": METRIC,
            "lambda_t": LAMBDA_T,
            "grad_mode": grad_mode_for_pass1,
            "grad_eps": GRAD_EPS,
            "physics_relative": PHYS_REL,
            "minmax": {"a": a, "b": b},
            "predict_fn": predict_fn_for_pass1,
            "predict_torch": predict_torch_for_pass1,
            "residuals": res_p1,
            "wind_std_for_model": Sv_s_m,
            "rho_std_for_model": (Sr_s_m if rho_for_model else None),
            "wind_std_for_clean": Sv_s_c,
            "rho_std_for_clean": (Sr_s_c if rho_for_clean else None),
            "y": S["power"].to_numpy(float),
            "y_hat": pred_p1
        },
        device=device
    )
    
    return {
        "model": mdl_p1,
        "predictions": pred_p1,
        "residuals": res_p1,
        "D_all": D_all,
        "zpos": zpos,
        "zneg": zneg,
        "pr_used": pr_used,
        "thresholds": outs1,
        "thr_pos": outs1.thr_pos,
        "thr_neg": outs1.thr_neg,
        "is_abnormal": outs1.is_abnormal
    }


def execute_pass2(S, idx_train, idx_val, is_p1, min_train_samples,
                  Xv_va_m, Xr_va_m, y_va, Sv_s_m, Sr_s_m, Sv_s_c, Sr_s_c,
                  rho_for_model, rho_for_clean, pass1_results,
                  mlp_cfg, th, device, gpu_cache_bytes, scaler, predict_fn_builder):
    """
    Execute Pass 2: Retrain on clean data and refine thresholds.
    
    Parameters
    ----------
    S : pd.DataFrame
        Scope data
    idx_train, idx_val : array-like
        Training and validation indices
    is_p1 : array-like
        Pass 1 anomaly flags
    min_train_samples : int
        Minimum training samples required
    Xv_va_m, Xr_va_m, y_va : array-like
        Validation data (model space)
    Sv_s_m, Sr_s_m : array-like
        Full data (model space)
    Sv_s_c, Sr_s_c : array-like
        Full data (clean space)
    rho_for_model : bool
        Whether model uses density
    rho_for_clean : bool
        Whether cleaning uses density
    pass1_results : dict
        Results from Pass 1
    mlp_cfg : dict
        MLP configuration
    th : dict
        Threshold configuration
    device : str
        Device for computation
    gpu_cache_bytes : int
        GPU cache limit
    scaler : Scaler
        Feature scaler
    predict_fn_builder : callable
        Function to build prediction functions
        
    Returns
    -------
    dict
        Dictionary containing Pass 2 results
    """
    from .data_prep import align_mask_to_index
    
    pr_used = pass1_results["pr_used"]
    mdl_p1 = pass1_results["model"]
    
    # Filter training data
    is_p1_train = align_mask_to_index(is_p1, S.index, idx_train, fill=False)
    keep_idx_tr = idx_train[~is_p1_train]
    
    # Retrain if enough samples
    if len(keep_idx_tr) >= min_train_samples:
        Xv_tr2_raw = S.loc[keep_idx_tr, "wind"].to_numpy(float)
        if rho_for_model:
            rho_series = S["rho"]
            Xr_tr2_raw_model = rho_series.loc[keep_idx_tr].to_numpy(float)
        else:
            Xr_tr2_raw_model = None
        Xv_tr2_m, Xr_tr2_m = scaler.transform(Xv_tr2_raw, Xr_tr2_raw_model, rho_for_model)
        y_tr2 = S.loc[keep_idx_tr, "power"].to_numpy(float)
        mdl_center, train_mode2 = fit_mlp_center(
            Xv_tr2_m, (Xr_tr2_m if rho_for_model else None), y_tr2,
            Xv_va_m, (Xr_va_m if rho_for_model else None), y_va,
            use_rho=rho_for_model, mlp_cfg=mlp_cfg, device=device, verbose=0,
            gpu_cache_limit_bytes=gpu_cache_bytes,
            thresholds_cfg=th, prated_used=pr_used
        )
    else:
        mdl_center = mdl_p1
    
    # Predict and compute residuals
    pred_c = predict_mlp_center(
        mdl_center, Sv_s_m, (Sr_s_m if rho_for_model else None),
        pr_used, rho_for_model
    )
    res_c = S["power"].to_numpy(float) - pred_c
    
    D_MODE = th.get("D_mode", "pred_or_both")
    EPS_RATIO = float(th.get("eps_ratio", 0.05))
    DELTA_POWER = float(th.get("delta_power", 50.0))
    D_all2 = build_D_np(pred_c, pr_used, D_MODE, EPS_RATIO, DELTA_POWER)
    
    zpos2 = np.clip(res_c, 0.0, None) / D_all2
    zneg2 = np.clip(-res_c, 0.0, None) / D_all2
    
    # Prepare Pass 2 KNN inputs
    keep_train_mask = S.index.isin(keep_idx_tr)
    v_keep = S.loc[keep_train_mask, "wind"].to_numpy(float)
    r_keep = S.loc[keep_train_mask, "rho"].to_numpy(float) if rho_for_clean else None
    v_keep_s, r_keep_s = scaler.transform(v_keep, r_keep, rho_for_clean)
    train2_X = np.c_[v_keep_s, (r_keep_s if rho_for_clean else np.zeros_like(v_keep_s))]
    train2_zp = zpos2[keep_train_mask]
    train2_zn = zneg2[keep_train_mask]
    query_X = np.c_[Sv_s_c, (Sr_s_c if rho_for_clean else np.zeros_like(Sv_s_c))]
    
    # Build prediction functions for Pass 2
    predict_fn_z2, predict_torch_z2 = predict_fn_builder(
        mdl_center, scaler, rho_for_model, rho_for_clean, pr_used
    )
    
    # Build minmax
    if scaler.method == "minmax":
        a = np.array([scaler.V_min, scaler.R_min], dtype=float)
        b = np.array([scaler.V_max - scaler.V_min,
                     scaler.R_max - scaler.R_min], dtype=float)
    else:
        a = np.array([0.0, 0.0], dtype=float)
        b = np.array([1.0, 1.0], dtype=float)
    
    # Grad mode
    METRIC = th.get("metric", "tanorm")
    LAMBDA_T = float(th.get("lambda_t", 6.0))
    GRAD_MODE = th.get("grad_mode", "auto")
    GRAD_EPS = th.get("grad_eps", 0.1)
    PHYS_REL = bool(th.get("physics_relative", True))
    
    grad_mode_for_pass2 = (
        GRAD_MODE if (((rho_for_clean and rho_for_model) or
                      ((not rho_for_clean) and (not rho_for_model))))
        else "physics"
    )
    predict_torch_for_pass2 = (predict_torch_z2 if grad_mode_for_pass2 == "auto" else None)
    predict_fn_for_pass2 = (predict_fn_z2 if grad_mode_for_pass2 == "auto" else None)
    
    # Compute Pass 2 thresholds
    TAU_HI = float(th.get("tau_hi", 0.98))
    TAU_LO = float(th.get("tau_lo", 0.98))
    
    method = get_method(th.get("method", "knn"))
    outs2 = method.compute(
        train_X=train2_X, train_zp=train2_zp, train_zn=train2_zn,
        query_X=query_X, D_all=D_all2,
        idx_train_mask=keep_train_mask,
        idx_val_mask=(S.index.isin(idx_val) & (~is_p1)),
        taus=(TAU_LO, TAU_HI),
        cfg={
            **th,
            "metric": METRIC,
            "lambda_t": LAMBDA_T,
            "grad_mode": grad_mode_for_pass2,
            "grad_eps": GRAD_EPS,
            "physics_relative": PHYS_REL,
            "minmax": {"a": a, "b": b},
            "predict_fn": predict_fn_for_pass2,
            "predict_torch": predict_torch_for_pass2,
            "residuals": res_c,
            "wind_std_for_model": Sv_s_m,
            "rho_std_for_model": (Sr_s_m if rho_for_model else None),
            "wind_std_for_clean": Sv_s_c,
            "rho_std_for_clean": (Sr_s_c if rho_for_clean else None),
            "y": S["power"].to_numpy(float),
            "y_hat": pred_c
        },
        device=device
    )
    
    # Combine Pass 1 and Pass 2 anomalies
    is_p2 = np.zeros_like(is_p1, dtype=bool)
    rem_mask = (~is_p1)
    is_p2[rem_mask] = outs2.is_abnormal[rem_mask]
    
    return {
        "model": mdl_center,
        "predictions": pred_c,
        "residuals": res_c,
        "D_all": D_all2,
        "thresholds": outs2,
        "thr_pos": outs2.thr_pos,
        "thr_neg": outs2.thr_neg,
        "is_abnormal": is_p2
    }
