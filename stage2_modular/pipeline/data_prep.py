# stage2_modular/pipeline/data_prep.py
# -*- coding: utf-8 -*-
"""
Data preparation utilities for the Stage 2 pipeline.
Handles data loading, filtering, and feature extraction.
"""
import numpy as np
import pandas as pd
import torch


def ensure_bool_flags(df_like, cols):
    """
    Ensure boolean columns exist in DataFrame with proper dtype.
    
    Parameters
    ----------
    df_like : pandas.DataFrame
        DataFrame to update
    cols : list
        List of column names to ensure as boolean
    """
    for c in cols:
        if c not in df_like.columns:
            df_like[c] = False
        else:
            df_like[c] = df_like[c].astype(bool, copy=False)


def align_mask_to_index(mask_full, full_index, target_index, fill=False):
    """
    Align a boolean mask from full index to a subset index.
    
    Parameters
    ----------
    mask_full : array-like or pd.Series
        Boolean mask for full_index
    full_index : pd.Index
        Full index the mask is defined on
    target_index : array-like or pd.Index
        Target subset index to align to
    fill : bool, default False
        Fill value for missing indices
        
    Returns
    -------
    numpy.ndarray
        Boolean array aligned to target_index
    """
    import pandas as _pd
    import numpy as np
    if isinstance(target_index, _pd.Index):
        tgt_idx = target_index
    else:
        tgt_idx = _pd.Index(target_index)
    if isinstance(full_index, _pd.Index):
        full_idx = full_index
    else:
        full_idx = _pd.Index(full_index)

    if isinstance(mask_full, (np.ndarray, list, tuple)):
        s = _pd.Series(mask_full, index=full_idx, dtype="boolean")
    elif isinstance(mask_full, _pd.Series):
        s = mask_full.astype("boolean")
        if not s.index.equals(full_idx):
            s = s.reindex(full_idx)
    else:
        raise TypeError(f"Unsupported mask type: {type(mask_full)}")

    s_sub = s.reindex(tgt_idx, fill_value=bool(fill)).astype("boolean")
    return s_sub.to_numpy(dtype=bool)


def split_indices_by_ratio(df_sorted_index, split=(0.8, 0.2, 0.0)):
    """
    Split indices by ratio.
    
    Parameters
    ----------
    df_sorted_index : array-like
        Sorted indices to split
    split : tuple, default (0.8, 0.2, 0.0)
        Train/validation/test split ratios
        
    Returns
    -------
    tuple
        (idx_train, idx_val, idx_test) arrays
    """
    tr, va, te = split
    n = len(df_sorted_index)
    i_tr = int(round(tr * n))
    i_va = int(round((tr + va) * n))
    idx_train = df_sorted_index[:i_tr]
    idx_val = df_sorted_index[i_tr:i_va]
    idx_test = df_sorted_index[i_va:] if te > 1e-12 else np.array([], dtype=df_sorted_index.dtype)
    return idx_train, idx_val, idx_test


def make_rho_model_array(base_series, ix, split_tag, rho_input_mode, rho_const_value,
                         rho_shuffle_cfg, station, tid, idx_train):
    """
    Create density array for model input with different modes.
    
    Parameters
    ----------
    base_series : pd.Series
        Base density series
    ix : array-like
        Indices to extract
    split_tag : str
        Tag for random seed (e.g., "train", "val", "all")
    rho_input_mode : str
        Mode: "normal", "constant", or "shuffle"
    rho_const_value : float or str
        Constant value or "train_mean"
    rho_shuffle_cfg : dict
        Shuffle configuration with seed
    station : str
        Station name for seed
    tid : int
        Turbine ID for seed
    idx_train : array-like
        Training indices (for computing mean)
        
    Returns
    -------
    numpy.ndarray
        Density array for model input
    """
    rng = np.random.default_rng(
        rho_shuffle_cfg.get("seed", 42) + hash((station, tid, split_tag)) % 100000
    )
    vals = base_series.loc[ix].to_numpy(float)
    
    if rho_input_mode == "normal":
        return vals
    
    if rho_input_mode == "constant":
        if isinstance(rho_const_value, (int, float)):
            const = float(rho_const_value)
        elif str(rho_const_value) == "train_mean":
            const = (float(np.nanmean(base_series.loc[idx_train].to_numpy(float)))
                    if len(idx_train) > 0 else 1.225)
        else:
            const = 1.225
        return np.full_like(vals, const, dtype=float)
    
    if rho_input_mode == "shuffle":
        perm = vals.copy()
        rng.shuffle(perm)
        return perm
    
    return vals


def build_predict_functions_for_pass(mdl, scaler, rho_for_model, rho_for_clean, pr_used):
    """
    Build prediction functions for KNN threshold computation.
    
    Parameters
    ----------
    mdl : torch.nn.Module
        Center model
    scaler : Scaler
        Feature scaler
    rho_for_model : bool
        Whether model uses density
    rho_for_clean : bool
        Whether cleaning uses density  
    pr_used : float
        Rated power for clipping
        
    Returns
    -------
    tuple
        (predict_fn_z, predict_torch_z) - numpy and torch prediction functions
    """
    import math
    
    def predict_fn_z(Z_batch):
        """NumPy prediction function in normalized space."""
        Z_batch = np.asarray(Z_batch, float)
        d_clean = Z_batch.shape[1]
        d_model = 2 if rho_for_model else 1
        if d_clean != d_model:
            raise RuntimeError(
                "predict_fn_z called with mismatched dim; rho_for_clean != rho_for_model"
            )
        zv = Z_batch[:, 0].reshape(-1)
        zr = (Z_batch[:, 1].reshape(-1) if rho_for_model else None)
        from ..models.center import predict_mlp_center
        y = predict_mlp_center(mdl, zv, zr, pr_used, use_rho=rho_for_model, clip_to_prated=True)
        return y.reshape(-1)

    def predict_torch_z(Zb_t: torch.Tensor) -> torch.Tensor:
        """PyTorch prediction function in normalized space (autograd-compatible)."""
        model = mdl
        dev = next(model.parameters()).device
        X = Zb_t[:, : (2 if rho_for_model else 1)].to(dev).float()
        model.eval()
        with torch.enable_grad():
            y = model(X).view(-1)
        if math.isfinite(pr_used):
            y = torch.clamp(y, 0.0, float(pr_used))
        return y

    return predict_fn_z, predict_torch_z
