# stage2_modular/thresholds/conformal_utils.py
# -*- coding: utf-8 -*-
"""
Conformal prediction and weighted quantile utilities for threshold calibration.
"""
import numpy as np


def weighted_quantile(values, weights, q):
    """
    Compute weighted quantile of values.
    
    Parameters
    ----------
    values : array-like
        Values to compute quantile from
    weights : array-like
        Weights for each value
    q : float
        Quantile level (0 to 1)
        
    Returns
    -------
    float
        Weighted quantile value
    """
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if v.size == 0 or np.sum(w) <= 0:
        return np.nan
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    csum = np.cumsum(w)
    tgt = float(q) * float(csum[-1])
    j = np.searchsorted(csum, tgt, side="right")
    j = min(max(int(j), 0), len(v) - 1)
    return float(v[j])


def conformal_scale(val_z_pos, q_hi_val, val_z_neg, q_lo_val, tau_hi, tau_lo):
    """
    Compute conformal scaling factors for positive and negative thresholds.
    
    Parameters
    ----------
    val_z_pos : array-like
        Validation set positive z-scores
    q_hi_val : array-like
        Validation set high quantile predictions
    val_z_neg : array-like
        Validation set negative z-scores
    q_lo_val : array-like
        Validation set low quantile predictions
    tau_hi : float
        High quantile level for conformal calibration
    tau_lo : float
        Low quantile level for conformal calibration
        
    Returns
    -------
    tuple of (float, float)
        Scaling factors (c_plus, c_minus) for positive and negative thresholds
    """
    def scale_one(z, q, tau):
        z = np.asarray(z, float)
        q = np.asarray(q, float)
        m = (q > 0) & (z >= 0)
        if not np.any(m):
            return 1.0
        r = z[m] / q[m]
        return float(np.quantile(r, tau))
    
    c_plus = scale_one(val_z_pos, q_hi_val, tau_hi)
    c_minus = scale_one(val_z_neg, q_lo_val, tau_lo)
    c_plus = max(c_plus, 1e-6)
    c_minus = max(c_minus, 1e-6)
    return c_plus, c_minus
