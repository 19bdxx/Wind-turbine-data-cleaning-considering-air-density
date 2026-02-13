# stage2_modular/thresholds/gradient_utils.py
# -*- coding: utf-8 -*-
"""
Gradient computation utilities for KNN threshold methods.
Supports physics-based gradients, finite differences, and PyTorch autograd.
"""
import numpy as np
import torch


def physics_grad_x_batch(x_t):
    """
    Compute physics-based gradient in physical space.
    
    For wind turbine power curve: P ≈ k * ρ * V^3
    Gradient: ∂P/∂V = 3ρV^2, ∂P/∂ρ = V^3
    
    Parameters
    ----------
    x_t : torch.Tensor
        Physical space coordinates (B, d) where d=1 (V only) or d=2 (V, ρ)
        
    Returns
    -------
    torch.Tensor
        Gradient tensor (B, d)
    """
    B, d = x_t.shape
    V = x_t[:, 0]
    if d == 1:
        gx0 = 3.0 * (V * V)
        return torch.stack([gx0], dim=1)
    rho = x_t[:, 1]
    gx0 = 3.0 * rho * (V * V)
    gx1 = V ** 3
    return torch.stack([gx0, gx1], dim=1)


def finite_diff_grad_z_batch(center_pred_fn, Z_b, eps):
    """
    Compute gradient using finite differences in normalized space.
    
    Parameters
    ----------
    center_pred_fn : callable
        Function accepting (N, d) numpy array and returning (N,) numpy array
    Z_b : numpy.ndarray
        Batch of normalized coordinates (B, d)
    eps : float or array
        Step size for finite differences (scalar or per-dimension)
        
    Returns
    -------
    numpy.ndarray
        Gradient array (B, d)
    """
    Z_b = np.asarray(Z_b, float)
    B, d = Z_b.shape
    ez = np.asarray(eps, float)
    if ez.size == 1:
        ez = np.full((d,), float(ez.ravel()[0]), dtype=float)

    f0 = np.asarray(center_pred_fn(Z_b), float).reshape(B)
    G = np.zeros((B, d), dtype=float)
    for k in range(d):
        Zk = Z_b.copy()
        Zk[:, k] += ez[k]
        fk = np.asarray(center_pred_fn(Zk), float).reshape(B)
        denom = ez[k] if ez[k] != 0.0 else 1.0
        G[:, k] = (fk - f0) / denom
    return G


def autograd_grad_z_batch(torch_predict, Zb_t):
    """
    Compute gradient using PyTorch autograd in normalized space.
    
    Parameters
    ----------
    torch_predict : callable
        Function accepting (B, d) torch.Tensor and returning (B,) torch.Tensor
    Zb_t : torch.Tensor
        Batch of normalized coordinates (B, d) on device
        
    Returns
    -------
    torch.Tensor
        Gradient tensor (B, d)
        
    Notes
    -----
    Uses per-sample backward pass for compatibility. Can be optimized with
    vmap/jacobian in newer PyTorch versions.
    """
    B, d = Zb_t.shape
    Zb_t = Zb_t.detach().clone().requires_grad_(True)
    grads = []
    # Single forward pass, then per-sample backward
    with torch.enable_grad():
        out = torch_predict(Zb_t)  # (B,)
        assert out.shape == (B,), f"torch_predict must return (B,), got {tuple(out.shape)}"
        for i in range(B):
            grad_i = torch.autograd.grad(out[i], Zb_t, retain_graph=True, 
                                         create_graph=False, allow_unused=False)[0][i]
            grads.append(grad_i.detach())
    G = torch.stack(grads, dim=0)  # (B, d)
    return G


def physics_dir_in_z_batch(Z_b, minmax):
    """
    Compute physics-based direction in normalized space.
    
    Transforms normalized coordinates to physical space, computes physics gradient,
    returns as direction (unnormalized).
    
    Parameters
    ----------
    Z_b : numpy.ndarray
        Batch of normalized coordinates (B, d)
    minmax : dict
        Normalization parameters with keys 'a' and 'b' for affine transform:
        X = a + b * Z
        
    Returns
    -------
    numpy.ndarray
        Direction vectors (B, d) based on physics gradient
    """
    Z_b = np.asarray(Z_b, float)
    B, d = Z_b.shape
    a = np.asarray(minmax.get("a") or minmax.get("A"), float).ravel()
    b = np.asarray(minmax.get("b") or minmax.get("B"), float).ravel()
    if a.size == 1:
        a = np.full((d,), float(a[0]), dtype=float)
    if b.size == 1:
        b = np.full((d,), float(b[0]), dtype=float)
    if a.size != d or b.size != d:
        raise RuntimeError(f"minmax a/b must be length {d}, got {a.size}/{b.size}")
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    X_b = a + b * Z_b  # (B, d)
    V = X_b[:, 0]
    if d == 1:
        gx0 = 3.0 * (V * V)
        return np.stack([gx0], axis=1)
    rho = X_b[:, 1]
    gx0 = 3.0 * rho * (V * V)
    gx1 = V ** 3
    return np.stack([gx0, gx1], axis=1)
