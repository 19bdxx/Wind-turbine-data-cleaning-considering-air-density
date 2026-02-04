# stage2_modular/thresholds/distance_utils.py
# -*- coding: utf-8 -*-
"""
Distance computation utilities for KNN threshold methods.
Supports physics-based, gradient-based, and tangent-normal metrics on GPU/CPU.
"""
import torch


def distances_chunk(metric, Zb_t, Xc_t, *,
                    lambda_t,
                    # grad/tanorm:
                    U_b_t=None,
                    # physics:
                    a_t=None, b_t=None, Gx_b_t=None, Xx_c_t=None,
                    physics_relative=True):
    """
    Compute distance matrix between query batch and candidate chunk on GPU.
    
    Parameters
    ----------
    metric : str
        Distance metric: "physics", "grad_dir", "grad", or "tanorm"
    Zb_t : torch.Tensor
        Query batch in normalized space (B, d)
    Xc_t : torch.Tensor
        Candidate chunk in normalized space (C, d)
    lambda_t : float
        Tangent weighting factor for tanorm metric
    U_b_t : torch.Tensor, optional
        Normalized direction vectors for query batch (B, d)
        Required for grad_dir, grad, and tanorm metrics
    a_t, b_t : torch.Tensor, optional
        Normalization parameters for physics metric
    Gx_b_t : torch.Tensor, optional
        Physics gradient for query batch (B, d)
        Required for physics metric
    Xx_c_t : torch.Tensor, optional
        Candidate chunk in physical space (C, d)
        Optional for physics metric (computed if not provided)
    physics_relative : bool
        Whether to use relative physics distance (default: True)
        
    Returns
    -------
    torch.Tensor
        Distance matrix (B, C)
    """
    B, d = Zb_t.shape
    C = Xc_t.shape[0]

    if metric == "physics":
        Xi_x = a_t + b_t * Zb_t                 # (B, d)
        Xc_x = Xx_c_t if Xx_c_t is not None else (a_t + b_t * Xc_t)  # (C, d)
        # dP = | (Xc_x @ Gx^T) - ⟨xi_x, Gx⟩ |
        S = Xc_x @ Gx_b_t.T                     # (C, B)
        c = torch.sum(Xi_x * Gx_b_t, dim=1)     # (B,)
        dP = (S - c).abs().T                    # (B, C)
        if physics_relative:
            if d >= 2:
                V = Xi_x[:, 0]
                rho = Xi_x[:, 1]
                denom = torch.abs(rho * (V ** 3)) + 1e-6
            else:
                V = Xi_x[:, 0]
                denom = torch.abs(V ** 3) + 1e-6
            dP = dP / denom.view(B, 1)
        return dP

    # —— grad_dir / tanorm ——（在 z 空间）
    # dn = | (Xc @ U^T) - ⟨Zb, U⟩ |
    S = Xc_t @ U_b_t.T                 # (C, B)
    c = torch.sum(Zb_t * U_b_t, dim=1) # (B,)
    dn = (S - c).abs().T               # (B, C)

    if metric in ("grad_dir", "grad"):
        return dn

    # tanorm: dt^2 = ||dz||^2 - dn^2
    X2 = torch.sum(Xc_t * Xc_t, dim=1).view(1, C)  # (1, C)
    Z2 = torch.sum(Zb_t * Zb_t, dim=1).view(B, 1)  # (B, 1)
    XZ = (Xc_t @ Zb_t.T).T                          # (B, C)
    dz2 = Z2 + X2 - 2.0 * XZ                       # (B, C)
    dt2 = torch.clamp(dz2 - dn * dn, min=0.0)
    return torch.sqrt(dn * dn + float(lambda_t) * dt2)
