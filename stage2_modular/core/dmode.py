# -*- coding: utf-8 -*-
import math
import numpy as np
import torch

def build_D_np(y_hat: np.ndarray, pr_used: float, d_mode: str, eps_ratio: float, delta_power: float):
    if d_mode == "pred_only":
        D = np.fmax(y_hat, 1.0)
    elif d_mode == "pred_or_epsPr":
        D = np.fmax(y_hat, eps_ratio*pr_used)
    elif d_mode == "pred_or_delta":
        D = np.fmax(y_hat, delta_power)
    else:
        D = np.fmax(np.fmax(y_hat, eps_ratio*pr_used), delta_power)
    return D

@torch.no_grad()
def build_D_from_yhat(y_hat: torch.Tensor, prated_used: float,
                      d_mode: str, eps_ratio: float, delta_power: float) -> torch.Tensor:
    D = torch.clamp(y_hat, min=1.0)
    if d_mode in ("pred_or_epsPr", "pred_or_both"):
        if math.isfinite(prated_used):
            D = torch.maximum(D, torch.full_like(D, float(eps_ratio * prated_used)))
    if d_mode in ("pred_or_delta", "pred_or_both"):
        D = torch.maximum(D, torch.full_like(D, float(delta_power)))
    return D
