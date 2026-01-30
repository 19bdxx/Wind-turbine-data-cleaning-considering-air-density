# -*- coding: utf-8 -*-
import numpy as np
import torch  # 新增：用于规范化 device
from .base import ThresholdMethod, ThresholdOutputs
from ..models.quantile import fit_quantile_mlp, predict_quantiles

class QuantilePower(ThresholdMethod):
    name = "quantile_power"

    def compute(self, *, train_X, train_zp, train_zn, query_X, D_all,
                idx_train_mask, idx_val_mask, taus, cfg, device=None):

        # ---- 统一规范化 device ----
        if device is None:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device

        tau_lo, tau_hi = float(taus[0]), float(taus[1])
        qcfg = (cfg.get("quantile") or {})
        use_rho = bool(qcfg.get("use_rho", True))
        qmlp_cfg = qcfg.get("mlp", {})

        # 选择用于拟合分位曲面的特征 X_all（风速/或风速+rho，均为已标准化）
        if use_rho:
            if ("wind_std_for_model" in cfg) and ("rho_std_for_model" in cfg) and (cfg["rho_std_for_model"] is not None):
                X_all = np.c_[cfg["wind_std_for_model"], cfg["rho_std_for_model"]]
            elif ("wind_std_for_clean" in cfg) and ("rho_std_for_clean" in cfg):
                X_all = np.c_[cfg["wind_std_for_clean"], cfg["rho_std_for_clean"]]
            else:
                X_all = query_X
        else:
            if "wind_std_for_model" in cfg and cfg["wind_std_for_model"] is not None:
                X_all = cfg["wind_std_for_model"].reshape(-1, 1)
            else:
                X_all = cfg["wind_std_for_clean"].reshape(-1, 1)

        y_all = cfg["y"]
        y_hat_all = cfg["y_hat"]
        tr = idx_train_mask
        va = idx_val_mask

        # 拟合上下分位（功率分位曲面）
        taus_list = [tau_lo, tau_hi]
        model, _ = fit_quantile_mlp(
            X_all[tr], y_all[tr],
            X_all[va], y_all[va],
            taus=taus_list, cfg=qmlp_cfg, device=dev, verbose=1
        )
        q_pred_all = predict_quantiles(model, X_all)
        q_lo_all = q_pred_all[:, 0]
        q_hi_all = q_pred_all[:, 1]

        # 简单共形偏移：保证在 val 上达到目标覆盖
        y_val = y_all[va]
        q_lo_val = q_lo_all[va]
        q_hi_val = q_hi_all[va]
        exceed_hi = np.maximum(0.0, y_val - q_hi_val)
        exceed_lo = np.maximum(0.0, q_lo_val - y_val)
        c_plus = float(np.quantile(exceed_hi, tau_hi)) if exceed_hi.size > 0 else 0.0
        c_minus = float(np.quantile(exceed_lo, tau_lo)) if exceed_lo.size > 0 else 0.0

        ub = q_hi_all + c_plus
        lb = q_lo_all - c_minus

        # 转化为“围绕 y_hat 的阈值”
        thr_pos = np.maximum(0.0, ub - y_hat_all)
        thr_neg = np.maximum(0.0, y_hat_all - lb)

        res = cfg["residuals"]
        is_abn = (res > thr_pos) | (res < -thr_neg)
        return ThresholdOutputs(thr_pos=thr_pos, thr_neg=thr_neg, is_abnormal=is_abn)
