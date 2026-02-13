# stage2_modular/thresholds/knn_local.py
# -*- coding: utf-8 -*-
"""
KNN-based threshold method for anomaly detection.
Uses GPU-accelerated distance computation with various metrics.
"""
import math
import numpy as np
import torch

from .base import ThresholdMethod, ThresholdOutputs
from .conformal_utils import weighted_quantile, conformal_scale
from .gradient_utils import (
    physics_grad_x_batch,
    finite_diff_grad_z_batch,
    autograd_grad_z_batch,
    physics_dir_in_z_batch
)
from .distance_utils import distances_chunk


class KNNLocal(ThresholdMethod):
    """
    KNN-based local threshold method with GPU acceleration.
    
    Supports multiple distance metrics:
    - physics: Physics-based distance in power curve space
    - grad_dir/grad: Gradient-based normal distance
    - tanorm: Tangent-normal decomposed distance
    """
    name = "knn"

    def compute(self, *, train_X, train_zp, train_zn, query_X, D_all,
                idx_train_mask, idx_val_mask, taus, cfg, device=None):
        """
        Compute KNN-based thresholds for anomaly detection.
        
        Parameters
        ----------
        train_X : numpy.ndarray
            Training features in normalized space (N, d)
        train_zp, train_zn : numpy.ndarray
            Training z-scores for positive/negative residuals (N,)
        query_X : numpy.ndarray
            Query features in normalized space (Q, d)
        D_all : numpy.ndarray
            Normalization scales for each query point (Q,)
        idx_train_mask, idx_val_mask : array-like
            Boolean masks for training and validation sets
        taus : tuple
            Quantile levels (tau_lo, tau_hi) for threshold computation
        cfg : dict
            Configuration dictionary with method parameters
        device : str, optional
            Device for computation ("cuda" or "cpu")
            
        Returns
        -------
        ThresholdOutputs
            Object containing thr_pos, thr_neg, and is_abnormal arrays
        """
        # ========= 配置 =========
        TAU_HI = float(taus[0] if isinstance(taus, (list, tuple)) else cfg.get("tau_hi", 0.98))
        TAU_LO = float(taus[1] if isinstance(taus, (list, tuple)) else cfg.get("tau_lo", 0.98))

        metric = (cfg.get("metric") or "tanorm").lower()
        lambda_t     = float(cfg.get("lambda_t", 6.0))
        grad_mode    = (cfg.get("grad_mode") or "auto").lower()
        grad_eps     = cfg.get("grad_eps", 0.1)
        predict_fn   = cfg.get("predict_fn", None)
        predict_tch  = cfg.get("predict_torch", None)
        minmax       = cfg.get("minmax", None)

        _val = cfg.get("physics_relative", True)
        if isinstance(_val, np.ndarray):
            if _val.size == 1:
                physics_relative = bool(_val.reshape(()).item())
            else:
                raise RuntimeError(f"physics_relative expects scalar/bool, got array shape={_val.shape}")
        else:
            physics_relative = bool(_val)

        K_NEI       = int(cfg.get("k_nei", 500))
        BATCH_Q     = int(cfg.get("knn_batch_q", 2048))
        TRAIN_CHUNK = int(cfg.get("knn_train_chunk", 65536))

        # ========= 数据准备 =========
        train_X = np.asarray(train_X, float)
        query_X = np.asarray(query_X, float)
        train_zp = np.asarray(train_zp, float)
        train_zn = np.asarray(train_zn, float)
        D_all = np.asarray(D_all, float)

        # ====== 稳健校验 ======
        def _fail(msg, **kv):
            lines = [f"[KNNLocal] {msg}"]
            for k, v in kv.items():
                lines.append(f"  - {k}: {v}")
            full = "\n".join(lines)
            print(full, flush=True)
            raise RuntimeError(full)

        if not (train_X.ndim == 2 and query_X.ndim == 2):
            _fail("train_X/query_X ndim wrong",
                  train_X_shape=getattr(train_X, "shape", None),
                  query_X_shape=getattr(query_X, "shape", None))
        d = int(train_X.shape[1])
        if d not in (1, 2):
            _fail("feature dim must be 1 or 2", d=d)
        if int(query_X.shape[1]) != d:
            _fail("query/train dim mismatch", query_dim=int(query_X.shape[1]), train_dim=d)

        d_model = 2 if (cfg.get("rho_std_for_model", None) is not None) else 1
        if metric in ("grad_dir", "grad", "tanorm", "tn") and (d != d_model):
            print(f"[KNNLocal] d_clean({d}) != d_model({d_model}) -> force grad_mode='physics'", flush=True)
            grad_mode = "physics"
            predict_fn = None
            predict_tch = None

        if metric == "physics":
            mm = minmax
            if not (isinstance(mm, dict) and (("a" in mm) or ("A" in mm)) and (("b" in mm) or ("B" in mm))):
                _fail("minmax is missing for physics metric",
                      minmax_type=type(mm), minmax_keys=list(mm.keys()) if isinstance(mm, dict) else None)
            a = np.asarray(mm.get("a") or mm.get("A"), float).ravel()
            b = np.asarray(mm.get("b") or mm.get("B"), float).ravel()
            if a.size == 1:
                a = np.full((d,), float(a[0]), dtype=float)
            if b.size == 1:
                b = np.full((d,), float(b[0]), dtype=float)
            if a.size != d or b.size != d:
                _fail("minmax a/b must be length d", a_size=a.size, b_size=b.size, d=d)

        if metric in ("grad_dir", "grad", "tanorm", "tn"):
            ge_arr = np.asarray(grad_eps, float)
            if ge_arr.size < 1:
                _fail("grad_eps must be scalar or array with >=1 elements", grad_eps=grad_eps)

        # ========= 设备 =========
        req_cuda = str(device).lower().startswith("cuda")
        cuda_ok = torch.cuda.is_available()
        use_gpu = (req_cuda and cuda_ok)
        torch_device = torch.device("cuda") if use_gpu else torch.device("cpu")
        if req_cuda and not cuda_ok:
            _fail("device='cuda' but torch.cuda.is_available()==False")

        if use_gpu:
            try:
                dev_name = torch.cuda.get_device_name(torch_device)
            except Exception:
                dev_name = "cuda"
            print(f"[KNNLocal] Using GPU: {dev_name} | candidates={len(train_X)}, queries={len(query_X)}", flush=True)
        else:
            print(f"[KNNLocal] Using CPU | candidates={len(train_X)}, queries={len(query_X)}", flush=True)

        # ========= 候选常驻设备 =========
        Xcand_z_t = torch.as_tensor(train_X, dtype=torch.float32, device=torch_device)
        N = int(Xcand_z_t.shape[0])
        Q = int(query_X.shape[0])

        # ========= physics 常量 =========
        a_t = b_t = None
        if metric == "physics":
            a_np = np.asarray(minmax.get("a") or minmax.get("A"), float).ravel()
            b_np = np.asarray(minmax.get("b") or minmax.get("B"), float).ravel()
            if a_np.size == 1:
                a_np = np.full((d,), float(a_np[0]), dtype=float)
            if b_np.size == 1:
                b_np = np.full((d,), float(b_np[0]), dtype=float)
            a_t = torch.as_tensor(a_np.reshape(1, -1), dtype=torch.float32, device=torch_device)
            b_t = torch.as_tensor(b_np.reshape(1, -1), dtype=torch.float32, device=torch_device)

        # ========= 输出容器 =========
        q_hi = np.full((Q,), np.nan, float)
        q_lo = np.full((Q,), np.nan, float)

        # ========= 批处理循环 =========
        for s in range(0, Q, BATCH_Q):
            e = min(Q, s + BATCH_Q)
            B = e - s

            Zb_np = query_X[s:e, :]
            Zb_t  = torch.as_tensor(Zb_np, dtype=torch.float32, device=torch_device)

            if metric == "physics":
                Xi_x = a_t + b_t * Zb_t
                Gx_b_t = physics_grad_x_batch(Xi_x)
            else:
                if grad_mode == "physics":
                    G_np = physics_dir_in_z_batch(Zb_np, minmax)
                    U_np = G_np / (np.linalg.norm(G_np, axis=1, keepdims=True) + 1e-12)
                    U_b_t = torch.as_tensor(U_np, dtype=torch.float32, device=torch_device)
                else:
                    if predict_tch is not None:
                        G_t = autograd_grad_z_batch(predict_tch, Zb_t)
                        if not isinstance(G_t, torch.Tensor):
                            G_t = torch.as_tensor(G_t, dtype=torch.float32, device=torch_device)
                        U_b_t = G_t / (torch.linalg.norm(G_t, dim=1, keepdim=True) + 1e-12)
                    elif predict_fn is not None:
                        G_np = finite_diff_grad_z_batch(predict_fn, Zb_np, grad_eps)
                        U_np = G_np / (np.linalg.norm(G_np, axis=1, keepdims=True) + 1e-12)
                        U_b_t = torch.as_tensor(U_np, dtype=torch.float32, device=torch_device)
                    else:
                        if minmax is not None:
                            G_np = physics_dir_in_z_batch(Zb_np, minmax)
                            U_np = G_np / (np.linalg.norm(G_np, axis=1, keepdims=True) + 1e-12)
                            U_b_t = torch.as_tensor(U_np, dtype=torch.float32, device=torch_device)
                        else:
                            U_b_t = torch.zeros((B, d), dtype=torch.float32, device=torch_device)
                            U_b_t[:, 0] = 1.0

            best_d = torch.full((B, K_NEI), float("inf"), dtype=torch.float32, device=torch_device)
            best_i = torch.full((B, K_NEI), -1, dtype=torch.int64, device=torch_device)

            for c0 in range(0, N, TRAIN_CHUNK):
                c1 = min(N, c0 + TRAIN_CHUNK)
                Xc_t = Xcand_z_t[c0:c1, :]
                C = c1 - c0

                if metric == "physics":
                    Xx_c_t = a_t + b_t * Xc_t
                    D_chunk = distances_chunk("physics", Zb_t, Xc_t,
                                              lambda_t=lambda_t,
                                              a_t=a_t, b_t=b_t, Gx_b_t=Gx_b_t, Xx_c_t=Xx_c_t,
                                              physics_relative=physics_relative)
                else:
                    D_chunk = distances_chunk(metric, Zb_t, Xc_t,
                                              lambda_t=lambda_t,
                                              U_b_t=U_b_t,
                                              physics_relative=physics_relative)

                idx_block = torch.arange(c0, c1, device=torch_device).view(1, C).expand(B, C)
                cand_d = torch.cat([best_d, D_chunk], dim=1)
                cand_i = torch.cat([best_i, idx_block], dim=1)
                dvals, idx = torch.topk(cand_d, k=K_NEI, dim=1, largest=False, sorted=False)
                rows = torch.arange(B, device=torch_device).view(B, 1)
                best_d = dvals
                best_i = cand_i[rows, idx]

                del D_chunk, idx_block, cand_d, cand_i, dvals, idx

            best_d_np = best_d.detach().cpu().numpy()
            best_i_np = best_i.detach().cpu().numpy()

            sigma = np.median(best_d_np, axis=1).reshape(-1, 1)
            sigma = np.maximum(sigma, 1e-9)

            for bi in range(B):
                idx_row = best_i_np[bi]
                d_row = best_d_np[bi]
                w = np.exp(-0.5 * (d_row / sigma[bi, 0]) ** 2)
                zpK = train_zp[idx_row]
                mp = (zpK > 0)
                znK = train_zn[idx_row]
                mn = (znK > 0)
                qhi_i = weighted_quantile(zpK[mp], w[mp], TAU_HI) if np.any(mp) else 0.0
                qlo_i = weighted_quantile(znK[mn], w[mn], TAU_LO) if np.any(mn) else 0.0
                q_hi[s + bi] = max(qhi_i, 0.0)
                q_lo[s + bi] = max(qlo_i, 0.0)

            del Zb_t
            if metric == "physics":
                del Xi_x, Gx_b_t
            else:
                del U_b_t
            torch.cuda.empty_cache()

        # ========= 验证集 conformal 标定 =========
        val_mask = np.asarray(idx_val_mask, bool)
        if val_mask.size != q_hi.size:
            c_plus, c_minus = 1.0, 1.0
        else:
            if "val_z_pos" in cfg and "val_z_neg" in cfg:
                val_z_pos = np.asarray(cfg["val_z_pos"], float)[val_mask]
                val_z_neg = np.asarray(cfg["val_z_neg"], float)[val_mask]
            else:
                resid_full = np.asarray(cfg.get("residuals", np.zeros_like(D_all)), float)
                zpos_full = np.clip(resid_full, 0.0, None) / np.maximum(D_all, 1e-12)
                zneg_full = np.clip(-resid_full, 0.0, None) / np.maximum(D_all, 1e-12)
                val_z_pos = zpos_full[val_mask]
                val_z_neg = zneg_full[val_mask]
            c_plus, c_minus = conformal_scale(val_z_pos, q_hi[val_mask], val_z_neg, q_lo[val_mask], TAU_HI, TAU_LO)

        thr_pos = np.asarray(c_plus * q_hi * D_all, float)
        thr_neg = np.asarray(c_minus * q_lo * D_all, float)

        resid_full = np.asarray(cfg.get("residuals", np.zeros_like(thr_pos)), float)
        is_abnormal = (resid_full > thr_pos) | (resid_full < -thr_neg)

        return ThresholdOutputs(
            thr_pos=thr_pos,
            thr_neg=thr_neg,
            is_abnormal=is_abnormal
        )
