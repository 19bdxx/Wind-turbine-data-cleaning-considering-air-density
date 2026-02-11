# stage2_modular/thresholds/knn_local.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import torch

from .base import ThresholdMethod, ThresholdOutputs



# ===================== 公共：加权分位 / conformal 标定 =====================

def _weighted_quantile(values, weights, q):
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

def _conformal_scale(val_z_pos, q_hi_val, val_z_neg, q_lo_val, tau_hi, tau_lo):
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

# ===================== 工具函数 =====================

def _ensure_1d(a):
    return np.asarray(a, dtype=float).reshape(-1)

def _physics_grad_x_batch(x_t):
    """
    x_t: torch tensor (B,d) 物理空间 (V[, rho])
    返回 g_x: torch (B,d)
    P≈k ρ V^3 → g_x = (3ρV^2, V^3)；若 d=1：g_x=(3V^2,)
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

def _finite_diff_grad_z_batch(center_pred_fn, Z_b, eps):
    """
    批量有限差分：Z_b (B,d) numpy → G (B,d) numpy
    center_pred_fn: 接受 (N,d) numpy，返回 (N,) numpy
    eps: 标量或长度 d 的数组（在 z 空间）
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

def _autograd_grad_z_batch(torch_predict, Zb_t):
    """
    使用 autograd 计算每个样本的 ∇_z f(z)
    - torch_predict: callable, 接受 (B,d) torch.Tensor (requires_grad 可用)，返回 (B,) torch.Tensor
    - Zb_t: (B,d) torch.Tensor on device
    返回: (B,d) numpy（在 CPU）或 torch.Tensor（在调用处转）
    说明：
      为了兼容性，这里用“小批内逐样本 backward”的实现（每样本 1 前向 + 1 backward）。
      若 PyTorch 版本支持 vmap/jacobian 的高效向量化，可以后续替换为矢量雅可比。
    """
    B, d = Zb_t.shape
    Zb_t = Zb_t.detach().clone().requires_grad_(True)
    grads = []
    # 一次前向，拿到 (B,)；然后逐样本 backward，避免多次重复前向
    with torch.enable_grad():
        out = torch_predict(Zb_t)  # (B,)
        assert out.shape == (B,), f"torch_predict must return (B,), got {tuple(out.shape)}"
        for i in range(B):
            grad_i = torch.autograd.grad(out[i], Zb_t, retain_graph=True, create_graph=False, allow_unused=False)[0][i]
            grads.append(grad_i.detach())
    G = torch.stack(grads, dim=0)  # (B,d)
    return G

def _physics_dir_in_z_batch(Z_b, minmax):
    """
    用物理方向作为 z 空间的法向（只用方向）。
    Z_b: (B,d) numpy
    minmax: {'a':..., 'b':...}
    返回 G (B,d) numpy（只用于取方向）
    """
    Z_b = np.asarray(Z_b, float)
    B, d = Z_b.shape
    a = np.asarray(minmax.get("a") or minmax.get("A"), float).ravel()
    b = np.asarray(minmax.get("b") or minmax.get("B"), float).ravel()
    if a.size == 1: a = np.full((d,), float(a[0]), dtype=float)
    if b.size == 1: b = np.full((d,), float(b[0]), dtype=float)
    if a.size != d or b.size != d:
        raise RuntimeError(f"minmax a/b must be length {d}, got {a.size}/{b.size}")
    a = a.reshape(1, -1); b = b.reshape(1, -1)
    X_b = a + b * Z_b  # (B,d)
    V = X_b[:, 0]
    if d == 1:
        gx0 = 3.0 * (V * V)
        return np.stack([gx0], axis=1)
    rho = X_b[:, 1]
    gx0 = 3.0 * rho * (V * V)
    gx1 = V ** 3
    return np.stack([gx0, gx1], axis=1)

# ===================== 窗口筛选 =====================

def _window_filter_candidates(Zb_t, Xcand_t, window_v, window_r, d, min_candidates, K_NEI):
    """
    对每个查询点，根据风速和密度窗口筛选候选点
    
    参数:
        Zb_t: (B,d) 查询点（z空间）
        Xcand_t: (N,d) 候选点（z空间）
        window_v: 风速窗口半径
        window_r: 空气密度窗口半径（当d=2时使用）
        d: 特征维度 (1或2)
        min_candidates: 最小候选数阈值
        K_NEI: K近邻数
        
    返回:
        filtered_indices: list of (B,) torch.Tensor，每个查询点的候选索引
        expand_count: 扩展次数统计
    """
    B = Zb_t.shape[0]
    N = Xcand_t.shape[0]
    device = Zb_t.device
    
    filtered_indices = []
    expand_count = 0
    max_expansions = 3
    
    for bi in range(B):
        query_point = Zb_t[bi]  # (d,)
        current_window_v = window_v
        current_window_r = window_r
        expansion = 0
        
        while expansion <= max_expansions:
            # 风速维度筛选
            ws_mask = torch.abs(Xcand_t[:, 0] - query_point[0]) <= current_window_v
            
            # 如果是2维，加上密度维度筛选
            if d == 2:
                rho_mask = torch.abs(Xcand_t[:, 1] - query_point[1]) <= current_window_r
                mask = ws_mask & rho_mask
            else:
                mask = ws_mask
            
            candidate_indices = torch.where(mask)[0]
            n_candidates = len(candidate_indices)
            
            # 检查候选数是否足够
            if n_candidates >= max(K_NEI, min_candidates):
                filtered_indices.append(candidate_indices)
                if expansion > 0:
                    expand_count += 1
                break
            
            # 扩大窗口
            expansion += 1
            if expansion <= max_expansions:
                current_window_v *= 1.5
                current_window_r *= 1.5
        else:
            # 达到最大扩展次数仍不足，使用全部候选
            filtered_indices.append(torch.arange(N, device=device))
            expand_count += 1
    
    return filtered_indices, expand_count

# ===================== 批量 GPU 距离（B×C 分块） =====================

def _distances_chunk(metric, Zb_t, Xc_t, *,
                     lambda_t,
                     # grad/tanorm：
                     U_b_t=None,
                     # physics：
                     a_t=None, b_t=None, Gx_b_t=None, Xx_c_t=None,
                     physics_relative=True):
    """
    计算一个查询批 Zb_t(B,d) 对 一个候选分块 Xc_t(C,d) 的距离矩阵 (B,C) —— 全在 GPU 上。
    """
    B, d = Zb_t.shape
    C = Xc_t.shape[0]

    if metric == "physics":
        Xi_x = a_t + b_t * Zb_t                 # (B,d)
        Xc_x = Xx_c_t if Xx_c_t is not None else (a_t + b_t * Xc_t)  # (C,d)
        # dP = | (Xc_x @ Gx^T) - ⟨xi_x, Gx⟩ |
        S = Xc_x @ Gx_b_t.T                     # (C,B)
        c = torch.sum(Xi_x * Gx_b_t, dim=1)     # (B,)
        dP = (S - c).abs().T                    # (B,C)
        if physics_relative:
            if d >= 2:
                V = Xi_x[:, 0]; rho = Xi_x[:, 1]
                denom = torch.abs(rho * (V ** 3)) + 1e-6
            else:
                V = Xi_x[:, 0]
                denom = torch.abs(V ** 3) + 1e-6
            dP = dP / denom.view(B, 1)
        return dP

    # —— grad_dir / tanorm ——（在 z 空间）
    # dn = | (Xc @ U^T) - ⟨Zb, U⟩ |
    S = Xc_t @ U_b_t.T                 # (C,B)
    c = torch.sum(Zb_t * U_b_t, dim=1) # (B,)
    dn = (S - c).abs().T               # (B,C)

    if metric in ("grad_dir", "grad"):
        return dn

    # tanorm: dt^2 = ||dz||^2 - dn^2
    X2 = torch.sum(Xc_t * Xc_t, dim=1).view(1, C)  # (1,C)
    Z2 = torch.sum(Zb_t * Zb_t, dim=1).view(B, 1)  # (B,1)
    XZ = (Xc_t @ Zb_t.T).T                          # (B,C)
    dz2 = Z2 + X2 - 2.0 * XZ                       # (B,C)
    dt2 = torch.clamp(dz2 - dn * dn, min=0.0)
    return torch.sqrt(dn * dn + float(lambda_t) * dt2)

# ===================== KNN（GPU 批量 + 候选分块 + 行级 topK 合并） =====================

class KNNLocal(ThresholdMethod):
    name = "knn"

    def compute(self, *, train_X, train_zp, train_zn, query_X, D_all,
                idx_train_mask, idx_val_mask, taus, cfg, device=None):
        """
        - train_X / query_X：z 空间特征，形如 (N,d) / (Q,d)，d ∈ {1,2}
        - train_zp / train_zn：与 train_X 对齐的 z-score 正/负部
        - cfg 里可选传入：
            - metric ∈ {"physics","grad_dir","tanorm"}
            - lambda_t
            - grad_mode ∈ {"auto","physics"}
            - grad_eps
            - predict_fn: (N,d)->(N,) numpy 版本（有限差分回退用）
            - predict_torch: (B,d) torch.Tensor -> (B,) torch.Tensor，可微（autograd 优先）
            - minmax: {"a":..., "b":...}，供 physics 用
            - physics_relative: bool
            - knn_batch_q, knn_train_chunk, k_nei
        """
        # ========= 配置 =========
        TAU_HI = float(taus[0] if isinstance(taus, (list, tuple)) else cfg.get("tau_hi", 0.98))
        TAU_LO = float(taus[1] if isinstance(taus, (list, tuple)) else cfg.get("tau_lo", 0.98))

        metric = (cfg.get("metric") or "tanorm").lower()
        lambda_t     = float(cfg.get("lambda_t", 6.0))
        grad_mode    = (cfg.get("grad_mode") or "auto").lower()
        grad_eps     = cfg.get("grad_eps", 0.1)
        predict_fn   = cfg.get("predict_fn", None)            # numpy 版，有限差分回退用
        predict_tch  = cfg.get("predict_torch", None)         # torch 版，可微
        minmax       = cfg.get("minmax", None)

        _val = cfg.get("physics_relative", True)
        if isinstance(_val, np.ndarray):
            if _val.size == 1: physics_relative = bool(_val.reshape(()).item())
            else: raise RuntimeError(f"physics_relative expects scalar/bool, got array shape={_val.shape}")
        else:
            physics_relative = bool(_val)

        K_NEI       = int(cfg.get("k_nei", 500))
        BATCH_Q     = int(cfg.get("knn_batch_q", 2048))
        TRAIN_CHUNK = int(cfg.get("knn_train_chunk", 65536))
        
        # ========= 窗口筛选配置 =========
        use_window_filter = cfg.get("use_window_filter", True)
        window_v = float(cfg.get("window_v", 0.1))
        window_r = float(cfg.get("window_r", 0.2))
        min_candidates = int(cfg.get("min_candidates", 1000))

        # ========= 数据准备 =========
        train_X = np.asarray(train_X, float);    query_X = np.asarray(query_X, float)
        train_zp = np.asarray(train_zp, float);  train_zn = np.asarray(train_zn, float)
        D_all = np.asarray(D_all, float)

        # ====== 稳健校验（显式打印+抛错） ======
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

        # —— 维度自检 & 自动兜底（防止 rho_for_clean != rho_for_model 时仍用 auto）——
        d_model = 2 if (cfg.get("rho_std_for_model", None) is not None) else 1
        if metric in ("grad_dir","grad","tanorm","tn") and (d != d_model):
            print(f"[KNNLocal] d_clean({d}) != d_model({d_model}) -> force grad_mode='physics' & disable predict_fn/predict_torch", flush=True)
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
            if a.size == 1: a = np.full((d,), float(a[0]), dtype=float)
            if b.size == 1: b = np.full((d,), float(b[0]), dtype=float)
            if a.size != d or b.size != d:
                _fail("minmax a/b must be length d", a_size=a.size, b_size=b.size, d=d, a_repr=a, b_repr=b)

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
            _fail("device='cuda' but torch.cuda.is_available()==False —— 请检查CUDA驱动/PyTorch GPU版/CUDA_VISIBLE_DEVICES")

        if use_gpu:
            try:
                dev_name = torch.cuda.get_device_name(torch_device)
            except Exception:
                dev_name = "cuda"
            print(f"[KNNLocal] Using GPU: {dev_name} | candidates={len(train_X)}, queries={len(query_X)}", flush=True)
        else:
            print(f"[KNNLocal] Using CPU path | device={device} | candidates={len(train_X)}, queries={len(query_X)}", flush=True)

        # ========= 候选常驻设备 =========
        Xcand_z_t = torch.as_tensor(train_X, dtype=torch.float32, device=torch_device)  # (N,d)
        N = int(Xcand_z_t.shape[0]); Q = int(query_X.shape[0])
        print(f"[KNNLocal] Xcand tensor on {Xcand_z_t.device}, shape={tuple(Xcand_z_t.shape)}", flush=True)
        
        # ========= 窗口筛选状态 =========
        if use_window_filter:
            print(f"[KNNLocal] Window filtering enabled: window_v={window_v}, window_r={window_r}, min_candidates={min_candidates}", flush=True)
        else:
            print(f"[KNNLocal] Window filtering disabled - using full candidate set", flush=True)

        # ========= physics 常量 =========
        a_t = b_t = None
        if metric == "physics":
            a_np = np.asarray(minmax.get("a") or minmax.get("A"), float).ravel()
            b_np = np.asarray(minmax.get("b") or minmax.get("B"), float).ravel()
            if a_np.size == 1: a_np = np.full((d,), float(a_np[0]), dtype=float)
            if b_np.size == 1: b_np = np.full((d,), float(b_np[0]), dtype=float)
            a_t = torch.as_tensor(a_np.reshape(1, -1), dtype=torch.float32, device=torch_device)
            b_t = torch.as_tensor(b_np.reshape(1, -1), dtype=torch.float32, device=torch_device)

        # ========= 输出容器 =========
        q_hi = np.full((Q,), np.nan, float)
        q_lo = np.full((Q,), np.nan, float)
        
        # ========= 窗口筛选统计 =========
        total_window_expansions = 0
        total_candidates_filtered = 0

        # ========= 批处理循环：查询按 B ；候选按 C =========
        for s in range(0, Q, BATCH_Q):
            e = min(Q, s + BATCH_Q)
            B = e - s

            # --- 本批查询（z） ---
            Zb_np = query_X[s:e, :]                                # (B,d) numpy
            Zb_t  = torch.as_tensor(Zb_np, dtype=torch.float32, device=torch_device)  # (B,d)

            # --- 本批方向：U_b_t (grad/tanorm) 或 Gx_b_t (physics) ---
            if metric == "physics":
                Xi_x = a_t + b_t * Zb_t            # (B,d)
                Gx_b_t = _physics_grad_x_batch(Xi_x)  # (B,d) torch
            else:
                if grad_mode == "physics":
                    G_np = _physics_dir_in_z_batch(Zb_np, minmax)     # (B,d) numpy
                    U_np = G_np / (np.linalg.norm(G_np, axis=1, keepdims=True) + 1e-12)
                    U_b_t = torch.as_tensor(U_np, dtype=torch.float32, device=torch_device)
                else:
                    # 优先使用 autograd 的 torch_predict；否则回退到有限差分
                    if predict_tch is not None:
                        # autograd：每样本 1 前向 + 1 backward
                        G_t = _autograd_grad_z_batch(predict_tch, Zb_t)   # (B,d) torch
                        # 若返回是 torch.Tensor：ok；若是 numpy：转回
                        if not isinstance(G_t, torch.Tensor):
                            G_t = torch.as_tensor(G_t, dtype=torch.float32, device=torch_device)
                        U_b_t = G_t / (torch.linalg.norm(G_t, dim=1, keepdim=True) + 1e-12)
                    elif predict_fn is not None:
                        G_np = _finite_diff_grad_z_batch(predict_fn, Zb_np, grad_eps)  # (B,d) numpy
                        U_np = G_np / (np.linalg.norm(G_np, axis=1, keepdims=True) + 1e-12)
                        U_b_t = torch.as_tensor(U_np, dtype=torch.float32, device=torch_device)
                    else:
                        # 最后兜底到物理方向（如可用），否则用 e1
                        if minmax is not None:
                            G_np = _physics_dir_in_z_batch(Zb_np, minmax)
                            U_np = G_np / (np.linalg.norm(G_np, axis=1, keepdims=True) + 1e-12)
                            U_b_t = torch.as_tensor(U_np, dtype=torch.float32, device=torch_device)
                        else:
                            U_b_t = torch.zeros((B, d), dtype=torch.float32, device=torch_device)
                            U_b_t[:, 0] = 1.0

            # --- 初始化本批“行级 topK”缓存 ---
            best_d = torch.full((B, K_NEI), float("inf"), dtype=torch.float32, device=torch_device)
            best_i = torch.full((B, K_NEI), -1, dtype=torch.int64, device=torch_device)

            # --- 窗口筛选或全量计算 ---
            if use_window_filter:
                # 对本批查询应用窗口筛选
                filtered_indices, expand_count = _window_filter_candidates(
                    Zb_t, Xcand_z_t, window_v, window_r, d, min_candidates, K_NEI
                )
                total_window_expansions += expand_count
                
                # 对每个查询点，使用筛选后的候选计算距离
                for bi in range(B):
                    cand_idx = filtered_indices[bi]
                    n_cand = len(cand_idx)
                    total_candidates_filtered += n_cand
                    
                    if n_cand == 0:
                        continue
                    
                    # 获取筛选后的候选点
                    Xc_t = Xcand_z_t[cand_idx, :]  # (n_cand, d)
                    Zb_single = Zb_t[bi:bi+1, :]    # (1, d)
                    
                    # 计算距离
                    if metric == "physics":
                        Xi_x_single = a_t + b_t * Zb_single
                        Gx_single = _physics_grad_x_batch(Xi_x_single)
                        Xx_c_t = a_t + b_t * Xc_t
                        D_row = _distances_chunk("physics", Zb_single, Xc_t,
                                                 lambda_t=lambda_t,
                                                 a_t=a_t, b_t=b_t, Gx_b_t=Gx_single, Xx_c_t=Xx_c_t,
                                                 physics_relative=physics_relative)  # (1, n_cand)
                        D_row = D_row.squeeze(0)  # (n_cand,)
                    else:
                        U_single = U_b_t[bi:bi+1, :]  # (1, d)
                        D_row = _distances_chunk(metric, Zb_single, Xc_t,
                                                 lambda_t=lambda_t,
                                                 U_b_t=U_single,
                                                 physics_relative=physics_relative)  # (1, n_cand)
                        D_row = D_row.squeeze(0)  # (n_cand,)
                    
                    # 选择 topK
                    k_actual = min(K_NEI, n_cand)
                    if k_actual > 0:
                        topk_vals, topk_idx_local = torch.topk(D_row, k=k_actual, largest=False, sorted=False)
                        best_d[bi, :k_actual] = topk_vals
                        best_i[bi, :k_actual] = cand_idx[topk_idx_local]
            else:
                # 原始的候选分块遍历（无窗口筛选）
                for c0 in range(0, N, TRAIN_CHUNK):
                    c1 = min(N, c0 + TRAIN_CHUNK)
                    Xc_t = Xcand_z_t[c0:c1, :]            # (C,d)
                    C = c1 - c0

                    if metric == "physics":
                        Xx_c_t = a_t + b_t * Xc_t           # (C,d)
                        D_chunk = _distances_chunk("physics", Zb_t, Xc_t,
                                                    lambda_t=lambda_t,
                                                    a_t=a_t, b_t=b_t, Gx_b_t=Gx_b_t, Xx_c_t=Xx_c_t,
                                                    physics_relative=physics_relative)   # (B,C)
                    else:
                        D_chunk = _distances_chunk(metric, Zb_t, Xc_t,
                                                    lambda_t=lambda_t,
                                                    U_b_t=U_b_t,
                                                    physics_relative=physics_relative)   # (B,C)

                    # 合并到本批 topK：把已有 K 与本块 C 拼接，再取前 K
                    idx_block = torch.arange(c0, c1, device=torch_device).view(1, C).expand(B, C)
                    cand_d = torch.cat([best_d, D_chunk], dim=1)              # (B, K+C)
                    cand_i = torch.cat([best_i, idx_block], dim=1)            # (B, K+C)
                    dvals, idx = torch.topk(cand_d, k=K_NEI, dim=1, largest=False, sorted=False)
                    rows = torch.arange(B, device=torch_device).view(B, 1)
                    best_d = dvals
                    best_i = cand_i[rows, idx]

                    # 释放临时张量，降低峰值
                    del D_chunk, idx_block, cand_d, cand_i, dvals, idx


            # --- 将本批 topK 拉回 CPU，做核权重和加权分位 ---
            best_d_np = best_d.detach().cpu().numpy()   # (B,K)
            best_i_np = best_i.detach().cpu().numpy()   # (B,K)

            sigma = np.median(best_d_np, axis=1).reshape(-1, 1)  # (B,1)
            sigma = np.maximum(sigma, 1e-9)

            for bi in range(B):
                idx_row = best_i_np[bi]; d_row = best_d_np[bi]
                w = np.exp(-0.5 * (d_row / sigma[bi, 0]) ** 2)
                zpK = train_zp[idx_row]; mp = (zpK > 0)
                znK = train_zn[idx_row]; mn = (znK > 0)
                qhi_i = _weighted_quantile(zpK[mp], w[mp], TAU_HI) if np.any(mp) else 0.0
                qlo_i = _weighted_quantile(znK[mn], w[mn], TAU_LO) if np.any(mn) else 0.0
                q_hi[s + bi] = max(qhi_i, 0.0)
                q_lo[s + bi] = max(qlo_i, 0.0)

            # 清理本批占用
            del Zb_t
            if metric == "physics":
                del Xi_x, Gx_b_t
            else:
                del U_b_t
            torch.cuda.empty_cache()

        # ========= 窗口筛选统计总结 =========
        if use_window_filter:
            avg_candidates = total_candidates_filtered / Q if Q > 0 else 0
            filter_ratio = (1.0 - avg_candidates / N) * 100 if N > 0 else 0
            print(f"[KNNLocal] Window filtering stats: avg_candidates={avg_candidates:.1f}/{N} ({filter_ratio:.1f}% filtered), expansions={total_window_expansions}/{Q}", flush=True)

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
            c_plus, c_minus = _conformal_scale(val_z_pos, q_hi[val_mask], val_z_neg, q_lo[val_mask], TAU_HI, TAU_LO)

        thr_pos = np.asarray(c_plus * q_hi * D_all, float)
        thr_neg = np.asarray(c_minus * q_lo * D_all, float)

        resid_full = np.asarray(cfg.get("residuals", np.zeros_like(thr_pos)), float)
        is_abnormal = (resid_full > thr_pos) | (resid_full < -thr_neg)

        return ThresholdOutputs(
            thr_pos=thr_pos,
            thr_neg=thr_neg,
            is_abnormal=is_abnormal
        )
