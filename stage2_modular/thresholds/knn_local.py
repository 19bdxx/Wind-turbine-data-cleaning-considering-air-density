# stage2_modular/thresholds/knn_local.py
# -*- coding: utf-8 -*-
"""
=============================================================================
KNN 局部阈值方法：基于K近邻的自适应异常检测
=============================================================================

一、核心思想与动机
-----------------
全局阈值（global）假设所有样本的残差分布相同，但在风机数据清洗场景中，这一假设往往不成立：
  1. **功率曲线分段特性**：低风速段、额定风速段、高风速段的数据密度和噪声特性差异巨大
  2. **环境条件影响**：空气密度 ρ 的变化会导致不同区域的物理特性差异
  3. **局部数据稀疏性**：边缘工况（极低/极高风速）的样本少，全局阈值容易误判
  
KNN 局部阈值通过以下方式解决这些问题：
  - **自适应性**：每个测试点根据其K个近邻的残差分布动态确定阈值
  - **鲁棒性**：使用加权分位数（近邻权重由距离决定），抑制离群邻居的影响
  - **共形校准**：利用验证集对阈值进行 conformal prediction 标定，保证统计覆盖率


二、三种距离度量方法
-------------------
选择合适的距离度量是KNN方法的核心。本实现提供三种方法，适用于不同场景：

1. **physics（物理距离）**
   - **原理**：基于风机功率物理模型 P ≈ k ρ V³，沿功率梯度方向测量距离
   - **优势**：无需训练模型，物理意义明确，适合冷启动或模型不可用场景
   - **计算**：
       * 功率梯度：∇P = (∂P/∂V, ∂P/∂ρ) = (3kρV², kV³)
       * 距离：d = |P(x_c) - P(x_q)| / |P(x_q)|（相对功率差）
   - **适用**：模型训练前的初步筛选、物理约束强的工况

2. **grad_dir（梯度方向距离）**
   - **原理**：使用模型预测的梯度 ∇f(z) 作为"数据流形"的法向量，测量垂直距离
   - **优势**：利用模型学到的数据分布，捕捉非线性功率曲线的局部特征
   - **计算**：
       * 获取梯度：U = ∇f(z) / ||∇f(z)||（单位化）
       * 距离：d = |⟨x_c - x_q, U_q⟩|（候选点到查询点沿法向的投影）
   - **适用**：模型训练后的精细清洗、数据分布复杂的场景
   - **梯度计算方式**：
       - auto（默认）：PyTorch autograd，精确但慢（每样本1次前向+1次反向）
       - finite_diff：有限差分，快速但需合理设置 eps
       - physics：退化为物理梯度（模型与清洗特征空间不一致时自动回退）

3. **tanorm（切+法距离）**
   - **原理**：综合考虑法向距离 d_n 和切向距离 d_t，平衡"功率方向"和"特征空间"的差异
   - **优势**：适合数据分布各向异性的场景，通过 λ 调节切向权重
   - **计算**：
       * d_n = |⟨x_c - x_q, U_q⟩|（法向距离，同 grad_dir）
       * d_t² = ||x_c - x_q||² - d_n²（切向距离，勾股定理）
       * d = √(d_n² + λ·d_t²)（λ 控制切向惩罚强度）
   - **适用**：数据密度各向异性、需要同时考虑"功率相似"和"工况相似"的场景
   - **参数建议**：λ ∈ [1, 10]，值越大越重视切向差异


三、GPU批量优化策略
-------------------
为处理大规模数据集（10万+ 样本），本实现采用多层批处理架构：

1. **查询批（BATCH_Q）**：将 Q 个测试点分成 B 个小批，逐批处理
   - 默认 2048 点/批，平衡显存占用和并行效率
   - 每批独立计算梯度/方向，避免一次性加载所有查询点

2. **候选分块（TRAIN_CHUNK）**：将 N 个训练样本分成 C 大小的块，逐块计算距离
   - 默认 65536 样本/块，适应 GPU 显存限制
   - 距离矩阵分块计算为 (B×C)，避免 (B×N) 的巨大显存峰值

3. **行级 topK 合并**：每处理完一个候选块，立即更新每行的 K 个最近邻
   - 流式处理：cat([已有K, 新块C]) → topK → 更新缓存
   - 内存高效：仅保留 (B×K) 的最近邻索引和距离，立即释放 (B×C) 的块矩阵
   - 时间复杂度：O(Q × N × K) → O(Q × N/C × (K+C))，显著降低峰值

4. **常驻策略**：训练样本特征 Xcand_z_t 一次性加载到 GPU，所有批次复用
   - 减少 CPU↔GPU 数据传输开销
   - 仅在每批查询时传输 (B×d) 的小数据量


四、共形预测校准（Conformal Calibration）
----------------------------------------
局部分位数阈值 q_hi/q_lo 基于训练集估计，可能在验证集上覆盖不足。
共形校准通过验证集计算缩放因子 c_plus/c_minus，保证预定义的覆盖率 τ：

  1. 对验证集每个点 i，计算其局部阈值 q_hi[i]
  2. 计算实际残差与阈值的比值：r_i = z_val[i] / q_hi[i]
  3. 缩放因子：c_plus = quantile(r_i, τ)
  4. 最终阈值：thr_pos = c_plus × q_hi × D_all

意义：即使局部估计有偏，校准后能保证 τ% 的正常样本不被误判为异常。


五、关键性能优化总结
-------------------
  - **为什么分块处理候选集？**  
      避免 (Q×N) 距离矩阵的显存爆炸，分块为 (B×C) 后峰值降至可控范围。
      
  - **为什么用行级topK合并？**  
      流式处理：每块计算完立即合并并丢弃，内存占用稳定在 O(Q×K)，而非 O(Q×N)。
      
  - **为什么常驻GPU？**  
      训练集仅传输一次，查询批次间复用，减少 PCIe 带宽瓶颈。
      
  - **为什么autograd慢？**  
      PyTorch autograd 对 batch 内逐样本求梯度需要 B 次 backward，无法向量化。
      有限差分虽精度略低，但可并行计算 d 个方向，实测快 5-10 倍。


六、使用建议
-----------
  - 初次清洗：用 physics 快速筛选明显异常
  - 精细清洗：模型训练后用 grad_dir 或 tanorm
  - 特征空间不一致（如清洗用2D、模型用1D）：自动回退到 physics
  - K值选择：一般 500-1000，数据量大可增至 2000
  - GPU显存不足：减小 BATCH_Q 或 TRAIN_CHUNK

=============================================================================
"""
import math
import numpy as np
import torch

from .base import ThresholdMethod, ThresholdOutputs



# ===================== 公共：加权分位 / conformal 标定 =====================

def _weighted_quantile(values, weights, q):
    """
    计算加权分位数（Weighted Quantile）
    
    背景：
    ------
    KNN 中不同邻居的"代表性"不同：距离近的邻居更能反映查询点的局部特性。
    因此使用加权分位数，而非简单的无权分位数，使得近邻的影响更大。
    
    参数：
    ------
    values : array-like
        待统计的数值（如 K 个邻居的 z-score）
    weights : array-like
        对应的权重（如高斯核权重 w = exp(-d²/2σ²)）
    q : float
        分位点，范围 [0, 1]。如 q=0.98 表示 98% 分位数
    
    返回：
    ------
    float
        加权分位数。若输入为空或权重和≤0，返回 NaN
    
    算法：
    ------
    1. 按 values 排序，对应调整 weights
    2. 计算累积权重 cumsum(weights)
    3. 目标权重：target = q × total_weight
    4. 找到第一个累积权重 ≥ target 的位置 j
    5. 返回 values[j]
    
    示例：
    ------
    >>> values = [1.0, 2.0, 3.0, 4.0]
    >>> weights = [0.5, 1.0, 1.0, 0.5]  # 中间两个样本权重更高
    >>> _weighted_quantile(values, weights, 0.5)  # 中位数偏向 2.5 附近
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

def _conformal_scale(val_z_pos, q_hi_val, val_z_neg, q_lo_val, tau_hi, tau_lo):
    """
    共形预测标定（Conformal Prediction Calibration）
    
    核心问题：
    ----------
    KNN 局部阈值 q_hi/q_lo 基于训练集的 K 近邻估计，可能在验证集上覆盖不足：
      - 训练集与验证集的局部分布可能有偏差
      - 选择的分位点 τ (如0.98) 未必精确对应真实覆盖率
    
    共形校准通过验证集计算全局缩放因子，保证统计覆盖率。
    
    原理：
    ------
    对验证集每个样本 i：
      - 实际 z-score：z_val[i]（真实残差的标准化值）
      - 预测阈值：q_hi_val[i]（KNN 估计的局部阈值）
      - 比值：r_i = z_val[i] / q_hi_val[i]
    
    若阈值估计完美，r_i 应满足：
      - 100×τ% 的 r_i ≤ 1.0（即真实残差在阈值内）
    
    实际中可能偏大/偏小，因此用 quantile(r_i, τ) 作为缩放因子 c：
      - c > 1：阈值低估，需放大
      - c < 1：阈值高估，需缩小
      - 最终阈值：thr_final = c × q_hi × D_all
    
    参数：
    ------
    val_z_pos : ndarray
        验证集正残差的 z-score（实际值）
    q_hi_val : ndarray
        验证集对应的上阈值预测（KNN 估计）
    val_z_neg : ndarray
        验证集负残差的 z-score（绝对值）
    q_lo_val : ndarray
        验证集对应的下阈值预测
    tau_hi : float
        上阈值目标覆盖率（如 0.98）
    tau_lo : float
        下阈值目标覆盖率
    
    返回：
    ------
    c_plus : float
        上阈值缩放因子
    c_minus : float
        下阈值缩放因子
    
    注意：
    ------
    - 仅使用 z > 0 且 q > 0 的样本计算比值（避免除零和无效数据）
    - 若验证集为空或无有效样本，返回 (1.0, 1.0)（不做校准）
    - 缩放因子下界为 1e-6，避免过小导致阈值失效
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

# ===================== 工具函数 =====================

def _ensure_1d(a):
    """确保输入为一维浮点数组"""
    return np.asarray(a, dtype=float).reshape(-1)

def _physics_grad_x_batch(x_t):
    """
    计算物理功率模型在 X 空间的梯度（批量版）
    
    物理模型：
    ----------
    P = k × ρ × V³  （k 为常数，可略去）
    
    其中：
      - V：风速（第1维）
      - ρ：空气密度（第2维，若存在）
    
    梯度计算：
    ----------
    1D（仅风速 V）：
        ∂P/∂V = 3 × V²
        g_x = (3V²,)
    
    2D（风速 V + 密度 ρ）：
        ∂P/∂V = 3 × ρ × V²
        ∂P/∂ρ = V³
        g_x = (3ρV², V³)
    
    参数：
    ------
    x_t : torch.Tensor, shape (B, d)
        物理空间特征，d ∈ {1, 2}
        - x_t[:, 0]：风速 V
        - x_t[:, 1]：空气密度 ρ（若 d=2）
    
    返回：
    ------
    g_x : torch.Tensor, shape (B, d)
        功率梯度向量
    
    用途：
    ------
    - physics 距离度量：沿功率梯度方向测量距离
    - 退化场景：模型不可用时，用物理梯度代替模型梯度
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
    批量有限差分法计算梯度（Finite Difference Gradient）
    
    优势：
    ------
    - 无需模型可微：仅需黑盒预测函数
    - 并行高效：d 个方向可独立计算，比 autograd 快 5-10 倍
    - 数值稳定：适合非光滑或高噪声的模型
    
    劣势：
    ------
    - 精度依赖 eps：过小则数值误差大，过大则逼近误差大
    - 多调用预测：需要 d+1 次前向传播（1次中心点 + d次扰动点）
    
    算法：
    ------
    对每个样本 z_i：
        f(z_i) = center_pred_fn(z_i)
        ∂f/∂z_k ≈ [f(z_i + eps·e_k) - f(z_i)] / eps
    
    其中 e_k 是第 k 维的单位向量。
    
    参数：
    ------
    center_pred_fn : callable
        预测函数，接受 (N, d) numpy 数组，返回 (N,) numpy 数组
    Z_b : ndarray, shape (B, d)
        批量查询点（z 空间特征）
    eps : float or array-like
        扰动步长。可以是标量（所有维度相同）或长度 d 的数组（每维不同）
        建议值：0.01 ~ 0.1，需根据特征归一化范围调整
    
    返回：
    ------
    G : ndarray, shape (B, d)
        梯度矩阵，G[i, k] = ∂f/∂z_k (z_i)
    
    性能提示：
    ----------
    - center_pred_fn 应支持批量输入，内部向量化计算可大幅加速
    - 若模型支持 GPU，将 Z_b 放在 GPU 上可进一步提速
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
    使用 PyTorch Autograd 计算每个样本的梯度 ∇_z f(z)
    
    优势：
    ------
    - 精度高：反向传播计算的梯度精确（无数值逼近误差）
    - 灵活：适用于任意可微模型（神经网络、复杂复合函数等）
    
    劣势与效率问题：
    ----------------
    **为什么慢？**
    PyTorch 的 autograd 设计用于"单次 loss.backward() 计算所有参数的梯度"，
    而非"对 batch 中每个样本独立计算输出对输入的梯度"。
    
    当前实现采用"逐样本 backward"策略：
      1. 一次前向：out = torch_predict(Zb_t)  # (B,)
      2. B 次反向：对每个 out[i] 单独 backward，获取 ∇out[i]/∇Zb_t[i]
    
    问题：
      - backward 无法向量化：每次只能计算一个标量对 Zb_t 的梯度
      - 计算图开销：需保留 graph 以支持多次 backward（retain_graph=True）
      - 实测比有限差分慢 5-10 倍（对于 d=1 或 d=2 的低维情况）
    
    改进方向（需 PyTorch >= 1.9）：
    --------------------------------
    可使用 torch.func.vmap + torch.func.jacrev 实现向量化雅可比计算，
    但需要模型支持函数式编程风格（functorch）。当前实现保持向后兼容。
    
    参数：
    ------
    torch_predict : callable
        可微预测函数，接受 (B, d) torch.Tensor（requires_grad=True），
        返回 (B,) torch.Tensor
    Zb_t : torch.Tensor, shape (B, d)
        批量输入，位于指定设备上
    
    返回：
    ------
    G : torch.Tensor, shape (B, d)
        梯度矩阵，G[i] = ∇f(z_i)
    
    注意事项：
    ----------
    - 模型必须支持 requires_grad 和反向传播
    - 若模型内部包含不可微操作（如 argmax、round），会报错
    - 梯度计算在同一设备上进行（GPU/CPU）
    
    使用建议：
    ----------
    - 对于低维特征（d ≤ 2），优先使用有限差分
    - 对于高维或梯度精度要求极高的场景，使用 autograd
    - 若模型复杂且 GPU 充足，autograd 的精度优势更明显
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
    在 Z 空间（归一化特征空间）中使用物理方向作为法向量
    
    应用场景：
    ----------
    当模型预测函数不可用时（predict_fn / predict_torch 为 None），
    使用物理功率梯度方向代替模型梯度方向进行距离度量。
    
    原理：
    ------
    虽然输入 Z_b 是归一化空间的坐标，但通过逆变换可还原到物理空间：
        X = a + b ⊙ Z  （⊙ 表示逐元素乘法）
    
    然后计算物理功率梯度 ∇P(X)，作为"数据流形"的法向量方向。
    
    注意：
    ------
    - 返回的梯度 G 在 X 空间中计算，但会被调用者归一化为单位向量
    - 仅使用方向信息（单位化后），数值大小无意义
    - 这是 grad_mode="physics" 或模型不可用时的退路方案
    
    参数：
    ------
    Z_b : ndarray, shape (B, d)
        归一化空间的查询点
    minmax : dict
        归一化参数，包含 {'a': a, 'b': b} 或 {'A': a, 'B': b}
        其中 a 是偏移，b 是缩放因子
    
    返回：
    ------
    G : ndarray, shape (B, d)
        物理梯度向量（未归一化）
    
    实现细节：
    ----------
    1. 逆变换：X_b = a + b * Z_b
    2. 计算功率梯度：
       - 1D：G = (3V²,)
       - 2D：G = (3ρV², V³)
    3. 调用者会进一步归一化：U = G / ||G||
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

# ===================== 批量 GPU 距离（B×C 分块） =====================

def _distances_chunk(metric, Zb_t, Xc_t, *,
                     lambda_t,
                     # grad/tanorm：
                     U_b_t=None,
                     # physics：
                     a_t=None, b_t=None, Gx_b_t=None, Xx_c_t=None,
                     physics_relative=True):
    """
    GPU 批量计算查询批次与候选分块之间的距离矩阵
    
    核心设计：
    ----------
    为支持大规模数据（10万+ 样本），采用"批量查询 × 分块候选"策略：
      - 查询批：Zb_t (B, d) —— B 个测试点（如 2048）
      - 候选块：Xc_t (C, d) —— C 个训练样本（如 65536）
      - 输出：D (B, C) —— 距离矩阵
    
    全 GPU 计算：
      - 所有张量常驻 GPU，无 CPU↔GPU 传输
      - 利用矩阵乘法并行化（cuBLAS 加速）
      - 分块处理避免 (Q×N) 的显存爆炸
    
    三种距离度量：
    ----------------
    
    1. **physics（物理功率距离）**
       原理：沿功率梯度方向测量功率差异
       计算：d[i,j] = |P(x_c[j]) - P(x_q[i])| / |P(x_q[i])|
       
       详细步骤：
         (1) 转换到物理空间：
             X_q = a + b ⊙ Zb_t  # (B,d)
             X_c = a + b ⊙ Xc_t  # (C,d)
         (2) 计算功率梯度：
             G_x[i] = ∇P(X_q[i])  # (B,d)
         (3) 功率投影（向量化矩阵乘法）：
             S = X_c @ G_x^T  # (C,B)  —— 候选功率
             c = ⟨X_q, G_x⟩  # (B,)    —— 查询功率
         (4) 功率差异：
             dP = |S - c|^T  # (B,C)
         (5) 相对化（可选）：
             dP /= |P(X_q[i])|  # 消除风速量级影响
       
       优势：
         - 无需模型，冷启动可用
         - 物理意义明确
       
       劣势：
         - 假设线性功率模型（实际曲线可能非线性）
    
    2. **grad_dir（梯度方向距离）**
       原理：使用模型梯度 ∇f(z) 作为局部法向，测量垂直距离
       计算：d[i,j] = |⟨x_c[j] - x_q[i], U_q[i]⟩|
       
       详细步骤：
         (1) 梯度单位化（在外部完成）：
             U_b[i] = ∇f(z_q[i]) / ||∇f(z_q[i])||  # (B,d)
         (2) 法向投影（向量化）：
             S = Xc_t @ U_b^T  # (C,B)
             c = ⟨Zb_t, U_b⟩  # (B,)
             d_n = |S - c|^T   # (B,C)
       
       优势：
         - 利用模型学到的数据流形
         - 适应非线性功率曲线
       
       劣势：
         - 需要模型梯度（autograd 或有限差分）
    
    3. **tanorm（切向+法向复合距离）**
       原理：同时考虑法向距离 d_n 和切向距离 d_t
       计算：d = √(d_n² + λ·d_t²)
       
       详细步骤：
         (1) 法向距离：d_n（同 grad_dir）
         (2) 总距离：||x_c - x_q||²（欧式距离）
         (3) 切向距离：d_t² = ||Δx||² - d_n²（勾股定理）
         (4) 加权组合：d = √(d_n² + λ·d_t²)
       
       参数 λ 的作用：
         - λ=0：退化为 grad_dir（仅法向）
         - λ=1：等权重（欧式距离的变形）
         - λ>1：加大切向惩罚（强调"工况相似"）
       
       适用场景：
         - 数据分布各向异性
         - 需要同时考虑"功率方向"和"特征空间"的相似性
    
    参数：
    ------
    metric : str
        距离度量类型，{"physics", "grad_dir", "grad", "tanorm", "tn"}
    Zb_t : torch.Tensor, shape (B, d)
        查询批次，位于 GPU
    Xc_t : torch.Tensor, shape (C, d)
        候选分块，位于 GPU
    lambda_t : float
        tanorm 的切向权重（仅 tanorm 使用）
    U_b_t : torch.Tensor, shape (B, d), optional
        查询点的单位梯度向量（grad_dir / tanorm 使用）
    a_t, b_t : torch.Tensor, shape (1, d), optional
        归一化参数（physics 使用）
    Gx_b_t : torch.Tensor, shape (B, d), optional
        查询点的功率梯度（physics 使用）
    Xx_c_t : torch.Tensor, shape (C, d), optional
        候选点的物理坐标（physics 使用，预计算优化）
    physics_relative : bool
        是否使用相对功率距离（physics 使用）
    
    返回：
    ------
    D : torch.Tensor, shape (B, C)
        距离矩阵，D[i,j] = distance(Zb_t[i], Xc_t[j])
    
    性能优化：
    ----------
    - 矩阵乘法代替循环：O(BCD) 的 GPU 并行操作
    - 广播机制：避免显式扩展张量
    - 向量化 clamp：批量限制数值稳定性
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
        KNN 局部阈值计算主函数
        
        算法流程概览：
        ===============
        1. **配置解析**：从 cfg 提取距离度量、梯度模式、批量参数等
        2. **数据验证**：检查维度一致性、minmax 参数完整性
        3. **设备选择**：根据 CUDA 可用性决定 GPU/CPU 执行
        4. **候选常驻**：将训练集加载到 GPU（一次传输，多次复用）
        5. **批处理循环**：
             For each query_batch(B)：
               - 计算梯度/方向（physics/autograd/finite_diff）
               - For each candidate_chunk(C)：
                   * 计算距离矩阵 (B×C)
                   * 行级 topK 合并
               - 加权分位数计算局部阈值
        6. **Conformal 校准**：验证集计算缩放因子
        7. **异常判定**：residuals 与阈值比较
        
        数据流图：
        ===========
        
                    ┌─────────────────────────────────────────────┐
                    │          输入数据（CPU/Numpy）               │
                    │  train_X(N,d), query_X(Q,d), train_zp/zn    │
                    └─────────────────┬───────────────────────────┘
                                      ↓
                    ┌─────────────────────────────────────────────┐
                    │     转换为 Torch 张量 + 移至 GPU            │
                    │   Xcand_z_t(N,d) → GPU (常驻)               │
                    └─────────────────┬───────────────────────────┘
                                      ↓
        ┌──────────────────────────────────────────────────────────┐
        │              查询批处理循环 (Q / BATCH_Q 次)              │
        └──────────────────────────────────────────────────────────┘
                │
                ↓ For s in range(0, Q, BATCH_Q):
                │
                ├─→ [Step 1] 查询批次切片：Zb_t = query_X[s:e] → GPU
                │
                ├─→ [Step 2] 计算方向向量：
                │      ┌─ physics: Gx_b_t = ∇P(a+b⊙Zb_t)
                │      ├─ autograd: U_b_t = ∇f(Zb_t) / ||∇f||
                │      ├─ finite_diff: U_b_t ≈ Δf/Δz
                │      └─ physics_dir: U_b_t = 物理梯度方向
                │
                ├─→ [Step 3] 初始化 topK 缓存：
                │      best_d = [∞] × (B, K)
                │      best_i = [-1] × (B, K)
                │
                ├─→ [Step 4] 候选分块循环 (N / TRAIN_CHUNK 次):
                │      For c0 in range(0, N, TRAIN_CHUNK):
                │        │
                │        ├─ 切片：Xc_t = Xcand_z_t[c0:c1]  # (C,d)
                │        │
                │        ├─ 距离计算：D_chunk = _distances_chunk(...)  # (B,C)
                │        │    └─ GPU 矩阵乘法：Xc_t @ U_b_t^T
                │        │
                │        ├─ topK 合并（流式）：
                │        │    cat([best_d(B,K), D_chunk(B,C)]) → (B, K+C)
                │        │    topK(dim=1, k=K) → best_d(B,K), best_i(B,K)
                │        │
                │        └─ 释放临时：del D_chunk, idx_block, ...
                │
                ├─→ [Step 5] 拉回 CPU：
                │      best_d_np, best_i_np = best_d.cpu(), best_i.cpu()
                │
                ├─→ [Step 6] 局部阈值计算（逐样本）：
                │      For bi in range(B):
                │        ├─ 核权重：w[k] = exp(-0.5 × (d[k]/σ)²)
                │        ├─ 正阈值：q_hi[bi] = weighted_quantile(train_zp[nei], w, τ_hi)
                │        └─ 负阈值：q_lo[bi] = weighted_quantile(train_zn[nei], w, τ_lo)
                │
                └─→ [Step 7] 清理：del Zb_t, U_b_t; torch.cuda.empty_cache()
        
        ┌──────────────────────────────────────────────────────────┐
        │           Conformal 校准 (验证集)                         │
        └──────────────────────────────────────────────────────────┘
                │
                ├─→ 计算比值：r[i] = val_z[i] / q_hi_val[i]
                ├─→ 缩放因子：c_plus = quantile(r, τ_hi)
                └─→ 最终阈值：thr_pos = c_plus × q_hi × D_all
        
        ┌──────────────────────────────────────────────────────────┐
        │                 异常判定                                  │
        └──────────────────────────────────────────────────────────┘
                │
                └─→ is_abnormal = (resid > thr_pos) | (resid < -thr_neg)
        
        关键性能优化策略：
        ===================
        
        1. **为什么分块处理候选集？**
           问题：Q×N 距离矩阵在大数据下显存爆炸
             - 如 Q=10000, N=100000：需 4GB（float32）
           方案：分块为 B×C，峰值显存：
             - B=2048, C=65536：仅 512MB
           代价：C 个块需 N/C 次迭代，但每次内存可控
        
        2. **为什么用行级 topK 合并？**
           朴素方案：存储所有 (Q×N) 距离 → 最后全局 topK
             - 显存：O(Q×N)，不可接受
           流式方案：每块立即合并 topK，丢弃非近邻
             - 显存：O(Q×K)，稳定可控
             - 算法：cat([已有K, 新块C]) → topK(K) → 更新缓存
             - 正确性：最终 K 个近邻一定在某个块的 topK 中
        
        3. **为什么常驻 GPU？**
           候选集 Xcand_z_t(N,d) 对所有查询批次复用：
             - 一次 CPU→GPU 传输（~1GB）
             - Q/B 次查询批次，每次仅传输 Zb_t(B,d)（~KB）
             - PCIe 带宽节省：O(N) → O(Q×d/B + N)
        
        4. **梯度计算的权衡**
           | 方法          | 精度 | 速度      | 依赖              |
           |---------------|------|-----------|-------------------|
           | autograd      | 高   | 慢(1×)    | 可微模型          |
           | finite_diff   | 中   | 快(5-10×) | 黑盒预测函数      |
           | physics       | 低   | 极快      | minmax参数        |
           
           自动回退机制：
             - 优先 autograd（若 predict_torch 可用）
             - 次选 finite_diff（若 predict_fn 可用）
             - 兜底 physics（始终可用）
        
        5. **维度不一致的处理**
           场景：清洗用 2D（V+ρ），但模型仅用 1D（V）
           问题：梯度维度不匹配
           方案：自动检测 d_clean ≠ d_model
             - 强制 grad_mode="physics"
             - 禁用 predict_fn/predict_torch
             - 打印警告日志
        
        参数说明：
        ==========
        train_X : ndarray, shape (N, d)
            训练集特征（归一化空间），d ∈ {1, 2}
        train_zp : ndarray, shape (N,)
            训练集正残差 z-score（用于上阈值估计）
        train_zn : ndarray, shape (N,)
            训练集负残差 z-score（用于下阈值估计）
        query_X : ndarray, shape (Q, d)
            查询集特征（待计算阈值的样本）
        D_all : ndarray, shape (Q,)
            不确定性估计（用于阈值缩放）
        idx_train_mask : array-like
            训练集mask（用于区分训练/验证）
        idx_val_mask : array-like
            验证集mask（用于 conformal 校准）
        taus : tuple of (tau_hi, tau_lo)
            分位点参数，如 (0.98, 0.98)
        cfg : dict
            配置字典，包含：
            
            **距离度量配置：**
              - metric : str, {"physics", "grad_dir", "tanorm"}
                  距离类型（默认 "tanorm"）
              - lambda_t : float
                  tanorm 切向权重（默认 6.0）
              - physics_relative : bool
                  physics 是否使用相对距离（默认 True）
            
            **梯度计算配置：**
              - grad_mode : str, {"auto", "physics"}
                  梯度计算模式（默认 "auto"）
              - grad_eps : float or array
                  有限差分步长（默认 0.1）
              - predict_fn : callable, (N,d) → (N,)
                  NumPy 预测函数（有限差分用）
              - predict_torch : callable, (B,d) → (B,)
                  PyTorch 可微预测函数（autograd 用）
            
            **归一化参数：**
              - minmax : dict, {"a": a, "b": b}
                  X = a + b * Z 的参数
            
            **批处理参数：**
              - k_nei : int
                  近邻数量（默认 500）
              - knn_batch_q : int
                  查询批大小（默认 2048）
              - knn_train_chunk : int
                  候选分块大小（默认 65536）
            
            **Conformal 校准：**
              - val_z_pos, val_z_neg : ndarray
                  验证集 z-score（可选，否则从 residuals 计算）
        
        device : str, optional
            计算设备，{"cuda", "cpu"}
            若为 "cuda" 但 CUDA 不可用，会报错
        
        返回：
        ======
        ThresholdOutputs : namedtuple
            包含：
            - thr_pos : ndarray, shape (Q,)
                上阈值
            - thr_neg : ndarray, shape (Q,)
                下阈值
            - is_abnormal : ndarray, shape (Q,)
                异常标记（True 为异常）
        
        异常处理：
        ==========
        函数内部对各种错误情况进行显式检查，并抛出 RuntimeError：
          - 维度不匹配（train_X.ndim ≠ 2 或 d ∉ {1, 2}）
          - minmax 参数缺失（physics 度量必需）
          - CUDA 请求但不可用（device="cuda" 但无 GPU）
          - 梯度计算失败（所有方法都回退失败）
        
        注意事项：
        ==========
        - 首次调用会有 GPU 预热开销（~1秒）
        - 显存占用峰值 ≈ BATCH_Q × TRAIN_CHUNK × 4 bytes
        - K 过大会降低速度但不会显著提高准确性（推荐 500-1000）
        - autograd 比 finite_diff 慢 5-10 倍，除非高维或精度要求极高
        """
        # ========= 配置解析：从 cfg 提取参数 =========
        # 分位点参数：决定异常覆盖率（如 0.98 表示保留 98% 正常数据）
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

        # ========= 数据准备：确保输入为 numpy 浮点数组 =========
        train_X = np.asarray(train_X, float);    query_X = np.asarray(query_X, float)
        train_zp = np.asarray(train_zp, float);  train_zn = np.asarray(train_zn, float)
        D_all = np.asarray(D_all, float)

        # ====== 稳健校验：提前检测错误配置，给出明确诊断信息 ======
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

        # ========= 设备选择：根据 CUDA 可用性决定 GPU/CPU 执行 =========
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

        # ========= 候选常驻 GPU：一次传输，所有批次复用 =========
        Xcand_z_t = torch.as_tensor(train_X, dtype=torch.float32, device=torch_device)  # (N,d)
        N = int(Xcand_z_t.shape[0]); Q = int(query_X.shape[0])
        print(f"[KNNLocal] Xcand tensor on {Xcand_z_t.device}, shape={tuple(Xcand_z_t.shape)}", flush=True)

        # ========= physics 度量预处理：归一化参数转为 GPU 张量 =========
        a_t = b_t = None
        if metric == "physics":
            a_np = np.asarray(minmax.get("a") or minmax.get("A"), float).ravel()
            b_np = np.asarray(minmax.get("b") or minmax.get("B"), float).ravel()
            if a_np.size == 1: a_np = np.full((d,), float(a_np[0]), dtype=float)
            if b_np.size == 1: b_np = np.full((d,), float(b_np[0]), dtype=float)
            a_t = torch.as_tensor(a_np.reshape(1, -1), dtype=torch.float32, device=torch_device)
            b_t = torch.as_tensor(b_np.reshape(1, -1), dtype=torch.float32, device=torch_device)

        # ========= 输出容器：预分配结果数组（NaN 填充） =========
        q_hi = np.full((Q,), np.nan, float)
        q_lo = np.full((Q,), np.nan, float)

        # ========= 批处理循环：分批处理查询点，避免一次性加载所有数据 =========
        # 外层循环：遍历查询批次（每批 BATCH_Q 个点）
        # 内层循环：遍历候选分块（每块 TRAIN_CHUNK 个样本）
        # 双层循环实现 O(Q×N/chunk) 的内存复杂度
        for s in range(0, Q, BATCH_Q):
            e = min(Q, s + BATCH_Q)
            B = e - s

            # --- 本批查询点：从 CPU 传输到 GPU ---
            Zb_np = query_X[s:e, :]                                # (B,d) numpy
            Zb_t  = torch.as_tensor(Zb_np, dtype=torch.float32, device=torch_device)  # (B,d)

            # --- 计算方向向量：用于距离度量 ---
            # physics: 功率梯度 Gx_b_t = ∇P(X)
            # grad_dir/tanorm: 单位梯度 U_b_t = ∇f(z) / ||∇f||
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

            # --- 候选分块遍历：分块计算距离，避免 (B×N) 的显存爆炸 ---
            # 每块大小 C = min(TRAIN_CHUNK, N-c0)
            # 块内计算距离矩阵 D_chunk(B, C)，立即与已有 topK 合并
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

                # ========= 行级 topK 合并（关键优化）=========
                # 策略：cat([已有K, 新块C]) → topK(K) → 更新缓存
                # 优势：内存稳定在 O(B×K)，避免存储完整 (B×N)
                # 正确性：最终 K 近邻一定在某个块的 topK 中
                idx_block = torch.arange(c0, c1, device=torch_device).view(1, C).expand(B, C)
                cand_d = torch.cat([best_d, D_chunk], dim=1)              # (B, K+C)
                cand_i = torch.cat([best_i, idx_block], dim=1)            # (B, K+C)
                dvals, idx = torch.topk(cand_d, k=K_NEI, dim=1, largest=False, sorted=False)
                rows = torch.arange(B, device=torch_device).view(B, 1)
                best_d = dvals
                best_i = cand_i[rows, idx]

                # 释放临时张量：立即回收显存，降低峰值占用
                del D_chunk, idx_block, cand_d, cand_i, dvals, idx

            # --- 将本批 topK 拉回 CPU：后续加权分位数计算在 numpy 中进行 ---
            # GPU → CPU 传输：仅 (B×K) 的距离和索引，数据量小
            best_d_np = best_d.detach().cpu().numpy()   # (B,K)
            best_i_np = best_i.detach().cpu().numpy()   # (B,K)

            # ========= 加权分位数计算：近邻越近，权重越大 =========
            # 高斯核带宽：σ = median(距离)，自适应数据密度
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

        # ========= Conformal 校准：保证统计覆盖率 =========
        # 原理：用验证集计算 c = quantile(真实残差/预测阈值, τ)
        # 作用：即使局部阈值有偏，校准后能保证 τ% 的覆盖率
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
