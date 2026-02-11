# -*- coding: utf-8 -*-
"""
分位数回归模型 (Quantile Regression Model) - 风机异常检测边界估计模块

本模块实现了基于多层感知器(MLP)的分位数回归模型，用于估计风机功率预测的上下边界。
与center.py中的中心模型不同，分位数模型不预测单个期望值，而是预测多个分位数值。

核心功能：
---------
1. **分位数MLP**: 神经网络输出多个分位数预测值（如0.01、0.05、0.5、0.95、0.99分位数）
2. **Pinball损失**: 使用不对称的分位数损失函数，确保预测值符合对应分位数的统计意义
3. **防交叉约束**: 添加惩罚项防止不同分位数曲线交叉（如0.95分位数应大于0.5分位数）
4. **混合精度训练**: 使用AMP加速训练，支持GPU-CACHED和STREAMING两种模式

与中心模型的区别：
----------------
**中心模型 (center.py)**:
- 目标: 预测功率的期望值(均值)
- 输出: 单个值
- 损失: MSE/WMSE/Huber_z
- 用途: 估计"正常"功率值

**分位数模型 (quantile.py)**:
- 目标: 预测功率的不同分位数（如5%、95%分位数）
- 输出: 多个值（每个分位数一个）
- 损失: Pinball loss (不对称损失)
- 用途: 估计功率的上下界，用于异常检测

应用场景：
---------
在风机数据清洗中，分位数模型用于设定异常检测的阈值：
1. 训练得到多个分位数模型（如τ=0.01, 0.05, 0.95, 0.99）
2. 对每个数据点，预测其对应的分位数值
3. 如果实际功率 < 0.01分位数 或 > 0.99分位数，则判定为异常
4. 通过调整分位数（如0.05/0.95 vs 0.01/0.99）可以控制异常检测的严格程度

分位数含义：
-----------
τ=0.5: 中位数（50%分位数），将数据平分为两半
τ=0.95: 95%分位数，只有5%的数据点功率高于此值（上界）
τ=0.05: 5%分位数，只有5%的数据点功率低于此值（下界）
τ=0.01/0.99: 更严格的1%/99%分位数边界
"""
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..core.device import resolve_device
from ..core.utils import Stopwatch

class QuantileMLP(nn.Module):
    """
    分位数多层感知器 (Quantile Multi-Layer Perceptron)
    
    与center.py中的MLP类似，但关键区别在于：
    - **MLP输出**: 1个值（功率的期望值）
    - **QuantileMLP输出**: len(taus)个值（多个分位数预测值）
    
    例如：taus=[0.05, 0.5, 0.95]时，网络输出3个值，分别对应5%、50%、95%分位数
    
    网络结构：
    ---------
    输入: [v] 或 [v, ρ]
    ↓ 隐藏层（与MLP相同）
    Linear -> 激活 -> Dropout -> ... -> Linear -> 激活 -> Dropout
    ↓ 输出层（关键区别）
    Linear(hidden_last -> len(taus))  [输出多个分位数值]
    
    参数说明：
    ---------
    in_dim : int
        输入特征维度（1: 仅风速, 2: 风速+密度）
    taus : list of float
        要预测的分位数列表，如[0.01, 0.05, 0.5, 0.95, 0.99]
        - 每个值必须在(0, 1)之间
        - 建议按升序排列，便于后续防交叉约束
    hidden : list of int, default=[512,512,256,128]
        隐藏层配置（与MLP相同）
    act : str, default="relu"
        激活函数类型（与MLP相同）
    dropout : float, default=0.05
        Dropout概率（与MLP相同）
    
    输出示例：
    ---------
    假设输入batch_size=1024, taus=[0.05, 0.5, 0.95]
    forward(x) -> shape: (1024, 3)
    - 第0列: 所有样本的5%分位数预测
    - 第1列: 所有样本的50%分位数预测（中位数）
    - 第2列: 所有样本的95%分位数预测
    """
    def __init__(self, in_dim: int, taus, hidden=[512,512,256,128], act="relu", dropout=0.05):
        super().__init__()
        self.taus = list(taus)  # 保存分位数列表
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}
        layers=[]; prev=in_dim
        # 构建隐藏层（与MLP完全相同）
        for h in hidden:
            layers += [nn.Linear(prev,h), acts.get(act, nn.ReLU())]
            if dropout and dropout>0:
                layers += [nn.Dropout(dropout)]
            prev=h
        # 输出层：输出len(taus)个值（关键区别）
        layers += [nn.Linear(prev, len(self.taus))]
        self.net = nn.Sequential(*layers)
    def forward(self,x): 
        return self.net(x)

def pinball_loss(pred: torch.Tensor, y: torch.Tensor, taus):
    """
    分位数损失函数 (Pinball Loss / Quantile Loss)
    
    分位数回归的核心损失函数，确保模型预测的是真实的τ分位数。
    与MSE等对称损失不同，Pinball损失是**不对称的**，对高估和低估的惩罚力度不同。
    
    数学原理：
    ---------
    对于分位数τ，Pinball损失定义为：
    
    L_τ(y, ŷ) = { τ * (y - ŷ)      if y ≥ ŷ  (预测偏低，实际值更大)
                { (τ-1) * (y - ŷ)  if y < ŷ  (预测偏高，实际值更小)
    
    简化形式：L_τ = max(τ * e, (τ-1) * e)，其中 e = y - ŷ
    
    为什么不对称：
    -----------
    - **高分位数(τ=0.95)**: 我们希望只有5%的点超过预测值
      → 如果预测偏低(y > ŷ)，惩罚应该很大：τ * e = 0.95 * e
      → 如果预测偏高(y < ŷ)，惩罚应该很小：(τ-1) * e = -0.05 * e
      → 这样模型会倾向于预测高一点，确保大部分点都在预测值之下
    
    - **低分位数(τ=0.05)**: 我们希望只有5%的点低于预测值
      → 如果预测偏低(y > ŷ)，惩罚应该很小：τ * e = 0.05 * e
      → 如果预测偏高(y < ŷ)，惩罚应该很大：(τ-1) * e = -0.95 * e
      → 这样模型会倾向于预测低一点，确保大部分点都在预测值之上
    
    - **中位数(τ=0.5)**: 对称损失，等价于MAE(平均绝对误差)
    
    参数说明：
    ---------
    pred : torch.Tensor, shape (batch_size, len(taus))
        模型预测的多个分位数值
    y : torch.Tensor, shape (batch_size, 1)
        真实的功率值
    taus : list of float
        分位数列表，如[0.05, 0.5, 0.95]
    
    返回：
    -----
    loss : torch.Tensor, scalar
        所有分位数的总损失（各分位数损失之和）
    
    计算示例：
    ---------
    假设 taus=[0.05, 0.95], y=100, pred=[80, 120]
    - 对τ=0.05, ŷ=80: e=100-80=20, loss=max(0.05*20, -0.95*20)=1 (预测偏低但合理)
    - 对τ=0.95, ŷ=120: e=100-120=-20, loss=max(0.95*(-20), -0.05*(-20))=1 (预测偏高但合理)
    """
    diff = y - pred  # shape: [B, T], 误差矩阵（实际值 - 预测值）
    losses = []
    for i, tau in enumerate(taus):
        e = diff[:, i]  # 第i个分位数的误差向量
        # Pinball损失：max(τ*e, (τ-1)*e)
        # 当e>0(预测偏低)：取τ*e
        # 当e<0(预测偏高)：取(τ-1)*e（负数）
        losses.append(torch.maximum(tau * e, (tau - 1.0) * e).mean())
    return torch.stack(losses).sum()  # 所有分位数损失之和

def non_crossing_penalty(pred: torch.Tensor, weight: float = 1.0):
    """
    防交叉惩罚项 (Non-Crossing Penalty)
    
    分位数回归的一个重要约束：**不同分位数曲线不应交叉**
    理论上应满足：q_0.05(x) ≤ q_0.5(x) ≤ q_0.95(x) ≤ q_0.99(x)
    
    问题背景：
    ---------
    在实际训练中，如果只使用Pinball损失，可能会出现分位数交叉现象：
    - 某些输入点上，预测的95%分位数 < 50%分位数
    - 某些输入点上，预测的50%分位数 < 5%分位数
    这违反了分位数的数学定义，导致结果不可解释。
    
    解决方案：
    ---------
    添加惩罚项，对所有相邻分位数的"逆序"情况进行惩罚：
    penalty = sum(ReLU(q_i - q_{i+1}))
    
    - 如果 q_i < q_{i+1}（正序）：ReLU(q_i - q_{i+1}) = 0，无惩罚
    - 如果 q_i > q_{i+1}（逆序）：ReLU(q_i - q_{i+1}) > 0，产生惩罚
    
    参数说明：
    ---------
    pred : torch.Tensor, shape (batch_size, len(taus))
        模型预测的多个分位数值，**假设按τ升序排列**
        例如：pred[:, 0]=q_0.05, pred[:, 1]=q_0.5, pred[:, 2]=q_0.95
    weight : float, default=1.0
        惩罚权重，控制约束的强度
        - 0: 不施加约束（可能出现交叉）
        - 1-10: 常用范围
        - 过大: 可能过度约束，导致所有分位数过于接近
    
    返回：
    -----
    penalty : torch.Tensor, scalar
        交叉惩罚值，值越大说明交叉越严重
    
    计算示例：
    ---------
    假设batch中某个样本的预测为 [90, 85, 95] (τ=[0.05, 0.5, 0.95])
    - diff[0] = 90 - 85 = 5 > 0  (第1、2个分位数交叉！)
    - diff[1] = 85 - 95 = -10 < 0 (第2、3个分位数正常)
    - ReLU([5, -10]) = [5, 0]
    - penalty = (5 + 0) / 2 = 2.5 (取平均)
    
    实际应用：
    ---------
    总损失 = pinball_loss + weight * non_crossing_penalty
    通过调整weight平衡两个目标：
    - 更小的weight: 更好地拟合数据，但可能交叉
    - 更大的weight: 严格防止交叉，但可能牺牲拟合精度
    """
    # 如果只有1个分位数，或不使用约束，则返回0
    if pred.shape[1] <= 1 or weight <= 0:
        return pred.new_tensor(0.0)
    
    # 计算相邻分位数的差值：q_i - q_{i+1}
    # pred[:, :-1]: [q_0, q_1, ..., q_{n-2}]
    # pred[:, 1:]:  [q_1, q_2, ..., q_{n-1}]
    diffs = pred[:, :-1] - pred[:, 1:]
    
    # ReLU(diff): 只惩罚逆序的情况（diff > 0）
    penalty = torch.relu(diffs).mean()
    
    return weight * penalty

def fit_quantile_mlp(Xtr, ytr, Xva, yva, taus, cfg=None, device="auto", verbose=1, gpu_cache_limit_bytes=20*1024**3):
    """
    训练分位数回归MLP模型
    
    本函数与center.py中的fit_mlp_center高度相似，但有关键区别。
    
    与fit_mlp_center的相似点：
    -----------------------
    1. **训练框架相同**: Adam优化器 + Early Stopping + 混合精度训练
    2. **双模式支持**: GPU-CACHED模式（显存充足）和STREAMING模式（显存不足）
    3. **超参数配置**: hidden、act、dropout、lr、l2、epochs、patience、batch等完全相同
    4. **训练流程**: 随机打乱 -> 分batch训练 -> 验证集评估 -> 保存最佳模型
    
    与fit_mlp_center的关键区别：
    -------------------------
    1. **模型类型**: QuantileMLP vs MLP
       - QuantileMLP输出len(taus)个值（多个分位数）
       - MLP输出1个值（期望值）
    
    2. **损失函数**: pinball_loss + non_crossing_penalty vs MSE/WMSE/Huber_z
       - pinball_loss: 不对称的分位数损失
       - non_crossing_penalty: 防止分位数曲线交叉
       - MSE/WMSE/Huber_z: 对称损失，最小化预测误差
    
    3. **输入参数**: 需要传入taus（分位数列表）
       - taus=[0.01, 0.05, 0.5, 0.95, 0.99]: 训练5个分位数
       - taus=[0.5]: 退化为单分位数（中位数回归）
    
    4. **超参数**: 额外的non_cross_penalty权重
       - 控制防交叉约束的强度
       - 默认值通常为0.0-10.0
    
    训练策略：
    ---------
    1. **初始化**: 创建QuantileMLP，输出维度=len(taus)
    2. **数据准备**: 
       - GPU-CACHED: 全部数据缓存到GPU（适用于数据量 < 20GB）
       - STREAMING: 通过DataLoader逐batch加载（适用于大数据集）
    3. **训练循环**: 
       - 每个epoch: 随机打乱 -> 分batch前向传播 -> 计算损失 -> 反向传播
       - 损失 = pinball_loss + λ * non_crossing_penalty
    4. **验证与早停**: 
       - 每个epoch结束，在验证集上计算总损失
       - 如果验证集损失连续patience个epoch不下降，提前终止
       - 返回验证集上表现最好的模型
    
    参数说明：
    ---------
    Xtr, ytr : np.ndarray
        训练集特征和标签
    Xva, yva : np.ndarray
        验证集特征和标签（用于Early Stopping）
    taus : list of float
        要预测的分位数列表，如[0.01, 0.05, 0.5, 0.95, 0.99]
        - 必须在(0, 1)之间
        - 建议按升序排列
    cfg : dict, optional
        超参数配置字典，支持的键：
        - hidden: list, 隐藏层配置，默认[512,512,256,128]
        - act: str, 激活函数，默认"relu"
        - dropout: float, Dropout概率，默认0.05
        - lr: float, 学习率，默认1e-3
        - l2: float, L2正则化系数，默认0.0
        - epochs: int, 最大训练轮数，默认40
        - patience: int, 早停patience，默认8
        - batch: int, batch大小，默认65536
        - non_cross_penalty: float, 防交叉惩罚权重，默认0.0
    device : str, default="auto"
        设备选择："cuda"、"cpu"或"auto"（自动选择）
    verbose : int, default=1
        日志详细程度
        - 0: 静默模式
        - 1: 每5个epoch打印一次验证损失
    gpu_cache_limit_bytes : int, default=20GB
        GPU缓存模式的显存阈值
        - 如果总数据大小 <= 此值，使用GPU-CACHED模式
        - 否则使用STREAMING模式
    
    返回：
    -----
    model : QuantileMLP
        训练好的分位数回归模型（已在验证集上取得最佳性能）
    mode : str
        实际使用的训练模式，"GPU-CACHED"或"STREAMING"
    
    使用示例：
    ---------
    >>> # 训练一个预测5%和95%分位数的模型（用于异常检测）
    >>> cfg = {"epochs": 50, "patience": 10, "non_cross_penalty": 1.0}
    >>> model, mode = fit_quantile_mlp(Xtr, ytr, Xva, yva, 
    ...                                 taus=[0.05, 0.95], 
    ...                                 cfg=cfg, device="cuda")
    >>> # 模型会输出2个值：pred[:, 0]=5%分位数, pred[:, 1]=95%分位数
    
    实际应用：
    ---------
    在风机数据清洗中，通常训练多个分位数：
    - taus=[0.01, 0.99]: 非常严格的边界（只有2%的数据在边界外）
    - taus=[0.05, 0.95]: 常用边界（只有10%的数据在边界外）
    - taus=[0.01, 0.05, 0.5, 0.95, 0.99]: 完整的分位数谱
    """
    sw = Stopwatch()  # 计时器
    dev = resolve_device(device)  # 解析设备
    use_cuda = (dev.type == "cuda")
    # 混合精度训练：GPU使用FP16加速，CPU使用FP32
    scaler_amp = torch.amp.GradScaler('cuda') if use_cuda else None
    autocast = torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()

    # 解析超参数配置
    hidden = cfg.get("hidden", [512,512,256,128]); act = cfg.get("act", "relu")
    dropout = cfg.get("dropout", 0.05); lr = cfg.get("lr", 1e-3); l2 = cfg.get("l2", 0.0)
    epochs = int(cfg.get("epochs", 40)); patience = int(cfg.get("patience", 8)); batch = int(cfg.get("batch", 65536))
    ncw = float(cfg.get("non_cross_penalty", 0.0))  # 防交叉惩罚权重

    # 创建分位数MLP模型
    in_dim = Xtr.shape[1]
    model = QuantileMLP(in_dim=in_dim, taus=taus, hidden=hidden, act=act, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    sw.lap("init model/opt")

    def to_tensor(X, y):
        """将numpy数组转换为PyTorch张量"""
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32).reshape(-1,1))
        return X, y

    Xtr_cpu, ytr_cpu = to_tensor(Xtr, ytr); Xva_cpu, yva_cpu = to_tensor(Xva, yva)
    sw.lap("prepare CPU tensors")

    # 根据数据大小决定训练模式
    total_bytes = (Xtr_cpu.element_size() * Xtr_cpu.nelement() + 
                   ytr_cpu.element_size() * ytr_cpu.nelement() +
                   Xva_cpu.element_size() * Xva_cpu.nelement() + 
                   yva_cpu.element_size() * yva_cpu.nelement())
    use_cached = use_cuda and (total_bytes <= gpu_cache_limit_bytes)
    mode = "GPU-CACHED" if use_cached else "STREAMING"
    print(f"[Quantile] bytes≈{total_bytes/1024**2:.1f}MiB; mode={mode}")

    # ========== 模式1: GPU-CACHED模式 ==========
    # 数据量较小，全部缓存到GPU显存，训练速度最快
    if use_cached:
        Xtr=Xtr_cpu.to(dev); ytr=ytr_cpu.to(dev)
        Xva=Xva_cpu.to(dev); yva=yva_cpu.to(dev)
        n=Xtr.shape[0]; steps=max(1,(n+batch-1)//batch)
        print(f"[Quantile] n_train={n}, steps/epoch={steps}, batch={batch}")
        best=float("inf"); bad=0
        for ep in range(epochs):
            model.train()
            # 每个epoch随机打乱训练数据
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch):
                idx = perm[i:i+batch]; xb=Xtr.index_select(0,idx); yb=ytr.index_select(0,idx)
                opt.zero_grad(set_to_none=True)
                with autocast:
                    pred = model(xb)
                    # 总损失 = Pinball损失 + 防交叉惩罚
                    loss = pinball_loss(pred.float(), yb, taus) + non_crossing_penalty(pred.float(), ncw)
                if scaler_amp:
                    scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
                else:
                    loss.backward(); opt.step()
            # 验证集评估
            model.eval()
            with torch.no_grad(), autocast:
                pv = model(Xva).float()
                vloss = pinball_loss(pv, yva, taus).item() + non_crossing_penalty(pv, ncw).item()
            if verbose and ((ep+1)%5==0 or ep==0):
                print(f"  [QMLP] epoch {ep+1:03d}, val_pinball={vloss:.6f}")
            # Early Stopping: 保存最佳模型
            if vloss < best - 1e-9:
                best=vloss; bad=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad+=1
                if bad>=patience: break
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        sw.total(f"fit_quantile_mlp ({mode})")
        model.eval(); return model, mode

    # ========== 模式2: STREAMING模式 ==========
    # 数据量较大，通过DataLoader逐batch从CPU加载到GPU
    tr_loader = DataLoader(TensorDataset(Xtr_cpu, ytr_cpu), batch_size=batch, shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva_cpu, yva_cpu), batch_size=batch, shuffle=False, drop_last=False)
    best=float("inf"); bad=0
    for ep in range(epochs):
        model.train()
        # 训练循环：逐batch从CPU加载到GPU
        for xb_cpu,yb_cpu in tr_loader:
            xb=xb_cpu.to(dev); yb=yb_cpu.to(dev)
            opt.zero_grad(set_to_none=True)
            with (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
                pred=model(xb); loss=pinball_loss(pred.float(), yb, taus) + non_crossing_penalty(pred.float(), ncw)
            if scaler_amp:
                scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
            else:
                loss.backward(); opt.step()
        # 验证集评估：逐batch计算总损失
        model.eval(); vsum=0.0; vcnt=0
        with torch.no_grad(), (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
            for xb_cpu,yb_cpu in va_loader:
                xb=xb_cpu.to(dev); yb=yb_cpu.to(dev)
                pred=model(xb)
                l = pinball_loss(pred.float(), yb, taus) + non_crossing_penalty(pred.float(), ncw)
                vsum += l.item()*len(xb); vcnt += len(xb)
        vloss=vsum/max(vcnt,1)
        if verbose and ((ep+1)%5==0 or ep==0):
            print(f"  [QMLP] epoch {ep+1:03d}, val_pinball={vloss:.6f}")
        # Early Stopping
        if vloss < best - 1e-9:
            best=vloss; bad=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    sw.total("fit_quantile_mlp (STREAMING)"); model.eval(); return model, "STREAMING"

@torch.no_grad()
def predict_quantiles(model, X):
    """
    使用训练好的分位数模型进行预测
    
    对输入数据X进行分位数预测，返回每个样本的多个分位数值。
    
    参数说明：
    ---------
    model : QuantileMLP
        训练好的分位数回归模型
    X : np.ndarray, shape (n_samples, n_features)
        输入特征矩阵，每行是一个样本
        - 如果模型输入维度为1: X只包含风速列
        - 如果模型输入维度为2: X包含[风速, 密度]两列
    
    返回：
    -----
    preds : np.ndarray, shape (n_samples, len(taus))
        预测的分位数矩阵
        - 每行对应一个样本
        - 每列对应一个分位数
        例如：如果model.taus=[0.05, 0.5, 0.95]，则返回shape为(n, 3)的矩阵
              preds[:, 0] = 所有样本的5%分位数
              preds[:, 1] = 所有样本的50%分位数（中位数）
              preds[:, 2] = 所有样本的95%分位数
    
    使用示例：
    ---------
    >>> # 假设模型预测5%和95%分位数
    >>> model, _ = fit_quantile_mlp(Xtr, ytr, Xva, yva, taus=[0.05, 0.95], ...)
    >>> preds = predict_quantiles(model, X_test)  # shape: (n_test, 2)
    >>> lower_bound = preds[:, 0]  # 5%分位数（下界）
    >>> upper_bound = preds[:, 1]  # 95%分位数（上界）
    >>> # 异常检测：如果y < lower_bound 或 y > upper_bound，则为异常
    >>> is_outlier = (y_test < lower_bound) | (y_test > upper_bound)
    
    实现细节：
    ---------
    1. 自动检测模型所在设备（CPU或GPU）
    2. 将输入数据转换为float32张量并移到对应设备
    3. 如果是CUDA，使用FP16混合精度推理加速
    4. 返回float32的numpy数组（CPU内存）
    """
    # 获取模型所在设备
    dev = next(model.parameters()).device
    use_cuda = (dev.type == "cuda")
    
    # 将输入转换为张量并移到模型所在设备
    X_t = torch.from_numpy(X.astype(np.float32)).to(dev)
    
    # 推理：如果是GPU，使用FP16加速
    with torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext():
        preds = model(X_t).float().cpu().numpy()
    
    return preds
