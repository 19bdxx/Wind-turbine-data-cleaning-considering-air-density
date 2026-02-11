# -*- coding: utf-8 -*-
"""
中心模型 (Center Model) - 风机功率预测核心模块

本模块实现了基于多层感知器(MLP)的风机功率预测模型，用于风机数据清洗的第二阶段。
这是整个数据清洗流程的核心预测引擎。

核心功能：
---------
1. **MLP神经网络**: 使用多层前馈神经网络学习 风速(v) 或 (风速v, 空气密度ρ) -> 功率(P) 的映射关系
2. **多种损失函数**: 支持MSE、WMSE(加权MSE)、Huber_z三种损失函数，适应不同的数据特点
3. **两种训练模式**: 
   - GPU-CACHED模式：数据较小时全部缓存到GPU显存，训练速度最快
   - STREAMING模式：数据较大时通过DataLoader流式加载，节省显存
4. **混合精度训练**: 使用AMP(Automatic Mixed Precision)加速训练，降低显存占用
5. **Early Stopping**: 防止过拟合，自动保存验证集上表现最好的模型

模型结构：
---------
输入层: [风速v] 或 [风速v, 空气密度ρ]  (1维或2维)
隐藏层: 多个全连接层，每层后接激活函数和Dropout
输出层: 单个神经元，输出预测功率值

训练策略：
---------
- 优化器: Adam (自适应学习率)
- 正则化: L2正则化 + Dropout
- 损失函数: 可选MSE/WMSE/Huber_z，根据数据分布特点选择
- 早停机制: 验证集损失不再下降时提前终止训练

使用场景：
---------
本模块在风机数据清洗的第二阶段使用：
1. 使用初步筛选后的"干净"数据训练中心模型
2. 用训练好的模型预测所有数据点的期望功率
3. 通过预测值与实际值的偏差识别异常数据点
"""
import math, contextlib
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..core.device import resolve_device
from ..core.utils import Stopwatch
from ..core.dmode import build_D_from_yhat

class MLP(nn.Module):
    """
    多层感知器 (Multi-Layer Perceptron) - 风机功率预测神经网络
    
    这是一个标准的全连接前馈神经网络，用于学习风机的功率曲线。
    网络结构为：输入层 -> 多个隐藏层(全连接+激活+Dropout) -> 输出层
    
    网络特点：
    ---------
    1. **可变深度**: 通过hidden参数控制隐藏层的数量和每层神经元数量
    2. **多种激活函数**: 支持ReLU、GELU、Tanh、SiLU，默认使用ReLU
    3. **Dropout正则化**: 每个隐藏层后可选Dropout，防止过拟合
    4. **单输出**: 输出层为单个神经元，直接预测功率值（回归任务）
    
    参数说明：
    ---------
    in_dim : int, default=2
        输入特征维度
        - 1: 仅使用风速 [v]
        - 2: 使用风速和空气密度 [v, ρ]
    hidden : list of int, default=[512,512,256,128]
        隐藏层配置，列表长度为隐藏层数量，每个元素为该层神经元数量
        默认配置：4个隐藏层，神经元数量逐渐减少（512->512->256->128）
        这种"漏斗型"结构有助于逐步提取抽象特征
    act : str, default="relu"
        激活函数类型，可选：
        - "relu": ReLU，最常用，计算快速，缓解梯度消失
        - "gelu": GELU，Transformer中常用，平滑版ReLU
        - "tanh": Tanh，输出范围(-1,1)，较少使用
        - "silu": SiLU/Swish，自门控激活函数
    dropout : float, default=0.05
        Dropout概率，取值范围[0,1]
        - 0或None: 不使用Dropout
        - 0.05: 每层随机丢弃5%的神经元（默认值，轻度正则化）
        - 0.1-0.3: 常用范围，根据数据量和过拟合程度调整
    
    网络结构示例：
    -------------
    输入: [v, ρ] (2维)
    ↓
    Linear(2 -> 512) -> ReLU -> Dropout(0.05)
    ↓
    Linear(512 -> 512) -> ReLU -> Dropout(0.05)
    ↓
    Linear(512 -> 256) -> ReLU -> Dropout(0.05)
    ↓
    Linear(256 -> 128) -> ReLU -> Dropout(0.05)
    ↓
    Linear(128 -> 1)  [输出功率值]
    
    使用示例：
    ---------
    >>> # 创建一个使用风速和密度的模型
    >>> model = MLP(in_dim=2, hidden=[256, 128, 64], act="relu", dropout=0.1)
    >>> 
    >>> # 创建一个仅使用风速的模型
    >>> model = MLP(in_dim=1, hidden=[128, 64], act="gelu", dropout=0.05)
    """
    def __init__(self, in_dim=2, hidden=[512,512,256,128], act="relu", dropout=0.05):
        super().__init__()
        
        # 激活函数字典：支持4种常用激活函数
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}
        
        # 构建网络层列表
        layers = []
        prev = in_dim  # 上一层的输出维度，初始为输入维度
        
        # 逐层构建隐藏层：全连接层 + 激活函数 + Dropout
        for h in hidden:
            # 添加全连接层：prev -> h
            layers += [nn.Linear(prev, h), acts.get(act, nn.ReLU())]
            
            # 如果指定了Dropout且大于0，则添加Dropout层
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            
            prev = h  # 更新为当前层的输出维度
        
        # 输出层：最后一个隐藏层 -> 1（功率预测值）
        # 注意：输出层不加激活函数，因为这是回归任务，需要输出连续值
        layers += [nn.Linear(prev, 1)]
        
        # 将所有层组合成一个顺序容器
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播：计算输入x对应的预测功率
        
        参数：
        -----
        x : torch.Tensor, shape=(batch_size, in_dim)
            输入特征，每行为一个样本
            - in_dim=1: x为风速 [v]
            - in_dim=2: x为 [v, ρ]
        
        返回：
        -----
        y_pred : torch.Tensor, shape=(batch_size, 1)
            预测的功率值
        """
        return self.net(x)

class LossBuilder:
    """
    损失函数构建器 - 支持多种损失函数用于风机功率预测模型训练
    
    本类封装了三种不同的损失函数，每种都有其独特的适用场景：
    
    1. **MSE (Mean Squared Error)** - 均方误差
       - 标准的回归损失函数
       - 对所有数据点一视同仁，不做加权
       - 适用场景：数据分布均匀，没有明显的异质性
       - 公式：L = mean((y_pred - y_true)²)
    
    2. **WMSE (Weighted Mean Squared Error)** - 加权均方误差
       - 根据预测值自适应调整权重，对不同区域给予不同重视程度
       - 权重D基于预测值动态计算，通常在功率曲线的某些区域（如额定功率附近）给予更大容忍度
       - 适用场景：希望模型在某些区域（如低功率区）更精确，而在其他区域（如额定功率区）允许更大误差
       - 公式：L = mean((y_pred - y_true)² / D²)
       - 特点：D越大的区域，该区域的误差对总损失贡献越小，模型可以"容忍"更大的误差
    
    3. **Huber_z (Standardized Huber Loss)** - 标准化Huber损失
       - 结合了MSE和MAE的优点：对小误差使用平方惩罚，对大误差使用线性惩罚
       - 先对误差进行标准化：z = (y_pred - y_true) / D
       - 然后应用Huber损失：对|z| < delta使用平方损失，对|z| >= delta使用线性损失
       - 适用场景：数据中存在一些离群点或异常值，希望降低它们对模型训练的影响
       - 优势：对异常值更鲁棒，不会因为少数极端值而扭曲模型
       - 公式：当|z| < δ时，L = 0.5*z²；当|z| >= δ时，L = δ*(|z| - 0.5*δ)
    
    参数说明：
    ---------
    kind : str, default="mse"
        损失函数类型，可选：
        - "mse": 标准均方误差
        - "wmse": 加权均方误差
        - "huber_z": 标准化Huber损失
    huber_delta_z : float, default=1.0
        Huber损失的阈值参数（仅在kind="huber_z"时使用）
        - 控制何时从平方损失切换到线性损失
        - 越小：对异常值越敏感，越早切换到线性损失
        - 越大：对异常值越不敏感，更接近MSE行为
        - 典型值：0.5-2.0之间
    d_mode : str, default="pred_or_both"
        权重D的计算模式（用于WMSE和Huber_z）
        - 影响如何根据预测值计算自适应权重D
        - 详见 build_D_from_yhat 函数说明
    eps_ratio : float, default=0.05
        权重计算中的epsilon比例参数
        - 用于数值稳定性，防止除零
    delta_power : float, default=50.0
        权重计算中的delta参数，控制权重在功率曲线不同区域的变化
        - 通常设置为接近额定功率的值
    prated_used : float or None
        额定功率值，用于权重计算
        - 如果为None，则不使用额定功率信息
    
    使用示例：
    ---------
    >>> # 创建标准MSE损失
    >>> loss_fn = LossBuilder(kind="mse")
    >>> loss = loss_fn(pred, y_true)
    >>> 
    >>> # 创建加权MSE损失，强调低功率区域的精度
    >>> loss_fn = LossBuilder(kind="wmse", d_mode="pred_or_both", prated_used=2000.0)
    >>> 
    >>> # 创建Huber损失，对异常值更鲁棒
    >>> loss_fn = LossBuilder(kind="huber_z", huber_delta_z=1.0, prated_used=2000.0)
    
    选择建议：
    ---------
    - 数据质量好，分布均匀 -> 使用 MSE
    - 希望模型在额定功率区域更宽容 -> 使用 WMSE
    - 数据中有离群点，需要鲁棒训练 -> 使用 Huber_z
    """
    def __init__(self, kind: str = "mse", huber_delta_z: float = 1.0,
                 d_mode: str = "pred_or_both", eps_ratio: float = 0.05,
                 delta_power: float = 50.0, prated_used: float | None = None):
        # 存储损失函数类型（转为小写，统一格式）
        self.kind = (kind or "mse").lower()
        
        # Huber损失的delta参数
        self.huber_delta_z = float(huber_delta_z)
        
        # 权重D的计算模式相关参数
        self.d_mode = d_mode
        self.eps_ratio = float(eps_ratio)
        self.delta_power = float(delta_power)
        
        # 额定功率（如果未提供，设为NaN）
        self.prated_used = (float(prated_used) if prated_used is not None else float("nan"))
    
    @torch.no_grad()
    def _make_D(self, pred: torch.Tensor) -> torch.Tensor:
        """
        计算自适应权重D
        
        根据预测值pred和配置参数，计算每个样本的权重D。
        D越大表示该区域允许更大的误差容忍度。
        
        参数：
        -----
        pred : torch.Tensor
            模型的预测值
        
        返回：
        -----
        D : torch.Tensor
            每个样本的权重值，shape与pred相同
        
        注意：
        -----
        - 使用@torch.no_grad()装饰器，因为权重计算不需要梯度
        - 权重D会被detach，不参与反向传播
        """
        return build_D_from_yhat(pred.detach(), self.prated_used, self.d_mode,
                                 self.eps_ratio, self.delta_power)
    
    def __call__(self, pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算损失值
        
        参数：
        -----
        pred : torch.Tensor, shape=(batch_size, 1) or (batch_size,)
            模型的预测值
        y_true : torch.Tensor, shape=(batch_size, 1) or (batch_size,)
            真实标签值
        
        返回：
        -----
        loss : torch.Tensor, scalar
            标量损失值，用于反向传播
        
        实现细节：
        ---------
        - MSE模式：直接计算均方误差
        - WMSE模式：先计算权重D，然后计算加权均方误差
        - Huber_z模式：先标准化误差z，然后应用Huber损失
        """
        # 模式1：标准均方误差
        if self.kind == "mse":
            return torch.mean((pred - y_true) ** 2)
        
        # 对于WMSE和Huber_z，需要先计算权重D
        D = self._make_D(pred)
        
        # 模式2：加权均方误差
        if self.kind == "wmse":
            # 除以D²实现加权：D越大，该样本的损失贡献越小
            return torch.mean(((pred - y_true) ** 2) / (D * D))
        
        # 模式3：标准化Huber损失
        if self.kind == "huber_z":
            # 先标准化误差：z = (pred - y_true) / D
            z = (pred - y_true) / D
            
            # 应用Huber损失（smooth_l1_loss是Huber损失的实现）
            # 目标是让z接近0，beta参数控制平方/线性损失的切换点
            return torch.nn.functional.smooth_l1_loss(
                z, torch.zeros_like(z), beta=self.huber_delta_z, reduction="mean"
            )
        
        # 默认情况：如果kind不是上述三种之一，回退到MSE
        return torch.mean((pred - y_true) ** 2)

def fit_mlp_center(Xv_tr, Xr_tr, y_tr, Xv_va, Xr_va, y_va, use_rho=True, mlp_cfg=None,
                   device="auto", verbose=1, gpu_cache_limit_bytes=20*1024**3,
                   thresholds_cfg=None, prated_used=None):
    """
    训练MLP中心模型 - 风机功率预测的核心训练函数
    
    本函数实现了完整的神经网络训练流程，包括：
    1. 自动选择训练模式（GPU-CACHED vs STREAMING）
    2. 混合精度训练（AMP）加速
    3. Early stopping防止过拟合
    4. 多种损失函数支持
    
    ====================================
    数据加载模式详解：为什么需要两种模式？
    ====================================
    
    **GPU-CACHED 模式** (数据全部缓存到GPU显存)
    优点：
    - 训练速度最快，无CPU-GPU数据传输开销
    - 每个batch直接从GPU显存中索引，延迟极低
    - 适合快速迭代实验
    
    缺点：
    - 需要足够的GPU显存（数据 + 模型 + 梯度 + 优化器状态）
    - 数据过大会导致显存溢出（OOM）
    
    使用条件：
    - 数据总大小 <= gpu_cache_limit_bytes（默认20GB）
    - 且使用CUDA设备
    
    **STREAMING 模式** (通过DataLoader流式加载)
    优点：
    - 显存占用小，只需容纳单个batch的数据
    - 可以处理任意大小的数据集
    - 稳定可靠，不会OOM
    
    缺点：
    - 每个batch需要CPU->GPU数据传输，增加延迟
    - 训练速度相对较慢（但对大数据集是必需的）
    
    使用条件：
    - 数据总大小 > gpu_cache_limit_bytes
    - 或使用CPU设备
    
    自动选择逻辑：
    - 计算训练集+验证集的总字节数
    - 如果 use_cuda=True 且 total_bytes <= gpu_cache_limit_bytes，选择GPU-CACHED
    - 否则选择STREAMING
    
    ====================================
    混合精度训练（AMP）详解
    ====================================
    
    混合精度训练使用float16进行前向传播和梯度计算，使用float32保存权重：
    
    优点：
    - 训练速度提升1.5-3倍（在Tensor Core GPU上）
    - 显存占用减少约50%，可以使用更大的batch size
    - 现代GPU（如V100、A100、RTX 30/40系列）对float16有硬件加速
    
    实现：
    - 使用torch.amp.autocast自动将前向传播转为float16
    - 使用GradScaler防止梯度下溢（gradient underflow）
    - 权重和优化器状态仍保持float32精度
    
    注意：
    - 只在CUDA设备上启用（CPU不支持float16加速）
    - 损失计算时转回float32，确保数值稳定性
    
    ====================================
    Early Stopping 机制详解
    ====================================
    
    Early stopping防止过拟合，提高模型泛化能力：
    
    工作原理：
    1. 每个epoch结束后，在验证集上评估模型
    2. 如果验证集损失下降，保存当前模型状态，重置patience计数器
    3. 如果验证集损失不下降，patience计数器+1
    4. 当patience计数器达到阈值（mlp_cfg["patience"]），停止训练
    5. 训练结束后，恢复验证集损失最低的模型状态
    
    为什么需要：
    - 神经网络容易过拟合，尤其是在训练数据有限时
    - 训练损失持续下降，但验证损失开始上升 -> 过拟合信号
    - 及时停止可以获得泛化能力最强的模型
    
    参数调整：
    - patience较小（如5）：更激进，训练时间短，可能欠拟合
    - patience较大（如20-50）：更保守，训练时间长，可能过拟合
    - 典型值：10-30，根据数据集大小和模型复杂度调整
    
    ====================================
    参数说明
    ====================================
    
    输入数据（训练集和验证集）：
    ---------------------------
    Xv_tr : np.ndarray, shape=(n_train,)
        训练集的风速数据（已标准化）
    Xr_tr : np.ndarray, shape=(n_train,) or None
        训练集的空气密度数据（已标准化）
        如果use_rho=False，可以为None
    y_tr : np.ndarray, shape=(n_train,)
        训练集的功率标签（目标值）
    Xv_va : np.ndarray, shape=(n_val,)
        验证集的风速数据（已标准化）
    Xr_va : np.ndarray, shape=(n_val,) or None
        验证集的空气密度数据（已标准化）
    y_va : np.ndarray, shape=(n_val,)
        验证集的功率标签
    
    模型配置：
    ---------
    use_rho : bool, default=True
        是否使用空气密度作为输入特征
        - True: 输入为 [风速, 密度]，2维
        - False: 输入仅为 [风速]，1维
    mlp_cfg : dict or None
        MLP模型的配置字典，包含以下键：
        - "hidden": list, 隐藏层配置，如[512,512,256,128]
        - "act": str, 激活函数，如"relu"
        - "dropout": float, Dropout概率，如0.05
        - "lr": float, 学习率，如1e-3
        - "l2": float, L2正则化系数，如1e-5
        - "batch": int, batch size，如256
        - "epochs": int, 最大训练轮数，如200
        - "patience": int, early stopping的patience，如20
        - "loss": str, 损失函数类型，如"mse"/"wmse"/"huber_z"
        - "huber_delta_z": float, Huber损失的delta参数（可选）
    device : str, default="auto"
        计算设备
        - "auto": 自动选择（优先GPU）
        - "cuda": 强制使用GPU
        - "cpu": 强制使用CPU
    verbose : int, default=1
        日志输出等级
        - 0: 静默模式，不输出训练日志
        - 1: 正常模式，每5个epoch输出一次
        - 2+: 详细模式，每个epoch都输出
    gpu_cache_limit_bytes : int, default=20GB
        GPU缓存模式的显存限制（字节）
        - 数据总大小小于此值时，使用GPU-CACHED模式
        - 否则使用STREAMING模式
        - 默认20GB，可根据GPU显存大小调整
    
    损失函数配置：
    -------------
    thresholds_cfg : dict or None
        阈值配置，用于损失函数的权重计算，包含：
        - "D_mode": str, 权重计算模式
        - "eps_ratio": float, epsilon比例
        - "delta_power": float, delta参数
    prated_used : float or None
        额定功率，用于损失函数的权重计算
    
    返回值：
    -------
    model : MLP
        训练好的模型（已加载验证集上最优的权重）
        模型处于eval模式，可直接用于预测
    mode : str
        使用的训练模式："GPU-CACHED" 或 "STREAMING"
    
    ====================================
    使用示例
    ====================================
    
    示例1：基本用法（自动选择模式）
    -------------------------------
    >>> mlp_cfg = {
    ...     "hidden": [512, 512, 256, 128],
    ...     "act": "relu",
    ...     "dropout": 0.05,
    ...     "lr": 1e-3,
    ...     "l2": 1e-5,
    ...     "batch": 256,
    ...     "epochs": 200,
    ...     "patience": 20,
    ...     "loss": "mse"
    ... }
    >>> model, mode = fit_mlp_center(
    ...     Xv_tr, Xr_tr, y_tr,
    ...     Xv_va, Xr_va, y_va,
    ...     use_rho=True,
    ...     mlp_cfg=mlp_cfg,
    ...     device="auto",
    ...     verbose=1
    ... )
    >>> print(f"训练完成，使用模式：{mode}")
    
    示例2：使用加权损失函数
    ----------------------
    >>> mlp_cfg["loss"] = "wmse"  # 使用加权MSE
    >>> thresholds_cfg = {
    ...     "D_mode": "pred_or_both",
    ...     "eps_ratio": 0.05,
    ...     "delta_power": 50.0
    ... }
    >>> model, mode = fit_mlp_center(
    ...     Xv_tr, Xr_tr, y_tr,
    ...     Xv_va, Xr_va, y_va,
    ...     mlp_cfg=mlp_cfg,
    ...     thresholds_cfg=thresholds_cfg,
    ...     prated_used=2000.0  # 额定功率2000kW
    ... )
    
    示例3：强制使用STREAMING模式（节省显存）
    ---------------------------------------
    >>> model, mode = fit_mlp_center(
    ...     Xv_tr, Xr_tr, y_tr,
    ...     Xv_va, Xr_va, y_va,
    ...     mlp_cfg=mlp_cfg,
    ...     gpu_cache_limit_bytes=0  # 设为0强制使用STREAMING
    ... )
    
    示例4：仅使用风速（不考虑空气密度）
    ----------------------------------
    >>> model, mode = fit_mlp_center(
    ...     Xv_tr, None, y_tr,  # Xr_tr=None
    ...     Xv_va, None, y_va,  # Xr_va=None
    ...     use_rho=False,      # 关闭密度特征
    ...     mlp_cfg=mlp_cfg
    ... )
    
    ====================================
    训练流程说明
    ====================================
    
    1. 初始化阶段：
       - 设置计算设备（CPU/GPU）
       - 初始化混合精度训练组件（GradScaler、autocast）
       - 创建MLP模型
       - 创建Adam优化器
       - 创建损失函数构建器
    
    2. 数据准备阶段：
       - 将NumPy数组转换为PyTorch张量（float32）
       - 计算数据总大小，决定使用GPU-CACHED还是STREAMING模式
       - 根据模式进行相应的数据加载准备
    
    3. 训练阶段（两种模式分别实现）：
       
       GPU-CACHED模式：
       - 将所有数据一次性加载到GPU
       - 每个epoch：随机打乱索引
       - 每个batch：通过索引直接从GPU显存采样
       - 前向传播 -> 计算损失 -> 反向传播 -> 更新权重
       - epoch结束：在验证集上评估，检查early stopping
       
       STREAMING模式：
       - 创建DataLoader，按batch流式加载
       - 每个epoch：遍历DataLoader
       - 每个batch：CPU->GPU传输 -> 前向传播 -> 反向传播
       - epoch结束：在验证集上评估，检查early stopping
    
    4. 结束阶段：
       - 恢复验证集损失最低的模型权重
       - 设置模型为eval模式
       - 返回模型和使用的训练模式
    
    ====================================
    性能调优建议
    ====================================
    
    1. Batch Size：
       - GPU-CACHED模式：可以使用较大batch（512-2048），充分利用GPU
       - STREAMING模式：受显存限制，通常256-512
       - 较大batch训练更稳定，但可能降低泛化能力
    
    2. 学习率：
       - 典型值：1e-3 到 1e-4
       - batch size越大，学习率可以略微提高
       - 如果损失震荡，降低学习率
    
    3. 网络深度：
       - 数据量大（>10万）：可以使用深网络[512,512,256,128]
       - 数据量小（<1万）：使用浅网络[256,128]避免过拟合
    
    4. 正则化：
       - L2系数：1e-5 到 1e-4
       - Dropout：0.0（无）到 0.2（强）
       - 数据量小时需要更强的正则化
    
    5. Early Stopping：
       - 数据量大：patience可以较大（30-50）
       - 数据量小：patience较小（10-20）
    """
    # ========================================
    # 阶段1：初始化 - 设备、混合精度、模型、优化器
    # ========================================
    
    sw = Stopwatch()  # 计时器，用于性能分析
    
    # 解析计算设备（auto -> cuda/cpu）
    dev = resolve_device(device)
    use_cuda = (dev.type == "cuda")
    
    # 混合精度训练组件（仅在CUDA设备上启用）
    # GradScaler: 梯度缩放器，防止float16下的梯度下溢
    scaler_amp = torch.amp.GradScaler('cuda') if use_cuda else None
    # autocast: 自动混合精度上下文管理器
    autocast = torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()
    
    # 确定输入维度：1维（仅风速）或 2维（风速+密度）
    in_dim = 2 if use_rho else 1
    
    # 创建MLP模型并移动到目标设备
    model = MLP(in_dim=in_dim, hidden=mlp_cfg["hidden"], act=mlp_cfg["act"], dropout=mlp_cfg["dropout"]).to(dev)
    
    # 创建Adam优化器
    # lr: 学习率，控制每次更新的步长
    # weight_decay: L2正则化系数，防止权重过大
    opt = torch.optim.Adam(model.parameters(), lr=mlp_cfg["lr"], weight_decay=mlp_cfg["l2"])
    
    # 解析损失函数配置
    loss_kind = (mlp_cfg.get("loss", "mse") if mlp_cfg else "mse")
    huber_delta_z = float(mlp_cfg.get("huber_delta_z", 1.0)) if mlp_cfg else 1.0
    d_mode      = (thresholds_cfg or {}).get("D_mode", "pred_or_both")
    eps_ratio   = float((thresholds_cfg or {}).get("eps_ratio", 0.05))
    delta_power = float((thresholds_cfg or {}).get("delta_power", 50.0))
    
    # 创建损失函数构建器
    loss_builder = LossBuilder(kind=loss_kind, huber_delta_z=huber_delta_z,
                               d_mode=d_mode, eps_ratio=eps_ratio,
                               delta_power=delta_power, prated_used=prated_used)
    sw.lap("init model/opt/loss")
    
    # ========================================
    # 阶段2：数据准备 - NumPy数组 -> PyTorch张量
    # ========================================
    
    def pack(v, r, y):
        """
        将NumPy数组打包为PyTorch张量
        
        参数：
        -----
        v : np.ndarray, 风速
        r : np.ndarray or None, 空气密度
        y : np.ndarray, 功率标签
        
        返回：
        -----
        X : torch.Tensor, shape=(n, in_dim), 输入特征
        y : torch.Tensor, shape=(n, 1), 标签
        """
        # 根据use_rho决定是否拼接密度特征
        X = np.c_[v, r].astype(np.float32) if use_rho else v.astype(np.float32).reshape(-1, 1)
        y = y.astype(np.float32).reshape(-1, 1)
        return torch.from_numpy(X), torch.from_numpy(y)
    
    # 打包训练集和验证集数据
    Xtr_cpu, Ytr_cpu = pack(Xv_tr, (Xr_tr if use_rho else None), y_tr)
    Xva_cpu, Yva_cpu = pack(Xv_va, (Xr_va if use_rho else None), y_va)
    sw.lap("prepare CPU tensors")
    
    # ========================================
    # 阶段3：决定训练模式 - GPU-CACHED vs STREAMING
    # ========================================
    
    # 计算训练集和验证集的总字节数
    total_bytes = (Xtr_cpu.element_size() * Xtr_cpu.nelement() +
                   Ytr_cpu.element_size() * Ytr_cpu.nelement() +
                   Xva_cpu.element_size() * Xva_cpu.nelement() +
                   Yva_cpu.element_size() * Yva_cpu.nelement())
    
    # 判断是否使用GPU缓存模式
    # 条件：(1) 使用CUDA设备 (2) 数据总大小不超过限制
    use_cached = use_cuda and (total_bytes <= gpu_cache_limit_bytes)
    mode = "GPU-CACHED" if use_cached else "STREAMING"
    
    print(f"[Data] bytes≈{total_bytes/1024**2:.1f}MiB; mode={mode}")
    
    # ========================================
    # 训练分支1：GPU-CACHED 模式
    # ========================================
    if use_cached:
        # 将所有数据一次性加载到GPU显存
        Xtr = Xtr_cpu.to(dev); Ytr = Ytr_cpu.to(dev)
        Xva = Xva_cpu.to(dev); Yva = Yva_cpu.to(dev)
        
        # 计算训练样本数和每个epoch的步数
        n = Xtr.shape[0]
        steps = max(1, (n + mlp_cfg["batch"] - 1) // mlp_cfg["batch"])
        print(f"[Data] n_train={n}, steps/epoch={steps}, batch={mlp_cfg['batch']}")
        
        # Early stopping 变量
        best = float("inf")  # 最佳验证集损失
        bad = 0              # 连续未改善的epoch数
        
        # 训练循环
        for ep in range(mlp_cfg["epochs"]):
            # 训练阶段
            model.train()
            
            # 在GPU上生成随机排列（用于batch采样）
            perm = torch.randperm(n, device=dev)
            
            # 遍历所有batch
            for i in range(0, n, mlp_cfg["batch"]):
                # 获取当前batch的索引
                idx = perm[i:i + mlp_cfg["batch"]]
                # 通过索引从GPU显存中采样（非常快，无CPU-GPU传输）
                xb = Xtr.index_select(0, idx)
                yb = Ytr.index_select(0, idx)
                
                # 清空梯度（set_to_none=True更高效）
                opt.zero_grad(set_to_none=True)
                
                # 前向传播（使用混合精度）
                with autocast:
                    pred = model(xb)  # shape: (batch_size, 1)
                    loss = loss_builder(pred.float(), yb)  # 损失计算用float32
                
                # 反向传播和参数更新
                if scaler_amp:
                    # 使用GradScaler：缩放损失 -> 反向传播 -> 缩放梯度 -> 更新参数 -> 更新scale
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(opt)
                    scaler_amp.update()
                else:
                    # 标准的反向传播和参数更新
                    loss.backward()
                    opt.step()
            
            # 验证阶段（每个epoch结束后）
            model.eval()
            with torch.no_grad(), autocast:
                # 在完整验证集上计算损失
                vloss = loss_builder(model(Xva).float(), Yva).item()
            
            # 打印日志（根据verbose设置）
            if verbose and ((ep + 1) % 5 == 0 or ep == 0):
                print(f"  [MLP] epoch {ep+1:03d}, val_loss={vloss:.6f} ({loss_kind})")
            
            # Early stopping 检查
            if vloss + 1e-9 < best:  # 加1e-9避免浮点误差
                # 验证集损失下降，更新最佳状态
                best = vloss
                bad = 0
                # 保存当前模型权重到CPU（避免占用GPU显存）
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                # 验证集损失未下降
                bad += 1
                if bad >= mlp_cfg["patience"]:
                    # 连续patience个epoch未改善，提前停止
                    break
        
        # 恢复最佳模型权重
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        
        sw.total(f"fit_mlp_center ({mode})")
        model.eval()
        return model, mode
    
    # ========================================
    # 训练分支2：STREAMING 模式
    # ========================================
    
    # 创建DataLoader：自动处理batch采样和shuffle
    tr_loader = DataLoader(TensorDataset(Xtr_cpu, Ytr_cpu), batch_size=mlp_cfg["batch"], shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva_cpu, Yva_cpu), batch_size=mlp_cfg["batch"], shuffle=False, drop_last=False)
    
    # Early stopping 变量
    best = float("inf")
    bad = 0
    
    # 训练循环
    for ep in range(mlp_cfg["epochs"]):
        # 训练阶段
        model.train()
        
        # 遍历训练集的所有batch
        for xb_cpu, yb_cpu in tr_loader:
            # 将batch从CPU传输到GPU（如果使用CUDA）
            xb = xb_cpu.to(dev)
            yb = yb_cpu.to(dev)
            
            # 清空梯度
            opt.zero_grad(set_to_none=True)
            
            # 前向传播（使用混合精度）
            with (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
                pred = model(xb)
                loss = loss_builder(pred.float(), yb)
            
            # 反向传播和参数更新
            if scaler_amp:
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                loss.backward()
                opt.step()
        
        # 验证阶段（每个epoch结束后）
        model.eval()
        vsum = 0.0  # 累计损失
        vcnt = 0    # 累计样本数
        
        with torch.no_grad(), (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
            # 遍历验证集的所有batch
            for xb_cpu, yb_cpu in va_loader:
                xb = xb_cpu.to(dev)
                yb = yb_cpu.to(dev)
                
                pred = model(xb)
                loss = loss_builder(pred.float(), yb)
                
                # 累加损失（加权平均，考虑batch大小）
                vsum += loss.item() * len(xb)
                vcnt += len(xb)
        
        # 计算平均验证集损失
        vloss = vsum / max(vcnt, 1)
        
        # 打印日志
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"  [MLP] epoch {ep+1:03d}, val_loss={vloss:.6f} ({loss_kind})")
        
        # Early stopping 检查
        if vloss + 1e-9 < best:
            best = vloss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= mlp_cfg["patience"]:
                break
    
    # 恢复最佳模型权重
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    
    sw.total("fit_mlp_center (STREAMING)")
    model.eval()
    return model, "STREAMING"

def predict_mlp_center(model, wind_std_vec, rho_std_vec, prated, use_rho=True, clip_to_prated=True):
    """
    使用训练好的MLP模型进行功率预测
    
    这是模型的推理（inference）函数，用于对新数据进行功率预测。
    函数自动处理数据格式转换、设备迁移、混合精度推理等细节。
    
    功能特点：
    ---------
    1. **自动设备检测**: 自动识别模型所在设备（CPU/GPU），将输入数据迁移到相应设备
    2. **混合精度推理**: 在GPU上自动使用float16加速推理（训练时的权重是float32）
    3. **批量预测**: 一次性处理整个数组，充分利用GPU并行能力
    4. **额定功率裁剪**: 可选将预测值裁剪到[0, Prated]区间，符合物理约束
    
    参数说明：
    ---------
    model : MLP
        训练好的MLP模型
        - 必须处于eval模式（fit_mlp_center返回的模型已自动设置）
        - 模型的权重可以在CPU或GPU上
    wind_std_vec : np.ndarray, shape=(n,)
        风速向量（已标准化）
        - 必须使用与训练时相同的标准化参数
        - 通常是 (v - mean) / std 的结果
    rho_std_vec : np.ndarray, shape=(n,) or None
        空气密度向量（已标准化）
        - 如果use_rho=False，可以为None或任意值（不会被使用）
        - 必须使用与训练时相同的标准化参数
    prated : float
        风机的额定功率（单位：kW）
        - 用于裁剪预测值（如果clip_to_prated=True）
        - 如果为非有限值（inf/nan），裁剪功能将被禁用
    use_rho : bool, default=True
        是否使用空气密度作为输入特征
        - 必须与训练时保持一致！
        - True: 输入 [风速, 密度]
        - False: 输入 [风速]
    clip_to_prated : bool, default=True
        是否将预测值裁剪到物理合理范围[0, Prated]
        - True: 预测值被裁剪到[0, prated]，确保物理合理性
        - False: 保留原始预测值（可能<0或>prated）
        - 推荐设为True，因为功率不可能为负或超过额定功率
    
    返回值：
    -------
    y_pred : np.ndarray, shape=(n,)
        预测的功率值（单位：kW）
        - 如果clip_to_prated=True且prated有限，则 0 <= y_pred <= prated
        - 返回的是NumPy数组（CPU），可直接用于后续处理
    
    使用示例：
    ---------
    
    示例1：基本预测（使用风速和密度）
    --------------------------------
    >>> # 假设已有训练好的模型和标准化后的数据
    >>> y_pred = predict_mlp_center(
    ...     model=model,
    ...     wind_std_vec=wind_normalized,     # 标准化后的风速
    ...     rho_std_vec=rho_normalized,       # 标准化后的密度
    ...     prated=2000.0,                    # 额定功率2000kW
    ...     use_rho=True,                     # 使用密度特征
    ...     clip_to_prated=True               # 裁剪到[0, 2000]
    ... )
    >>> print(f"预测功率范围: [{y_pred.min():.2f}, {y_pred.max():.2f}] kW")
    
    示例2：仅使用风速预测
    ---------------------
    >>> y_pred = predict_mlp_center(
    ...     model=model,
    ...     wind_std_vec=wind_normalized,
    ...     rho_std_vec=None,                 # 不需要密度
    ...     prated=2000.0,
    ...     use_rho=False,                    # 不使用密度特征
    ...     clip_to_prated=True
    ... )
    
    示例3：不裁剪预测值（用于分析模型行为）
    --------------------------------------
    >>> y_pred_raw = predict_mlp_center(
    ...     model=model,
    ...     wind_std_vec=wind_normalized,
    ...     rho_std_vec=rho_normalized,
    ...     prated=2000.0,
    ...     use_rho=True,
    ...     clip_to_prated=False              # 保留原始预测值
    ... )
    >>> # 可以分析预测值是否超出合理范围
    >>> over_rated = np.sum(y_pred_raw > 2000.0)
    >>> print(f"有{over_rated}个点预测值超过额定功率")
    
    示例4：完整的预测流程（包括标准化）
    ----------------------------------
    >>> # 1. 准备原始数据
    >>> wind_raw = df['WindSpeed'].values
    >>> rho_raw = df['AirDensity'].values
    >>> 
    >>> # 2. 使用训练时保存的标准化参数
    >>> wind_std = (wind_raw - wind_mean) / wind_std
    >>> rho_std = (rho_raw - rho_mean) / rho_std
    >>> 
    >>> # 3. 进行预测
    >>> power_pred = predict_mlp_center(
    ...     model=model,
    ...     wind_std_vec=wind_std,
    ...     rho_std_vec=rho_std,
    ...     prated=turbine_rated_power,
    ...     use_rho=True,
    ...     clip_to_prated=True
    ... )
    >>> 
    >>> # 4. 计算预测误差（用于异常检测）
    >>> power_actual = df['Power'].values
    >>> residual = power_actual - power_pred
    >>> outliers = np.abs(residual) > threshold
    
    性能说明：
    ---------
    - 推理速度：在GPU上处理10万条数据通常只需几十毫秒
    - 内存占用：取决于数据量，float32数组占用 n*4 字节
    - 批处理：函数自动批量处理所有数据，无需手动分批
    
    注意事项：
    ---------
    1. **标准化一致性**: 预测时必须使用与训练时相同的标准化参数（均值和标准差）
       - 错误的标准化会导致预测结果完全错误
       - 建议在训练时保存标准化参数，预测时重新使用
    
    2. **use_rho参数**: 必须与训练时保持一致
       - 如果训练时use_rho=True，预测时也必须为True
       - 不一致会导致输入维度错误，引发运行时错误
    
    3. **设备兼容性**: 函数自动处理CPU/GPU切换
       - 输入数据在CPU上（NumPy数组）
       - 自动迁移到模型所在设备进行推理
       - 结果自动转回CPU（NumPy数组）
    
    4. **混合精度**: GPU推理自动使用float16加速
       - 模型权重仍是float32（训练时的精度）
       - 中间计算使用float16（更快，显存更小）
       - 输出结果转回float32（保证精度）
    
    5. **额定功率裁剪**: 
       - 物理上，功率应该在[0, Prated]范围内
       - 但模型可能预测出超出范围的值（尤其是外插时）
       - 裁剪可以确保预测值符合物理约束
       - 如果prated为inf或nan，裁剪将不起作用
    
    实现细节：
    ---------
    1. 从模型参数中推断设备位置（CPU/GPU）
    2. 根据use_rho决定输入特征的拼接方式
    3. 将NumPy数组转换为PyTorch张量（float32）
    4. 将张量迁移到模型所在设备
    5. 在no_grad()模式下进行推理（不计算梯度，节省内存）
    6. 如果在GPU上，使用autocast启用混合精度
    7. 将预测结果转回float32并移动到CPU
    8. 转换为NumPy数组
    9. 如果需要，裁剪到[0, prated]区间
    10. 返回1维NumPy数组
    """
    # ========================================
    # 步骤1：推断模型所在设备
    # ========================================
    # 通过查看模型参数的device属性来确定模型在CPU还是GPU上
    dev = next(model.parameters()).device
    use_cuda = (dev.type == "cuda")
    
    # ========================================
    # 步骤2：准备输入数据
    # ========================================
    # 根据use_rho决定是否拼接密度特征
    if use_rho:
        # 拼接风速和密度：X.shape = (n, 2)
        X = np.c_[wind_std_vec, rho_std_vec].astype(np.float32)
    else:
        # 仅使用风速：X.shape = (n, 1)
        X = wind_std_vec.astype(np.float32).reshape(-1, 1)
    
    # ========================================
    # 步骤3：推理（前向传播）
    # ========================================
    # 使用torch.no_grad()：
    # - 禁用梯度计算，节省内存
    # - 加快推理速度
    # - 防止梯度累积导致内存泄漏
    with torch.no_grad(), (torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else contextlib.nullcontext()):
        # 将NumPy数组转换为PyTorch张量并迁移到模型所在设备
        X_tensor = torch.from_numpy(X).to(dev)
        
        # 前向传播：得到预测值（可能是float16）
        y_tensor = model(X_tensor)
        
        # 转回float32（确保精度）并移动到CPU
        y_tensor = y_tensor.float().cpu()
        
        # 转换为NumPy数组并展平为1维
        y = y_tensor.numpy().reshape(-1)
    
    # ========================================
    # 步骤4：裁剪到物理合理范围（可选）
    # ========================================
    if clip_to_prated and math.isfinite(prated):
        # 将预测值裁剪到 [0, prated] 区间
        # - 小于0的值设为0（功率不能为负）
        # - 大于prated的值设为prated（功率不能超过额定功率）
        y = np.clip(y, 0.0, prated)
    
    return y
