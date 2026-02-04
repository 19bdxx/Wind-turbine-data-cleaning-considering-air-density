# -*- coding: utf-8 -*-
"""
D 尺度（Scale）计算模块 - 异常检测中的残差标准化因子

本模块定义了用于计算 D 尺度的函数，D 是异常检测中用于标准化残差的重要参数。
D 尺度的物理意义：代表在不同功率区间下，残差的"典型幅度"或"期望尺度"。

--------------------------------------
为什么需要 D 尺度？
--------------------------------------
在风机功率曲线异常检测中，如果直接使用绝对残差 |y - ŷ| 作为异常指标，会遇到以下问题：

1. 低功率区问题：
   - 在低风速、低功率区（如 0-100 kW），即使模型很精确，残差也可能较小（如 5-10 kW）
   - 如果用统一阈值（如 50 kW），会漏检许多低功率区的异常

2. 高功率区问题：
   - 在高功率区（如 1500-2000 kW），由于湍流、环境变化等因素，残差波动较大（如 50-100 kW）
   - 如果用统一阈值（如 20 kW），会产生大量误报

3. 标准化残差：
   - 通过引入 D 尺度，将残差标准化为 z = (y - ŷ) / D
   - 标准化残差 z 在不同功率区间具有可比性
   - 可以用统一的 z 阈值（如 z > 3）进行异常判定

--------------------------------------
D 尺度的计算模式（d_mode）
--------------------------------------
本模块支持四种 d_mode，用于在不同场景下计算 D 尺度：

1. "pred_only" - 仅使用预测值
   D = max(ŷ, 1.0)
   - 最简单的模式，D 完全跟随预测功率
   - 适用于模型预测较准确的场景
   - 保证 D ≥ 1.0 避免除零错误

2. "pred_or_epsPr" - 预测值或 ε×P_rated
   D = max(ŷ, ε × P_rated)
   - 引入额定功率的比例下界（通常 ε = 0.01 ~ 0.05，即 1%~5% 的额定功率）
   - 防止在极低功率区 D 过小导致过度敏感
   - 适用于需要在全功率范围保持稳定灵敏度的场景

3. "pred_or_delta" - 预测值或 δ_power
   D = max(ŷ, δ_power)
   - 引入绝对功率下界 δ_power（如 10 kW 或 20 kW）
   - 防止 D 过小导致误报
   - 适用于不依赖额定功率、使用固定功率阈值的场景

4. "pred_or_both" - 三者取最大（默认推荐）
   D = max(ŷ, ε × P_rated, δ_power)
   - 综合考虑预测值、额定功率比例、绝对功率阈值
   - 最稳健的模式，兼顾低、中、高功率区的异常检测性能
   - 推荐用于生产环境

--------------------------------------
函数说明
--------------------------------------
本模块提供两个函数：
- build_D_np: NumPy 版本，用于数据预处理和离线分析
- build_D_from_yhat: PyTorch 版本，用于模型训练时的 loss 计算（带梯度禁用）

依赖：
    - numpy: 用于 NumPy 数组计算
    - torch: 用于 PyTorch 张量计算
    - math: 用于数值判断

作者：项目团队
"""

import math
import numpy as np
import torch


def build_D_np(y_hat: np.ndarray, pr_used: float, d_mode: str, eps_ratio: float, delta_power: float):
    """
    构建 D 尺度 - NumPy 版本
    
    功能：
        根据预测功率和 d_mode 参数，计算用于标准化残差的 D 尺度。
        该函数用于数据预处理和离线分析阶段。
    
    参数：
        y_hat (np.ndarray): 模型预测的功率值，形状为 (n_samples,)
        pr_used (float): 使用的额定功率 P_rated（单位：kW）
        d_mode (str): D 尺度计算模式，可选值：
            - "pred_only": D = max(ŷ, 1.0)
            - "pred_or_epsPr": D = max(ŷ, ε × P_rated)
            - "pred_or_delta": D = max(ŷ, δ_power)
            - "pred_or_both": D = max(ŷ, ε × P_rated, δ_power)
        eps_ratio (float): 额定功率比例因子 ε（通常为 0.01 ~ 0.05）
        delta_power (float): 绝对功率下界 δ_power（单位：kW）
    
    返回：
        D (np.ndarray): 计算得到的 D 尺度，形状与 y_hat 相同
    
    实现细节：
        - 使用 np.fmax 而非 np.maximum：fmax 会忽略 NaN 值
        - 保证 D ≥ 1.0，避免在计算 z = (y - ŷ) / D 时出现除零错误
        - 不同 d_mode 通过条件分支实现不同的计算逻辑
    
    使用示例：
        # 模式1：仅使用预测值
        D = build_D_np(y_hat, pr_used=2000, d_mode="pred_only", 
                       eps_ratio=0.03, delta_power=10.0)
        
        # 模式4：综合考虑三者（推荐）
        D = build_D_np(y_hat, pr_used=2000, d_mode="pred_or_both", 
                       eps_ratio=0.03, delta_power=10.0)
        # 结果：D = max(y_hat, 60, 10) = max(y_hat, 60)
    """
    if d_mode == "pred_only":
        # 模式1：仅使用预测值，最小值为 1.0
        D = np.fmax(y_hat, 1.0)
    elif d_mode == "pred_or_epsPr":
        # 模式2：预测值与 ε × P_rated 取最大值
        D = np.fmax(y_hat, eps_ratio*pr_used)
    elif d_mode == "pred_or_delta":
        # 模式3：预测值与绝对功率下界 δ_power 取最大值
        D = np.fmax(y_hat, delta_power)
    else:
        # 模式4（默认）：三者取最大值
        # 先比较 y_hat 和 ε × P_rated，再与 δ_power 比较
        D = np.fmax(np.fmax(y_hat, eps_ratio*pr_used), delta_power)
    return D


@torch.no_grad()
def build_D_from_yhat(y_hat: torch.Tensor, prated_used: float,
                      d_mode: str, eps_ratio: float, delta_power: float) -> torch.Tensor:
    """
    构建 D 尺度 - PyTorch 版本（用于训练时的 loss 计算）
    
    功能：
        根据预测功率和 d_mode 参数，计算用于标准化残差的 D 尺度。
        该函数用于模型训练时的 loss 计算，使用 @torch.no_grad() 装饰器禁用梯度计算。
    
    参数：
        y_hat (torch.Tensor): 模型预测的功率值，形状为 (batch_size,) 或 (batch_size, 1)
        prated_used (float): 使用的额定功率 P_rated（单位：kW）
                             如果为 NaN 或 Inf，则忽略额定功率相关的计算
        d_mode (str): D 尺度计算模式，可选值：
            - "pred_only": D = max(ŷ, 1.0)
            - "pred_or_epsPr": D = max(ŷ, ε × P_rated)
            - "pred_or_delta": D = max(ŷ, δ_power)
            - "pred_or_both": D = max(ŷ, ε × P_rated, δ_power)
        eps_ratio (float): 额定功率比例因子 ε（通常为 0.01 ~ 0.05）
        delta_power (float): 绝对功率下界 δ_power（单位：kW）
    
    返回：
        D (torch.Tensor): 计算得到的 D 尺度，形状与 y_hat 相同
    
    实现细节：
        - 使用 @torch.no_grad() 装饰器：D 尺度不参与梯度反向传播
        - 使用 torch.clamp(y_hat, min=1.0) 初始化 D，保证 D ≥ 1.0
        - 使用 torch.maximum 进行逐元素最大值计算（PyTorch 1.7+）
        - 使用 torch.full_like 创建与 y_hat 形状相同的常量张量
        - 检查 prated_used 是否为有限值（math.isfinite），避免 NaN/Inf 导致错误
    
    与 NumPy 版本的区别：
        1. 返回类型为 torch.Tensor
        2. 初始化时直接用 torch.clamp 保证最小值为 1.0
        3. 通过多次 torch.maximum 累积计算，而非嵌套调用
        4. 检查 prated_used 的有效性（防止 NaN/Inf）
    
    使用示例：
        # 在训练循环中
        y_pred = model(X_batch)  # 形状: (32,)
        D = build_D_from_yhat(y_pred, prated_used=2000.0, 
                              d_mode="pred_or_both", 
                              eps_ratio=0.03, delta_power=10.0)
        # D 形状: (32,)，每个样本的 D = max(y_pred[i], 60, 10)
        
        # 计算标准化残差用于 loss
        residual = y_batch - y_pred
        z = residual / D  # 标准化残差
        loss = robust_loss_function(z)  # 使用 z 计算 loss
    """
    # 初始化 D：至少为 1.0（防止除零）
    D = torch.clamp(y_hat, min=1.0)
    
    # 根据 d_mode 逐步更新 D
    if d_mode in ("pred_or_epsPr", "pred_or_both"):
        # 如果需要考虑额定功率比例 ε × P_rated
        if math.isfinite(prated_used):
            # 只有当 prated_used 是有限值时才计算
            # 创建形状与 D 相同、值为 ε × P_rated 的张量
            eps_pr_tensor = torch.full_like(D, float(eps_ratio * prated_used))
            # 更新 D：取 D 和 ε × P_rated 的较大值
            D = torch.maximum(D, eps_pr_tensor)
    
    if d_mode in ("pred_or_delta", "pred_or_both"):
        # 如果需要考虑绝对功率下界 δ_power
        # 创建形状与 D 相同、值为 δ_power 的张量
        delta_tensor = torch.full_like(D, float(delta_power))
        # 更新 D：取 D 和 δ_power 的较大值
        D = torch.maximum(D, delta_tensor)
    
    return D
