# -*- coding: utf-8 -*-
"""
标准化残差分位数回归阈值方法

本模块实现了基于标准化残差的分位数回归阈值计算方法（QuantileZResid）。

核心思想：
    不直接对功率 P 建模阈值，而是对标准化残差 z = (P - P̂) / D 建模，
    其中 D 是期望离差（标准差代理）。通过标准化，可以缓解异方差问题，
    使得不同风速区间的波动具有可比性。

方法流程：
    1. 计算标准化残差: z = (y - ŷ) / D
    2. 使用 MLP 拟合分位数曲面: Q_z(v, ρ) = f(v, ρ; τ)
    3. 共形预测校准: 在验证集上计算偏移量 c_plus 和 c_minus
    4. 还原到功率域: thr_pos = (Q_hi + c_plus) * D, thr_neg = (c_minus - Q_lo) * D

与 QuantilePower 方法的区别：
    - QuantilePower: 直接拟合 P 的分位数，Q_P(v, ρ)
      优点: 直观，阈值直接在功率域
      缺点: 高风速区间波动大，低风速区间波动小，异方差显著
      
    - QuantileZResid: 拟合标准化残差 z 的分位数，Q_z(v, ρ)
      优点: 标准化后各区间波动相当，模型更稳定，泛化性更好
      缺点: 需要额外的标准差估计 D
      推荐: 当功率波动随风速变化显著时使用（通常情况）

共形预测校准的作用：
    即使分位数模型拟合良好，也可能存在系统性偏差（过于乐观或保守）。
    共形预测通过验证集的超出量（exceed）统计，计算校准偏移 c，
    确保阈值在新数据上仍满足预期覆盖率（如 90%）。
    
    具体步骤:
        1. 在验证集上计算超出量: exceed_hi = max(0, z_val - Q_hi_val)
        2. 计算校准常数: c_plus = quantile(exceed_hi, tau_hi)
        3. 调整阈值: Q_hi_adjusted = Q_hi + c_plus
        
    优点: 提供分布自由的覆盖保证，无需假设数据分布

参考文献：
    Conformal Prediction: 
    Vovk et al., "Algorithmic Learning in a Random World" (2005)
"""
import numpy as np
import torch  # 用于规范化 device（CPU 或 GPU）
from .base import ThresholdMethod, ThresholdOutputs
from ..models.quantile import fit_quantile_mlp, predict_quantiles

class QuantileZResid(ThresholdMethod):
    """
    标准化残差分位数回归阈值方法
    
    通过对标准化残差 z = (P - P̂) / D 进行分位数回归，
    学习残差的上下分位数曲面，然后还原到功率域作为阈值。
    
    相比直接对功率 P 建模，标准化残差方法能更好地处理异方差问题，
    使得模型在不同风速区间都有较好的性能。
    """
    name = "quantile_zresid"

    def compute(self, *, train_X, train_zp, train_zn, query_X, D_all,
                idx_train_mask, idx_val_mask, taus, cfg, device=None):
        """
        计算基于标准化残差分位数回归的阈值
        
        主要步骤：
        1. 设备规范化（CPU/GPU）
        2. 准备特征矩阵 X（风速 + 可选的空气密度）
        3. 计算标准化残差 z = (y - ŷ) / D
        4. 在训练集上拟合分位数 MLP，验证集上评估
        5. 共形预测校准：计算验证集超出量的分位数作为偏移
        6. 还原到功率域得到最终阈值
        7. 标记异常点
        
        参数说明见 ThresholdMethod.compute 的文档
        
        返回:
            ThresholdOutputs: 包含正负阈值和异常标记的结果对象
        """

        # ========== 1. 设备规范化 ==========
        # 统一 device 参数格式为 torch.device 对象
        if device is None:
            # 默认：如果有 GPU 则使用 GPU，否则使用 CPU
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            # 字符串形式（如 "cpu" 或 "cuda:0"）转为 torch.device
            dev = torch.device(device)
        else:
            # 已经是 torch.device 对象，直接使用
            dev = device

        # ========== 2. 解析配置参数 ==========
        # 提取分位数水平：tau_lo 为下分位（如 0.05），tau_hi 为上分位（如 0.95）
        tau_lo, tau_hi = float(taus[0]), float(taus[1])
        
        # 获取分位数回归相关配置
        qcfg = (cfg.get("quantile") or {})
        
        # 是否使用空气密度 ρ 作为特征（默认 True）
        use_rho = bool(qcfg.get("use_rho", True))
        
        # 获取 MLP 模型配置（隐藏层、学习率等）
        qmlp_cfg = qcfg.get("mlp", {})

        # ========== 3. 准备特征矩阵 X_all ==========
        if use_rho:
        # 构造用于拟合标准化残差分位曲面的特征
        # 优先使用 "_for_model"（建模用标准化数据），其次使用 "_for_clean"（清洗用数据）
            # 使用风速 + 空气密度作为特征（二维输入）
            if ("wind_std_for_model" in cfg) and ("rho_std_for_model" in cfg) and (cfg["rho_std_for_model"] is not None):
                # 优先：建模阶段的标准化风速和密度
                X_all = np.c_[cfg["wind_std_for_model"], cfg["rho_std_for_model"]]
            elif ("wind_std_for_clean" in cfg) and ("rho_std_for_clean" in cfg):
                # 备选：清洗阶段的标准化风速和密度
                X_all = np.c_[cfg["wind_std_for_clean"], cfg["rho_std_for_clean"]]
            else:
                # 兜底：使用传入的 query_X
                X_all = query_X
        else:
            # 仅使用风速作为特征（一维输入）
            if "wind_std_for_model" in cfg and cfg["wind_std_for_model"] is not None:
                # 优先：建模阶段的标准化风速
                X_all = cfg["wind_std_for_model"].reshape(-1, 1)
            else:
                # 备选：清洗阶段的标准化风速
                X_all = cfg["wind_std_for_clean"].reshape(-1, 1)

        # ========== 4. 计算标准化残差 ==========
        # 获取真实功率和预测功率
        y_all = cfg["y"]          # 真实功率 (n_samples,)
        y_hat_all = cfg["y_hat"]  # 预测功率 (n_samples,)
        D_all_local = D_all       # 期望离差（标准差代理）(n_samples,)
        
        # 计算标准化残差: z = (y - ŷ) / D
        # 使用 maximum 避免除以零或极小值，确保数值稳定
        z_all = (y_all - y_hat_all) / np.maximum(D_all_local, 1e-12)
        
        # ========== 5. 准备训练集和验证集索引 ==========
        tr = idx_train_mask  # 训练集掩码（布尔数组）
        va = idx_val_mask    # 验证集掩码（布尔数组）

        # ========== 6. 拟合标准化残差的分位数曲面 ==========
        # 使用 MLP 拟合 z 的上下分位数：Q_lo(X) 和 Q_hi(X)
        taus_list = [tau_lo, tau_hi]  # 如 [0.05, 0.95]
        
        # 在训练集上拟合，验证集上评估
        model, _ = fit_quantile_mlp(
            X_all[tr], z_all[tr],    # 训练数据
            X_all[va], z_all[va],    # 验证数据
            taus=taus_list,          # 分位数水平
            cfg=qmlp_cfg,            # MLP 配置（隐藏层等）
            device=dev,              # 训练设备
            verbose=1                # 打印训练信息
        )
        
        # 对所有样本进行预测，得到分位数曲面
        q_pred_all = predict_quantiles(model, X_all)  # (n_samples, 2)
        q_lo_z = q_pred_all[:, 0]  # 下分位数 Q_lo(X)
        q_hi_z = q_pred_all[:, 1]  # 上分位数 Q_hi(X)

        # ========== 7. 共形预测校准 ==========
        # 在验证集上计算超出量，用于校准分位数预测的偏差
        # 确保阈值在新数据上仍满足预期覆盖率
        
        z_val = z_all[va]        # 验证集的真实标准化残差
        q_lo_val = q_lo_z[va]    # 验证集的下分位数预测
        q_hi_val = q_hi_z[va]    # 验证集的上分位数预测
        
        # 计算超出上分位数的量：exceed_hi = max(0, z - Q_hi)
        # 只关心超出部分（正值），未超出则为 0
        exceed_hi = np.maximum(0.0, z_val - q_hi_val)
        
        # 计算超出下分位数的量：exceed_lo = max(0, Q_lo - z)
        # 只关心低于下界的部分（正值），未超出则为 0
        exceed_lo = np.maximum(0.0, q_lo_val - z_val)
        
        # 计算校准常数：超出量的 tau_hi 分位数
        # c_plus 用于扩大上界，确保 (1 - tau_hi) 的数据在阈值内
        c_plus = float(np.quantile(exceed_hi, tau_hi)) if exceed_hi.size > 0 else 0.0
        
        # c_minus 用于扩大下界，确保 tau_lo 的数据在阈值内
        c_minus = float(np.quantile(exceed_lo, tau_lo)) if exceed_lo.size > 0 else 0.0


        # ========== 8. 还原到功率域 ==========
        # 标准化残差的阈值需要乘以 D 还原到功率域
        # 校准后的上界: (Q_hi + c_plus) * D
        thr_pos = np.maximum(0.0, (q_hi_z + c_plus)) * D_all_local
        
        # 校准后的下界: (c_minus - Q_lo) * D
        # 注意：Q_lo 通常为负值，c_minus - Q_lo 得到正的偏差量
        thr_neg = np.maximum(0.0, (c_minus - q_lo_z)) * D_all_local

        # ========== 9. 判断异常点 ==========
        # 获取原始残差（功率域）
        res = cfg["residuals"]  # res = y - y_hat
        
        # 判断是否异常：
        # - 正向异常: res > thr_pos（实际功率远高于预测）
        # - 负向异常: res < -thr_neg（实际功率远低于预测）
        is_abn = (res > thr_pos) | (res < -thr_neg)
        
        # ========== 10. 返回结果 ==========
        return ThresholdOutputs(thr_pos=thr_pos, thr_neg=thr_neg, is_abnormal=is_abn)
