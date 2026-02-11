# -*- coding: utf-8 -*-
"""
功率分位数回归阈值方法

本模块实现了基于功率的分位数回归阈值计算方法（QuantilePower）。

核心思想：
    直接在功率空间拟合分位数曲面，学习 P 的上下分位数函数：
    Q_P(v, ρ; τ) = f(v, ρ; τ)
    
    其中：
    - v: 风速
    - ρ: 空气密度（可选）
    - τ: 分位数水平（如 0.05 和 0.95）
    
    拟合得到的 Q_lo(v, ρ) 和 Q_hi(v, ρ) 直接代表功率的下界和上界。

方法流程：
    1. 准备特征矩阵: X = [v] 或 X = [v, ρ]（已标准化）
    2. 使用 MLP 拟合功率分位数曲面: Q_P(X; τ) = f_MLP(X; τ)
    3. 共形预测校准: 在验证集上计算偏移量 c_plus 和 c_minus
    4. 调整阈值: lb = Q_lo - c_minus, ub = Q_hi + c_plus
    5. 转化为"围绕 ŷ 的阈值": thr_pos = ub - ŷ, thr_neg = ŷ - lb

与 QuantileZResid 方法的区别：
    - QuantilePower（本方法）:
      * 直接拟合功率 P 的分位数 Q_P(v, ρ)
      * 优点: 直观易懂，阈值直接在功率域，无需额外的标准差估计
      * 缺点: 异方差问题显著（高风速区间波动大，低风速区间波动小）
               MLP 需要同时捕捉不同风速区间的不同波动尺度，泛化性较弱
      * 适用场景: 功率波动相对均匀的情况，或数据量充足时
      
    - QuantileZResid（标准化残差方法）:
      * 拟合标准化残差 z = (P - P̂) / D 的分位数 Q_z(v, ρ)
      * 优点: 标准化后各区间波动相当，模型更稳定，泛化性更好
      * 缺点: 需要额外的标准差估计 D，计算稍复杂
      * 适用场景: 功率波动随风速变化显著（常见情况）
      * 推荐: 通常情况下推荐使用 QuantileZResid

为什么需要转化为"围绕 ŷ 的阈值"？
    1. 统一接口: 
       所有阈值方法最终需要提供统一格式的输出，即相对于预测值 ŷ 的偏差阈值：
       - thr_pos: 允许的正向偏差（实际功率 > 预测功率）
       - thr_neg: 允许的负向偏差（实际功率 < 预测功率）
       
    2. 异常判断: 
       统一为 "res > thr_pos 或 res < -thr_neg" 即为异常，
       其中 res = y - ŷ 是残差
       
    3. 等价性: 
       原始判断: y < lb 或 y > ub（功率超出绝对边界）
       转化后判断: y - ŷ < -(ŷ - lb) 或 y - ŷ > (ub - ŷ)
       即: res < -thr_neg 或 res > thr_pos（残差超出相对阈值）
       
    4. 便于可视化和分析: 
       相对阈值更直观地反映了"预测的不确定性"，
       而不是"功率的绝对范围"

共形预测校准的作用：
    即使分位数模型拟合良好，也可能存在系统性偏差（过于乐观或保守）。
    共形预测通过验证集的超出量（exceed）统计，计算校准偏移 c，
    确保阈值在新数据上仍满足预期覆盖率（如 90%）。
    
    具体步骤:
        1. 在验证集上计算超出量: exceed_hi = max(0, y_val - Q_hi_val)
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

class QuantilePower(ThresholdMethod):
    """
    功率分位数回归阈值方法
    
    通过对功率 P 进行分位数回归，直接学习功率的上下分位数曲面，
    然后转化为围绕预测值 ŷ 的阈值。
    
    相比标准化残差方法，本方法更直观，但在处理异方差时效果较弱。
    """
    name = "quantile_power"

    def compute(self, *, train_X, train_zp, train_zn, query_X, D_all,
                idx_train_mask, idx_val_mask, taus, cfg, device=None):
        """
        计算基于功率分位数回归的阈值
        
        主要步骤：
        1. 设备规范化（CPU/GPU）
        2. 准备特征矩阵 X（风速 + 可选的空气密度）
        3. 在训练集上拟合功率分位数 MLP，验证集上评估
        4. 共形预测校准：计算验证集超出量的分位数作为偏移
        5. 得到功率的绝对上下界: lb, ub
        6. 转化为围绕 ŷ 的相对阈值: thr_pos = ub - ŷ, thr_neg = ŷ - lb
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
        # 构造用于拟合功率分位曲面的特征
        # 优先使用 "_for_model"（建模用标准化数据），其次使用 "_for_clean"（清洗用数据）
        if use_rho:
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

        # ========== 4. 准备目标变量和预测值 ==========
        y_all = cfg["y"]          # 真实功率 (n_samples,)
        y_hat_all = cfg["y_hat"]  # 预测功率 (n_samples,)
        
        # ========== 5. 准备训练集和验证集索引 ==========
        tr = idx_train_mask  # 训练集掩码（布尔数组）
        va = idx_val_mask    # 验证集掩码（布尔数组）

        # ========== 6. 拟合功率的分位数曲面 ==========
        # 使用 MLP 直接拟合功率 P 的上下分位数：Q_lo(X) 和 Q_hi(X)
        taus_list = [tau_lo, tau_hi]  # 如 [0.05, 0.95]
        
        # 在训练集上拟合，验证集上评估
        model, _ = fit_quantile_mlp(
            X_all[tr], y_all[tr],    # 训练数据：特征 X 和目标 y（功率）
            X_all[va], y_all[va],    # 验证数据：用于早停和评估
            taus=taus_list,          # 分位数水平列表
            cfg=qmlp_cfg,            # MLP 配置（隐藏层、学习率等）
            device=dev,              # 训练设备（CPU 或 GPU）
            verbose=1                # 打印训练信息
        )
        
        # 对所有样本进行预测，得到功率分位数曲面
        q_pred_all = predict_quantiles(model, X_all)  # (n_samples, 2)
        q_lo_all = q_pred_all[:, 0]  # 功率下分位数 Q_lo(X)
        q_hi_all = q_pred_all[:, 1]  # 功率上分位数 Q_hi(X)

        # ========== 7. 共形预测校准 ==========
        # 在验证集上计算超出量，用于校准分位数预测的偏差
        # 确保阈值在新数据上仍满足预期覆盖率
        
        y_val = y_all[va]        # 验证集的真实功率
        q_lo_val = q_lo_all[va]  # 验证集的功率下分位数预测
        q_hi_val = q_hi_all[va]  # 验证集的功率上分位数预测
        
        # 计算超出上分位数的量：exceed_hi = max(0, y - Q_hi)
        # 只关心超出部分（正值），未超出则为 0
        exceed_hi = np.maximum(0.0, y_val - q_hi_val)
        
        # 计算超出下分位数的量：exceed_lo = max(0, Q_lo - y)
        # 只关心低于下界的部分（正值），未超出则为 0
        exceed_lo = np.maximum(0.0, q_lo_val - y_val)
        
        # 计算校准常数：超出量的 tau_hi 分位数
        # c_plus 用于扩大上界，确保 (1 - tau_hi) 的数据在阈值内
        c_plus = float(np.quantile(exceed_hi, tau_hi)) if exceed_hi.size > 0 else 0.0
        
        # c_minus 用于扩大下界，确保 tau_lo 的数据在阈值内
        c_minus = float(np.quantile(exceed_lo, tau_lo)) if exceed_lo.size > 0 else 0.0

        # ========== 8. 得到校准后的功率绝对上下界 ==========
        # 上界：原始上分位数 + 正向校准偏移
        ub = q_hi_all + c_plus
        
        # 下界：原始下分位数 - 负向校准偏移
        lb = q_lo_all - c_minus

        # ========== 9. 转化为"围绕 ŷ 的阈值" ==========
        # 为了与其他阈值方法保持统一接口，需要将绝对边界转化为相对阈值
        # thr_pos: 允许的正向偏差（y - ŷ 的上限）
        # thr_neg: 允许的负向偏差（ŷ - y 的上限）
        # 
        # 原始异常判断: y > ub 或 y < lb
        # 等价于: y - ŷ > (ub - ŷ) 或 y - ŷ < -(ŷ - lb)
        # 即: res > thr_pos 或 res < -thr_neg
        
        # 正向阈值 = 上界 - 预测值（最大允许的正残差）
        thr_pos = np.maximum(0.0, ub - y_hat_all)
        
        # 负向阈值 = 预测值 - 下界（最大允许的负残差的绝对值）
        thr_neg = np.maximum(0.0, y_hat_all - lb)

        # ========== 10. 判断异常点 ==========
        # 获取原始残差（功率域）
        res = cfg["residuals"]  # res = y - y_hat
        
        # 判断是否异常：
        # - 正向异常: res > thr_pos（实际功率远高于预测，超出上界）
        # - 负向异常: res < -thr_neg（实际功率远低于预测，超出下界）
        is_abn = (res > thr_pos) | (res < -thr_neg)
        
        # ========== 11. 返回结果 ==========
        return ThresholdOutputs(thr_pos=thr_pos, thr_neg=thr_neg, is_abnormal=is_abn)
