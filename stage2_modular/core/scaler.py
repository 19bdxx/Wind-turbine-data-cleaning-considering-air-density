# -*- coding: utf-8 -*-
"""
数据标准化模块 - 风速和空气密度的归一化处理

本模块提供了 Scaler 类，用于对风机数据中的风速和空气密度进行标准化（归一化）处理。
标准化是机器学习模型训练前的关键步骤，可以加速收敛并提高模型性能。

主要功能：
1. 支持两种标准化方法：
   - MinMax 归一化：将数据缩放到 [0, 1] 区间
   - Z-score 标准化：转换为均值为0、标准差为1的分布
   
2. 支持固定范围和自适应范围：
   - 固定范围（fixed=True）：使用预设的风速和密度范围
   - 自适应范围（fixed=False）：从训练数据中学习范围参数
   
3. 计算标准化后的密度/风速比例尺度：
   - 用于权衡两种输入特征的相对重要性
   - 在模型训练中用于平衡梯度

依赖：
    - numpy: 数值计算
    - pandas: 数据处理（在 fit_from_train 中使用）

使用场景：
    - 在数据预处理阶段对训练集进行标准化
    - 使用训练集的统计参数对验证集、测试集进行相同的标准化
    - 在模型推理时对新数据进行标准化

作者：项目团队
"""
import numpy as np

class Scaler:
    """
    数据标准化器类 - 对风速和空气密度进行归一化处理
    
    功能：
        1. 提供 MinMax 和 Z-score 两种标准化方法
        2. 支持固定范围和从数据学习范围
        3. 同时处理风速和空气密度两种特征
        4. 计算标准化后的特征尺度比例
    
    属性：
        method (str): 标准化方法，"minmax" 或 "zscore"
        fixed (bool): 是否使用固定范围（True）或从数据学习（False）
        
        # MinMax 方法的范围参数：
        V_min (float): 风速的最小值（默认 0 m/s）
        V_max (float): 风速的最大值（默认 15 m/s）
        R_min (float): 空气密度的最小值（默认 1.07 kg/m³）
        R_max (float): 空气密度的最大值（默认 1.37 kg/m³）
        
        # Z-score 方法的统计参数：
        V_mu (float): 风速的均值（初始为 0.0）
        V_sigma (float): 风速的标准差（初始为 1.0）
        R_mu (float): 空气密度的均值（初始为 0.0）
        R_sigma (float): 空气密度的标准差（初始为 1.0）
    
    使用流程：
        # 1. 创建 Scaler 对象
        scaler = Scaler(method="zscore", fixed=False)
        
        # 2. 从训练数据中学习统计参数（仅当 method="zscore" 且 fixed=False 时有效）
        scaler.fit_from_train(train_wind, train_rho, use_rho=True)
        
        # 3. 对数据进行标准化
        wind_scaled, rho_scaled = scaler.transform(wind_data, rho_data, use_rho=True)
        
        # 4. 获取密度/风速的尺度比例（用于模型权重调整）
        ratio = scaler.scale_ratio_r_over_v()
    
    参数范围说明：
        - 风速范围 [0, 15]：覆盖大部分风机的工作风速区间
        - 密度范围 [1.07, 1.37]：覆盖海平面附近常见的空气密度变化范围
          （约对应 -10°C ~ 40°C 的温度变化）
    """
    
    def __init__(self, method="minmax", fixed=True, wind_range=(0,15), rho_range=(1.07,1.37)):
        """
        初始化 Scaler 对象
        
        参数：
            method (str): 标准化方法，可选值：
                - "minmax": MinMax 归一化，将数据缩放到 [0, 1] 区间
                  公式：x_scaled = (x - min) / (max - min)
                - "zscore": Z-score 标准化，转换为标准正态分布
                  公式：x_scaled = (x - mean) / std
                默认为 "minmax"
            
            fixed (bool): 是否使用固定的标准化范围
                - True: 使用预设的风速和密度范围（wind_range, rho_range）
                - False: 从训练数据中学习范围（仅对 zscore 有效）
                默认为 True
            
            wind_range (tuple): 风速的 (最小值, 最大值)，单位 m/s
                - 仅在 method="minmax" 时使用
                - 默认 (0, 15)，覆盖风机的主要工作区间
                - 常见风机的切入风速约 3 m/s，切出风速约 20-25 m/s
            
            rho_range (tuple): 空气密度的 (最小值, 最大值)，单位 kg/m³
                - 仅在 method="minmax" 时使用
                - 默认 (1.07, 1.37)，覆盖常见环境条件下的密度变化
                - 标准大气密度约 1.225 kg/m³
        
        返回：
            None
        
        初始化后的状态：
            - MinMax 方法：V_min, V_max, R_min, R_max 已设置
            - Z-score 方法：V_mu, V_sigma, R_mu, R_sigma 使用默认值（0, 1）
                           需要调用 fit_from_train() 从数据中学习真实参数
        """
        # 保存标准化方法和是否固定范围
        self.method = method
        self.fixed = fixed
        
        # 解包风速范围
        self.V_min, self.V_max = wind_range
        
        # 解包空气密度范围
        self.R_min, self.R_max = rho_range
        
        # 初始化 Z-score 的统计参数（均值和标准差）
        # 默认值（0, 1）表示未训练状态，需要通过 fit_from_train() 更新
        self.V_mu, self.V_sigma = 0.0, 1.0      # 风速的均值和标准差
        self.R_mu, self.R_sigma = 0.0, 1.0      # 空气密度的均值和标准差
    
    def fit_from_train(self, wind, rho, use_rho: bool):
        """
        从训练数据中学习标准化参数（仅对 Z-score 方法有效）
        
        功能：
            计算训练集的均值和标准差，用于后续的 Z-score 标准化。
            这是 Z-score 方法的"训练"过程，确保：
            1. 训练集标准化后均值≈0，标准差≈1
            2. 验证集和测试集使用相同的参数，保证数据分布一致
        
        参数：
            wind (pd.Series): 训练集的风速序列
                - 可能包含非数值或 NaN，会自动清洗
            
            rho (pd.Series): 训练集的空气密度序列
                - 可能包含非数值或 NaN，会自动清洗
            
            use_rho (bool): 是否使用空气密度特征
                - True: 同时学习风速和密度的统计参数
                - False: 仅学习风速的统计参数
        
        返回：
            None（结果保存在对象的属性中）
        
        副作用：
            - 更新 self.V_mu, self.V_sigma（风速的均值和标准差）
            - 如果 use_rho=True，更新 self.R_mu, self.R_sigma
        
        注意事项：
            1. 仅在 method="zscore" 时有效，MinMax 方法不需要调用此函数
            2. 必须在 transform() 之前调用，否则使用默认值（0, 1）
            3. 数据清洗策略：
               - 使用 pd.to_numeric(errors="coerce") 将无效值转为 NaN
               - 使用 dropna() 删除 NaN 值
            4. 空数据保护：
               - 如果清洗后数据为空，均值设为 0.0，标准差设为 1.0
               - 避免除零错误和数值异常
        """
        import pandas as pd
        
        # 仅对 Z-score 方法执行参数学习（MinMax 使用固定范围，不需要训练）
        if self.method == "zscore":
            # ========== 1. 处理风速数据 ==========
            # 将风速转换为数值类型，无法转换的值变为 NaN
            v = pd.to_numeric(wind, errors="coerce").dropna().values
            
            # 计算风速的均值和标准差
            # 如果数据为空（v.size == 0），使用默认值避免异常
            self.V_mu = float(v.mean()) if v.size else 0.0        # 均值
            self.V_sigma = float(v.std()) if v.size else 1.0      # 标准差（保证非零）
            
            # ========== 2. 处理空气密度数据（如果启用）==========
            if use_rho:
                # 将空气密度转换为数值类型，无法转换的值变为 NaN
                r = pd.to_numeric(rho, errors="coerce").dropna().values
                
                # 计算空气密度的均值和标准差
                # 如果数据为空，使用默认值避免异常
                self.R_mu = float(r.mean()) if r.size else 0.0        # 均值
                self.R_sigma = float(r.std()) if r.size else 1.0      # 标准差（保证非零）
    
    def transform(self, v_arr, r_arr, use_rho: bool):
        """
        对风速和空气密度数据进行标准化转换
        
        功能：
            根据初始化时指定的方法（MinMax 或 Z-score）和学习到的参数，
            对输入的风速和密度数据进行标准化，使其适合输入到机器学习模型。
        
        参数：
            v_arr (np.ndarray): 待标准化的风速数组，shape 任意
                - 单位：m/s
                - 可以是 1D、2D 或多维数组
            
            r_arr (np.ndarray): 待标准化的空气密度数组，shape 与 v_arr 相同
                - 单位：kg/m³
                - 如果 use_rho=False，此参数会被忽略
            
            use_rho (bool): 是否对空气密度进行标准化
                - True: 返回标准化后的风速和密度
                - False: 仅返回标准化后的风速，密度返回 None
        
        返回：
            tuple: (v_std, r_std)
                - v_std (np.ndarray): 标准化后的风速数组
                  - MinMax: 缩放到 [0, 1] 区间
                  - Z-score: 均值≈0，标准差≈1
                
                - r_std (np.ndarray | None): 标准化后的空气密度数组
                  - use_rho=True: 返回标准化后的密度
                  - use_rho=False: 返回 None
        
        标准化公式：
            # MinMax 归一化：
            v_std = (v - V_min) / (V_max - V_min)
            r_std = (r - R_min) / (R_max - R_min)
            
            # Z-score 标准化：
            v_std = (v - V_mu) / V_sigma
            r_std = (r - R_mu) / R_sigma
        
        数值稳定性保护：
            - 在除法运算中，分母至少为 1e-12，避免除零错误
            - max(divisor, 1e-12) 确保即使范围或标准差为 0，也不会崩溃
        
        使用示例：
            # MinMax 标准化
            scaler = Scaler(method="minmax", wind_range=(0, 20))
            v_scaled, rho_scaled = scaler.transform(wind_data, rho_data, use_rho=True)
            
            # Z-score 标准化（需要先 fit）
            scaler = Scaler(method="zscore")
            scaler.fit_from_train(train_wind, train_rho, use_rho=True)
            v_scaled, rho_scaled = scaler.transform(test_wind, test_rho, use_rho=True)
        """
        import numpy as np
        
        if self.method == "zscore":
            # ========== Z-score 标准化 ==========
            # 转换为标准正态分布：均值0，标准差1
            
            # 标准化风速：(v - 均值) / 标准差
            # max(self.V_sigma, 1e-12) 防止标准差为 0 导致除零错误
            v_std = (v_arr - self.V_mu) / max(self.V_sigma, 1e-12)
            
            # 标准化空气密度（如果启用）
            if use_rho:
                # (rho - 均值) / 标准差
                r_std = (r_arr - self.R_mu) / max(self.R_sigma, 1e-12)
            else:
                r_std = None  # 不使用密度时返回 None
            
            return v_std, r_std
        else:
            # ========== MinMax 归一化 ==========
            # 缩放到 [0, 1] 区间
            
            # 归一化风速：(v - 最小值) / (最大值 - 最小值)
            # max(V_max - V_min, 1e-12) 防止范围为 0 导致除零错误
            v_std = (v_arr - self.V_min) / max(self.V_max - self.V_min, 1e-12)
            
            # 归一化空气密度（如果启用）
            if use_rho:
                # (rho - 最小值) / (最大值 - 最小值)
                r_std = (r_arr - self.R_min) / max(self.R_max - self.R_min, 1e-12)
            else:
                r_std = None  # 不使用密度时返回 None
            
            return v_std, r_std
    
    def scale_ratio_r_over_v(self):
        """
        计算标准化后空气密度相对风速的尺度比例
        
        功能：
            计算密度和风速在标准化后的"量纲比"，用于：
            1. 权衡两种输入特征的相对重要性
            2. 在神经网络中调整切向（tangent）方向的权重
            3. 平衡梯度，避免某个特征主导训练过程
        
        返回：
            float: 密度尺度 / 风速尺度 的比例
                - Z-score: R_sigma / V_sigma（密度标准差 / 风速标准差）
                - MinMax: (R_max - R_min) / (V_max - V_min)（密度范围 / 风速范围）
        
        物理意义：
            - 比值 > 1：密度的变化范围（标准化前）比风速大
                        在标准化后，密度特征的数值波动更大
            - 比值 < 1：风速的变化范围比密度大
                        在标准化后，风速特征的数值波动更大
            - 比值 ≈ 1：两者的变化范围相当
        
        使用场景：
            在混合密度-风速模型中，用于调整切向方向的权重：
            - 例如：tangent_weight = base_weight * scale_ratio
            - 确保密度和风速在模型中的影响力适当平衡
        
        数值稳定性保护：
            - 分母至少为 1e-12，避免除零错误
            - 使用 max() 确保即使标准差或范围为 0，也能返回合理的值
        
        典型值：
            - 风速范围 [0, 15] m/s，密度范围 [1.07, 1.37] kg/m³
              ratio = (1.37 - 1.07) / (15 - 0) = 0.3 / 15 ≈ 0.02
            - 这意味着密度的变化范围远小于风速，需要适当放大其权重
        """
        if self.method == "zscore":
            # Z-score 方法：使用标准差的比值
            # ratio = σ_rho / σ_wind
            # 表示密度和风速在标准化后的相对波动幅度
            return float(self.R_sigma / max(self.V_sigma, 1e-12))
        
        # MinMax 方法：使用范围的比值
        # ratio = (R_max - R_min) / (V_max - V_min)
        # 表示密度和风速的原始变化范围之比
        return float((self.R_max - self.R_min) / max(self.V_max - self.V_min, 1e-12))
