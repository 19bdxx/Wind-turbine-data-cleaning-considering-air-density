# -*- coding: utf-8 -*-
"""
阈值方法基类模块

本模块定义了异常检测阈值方法的基础架构，包括：
1. ThresholdOutputs: 阈值计算结果的数据类
2. ThresholdMethod: 所有阈值方法的抽象基类（模板方法模式）

设计思想：
- 采用模板方法模式，定义了阈值计算的统一接口
- 所有具体的阈值方法（KNN、分位数等）都继承此基类
- 确保不同方法的输出格式统一，便于上层调用
"""
from dataclasses import dataclass

@dataclass
class ThresholdOutputs:
    """
    阈值计算输出的数据类
    
    封装了阈值方法的三个核心输出结果，用于后续的异常判断和可视化。
    
    属性:
        thr_pos: 正向阈值（上界）
                 表示允许的最大正向偏差（功率高于预测值的上限）
                 通常为 numpy.ndarray，形状为 (n_samples,)
                 
        thr_neg: 负向阈值（下界）
                 表示允许的最大负向偏差（功率低于预测值的下限）
                 通常为 numpy.ndarray，形状为 (n_samples,)
                 注意：thr_neg 是正数，实际下界为 y_hat - thr_neg
                 
        is_abnormal: 异常标记
                     布尔数组，标记每个样本是否为异常点
                     True 表示该点被判定为异常（超出阈值范围）
                     通常为 numpy.ndarray，形状为 (n_samples,)，dtype=bool
    
    使用示例:
        outputs = ThresholdOutputs(
            thr_pos=np.array([100, 150, 200]),
            thr_neg=np.array([80, 120, 180]),
            is_abnormal=np.array([False, True, False])
        )
    """
    thr_pos: object
    thr_neg: object
    is_abnormal: object

class ThresholdMethod:
    """
    阈值方法抽象基类
    
    采用模板方法模式，定义了所有阈值计算方法的统一接口。
    所有具体的阈值方法（KNN、QuantilePower、QuantileZResid 等）
    都必须继承此类并实现 compute 方法。
    
    属性:
        name: 方法的唯一标识符，用于注册和查找
    
    设计模式:
        - 模板方法模式: 定义算法框架，具体实现由子类完成
        - 策略模式: 不同的阈值计算策略可以互换使用
    """
    name = "base"
    
    def compute(self, *, train_X, train_zp, train_zn, query_X, D_all,
                idx_train_mask, idx_val_mask, taus, cfg, device=None):
        """
        计算异常检测阈值的统一接口（抽象方法）
        
        这是所有阈值方法必须实现的核心方法。根据训练数据和配置参数，
        计算出每个样本的正负阈值，并判断哪些样本为异常。
        
        参数:
            train_X: 训练集特征（风速、空气密度等）
                    numpy.ndarray, 形状 (n_train, n_features)
                    
            train_zp: 训练集的正向标准化残差
                     numpy.ndarray, 形状 (n_train,)
                     计算方式: max(0, (y - y_hat) / D)
                     
            train_zn: 训练集的负向标准化残差
                     numpy.ndarray, 形状 (n_train,)
                     计算方式: max(0, (y_hat - y) / D)
                     
            query_X: 查询集特征（需要计算阈值的数据点）
                    numpy.ndarray, 形状 (n_query, n_features)
                    
            D_all: 所有样本的期望离差（标准差代理）
                  numpy.ndarray, 形状 (n_all,)
                  用于将标准化残差还原到功率域
                  
            idx_train_mask: 训练集索引掩码
                           布尔数组，指示哪些样本用于训练
                           
            idx_val_mask: 验证集索引掩码
                         布尔数组，指示哪些样本用于验证/校准
                         
            taus: 分位数水平 (tau_lo, tau_hi)
                 tuple of float，如 (0.05, 0.95)
                 表示期望的覆盖率（如 90% 的数据在阈值内）
                 
            cfg: 全局配置字典
                包含数据、模型预测、方法参数等所有配置
                
            device: PyTorch 设备（'cpu' 或 'cuda'）
                   用于神经网络模型的训练，可选参数
        
        返回:
            ThresholdOutputs: 包含 thr_pos、thr_neg 和 is_abnormal 的结果对象
        
        异常:
            NotImplementedError: 子类必须实现此方法
        
        实现要点:
            1. 不同方法可以使用不同的算法（KNN、分位数回归等）
            2. 最终输出格式必须统一为 ThresholdOutputs
            3. 阈值应为正数，表示偏离预测值的最大允许距离
            4. is_abnormal 应基于 residuals 与阈值的比较得出
        """
        raise NotImplementedError
