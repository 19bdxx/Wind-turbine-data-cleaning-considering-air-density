# -*- coding: utf-8 -*-
"""
阈值方法注册器模块

本模块实现了阈值方法的工厂模式注册机制，负责管理和创建不同的阈值计算方法。

核心功能：
1. 方法注册表（_REGISTRY）: 维护所有可用阈值方法的映射
2. 工厂函数（get_method）: 根据名称动态创建方法实例

已注册的阈值方法：
1. knn (KNNLocal):
   - 基于 K 近邻的局部阈值方法
   - 原理: 使用训练集中 K 个最近邻的残差分位数作为阈值
   - 优点: 自适应性强，能捕捉局部特性
   - 适用: 数据分布不均匀的场景

2. quantile_power (QuantilePower):
   - 功率域分位数回归方法
   - 原理: 直接对功率值 P 拟合分位数曲面 Q(v, ρ)
   - 优点: 直观，阈值直接在功率域
   - 适用: 功率波动范围随风速变化较规律的场景

3. quantile_zresid (QuantileZResid):
   - 标准化残差分位数回归方法（默认方法，别名 "quantile"）
   - 原理: 对标准化残差 z = (P - P̂) / D 拟合分位数曲面
   - 优点: 异方差问题得到缓解，泛化性能更好
   - 适用: 功率波动范围随风速变化显著的场景（推荐）

添加新方法的步骤：
1. 在 thresholds/ 目录下创建新的方法模块（如 new_method.py）
2. 继承 ThresholdMethod 基类并实现 compute 方法
3. 在 _REGISTRY 字典中注册新方法
4. 通过 get_method("new_method") 即可使用

设计模式：
- 工厂模式: 通过 get_method 工厂函数创建方法实例
- 注册表模式: 使用字典维护方法名到类的映射
- 单一职责: 本模块仅负责方法的注册和创建，不涉及具体算法
"""
from typing import Dict, Type
from .base import ThresholdMethod
from .knn_local import KNNLocal
from .quantile_power import QuantilePower
from .quantile_zresid import QuantileZResid

# 阈值方法注册表
# 键: 方法名称（字符串），值: 方法类（ThresholdMethod 的子类）
_REGISTRY: Dict[str, Type[ThresholdMethod]] = {
    "knn": KNNLocal,                     # K 近邻局部阈值方法
    "quantile_power": QuantilePower,     # 功率域分位数回归方法
    "quantile_zresid": QuantileZResid,   # 标准化残差分位数回归方法
    "quantile": QuantileZResid,          # 默认分位数方法（指向 quantile_zresid）
}

def get_method(name: str) -> ThresholdMethod:
    """
    工厂函数: 根据方法名称创建阈值方法实例
    
    这是获取阈值方法的统一入口，通过方法名称字符串动态创建对应的方法对象。
    使用工厂模式，调用方无需知道具体类名，降低耦合度。
    
    参数:
        name: 方法名称（字符串），如 "knn"、"quantile_power"、"quantile_zresid"
             - 不区分大小写（会自动转为小写）
             - 如果为 None 或空字符串，默认使用 "knn"
             - 支持别名，如 "quantile" 指向 "quantile_zresid"
    
    返回:
        ThresholdMethod: 已实例化的阈值方法对象
                        可以直接调用其 compute 方法进行阈值计算
    
    异常:
        KeyError: 当提供的方法名称未在注册表中时抛出
                 错误信息会列出所有可用的方法名称
    
    使用示例:
        # 创建 KNN 方法实例
        method = get_method("knn")
        outputs = method.compute(...)
        
        # 创建分位数方法实例（使用默认别名）
        method = get_method("quantile")
        outputs = method.compute(...)
        
        # 方法名称不区分大小写
        method = get_method("KNN")  # 等同于 "knn"
    
    扩展说明:
        添加新方法后，只需在 _REGISTRY 中注册，
        即可通过 get_method 获取，无需修改此函数。
    """
    # 标准化方法名称：转小写，None 时使用默认值
    key = (name or "knn").lower()
    
    # 检查方法是否已注册
    if key not in _REGISTRY:
        raise KeyError(f"未知阈值方法: {name}。可选: {list(_REGISTRY.keys())}")
    
    # 实例化并返回方法对象
    return _REGISTRY[key]()
