# -*- coding: utf-8 -*-
"""
核心工具模块 - 通用工具函数集合

本模块提供了项目中常用的工具函数，包括：
1. Stopwatch 类：性能计时器，用于测量代码执行时间
2. read_csv_any：智能 CSV 读取函数，自动尝试多种编码
3. load_rho_table：从宽表数据中提取空气密度时间序列
4. estimate_prated_from_series：从功率数据估计风机额定功率

依赖：
    - pandas: 数据处理
    - numpy: 数值计算

作者：项目团队
"""

import os, math
from time import perf_counter
import numpy as np
import pandas as pd

# 设置 pandas 选项：明确禁用静默向下转型警告（pandas 2.x+）
# 这可以避免在数据类型转换时出现难以察觉的精度损失
pd.set_option('future.no_silent_downcasting', True)


class Stopwatch:
    """
    简易性能计时器类
    
    功能：
        - 记录代码块的执行时间
        - 支持多个计时点（lap），便于分析性能瓶颈
        - 自动打印计时结果
    
    使用示例：
        sw = Stopwatch()
        # ... 代码块1 ...
        sw.lap("加载数据")       # 打印从上一个计时点到现在的耗时
        # ... 代码块2 ...
        sw.lap("数据预处理")
        # ... 代码块3 ...
        sw.total("总耗时")      # 打印从创建 Stopwatch 到现在的总耗时
    """
    
    def __init__(self): 
        """初始化计时器，记录起始时间"""
        self.t0 = perf_counter()     # 创建时的时间戳（总计时起点）
        self.last = self.t0           # 上一个计时点的时间戳
    
    def lap(self, label): 
        """
        记录一个计时点（lap），打印距离上一个计时点的耗时
        
        参数：
            label (str): 当前计时点的描述标签
        
        返回：
            float: 距离上一个计时点的时间差（秒）
        """
        now = perf_counter()          # 当前时间戳
        dt = now - self.last          # 计算时间差
        self.last = now               # 更新"上一个计时点"为当前时刻
        print(f"[Time] {label}: {dt:.3f}s")  # 打印耗时（保留3位小数）
        return dt
    
    def total(self, label="Total"): 
        """
        打印从创建 Stopwatch 到现在的总耗时
        
        参数：
            label (str): 总计时的描述标签，默认为 "Total"
        
        返回：
            float: 从创建到现在的总时间（秒）
        """
        now = perf_counter()
        dt = now - self.t0            # 相对于起始时间的总时间差
        print(f"[Time] {label}: {dt:.3f}s")
        return dt


def read_csv_any(path: str) -> pd.DataFrame:
    """
    智能 CSV 读取函数 - 自动尝试多种常见编码
    
    功能：
        尝试按顺序使用以下编码读取 CSV 文件：
        1. utf-8-sig  （带 BOM 的 UTF-8，Excel 导出的 CSV 常用此编码）
        2. utf-8      （标准 UTF-8）
        3. gbk        （中文 Windows 系统常用编码）
        4. cp936      （GBK 的别名，兼容性更好）
        
        如果所有编码都失败，则抛出异常并报告最后一个错误。
    
    参数：
        path (str): CSV 文件的完整路径
    
    返回：
        pd.DataFrame: 读取成功的数据框
    
    异常：
        RuntimeError: 如果所有编码尝试都失败
    
    使用场景：
        当 CSV 文件编码不确定时（例如来自不同数据源），
        使用本函数可以避免因编码问题导致读取失败。
    """
    last = None  # 记录最后一个错误，供最终异常信息使用
    
    # 按优先级尝试各种编码
    for enc in ["utf-8-sig","utf-8","gbk","cp936"]:
        try: 
            return pd.read_csv(path, encoding=enc)
        except Exception as e: 
            last = e  # 记录错误，继续尝试下一个编码
    
    # 所有编码都失败，抛出异常
    raise RuntimeError(f"读取失败：{path}；最后错误：{last}")


def load_rho_table(wide_csv: str, station: str):
    """
    从宽表数据文件中提取空气密度时间序列
    
    功能：
        1. 读取包含多站点环境数据的"宽表"CSV 文件
        2. 提取指定站点的空气密度列
        3. 返回 [timestamp, rho] 的二列数据框，供后续 merge 使用
    
    参数：
        wide_csv (str): 宽表数据文件路径
                       该文件应包含 timestamp 列和各站点的空气密度列
        station (str): 站点名称，用于匹配空气密度列名
    
    返回：
        pd.DataFrame 或 None: 
            - 成功时返回 DataFrame，包含 ["timestamp", "rho"] 两列
            - 如果找不到匹配的空气密度列，返回 None
    
    列名匹配规则：
        按以下优先级查找空气密度列：
        1. "{station}_空气密度"  （站点专属列，优先级最高）
        2. "空气密度"            （通用空气密度列）
        3. "rho"                （英文列名）
        4. "density"            （英文列名）
    
    异常：
        KeyError: 如果宽表文件缺少 timestamp 列
    
    注意事项：
        - 返回的数据框已按 timestamp 排序
        - 重复的 timestamp 会保留最后一条记录（keep="last"）
        - 无效的 timestamp 会被删除（dropna）
    """
    # 读取宽表数据（自动处理编码问题）
    dfw = read_csv_any(wide_csv)
    
    # 检查必需的 timestamp 列是否存在
    if "timestamp" not in dfw.columns:
        import os
        raise KeyError(f"{os.path.basename(wide_csv)} 缺少 timestamp 列")
    
    # 定义空气密度列名的候选列表（按优先级排序）
    # 优先匹配站点专属列，其次是通用列名
    cand = [f"{station}_空气密度","空气密度","rho","density"]
    
    # 查找第一个匹配的列名
    rho_col = next((c for c in cand if c in dfw.columns), None)
    
    # 如果找不到任何匹配的列，返回 None
    if rho_col is None:
        return None
    
    # 提取 timestamp 和空气密度列，创建新的数据框
    out = dfw[["timestamp", rho_col]].copy()
    
    # 将空气密度列重命名为统一的 "rho"
    out.rename(columns={rho_col:"rho"}, inplace=True)
    
    # 数据清洗：
    # 1. 将 timestamp 转换为 datetime 类型（errors="coerce" 会将无效值转为 NaT）
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    
    # 2. 删除 timestamp 为 NaT 的行
    out = out.dropna(subset=["timestamp"])
    
    # 3. 去除重复的 timestamp，保留最后一条（keep="last"）
    #    这样可以处理同一时刻有多条记录的情况
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    
    # 4. 按 timestamp 排序，便于后续 merge 操作
    return out.sort_values("timestamp")


def estimate_prated_from_series(p: pd.Series) -> float:
    """
    从功率数据中估计风机的额定功率（P_rated）
    
    功能：
        使用统计方法从历史功率数据中估算风机的额定功率。
        采用 99.5% 分位数和最大值的较小者，以避免异常峰值影响。
    
    参数：
        p (pd.Series): 功率时间序列（可能包含 NaN 或非数值）
    
    返回：
        float: 估计的额定功率（kW）
               - 如果输入为空或全为无效值，返回 NaN
               - 否则返回一个非负值
    
    估算方法：
        1. 清洗数据：将非数值转为 NaN 并删除
        2. 计算 99.5% 分位数（q）：排除异常高值
        3. 计算最大值（m）
        4. 取 min(max(q, 0), m) 作为估计值
        
        这种方法的优点：
        - 对异常峰值有较强的鲁棒性
        - 避免因个别测量误差导致过高的额定功率估计
        - 保证估计值非负且不超过实际观测最大值
    
    使用场景：
        当配置文件中未提供风机额定功率时，或需要验证配置的额定功率是否合理时使用。
    """
    # 将 Series 转换为数值类型，无法转换的值变为 NaN
    pc = pd.to_numeric(p, errors="coerce").dropna()
    
    # 如果清洗后为空，无法估算，返回 NaN
    if pc.empty:
        return float("nan")
    
    # 计算 99.5% 分位数（排除最高的 0.5% 数据，避免异常值影响）
    q = float(np.quantile(pc, 0.995))
    
    # 计算最大值
    m = float(pc.max())
    
    # 取分位数和最大值的较小者，并确保非负
    # min(max(q, 0), m) 的逻辑：
    # - max(q, 0)：确保分位数非负
    # - min(..., m)：确保不超过实际观测的最大值
    return float(min(max(q, 0.0), m))
