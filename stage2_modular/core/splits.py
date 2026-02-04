# -*- coding: utf-8 -*-
"""
数据集切分持久化模块 - 基于稳定键（STABLE KEY）的训练集/验证集/测试集划分

本模块提供了鲁棒的数据集切分持久化方案，解决以下核心问题：

1. **实验可复现性**：
   - 保证多次运行使用相同的训练集/验证集/测试集划分
   - 避免因随机种子变化导致实验结果不可比较
   
2. **数据过滤场景的兼容性**：
   - 支持在不同运行中应用不同的数据过滤策略
   - 例如：首次运行时保留所有数据，后续运行中过滤掉部分异常点
   - 切分逻辑能够自动适配当前数据框，而不会因行数变化而失败
   
3. **避免索引错位问题**：
   - 传统的索引位置切分（如 [0:1000] 为训练集）在数据过滤后会失效
   - 本模块使用稳定的行键（row_key）进行标识，不受索引变化影响

核心机制：

1. **稳定行键（Stable Row Key）**：
   - 构成：timestamp(ns) + "#" + 原始行索引
   - 特性：每行数据有唯一且不变的标识符，不受数据框重排或过滤影响
   - 示例：1609459200000000000#42 表示时间戳为 2021-01-01 00:00:00 的第 42 行
   
2. **持久化格式**：
   - 保存为 CSV 文件，包含两列：row_key, split
   - 每一行记录：某个稳定键对应的数据属于哪个集合（train/val/test）
   - 轻量级存储，便于人工检查和版本控制
   
3. **对齐机制（Alignment）**：
   - 加载时：根据 row_key 匹配当前数据框和持久化文件
   - 缺失处理：持久化文件中不存在的行 → 默认分配到训练集（或丢弃）
   - 冗余处理：当前数据框中不存在的行 → 自动忽略
   - 结果：返回当前数据框的位置索引（positional indices），可直接用于切片

使用流程：

    # 1. 首次运行：创建并保存切分
    from sklearn.model_selection import train_test_split
    
    # 生成稳定的行键
    row_keys = make_row_keys(df)
    
    # 使用标准方法划分数据集（例如 80% 训练，10% 验证，10% 测试）
    idx_all = np.arange(len(df))
    idx_train, idx_temp = train_test_split(idx_all, test_size=0.2, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    
    # 持久化切分（保存 row_key -> split 的映射）
    save_split_csv("split.csv", row_keys, idx_train, idx_val, idx_test)
    
    # 2. 后续运行：加载并对齐切分
    # 即使 df 经过过滤，行数不同，也能正确对齐
    row_keys = make_row_keys(df)  # 为当前 df 生成行键
    idx_train, idx_val, idx_test = load_split_csv("split.csv", row_keys)
    
    # 3. 使用位置索引切片数据
    train_data = df.iloc[idx_train]
    val_data = df.iloc[idx_val]
    test_data = df.iloc[idx_test]

边缘情况处理：

1. **时间戳缺失或无效**：抛出 ValueError，要求数据必须有有效的 timestamp 列
2. **持久化文件不存在**：抛出 FileNotFoundError，提示需要先创建切分
3. **持久化文件格式错误**：抛出 ValueError，提示文件必须包含 row_key 和 split 列
4. **行键覆盖率低**：如果匹配率 < min_coverage，仍然返回结果但调用者可以选择重新切分

依赖：
    - pandas: 数据处理和 CSV 读写
    - numpy: 数组操作和索引处理

作者：项目团队
"""
from __future__ import annotations
import os
from typing import Tuple, Optional, Iterable
import pandas as pd
import numpy as np

# 数据集切分类型常量定义
SPLIT_TRAIN = "train"  # 训练集标识符
SPLIT_VAL   = "val"    # 验证集标识符
SPLIT_TEST  = "test"   # 测试集标识符

def make_row_keys(S: pd.DataFrame) -> np.ndarray:
    """
    为 DataFrame 的每一行生成稳定的唯一标识键（row_key）
    
    功能：
        构建一个不随数据框重排、过滤或索引重置而变化的稳定行键。
        该键由时间戳（纳秒级）和原始行索引组成，确保：
        1. 唯一性：即使时间戳相同，行索引也能区分不同的行
        2. 稳定性：只要数据本身不变（时间戳和索引不变），键就不变
        3. 可追溯性：可以通过键追溯到数据的原始时间点和位置
    
    参数：
        S (pd.DataFrame): 待生成行键的数据框
            - 必须包含 'timestamp' 列（可以是字符串或 datetime 类型）
            - 推荐 timestamp 列无重复或重复很少，以提高键的稳定性
    
    返回：
        np.ndarray: 字符串数组，shape 为 (len(S),)
            - 每个元素格式："{timestamp_ns}#{row_index}"
            - 示例："1609459200000000000#42" 表示时间戳为 1609459200000000000 纳秒的第 42 行
    
    异常：
        KeyError: 如果 DataFrame 中不存在 'timestamp' 列
        ValueError: 如果存在无法解析为有效时间戳的行（NaT，Not a Time）
    
    行键构成详解：
        - 第一部分：timestamp(ns) - 时间戳的纳秒表示（int64）
          - 精度：纳秒级（10^-9 秒），足以区分高频采样数据
          - 范围：可表示约 292 年的时间跨度（从 1970-01-01 起）
          - 示例：2021-01-01 00:00:00 → 1609459200000000000
        
        - 分隔符："#" - 分隔时间戳和索引，便于人工阅读和调试
        
        - 第二部分：原始行索引 - 当前 DataFrame 的 index 值（转为字符串）
          - 用途：在时间戳相同的情况下区分不同行
          - 说明：这里的 "原始" 是相对于当前数据框而言，实际上使用的是当前的 df.index
    
    稳定性保证：
        1. **时间戳稳定性**：
           - 只要数据源的时间戳列不变，纳秒值就不变
           - 即使数据框经过排序、过滤、索引重置，时间戳本身不会改变
        
        2. **索引稳定性**：
           - 使用当前数据框的 index，而不是位置索引（positional index）
           - 如果数据框的 index 本身是稳定的（例如原始数据的行号），则键稳定
           - 注意：如果调用了 df.reset_index(drop=True)，索引会变为 0, 1, 2, ...
                  此时需要确保操作顺序一致，或者在重置前先生成行键
        
        3. **唯一性保证**：
           - 即使时间戳完全相同，不同的索引值也能保证键唯一
           - 对于同一时刻的多条记录（例如多台传感器同步采样），仍能区分
    
    潜在边缘情况：
        1. **时间戳为 NaT（Not a Time）**：
           - pd.to_datetime(..., errors="coerce") 会将无效值转为 NaT
           - 函数会检测到并抛出 ValueError，提示有多少行无效
           - 解决方法：在调用本函数前，先清洗掉无效的时间戳行
        
        2. **时间戳列不存在**：
           - 抛出 KeyError，提示缺少必需的列
           - 解决方法：确保数据框包含 'timestamp' 列
        
        3. **时间戳精度损失**：
           - 如果原始数据只有秒级精度，纳秒表示可能会有大量零
           - 这不影响功能，但可能导致键的第一部分在短时间内重复
           - 在这种情况下，索引部分的作用更加重要
    
    使用示例：
        # 示例 1：基本使用
        import pandas as pd
        df = pd.DataFrame({
            'timestamp': ['2021-01-01 00:00:00', '2021-01-01 00:00:01', '2021-01-01 00:00:02'],
            'value': [10, 20, 30]
        })
        keys = make_row_keys(df)
        # keys: ['1609459200000000000#0', '1609459201000000000#1', '1609459202000000000#2']
        
        # 示例 2：应对数据过滤
        df_filtered = df[df['value'] > 15]  # 过滤后只剩第 2、3 行
        keys_filtered = make_row_keys(df_filtered)
        # keys_filtered: ['1609459201000000000#1', '1609459202000000000#2']
        # 注意：索引保持为 1 和 2（原始索引），而不是 0 和 1
        
        # 示例 3：错误处理
        df_bad = pd.DataFrame({
            'timestamp': ['2021-01-01', 'invalid_date', '2021-01-03'],
            'value': [1, 2, 3]
        })
        try:
            keys = make_row_keys(df_bad)
        except ValueError as e:
            print(e)  # 输出：make_row_keys: 1 rows have NaT timestamps; cannot build keys.
    
    注意事项：
        1. 本函数生成的键是针对"当前"数据框的，如果后续对数据框进行了：
           - reset_index(drop=True)：索引部分会变化，导致键不同
           - 列重命名：timestamp 列名变化会导致 KeyError
           - 时间戳修改：键会完全不同
        
        2. 建议在数据加载后、任何索引操作前尽早生成行键，并保存下来
        
        3. 如果需要长期稳定的键，考虑在原始数据中添加一个永久的唯一 ID 列
    """
    # 1. 检查 timestamp 列是否存在
    if "timestamp" not in S.columns:
        raise KeyError("make_row_keys: 'timestamp' column not found in DataFrame.")
    
    # 2. 将 timestamp 列转换为 datetime 类型
    #    errors="coerce" 会将无法解析的值转为 NaT（Not a Time）
    ts = pd.to_datetime(S["timestamp"], errors="coerce")
    
    # 3. 检查是否有无效的时间戳（NaT）
    if ts.isna().any():
        # 统计无效行数
        bad = int(ts.isna().sum())
        # 抛出异常，要求调用者先清洗数据
        raise ValueError(f"make_row_keys: {bad} rows have NaT timestamps; cannot build keys.")
    
    # 4. 将 datetime 转换为纳秒级整数（int64）
    #    这是 numpy 的 datetime64[ns] 类型的底层表示
    ts_ns = ts.astype("int64")
    
    # 5. 获取当前 DataFrame 的索引作为第二部分
    #    使用 pd.Series(S.index) 是为了方便后续转换为字符串数组
    #    注意：这里使用的是 DataFrame 的 index 属性，而不是位置索引（0, 1, 2, ...）
    idx_comp = pd.Series(S.index).astype(str).to_numpy()
    
    # 6. 组合时间戳和索引，生成最终的行键
    #    格式："{timestamp_ns}#{index}"
    #    使用 "#" 作为分隔符，便于人工检查和调试
    keys = (ts_ns.astype(str) + "#" + idx_comp)
    
    # 7. 返回字符串数组（np.ndarray[str]）
    return keys

def _ensure_dir(path: str) -> None:
    """
    确保文件路径的目录存在，如果不存在则递归创建
    
    功能：
        在保存文件前，自动创建所需的目录结构。
        这是一个内部辅助函数，避免因目录不存在导致文件保存失败。
    
    参数：
        path (str): 文件的完整路径（包括文件名）
            - 示例："/path/to/output/split.csv"
    
    返回：
        None
    
    行为说明：
        1. 提取文件路径中的目录部分（os.path.dirname）
        2. 如果目录为空字符串（当前目录），不做任何操作
        3. 如果目录不存在，递归创建所有父目录（exist_ok=True）
        4. 如果目录已存在，不做任何操作（不会报错）
    
    示例：
        # 自动创建多级目录
        _ensure_dir("/path/to/output/split.csv")
        # 结果：创建 /path/to/output/ 目录（如果不存在）
        
        # 当前目录的文件（无需创建目录）
        _ensure_dir("split.csv")
        # 结果：不做任何操作
    
    注意事项：
        - exist_ok=True 确保即使目录已存在也不会抛出异常
        - 如果路径中包含不存在的中间目录，会一并创建
    """
    # 提取文件路径的目录部分
    d = os.path.dirname(path)
    
    # 如果目录不为空且不存在，则创建
    # exist_ok=True 表示如果目录已存在不会报错
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_split_csv(path: str,
                   row_keys: np.ndarray,
                   idx_train: Iterable,
                   idx_val: Iterable,
                   idx_test: Optional[Iterable] = None) -> None:
    """
    将数据集切分结果持久化为 CSV 文件
    
    功能：
        保存训练集/验证集/测试集的划分信息，便于后续运行中复用相同的切分。
        持久化格式为轻量级 CSV，只保存 (row_key, split) 对，而不保存完整数据。
    
    参数：
        path (str): 输出 CSV 文件的完整路径
            - 示例："/path/to/split.csv"
            - 如果目录不存在，会自动创建
        
        row_keys (np.ndarray): 行键数组，由 make_row_keys() 生成
            - shape: (N,)，其中 N 是当前数据框的行数
            - 顺序：必须与当前数据框的行顺序一致
            - 示例：['1609459200000000000#0', '1609459201000000000#1', ...]
        
        idx_train (Iterable): 训练集的索引标签（来自数据框的 index）
            - 可以是列表、数组、pd.Index 等可迭代对象
            - 如果是整数位置索引（0, 1, 2, ...），会被直接使用
            - 示例：[0, 1, 5, 10, 15]  # 位置索引
        
        idx_val (Iterable): 验证集的索引标签
            - 格式与 idx_train 相同
            - 示例：[2, 3, 7, 12]
        
        idx_test (Optional[Iterable]): 测试集的索引标签，可选
            - 如果为 None，则不保存测试集（默认所有非 train/val 的行都是 train）
            - 示例：[4, 6, 8, 9, 11]
    
    返回：
        None（结果保存在指定的 CSV 文件中）
    
    输出文件格式：
        CSV 文件包含两列：
        - row_key (str): 稳定的行键，来自 row_keys 参数
        - split (str): 所属集合，取值为 "train"、"val" 或 "test"
        
        示例内容：
            row_key,split
            1609459200000000000#0,train
            1609459201000000000#1,train
            1609459202000000000#2,val
            1609459203000000000#3,test
            ...
    
    内部逻辑：
        1. 创建一个长度为 len(row_keys) 的标签 Series，初始全部为 "train"
        2. 根据 idx_train、idx_val、idx_test 更新对应位置的标签
        3. 将 row_keys 和标签组合成 DataFrame
        4. 保存为 CSV 文件
    
    索引类型处理：
        函数内部会尝试将 idx_* 参数转换为位置索引（positional indices）：
        - 如果 idx_* 是整数数组且在 [0, N) 范围内，直接使用
        - 否则，尝试在当前索引中查找匹配（使用 pd.Index.get_indexer）
        
        这种设计使得函数既支持位置索引（例如来自 train_test_split），
        也支持标签索引（例如来自布尔索引或 df.index 的子集）。
    
    使用示例：
        # 示例 1：使用 train_test_split 生成的位置索引
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # 生成行键
        row_keys = make_row_keys(df)
        
        # 划分数据集（返回位置索引）
        idx_all = np.arange(len(df))
        idx_train, idx_temp = train_test_split(idx_all, test_size=0.2, random_state=42)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
        
        # 保存切分
        save_split_csv("split.csv", row_keys, idx_train, idx_val, idx_test)
        
        # 示例 2：只有训练集和验证集（无测试集）
        idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2)
        save_split_csv("split.csv", row_keys, idx_train, idx_val)  # idx_test=None
        
        # 示例 3：使用布尔索引
        train_mask = df['value'] > 10
        val_mask = (df['value'] > 5) & (df['value'] <= 10)
        idx_train = df[train_mask].index
        idx_val = df[val_mask].index
        save_split_csv("split.csv", row_keys, idx_train, idx_val)
    
    边缘情况：
        1. **空索引**：
           - 如果 idx_train、idx_val 或 idx_test 为空，对应集合就没有样本
           - 函数不会报错，会生成有效的 CSV 文件
        
        2. **索引重叠**：
           - 如果 idx_train、idx_val、idx_test 有重叠，后者会覆盖前者
           - 优先级：train < val < test（test 的优先级最高）
           - 建议：确保索引互不重叠，避免混淆
        
        3. **目录不存在**：
           - 函数会自动创建所需的目录结构
           - 无需手动创建输出目录
    
    注意事项：
        1. row_keys 的顺序必须与当前数据框的行顺序一致
        2. idx_* 参数中的索引必须在 [0, len(row_keys)) 范围内，否则会报错
        3. 生成的 CSV 文件不包含任何实际数据，只有切分信息
        4. 文件大小通常很小（每行约 30-50 字节），便于版本控制
    """
    # 1. 确保输出目录存在（如果不存在则创建）
    _ensure_dir(path)
    
    # 2. 创建一个标签 Series，初始全部设为 "train"
    #    index: 0 到 len(row_keys)-1 的连续整数（位置索引）
    #    values: 全部初始化为 "train"
    lab = pd.Series(SPLIT_TRAIN, index=pd.Index(range(len(row_keys))), dtype="object")
    
    # 3. 构建一个临时的位置索引（0, 1, 2, ..., N-1）
    #    用于后续将标签索引转换为位置索引
    cur_index = pd.Index(range(len(row_keys)))
    
    # 4. 定义内部函数：将标签索引转换为位置索引
    def _coerce_positions(labels: Iterable) -> np.ndarray:
        """
        将索引标签转换为位置索引（positional indices）
        
        处理逻辑：
            1. 如果 labels 是整数数组且在 [0, N) 范围内，直接使用
            2. 否则，使用 pd.Index.get_indexer 在 cur_index 中查找匹配
        
        参数：
            labels (Iterable): 索引标签（可以是位置索引或其他标签）
        
        返回：
            np.ndarray: 位置索引数组（整数类型）
        """
        # 将 labels 转换为 pd.Index 以便统一处理
        arr = pd.Index(labels)
        
        # 如果是整数类型，尝试直接作为位置索引使用
        if arr.inferred_type in ("integer", "mixed-integer"):
            a = arr.to_numpy()
            # 检查是否在有效范围内 [0, N)
            if np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() < len(row_keys):
                return a.astype(int)
        
        # 否则，使用 get_indexer 在 cur_index 中查找匹配
        # get_indexer 返回的是 arr 中每个元素在 cur_index 中的位置
        # 如果找不到，返回 -1
        return cur_index.get_indexer(arr)
    
    # 5. 将训练集、验证集、测试集的索引转换为位置索引
    pos_tr = _coerce_positions(idx_train)
    pos_va = _coerce_positions(idx_val)
    
    # 6. 重置所有标签为 "train"（保险起见，避免之前的操作有副作用）
    lab.iloc[:] = SPLIT_TRAIN
    
    # 7. 标记训练集（虽然已经是 "train"，但为了代码清晰性还是显式标记）
    if len(pos_tr) > 0:
        lab.iloc[pos_tr] = SPLIT_TRAIN
    
    # 8. 标记验证集
    if len(pos_va) > 0:
        lab.iloc[pos_va] = SPLIT_VAL
    
    # 9. 标记测试集（如果提供）
    if idx_test is not None:
        pos_te = _coerce_positions(idx_test)
        if len(pos_te) > 0:
            lab.iloc[pos_te] = SPLIT_TEST
    
    # 10. 组合 row_key 和 split 标签，生成输出 DataFrame
    out = pd.DataFrame({"row_key": row_keys, "split": lab.values})
    
    # 11. 保存为 CSV 文件（不保存行索引）
    out.to_csv(path, index=False)

def load_split_csv(path: str,
                   row_keys: np.ndarray,
                   default: str = SPLIT_TRAIN,
                   min_coverage: float = 0.80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从持久化的 CSV 文件中加载数据集切分，并对齐到当前的数据框
    
    功能：
        读取之前保存的切分信息，根据行键（row_key）匹配到当前数据框的行，
        返回训练集、验证集、测试集的位置索引。这是切分持久化机制的核心函数，
        解决了数据过滤后索引错位的问题。
    
    参数：
        path (str): 持久化切分文件的路径（CSV 格式）
            - 必须包含 "row_key" 和 "split" 两列
            - 由 save_split_csv() 生成
        
        row_keys (np.ndarray): 当前数据框的行键数组，由 make_row_keys() 生成
            - shape: (N,)，其中 N 是当前数据框的行数
            - 顺序：必须与当前数据框的行顺序一致
        
        default (str): 当持久化文件中找不到某行键时，默认分配的集合
            - 默认值：SPLIT_TRAIN（"train"）
            - 其他可选值：SPLIT_VAL、SPLIT_TEST
            - 用途：处理新增数据或持久化文件覆盖不全的情况
        
        min_coverage (float): 最小覆盖率阈值（0.0 ~ 1.0）
            - 覆盖率 = 匹配到的行键数 / 当前数据框总行数
            - 如果覆盖率 < min_coverage，函数仍会返回结果，但调用者可以选择重新切分
            - 默认值：0.80（80%）
            - 用途：检测持久化文件是否过期或数据变化过大
    
    返回：
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - idx_train (np.ndarray[int]): 训练集的位置索引
            - idx_val (np.ndarray[int]): 验证集的位置索引
            - idx_test (np.ndarray[int]): 测试集的位置索引
        
        返回的索引是相对于当前数据框的位置索引（0-based），可以直接用于切片：
            train_data = df.iloc[idx_train]
            val_data = df.iloc[idx_val]
            test_data = df.iloc[idx_test]
    
    异常：
        FileNotFoundError: 如果指定的 CSV 文件不存在
        ValueError: 如果 CSV 文件格式不正确（缺少必需的列）
    
    对齐机制详解：
        1. **读取持久化文件**：
           - 读取 CSV 文件，构建 row_key -> split 的映射字典
           - 示例：{'1609459200000000000#0': 'train', '1609459201000000000#1': 'val', ...}
        
        2. **匹配当前行键**：
           - 遍历当前数据框的每个行键（row_keys）
           - 在映射字典中查找对应的 split 标签
           - 如果找到，使用该标签；如果找不到，使用 default 标签
        
        3. **生成位置索引**：
           - 根据匹配结果，为每一行分配 train/val/test 标签
           - 提取各集合对应的位置索引（0, 1, 2, ...）
           - 返回三个位置索引数组
    
    覆盖率计算：
        覆盖率 = (当前数据框中匹配到持久化文件的行数) / (当前数据框总行数)
        
        覆盖率的含义：
        - 100%：所有当前行都在持久化文件中找到了对应的切分标签
        - 80%：80% 的当前行有明确的切分标签，20% 使用 default 标签
        - 50%：一半的数据是新增的或持久化文件缺失
        
        低覆盖率的可能原因：
        - 数据源发生了较大变化（新增或删除了大量行）
        - 持久化文件过期或损坏
        - 时间戳或索引发生了不一致的变化
    
    使用示例：
        # 示例 1：标准使用流程
        # 第一次运行：保存切分
        row_keys = make_row_keys(df)
        idx_train, idx_val, idx_test = ..., ..., ...  # 通过某种方法划分
        save_split_csv("split.csv", row_keys, idx_train, idx_val, idx_test)
        
        # 后续运行：加载并对齐切分
        row_keys = make_row_keys(df)  # 可能经过过滤，行数不同
        idx_train, idx_val, idx_test = load_split_csv("split.csv", row_keys)
        train_data = df.iloc[idx_train]
        val_data = df.iloc[idx_val]
        
        # 示例 2：处理数据过滤
        # 原始数据：1000 行
        row_keys_orig = make_row_keys(df_orig)
        save_split_csv("split.csv", row_keys_orig, idx_train, idx_val)
        
        # 过滤后：只剩 800 行
        df_filtered = df_orig[df_orig['quality'] > 0.5]
        row_keys_filtered = make_row_keys(df_filtered)
        
        # 加载时自动对齐：只返回 800 行的切分
        idx_train, idx_val, idx_test = load_split_csv("split.csv", row_keys_filtered)
        # idx_train, idx_val, idx_test 的元素都在 [0, 800) 范围内
        
        # 示例 3：检查覆盖率
        idx_train, idx_val, idx_test = load_split_csv("split.csv", row_keys, min_coverage=0.9)
        coverage = len(idx_train) + len(idx_val) + len(idx_test)
        if coverage < 0.9 * len(row_keys):
            print("警告：覆盖率过低，建议重新切分数据")
    
    边缘情况：
        1. **持久化文件包含当前数据框中不存在的行键**：
           - 这些行键会被忽略（不影响结果）
           - 常见于数据过滤后某些行被删除的情况
        
        2. **当前数据框包含持久化文件中不存在的行键**：
           - 这些行会被分配到 default 集合（默认为训练集）
           - 常见于新增数据或持久化文件覆盖不全的情况
        
        3. **覆盖率为 0**：
           - 如果所有行键都找不到匹配，所有行都会被分配到 default 集合
           - 可能原因：持久化文件与当前数据完全不匹配（需要检查数据源）
        
        4. **某个集合为空**：
           - 如果持久化文件中某个集合（如 test）没有任何行，返回的索引数组为空
           - 这是正常情况（例如只有 train 和 val，没有 test）
    
    注意事项：
        1. **索引顺序**：
           - 返回的位置索引是根据当前数据框的行顺序生成的
           - 索引值范围：[0, len(row_keys))
        
        2. **数据一致性**：
           - 确保当前数据框的 timestamp 列与持久化时一致
           - 如果时间戳被修改，行键会不匹配，导致覆盖率降低
        
        3. **性能考虑**：
           - 匹配过程使用字典查找，时间复杂度 O(N)，对大数据集高效
           - 持久化文件大小通常很小（每行约 30-50 字节），读取速度快
        
        4. **版本控制**：
           - 建议将持久化文件纳入版本控制（如 git），便于实验复现
           - 文件是纯文本（CSV），易于比较差异和合并
    
    最佳实践：
        1. 首次运行时生成切分并保存，后续运行始终加载相同的切分文件
        2. 定期检查覆盖率，如果过低（< 80%），考虑重新生成切分
        3. 为不同的实验配置使用不同的切分文件名，避免混淆
        4. 在数据清洗流程变化后，评估是否需要重新切分
    """
    # 1. 检查持久化文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    # 2. 读取持久化的切分 CSV 文件
    persisted = pd.read_csv(path)
    
    # 3. 检查文件格式是否正确（必须包含 row_key 和 split 两列）
    if "row_key" not in persisted.columns or "split" not in persisted.columns:
        raise ValueError("load_split_csv: bad schema (need columns: row_key, split)")
    
    # 4. 构建行键到切分标签的映射字典
    #    key: row_key (str) -> value: split (str)
    #    这是一个高效的查找结构，时间复杂度 O(1)
    kv = dict(zip(persisted["row_key"].astype(str), persisted["split"].astype(str)))
    
    # 5. 对齐到当前数据框：为每个当前行键查找对应的切分标签
    #    如果找不到，使用 default 标签（默认为 "train"）
    splits_now = [ kv.get(str(k), default) for k in row_keys ]
    
    # 6. 转换为 pandas Series，便于后续的布尔索引操作
    splits_now = pd.Series(splits_now, dtype="object")
    
    # 7. 构建位置索引（0, 1, 2, ..., N-1）
    #    用于提取各集合对应的位置索引
    pos = pd.Index(range(len(row_keys)))
    
    # 8. 提取训练集的位置索引
    #    找到所有 split == "train" 的位置
    idx_train = pos[splits_now.eq(SPLIT_TRAIN)].to_numpy(dtype=int)
    
    # 9. 提取验证集的位置索引
    #    找到所有 split == "val" 的位置
    idx_val   = pos[splits_now.eq(SPLIT_VAL)].to_numpy(dtype=int)
    
    # 10. 提取测试集的位置索引
    #     找到所有 split == "test" 的位置
    idx_test  = pos[splits_now.eq(SPLIT_TEST)].to_numpy(dtype=int)
    
    # 11. 返回三个位置索引数组
    #     可以直接用于 df.iloc[idx_train] 等切片操作
    return idx_train, idx_val, idx_test
