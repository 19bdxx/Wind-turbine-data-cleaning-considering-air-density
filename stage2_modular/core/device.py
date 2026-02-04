# -*- coding: utf-8 -*-
"""
设备管理模块 - PyTorch 计算设备选择与配置

本模块提供了智能的计算设备（CPU/GPU）选择功能，主要包括：
1. 自动检测 CUDA 可用性
2. 根据用户配置选择合适的设备
3. 优化 PyTorch 性能设置

依赖：
    - torch: PyTorch 深度学习框架

作者：项目团队
"""

import torch

# ========== PyTorch 性能优化配置 ==========

# 启用 cuDNN 自动调优（benchmark mode）
# 功能：在第一次运行时尝试多种卷积算法，选择最快的并缓存
# 适用场景：输入尺寸固定的模型，可以显著提升训练速度
# 注意：如果输入尺寸变化频繁，反而会降低性能
torch.backends.cudnn.benchmark = True

# 设置矩阵乘法精度为 "high"（PyTorch 1.12+ 支持）
# 功能：在 Ampere 架构（RTX 30系列、A100等）的 GPU 上启用 TF32 加速
# 效果：牺牲微小精度换取显著的速度提升（对大多数应用影响极小）
# 注意：某些 PyTorch 版本可能不支持此功能，用 try-except 包裹
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass  # 如果不支持，静默跳过（不影响主要功能）


def resolve_device(want="auto"):
    """
    智能设备选择函数 - 根据需求和硬件可用性选择计算设备
    
    功能：
        1. 检测 CUDA 是否可用
        2. 根据用户需求（want 参数）选择合适的设备
        3. 打印设备信息（GPU 型号、数量等）
        4. 返回 torch.device 对象供模型使用
    
    参数：
        want (str | torch.device): 期望使用的设备，支持以下值：
            - "auto" (默认): 自动选择，优先使用 GPU
            - "cpu": 强制使用 CPU
            - "cuda" 或 "cuda:0": 使用指定的 GPU
            - torch.device 对象: 直接使用该设备对象
    
    返回：
        torch.device: 实际使用的设备对象
            - 如果请求 GPU 但不可用，自动降级到 CPU
            - 如果请求 CPU，则使用 CPU
            - 如果 auto，优先使用 GPU（cuda:0）
    
    设备选择逻辑：
        1. 如果 want 已经是 torch.device 对象，直接使用
        2. 如果 want="cpu"，使用 CPU
        3. 如果 want 以 "cuda" 开头：
           - CUDA 可用：使用请求的 GPU
           - CUDA 不可用：降级到 CPU（并打印提示）
        4. 如果 want="auto" 或其他值：
           - CUDA 可用：使用 cuda:0
           - CUDA 不可用：使用 CPU
    
    使用示例：
        # 自动选择（优先 GPU）
        device = resolve_device("auto")
        
        # 强制 CPU
        device = resolve_device("cpu")
        
        # 指定 GPU 编号
        device = resolve_device("cuda:1")
        
        # 使用现有设备对象
        existing_dev = torch.device("cuda:0")
        device = resolve_device(existing_dev)
    """
    # ========== 1. 处理输入参数 ==========
    
    # 如果传入的已经是 torch.device 对象，直接使用
    if isinstance(want, torch.device):
        dev = want
    else:
        # 确保 want 是字符串类型，避免意外类型导致错误
        want = (want or "auto")  # None 或空字符串默认为 "auto"
        if isinstance(want, str):
            want_l = want.lower()  # 转小写，便于比较
        else:
            # 如果既不是 device 也不是字符串，使用 auto 作为兜底
            want_l = "auto"

        # ========== 2. 检测 CUDA 可用性 ==========
        has = torch.cuda.is_available()
        
        # ========== 3. 根据需求选择设备 ==========
        if want_l == "cpu":
            # 用户明确要求 CPU
            dev = torch.device("cpu")
        elif want_l.startswith("cuda"):
            # 用户要求使用 GPU
            if has:
                # CUDA 可用：使用请求的 GPU（保留完整设备名，如 "cuda:1"）
                dev = torch.device(want)
            else:
                # CUDA 不可用：降级到 CPU
                dev = torch.device("cpu")
        else:
            # 默认情况（auto 或其他）：优先使用 GPU
            dev = torch.device("cuda:0" if has else "cpu")

    # ========== 4. 打印设备信息 ==========
    # 提供详细的设备信息，便于调试和性能分析
    has = torch.cuda.is_available()
    if has:
        # GPU 可用，打印 GPU 详细信息
        try:
            # 获取 GPU 数量
            n = torch.cuda.device_count()
            # 获取当前使用的 GPU 型号
            # 如果 dev 是 CPU 设备，则查询 GPU 0 的信息
            name = torch.cuda.get_device_name(dev if dev.type == "cuda" else 0)
        except Exception:
            # 某些情况下可能获取失败（例如权限问题），使用占位符
            n, name = "?", "?"
        print(f"[Device] torch.cuda.is_available()={has}; GPUs={n}; using={dev}; name={name}")
    else:
        # GPU 不可用，提示使用 CPU
        print("[Device] CUDA 不可用，使用 CPU")
    
    return dev
