# -*- coding: utf-8 -*-
import torch

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def resolve_device(want="auto"):
    # 兼容 torch.device 直接传入
    if isinstance(want, torch.device):
        dev = want
    else:
        want = (want or "auto")
        if isinstance(want, str):
            want_l = want.lower()
        else:
            # 避免奇怪类型
            want_l = "auto"

        has = torch.cuda.is_available()
        if want_l == "cpu":
            dev = torch.device("cpu")
        elif want_l.startswith("cuda"):
            dev = torch.device(want if has else "cpu")
        else:
            dev = torch.device("cuda:0" if has else "cpu")

    # 打印信息
    has = torch.cuda.is_available()
    if has:
        try:
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(dev if dev.type == "cuda" else 0)
        except Exception:
            n, name = "?", "?"
        print(f"[Device] torch.cuda.is_available()={has}; GPUs={n}; using={dev}; name={name}")
    else:
        print("[Device] CUDA 不可用，使用 CPU")
    return dev
