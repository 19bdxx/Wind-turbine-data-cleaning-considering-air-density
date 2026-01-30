# -*- coding: utf-8 -*-
from typing import Dict, Type
from .base import ThresholdMethod
from .knn_local import KNNLocal
from .quantile_power import QuantilePower
from .quantile_zresid import QuantileZResid

_REGISTRY: Dict[str, Type[ThresholdMethod]] = {
    "knn": KNNLocal,
    "quantile_power": QuantilePower,
    "quantile_zresid": QuantileZResid,
    "quantile": QuantileZResid,
}

def get_method(name: str) -> ThresholdMethod:
    key = (name or "knn").lower()
    if key not in _REGISTRY:
        raise KeyError(f"未知阈值方法: {name}。可选: {list(_REGISTRY.keys())}")
    return _REGISTRY[key]()
