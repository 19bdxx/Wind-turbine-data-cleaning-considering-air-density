# -*- coding: utf-8 -*-
import os, math
from time import perf_counter
import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

class Stopwatch:
    def __init__(self): self.t0 = perf_counter(); self.last = self.t0
    def lap(self, label): 
        now = perf_counter(); dt = now - self.last; self.last = now
        print(f"[Time] {label}: {dt:.3f}s"); return dt
    def total(self, label="Total"): 
        now = perf_counter(); dt = now - self.t0
        print(f"[Time] {label}: {dt:.3f}s"); return dt

def read_csv_any(path: str) -> pd.DataFrame:
    last = None
    for enc in ["utf-8-sig","utf-8","gbk","cp936"]:
        try: return pd.read_csv(path, encoding=enc)
        except Exception as e: last = e
    raise RuntimeError(f"读取失败：{path}；最后错误：{last}")

def load_rho_table(wide_csv: str, station: str):
    dfw = read_csv_any(wide_csv)
    if "timestamp" not in dfw.columns:
        import os
        raise KeyError(f"{os.path.basename(wide_csv)} 缺少 timestamp 列")
    cand = [f"{station}_空气密度","空气密度","rho","density"]
    rho_col = next((c for c in cand if c in dfw.columns), None)
    if rho_col is None:
        return None
    out = dfw[["timestamp", rho_col]].copy()
    out.rename(columns={rho_col:"rho"}, inplace=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    return out.sort_values("timestamp")

def estimate_prated_from_series(p: pd.Series) -> float:
    pc = pd.to_numeric(p, errors="coerce").dropna()
    if pc.empty:
        return float("nan")
    q = float(np.quantile(pc, 0.995)); m = float(pc.max())
    return float(min(max(q, 0.0), m))
