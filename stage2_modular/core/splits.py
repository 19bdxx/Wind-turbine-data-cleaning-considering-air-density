# -*- coding: utf-8 -*-
"""
Robust split persistence with STABLE KEYS and alignment.

Key idea:
  - Compute a stable per-row key using timestamp(ns) + original row index.
  - When saving a split, persist only (row_key, split) pairs.
  - When loading a split, align back to the CURRENT dataframe by matching row_key.
    Any rows missing in persisted file are safely treated as 'train' (or dropped),
    avoiding boolean-index shape mismatches.
"""
from __future__ import annotations
import os
from typing import Tuple, Optional, Iterable
import pandas as pd
import numpy as np

SPLIT_TRAIN = "train"
SPLIT_VAL   = "val"
SPLIT_TEST  = "test"

def make_row_keys(S: pd.DataFrame) -> np.ndarray:
    """
    Build a STABLE key for each row using timestamp(ns) + original row index.
    This is robust to different filtering in later runs.
    Returns: np.ndarray[str] with shape (len(S),)
    """
    if "timestamp" not in S.columns:
        raise KeyError("make_row_keys: 'timestamp' column not found in DataFrame.")
    ts = pd.to_datetime(S["timestamp"], errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"make_row_keys: {bad} rows have NaT timestamps; cannot build keys.")
    ts_ns = ts.astype("int64")
    # Use the CURRENT DataFrame index as second component to guarantee uniqueness.
    idx_comp = pd.Series(S.index).astype(str).to_numpy()
    keys = (ts_ns.astype(str) + "#" + idx_comp)
    return keys

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_split_csv(path: str,
                   row_keys: np.ndarray,
                   idx_train: Iterable,
                   idx_val: Iterable,
                   idx_test: Optional[Iterable] = None) -> None:
    """
    Persist split as CSV with columns: row_key, split.
    - row_keys: array of str keys for the CURRENT S (same order as S.index)
    - idx_train/idx_val/idx_test: index labels (from S.index) indicating membership
    """
    _ensure_dir(path)
    # Build a Series of labels indexed by current S.index order
    lab = pd.Series(SPLIT_TRAIN, index=pd.Index(range(len(row_keys))), dtype="object")
    # Build a synthetic current index (0..n-1) aligned with row_keys order.
    cur_index = pd.Index(range(len(row_keys)))
    # If idx_* are integer positions, use directly; otherwise assume they are labels equal to positions.
    def _coerce_positions(labels: Iterable) -> np.ndarray:
        arr = pd.Index(labels)
        # If already integer and within [0, n), take directly
        if arr.inferred_type in ("integer", "mixed-integer"):
            a = arr.to_numpy()
            if np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() < len(row_keys):
                return a.astype(int)
        # Else try treating labels as positions equal to their values
        return cur_index.get_indexer(arr)

    pos_tr = _coerce_positions(idx_train)
    pos_va = _coerce_positions(idx_val)
    lab.iloc[:] = SPLIT_TRAIN
    if len(pos_tr) > 0:
        lab.iloc[pos_tr] = SPLIT_TRAIN
    if len(pos_va) > 0:
        lab.iloc[pos_va] = SPLIT_VAL
    if idx_test is not None:
        pos_te = _coerce_positions(idx_test)
        if len(pos_te) > 0:
            lab.iloc[pos_te] = SPLIT_TEST

    out = pd.DataFrame({"row_key": row_keys, "split": lab.values})
    out.to_csv(path, index=False)

def load_split_csv(path: str,
                   row_keys: np.ndarray,
                   default: str = SPLIT_TRAIN,
                   min_coverage: float = 0.80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load persisted split CSV and ALIGN to CURRENT row_keys.
    Returns positional indices (np.ndarray[int]) for train/val/test relative to current S.
    - Any missing keys fallback to `default` (train).
    - If coverage < min_coverage, we still align but caller may choose to rebuild.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    persisted = pd.read_csv(path)
    if "row_key" not in persisted.columns or "split" not in persisted.columns:
        raise ValueError("load_split_csv: bad schema (need columns: row_key, split)")

    # Build mapping from key->split from persisted file
    kv = dict(zip(persisted["row_key"].astype(str), persisted["split"].astype(str)))
    # Align to current keys
    splits_now = [ kv.get(str(k), default) for k in row_keys ]
    splits_now = pd.Series(splits_now, dtype="object")

    # Build positional lists
    pos = pd.Index(range(len(row_keys)))
    idx_train = pos[splits_now.eq(SPLIT_TRAIN)].to_numpy(dtype=int)
    idx_val   = pos[splits_now.eq(SPLIT_VAL)].to_numpy(dtype=int)
    idx_test  = pos[splits_now.eq(SPLIT_TEST)].to_numpy(dtype=int)
    return idx_train, idx_val, idx_test
