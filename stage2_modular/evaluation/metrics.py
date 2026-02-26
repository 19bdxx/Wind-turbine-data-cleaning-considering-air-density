# stage2_modular/evaluation/metrics.py
# -*- coding: utf-8 -*-
"""
评估模块：用于计算清洗质量指标和模型性能指标。

主要功能：
  - compute_cleaning_stats: 计算各 Pass 的异常率和清洗率
  - compute_model_metrics: 计算 RMSE/MAE/R² 等建模指标
  - compute_seasonal_metrics: 按季节分组的指标分解
  - summarize_run: 读取某次实验的所有输出 CSV 并汇总
  - compare_runs: 将多次实验结果汇总为对比表格
"""
from __future__ import annotations

import os
import glob
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

_SEASON_MAP = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
               5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
               9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter"}


def _season_of_month(month: int) -> str:
    return _SEASON_MAP.get(int(month), "Unknown")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]; yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _read_csv_any(path: str) -> pd.DataFrame:
    last = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "cp936"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"读取失败：{path}；最后错误：{last}")


# ─────────────────────────────────────────────────────────────────────────────
# 公共 API
# ─────────────────────────────────────────────────────────────────────────────

def compute_cleaning_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    从 stage2 输出的 DataFrame 计算清洗统计指标。

    输入列（若存在）：
      - Pass1_异常, Pass2_异常: 布尔型异常标志
      - S_scope: 布尔型，表示在分析范围内的样本

    返回字典包含：
      - n_total:       总样本数（所有行）
      - n_scope:       在 wind_scope 且规则通过的有效样本数
      - n_pass1_abn:   Pass1 检出异常数
      - n_pass2_abn:   Pass2 增量检出异常数
      - n_clean:       清洗后剩余干净样本数
      - rate_pass1:    Pass1 异常率（占 scope 样本）
      - rate_pass2:    Pass2 增量异常率（占 scope 样本）
      - rate_total_abn:总异常率（Pass1 ∪ Pass2，占 scope 样本）
      - rate_clean:    清洗后保留率（占 scope 样本）
    """
    n_total = len(df)

    scope_col = "S_scope"
    if scope_col in df.columns:
        scope_mask = df[scope_col].fillna(False).astype(bool)
    else:
        scope_mask = pd.Series(True, index=df.index)

    n_scope = int(scope_mask.sum())

    p1_col = "Pass1_异常"; p2_col = "Pass2_异常"
    p1 = df[p1_col].fillna(False).astype(bool) if p1_col in df.columns else pd.Series(False, index=df.index)
    p2 = df[p2_col].fillna(False).astype(bool) if p2_col in df.columns else pd.Series(False, index=df.index)

    p1_scope = p1 & scope_mask
    p2_scope = p2 & scope_mask
    abn_scope = (p1 | p2) & scope_mask

    n_pass1_abn = int(p1_scope.sum())
    n_pass2_abn = int(p2_scope.sum())
    n_total_abn = int(abn_scope.sum())
    n_clean = n_scope - n_total_abn

    denom = max(n_scope, 1)
    return {
        "n_total": n_total,
        "n_scope": n_scope,
        "n_pass1_abn": n_pass1_abn,
        "n_pass2_abn": n_pass2_abn,
        "n_total_abn": n_total_abn,
        "n_clean": n_clean,
        "rate_pass1": n_pass1_abn / denom,
        "rate_pass2": n_pass2_abn / denom,
        "rate_total_abn": n_total_abn / denom,
        "rate_clean": n_clean / denom,
    }


def compute_model_metrics(
    df: pd.DataFrame,
    split: str = "val",
    power_col: str = "power",
    pred_col: str = "pred_center",
    exclude_abn: bool = True,
) -> Dict[str, float]:
    """
    计算模型预测精度指标（RMSE / MAE / R²）。

    参数：
      df:          stage2 输出的 DataFrame（含 split 列、power 列、pred_center 列）
      split:       评估集标签，'val' / 'test' / 'train' / 'all'
      power_col:   实测功率列名
      pred_col:    预测功率列名（pred_center 或 pred_center_p1）
      exclude_abn: 若为 True，在 val/test 上排除 Pass1_异常 和 Pass2_异常 点（仅统计干净点精度）

    返回：
      rmse, mae, r2, n_samples
    """
    if "split" not in df.columns:
        sub = df.copy()
    elif split == "all":
        sub = df.copy()
    else:
        sub = df[df["split"] == split].copy()

    if exclude_abn and split in ("val", "test"):
        p1 = sub["Pass1_异常"].fillna(False).astype(bool) if "Pass1_异常" in sub.columns else pd.Series(False, index=sub.index)
        p2 = sub["Pass2_异常"].fillna(False).astype(bool) if "Pass2_异常" in sub.columns else pd.Series(False, index=sub.index)
        sub = sub[~(p1 | p2)]

    if pred_col not in sub.columns or power_col not in sub.columns:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_samples": 0}

    y_true = pd.to_numeric(sub[power_col], errors="coerce").to_numpy(float)
    y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(float)

    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
        "n_samples": int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred))),
    }


def compute_seasonal_metrics(
    df: pd.DataFrame,
    split: str = "val",
    power_col: str = "power",
    pred_col: str = "pred_center",
    exclude_abn: bool = True,
) -> pd.DataFrame:
    """
    按季节（Spring/Summer/Fall/Winter）分组计算 RMSE / MAE / R²。

    返回：
      DataFrame，index 为季节名称，列为 rmse / mae / r2 / n_samples / abn_rate。
    """
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["_season"] = df["timestamp"].dt.month.map(_SEASON_MAP)

    rows = []
    for season in ["Spring", "Summer", "Fall", "Winter"]:
        mask = df["_season"] == season
        sub = df[mask]
        if len(sub) == 0:
            continue
        metrics = compute_model_metrics(sub, split=split, power_col=power_col,
                                        pred_col=pred_col, exclude_abn=exclude_abn)
        cs = compute_cleaning_stats(sub)
        rows.append({
            "season": season,
            "n_scope": cs["n_scope"],
            "abn_rate": cs["rate_total_abn"],
            **metrics,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("season")


def summarize_run(
    run_dir: str,
    split: str = "val",
    power_col: str = "power",
    pred_col: str = "pred_center",
    exclude_abn: bool = True,
) -> Dict[str, object]:
    """
    读取一次实验下所有风机的 stage2 输出 CSV，汇总清洗统计与建模指标。

    run_dir 应为 <out_root>/<run_tag>/<station>_mlp/ 结构或其父目录。
    该函数会递归搜索所有符合 *_stage2_mlp.csv 的文件。

    返回：
      {
        "cleaning":  汇总的清洗统计（所有风机合并）,
        "metrics":   汇总的建模指标（所有风机合并）,
        "seasonal":  按季节汇总的 DataFrame,
        "per_turbine": {csv_path: {...}},
      }
    """
    pattern = os.path.join(run_dir, "**", "*_stage2_mlp.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))
    if not csv_files:
        return {"cleaning": {}, "metrics": {}, "seasonal": pd.DataFrame(), "per_turbine": {}}

    all_dfs: List[pd.DataFrame] = []
    per_turbine: Dict[str, Dict] = {}
    for path in csv_files:
        try:
            df = _read_csv_any(path)
            cs = compute_cleaning_stats(df)
            mm = compute_model_metrics(df, split=split, power_col=power_col,
                                       pred_col=pred_col, exclude_abn=exclude_abn)
            per_turbine[path] = {"cleaning": cs, "metrics": mm}
            all_dfs.append(df)
        except Exception as e:
            per_turbine[path] = {"error": str(e)}

    if not all_dfs:
        return {"cleaning": {}, "metrics": {}, "seasonal": pd.DataFrame(), "per_turbine": per_turbine}

    combined = pd.concat(all_dfs, ignore_index=True)
    cleaning_agg = compute_cleaning_stats(combined)
    metrics_agg = compute_model_metrics(combined, split=split, power_col=power_col,
                                        pred_col=pred_col, exclude_abn=exclude_abn)
    seasonal_agg = compute_seasonal_metrics(combined, split=split, power_col=power_col,
                                            pred_col=pred_col, exclude_abn=exclude_abn)

    return {
        "cleaning": cleaning_agg,
        "metrics": metrics_agg,
        "seasonal": seasonal_agg,
        "per_turbine": per_turbine,
    }


def compare_runs(
    results: Dict[str, Dict],
    metric_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    将多次实验（summarize_run 的输出）汇总为横向对比表格。

    参数：
      results:     {run_name: summarize_run(...) 的返回值}
      metric_keys: 需要展示的指标键列表（默认涵盖清洗率和建模指标）

    返回：
      DataFrame，行为 run_name，列为各指标。
    """
    if metric_keys is None:
        metric_keys = [
            "rate_pass1", "rate_pass2", "rate_total_abn", "rate_clean",
            "rmse", "mae", "r2", "n_samples",
        ]

    rows = []
    for run_name, res in results.items():
        row: Dict[str, object] = {"run": run_name}
        cleaning = res.get("cleaning", {})
        metrics = res.get("metrics", {})
        combined = {**cleaning, **metrics}
        for k in metric_keys:
            row[k] = combined.get(k, float("nan"))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("run")
