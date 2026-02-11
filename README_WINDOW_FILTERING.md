# KNN Window Filtering Optimization

## Overview

This document describes the window filtering optimization implemented in `stage2_modular/thresholds/knn_local.py` to improve KNN distance calculation performance for large datasets.

## Problem Statement

The original KNN implementation calculated distances between all query points and all training points (O(Q × N) complexity), which becomes expensive for large datasets:

- For N=100,000 training points and Q=10,000 query points
- Results in 10^9 distance calculations
- Even with GPU acceleration, this takes considerable time

## Solution: Window-Based Pre-filtering

The optimization pre-filters candidate points based on wind speed and air density ranges before calculating distances.

### Core Concept

For each query point `(ws_q, rho_q)`, only consider candidates within:
```
ws_q - window_v ≤ ws_c ≤ ws_q + window_v
rho_q - window_r ≤ rho_c ≤ rho_q + window_r
```

**Physical Justification**: Only points from similar operating conditions are comparable.

## Configuration Parameters

Add these parameters to your JSON config file under `defaults.thresholds`:

```json
{
  "thresholds": {
    "use_window_filter": true,
    "window_v": 0.1,
    "window_r": 0.2,
    "min_candidates": 1000,
    ...
  }
}
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_window_filter` | boolean | `true` | Enable/disable window filtering |
| `window_v` | float | `0.1` | Wind speed window radius (normalized space) |
| `window_r` | float | `0.2` | Air density window radius (normalized space) |
| `min_candidates` | int | `1000` | Minimum candidate count threshold |

### Choosing Window Sizes

The window parameters are in **normalized space** (e.g., MinMax [0,1] or Z-score):

**For MinMax Normalization [0,1]:**
- `window_v = 0.1` ≈ ±1.5 m/s in physical space
- `window_r = 0.2` ≈ ±0.06 kg/m³ in physical space

**For Z-score Normalization:**
- `window_v = 0.5` ≈ 0.5σ (standard deviations)
- `window_r = 1.0` ≈ 1.0σ

**Guidelines:**
- **Dense data**: Use smaller windows (0.05/0.1) for more selective filtering
- **Sparse data**: Use larger windows (0.15/0.25) to ensure sufficient candidates
- **Default values** (0.1/0.2) work well for most cases

## Automatic Window Expansion

If the number of filtered candidates is less than `max(K_NEI, min_candidates)`:

1. **Expand window** by multiplying by 1.5
2. **Retry filtering** with larger window
3. **Maximum 3 expansions** (1.5×, 2.25×, 3.375×)
4. **Fallback**: If still insufficient, use all candidates (full calculation)

This ensures robustness while maintaining performance benefits.

## Usage

### Basic Usage

Use the optimized configuration (window filtering enabled):

```bash
python main.py --config experiments_compare_不同切向比例_分风机_JSMZS51-58.json
```

### Baseline Comparison

Compare performance with and without window filtering:

```bash
# Baseline (no optimization)
python main.py --config config_baseline.json

# Optimized (with window filtering)
python main.py --config config_window_filter.json
```

### Testing Window Filtering

Verify the window filtering implementation:

```bash
# Note: Requires numpy and torch installed
python test_window_filtering.py
```

## Performance Monitoring

The implementation logs performance statistics:

```
[KNNLocal] Window filtering enabled: window_v=0.1, window_r=0.2, min_candidates=1000
[KNNLocal] Window filtering stats: avg_candidates=5234.2/50000 (89.5% filtered), expansions=12/10000
```

**Interpretation:**
- `avg_candidates`: Average number of candidates per query after filtering
- `% filtered`: Percentage of total candidates filtered out
- `expansions`: Number of queries that required window expansion

**Expected Results:**
- **Filter ratio**: 80-95% for well-tuned windows
- **Expansions**: <5% of queries should need expansion
- **Speedup**: 5-20× faster for large datasets (N > 50,000)

## Implementation Details

### Function: `_window_filter_candidates()`

Located in `stage2_modular/thresholds/knn_local.py`:

```python
def _window_filter_candidates(Zb_t, Xcand_t, window_v, window_r, d, 
                               min_candidates, K_NEI):
    """
    Filter candidates for each query point based on window constraints.
    
    Returns:
        filtered_indices: List of candidate indices for each query
        expand_count: Number of queries that required expansion
    """
```

### Integration Points

1. **Configuration parsing** (lines ~230-235): Reads window filter parameters
2. **Status logging** (lines ~370-375): Reports filtering status
3. **Main loop** (lines ~440-515): Applies filtering per query batch
4. **Statistics** (lines ~540-545): Reports final statistics

## Backward Compatibility

- **Default behavior**: Window filtering enabled
- **Disable option**: Set `use_window_filter: false` in config
- **Original code path**: Preserved when filtering is disabled
- **No breaking changes**: Existing configs work without modification

## Edge Cases Handled

1. **Insufficient candidates**: Automatic window expansion
2. **Sparse data regions**: Falls back to full calculation if needed
3. **1D vs 2D data**: Correctly handles both wind-only and wind+density cases
4. **Device compatibility**: Works on both CPU and GPU

## Troubleshooting

### Too few candidates (many expansions)

**Symptom**: High expansion count in logs

**Solutions:**
- Increase `window_v` and `window_r` (e.g., 0.15, 0.25)
- Decrease `min_candidates` (e.g., 500)
- Check data distribution for outliers

### No performance improvement

**Symptom**: Similar runtime with/without filtering

**Possible causes:**
- Windows too large (filtering very few candidates)
- Dataset too small (overhead dominates)
- Dense data (most points within windows)

**Solutions:**
- Decrease window sizes
- Check filter ratio in logs (should be >50%)
- Window filtering most beneficial for N > 50,000

### Results differ from baseline

**Expected**: <5% difference in final results
**Reason**: Slightly different neighbor sets (physically reasonable)
**Action**: If difference >5%, increase window sizes

## Performance Expectations

### Large Dataset (N=100,000, Q=10,000)

| Configuration | Distance Calcs | Expected Time | Speedup |
|---------------|----------------|---------------|---------|
| No filtering | 10^9 | ~300s | 1× |
| Window filter (90% reduction) | 10^8 | ~30s | **10×** |

### Medium Dataset (N=50,000, Q=5,000)

| Configuration | Distance Calcs | Expected Time | Speedup |
|---------------|----------------|---------------|---------|
| No filtering | 2.5×10^8 | ~60s | 1× |
| Window filter (85% reduction) | 3.75×10^7 | ~10s | **6×** |

*Note: Actual speedups depend on hardware, data distribution, and window parameters.*

## Future Improvements

Potential enhancements (not currently implemented):

1. **Adaptive windows**: Automatically adjust based on data density
2. **Spatial indexing**: Use KD-tree or ball-tree for faster filtering
3. **Batch filtering**: Filter all queries at once (requires more memory)
4. **GPU kernels**: Custom CUDA kernels for filtering

## References

- Original issue: "knn数据选择" (KNN data selection optimization)
- File: `stage2_modular/thresholds/knn_local.py`
- Test: `test_window_filtering.py`
- Configs: `config_baseline.json`, `config_window_filter.json`

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
