# 风电数据清洗——考虑空气密度

> 两阶段风电数据清洗框架，支持空气密度特征引入，含完整消融实验配置。  
> 详细方法和实验设计见 [实验方案报告.md](实验方案报告.md)。

---

## 环境依赖

```bash
pip install torch numpy pandas
```

---

## 运行指南

整个工作流分为两步：**第一步运行清洗流程**，**第二步分析实验结果**。

---

### 第一步：运行 Stage2 清洗流程

```bash
# 消融实验（论文主实验，A→J 共 10 组，A 组必须最先运行）
python main.py --config experiments_paper_ablation.json

# 其他对比实验（切向比例对比）
python main.py --config "experiments_compare_不同切向比例_分风机_JSMZS51-58.json"
```

**输出目录结构（以消融实验为例）：**

```
paper_ablation_runs/
  _splits/                         # 划分缓存（B~J 自动复用 A 的划分，保证公平对比）
  A_rules_only/
    JMZSFD_mlp/
      51号机/
        JMZSFD_51号机_stage2_mlp.csv
      ...
  B_no_rho/
  C_rho_model_only/
  D_rho_clean_only/
  E_rho_both/
  F_rho_shuffle/
  G_rho_constant/
  H_seasonal_winter_test/
  I_seasonal_winter_test_no_rho/
  J_one_pass_rho_both/
```

> **注意**：B~J 组配置了 `reuse_split_from: A_rules_only`，必须在 A 组完成后才能正确加载划分缓存。按 JSON 中的顺序执行即可，`main.py` 会自动处理依赖关系。

---

### 第二步：按风机分析实验结果

Stage2 流程完成后，用 `analyze_stage2_by_turbine.py` 对输出 CSV 做逐风机统计与 MLP 重训练对比。

#### 基本用法

```bash
python analyze_stage2_by_turbine.py --config experiments_paper_ablation.json
```

#### 常用参数示例

```bash
# 完整运行（分析所有 run、所有风机）
python analyze_stage2_by_turbine.py \
    --config experiments_paper_ablation.json \
    --out_turbines turbine_results.csv \
    --out_runs    run_summary.csv

# 只分析某个站点的特定风机（51~58 号）
python analyze_stage2_by_turbine.py \
    --config experiments_paper_ablation.json \
    --station JMZSFD \
    --turbine_range 51 58

# 快速验证（限制 MLP 训练轮数，不改 JSON）
python analyze_stage2_by_turbine.py \
    --config experiments_paper_ablation.json \
    --fast_epochs 20 \
    --debug

# 不使用 split_repo，直接读 CSV 中的 split 列
python analyze_stage2_by_turbine.py \
    --config experiments_paper_ablation.json \
    --no_split_repo
```

#### 全部参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | **必填** | 实验 JSON 配置文件路径 |
| `--pattern` | `**/*_stage2_mlp.csv` | 在各 run 目录下匹配输出文件的 glob 模式 |
| `--station` | 全部 | 只分析指定站点（如 `JMZSFD`） |
| `--turbine_range N M` | 全部 | 只分析编号在 `[N, M]` 范围内的风机 |
| `--out_turbines` | `turbine_results.csv` | 逐风机结果输出路径 |
| `--out_runs` | `run_summary.csv` | 按 run 加权汇总结果输出路径 |
| `--fast_epochs` | 不限 | 临时限制 MLP 训练轮数（调试用，不修改 JSON） |
| `--no_split_repo` | 否（默认使用） | 禁用 split_repo，改为读 CSV 中的 `split` 列 |
| `--debug` | 否 | 打印详细调试信息 |

#### 输出文件

- **`turbine_results.csv`**：每行一台风机×一个 run，包含：
  - `abn_rules`：规则异常率（S_scope==FALSE 占比）
  - `abn_method_scope`：方法异常率（在 S_scope 内）
  - `abn_increment`：方法新增异常率（相对总样本）
  - `abn_union`：规则+方法联合异常率
  - `mlp_v_val_mse`：仅用风速建模的验证集 MSE
  - `mlp_vr_val_mse`：风速+空气密度建模的验证集 MSE
- **`run_summary.csv`**：按 `n_scope`（方法作用域样本数）加权汇总各指标。

---

## 关键配置参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `rho_for_clean` | false | KNN 清洗是否使用空气密度 |
| `rho_for_model` | false | MLP 建模是否使用空气密度 |
| `cleaning_passes` | 2 | 迭代清洗次数（0=仅规则，1=单次，2=两次） |
| `thresholds.lambda_t` | 6.0 | tanorm 度量切向权重 |
| `thresholds.k_nei` | 500 | KNN 邻居数 |
| `split.ratio` | `[0.70, 0.15, 0.15]` | 训练/验证/测试比例 |

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `main.py` | Stage2 清洗流程入口（调用 `stage2_modular/pipeline/orchestrator.py`） |
| `analyze_stage2_by_turbine.py` | 实验结果分析脚本（逐风机统计 + MLP 对比） |
| `experiments_paper_ablation.json` | 论文消融实验配置（A~J 共 10 组） |
| `experiments_compare_不同切向比例_分风机_JSMZS51-58.json` | 切向比例对比实验配置 |
| `stage2_modular/` | 核心模块包（device, scaler, splits, models, thresholds, evaluation） |
| `实验方案报告.md` | 详细实验方案、指标定义、问答与代码使用指引 |
