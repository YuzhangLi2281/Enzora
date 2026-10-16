# PHENOME 统一评估协议

`Phenome_Evaluation` 是 PHENOME 与所有 baseline 的唯一指标计算入口。模型负责
推理并保存逐样本预测；本模块负责读取固定测试集、严格核对样本、计算统一指标，
以及保存可复现的评估协议。

训练途中评估和离线模型评估使用同一个 `phenome_evaluation.py`，不再由
`train/main.py` 维护第二套指标实现。

## 目录结构

```text
Phenom/Phenome_Evaluation/
├── phenome_evaluation.py
├── metrics/
│   ├── kinatic.py
│   ├── ph.py
│   ├── mutation_effects.py
│   └── esp.py
└── Results/
    └── <评估任务>/<模型或训练步骤>/
        ├── predictions.csv
        ├── metrics.yaml
        └── protocol.yaml
```

Baseline 的源码和权重仍存放在 `../../Baselines`。任何模型都不能直接写入其他
模型的结果目录。

## 核心原则

1. 预测文件只提供预测值，不提供可信标签。
2. 标签和 benchmark 元数据只从 `../../data/Test` 读取。
3. `sample_id` 必须与 benchmark 的 `record_id` 一一对应。
4. 缺失预测、额外预测和重复 ID 都是致命协议错误。
5. 二分类 `prediction` 必须是正类概率 `P(y=1)`，方向必须与测试集一致。
6. 固定阈值默认为 `0.5`，禁止使用测试标签选择最佳阈值。
7. 所有结果必须同时保存逐样本预测、指标和协议文件。

## 预测文件格式

标准输入 CSV：

| 列名 | 必需 | 含义 |
|---|---:|---|
| `sample_id` | 是 | benchmark 中原始 `record_id` |
| `task` | 是 | `kcat`、`km`、`kcat_km`、`ph`、`mutation` 或 `esp` |
| `prediction` | 是 | 回归预测值，或二分类正类概率 |
| `sigma` | 否 | 与连续预测目标同尺度的预测标准差 |

PHENOME 训练产生的 `raw_predictions_step_*.csv` 使用：

```bash
--input-format phenome
```

该模式执行以下固定字段映射：

```text
ID                  -> sample_id
selected_prediction -> prediction
selected_sigma      -> sigma
```

新 baseline 应直接输出标准格式，不应依赖 PHENOME 字段映射。

## 注册测试集

| 评估任务 | 固定测试集 | 当前行数 |
|---|---|---:|
| 动力学参数 | `../../data/Test/kinetic_prediction/test.csv` | 9,532 |
| 最适 pH | `../../data/Test/ph_prediction/test.csv` | 1,970 |
| 突变效应 | `../../data/Test/mutation_effect/test.csv` | 23,262 |
| 酶-底物配对 | `../../data/Test/esp_prediction/test.csv` | 12,961 |

突变测试集当前包括：

| 来源 | 行数 | 突变类型 |
|---|---:|---|
| TrpB | 10,411 | 双点 |
| IRED lrDMS | 11,909 | 双点 |
| OmniESI AMP | 788 | 数据源定义的测试突变 |
| OmniESI CTX | 118 | 数据源定义的测试突变 |
| OmniESI mutation classify | 36 | 单点、18 正/18 负 |

因此总体 mutation 指标不等同于旧的 36 条 OmniESI 单点结果。统一 evaluator 会
同时报告总体结果与 `mutation_count` 分层结果。

另有两个独立协议，不与上述四类固定测试集混合：

- CataPro mutant ranking：使用 kinetic benchmark 的 reaction group；
- CataPro Fig. 6 DMS：`../../data/catapro_test/03_Fig6_DMS/dms_benchmark.csv`。

## 指标定义

### 动力学参数

对 `kcat`、`Km`、`kcat/Km` 分别报告，并给出三个分项的等权宏平均：

- 样本数；
- MAE、Median AE、RMSE、R2；
- Pearson 相关系数、Spearman 相关系数；
- 提供 `sigma` 时计算 Gaussian NLL、平均 sigma；
- 50%、90%、95% 预测区间覆盖率。

`parameters` 是主要结果：它分别衡量每类动力学参数内部的预测排序和误差。
`macro_average` 是三个 `parameters` 指标的等权算术平均。协议禁止把三类样本直接
合并后计算相关系数，因为不同参数的标签基线会产生跨任务排序捷径，使 Spearman
虚高。模型比较应报告三个分项结果，并可用 `macro_average` 作为单值汇总。

消融实验可用 `--exclude-task kcat_km` 同时过滤预测文件和 benchmark；此时
`macro_average` 只对剩余的 `kcat`、`Km` 两项计算，排除项记录在 `protocol.yaml`。

当 benchmark 存在 reaction group 时，还报告：

- 每个参数的组内平均 Spearman；
- `N >= 20` 和 `N >= 30` 的 pairwise ranking accuracy；
- 有效 group 数与 comparable pair 数。

### 最适 pH

回归指标与动力学参数一致，并增加：

酸性/碱性二分类：

- 排除真实值恰好为 pH 7 的样本；
- `<7` 为酸性，`>7` 为碱性；
- AUROC、Accuracy、Balanced Accuracy、F1、MCC、specificity；
- TP、TN、FP、FN 与完整 ROC 坐标。

三分类：

- `pH < 5`；
- `5 <= pH <= 9`；
- `pH > 9`；
- Accuracy、Balanced Accuracy、Macro-F1、MCC 和 confusion matrix。

### 突变效应

总体及可用的突变数量分层均报告：

- AUROC、AUPRC；
- Brier score、log loss、10-bin ECE；
- 每个 calibration bin 的样本数、平均概率和真实正例率；
- `classification_at_threshold`：预先声明阈值下的 MCC、Accuracy、Balanced Accuracy、F1；
- precision、recall、specificity、TP、TN、FP、FN；
- 提供 `sigma` 时报告平均 sigma。

默认决策阈值为 `0.5`，具体数值保存在 `classification_at_threshold.threshold`。
其中 `decision_rule` 为 `prediction >= threshold`。mutation 的正类定义为
`mutant_outperforms_wild_type`，即突变体预测 fitness 高于 WT；ESP 的正类定义为
`active_enzyme_substrate_pair`。正类语义同时写入 `protocol.yaml`。
可通过 `--positive-prevalence` 生成明确标记的
`classification_at_predeclared_prevalence` 辅助结果，但它不能替代预声明阈值结果，
也不能使用测试标签选择阈值。

### 酶-底物配对

报告与 mutation 二分类相同的完整排序、分类和校准指标。如果 benchmark 提供
sequence identity，则额外按以下区间分别评估：

```text
<20, 20-40, 40-60, 60-80, >=80
```

## 训练途中评估

`train/main.py` 每到 `eval_every`：

1. 在完整固定 test loader 上做一次推理；
2. 保存五个模型输出头对应的 `raw_predictions_step_<step>.csv`；
3. 直接调用本目录的 `phenome_evaluation.py`；
4. 对 kinetic、pH、mutation effect、ESP 分别执行严格 benchmark 对齐；
5. 将 canonical 结果保存到本目录的 `Results`；
6. 从 canonical `metrics.yaml` 读取关键指标写入 W&B。

训练步骤结果命名规则：

```text
Results/<task>/<run_name>_step_<9位step>/
```

例如：

```text
Results/kinetic/phenome_esmc600m_scratch_full_step_000075000/
Results/ph/phenome_esmc600m_scratch_full_step_000075000/
Results/mutation_effect/phenome_esmc600m_scratch_full_step_000075000/
Results/esp/phenome_esmc600m_scratch_full_step_000075000/
```

训练 run 目录只保留原始推理结果。旧的
`runs/<run>/evaluation/metrics_step_*.json` 属于已废弃的训练内简化指标，不再生成。

## 命令

从本目录执行：

```bash
cd /home/ext-yuzli/nd32_scratch/yuzhangli/PHENOM/Phenom/Phenome_Evaluation
source /home/ext-yuzli/nd32_scratch/miniconda3/bin/activate phenome
```

### 标准 baseline 结果

```bash
python phenome_evaluation.py \
  --task esp \
  --model FusionESP \
  --predictions /path/to/fusionesp_predictions.csv \
  --test-set ../../data/Test/esp_prediction/test.csv
```

### PHENOME checkpoint 结果

```bash
python phenome_evaluation.py \
  --task mutation_effect \
  --model phenome_esmc600m_scratch_full_step_000075000 \
  --input-format phenome \
  --predictions ../runs/phenome_esmc600m_scratch_full/evaluation/05_fitness/raw_predictions_step_000075000.csv \
  --test-set ../../data/Test/mutation_effect/test.csv
```

### CataPro mutant ranking

```bash
python phenome_evaluation.py \
  --task mutant_ranking \
  --model PHENOME_model_name \
  --predictions Results/kinetic/PHENOME_model_name/predictions.csv \
  --test-set ../../data/Test/kinetic_prediction/test.csv
```

### CataPro DMS

```bash
python phenome_evaluation.py \
  --task dms \
  --model PHENOME_model_name \
  --input-format phenome \
  --predictions /path/to/raw_predictions.csv \
  --test-set ../../data/catapro_test/03_Fig6_DMS/dms_benchmark.csv
```

## 输出文件

每次评估固定生成：

```text
predictions.csv  # benchmark 标签、元数据和模型预测的一一对应表
metrics.yaml     # 本协议定义的完整指标
protocol.yaml    # 模型名、任务、输入文件、测试集、行数、阈值和输入格式
```

`protocol.yaml` 是结果可复现性的最低要求。比较两个模型前，应先确认它们使用相同的
benchmark 路径、行数、任务定义和阈值。

## 验证

```bash
python -m py_compile phenome_evaluation.py metrics/*.py
python phenome_evaluation.py --help
```
