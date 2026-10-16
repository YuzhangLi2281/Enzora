# Phenom 最终版本

Phenom 基于蛋白质序列与底物分子图，通过残基–口袋、原子–motif 分层图网络及双向注意力，精心构造可解释的酶–底物交互指纹。原生 Gaussian head 同时输出预测均值与标准差，生产接口预测 `log10(kcat)` 和 `log10(Km)`。注意力与池化权重用于检查模型关注的残基和化学片段，不等同于实验验证的催化机制；原生不确定性不等于已校准的置信区间。

模型类与架构标识统一为 `Phenom`；训练、推理和导出的 `config.json` 必须使用相同标识。已有权重张量键和尺寸保持不变；转移既有 checkpoint 时需同步更新配置标识，不提供旧名称的静默兼容。

`models/net.py` 已只保留 `Phenom` 生产类的构造、forward 和所需 RBF 工具；
`models/layers.py` 与 `models/pooling.py` 也只保留该 forward 实际依赖的层。
旧优化器、未调用的聚类/交互层、旧图连通性工具及实验架构分支均已移除。
`inference/` 在命令行和函数边界都会拒绝 kcat/Km 之外的任务，避免误用
checkpoint 中为张量兼容而保留、但未经最终校准的其他维度。

发布接口只有两层：命令行 `inference/inference.py` 接收包含 `Protein`、`Ligand`
的 CSV，输出 `inference_results.csv`；Python 接口 `load_model(model_dir, device)`
和 `predict_frame(frame, *, task, model, protein_cache_dir, ligand_cache_dir, ...)`
返回带预测列的 DataFrame。`task` 只能是 `kcat` 或 `km`；输出中的 `mu_log10`
可用于排序，`prediction` 是 `10**mu_log10`，`nominal_*_95` 不是校准置信区间。
两种接口均要求预计算的 ESMC 蛋白缓存及配体缓存。训练入口仍保留六维输出布局
和历史辅助任务评测能力以兼容 checkpoint，但这些任务不是最终发布模型的推理接口。

代码入口按职责区分：`inference/inference.py` 提供唯一的生产预测命令及
`load_model`、`predict_frame` 两个可调用接口；`train/main.py` 是 模型训练和
checkpoint 评估入口；`scripts/train.slurm`、`scripts/evaluate.slurm` 是对应的
集群参数模板；`train/build_esmc_cache.py` 与
`train/build_ligand_fingerprints.py` 负责生成推理所需缓存；
`Phenome_Evaluation/phenome_evaluation.py` 负责统一指标计算。
`models/` 仅保留 模型前向计算实际调用的层。已删除没有任何生产引用的旧
virtual-screening 指标、旧 ligand encoder、失效的 molecule/dict loader、
注释占位模块和对应辅助函数；保留历史训练数据和评测文件以便复核，
不把数据清理误当作代码清理。
本次进一步移除了预测头中无效的 `task_index`/`all_task_outputs` 参数，以及主干
只累积、不参与预测或解释性输出的两个中间列表；这不改变 checkpoint 参数或预测。
生成的 Python 字节码缓存已清理，后续导入时可自动重建。
`utils/protein_init.py` 只提供最终缓存使用的 33 维氨基酸特征，不再加载旧 ESM2；
清理后与 20 条现有 ESMC 缓存的特征逐元素完全一致。ligand 图编码也与现有缓存
逐张量一致，保证代码整理不改变发布 checkpoint 的输入语义。

## 1. 最终模型文件

```text
ckpts/
├── model.pt             # kcat 最优版本的唯一模型权重
├── config.json          # 固定 Phenom 架构和 train-only scaler
├── degree.pt            # PNA degree histogram
└── selection.json       # 选模协议
```

生产目录只保留导出的
`model.pt`，不保留 optimizer、scheduler 或训练中间状态。

模型模块按职责划分：`net.py` 定义 `Phenom`；`interaction_layer.py` 包含 `PhenomInteraction`、`InteractionLayer`、`AttentionConv`、`CatalyticInteraction` 和 `SuperNode2TokenConv`；`pooling.py` 统一管理 `MotifPool` 与 `dense_mincut_pool`；`enzyme_prediction.py` 提供原生 Gaussian head。

## 2. 固定架构

架构标识：`Phenom`。

### 2.1 蛋白质表示

- ESMC-600M 预计算 1152 维 residue representation；
- contact graph 同样由 ESMC-600M 最后四层 attention 预计算，经对称化、APC 和 residue top-16 构图；
- 叠加 33 维氨基酸理化特征；
- 最长保留 1000 个 residue（前 500 + 后 500）。

因此，训练与推理中的 sequence information 和 contact map 均来自 ESMC-600M，不读取 ESM2 contact 或 EnzGFM residue embedding。

### 2.2 配体表示

- 标准化 SMILES 的 atom/bond 图；
- motif/clique 图；
- MACCS-167 + Morgan-2048 指纹；
- 预计算 MolT5 768 维 ligand language embedding。

OmniESI 记录保留完整反应物侧 SMILES；EnzGFM 只提供单底物时不虚构辅因子。

### 2.3 分层图网络与催化交互指纹

采用三层 PNA、蛋白 min-cut pooling、配体 motif pooling 和双向 pocket–motif attention。关键参数如下：

| 参数 | 数值 |
|---|---:|
| hidden channels | 200 |
| PNA layers | 3 |
| protein clusters | 5 / 10 / 20 |
| attention heads | 5 |
| attention dropout | 0.2 |

催化交互使用 ligand-guided catalytic interaction：根据配体全局表示和 task embedding 对 residue 打分，选择 top-64 residue，依次执行 residue→atom 与 atom→residue 多头 cross-attention，再把催化交互残差写回 400 维融合表示。该残差与分层图网络共同形成酶–底物交互指纹。

### 2.4 独立高斯头

模型保留六维 task-independent 高斯输出布局，以保证 checkpoint 张量布局稳定；最终训练只覆盖第 0、1 维。这里的训练不意味着 `sigma` 已完成概率校准：

```text
dim 0: log10(kcat)
dim 1: log10(Km)
```

每个任务使用独立 adapter、mu head 和 sigma head。均值无输出范围约束；sigma 使用有界 sigmoid：

```text
sigma = 1.99 * sigmoid(raw_sigma) + 0.01
```

sigma head 输入 `z.detach()`，其损失不向 fusion/backbone 反传。最终排序使用原始 `mu`，不把 sigma 与 mu 固定融合。

## 3. 数据划分与防泄漏

最终数据位于 `training_data/v2/`：

```text
kinetic_specialization.csv        # fold 0-7 + 去重额外训练数据
validation.csv                    # EnzGFM fold 8
kinetic_specialization_final.csv  # fold 0-8 + 去重额外训练数据，用于定步 refit
test.csv                          # 固定 benchmark，动力学部分为 fold 9
kinetic_scalers.json              # 仅训练集统计量
```

划分协议：

1. fold 0–7 训练，fold 8 选择训练步数；
2. 选定 `4000` step 后，将 fold 8 加入训练并从同一底座定步 refit；
3. fold 9 未用于最终 refit 的 checkpoint 步数选择或预测融合，但项目此前曾参考 fold 9 指标迭代架构；因此它是固定 benchmark，不应宣称为全研发过程未触碰的一次性独立测试；
4. 额外 CatPred/OmniESI 数据删除与 validation/test 重合的 protein–ligand pair；
5. 重复 `(Protein, Ligand, task)` 测量取 median，`sample_weight=clip(1/(1+MAD), 0.5, 1.0)`；
6. kcat/Km batch 严格 50:50，batch 内采用标签四分位、同 ligand/异 protein、同 protein/异 ligand 的排序友好采样。

train-only robust scaling：

| 任务 | median | IQR |
|---|---:|---:|
| kcat | 0.6989700043 | 1.9583247348 |
| Km | -0.7447274950 | 1.6795075246 |

推理后还原到原始 log10 单位。Spearman 在还原前后不变。

## 4. 训练实施

最终 refit 使用：

| 配置 | 数值 |
|---|---:|
| 总步数 | 4000 |
| batch size | 24 |
| kcat/Km 配额 | 50% / 50% |
| learning rate | 1e-6 |
| warmup | 500 step |
| head-only | 1000 step |
| mean loss | Smooth L1 + ranking loss |
| min-cut / orthogonality | 0 / 0 |
| checkpoint / evaluation | step 4000 |

图辅助损失在本次 specialization 中关闭，因为 validation 没有证明其能提高 Spearman。最终模型不是从 scratch 训练，而是加载已有多任务 backbone 后进行 kinetic specialization。

训练入口固定为 Phenom，不再暴露其他架构开关：

```bash
/home/ext-yuzli/nd32_scratch/miniconda3/envs/phenome/bin/python train/main.py \
  --train-csv training_data/v2/kinetic_specialization.csv \
  --validation-csv training_data/v2/validation.csv \
  --test-csv training_data/v2/test.csv \
  --scaler-json training_data/v2/kinetic_scalers.json \
  --output-dir runs/phenom \
  --init-checkpoint /path/to/multitask_backbone_checkpoint.pt \
  --degree-path ckpts/degree.pt \
  --protein-cache-dir ../data/Cache_ESMC_600M \
  --ligand-cache-dir ../data/Ligand_Cache \
  --task-weights kcat=50,km=50 \
  --total-steps 4000 \
  --head-only-steps 1000 \
  --warmup-steps 500 \
  --training-phase mean \
  --exclude-task kcat_km --exclude-task ph --exclude-task esp --exclude-task mutation \
  --prediction-head-layout task_independent
```

## 5. Fold 9 最终结果

统一测试集：`data/Test/kinetic_prediction/test.csv`。所有数值来自单个 Phenom checkpoint 的原始预测。

| 任务 | n | Spearman | Pearson | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|---:|
| kcat | 2765 | **0.5317** | 0.5324 | 1.2735 | 0.9554 | 0.2493 |
| Km | 4185 | **0.6032** | 0.6174 | 1.0331 | 0.7943 | 0.3649 |
| macro | 6950 | **0.5674** | 0.5749 | 1.1533 | 0.8748 | 0.3071 |

这组结果替换旧 README 中 stage5 merged 的 `kcat=0.5143`、`Km=0.5914`；旧数值不再代表最终模型。

### 5.1 同样本比较与点估计优势

完整 PHENOM fold-9 集上的 `kcat=0.5317`、`Km=0.6032` 是模型本身的结果，
不能与缺失样本的 baseline 直接相减宣称胜出。四模型共同支持的 kcat 2,759 条、
Km 4,177 条记录重算后，主要指标 Spearman 如下：

| 任务 | n | Phenom | CatPred | CataPro-fold9 | OmniESI |
|---|---:|---:|---:|---:|---:|
| kcat | 2759 | 0.5305 | 0.5167 | 0.4818 | 0.5420 |
| Km | 4177 | 0.6032 | 0.7937 | 0.6091 | 0.7767 |

PHENOM 在共同 kcat 样本上的 Spearman 点估计高于 CatPred，差值为 +0.0138；
行级配对 bootstrap 95% 描述性区间为 `[−0.0282, 0.0539]`，按完全相同
protein ID 重采样的区间为 `[−0.0959, 0.1286]`。两者均跨零，而相同 protein ID
并不等于 40% 同源簇。为进一步检查家族依赖，用 MMseqs2 在固定测试集内部
以 40% identity、双侧 80% coverage 建立连通簇；共同 kcat 样本有 364 个簇，
按簇配对重采样的区间为 `[−0.1035, 0.1388]`，仍跨零，因此**不能声称统计显著胜出**。
这里的测试集内部聚类也不能证明训练与测试之间完全同源隔离。共同样本上的 Pearson、RMSE、
MAE、R² 也优于 CatPred。完整计算记录见
`../materials/FigsScripts/kinetic_common_pair_audit.yaml`。

在双方都覆盖完整对应任务集时，下表仅列 Spearman 点估计优势，不当作显著性结论：

| 任务 | Phenom | 被超过的模型 | baseline | 提升 |
|---|---:|---|---:|---:|
| kcat（共同 2759 条） | 0.5305 | CatPred | 0.5167 | +0.0138 |
| kcat | 0.5317 | CataPro-fold9 | 0.4834 | +0.0483 |
| Km | 0.6032 | UniKP | 0.3721 | +0.2311 |
| macro(kcat, Km) | 0.5674 | CataPro-fold9 | 0.5460 | +0.0214 |
| macro(kcat, Km) | 0.5674 | UniKP | 0.5144 | +0.0530 |

Phenom 的 kcat Spearman 在共同样本上没有超过 OmniESI，
完整集上也没有超过 DLKcat 或 UniKP；Km 没有超过 CatPred、OmniESI 或
CataPro。README 不作 SOTA 声明。PHENOM 对 OmniESI kcat 的 Pearson、RMSE、
MAE、R²更好，但 Spearman 较低，因此只记为次要指标胜出。

## 6. ESP benchmark 状态

最终 Phenom checkpoint 只训练 kcat/Km，不能把旧 ESP head 冒充为 当前模型输出。仓库保留的 PHENOM framework ESP 结果来自 stage5 ESP validation-best head，仅用于项目历史横向比较：

| 模型 | AUROC | AUPRC | Balanced Acc. | F1 | MCC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| PHENOM archived ESP head | 0.8924 | 0.8129 | 0.8082 | 0.7244 | 0.6278 | 0.1155 |
| ESP | 0.9480 | 0.9035 | 0.8718 | 0.8146 | 0.7476 | 0.0719 |
| OmniESI | 0.9667 | 0.9406 | 0.8988 | 0.8740 | 0.8358 | 0.0557 |

结论：当前公平 ESP benchmark 上，PHENOM 没有超过 ESP 或 OmniESI，胜出列表为空。`ESP-production` 因无法排除训练重叠，只是非公平上界，不进入结果比较。

论文主文目前聚焦于 kcat 的公平比较；Km 的较弱结果完整保留在 `Phenom_Evaluation_Results/final.yaml` 和补充图，不能隐去。上述 ESP 与下面的 mutation 优势属于早期多任务 head，不能宣称由最终动力学 checkpoint 同时取得。

## 7. Mutation 和 pH benchmark

最终 Phenom checkpoint 没有训练 mutation 或 pH；以下结果来自 PHENOM framework 的历史专项 head，不能标成 当前 checkpoint 的能力。

Mutation 在 UniKP 支持的同一 10,411 条测试样本上存在一项明确胜出。双方逐行使用同一份 `supported_test.csv`，不存在通过改变测试样本制造优势的问题。

| Mutation 指标 | PHENOM | UniKP | 胜者 |
|---|---:|---:|---|
| AUROC | **0.9093** | 0.6255 | PHENOM |
| AUPRC | **0.2582** | 0.0166 | PHENOM |
| Balanced Accuracy | **0.6722** | 0.5877 | PHENOM |
| F1 | **0.2727** | 0.0285 | PHENOM |
| MCC | **0.2706** | 0.0377 | PHENOM |
| Brier | **0.0171** | 0.2627 | PHENOM |
| ECE-10 | **0.0173** | 0.5006 | PHENOM |

该测试集只有 117/10,411 个正样本，因此 accuracy 不作为主要结论。UniKP recall 更高（0.7692 对 0.3590），但以大量假阳性为代价；PHENOM 在 AUROC、AUPRC、特异性和校准上明显更好。PHENOM 在 OmniESI 共同支持的 942 条样本上未胜出，因此不列入优势。

pH 固定 1,970 条测试集上，PHENOM Spearman 为 0.3076，EpHod 为 0.5709；PHENOM 未超过 EpHod。

完整、机器可读的最终比较见 `../Phenom_Evaluation_Results/final.yaml`。

## 8. 推理

输入 CSV 至少包含 `Protein` 和 `Ligand`：

```bash
/home/ext-yuzli/nd32_scratch/miniconda3/envs/phenome/bin/python inference/inference.py \
  --input-file input.csv \
  --output-folder output \
  --model-dir ckpts \
  --protein-cache-dir ../data/Cache_ESMC_600M \
  --ligand-cache-dir ../data/Ligand_Cache \
  --task kcat \
  --device cuda:0
```

`--task` 对最终模型只应使用 `kcat` 或 `km`。输出均值已还原到原始 log10 单位，并同时给出 sigma。
Python 调用者可以先用 `load_model(Path("ckpts"), "cpu")` 加载一次模型，
再用 `predict_frame(frame, task="kcat", model=model,
protein_cache_dir=..., ligand_cache_dir=..., device="cpu")` 对多个输入表复用模型。
`predict_frame` 返回原始输入列及预测列，空表或非 kcat/Km 任务会明确报错；
不需要通过临时 CSV 调用命令行入口。
动力学输出中的 `nominal_lower_95`、`nominal_upper_95` 按
`10 ** (mu ± 1.96 * sigma)` 计算，仅表示高斯假设下的名义范围，并不是经过
fold-8 coverage calibration 的置信区间；`nominal_factor_95 = 10 ** (1.96 * sigma)`
直接显示该范围相对点预测的乘除倍数。排序和 agent 综合分使用 `mu_log10`，sigma
只作为相对可靠度，不使用区间上界提高候选排名。
固定 fold-9 的事后诊断显示，`sigma` 与绝对预测误差的 Spearman 仅为
`−0.034`（kcat）和 `+0.029`（Km）。因此相对可靠度目前只是待检验的决策
启发式，不能据此声称候选级误差得到可靠校准。名义 95% 高斯范围在 fold-9
上的实际覆盖率约 99%，表明范围偏宽。诊断代码与完整分位数结果见
`../materials/FigsScripts/diagnose_uncertainty_and_policy.py` 和相应 YAML。

## 9. 结果文件

- 最终汇总：`../Phenom_Evaluation_Results/final.yaml`
- Phenom kinetic：`../Phenom_Evaluation_Results/kinetic/PHENOM/metrics.yaml`
- 评测协议：`../Phenom_Evaluation_Results/kinetic/PHENOM/protocol.yaml`

上述结果文件与 checkpoint 是发布产物；`training_data/v2/` 的其他文件保留作
历史实验和数据划分复核，不是推理接口的必需输入。旧版辅助任务分割、40% 聚类
实验和多 checkpoint 选模脚本已移到 `../.legacy_archive/Phenom_pre_v2/`，可复核但
不再出现在 Phenom 生产代码路径中；它们不代表当前生产 checkpoint 的输出能力。

最终发布前可运行完整模型目录审计：

```bash
/home/ext-yuzli/nd32_scratch/miniconda3/envs/phenome/bin/python scripts/audit_final.py
```

该审计会检查 checkpoint 目录唯一性、Phenom 架构与类标识、训练任务、选模步数、
严格权重加载，以及推理任务边界。

## 10. 研究性可解释性诊断（不属于生产推理接口）

在冻结的最终 Phenom 交互向量上，我们从 fold 0–7 的 24,000 条 kcat 样本训练了 TopK 稀疏自编码器，并将 fold 8 的 2,765 对用于 SAE 重构检查。当前展示的 **6,400 个潜变量 / TopK-128** 配置在 fold 8 的重构 R² 为 **0.9193**，182/6,400（2.84%）个潜变量未激活。训练折选定的单潜变量对芳香、含磷、含硫底物的 fold-8 配对 AUROC 分别为 **0.7618、0.6917、0.8153**；将重复蛋白–底物对按底物汇总为 **738 个独特底物**后，AUROC 为 **0.7648、0.6433、0.7676**。图中的含硫示例是在三种已检查概念中事后选择展示，不能视为预注册发现。先前的 1,600/TopK-64 配置仍保留作历史对照：R² 0.8915、5.81% 潜变量未激活，芳香/含磷/含硫配对 AUROC 0.7041/0.6527/0.5514。64 维 PCA 的重构 R² 0.8697，但与过完备 SAE 不等参数匹配。此诊断只说明部分配体化学信息可由稀疏特征表示，**没有证明催化位点机制**；最终 PHENOM checkpoint 定步 refit 时已经使用 fold 8，所以这里的 fold 8 仅相对于 SAE 训练留出。

碳酸酐酶 CsoS3 的真实模型残基池化权重已映射到现有**预测** CIF 的 PyMOL 风格 cartoon。设计输入比文献天然序列多 30 个 N 端残基，文献的 Cys173/His242/Cys253 锌结合位点对应输入 Cys203/His272/Cys283。展示的结构核心有 514 位，其中距推定 Zn 中心 10 Å 内的 44 位占核心注意力质量 9.91%，与其 8.56% 的残基比例接近；独立 top-64 选择在核心中有 63 位，**0 位**落在该 shell。另对第一轮 VESM 选出的 1,000 个单点突变进行描述性分析，覆盖 319 个位置；位点距离与位置中位预测 Δlog10(kcat) 的 Spearman 仅为 **−0.0848**。结构、注意力和动力学变化均是计算结果，不是实测位点机制。图和逐点数据见 `../materials/FigsScripts/` 中的 `supplement_carbonic_anhydrase_case.svg` 与 `source_data/`；完整统计见 `../materials/FigsScripts/NUMERICAL_HANDOFF.md`。
