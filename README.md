# Enzora

Enzora 是面向酶工程的 Agent 工作流：通过 VESM 构建突变库，使用 PHENOM、EpHod 和 UniStab 分别预测活性、最适 pH 和稳定性，再由 Agent 与用户共同调整评分权重，完成 Top-K 筛选、密码子优化和结果导出。

正式前端：**[enzora-agent.com](https://enzora-agent.com/)**；受保护 API：`https://api.enzora-agent.com/api`。GitHub Pages 承载静态网页，Cloudflare 命名隧道连接 M3 上的 API，推理由 Slurm 分配 GPU。HTTPS、访问码登录、跨域与真实建库/Phenom 已通过验证；其余评分器和最终输出仍按真实全流程联调结果更新。模型权重仅存服务器，本地不下载。

## 一、工程目录

以下路径均相对于 Enzora 根目录。

```text
Enzora/
├── config/                         # 工作流与服务器的 YAML 配置
│   ├── server.yaml                 # 服务端路径、队列、默认参数、Gemini 模型
│   ├── mutation.yaml               # VESM 建库模型、采样与输出配置
│   └── request.example.yaml        # 独立 CLI 请求示例
├── model_weights/                  # 建库与评分所需的本地模型资源
├── resource/                       # 网页素材与工程文档；新增网页图片使用 SVG
│   ├── fonts/                     # 指定 GitHub 图标与许可字体素材
│   ├── models/                    # 三个生成模型的 SVG 概念图
│   └── docs/                      # 设计/部署说明与 AnewOmni 蛋白装饰 SVG
├── src/
│   ├── Enzora_Agent/               # 工作流、模型适配器、工具、配平、报告
│   ├── Enzora_Mutation/            # VESM 建库代码及生成模型
│   ├── Enzora_Screening/
│   │   ├── PheNom/                 # PHENOM 动力学预测与特征预计算
│   │   ├── EpHod/                  # 最适 pH 预测
│   │   └── UniStab/                # 稳定性预测
│   ├── Enzora_Software_Cache/       # 输入、运行产物、模型缓存与日志
│   └── Enzora_Software_Webs/
│       ├── utils/server/           # FastAPI、任务队列、SSE、文件导出
│       ├── utils/plugins/          # 序列翻译等插件
│       └── web/                    # React + TypeScript + Vite 前端
└── README.md
```

运行配置统一放在 `config/*.yaml`，其中的相对路径以配置文件所在目录为基准。模型原生 checkpoint 描述、分词器和前端工具链配置保留其要求的文件格式。

## 二、前端目录与职责

前端按“应用装配 → 业务功能 → 公共能力”组织。页面、组件和业务逻辑放在对应功能目录，修改一个功能时可以直接找到相关文件。

```text
src/Enzora_Software_Webs/web/
├── src/
│   ├── app/
│   │   ├── main.tsx               # React 挂载、路由及 QueryClient 初始化
│   │   ├── App.tsx                # 页面路由与应用布局
│   │   ├── Header.tsx             # 全局导航、语言与演示模式切换
│   │   └── store.ts               # 全局状态与当前实验状态
│   ├── features/
│   │   ├── experiment/            # 实验输入、Agent 对话、参数确认
│   │   │   ├── Wizard.tsx
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── InputFiles.tsx
│   │   │   ├── ReviewParameters.tsx
│   │   │   └── review.ts
│   │   ├── workbench/             # 三段全屏工作台、阶段控制、建库可视化
│   │   │   ├── Workbench.tsx
│   │   │   └── WorkspaceIntro.tsx
│   │   ├── runs/                  # 运行记录、进度订阅与请求恢复
│   │   │   ├── History.tsx
│   │   │   ├── StageTimeline.tsx
│   │   │   ├── useRun.ts
│   │   │   └── requestStorage.ts
│   │   ├── results/               # 分析报告、候选表、序列对照与结果图表
│   │   │   ├── AnalysisPanel.tsx
│   │   │   ├── CandidatePanel.tsx
│   │   │   ├── CandidateTable.tsx
│   │   │   ├── MoleculePanel.tsx
│   │   │   ├── SequenceCompare.tsx
│   │   │   ├── WeightDonut.tsx
│   │   │   ├── Charts.tsx
│   │   │   └── rerank.ts
│   │   ├── tools/Tools.tsx         # 工具目录与工作流规划预览
│   │   └── settings/Settings.tsx   # 连接、模型路径与会话设置
│   ├── shared/
│   │   ├── api/
│   │   │   ├── client.ts          # REST、SSE 与后端调用入口
│   │   │   ├── config.ts          # API 地址及演示模式解析
│   │   │   └── contracts.ts       # 前后端数据类型合同
│   │   ├── demo/                  # 明确标识的演示数据与离线序列工具
│   │   │   ├── backend.ts
│   │   │   └── sequence.ts
│   │   ├── lib/                   # CSV、下载与多语言工具
│   │   └── ui/                    # 公共 UI 基础组件与 Plotly 容器
│   ├── styles/theme.css           # 主题、页面布局与交互样式
│   └── types/                     # Vite、Plotly 的环境及类型声明
├── public/config.js               # 部署时可修改的公开前端连接配置
├── tests/                         # API、演示模式和实验参数回归测试
├── deploy/                        # Nginx、Podman 和服务部署示例
├── Containerfile                  # 前端容器构建入口
├── index.html                     # HTML 入口
├── package.json                   # 依赖与开发命令
├── package-lock.json              # npm 依赖锁定文件
├── tsconfig.json                  # TypeScript 配置
└── vite.config.ts                 # Vite 构建与开发代理配置
```

维护约定：

- 新的页面交互优先放入所属 `features` 目录；公共类型和后端接口分别由 `shared/api/contracts.ts`、`shared/api/client.ts` 管理。
- 业务代码直接导入实际文件，不添加仅用于转发的 `index.ts` 或薄包装。
- `node_modules/` 是依赖目录，`dist/` 是构建产物，TypeScript 增量缓存位于 `node_modules/.cache/typescript/`；这些内容不作为源码提交。
- `public/config.js` 和 `VITE_*` 变量会进入浏览器可见范围，只能保存公开连接信息，不能保存密钥或内部服务器地址。
- 现有源码按职责迁移；本轮同时重做工作台与交互，保留独立的导入向导、工具、设置和运行记录。

## Agent 目录与调用边界

```text
src/Enzora_Agent/
├── cli.py / config.py / workflow.py    # CLI、严格配置与生产全流程
├── diagnostics.py                     # 路径预检、GPU 预检、真实评分器检查
├── adapters/                          # vesm、phenom、ephod、unistab
├── core/
│   ├── runtime.py                     # 子进程、离线环境、CSV/JSON、tryN 目录
│   ├── sequence.py                    # 序列和多点替换解析
│   ├── screening.py                   # 配平、过滤、排序、突变特征与 Pareto
│   └── types.py                       # 候选数据结构
├── planning/
│   ├── models.py                      # 能力目标、步骤和计划
│   └── engine.py                      # 规划、依赖校验与执行
├── outputs/                           # 密码子优化、UMAP/图表、报告
├── llm/                               # Gemini、参数解析、结果分析
└── tools/                             # 工具合同、注册、内置工具与插件发现
```

Agent 从 32 个 Python 文件整理为 28 个。合并运行/IO、预检/验证、排序/突变特征、规划/校验/执行等相关逻辑，模型适配与输出保留各自职责。生产网页任务由 `workflow.py` 执行完整固定流程；`planning/engine.py` 的能力规划用于工具计划和依赖执行，不将规划预览冒充生产任务已运行。计划必须声明真实依赖；不允许无关步骤的产物覆盖依赖步骤的数据。

### 新版界面预览

开发服务器启动后打开：

- **主版本（马卡龙）**：http://127.0.0.1:5173/workbench?demo=1&theme=macaron
- Atlas 蓝黑对照：http://127.0.0.1:5173/workbench?demo=1&theme=deep-space

顶部可切换配色，保留当前任务。首次默认使用马卡龙；明确的 `theme` URL 参数优先于已保存偏好。页面从近白、淡天蓝连续过渡到淡紫，三个阶段共享同一背景，不在屏幕边缘切换色块。英文使用 Arial，中文使用 Noto Sans SC，统一 700 粗体；网络受限时回退至苹方/微软雅黑。图表刻度、说明、输入、按钮与 Assistant 同样使用该字体体系。

点击“载入示例 / Load sample”，再构建突变库。演示默认生成 1000 条唯一单点序列，等待两次明确 Next 才进入后续流程。右侧阶段导航只切换视图，不启动计算。页面先后提供：

- 建库：输入、模型/参数、突变库三个翻转视图；真实模式根据已保存序列更新进度和吞吐动效。最低校准 LLR 模型占比来自真实生成结果，并列均分；它不是因果贡献。未收到归因数据时明确等待，不补造百分比。
- 筛选：三个评分器逐步发布真实预测快照，展示已有分数分布；运行阶段 1–2 可应用偏好 × 可信度。目标 pH、阈值等固定条件在启动前设置。
- 输出：整个库的嵌入图、缩放/拖拽/悬浮、Top-5 箭头、Top-K 全部替换残基深蓝高亮、报告、CSV/FASTA/CDS 导出。

演示图是明确标识的示意投影，**不是 UMAP**；演示分数没有生物学效力，Assistant 演示回复没有调用 Gemini。真实模式只显示后端生成的 UMAP 坐标，不将缺失预测填成分数。已在 Chrome 检查 1440×900、1366×768、768×1024、390×844 布局，并验证阶段推进、配平、聊天、拖动、缩放及 CSV/FASTA/CDS/报告下载。M3 上真实 Gemini 短对话与基于实测预测表的配平工具调用已通过；新鲜 GPU 全流程继续独立验收。

## 三、本地启动与检查

### 后端

在 Enzora 根目录、已激活的 Python 环境中执行：

```bash
python -m pip install -r src/Enzora_Software_Webs/utils/server/requirements.txt
export PYTHONPATH="$PWD/src"
python -m Enzora_Agent.cli check-paths config/request.example.yaml
python src/Enzora_Software_Webs/utils/server/run.py --host 127.0.0.1 --port 8000
```

模型资源未就绪时，`check-paths` 会明确报告缺少的权重、源码或结构文件。各科学模型使用 YAML 中指定的独立 Python 环境；API 主进程不需要安装 torch。

`request.example.yaml` 中的序列、底物和结构路径仅用于展示请求格式，运行真实实验前必须替换为匹配的数据。

### 前端

在 Enzora 根目录执行：

```bash
cd src/Enzora_Software_Webs/web
npm ci
npm run dev
```

开发服务默认使用 `http://localhost:5173`，`/api` 默认代理到本机 `http://127.0.0.1:8000`。可在本地环境配置中设置 `VITE_PROXY_TARGET`；公开部署的 API 地址通过 `public/config.js` 或 `VITE_API_BASE_URL` 配置。

检查与构建命令：

```bash
npm test
npm run typecheck
npm run build
```

构建结果位于 `web/dist/`。字体由 `resource/fonts/` 统一管理，因此容器构建上下文为 **Enzora 根目录**：`podman build -f src/Enzora_Software_Webs/web/Containerfile -t localhost/enzyme-web:v1 .`。根目录 `.containerignore` 排除权重、缓存与科学模型代码；容器构建尚未在本机实际执行。演示模式只用于界面预览，不代表模型已执行。

### Gemini 凭据

在私有服务端环境中配置 `GEMINI_API_KEY`，或通过设置页建立仅保存在服务端内存中的密钥会话。Gemini 模型名称在 YAML 中配置。凭据不能写入请求文件、前端配置、日志或 GitHub Pages。

## 四、科学计算与数据合同

| 参数或结果 | 含义 |
| --- | --- |
| `candidate.top_k` | 完整建库数量，默认 1000 |
| `selection.top_k` | 最终保留数量，默认 5 |
| `mutation.counts` | 各突变阶数的配额；默认 `{1: 1000}` |
| `activity.ligand_smiles` | PHENOM 所需的明确底物 SMILES，必填 |
| `activity_log10` | PHENOM 预测的 `log10(kcat / s⁻¹)` |
| `activity_kcat` | PHENOM 预测的 kcat，单位为 s⁻¹ |
| `activity_sigma_log10` | 模型在 log10 空间中的名义不确定性 |
| `selection.weights` | 用户的 pH、活性和稳定性偏好 |
| `selection.confidence` | 三项评分的显式可信度控制，范围为 [0, 1] |

`mutation.counts: {1: 600, 2: 400}` 表示生成 600 条单点和 400 条双点突变序列，各配额之和必须等于 `candidate.top_k`。无法满足的唯一序列数量会报错，不会自动缩减或重复填充。

允许与排除位点在建库前生效。输入 CIF 链必须与野生型序列完全匹配；当前联合流程支持不超过 1000 个标准氨基酸的蛋白，不截断序列。

活性预测全部由 PHENOM 执行，EnzGFM 仅参与 VESM 建库。PHENOM 发布权重中的内部 `enzgfm_adapter` 张量键保留原有序列化合同，其当前输入为 ESMC 表征。训练表及现有 92 字段合同没有增加旧字段兼容层。

最终权重按 `normalize(weights × confidence)` 计算。可信度控制不等同于已校准的准确率概率，`sigma_log10` 也不应被解释为经实验校准的置信区间。

UMAP 使用完整突变库的位点氨基酸身份特征及 Hamming 距离，产物记录方法、随机种子和 Top-K 标记。同义密码子优化不会改变蛋白序列及其坐标。宿主按用户的明确请求选择，支持 E. coli、Bacillus 等内置宿主。

## 五、三阶段推进与产物管理

网页请求使用 `runtime.phase_control: true`：

1. **突变库构建**：提交后启动 VESM，记录模型进度和实际保存的序列数量。
2. **评分与配平**：建库完成后，通过 Next Process 启动三个评分器，并允许调整偏好和可信度。
3. **筛选与输出**：评分完成后，再次通过 Next Process 启动最终筛选、密码子优化、UMAP 与报告生成。

阶段推进接口为 `POST /api/runs/{id}/advance`，请求体分别为 `{"phase":2}`、`{"phase":3}`。当前阶段尚未完成或推进顺序错误时返回 409。

阶段 1–2 可通过 `POST /api/runs/{id}/balance` 提交完整的 `weights` 和 `confidence`。`/api/chat` 中的 Gemini Assistant 也通过同一套校验执行配平工具。对已完成的任务，可按明确自然语言指令执行 `optimize_codons`，宿主限定为 `e_coli` 或 `b_subtilis`，对现有 Top-K 蛋白反向翻译，返回可导出的 CDS；不会生成或替换蛋白序列。

每个任务包含以下运行文件：

```text
任务目录/
├── request.yaml                   # 本次请求参数，不包含密钥
├── meta.json                      # 状态与阶段记录
├── process.log                    # 主进程输出
└── outputs/
    └── tryN/                      # 单次尝试的预测、配平、序列及报告
```

运行进度与模型日志通过 SSE 传递。最终产物包括预测结果、排序及 Top-K CSV、配平记录、密码子优化 CDS、UMAP JSON、SVG 图表和报告。`keep_temp: false` 会在成功或失败后清理每轮蛋白/配体图缓存、VESM 原始 logits 和任务临时目录；取消运行后也须等调度器确认进程结束再清理。预测表、模型日志与诊断文件始终保留，无效提交不会残留孤立任务目录。

## 六、验证范围与后续工作

已完成的本地验证包括：前端 31 项回归测试、TypeScript 检查、生产构建，以及后端数据合同、配额算法、隔离模型后的流程编排、密码子优化、SVG 图表和实际 UMAP 计算。

M3 部署已完成 Phenom 569 个参数键的严格兼容加载、VESM3 基础权重与末五层微调合同核验，以及 10 条候选的真实三模型建库、Phenom/EpHod/UniStab 三评分器推理。Gemini 已读取本次三模型不确定性证据并实际调用配平工具；这些控制值是可审查的启发式权重，不是校准概率。最终 UMAP 暴露科学环境的依赖冲突，正在修复；报告、完整端到端和 1,000 条流程仍需分别验收，不能用小规模联调或历史回放代替。Slurm 队列、取消、超时、重启接管及缓存清理共 20 项隔离测试通过。

新版工作台保留三段全屏、显式 Next 推进与原生纵向滚动。右侧导航只切换视图；评分器与报告只在用户确认阶段边界后启动。滚轮自由滚动，Next 精确整屏定位且仅滚动主容器，手机和平板展开为纵向布局。装饰图不占文档流、不拦截点击。空地图允许正常滚动；有数据的地图支持从节点拖拽、滚轮/键盘缩放、复位和有界平移，缩放范围为 80%–800%。

评分图显示已有真实预测的计数直方图与区间刻度，不用预设曲线冒充模型输出。配平面板明确区分“未应用修改”“已应用”和“下轮运行设置”，可信度只作为权重系数。Top-K 表格显示综合分、活性、pH 与 ΔΔG，点击表格或全库节点可检查对应蛋白，所有替换位点深蓝高亮。E. coli 与 Bacillus subtilis 可分别生成同义密码子 CDS。浏览器下载的 5 条 Bacillus CDS 已逐一翻译核验，与选中蛋白完全一致。

### 生成模型说明与视觉素材

模型卡可折叠查看文章、官方代码与当前接入边界：

| 模型 | 当前用途与边界 | 原文 |
| --- | --- | --- |
| ProteinDPO | 用实验稳定性偏好对齐的 ESM-IF1 逆折叠模型；输入骨架与序列，参与建库。稳定性改善不保证催化功能保留。 | [Nature Methods](https://www.nature.com/articles/s41592-026-03137-3) |
| EnzGFM | 通用蛋白预训练后进行酶领域适配的 Mamba–Transformer 模型；在本项目仅参与建库。 | [Nature Communications](https://www.nature.com/articles/s41467-026-75283-3) |
| VESM3 | 将 VESM-3B 变异效应知识蒸馏进 ESM3 序列模块，保留多模态能力；当前适配器仅传序列，未启用结构 token。VESM++ 是 VESM3 与 VESM-3B 的组合。 | [Nature Methods](https://www.nature.com/articles/s41592-026-03050-9) |

视觉参考：[AnewOmni 页面](https://anewbt.com/research/anewomni/)与本地 `anewomni.pdf` 的图 1–3。`resource/models/*.svg` 是项目自绘概念图。`resource/docs/anewomni-protein.svg` 改编自 Kong 等 *Programming Biomolecular Interactions with All-Atom Generative Model*，图 1、PDF 第 4 页，[DOI: 10.64898/2026.03.12.711044](https://doi.org/10.64898/2026.03.12.711044)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。改编包含去背景、颜色量化与轮廓矢量化，SVG 无嵌入位图；仅用于装饰，不代表本次候选的结构预测。原作者、来源、许可与改编信息保留在 SVG 描述、页面署名和本节中。

最终仅发布网站的 software 部分，保留其他 wiki 页面；GitHub Pages 仅承载静态前端，科学推理由后端提供。前端已部署正式域名；Software Wiki 的英文稿仍保留本地，按用户要求暂不推送。


## 生物对象交换与服务器评分核验

工具执行器以 `BioArtifact` 传递序列记录、预测与说明，JSON 标识为 `enzora.bioartifact/1`。网页导入支持 FASTA、CSV、JSON、GenBank 和显式序列型 SBOL；导入单个文件后可选择格式导出，API 为 `POST /api/input/export`。CSV 要求 `id,name,type,sequence` 四列。JSON、Enzora CSV/GenBank/SBOL 可保留自身的预测和说明；FASTA 明确提示这些注释会丢失。普通 GenBank 的所有 feature、完整 SBOL 装配/电路语义不在本次转换范围内。

服务器 WT 参考评分已迁移到 `core/screening.py`。`selection.score_policy` 为 null 时使用原有单轮库内评分；提供已冻结的服务器策略对象后，使用相对 WT 的有符号活性/pH/稳定性增益，以及来源明确的不确定性收缩。策略包含 method、target_ph、weights、scales、uncertainty_scales；策略要求活性为主要目标，原始 UniStab ddG 为 lower-is-better。WT 预测由本次真实推理生成，不能替换成示例酶的常量。运行产物 `scoring_reference.json` 保存参考与策略，重排读取同一对象，前端保留冻结分量。

EpHod 使用两子模型分歧作为代理，UniStab 使用仅回归头的 3 次 MC-dropout 标准差，Phenom 使用 Gaussian head sigma。它们不等同于经过实验校准的置信区间。MC 路径已补入本地，模型权重参数名保持不变；尚未执行本地 GPU 端到端推理。真实服务器三轮 3,000 条记录的本地评分回放误差小于 `2e-16`。

三轮工程迭代材料以 [DBTL.md](../DBTL.md) 为准；[图稿目录](../pptx/Fig) 仅保存 SVG 成图，原始数据和协议在 `source_data`。现有网页生产流程仍为单轮三阶段，多轮结果来自服务器已核验实验，不把实验脚本能力冒充网页已完成能力。

## M3 运行部署与服务接续

服务目录按 `releases/`（源码）、`config/`（私有运行配置）、`shared/weights/`（权重映射）、`shared/cache/`（跨任务缓存）、`runtime/`（服务环境）、`logs/`、`run/live/<job_id>/`（独立运行）分开。API Python 环境与科学模型环境隔离。模型解释器必须保留 conda 符号链接路径，不能通过 realpath 改变解释器所属环境。

VESM3 先严格加载 ESM3 完整基础权重，再核对并覆盖末五层微调的精确 50 个键；它不是独立的 50 参数完整模型。残基注释 tokenizer CSV 也必须显式配置为服务器本地资源。EnzGFM 需要匹配的自定义模型源码和 tokenizer；Phenom 的蛋白缓存需要包含 `EsmcModel` 的匹配 ESMC 实现，不能直接换成同名官方接口。当前服务器复用了既有 PHENOM/VESM 的受控源码副本；这些外部源码与权重路径是部署依赖，不能假设本仓库已包含它们。UniStab 同样需要原配套 `vendor/fair_esm`、`vendor/openfold` 与 `vendor/dllogger`，适配器会核对这些依赖并为独立子进程设置搜索路径。

命名隧道凭据与 API/Gemini 凭据仅保存在服务器私有配置，公开静态仓库只包含 API 域名。CPU API 服务与 GPU 推理分开申请 Slurm 资源。服务分配有时限；已配置的接续作业必须等前一服务结束后才启动，避免两个 API manager 同时写同一任务目录。GPU 作业在 API 接续后由持久化作业 ID 接管，不能重新提交冒充恢复。运维停止时须先取消待接续作业，再停止当前服务；固定域名和自动接续均不能保证集群停机期间可用。

每轮缓存采用服务器保留策略：`runtime.keep_temp: false` 时，网页请求不能开启长期调试缓存。任务结束后从 API 内存回收运行 handle、事件队列和任务级 Key 引用；历史信息从磁盘元数据与结果读取，Assistant 也从持久化结果获取上下文。清理不作用于共享权重、tokenizer、公共密码子表或仍运行的作业；`cache_cleanup.json` 记录释放字节数和明确错误。模型子进程退出后由操作系统/Slurm 回收 CPU/GPU 内存，磁盘清理不等同于清空显存。

M3 实测完整 UniStab 推理进程峰值内存约 59 GiB，因此后续 GPU 作业配置申请 64G 系统内存；它与服务 API 的 8G CPU 分配分开。Slurm 模式保留调度器提供的 `CUDA_VISIBLE_DEVICES` 掩码，模型使用逻辑 `cuda:0`，不能用网页 GPU 编号覆盖物理分配。
