# Enzora 使用手册

本手册对应当前正式网站 https://enzora-agent.com/ 。英文可编辑版本为 `pptx/Enzora_User_Guide.pptx`。网页分为建库、评分、结果三屏；滚动只切换视图，计算由明确的执行按钮启动。

## 1. 打开工作区

访问正式网站并按提示输入工作区访问码。访问码用于登录，不是 MiniMax API Key。浏览器不需要也不会接收模型权重或服务端 API Key。

点击 Demo 可以离线体验。默认示例固定为项目使用的 544 aa 碳酸酐酶，底物为二氧化碳，SMILES 为 `O=C=O`。Demo 的预测分数和投影是模拟数据，正式运行的模型输出与 Demo 分开保存。

## 2. 准备酶与底物

在 Input 中输入实验名称、完整蛋白序列与底物 SMILES。当前联合流程接受 1–1,000 个标准氨基酸残基。Phenom 的活性预测依赖酶–底物配对；只输入目标 pH 不能替代底物。

通过 FASTA / DNA / Database import 可以导入文件、解析核酸或检索蛋白。DNA/RNA 需要先翻译，再核对蛋白。逆折叠模型使用的结构与链必须匹配 WT；更换起始酶时，请先核对管理员配置的结构映射。

文件入口支持 FASTA、CSV、BioArtifact JSON、GenBank 与包含显式序列的 SBOL，单文件最多 100 条记录。CSV 的序列字段使用 `id,name,type,sequence`；JSON schema 为 `enzora.bioartifact/1`。SBOL 入口处理序列，不解析完整电路装配。解析后选择本次要运行的蛋白。

## 3. 设置候选库

Settings 中设置文库大小、突变阶数和最终 Top-K。默认生成 1,000 条单点候选并最终选择五条。双点突变改变两个不同位置。高级请求中的允许/排除位点、各突变阶数配额需要在参数预览中确认，配额之和必须等于文库大小。

VESM 使用 ProteinDPO、EnzGFM 与 VESM3 的单点评分。模型分数校准后取最低值进行保守集成，再按配额生成候选。组合设计使用不同位点的单点信息，不代表已经测得位点间的相互作用。

## 4. 建库与 LLR 图

点击 Build library 开始建库。圆环显示实际保存的序列数量；提交成功、排队、模型计算与保存完成是不同状态。

LLR 曲线展示已保存替换在三个生成模型中的校准评分分布。上方色条将每个替换归因给最低分模型，平局平分。五秒动画是已保存数据的回放，不是推理计时。图表已放大，整屏查看按钮可进一步放大，再点击关闭整屏视图返回。

## 5. 启动评分

建库完成后滚动至第二屏，点击 Start Screening。Phenom 返回预测活性及原生 Gaussian 标准差；EpHod 返回最适 pH 与模型分歧；UniStab 返回相对 WT 的稳定性变化与回归头采样变异。曲线和均值对应实际返回的预测，模型失败不等于零分。

## 6. 偏好与可信度

Preference 表示你有多看重该性质。Confidence 控制该模型对本轮排序的影响。二者相乘并归一化得到三个有效权重，权重和为一。pH 权重大于零时必须设置目标 pH。

MiniMax 驱动的 Enzora Assistant 会读取预测和不确定性摘要，通过 `balance_scores` 提交受校验的调整。也可以手动修改面板并点击 Apply balance；检查状态是否为 Applied。可信度是控制参数，不是实验成功概率。

## 7. 理解综合分

当前默认使用冻结的 WT 参考评分。活性增益为候选与 WT 的预测差；pH 增益为到目标距离的缩小；稳定性增益遵循当前 ΔΔG 越低越好的方向。第一批评分确定尺度并写入 `scoring_reference.json`。之后复用同一参考与策略时，不会因加入极端候选而把旧分数重新拉伸。

不确定性使候选的增益或损失向中性分数 0.5 收缩。Phenom sigma、EpHod 分歧与 UniStab MC-dropout 标准差来源不同，并未统一校准为概率。最终分数按实际有效权重加和。硬阈值与排名分别判断，高排名不代表达到目标。

贡献图中每一行代表一条候选，色带宽度代表该模型对其分数的加权贡献。可整屏查看完整曲线。当前网页执行单轮三阶段任务；跨轮实验需要复用导出候选和冻结的参考策略。

## 8. 使用 Assistant

拖动窗口标题可移动 Assistant，拖动边缘可调整大小。对话保留最近上下文，运行中显示实际工具步骤与耗时。

可以提出具体请求：

- “解释当前第一名候选的评分依据。”
- “从 ESM Atlas 查询 UniProt P01308，并给出来源和序列。”
- “从输入蛋白截取第 25–300 位残基。”
- “按枯草芽孢杆菌偏好导出 WT 与入选候选的编码 DNA。”

区域剪切必须提供明确的一起始、一终止位点；工具不自行预测成熟肽切割位置。Atlas 检索返回来源、序列、SAE 特征及已有预测结构；特征相似性不等于序列一致性。

## 9. 结构与工具页

Model&Tools 可以独立执行数据库检索、核酸翻译、明确区域提取、结构查看和密码子转换，不必启动完整突变流程。输入后选择操作，核对返回卡片、来源与序列。

结构查看显示已有或用户提供的 WT 结构，可旋转、缩放和切换表示。ESMFold2 是单独的外部预测请求，需要管理员配置 Biohub Token 并由用户明确提出。未配置的工具会禁用；当前不把 Rosetta 当作已部署工具。

## 10. Top-K 与 UMAP

评分完成后进入结果屏，点击 Select Top-K & generate report。此按钮启动最终选择、UMAP 与报告。点击 Top-K 节点可检查序列，深蓝色残基表示相对初始 WT 的替换。地图支持拖动、滚轮与按钮缩放。

UMAP 展示序列空间中的相对位置，使用位置氨基酸身份与 Hamming 距离；不能据此直接证明活性或实验多样性。Demo 的投影会明确标记为非真实 UMAP。

## 11. 导出

报告生成后可导出分析文本与结果。格式选择器支持 JSON、CSV、FASTA、GenBank 和 SBOL。需要预测与溯源时保留 JSON；FASTA 主要保存序列身份，不能保留任意预测注释。导出时核对候选 ID、序列长度、替换位点、底物和宿主。

## 12. 宿主编码 DNA

选择表达宿主，点击 Optimize & export CDS，为 WT 和入选候选生成同义编码 DNA。11 种使用表内置离线可用，包括大肠杆菌和枯草芽孢杆菌；未收录宿主走在线补充分支。

当前实现选择宿主偏好的同义密码子。翻译回蛋白后必须与输入一致。它不联合优化 GC、重复序列、限制性酶切位点或 mRNA 结构，订购前仍需检查合成要求。

## 13. 本地启动

从项目根目录执行，科学模型使用独立配置的环境：

```bash
git clone https://github.com/YuzhangLi2281/Enzora.git
cd Enzora
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/Enzora_Software_Webs/utils/server/requirements.txt
npm --prefix src/Enzora_Software_Webs/web ci
```

编辑 `config/server.yaml` 和 `config/mutation.yaml` 中实际模型路径、解释器与权重目录。MiniMax 配置使用 `MiniMax-M2.5` 和 `https://api.minimax.cn/v1`，私下提供 `MINIMAX_API_KEY`。工作区访问码由 `ENZORA_ACCESS_TOKEN` 配置；可选折叠工具另用 `BIOHUB_API_TOKEN`。

```bash
python src/Enzora_Software_Webs/utils/server/run.py --host 127.0.0.1 --port 8000
```

在第二个终端启动前端：

```bash
VITE_PROXY_TARGET=http://127.0.0.1:8000 npm --prefix src/Enzora_Software_Webs/web run dev
```

打开 Vite 输出的地址。附加 `?demo=1#/workbench` 可体验离线 Demo。静态部署只在 `web/public/config.js` 配置公共 API 地址，不能放入私钥。

## 14. 模型资源与缓存

Phenom 需要发布的 `model.pt`、`config.json`、`degree.pt` 与匹配的 ESMC/MolT5 资源。VESM3 先加载 ESM3 基础权重，再覆盖微调层。EpHod 和 UniStab 需要各自模型文件与对应依赖。Enzora 仓库不包含所有权重；从各原项目按其许可获取。

Slurm 部署将模型任务与轻量 API 分开调度。共享权重、输入、预测与临时缓存分开存放。成功、失败和取消后清理可重建的图缓存、原始 logits 与 tmp；取消后先确认作业已结束。预测、报告和审计记录继续保留。

## 15. 常见问题

访问码错误时检查登录凭据；长时间排队时查看现有任务，不重复提交。模型失败时检查该阶段日志、环境、资源路径与输入合同。Assistant 最终解释可能慢于工具执行，应查看真实步骤，而不是把“正在解释”当作模型尚未计算。

报告未生成时导出按钮不会提供空的成功报告。没有候选达标时，检查约束与设计方向，可以另建一轮任务；不能把第一名自动称为达标。

## 16. 验证边界

目前的真实端到端验收使用 544 aa WT、CO2、10 条单点候选和 Top-5。三生成模型、三评分器、不确定性、MiniMax 原生配平、UMAP 和报告均已完成，且独立复算了冻结参考分数与实际有效权重。该记录证明集成链路能够运行，不替代 1,000 条规模性能或湿实验验证。
