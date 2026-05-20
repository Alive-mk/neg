# 生成式大语言模型中的否定盲目性：诊断与多粒度训练修复

---

## 摘要

大语言模型（LLM）存在一种持续性失效模式，我们称之为**否定盲目性（Negation Blindness）**：当提示中明确禁止某一概念时，模型仍频繁将其作为输出。本文做出三项贡献。其一，我们提出 **E4** 基准，当前 v2 实验分割包含 471 条训练记录和 525 条原始测试记录，覆盖三类结构上截然不同的否定现象——目标压制、范围保留与否定排序，实体级 holdout 率 99.6%。其二，我们提出 **MGNM（Multi-Grained Negation Modeling）**，一个基于 LoRA 的多粒度微调框架，通过调用方指定的行为条件 token 与五项互补损失函数联合训练，在单次微调中同时优化压制准确率与范围控制能力。其三，我们在 397 条经过标签审计的 E4 v2 测试子集上报告主要候选排名结果：MGNM 在 Qwen2.5-7B（base）上取得 **NegationFlipAcc（NegFlipAcc）64.2%**、**ScopeCtrl 92.9%**，同时优于所有基线模型（与 NC-SFT 相比：NegFlipAcc +13.4pp，ScopeCtrl +15.3pp），过度否定率（OverNeg 7.1%）低于未经微调的基础模型；在 BoolQ 上，Qwen MGNM 经 PMI 校准达到 87.2%；Llama 则暴露出 No-prior 校准偏移与训练权衡，标准 PMI 后由 69.0% 恢复至 81.4%。相同趋势也出现在 Llama-3.1-8B（base）上，并由 Mistral-7B-Instruct-v0.3 提供部分跨架构验证。上述峰值 E4 结果以正确行为 token 给定为条件；结构化/元数据辅助路由仅作为已知任务结构下的单独部署校准评测，而非纯文本通用自动路由的已解决问题。在 200 条 E4 候选排名探针上的机制分析表明，否定失效在候选排名设置下并非主要由对否定词的注意力缺失引起，而更可能源于否定信号未能被转化为候选层面的排名偏好。

---

## 1. 引言

考虑以下向生成式大语言模型发出的提示：

> *"请说出一个**不属于**鸟类的名词。"*

一个语义理解正确的模型应当输出非鸟类实体。然而，现有 7B 至 70B 参数规模的 LLM 往往输出"麻雀"或"老鹰"——恰恰是被否定明确排除的概念。我们将这一失效模式命名为**否定盲目性**。

否定盲目性并非小概率边缘案例。在现实部署中，否定是表达约束的主要语言手段："不要包含 X"、"避免 Y"、"说明什么**不是**有效方案"。一个无视否定的模型所产生的输出违背用户意图，且这种违背往往隐蔽而难以发现。在医疗指令跟随、法律文档生成、代码安全约束等高风险场景中，此类错误可能带来严重后果。

**为何难以修复？** 两种直觉解法均以失败告终：

- **提示工程**：添加明确警告（"严格遵守否定约束"）或思维链脚手架，并不能改变模型的内部概率分布。我们的实验表明，此类干预不仅无效，反而会降低准确率——因为它们干扰了流畅性，却无法触及表示层的根本缺陷。
- **对比解码（Contrastive Decoding）**：放大专家模型与业余模型的 logit 差值可以提高输出分化程度，但无法区分"应当压制"与"应当保留"的上下文，导致范围控制大幅恶化（−16.4pp ScopeCtrl）。

我们的机制探针表明，在 E4 候选排名设置下，失效并非主要来自模型没有注意到否定词；更常见的问题是，否定信号未能转化为候选层面的排名偏好。因此，单纯的下游 logit 操控很难稳定地区分 suppress、preserve 和 select 三类行为。

**本文方法。** 我们通过有针对性的微调解决否定盲目性。核心洞察是：否定处理并非单一能力，而是三类结构上截然不同的行为：（1）*压制*——明确禁止的概念必须排名最低；（2）*保留*——提示中出现否定词，但否定并不作用于目标概念，正向答案仍然正确；（3）*选择*——存在否定版本的正确答案，应排名最高。将三类行为混同处理（如 Vanilla SFT 和 DPO 所为）会产生相互冲突的训练信号，导致压制能力与范围控制的不可调和的权衡。

MGNM 为每类行为配备专属损失项和调用方指定的行为条件 token，在单次 LoRA 微调中联合优化；本文主结果衡量的是正确行为给定时的条件控制能力。

**本文贡献：**

1. **E4 基准**（§3）：当前 v2 实验分割包含 471 条训练记录和 525 条原始测试记录，涵盖三类否定现象，6 个领域，严格实体与模板 holdout，可支持后续研究。
2. **MGNM**（§4）：五项互补训练目标配合行为条件 token，形成一个条件控制接口；单张 80GB A100 训练时间 15-40 分钟。
3. **全面评测**（§5）：8 条基线、3 种架构、2 项域外任务（WikiFact、BoolQ）、McNemar 显著性检验、数据规模消融。
4. **机制诊断**（§6）：logit-lens、注意力归因与激活补丁实验，定位失效的表示层级根因。

---

## 2. 相关工作

### 2.1 否定在 NLP 与语言模型中的研究

否定长期以来是 NLP 的核心难题，早期工作集中于否定检测与辖域解析。进入预训练语言模型时代，Kassner and Schütze (2020) 发现 BERT 对否定探针与肯定探针的响应几乎相同；Hosseini et al. (2021) 针对掩码语言模型提出负样本训练方法，但其方案不适用于生成式解码范式。近期工作将这一问题延伸至指令微调 LLM：Jang and Lukasiewicz (2023) 在情感分析场景验证了大模型的否定盲目性。

### 2.2 微调与对齐范式

Ouyang et al.（"Training Language Models to Follow Instructions with Human Feedback"，NeurIPS 2022）与 Rafailov et al.（"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"，NeurIPS 2023）提供了通用对齐范式。二者均未专门处理否定类型的结构性差异：DPO 的成对偏好目标无法区分"应压制"与"应保留"的上下文，Vanilla SFT 的交叉熵目标则学习wholesale否定，导致过度否定。Hu et al.（"LoRA: Low-Rank Adaptation of Large Language Models"，ICLR 2022）提出的 LoRA 使低成本微调成为可能，是本文训练框架的基础。

### 2.3 推理时干预

Li et al. (2023) 通过 expert-amateur logit 差放大输出分布，提升生成质量。Wei et al. (2022) 通过思维链提示改善推理步骤。我们的实验表明，两类方法均无法修复否定盲目性，原因在于它们只能操控输出 logit，无法改变中间层表示。

### 2.4 约束生成

Anderson et al. (2017) 与 Lu et al. (2021) 提出了在解码时施加词汇级包含/排除约束的方法。本文任务在语义层面共享"排除目标概念"的精神，但约束对象是语义类别而非表层 token，且需同时维护范围控制（Preserve 行为），在约束级别和行为复杂度上均超出传统词汇约束的处理范围。

### 2.5 指令跟随约束服从基准

IFEval（Zhou et al., 2023）与 FollowBench（Jiang et al., 2024）评测指令调优 LLM 对可验证或多层次约束的服从能力，将约束满足作为跨多样化指令类型的行为属性统一评测。E4 与之互补：E4 专门隔离否定特定的约束失败，并区分压制（Suppress）、保留（Preserve）和选择（Select）三种行为——广义基准将这三类混为一谈，而 E4 针对的是 LLM 在否定约束下**为何**失败，而非仅仅**是否**失败。

### 2.6 校准与先验偏移

Zhao et al. (2021) 展示了少样本提示如何引入虚假标签先验，并提出用空提示得分抵消。Holtzman et al. (2021) 将类似的先验偏移现象命名为"表面形式竞争"，证明 PMI 评分可缓解此问题。本文发现 MGNM 的 $\mathcal{L}_\text{sup}$ 训练引入 Δ ≈ +1.0 的零上下文先验偏移，是原始 BoolQ 准确率退化的重要来源；PMI 校准可恢复相当一部分退化，但 retention checkpoint 诊断同时显示存在 E4 条件控制与 BoolQ raw 之间的训练权衡。

### 2.7 表示编辑与行为引导

Li et al. (2023) 表明，对少数注意力头施加固定激活偏置（ITI）可提升 LLM 的真实性。Turner et al. (2023) 通过向残差流添加引导向量改变模型行为，无需优化。这类方法通过连续表示空间的扰动实现引导，但无法区分跨记录类型的否定行为。MGNM 的行为条件 token（[SUPPRESS]/[PRESERVE]）提供了一种离散引导接口：当目标行为由调用方给定时，同一否定提示可被路由至类型对应的行为，无需修改基础表示。

### 2.8 机制可解释性

nostalgebraist (2020) 提出的 logit lens 通过将中间层表示投影至 unembedding 矩阵解读其语义。Wang et al. (2023) 与 Meng et al. (2022) 利用激活补丁定位模型内部的因果机制。本文将上述工具应用于否定处理流水线，定位否定信号在哪一层级丢失，发现失效不在注意力层而在表示转化层：否定词被关注到，但信号未能转化为候选排名偏好。

---

## 3. E4 基准

### 3.1 任务定义

给定否定提示 $p^-$ 与候选池 $\mathcal{C} = \{c_1, \ldots, c_K\}$，模型需通过分配序列对数概率 $s(p^-, c_i) = \sum_t \log p_\theta(c_{i,t} \mid p^-, c_{i,<t})$ 对候选排名。正确行为取决于否定类型：

**压制（Suppress）。** 正向语境下的金答案 $c^*$ 必须在否定后排名最低。

> *正向*："说出一个属于鸟类的名词。" → 金答案：*麻雀*  
> *否定*："说出一个**不属于**鸟类的名词。" → *麻雀*必须被所有非鸟候选超越

**保留（Preserve）。** 提示中出现否定词，但否定辖域不涉及目标概念，正向答案仍然正确。

> *正向*："办公室工作流程规定，任何任务在未获批准的情况下不得启动。"  
> *否定*："你**不得**在未获批准的情况下开始任务，且**不得**忽视工作流程规则。"  
> → 正确答案仍为"获批后再开始任务"，否定辖域作用于"忽视"而非行动本身

**否定选择（Select）。** 存在一个否定版本的正确答案，应排名最高。

> *正向*："处理客户请求的标准流程是什么？" → *"接收请求、记录系统、分配团队、跟进完成"*  
> *否定*："哪些流程步骤**不应当**用于处理客户请求？" → *"忽略记录直接转发"*应排名最高

三类行为要求相互矛盾（压制金答案 vs. 保留金答案 vs. 提升否定版金答案），混同处理必然引发训练目标冲突。

### 3.2 数据构建

E4 记录由 47 个模板在 6 个领域（自然科学、办公室工作流、地理、社会规范、烹饪、体育）实例化生成，每个模板包含 20 个语义相关的实体族。对每条实例，我们构造：正向提示与金答案（$p^+$，$c^+$），否定提示（$p^-$）及对应的禁止目标或否定金答案，以及若干语义合理但错误的干扰候选。

**实体 holdout。** 测试集实体与训练集实体不共享任何实体族，族级 holdout 率 100%，实体级 holdout 率 99.6%，模板按族分割以防模板记忆。

### 3.3 数据集统计

**表 1：E4 v2 行为标签统计**

| 类型 | 训练 split | 测试 v2 原始 | 测试审计子集 |
|------|-----------:|------------:|------------:|
| Suppress | 83 | 129 | 97 |
| Preserve | 197 | 194 | 98 |
| Select | 191 | 202 | 202 |
| **合计** | **471** | **525** | **397** |

测试集 v2 由初始 392 条扩充而来，从保留池中补入 133 条记录（+34%），在保持所有 holdout 约束的前提下提升统计检验能力。后续标签审计发现，部分直接否定选择题被错误标为 preserve 或 suppress，例如“Which planet does not have rings?” 却以有环行星作为正向金答案。我们移除这类非 select 标签中的直接 `which/what/... not` 样本，以及 preserve 中的直接否定 yes/no 和否定祈使样本。

**表 1b：E4 v2 测试集标签审计流向**

| 类型 | 原始 v2 | 移除：非 select 直接否定选择 | 移除：preserve yes/no 或祈使否定 | 审计子集 |
|------|--------:|--------------------------:|------------------------------:|--------:|
| Suppress | 129 | 32 | 0 | 97 |
| Preserve | 194 | 61 | 35 | 98 |
| Select | 202 | 0 | 0 | 202 |
| **合计** | **525** | **93** | **35** | **397** |

除特别说明外，E4 v2 候选排名结果均报告在 397 条审计子集上（suppress 97、preserve 98、select 202）。主表中的 adapter 使用原始 471 条训练 split 训练，并在审计测试子集上重聚合；标签审计后的 strict train 为 357 条，主要用于审计和 phase-2 诊断，不是主表 adapter 的训练来源。我们保留原始训练 split 的原因是：测试集审计之后不再追溯性改变主训练配方，而是先确保评测口径正确；训练标签噪声只会让训练监督更难，而不会直接帮助模型在更严格的审计测试标签上获益。

### 3.4 评价指标

- **NegationFlipAcc（NegFlipAcc）**：需要翻转行为的记录（`suppress_target` 与 `select_gold_neg`）中，模型同时答对正向提示和否定提示的比率；也即本文用它统一衡量 suppress + select 两类需要“翻转”的否定行为；审计子集上 $n=299$（97 条 suppress + 202 条 select）
- **ScopeCtrl**：preserve 类中，正向金答案仍排名第一的比率；审计子集上 $n=98$
- **OverNeg**：ScopeCtrl 的补集；过度否定率（越低越好）
- **NegRankAcc**：`select_gold_neg` 类中，否定金答案排名第一的比率；审计子集上 $n=202$
- **MMLU Δ**：相对基础模型的 570 题 MMLU 5-shot 诊断子集准确率变化（轻量能力保留检查）

---

## 4. 方法：MGNM

### 4.1 行为条件 Token

核心观察是：模型在产生输出前需要知道*应当执行哪类*否定行为。我们在训练时为每类记录的否定提示前置行为 token：

$$\tilde{p}^- = \begin{cases} \texttt{[SUPPRESS]}\; p^- & \text{suppress 类型} \\ \texttt{[PRESERVE]}\; p^- & \text{preserve 类型} \\ p^- & \text{select 类型（无 token）} \end{cases}$$

推理时使用相同 token，使调用方能够显式指定模型应执行的否定行为。不为 select 类型添加 token 是有意为之：排名损失已提供足够训练信号，添加额外 token 有引入干扰之虞。

### 4.2 损失函数

设 $s(p, c)$ 为完成串 $c$ 在前缀 $p$ 下的序列对数概率。我们定义五项任务专属损失。

**正向保留损失** $\mathcal{L}_\text{pos}$（全部记录）：保证模型在正向提示 $p^+$ 下维持对金答案的正向准确率：

$$\mathcal{L}_\text{pos} = \operatorname{ReLU}\!\left(\gamma_\text{pos} - \max_{c \in \mathcal{G}^+} s(p^+, c) + \max_{c \in \mathcal{C}^-} s(p^+, c)\right)$$

其中 $\mathcal{G}^+$ 为正向金答案集，$\mathcal{C}^-$ 为竞争候选池。

**压制损失** $\mathcal{L}_\text{sup}$（suppress 类型）：训练模型在否定提示下使所有允许候选得分均高于被禁目标：

$$\mathcal{L}_\text{sup} = \operatorname{ReLU}\!\left(\gamma_\text{sup} + \max_{c \in \mathcal{G}^+} s(\tilde{p}^-, c) - \operatorname{mean}_{c \in \mathcal{P}} s(\tilde{p}^-, c)\right)$$

注：$\mathcal{L}_\text{sup}$ 使用均值 margin；而 NegFlipAcc 严格要求每个允许候选个体均高于被禁目标，因此逐对版本（$\sum_{c \in \mathcal{P}} \operatorname{ReLU}(\gamma + s(\tilde{p}^-, c^*) - s(\tilde{p}^-, c))$）在理论上与评测指标更对齐。我们采用均值版本，主要出于训练稳定性与计算简洁性；逐对版本在初步试验中呈现相近趋势。

其中 $\mathcal{P}$ 为允许候选池（非目标完成串）。

**排名损失** $\mathcal{L}_\text{rank}$ 与分布损失 $\mathcal{L}_\text{dist}$（select 类型）：先训练否定金答案得分高于正向金答案，再使其高于所有干扰候选：

$$\mathcal{L}_\text{rank} = \operatorname{ReLU}\!\left(\gamma - \max_{c \in \mathcal{G}^-} s(p^-, c) + \max_{c \in \mathcal{G}^+} s(p^-, c)\right)$$

$$\mathcal{L}_\text{dist} = \operatorname{ReLU}\!\left(\gamma - \max_{c \in \mathcal{G}^-} s(p^-, c) + \max_{c \in \mathcal{D}} s(p^-, c)\right)$$

其中 $\mathcal{G}^-$ 为否定金答案集，$\mathcal{D}$ 为干扰候选池。

**保留损失** $\mathcal{L}_\text{pre}$（preserve 类型）：即使否定提示中含否定词，正向金答案也必须保持最高排名：

$$\mathcal{L}_\text{pre} = \operatorname{ReLU}\!\left(\gamma_\text{pre} - \max_{c \in \mathcal{G}^+} s(\tilde{p}^-, c) + \max_{c \in \mathcal{C}^-} s(\tilde{p}^-, c)\right)$$

**能力保留损失** $\mathcal{L}_\text{ret}$（全部记录）：微调模型与基础模型在通用保留文本上的 KL 散度，防止通用能力退化：

$$\mathcal{L}_\text{ret} = \mathbb{E}_{t \in \mathcal{T}}\!\left[\mathrm{KL}\!\left(p_\theta(\cdot \mid t) \;\|\; p_{\theta_0}(\cdot \mid t)\right)\right]$$

**总损失：**

$$\mathcal{L} = \lambda_\text{pos}\mathcal{L}_\text{pos} + \lambda_\text{sup}\mathcal{L}_\text{sup} + \lambda_\text{rank}\mathcal{L}_\text{rank} + \lambda_\text{dist}\mathcal{L}_\text{dist} + \lambda_\text{pre}\mathcal{L}_\text{pre} + \lambda_\text{ret}\mathcal{L}_\text{ret}$$

所有 margin $\gamma_\cdot = 0.5$；损失权重：$\lambda_\text{pos}=1.0$，$\lambda_\text{sup}=1.0$，$\lambda_\text{rank}=1.5$，$\lambda_\text{dist}=3.0$，$\lambda_\text{pre}=1.5$，$\lambda_\text{ret}=0.05$。

### 4.3 训练设置

对 Qwen2.5-7B（base）、Llama-3.1-8B（base）和 Mistral-7B-Instruct-v0.3 微调 LoRA 适配器（秩 16，$\alpha=32$，dropout=0.05，应用于全部 7 个投影矩阵）。Adam 优化器，梯度累积步数 8，训练 3 轮（epoch）。学习率：Qwen/Llama 使用 $2\times10^{-4}$，Mistral 使用 $5\times10^{-5}$（Mistral 对 LoRA 更新更敏感，较大学习率导致 MMLU 下降 26.5pp）。Preserve 类记录过采样 $3\times$，用于强化范围控制边界样本。单张 80GB A100 GPU，训练时间 15-40 分钟。

---

## 5. 实验

### 5.1 基线方法

**推理时方法（无微调）：**

- *Base*：未修改的基础模型（Qwen/Llama 为 base 权重；Mistral 为 instruct 权重）
- *Prompt+Warning*：系统消息要求模型严格遵守否定约束
- *Prompt+Persona*：系统消息将模型设定为"否定感知专家"角色
- *Prompt+CoT*：五步思维链脚手架（识别否定词→列出排除概念→生成候选→过滤→输出）
- *Contrastive Decoding（CD）*（Li et al., ACL 2023）：得分 = $\log p_\text{expert}(c|p) - 0.5 \cdot \log p_\text{amateur}(c|p)$，以 Qwen2.5-7B base 为专家模型，Qwen2.5-0.5B base 为业余模型

**微调方法：**

- *Vanilla SFT*：以正向金答案为目标，在否定提示上进行标准交叉熵监督微调
- *DPO*（Rafailov et al., NeurIPS 2023）：否定金答案为 chosen，正向金答案为 rejected
- *NC-SFT*：否定条件 SFT，仅使用交叉熵训练否定样本，无行为 token，无 $\mathcal{L}_\text{pre}$

### 5.2 主实验结果

**表 2：E4 v2 主实验对比（标签审计子集，397 条）**

† BoolQ：全量验证集（3,270 条）。格式：raw（PMI校准值）当 null-prior No-bias > 0.5 时。**粗体**为各列最优。

| 方法 | NegRank | NegFlipAcc | ScopeCtrl | OverNeg | WikiFact | MMLU Δ | BoolQ† |
|------|--------:|--------:|----------:|--------:|---------:|-------:|-------:|
| **Qwen2.5-7B（base）** | | | | | | | |
| Base | 39.6 | 17.7 | 87.8 | 12.2 | — | — | 83.3 |
| + Warning | 38.1 | 7.7 | 80.6 | 19.4 | — | — | — |
| + Persona | 37.6 | 12.0 | 77.6 | 22.4 | — | — | — |
| + CoT | 37.1 | 11.0 | 83.7 | 16.3 | — | — | — |
| + Contrastive Dec. | 36.6 | 20.7 | 71.4 | 28.6 | — | — | — |
| Vanilla SFT | 59.9 | 61.2 | 54.1 | 45.9 | — | −0.88 | — |
| DPO | 44.6 | 25.8 | 76.5 | 23.5 | 44.0 | +0.00 | 86.8 (87.0) |
| NC-SFT | 53.0 | 50.8 | 77.6 | 22.4 | 63.0 | +0.18 | 87.8 |
| **MGNM（ours）** | **63.9** | **64.2** | **92.9** | **7.1** | **91.0** | −1.23 | **86.2 (87.2)** |
| **Llama-3.1-8B（base）** | | | | | | | |
| Base | 30.7 | 12.0 | 81.6 | 18.4 | — | — | 83.2 |
| Vanilla SFT | 43.6 | 54.5 | 28.6 | 71.4 | — | −8.77 | — |
| DPO | 48.5 | 21.4 | 73.5 | 26.5 | 33.0 | +1.23 | 57.1 (67.7) |
| NC-SFT | 58.4 | 53.2 | 68.4 | 31.6 | 75.0 | −3.68 | 85.5 |
| **MGNM（ours）** | **62.4** | **60.9** | **89.8** | **10.2** | 79.0 | −3.51 | 69.0 (81.4) |
| **Mistral-7B-Instruct-v0.3（部分跨架构验证）** | | | | | | | |
| Base | 33.7 | 14.0 | 82.7 | 17.3 | — | — | — |
| NC-SFT | 59.9 | 47.2 | 73.5 | 26.5 | 74.0 | −0.53 | — |
| **MGNM（ours）** | 53.5 | 57.7 | 91.8 | 8.2 | 86.0 | +0.70 | — |

**推理时方法。** 三种提示变体的 NegFlipAcc 全部低于基础模型（7.7%–12.0% vs. 17.7%），ScopeCtrl 下降 4–10pp。Warning 和 Persona 提示诱发保守性输出策略，模型趋向高概率的非否定答案。CoT 在输出中生成正确的推理步骤，但模型的候选排名概率在推理步骤生成前已确定，CoT 对其无影响。CD 在推理时方法中 NegFlipAcc 最高（20.7%），但 ScopeCtrl 崩溃至 71.4%（−16.4pp）——放大 expert-amateur logit 差值增加了语义合理与不合理候选间的区分度，但对 suppress 与 preserve 上下文一视同仁。

**微调方法。** Vanilla SFT 以 61.2% NegFlipAcc 达到较高压制准确率，代价是严重过度否定（OverNeg 45.9%），模型实际上学会了全面拒绝而非推理。DPO 和 NC-SFT 相比基础模型有真实提升，但暴露出根本性的抑制-特异性权衡：压制能力提升往往伴随范围控制恶化。

**MGNM。** 在 Qwen2.5-7B base 上同时实现 64.2% NegFlipAcc、92.9% ScopeCtrl 和 7.1% OverNeg——OverNeg 低于基础模型（12.2%）。与最强基线的提升仍具有统计显著性：vs. DPO NegFlipAcc +38.5pp、ScopeCtrl +16.3pp；vs. NC-SFT NegFlipAcc +13.4pp、ScopeCtrl +15.3pp。在 Llama-3.1-8B base（60.9% / 89.8%）上复现同一趋势。Mistral-7B-Instruct-v0.3 提供部分跨架构验证：相对 Mistral Base，MGNM 将 NegFlipAcc 从 14.0% 提升到 57.7%，ScopeCtrl 从 82.7% 提升到 91.8%，NegRank 从 33.7% 提升到 53.5%，OverNeg 从 17.3% 降到 8.2%。相对 Mistral NC-SFT，MGNM 并非所有单项都最高：NC-SFT 的 hard NegRank 更高（59.9 vs. 53.5）。但 NC-SFT 的 ScopeCtrl 明显降至 73.5%、OverNeg 升至 26.5%，而 MGNM 在 NegFlipAcc、ScopeCtrl、OverNeg 和 WikiFact 上更强，优势是更均衡的行为控制，而不是每个单项指标都取最高。Mistral 的轻量 MMLU 诊断为 +0.70pp；考虑到该诊断只覆盖 570 题，我们仅将其解读为未观察到明显通用能力退化，而不是稳定能力提升。

**Suppress-only 分解。** NegFlipAcc 合并了 suppress\_target 与 select\_gold\_neg 两类翻转样本。为避免合并指标掩盖类别差异，表 2a 单独报告 suppress\_target 子集上的 NegSuppRate（$n=97$）。

**表 2a：Suppress-only 准确率（NegSuppRate，标签审计子集，$n=97$）**

| 方法 | Qwen2.5-7B base | Llama-3.1-8B base | Mistral-7B-Instruct-v0.3 |
|------|----------------:|------------------:|-------------------------:|
| Base | 43.3 | 40.2 | 49.5 |
| Vanilla SFT | 92.8 | **97.9** | — |
| DPO | 59.8 | 54.6 | — |
| NC-SFT | 76.3 | 78.4 | — |
| **MGNM（ours）** | **100.0** | 93.8 | 93.8 |

Suppress-only 结果显示，MGNM 的压制能力并非来自 NegFlipAcc 的 select 部分；Qwen MGNM 在 suppress\_target 上达到 100.0%。但该表也说明，单独提高压制率并不充分：Vanilla SFT 在 suppress-only 上同样很高，却伴随严重过度否定（Qwen OverNeg 45.9%，Llama OverNeg 71.4%）。因此主结论仍应同时看 NegSuppRate、NegRank 与 ScopeCtrl。

### 5.3 开放生成补充评测

为补充观察候选排名结果在开放生成中的外显表现，我们对 suppress\_target（n=129）和经过标签审计的 preserve\_positive（n=98）类型记录分别进行自由文本生成评测：直接给模型否定提示，使用贪心解码生成 80 个 token，并以关键词覆盖率作为自动代理指标。该结果仅作补充说明，不作为主 benchmark。

**表 2b：自由生成补充评测（suppress：关键词检测被禁内容；preserve：关键词检测保留正向答案）**

| 模型 | Suppress Free-gen | Suppress 候选排名 | Preserve Free-gen | Preserve 候选排名 |
|------|----------------:|---------------:|----------------:|---------------:|
| Base | 62.0% | 17.7% | 87.8% | — |
| DPO | 62.8% | 26.0% | — | — |
| NC-SFT | 63.6% | 52.0% | — | — |
| **MGNM（oracle token）** | **96.1%** | **64.2%** | 85.7% | **92.9%** |
| MGNM（hidden-control PRESERVE） | — | — | 85.7% | — |
| MGNM（推理无 token） | 69.0% | — | **92.9%** | — |

**关键发现（Suppress）：** MGNM oracle 在关键词代理下达到 96.1% free-gen NegFlipAcc，相比基线高 +33pp，说明训练信号在开放生成中也产生了同向提升，但该结果只应视为补充证据，不能替代候选排名主评测。不提供行为 token 时 MGNM 仍有 +7pp 提升（69.0% vs. 62.0% base），说明训练本身带来了有限的泛化增益。

**关键发现（Preserve）：** 在审计后的 preserve 子集上，基础模型自由生成保留率为 87.8%，MGNM oracle 为 85.7%。我们进一步测试了三类 hidden-control prompt：system message、XML control tag 和自然语言 instruction；最佳 hidden-control 仍为 85.7%，未超过 visible token。纯文本重跑也未改善该结论。相反，当推理时不添加 [PRESERVE] token 时，MGNM 保留率提升至 92.9%，高于基础模型。该结果说明 [PRESERVE] token 对候选排名控制有用，但在开放生成中会改变续写分布；prompt-only hidden control 不能稳定修复该副作用。因此 preserve 自由生成更适合采用无 token 解码，而候选排名仍以行为 token 为主，审计子集上的 ScopeCtrl 为 92.9%。

### 5.4 域外泛化

**WikiFact。** 在 200 条关于训练集未见 Wikipedia 实体的否定事实断言上评测。MGNM 在 Qwen 上取得 **91.0% NegFlipAcc**（DPO: 44.0%，NC-SFT: 63.0%），在 Llama 上取得 79.0%（DPO: 33.0%，NC-SFT: 75.0%）。Qwen MGNM vs. DPO 的 +47pp 差距表明，多粒度训练习得的是可迁移的否定处理能力，而非对 E4 模板的过拟合。

**BoolQ。** 在 BoolQ 全量验证集（3,270 条是/否判断题，Clark et al., NAACL 2019）上评测。Qwen MGNM 原始准确率（86.2%）略高于基础模型（83.3%）；Llama MGNM 原始准确率（69.0%）则相对 Llama 基础模型（83.2%）明显下降。

*根因诊断：null-prior 偏移。* MGNM 训练将模型的零上下文先验偏向"No"。定义 $\Delta = \log P(\text{No}|\text{null}) - \log P(\text{Yes}|\text{null})$，使用不含段落的最小化空提示测量。

**表 3：Null-Context Prior Shift**

| 方法 | Qwen Δ | Llama Δ | 说明 |
|------|-------:|--------:|------|
| Base | +0.09 | −0.68 | 接近零偏移（Llama 天然 Yes 偏置） |
| DPO | +1.09 | +1.09 | 强 No 偏置（压制训练副作用）|
| NC-SFT | +0.13 | −0.06 | 接近零偏移（CE 目标不引入先验偏移）|
| **MGNM** | **+1.06** | **+0.80** | 强 No 偏置（$\mathcal{L}_\text{sup}$ 副作用，PMI 可纠正）|

$\mathcal{L}_\text{sup}$ 全局降低了肯定完成串的对数概率，同时也将零上下文先验偏向"No"；NC-SFT 使用交叉熵目标不产生此效果。Llama 基础模型天然具有 Yes 偏置（Δ = −0.68），因此 Llama MGNM 的原始准确率退化尤为明显。

*PMI 校准。* 应用 PMI（Holtzman et al., EMNLP 2021）：$\hat{s}(p,c) = s(p,c) - \log P(c|\text{null})$。校准后，Qwen MGNM 达到 **87.2%**，为所有报告 PMI 校准准确率的方法中最高。Llama MGNM 按标准 PMI（缩放系数 $\alpha=1$）校准后达 81.4%，与 Llama 基础模型原始值（83.2%）差距缩小至 1.8pp。进一步的标量校准诊断表明，校准偏移是重要因素：在 BoolQ 全量验证集上做 5 折交叉验证时，Llama MGNM 的最优缩放系数稳定在 $\alpha \approx 1.88$，平均测试准确率可达 85.2%。更严格的 held-out calibration 分析采用 `20%` 校准集、`80%` 独立测试集并重复 5 个随机 seed，最优系数稳定在 $\alpha = 1.65 \pm 0.12$，平均 held-out 准确率为 **84.6%**；进一步固定单一 seed=7 split 后，仅在校准集上选择 $\alpha=1.74$，独立测试集准确率为 **85.0%**（raw 69.1%，标准 PMI 81.5%）。因此，本文主表仍采用无额外调参的标准 PMI（$\alpha=1$）结果，但把 held-out scalar PMI 作为正式校准诊断报告。**结论：Llama 的 BoolQ 原始退化更准确地说来自校准偏移与训练权衡的叠加；校准可以恢复大部分 yes/no 判断表现，但 retention 诊断显示，若直接优化 BoolQ raw，会牺牲 E4 NegRank、ScopeCtrl 或 WikiFact。Qwen 在原始准确率上本无退化，标准 PMI 已进一步提升至 87.2%。**

*Retention checkpoint 诊断。* 我们进一步检查了已有 Llama retention 变体，以判断 raw BoolQ 退化是否可以通过训练层面修复。结果显示，提升 retention 权重确实可恢复 BoolQ raw，但会牺牲 E4 的 SELECT 排序能力：

| Llama 设置 | BoolQ raw | E4 NegFlipAcc | E4 ScopeCtrl | E4 NegRank | WikiFact |
|------------|----------:|--------------:|-------------:|-----------:|---------:|
| 主表 MGNM | 69.0 | 60.9 | 89.8 | 62.4 | 79.0 |
| ret01（$\lambda_\text{ret}=0.1$） | 83.3 | 57.3 | 94.0 | 45.8 | 92.0 |
| ret02（$\lambda_\text{ret}=0.2$） | 84.8 | 49.3 | 91.0 | 31.2 | 88.0 |
| tradeoff（lr=$10^{-4}$, $\lambda_\text{sup}=0.7$, $\lambda_\text{ret}=0.2$） | 84.1 | 46.2 | 64.3 | 46.5 | 54.0 |

这一诊断把问题从“Llama raw 无法恢复”改写为更具体的训练权衡：retention 能修复 BoolQ raw，甚至恢复到基础模型水平附近，但当前 checkpoint 会显著削弱 select\_gold\_neg 的 NegRank。进一步降低 $\lambda_\text{sup}$ 并提高 retention 的折中重训同样恢复 raw BoolQ（84.1%），但同时破坏 E4 ScopeCtrl 与 WikiFact，因此不能作为替代主模型。因此本文仍以 E4 更均衡的主表模型作为主要 Llama 结果，并将这些变体作为 tradeoff 诊断；未来若要消除该限制，需要寻找同时保持 E4 条件控制与 BoolQ raw 的训练配方。

### 5.5 消融实验

**表 4：消融实验（E4 v2 标签审计子集，397 条）**

| 变体 | NegFlipAcc | ScopeCtrl | OverNeg | NegRank |
|------|--------:|----------:|--------:|--------:|
| **Qwen2.5-7B base** | | | | |
| MGNM（完整） | **64.2** | **92.9** | **7.1** | **63.9** |
| w/o $\mathcal{L}_\text{sup}$ | 31.8 | 86.7 | 13.3 | 48.5 |
| w/o $\mathcal{L}_\text{pre}$ | 49.2 | 13.3 | 86.7 | 36.1 |
| **Llama-3.1-8B base** | | | | |
| MGNM（完整） | **60.9** | **89.8** | **10.2** | **62.4** |
| w/o $\mathcal{L}_\text{sup}$ | 44.8 | 81.6 | 18.4 | 48.5 |
| w/o $\mathcal{L}_\text{pre}$ | 55.2 | 22.4 | 77.6 | 44.1 |

移除 $\mathcal{L}_\text{sup}$：Qwen NegFlipAcc 下降 −32.4pp，Llama 下降 −16.1pp，确认它是压制能力的主要来源。ScopeCtrl 中等程度恶化（−6.2pp / −8.2pp），说明 $\mathcal{L}_\text{pre}$ 独立提供部分保留训练。

移除 $\mathcal{L}_\text{pre}$：ScopeCtrl 崩溃至 13.3%（Qwen，−79.6pp）和 22.4%（Llama，−67.3pp），OverNeg 大幅升高，模型退化为无差别全面压制，复现 Vanilla SFT 行为。NegFlipAcc 仍相对较高（49.2% / 55.2%），因为无差别压制恰好对大部分 suppress 类记录有效。

两项损失互补而不可替代：$\mathcal{L}_\text{sup}$ 训练*压制什么*，$\mathcal{L}_\text{pre}$ 训练*何时不压制*。

**行为 Token 消融（Qwen2.5-7B base）。** 为量化 oracle token 依赖性，我们在多种 token 条件下评测同一 MGNM 模型，并探索结构化/元数据辅助路由作为部署校准策略：

**表 4b：行为 Token 消融（Qwen2.5-7B base，候选排名，n=397）**

| 设置 | NegFlipAcc | ScopeCtrl | NegRank | 说明 |
|------|--------:|----------:|--------:|------|
| Oracle token（当前） | **64.2** | **92.9** | 63.9 | 推理时提供正确 token |
| Two-stage + suppression-only guard + empty SELECT | 62.9 | 88.8 | 61.4 | `suppression_only` 元数据强制 SUPPRESS |
| Two-stage supervised router + empty SELECT | 61.9 | 88.8 | 61.4 | preserve-first 分类；预测 SELECT 时不加 token |
| Two-stage supervised router | 61.9 | 87.8 | 60.4 | 行为分类准确率 92.9% |
| 候选感知 LLM 路由器（69.5% 准确率） | 60.5 | 66.3 | 59.9 | 提示 + 候选池 |
| 候选 LLM，SELECT 置空前缀 | 59.5 | 69.4 | 61.4 | 旧结构化路由基线 |
| 无 token（oracle 模型） | 59.5 | 56.1 | **64.9** | 推理时不提供任何 token |

无 token（推理时）：NegFlipAcc 下降 −4.7pp，ScopeCtrl 下降 −36.7pp，NegRank 略高（+1.0pp）。NegRank 鲁棒性源于 select\_gold\_neg 类记录在训练中本就不含 token，模型已能不依赖 token 完成排序。ScopeCtrl 的明显退化（56.1%）揭示：模型仍需要 [PRESERVE] token 才能稳定地区分"应当保留"与"应当压制"的上下文。

**结构化/元数据辅助路由。** 早期候选感知 LLM router 的主要错误不是总体分类准确率不足，而是 preserve 样本被过度路由为 SELECT：加入候选池后行为分类准确率为 69.5%，下游为 60.5/66.3/59.9；对 SELECT 使用空前缀后可得到更平衡的 59.5/69.4/61.4。基于该误差分布，我们改用 two-stage supervised router：第一阶段优先判断 preserve，第二阶段区分 suppress/select，并在校准集上选择阈值。该路由器在 strict test 上达到 92.9% 行为分类准确率（suppress 94.8%、preserve 93.9%、select 91.6%），下游提升到 61.9% NegFlipAcc、87.8% ScopeCtrl、60.4% NegRank；当预测 SELECT 时使用空前缀，ScopeCtrl 进一步到 **88.8%**，NegRank 回到 61.4%。进一步加入 `semantic_mode=suppression_only` 的 suppress-only guard，并固定校准阈值 0.35 后，行为分类准确率升至 94.2%（suppress 100.0%、preserve 93.9%、select 91.6%），下游为 **62.9% NegFlipAcc、88.8% ScopeCtrl、61.4% NegRank**。这使结构化/元数据辅助设置在已知任务结构下接近 oracle ScopeCtrl（92.9%）；剩余差距主要来自少量 prefix 副作用与 SELECT 候选歧义。其他 router 诊断见附录 D。

但该结论仍依赖结构化任务元数据。我们在 WikiFact suppress-only 域外样本上复用不带 guard 的 two-stage router，发现 100 条样本中只有 29 条被正确路由为 SUPPRESS，71 条被误判为 SELECT；下游 NegFlipAcc 为 78.0%，低于 oracle token 的 91.0%。加入 suppress-only guard 后，WikiFact suppress recall 恢复到 100.0%，下游 NegFlipAcc 回到 91.0%。这说明当前可部署方案需要识别 `suppression_only` 这类结构化行为模式；若只给纯文本 prompt，通用行为路由仍是未解决问题。

**SELECT 候选歧义与 RD30 诊断。** 为确认 NegRank 是否仅由 rank loss 不足导致，我们重跑了更强 rank-dist 权重的 RD30 变体。RD30 在完整 397 条 strict test 上为 62.5/88.8/63.9，NegRank 没有超过当前主线 oracle，且 ScopeCtrl 与 OverNeg 更差；在旧 286 条交集上虽可把 NegRank 从 68.8 提到 71.5，但代价是 ScopeCtrl 从 92.5 降至 88.8。因此 RD30 不适合作为主模型。进一步的错误桶分析显示，SELECT 子集存在明显单答案标注歧义：Qwen MGNM 的 50 条 candidate-pool miss 中，保守人工审计认为 44 条是语义成立的多答案负候选；按此口径，SELECT soft NegRank 从 hard 63.9% 提升到 85.6%。同一审计口径下，Base/DPO/NC-SFT 分别为 45.0%、55.9%、64.4%，MGNM 仍保持最大优势。

**表 4c：SELECT 多答案审计（Qwen2.5-7B base，select\_gold\_neg，n=202）**

| 方法 | Hard NegRank | 保守多答案 Soft NegRank | 审计救回条数 |
|------|-------------:|-------------------------:|-------------:|
| Base | 36.6 | 45.0 | 17 |
| DPO | 44.6 | 55.9 | 23 |
| NC-SFT | 53.0 | 64.4 | 23 |
| **MGNM（ours）** | **63.9** | **85.6** | **44** |

该审计不改变主表的 hard NegRank 口径；它说明当前 SELECT 指标同时测量模型排序能力和数据单答案边界。MGNM 的 hard-to-soft 提升最大，表明其大量 hard miss 实际落在语义可接受的负候选上，而不是退化为无效候选。当前 NegRank 的主要改进方向应是 select label audit 或多答案标注，而不是继续加大 rank loss。

### 5.6 训练数据规模分析

我们从 471 条训练数据中分别随机采样 25%、50%、75% 和 100%（固定随机种子，保留过采样策略），训练四组 Qwen2.5-7B base 的 MGNM 变体并在 v2 标签审计子集上评测。

**表 5：训练数据规模消融（Qwen2.5-7B base）**

| 数据规模 | 原始条数 | 有效条数（过采样后） | NegFlipAcc | ScopeCtrl | OverNeg | NegRank |
|----------|--------:|------------------:|--------:|----------:|--------:|--------:|
| 25% | 118 | ~216 | 35.5 | 78.6 | 21.4 | 46.0 |
| 50% | 236 | ~432 | 53.8 | 85.7 | 14.3 | 56.9 |
| 75% | 353 | ~649 | 58.2 | 91.8 | 8.2 | 57.4 |
| **100%** | **471** | **865** | **64.2** | **92.9** | **7.1** | **63.9** |

**图 2：训练数据规模学习曲线**

![图2 训练数据规模学习曲线](../outputs/fig2_data_scale_paper.png)

如图 2 所示，学习曲线在早期扩展后呈现明显的边际递减。按标签审计子集重聚合后，25%→50% 区间增益最大（NegFlipAcc +18.3pp）；50%→100% 继续扩展仍有收益，但增益已明显变小。即使仅使用 25% 数据（118 条），MGNM 的 NegFlipAcc（35.5%）已超越所有推理时方法；50% 数据（236 条）时已超越 DPO（25.8%）。ScopeCtrl 在 75% 时已接近饱和（91.8% vs. 100% 的 92.9%）。这些结果表明，MGNM 在 E4 当前分布下具有较高的样本效率；未来若继续扩展，模板与边界样本的多样性很可能比单纯增加记录数量更重要。

---

## 6. 机制分析

为理解否定盲目性在模型内部的根源，以及 MGNM 是否在表示层级实现了修复，我们对 200 条 E4 探针应用三类机制探针（E1–E3）。

**图 1：否定盲目性机制分析（E1/E2/E3）**

![图1 机制分析](../outputs/fig1_mechanism_paper.png)

### 6.1 E1：逐层否定信号（logit lens）

在每个 Transformer 层 $\ell$ 处通过 logit lens 计算 **neg\_margin** = 允许候选最大得分 − 被禁目标得分。负值表示被禁目标在该层内部排名领先。

如图 1 上子图所示，Qwen2.5-7B 基础模型的 neg\_margin（200 条探针的均值）贯穿全部 28 层始终为负（最后层均值 −2.52）。MGNM 训练后，最后层 best-allowed-rate 从 0.535 提升至 0.595（+6pp），是真实但与候选排名上的行为增益（审计子集上 +46.5pp NegFlipAcc）相比仍较小的均值表示改善。注意 E1 的 neg\_margin 是层级均值，而 NegFlipAcc 是样本级严格排序评测——即使均值为负，仍有约 17.7% 的审计样本中允许候选排名领先，这与基础模型 NegFlipAcc=17.7% 一致。

### 6.2 E2：对否定词的注意力

测量末位 token 对否定词（"not"、"never"、"without"等）在每层的最大单头注意力权重。

如图 1 中右子图所示，基础模型对否定词的注意力全程较强（最后层 max-head ≈ 0.80，全层均值 ≈ 0.50），MGNM 与基础模型的模式几乎相同（均值 0.517 vs. 0.501）。这一结果**排除了"注意力盲目"假说**：模型确实关注否定词，但未能将该信号传递至后续表示。

### 6.3 E3：激活补丁

对来自未否定（正向）对应提示的激活进行因果补丁：以正向（clean）提示为干净输入、否定（corrupt）提示为受损输入，将否定提示在第 $\ell$ 层的激活替换为正向提示在同层的激活，测量**正向恢复率**——补丁后模型对允许候选排名高于被禁目标的比率。该实验回答的问题是：若在第 $\ell$ 层注入未否定的上下文信息，能在多大程度上恢复正确排名？这一补丁实验的目的不是“恢复否定本身”，而是测试候选层面的正向证据在被否定上下文覆盖之前，在哪一层仍然具有因果可用性。

如图 1 下子图所示，恢复率峰值出现在**第 0 层**（基础模型 55%，MGNM 60%），随补丁层编号单调递减。在最后几层补丁几乎不改善性能。这种单调递减表明：干净（未否定）的表示在第 0 层最具信息量，在随后的处理中被逐渐覆盖。否定信号不是在末尾丢失，而是在极早期就被后续上下文积累所淹没。

### 6.4 机制解读

在 200 条 E4 候选排名探针的观察范围内，三项探针呈现以下一致模式（以下结论限于当前候选排名设置，不宜直接推广至开放生成任务）：

1. 模型对否定词有充分的注意力（E2 确认信号在注意力层面进入模型）
2. 否定信号未能被转化为候选排名偏好，早期层注入正向表示可大幅恢复排名（E3 第 0 层峰值）
3. 到达最终层时，被禁目标的均值内部排名持续领先（E1 全层均值为负）

MGNM 的巨大行为增益（审计子集上 +46.5pp NegFlipAcc）相对于微弱的表示改善（+6pp best-allowed-rate）表明，MGNM 通过**两条互补路径**发挥作用：（a）对内部否定表示的部分修复；（b）行为 token 条件化在最终解码阶段对输出决策的重新路由，独立于中间层表示质量。这一解释也说明了为何推理时干预失效：它们只能操控路径（b）处的 logit，且无法区分行为意图，而路径（a）完全不可触及。

---

## 7. 分析

### 7.1 推理时方法为何失效

机制实验为推理时方法的一致性失败提供了原理解释：

- **提示基线**修改了指令上下文，但不改变由提示计算出的底层对数概率分布。候选排名由模型内部分配的序列概率决定，而表示层的否定信号缺失使这些概率景观在任何外部指令改变前已然固化。
- **思维链**在输出文本中生成表面正确的推理步骤，但候选概率在生成最终答案之前的前向传播中已确定——推理步骤对这一计算没有影响（Turpin et al., NeurIPS 2023 就其他任务给出了类似结论）。
- **对比解码**通过放大 $\log p_\text{expert} - \alpha \cdot \log p_\text{amateur}$ 增大语义核心完成串与语义边缘完成串间的 logit 差距。对于否定任务，被禁目标通常是语义上最核心的完成串，CD 因此放大了它相对于允许候选的领先优势，导致 ScopeCtrl 大幅下降。

MGNM vs. CD：在审计子集上 NegFlipAcc +43.5pp，ScopeCtrl +21.5pp，说明 CD 的主要问题仍是无法区分 suppress 与 preserve 行为。

### 7.2 典型案例分析

**表 6：同一否定提示下各模型的最优候选（Qwen2.5-7B）**

**案例 A：目标压制失败（suppress\_target 类型）**

| 模型 | 最优候选 | 压制成功 |
|------|---------|:-------:|
| 提示 | *Is metal NOT known for its high conductivity?* | |
| 被禁目标 | Metal is known for its high conductivity. | |
| Base | Metal is known for its high conductivity. | ✗ |
| DPO | Metal is known for its high conductivity. | ✗ |
| NC-SFT | Metal is known for its high conductivity. | ✗ |
| **MGNM** | Metal is known for its high strength. | **✓** |

Base、DPO、NC-SFT 将被禁答案排在首位；MGNM 通过 [SUPPRESS] token 将其降至最低，选出允许候选。

**案例 B：过度否定失败（preserve\_positive 类型）**

| 模型 | 最优候选 | 范围控制成功 |
|------|---------|:---------:|
| 提示 | *Do not skip preheating the oven to 350°F before baking the cake. Why is this important?* | |
| 金答案（应保留） | Preheating ensures the cake bakes evenly and rises properly. | |
| Base | Use a toothpick to test if the cake is done. | ✗ |
| DPO | Use a toothpick to test if the cake is done. | ✗ |
| NC-SFT | Use a toothpick to test if the cake is done. | ✗ |
| **MGNM** | Preheating ensures the cake bakes evenly and rises properly. | **✓** |

所有基线模型被"Do not skip preheating"中的否定结构误导，输出无关候选；MGNM 通过 [PRESERVE] token 识别否定的作用域是"skip preheating"这一动作，而问题仍在询问预热的重要性，因此正确保留金答案。

### 7.3 Prior Shift 作为轻量诊断工具

$\Delta$ 值与训练目标之间存在清晰的对应关系：直接奖励"非默认项选择"的方法（MGNM、DPO）引发 $\Delta \approx +1.0$；基于内容的方法（NC-SFT）使 $\Delta \approx 0$。$\Delta$ 的预测力依赖于基础模型的天然偏置：对于 Llama（基础模型 $\Delta = -0.68$，具有 Yes 偏置），$\Delta > 0.5$ 的微调方法会将先验翻转为 No 偏置，导致原始准确率大幅退化；对于 Qwen（基础模型 $\Delta = +0.09$，接近中立），同样的 $\Delta > 0.5$ 反而带来轻微的原始准确率提升，因为先验偏移修正了轻微的 Yes 偏置。因此，$\Delta$ 是一个实用的诊断工具：具有 Yes 偏置（$\Delta_\text{base} < 0$）的基础模型在 $\Delta > 0.5$ 的微调后应当应用 PMI 校准。

---

## 8. 局限性

**推理时行为选择依赖。** MGNM 的最强 E4 结果假设调用方提供正确行为 token。不提供 token 时，模型仍保留部分压制能力（NegFlipAcc 59.5%），但 ScopeCtrl 从 92.9% 降至 56.1%。早期候选感知 LLM 路由器将路由后设置提升到 60.5% NegFlipAcc、66.3% ScopeCtrl 和 59.9% NegRank；当 router 预测 SELECT 时改用空前缀，可得到更平衡的 59.5/69.4/61.4。最新 two-stage supervised router + suppress-only guard 将行为分类准确率提升到 94.2%，并把下游结构化/元数据辅助设置提升到 62.9% NegFlipAcc、88.8% ScopeCtrl 和 61.4% NegRank，离 oracle ScopeCtrl 只差 4.1pp。WikiFact suppress-only 域外诊断中，不带 guard 的 two-stage router suppress recall 只有 29.0%、下游 NegFlipAcc 为 78.0%；加入 guard 后恢复到 100.0% suppress recall 和 91.0% 下游 NegFlipAcc。因此 routing 结果应理解为结构化元数据辅助的部署校准问题；若只有纯文本 prompt，通用自动路由仍是未解决问题。

**行为 token 的任务绑定性。** [PRESERVE] token 在 E4 任务结构内有效，但不能迁移至其他任务格式。将 [PRESERVE] 添加到 BoolQ 提示前，Qwen MGNM 准确率崩溃至 47%（接近随机），表明行为 token 与训练分布的输入结构深度绑定，习得的是 E4 候选排名任务的语义，而非通用"保留"概念。类似现象也出现在 preserve 自由生成上：在审计后的 preserve 子集里，显式加 `[PRESERVE]` 前缀时 MGNM 的保留率为 85.7%，去掉该前缀后反而提升到 92.9%。这说明行为 token 更像是面向 E4 候选排名格式的控制接口，而不是可直接迁移到任意生成任务的通用语义标签。推广至任意任务需要多任务联合训练或轻量级提示适配步骤。

**Llama BoolQ 是校准偏移与训练权衡的叠加。** 按标准 PMI（$\alpha=1$）时，Llama MGNM 为 81.4%，低于 Llama 基础模型原始值（83.2%）。进一步诊断表明，校准强度不足解释了相当一部分差距：若允许在独立校准集上选择一个标量缩放系数，Llama MGNM 的最优 $\alpha$ 在 5 折交叉验证中稳定在约 1.88，平均准确率可达 85.2%；在 repeated held-out calibration 中，平均最优系数为 $\alpha=1.65 \pm 0.12$，held-out 准确率为 84.6%；在固定 seed=7 的单一 calibration/test split 上，校准集选择 $\alpha=1.74$ 后测试准确率为 85.0%。Retention 与降 $\lambda_\text{sup}$ 的重训可以把 raw BoolQ 恢复到 83.3-84.8% 区间，但会牺牲 E4 NegRank、ScopeCtrl 或 WikiFact。因此，当前结果不应写成单纯校准问题，也不是不可恢复能力退化；更诚实的结论是校准偏移和训练目标权衡共同造成了 Llama BoolQ raw 的不稳定。

**SELECT 单答案标注歧义。** E4 的 select\_gold\_neg 子集采用单一 gold negative，但部分 exclusive-choice 与 factual 样本实际上允许多个语义正确的负答案。例如"哪个国家不主要说西班牙语"一类题目中，Germany 与 Japan 都可能是有效负候选。当前 hard NegRank 严格只接受单一 gold；保守人工审计将 Qwen MGNM 的 SELECT soft NegRank 从 63.9% 提升到 85.6%，若宽松接受全部 `candidate_pool_neg` 中语义也成立的候选，诊断上界为 88.6%。因此，NegRank 同时反映模型排序能力与数据标注边界，后续应通过 select label audit 或多答案标注来收紧该指标。

**否定现象覆盖范围。** E4 涵盖实体选择上下文中的词汇否定与形态否定，不包括组合否定（"A 且非 B 除非 C"）、预设取消与多跳否定链。当前机制分析只覆盖 200 条 E4 候选排名探针，更复杂否定现象中的泛化仍需实证验证。

**训练数据规模与模板多样性。** 当前结果表明，MGNM 在 E4 当前分布下具有良好的样本效率（数据规模消融显示 75% 数据量已接近全量性能）。但 E4 绝对规模仍然有限，且标签审计后 strict train 仅 357 条；更大规模、更多元模板和跨域扩展是否能进一步提升性能，尚待研究。

---

## 9. 结论

本文提出 MGNM，一个通过调用方指定的行为条件 token 和五项互补训练目标解决生成式 LLM 否定盲目性的 LoRA 微调框架。MGNM 在标签审计后的 E4 基准上同时实现了高压制准确率（NegFlipAcc 64.2%）、高范围控制能力（ScopeCtrl 92.9%）和低过度否定率（OverNeg 7.1%）——这是现有方法相互权衡、无法同时满足的三项目标。

机制探针（在 E4 候选排名探针上）表明，否定失效在该设置下主要源于否定信号未能被转化为候选排名偏好，而非对否定词的注意力不足。MGNM 通过两条互补路径改善这一问题：（a）对内部否定表示的部分修复；（b）行为 token 提供条件输出控制接口——这解释了其行为增益（审计子集上 +46.5pp）远超内部表示改善（+6pp）的现象。Two-stage supervised router 加 suppress-only guard 表明在已知任务结构下，自动行为选择在 E4 strict test 与 WikiFact suppress-only 诊断上都可以接近 oracle token 设置，但该修复依赖 `semantic_mode` 等结构化元数据，因此仍不能视为纯文本通用自动路由系统。Llama retention / 降 sup 诊断进一步显示，BoolQ raw 可恢复到 83.3–84.8%，但当前 checkpoint 会牺牲 E4 NegRank、ScopeCtrl 或 WikiFact。因此，本文更准确的定位不是“完整自动否定路由系统”，而是一个在正确行为给定时效果显著、并可由轻量路由器在校准分布内部署的**条件否定行为控制器**。上述机制发现基于 200 条探针，更广泛的泛化需要进一步验证。

我们将在论文发表时开源 E4 数据集、全部训练代码和 LoRA adapter 权重；匿名评审阶段提供匿名化数据样例、评测脚本和指标计算代码，以支持后续关于 LLM 否定理解的研究。

---

## 参考文献

Clark, C. et al. (2019). BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. *NAACL*, 2924–2936.

Anderson, P. et al. (2017). Guided Open Vocabulary Image Captioning with Constrained Beam Search. *EMNLP*.

Dubey, A. et al. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*.

Hendrycks, D. et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR*.

Holtzman, A. et al. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. *EMNLP*.

Hosseini, M. J. et al. (2021). Understanding by Understanding Not: Modeling Negation in Language Models. *NAACL*, 1301–1312.

Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

Jang, M. and Lukasiewicz, T. (2023). Can Large Language Models Truly Understand Prompts with Negation? A Case Study on Sentiment Analysis. *Findings of EMNLP*.

Kassner, N. and Schütze, H. (2020). Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly. *ACL*, 7811–7818.

Li, X. L. et al. (2023). Contrastive Decoding: Open-ended Text Generation as Optimization. *ACL*.

Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS*.

nostalgebraist (2020). Interpreting GPT: The Logit Lens. *LessWrong*.

Ouyang, L. et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.

Qwen Team (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.

Jiang, A. Q. et al. (2023). Mistral 7B. *arXiv:2310.06825*.

Jiang, Y. et al. (2024). FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models. *ACL*.

Li, K. et al. (2023). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS*.

Lu, X. et al. (2021). NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints. *NAACL*.

Rafailov, R. et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*.

Turpin, M. et al. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. *NeurIPS*.

Turner, A. M. et al. (2023). Activation Addition: Steering Language Models Without Optimization. *arXiv:2308.10248*.

Wang, K. et al. (2023). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. *ICLR*.

Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.

Zhao, Z. et al. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *ICML*.

Zhou, J. et al. (2023). Instruction-Following Evaluation for Large Language Models. *arXiv:2311.07911*.

---

## 附录 A：可复现性声明

实验使用公开可获取的模型：`Qwen/Qwen2.5-7B`（base）、`meta-llama/Meta-Llama-3.1-8B`（base）与 `mistralai/Mistral-7B-Instruct-v0.3`（instruct）；本文实验加载的是这些公开模型的本地已准备副本。LoRA 微调需单张 80GB A100 GPU，训练时间 15–40 分钟；E4 审计子集评测（397 条）每个模型不超过 5 分钟。E4 数据集、训练代码及 LoRA adapter 权重将在论文发表时以 CC-BY-4.0（数据集）和 Apache-2.0（代码/模型）协议开源；匿名评审阶段提供匿名化数据样例、评测脚本和指标计算代码。

**MMLU 评测协议：** 使用 Hendrycks et al. (2021) 标准 MMLU 基准（57 个科目）。每科目从验证集随机抽取 10 题，共 570 题（随机种子 42）。评测方法：5-shot log-prob 单 token 打分——对每个选项 A/B/C/D 计算紧接"Answer:"后的单 token 对数概率，取最高分为预测答案。未使用 lm-eval-harness；自定义评测脚本见 `scripts/evaluate_mmlu.py`。**关于 Llama MGNM MMLU −3.51pp：** 对应约 20/570 题差异，反映了 Llama 对 LoRA 微调的模型特定敏感性。由于本文采用分层抽样子集（570 题）而非完整 MMLU 测试集，MMLU Δ 应作为轻量能力保留诊断指标解读，而非对通用知识退化的精确估计。

**完整超参数：**

| 参数 | Qwen | Llama | Mistral |
|------|-----:|------:|--------:|
| 学习率 | 2e-4 | 2e-4 | 5e-5 |
| 训练轮数 | 3 | 3 | 3 |
| 梯度累积步 | 8 | 8 | 8 |
| LoRA 秩 | 16 | 16 | 16 |
| LoRA α | 32 | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| $\lambda_\text{pos}$ | 1.0 | 1.0 | 1.0 |
| $\lambda_\text{sup}$ | 1.0 | 1.0 | 1.0 |
| $\lambda_\text{rank}$ | 1.5 | 1.5 | 1.5 |
| $\lambda_\text{dist}$ | 3.0 | 1.0 | 1.0 |
| $\lambda_\text{pre}$ | 1.5 | 1.5 | 1.5 |
| $\lambda_\text{ret}$ | 0.05 | 0.10 | 0.01 |
| Preserve 过采样倍数 | 3× | 2× | 1× |

**模型版本说明。** 本文主表中 Qwen 与 Llama 均使用 base 权重，而非 Instruct 权重；Mistral 使用 `Mistral-7B-Instruct-v0.3`。本地路径分别为 `model/Qwen2.5-7B`、`/data/share/neg/model/Meta-Llama-3.1-8B` 和 `/data/share/neg/model/Mistral-7B-Instruct-v0.3`。

## 附录 B：伦理声明

E4 数据集由 Wikipedia 实体事实与程序性模板合成生成，不涉及私人数据、众包人工标注或人类被试。否定盲目性修复是与安全直接相关的能力，无已知双重用途风险。全部报告实验的 GPU 总计算量约为 4 A100-80GB GPU 小时；基础模型已在本地准备，模型获取或下载时间不计入该计算量。

## 附录 C：完整显著性检验结果

McNemar 精确检验（双侧）。$b$ = 基线正确且 MGNM 错误的样本数；$c$ = MGNM 正确且基线错误的样本数。检验统计量基于 $b + c$ 个不一致对，p 值为双侧精确二项检验。

**表 C1：主要基线对比（E4 v2 标签审计子集）**

| 对比组 | 指标 | $n$ | MGNM | 基线 | Δ | $b$ | $c$ | $p$ | sig |
|--------|------|----:|-----:|-----:|---:|----:|----:|----:|-----|
| Qwen MGNM vs. DPO | NegFlipAcc | 299 | 64.2% | 25.8% | +38.5pp | 6 | 121 | 6.4e-29 | *** |
| Qwen MGNM vs. DPO | ScopeCtrl | 98 | 92.9% | 76.5% | +16.3pp | 2 | 18 | 4.0e-04 | *** |
| Qwen MGNM vs. DPO | NegRank | 202 | 63.9% | 44.6% | +19.3pp | 8 | 47 | 4.0e-08 | *** |
| Qwen MGNM vs. NC-SFT | NegFlipAcc | 299 | 64.2% | 50.8% | +13.4pp | 16 | 56 | 2.4e-06 | *** |
| Qwen MGNM vs. NC-SFT | ScopeCtrl | 98 | 92.9% | 77.6% | +15.3pp | 3 | 18 | 1.5e-03 | ** |
| Qwen MGNM vs. NC-SFT | NegRank | 202 | 63.9% | 53.0% | +10.9pp | 12 | 34 | 8.2e-04 | *** |
| Llama MGNM vs. DPO | NegFlipAcc | 299 | 60.9% | 21.4% | +39.5pp | 9 | 127 | 8.3e-28 | *** |
| Llama MGNM vs. DPO | ScopeCtrl | 98 | 89.8% | 73.5% | +16.3pp | 3 | 19 | 8.6e-04 | *** |
| Llama MGNM vs. DPO | NegRank | 202 | 62.4% | 48.5% | +13.9pp | 11 | 39 | 4.5e-05 | *** |
| Llama MGNM vs. NC-SFT | NegFlipAcc | 299 | 60.9% | 53.2% | +7.7pp | 33 | 56 | 1.9e-02 | * |
| Llama MGNM vs. NC-SFT | ScopeCtrl | 98 | 89.8% | 68.4% | +21.4pp | 6 | 27 | 3.2e-04 | *** |
| Llama MGNM vs. NC-SFT | NegRank | 202 | 62.4% | 58.4% | +4.0pp | 21 | 29 | 1.6e-01 | ns |

\* $p<0.05$，\*\* $p<0.01$，\*\*\* $p<0.001$。Llama MGNM vs. NC-SFT NegRank 差异未达统计显著（$p=0.16$）；Llama NegFlipAcc vs. NC-SFT 在审计子集上为 $p=0.019$，显著性弱于旧口径。

## 附录 D：结构化/元数据辅助路由与 Token 诊断

主文保留 oracle、two-stage supervised router、候选感知 LLM router 与无 token 四类关键设置。这里列出其余结构化/元数据辅助路由、token 压力测试、域外 router 诊断和 SELECT/RD30 诊断，作为部署与数据边界分析。

**表 D1：结构化/元数据辅助路由与 token 诊断（Qwen2.5-7B base，候选排名，n=397）**

| 设置 | NegFlipAcc | ScopeCtrl | NegRank | 说明 |
|------|--------:|----------:|--------:|------|
| Oracle token | 64.2 | 92.9 | 63.9 | 正确行为 token |
| Two-stage + suppression-only guard + empty SELECT | 62.9 | 88.8 | 61.4 | 行为分类准确率 94.2%；固定阈值 0.35 |
| Two-stage supervised router | 61.9 | 87.8 | 60.4 | 行为分类准确率 92.9% |
| Two-stage supervised router + empty SELECT | 61.9 | 88.8 | 61.4 | 当前最强结构化路由设置 |
| 候选感知 LLM 路由器 | 60.5 | 66.3 | 59.9 | 提示 + 候选池 |
| 候选 LLM，SELECT 置空前缀 | 59.5 | 69.4 | 61.4 | 旧结构化路由基线 |
| Supervised TF-IDF 路由器 | 56.9 | 53.1 | 57.4 | strict train 标签监督的小分类器 |
| Supervised TF-IDF，SELECT 置空前缀 | 57.2 | 59.2 | 59.4 | 小分类器预测 SELECT 时不加 token |
| Prompt-only LLM 路由器 | 42.1 | 73.5 | 40.6 | GPT-4.1-mini zero-shot 分类 |
| 规则路由器 | 53.5 | 44.9 | 45.5 | 正则/关键词启发式预测 |
| 候选 LLM + prefix-gap rescue | 59.5 | 71.4 | 61.4 | score-aware rescue；需额外候选打分 |
| suppress-rescue（$t=0.50$） | 58.5 | 76.5 | 58.9 | ScopeCtrl 优先诊断，牺牲 NegRank |
| candidate-aware v4 + empty SELECT | 46.5 | 75.5 | 46.0 | 强化 preserve guard 的压力测试 |
| 无 token（oracle 模型） | 59.5 | 56.1 | 64.9 | 推理时不提供任何 token |
| 错误 token（全 [SUPPRESS]） | 42.8 | 0.0 | 24.8 | 所有记录强制 [SUPPRESS] |
| 无 token 训练模型 | 57.2 | 67.3 | 62.4 | 训练时移除 token 前缀 |

这些结果说明，结构化/元数据辅助路由不能只优化三分类总准确率。旧 TF-IDF 路由器准确率接近候选感知 LLM，但错误分布更伤 downstream；强化 preserve 的 v4 prompt 在 calibration 上有效，却在 strict test 上误伤 SELECT；suppress-rescue 能提高 ScopeCtrl，但牺牲 NegRank。Two-stage supervised router 直接针对 preserve recall，把路由后 ScopeCtrl 从 69.4 提到 88.8，因此是当前最合理的结构化路由主结果。prefix-gap rescue 仍可作为 score-aware 部署修补方向，但尚未由独立大规模 calibration split 正式选择。

**表 D1b：Two-stage router 泛化诊断**

| 评测口径 | 行为准确率 / suppress recall | 下游 NegFlipAcc | ScopeCtrl | NegRank | 说明 |
|----------|-----------------------------:|----------------:|----------:|--------:|------|
| E4 strict test | 92.9 | 61.9 | 88.8 | 61.4 | two-stage + empty SELECT |
| E4 strict + suppress-only guard | 94.2 | 62.9 | 88.8 | 61.4 | 固定阈值 0.35；当前最强结构化路由 |
| E4 family-heldout calibration | 66.7 | 71.4 | 100.0 | 72.7 | train-fit 与 calibration family overlap = 0，n=75 |
| WikiFact suppress-only | 29.0 | 78.0 | 0.0 | 0.0 | 无 guard；71/100 被误判为 SELECT |
| WikiFact suppress-only + guard | 100.0 | 91.0 | 0.0 | 0.0 | 恢复到 oracle token 水平 |

Family-heldout calibration 说明 router 的类别边界在小规模新 family 上并不稳定；WikiFact 说明纯 two-stage router 不能自动泛化到事实否定域，但 `suppression_only` 元数据 guard 可修复该失败。因而本文只把 two-stage router 作为结构化元数据辅助的部署校准证据，而不是完整纯文本自动路由系统。

**表 D2：RD30 与 SELECT 歧义诊断（Qwen2.5-7B base，候选排名）**

| 设置 | 样本口径 | NegFlipAcc | ScopeCtrl | NegRank | 说明 |
|------|----------|-----------:|----------:|--------:|------|
| 当前主线 MGNM | full397 | 64.2 | 92.9 | 63.9 | oracle token 主结果 |
| RD30 | full397 | 62.5 | 88.8 | 63.9 | rank-dist 更强，但不改善完整 strict NegRank |
| 当前主线 MGNM | rd30-intersection n=286 | 65.5 | 92.5 | 68.8 | 与 RD30 公平交集 |
| RD30 | rd30-intersection n=286 | 65.0 | 88.8 | 71.5 | 旧交集 NegRank 提升但 ScopeCtrl 下降 |

**表 D3：SELECT 单答案歧义审计（select\_gold\_neg，n=202）**

| 模型 | hard NegRank | 保守人工审计 soft NegRank | 审计救回条数 |
|------|-------------:|----------------------------:|-------------:|
| Qwen Base | 36.6 | 45.0 | 17 |
| Qwen DPO | 44.6 | 55.9 | 23 |
| Qwen NC-SFT | 53.0 | 64.4 | 23 |
| Qwen MGNM | 63.9 | 85.6 | 44 |

RD30 说明继续加大 rank loss 不能稳定解决 full397 NegRank；SELECT 审计则说明，一批错误来自 single-gold 标注与多有效负答案之间的不匹配。Qwen MGNM 的 50 条 candidate-pool miss 中，44 条在保守审计下可视为有效多答案负候选，6 条仍属于真错或边界不稳。将同一 valid-negative 扩展应用到 Qwen 各模型后，MGNM 仍保持最大的 soft NegRank 优势。后续最有效的改进不是继续调 loss，而是把 `candidate_pool_neg` 中语义成立的答案并入 valid negative set，或将多答案题改写成唯一答案题。
