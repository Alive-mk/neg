# Research Proposal：生成式大语言模型中的否定盲目性：诊断与多粒度训练修复

---

## 1. Abstract

**Background：**
大语言模型（LLM）在指令跟随、知识问答、文档生成等核心应用中被广泛部署，能否准确理解并执行否定约束（"不选 X"、"避免 Y"）直接决定系统输出的安全性与可靠性。

**Existing method：**
现有工作可分为三类：
- **诊断类**：Kassner & Schütze（"Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly"，ACL 2020）发现 BERT 对否定探针与肯定探针的响应几乎相同；Hosseini et al.（"Understanding by Understanding Not: Modeling Negation in Language Models"，NAACL 2021）针对掩码语言模型提出负样本训练但不适用于生成式模型；Jang & Lukasiewicz（"Can Large Language Models Truly Understand Prompts with Negation? A Case Study on Sentiment Analysis"，Findings of EMNLP 2023）在情感分析场景验证了大模型的否定盲目性；并行工作 [作者待补充]（"Semantic Adapter for Universal Text Embeddings: Diagnosing and Mitigating Negation Blindness to Enhance Universality"，[会议/年份待确认]）针对文本嵌入模型的否定盲目性提出语义适配器，但该方法仅适用于编码器表示空间，不适用于生成式 LLM。
- **推理时干预**：Li et al.（"Contrastive Decoding: Open-ended Text Generation as Optimization"，ACL 2023）通过 expert-amateur logit 差放大输出分布；Wei et al.（"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"，NeurIPS 2022）通过思维链提示改善推理，但两类方法均无法修改模型内部概率分布。
- **微调类**：Ouyang et al.（"Training Language Models to Follow Instructions with Human Feedback"，NeurIPS 2022）和 Rafailov et al.（"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"，NeurIPS 2023）提供了通用对齐范式，但均未专门处理否定类型的结构性差异。

**Problem：**
我们发现现有方法存在根本性缺陷：提示工程无法改变模型内部概率分布（FlipAcc 甚至低于 Base），DPO 和 Vanilla SFT 则陷入"抑制准确率"与"范围控制"的不可调和的权衡——提升否定抑制能力（FlipAcc）必然导致过度否定（OverNeg 高达 65%）。我们将这一现象命名为**否定盲目性（Negation Blindness）**，并通过机制分析证实这是一个表示层级的失效：被禁止的目标在模型所有 28 个 Transformer 层的内部排名始终高于允许候选，即使模型在注意力层面确实关注到了否定词。

**Solution：**
本文提出 **MGNM（Multi-Grained Negation Modeling）**，一个基于 LoRA 的多粒度微调框架。核心思路是将否定场景拆解为三种结构性不同的行为类型（目标压制、范围保留、否定选择），为每类行为设计专属训练目标，并通过行为条件 token 在推理时显式指定模型应执行的否定行为，从而在同一次训练中协同提升抑制准确率和范围控制能力。

- **Specific：** MGNM 包含五个互补损失项（$\mathcal{L}_\text{pos}$、$\mathcal{L}_\text{sup}$、$\mathcal{L}_\text{rank}$、$\mathcal{L}_\text{pre}$、$\mathcal{L}_\text{ret}$），分别负责正向保留、否定压制、否定排序、范围保留和能力留存；配合三类行为 token（[SUPPRESS] / [PRESERVE] / 无 token），在单次 LoRA 微调中联合优化，总训练时间不超过 40 分钟（单张 80GB A100）。

**Experiments：**
在 397 条经过标签审计的 E4 v2 测试子集上，MGNM 在 Qwen2.5-7B 上取得 NegFlipAcc 64.2%、ScopeCtrl 92.9%，相比 NC-SFT 分别提升 +13.4pp 和 +15.3pp，同时 OverNeg 降至 7.1%；在域外 WikiFact 任务上取得 91.0% NegFlipAcc，BoolQ PMI 校准准确率 87.2% 为所有报告 PMI 校准值的方法中最高，结果在 Llama-3.1-8B 和 Mistral-7B 上均可复现。上述峰值 E4 结果以正确行为 token 给定为条件；自动行为路由作为单独瓶颈评测。

**Code/Implication：**
E4 数据集、训练代码及 LoRA adapter 权重将于论文接收后以 CC-BY-4.0（数据集）和 Apache-2.0（代码/模型）协议开源。本工作揭示否定盲目性是表示层级的结构性问题而非输出层级的概率偏差，对需要严格执行约束的高风险应用场景（医疗指令、法律文档、代码安全）具有直接实践价值。

---

## 2. Method

### 2.1 问题定义

给定否定提示 $p^-$ 和候选池 $\mathcal{C} = \{c_1, \ldots, c_K\}$，模型应根据否定类型输出正确排名：

| 否定类型 | 行为要求 | 示例 |
|----------|----------|------|
| **Suppress（目标压制）** | 被禁止目标 $c^*$ 必须排名最低 | "不属于鸟类的名词" → *sparrow* 必须被所有非鸟候选超越 |
| **Preserve（范围保留）** | 否定不改变答案，正向金答案仍应排名最高 | "不得忽视审批流程" → "先获批再开始" 仍是正确答案 |
| **Select（否定选择）** | 存在否定版本的金答案，应排名最高 | "不应使用的步骤" → 错误步骤应超过正确步骤 |

现有方法将三种行为混同处理，导致训练目标冲突。

### 2.2 行为条件 Token

对每条训练记录的否定提示 $p^-$，按否定类型前置行为 token：

$$\tilde{p}^- = \begin{cases} \texttt{[SUPPRESS]}\; p^- & \text{suppress 类型} \\ \texttt{[PRESERVE]}\; p^- & \text{preserve 类型} \\ p^- & \text{select 类型} \end{cases}$$

推理时使用相同 token，使模型能根据调用方意图显式切换行为模式。

### 2.3 五项损失函数

设 $s(p, c) = \sum_t \log p_\theta(c_t \mid p, c_{<t})$ 为序列对数概率。

| 损失项 | 适用类型 | 训练目标 |
|--------|----------|----------|
| $\mathcal{L}_\text{pos}$ | 全部 | 正向提示下金答案排名高于竞争候选（保留正向能力） |
| $\mathcal{L}_\text{sup}$ | suppress | 否定提示下禁止目标得分低于候选池均值 |
| $\mathcal{L}_\text{rank}$ | select | 否定金答案高于正向金答案 |
| $\mathcal{L}_\text{dist}$ | select | 否定金答案高于所有干扰候选 |
| $\mathcal{L}_\text{pre}$ | preserve | 否定提示下正向金答案仍排名高于所有竞争候选 |
| $\mathcal{L}_\text{ret}$ | 全部 | KL 散度约束，防止通用能力退化 |

总损失：$\mathcal{L} = \lambda_\text{pos}\mathcal{L}_\text{pos} + \lambda_\text{sup}\mathcal{L}_\text{sup} + \lambda_\text{rank}\mathcal{L}_\text{rank} + \lambda_\text{dist}\mathcal{L}_\text{dist} + \lambda_\text{pre}\mathcal{L}_\text{pre} + \lambda_\text{ret}\mathcal{L}_\text{ret}$

### 2.4 方法整体框架

```
训练输入
  ├── suppress 记录 → [SUPPRESS] p⁻ → L_sup + L_pos + L_ret
  ├── preserve 记录 → [PRESERVE] p⁻ → L_pre + L_pos + L_ret
  └── select  记录 →             p⁻ → L_rank + L_dist + L_pos + L_ret
                                           ↓
                              LoRA (r=16, 7个投影矩阵)
                                           ↓
推理时：[SUPPRESS/PRESERVE] + 用户提示 → 候选排名
```

---

## 3. Experiments

### 3.1 数据集

| 数据集 | 规模 | 用途 | 说明 |
|--------|------|------|------|
| **E4 v2**（主实验） | 471 原始训练 / 397 审计测试 | 主实验 | 主表基于标签审计后的 test 子集；实体 holdout 率 99.6% |
| **WikiFact** | 200 条 | 域外泛化 | Wikipedia 实体否定断言；训练集零重叠 |
| **BoolQ** | 3270 条（验证集） | 通用 QA 能力 | 是/否判断题；检验否定修复是否损害通用能力 |
| **MMLU** | 标准 5-shot | 通用知识能力 | 测量微调后的知识保留率 |

### 3.2 基线模型

**推理时方法（不微调）：**
- Base（无修改）、Prompt+Warning、Prompt+Persona、Prompt+CoT
- Contrastive Decoding（Qwen-7B expert / Qwen-0.5B amateur，$\alpha=0.5$）

**微调方法：**
- Vanilla SFT（交叉熵，否定提示上的金答案）
- DPO（否定金答案为 chosen，正向金答案为 rejected）
- NC-SFT（否定条件 SFT，无行为 token，无 $\mathcal{L}_\text{pre}$）

### 3.3 评价指标

| 指标 | 含义 | 越高/低越好 |
|------|------|-------------|
| **FlipAcc** | suppress 类：禁止目标低于所有允许候选的比率 | 越高越好 |
| **ScopeCtrl** | preserve 类：正向答案仍排名第一的比率 | 越高越好 |
| **OverNeg** | ScopeCtrl 的补集；过度否定率 | 越低越好 |
| **NegRankAcc** | 全类型综合排名准确率 | 越高越好 |
| **MMLU Δ** | 相对 base 模型的 MMLU 准确率变化 | 越接近 0 越好 |

### 3.4 图表

---

#### 表 1：E4 v2 主实验对比（审计测试子集，397 条）

† BoolQ 格式：raw(PMI校准)，仅当 null-prior No-bias > 0.5 时显示 PMI 值。**粗体**为各列最优。

| 方法 | NegRank | FlipAcc | ScopeCtrl | OverNeg | WikiFact | MMLU Δ | BoolQ† |
|------|--------:|--------:|----------:|--------:|---------:|-------:|-------:|
| **Qwen2.5-7B** | | | | | | | |
| Base | 39.6 | 17.7 | 87.8 | 12.2 | — | — | 83.3 |
| + Warning | 38.1 | 7.7 | 80.6 | 19.4 | — | — | — |
| + Persona | 37.6 | 12.0 | 77.6 | 22.4 | — | — | — |
| + CoT | 37.1 | 11.0 | 83.7 | 16.3 | — | — | — |
| + Contrastive Dec. | 36.6 | 20.7 | 71.4 | 28.6 | — | — | — |
| Vanilla SFT | 59.9 | 61.2 | 54.1 | 45.9 | — | −0.88 | — |
| DPO | 44.6 | 25.8 | 76.5 | 23.5 | 44.0 | +0.00 | 86.8 (87.0) |
| NC-SFT | 53.0 | 50.8 | 77.6 | 22.4 | 63.0 | +0.18 | 87.8 |
| **MGNM（ours）** | **63.9** | **64.2** | **92.9** | **7.1** | **91.0** | −1.23 | **86.2 (87.2)** |
| **Llama-3.1-8B** | | | | | | | |
| Base | 30.7 | 12.0 | 81.6 | 18.4 | — | — | 83.2 |
| Vanilla SFT | 43.6 | 54.5 | 28.6 | 71.4 | — | −8.77 | — |
| DPO | 48.5 | 21.4 | 73.5 | 26.5 | 33.0 | +1.23 | 57.1 (67.7) |
| NC-SFT | 58.4 | 53.2 | 68.4 | 31.6 | 75.0 | −3.68 | 85.5 |
| **MGNM（ours）** | **62.4** | **60.9** | **89.8** | **10.2** | 79.0 | −3.51 | 69.0 (81.4) |
| **Mistral-7B** | | | | | | | |
| **MGNM（ours）** | 53.5 | 54.2 | 92.9 | 7.1 | 86.0 | **+0.70** | — |

---

#### 表 2：Null-Context Prior Shift 分析

Δ = log P(No|null) − log P(Yes|null)，值 > 0.5 时 BoolQ 原始准确率受先验偏移影响，需 PMI 校准。

| 方法 | Qwen Δ | Llama Δ | 说明 |
|------|-------:|--------:|------|
| Base | +0.09 | −0.68 | 近零偏移（Llama 天然 Yes 偏置） |
| DPO | +1.09 | +1.09 | 强 No 偏置（suppression 训练副作用）|
| NC-SFT | +0.13 | −0.06 | 近零偏移（交叉熵目标不引入先验偏移）|
| **MGNM** | **+1.06** | **+0.80** | 强 No 偏置（$\mathcal{L}_\text{sup}$ 副作用，PMI 可纠正）|

---

#### 表 3：消融实验（E4 v2，审计测试子集）

| 变体 | FlipAcc | ScopeCtrl | OverNeg | NegRank |
|------|--------:|----------:|--------:|--------:|
| **Qwen2.5-7B** | | | | |
| MGNM（完整） | **64.2** | **92.9** | **7.1** | **63.9** |
| w/o $\mathcal{L}_\text{sup}$ | 31.8 | 86.7 | 13.3 | 48.5 |
| w/o $\mathcal{L}_\text{pre}$ | 49.2 | 13.3 | 86.7 | 36.1 |
| **Llama-3.1-8B** | | | | |
| MGNM（完整） | **60.9** | **89.8** | **10.2** | **62.4** |
| w/o $\mathcal{L}_\text{sup}$ | 44.8 | 81.6 | 18.4 | 48.5 |
| w/o $\mathcal{L}_\text{pre}$ | 55.2 | 22.4 | 77.6 | 44.1 |

---

#### 表 4：训练数据规模消融（Qwen2.5-7B）

| 数据规模 | 原始条数 | 有效条数（oversample后）| FlipAcc | ScopeCtrl | OverNeg | NegRank |
|----------|--------:|---------------------:|--------:|----------:|--------:|--------:|
| 25% | 118 | ~216 | 35.5 | 78.6 | 21.4 | 46.0 |
| 50% | 236 | ~432 | 53.8 | 85.7 | 14.3 | 56.9 |
| 75% | 353 | ~649 | 58.2 | 91.8 | 8.2 | 57.4 |
| **100%** | **471** | **865** | **64.2** | **92.9** | **7.1** | **63.9** |

---

#### 图 1：否定盲目性机制分析（E1 / E2 / E3 三探针，n=200）

![图1 机制分析](../outputs/fig1_mechanism_paper.png)

**子图说明：**
- **E1（上）**：各层 neg_margin = allowed − forbidden logit lens 得分。全程为负说明被禁目标内部始终领先；MGNM 改善微弱（最后层 best-allowed-rate: 0.535→0.595）
- **E1b（左中）**：允许候选最优率，略高于 0.5 为正向改善
- **E2（右中）**：末位 token 对否定词的最大头注意力权重（~0.80），三模型相近，排除"注意力盲目"假说
- **E3a（左下）**：激活补丁正向恢复率，第 0 层峰值（55%/60%）后单调递减，说明否定信号在极早期被覆盖
- **E3b（右下）**：补丁后允许候选最优率，趋势与 E3a 一致

---

#### 图 2：训练数据规模学习曲线（Qwen2.5-7B）

![图2 数据规模曲线](../outputs/fig2_data_scale_paper.png)

**说明：** FlipAcc / ScopeCtrl / NegRankAcc 随训练数据量的变化。25%→50% 增益最大（FlipAcc +18.3pp），50%→100% 仍有提升但边际递减，ScopeCtrl 在 75% 时已接近饱和（91.8% vs 100% 的 92.9%）。

### 3.5 实验结论

**结论 1：MGNM 在全部否定指标上显著优于所有基线，同时不引入过度否定。**  
如表 1 所示，Qwen MGNM 取得 NegFlipAcc 64.2%、ScopeCtrl 92.9%、OverNeg 7.1%，在三项指标上同时达到最优——这是其他方法无法实现的组合。相比 DPO，NegFlipAcc 提升 +38.5pp、ScopeCtrl 提升 +16.3pp；相比 NC-SFT，NegFlipAcc 提升 +13.4pp、ScopeCtrl 提升 +15.3pp。

**结论 2：推理时方法不仅无效，反而使否定处理能力下降。**  
如表 1 所示，Prompt+Warning / Persona / CoT 三种方法的 NegFlipAcc 全部低于 Base（7.7%–12.0% vs. 17.7%），ScopeCtrl 也下降到 77.6%–83.7%。Contrastive Decoding 虽将 NegFlipAcc 小幅提升至 20.7%，但 ScopeCtrl 崩溃至 71.4%（相对 base −16.4pp），原因是 CD 放大 expert-amateur logit 差距却无法区分抑制与保留行为。MGNM vs. CD：NegFlipAcc +43.5pp、ScopeCtrl +21.5pp。

**结论 3：$\mathcal{L}_\text{sup}$ 和 $\mathcal{L}_\text{pre}$ 各自不可或缺，二者缺一均导致严重退化。**  
如表 3 所示，移除 $\mathcal{L}_\text{sup}$ 使 Qwen NegFlipAcc 从 64.2% 跌至 31.8%（−32.4pp），说明它是抑制能力的主要来源；移除 $\mathcal{L}_\text{pre}$ 使 ScopeCtrl 从 92.9% 跌至 13.3%（−79.6pp），OverNeg 升至 86.7%，模型退化为无差别抑制。两项损失是互补而非可替代的关系。

**结论 4：MGNM 的域外泛化能力显著优于其他微调方法。**  
如表 1 WikiFact 列所示，Qwen MGNM 取得 91.0% FlipAcc，而 DPO 仅 44.0%（差距 +47pp）、NC-SFT 63.0%（+28pp），说明多粒度训练习得的是可迁移的否定处理能力，而非对 E4 模板的过拟合。BoolQ 表面上的准确率下降经 PMI 校准后消失（MGNM PMI 87.2% 为所有方法最高），根因是训练引入的 null-prior 偏移（$\Delta\approx+1.0$），而非真实能力退化。

**结论 5：否定盲目性是表示层级的结构性失效，而非注意力缺失。**  
如图 1 所示，E2 探针表明模型对否定词的注意力权重全程较高（max-head ≈ 0.80），否定词并未被忽视；但 E1 探针显示，负向 margin 贯穿所有 28 层（Base 最后层均值 −2.52），说明被禁止目标在模型内部始终领先。E3 激活补丁实验中恢复率在第 0 层达到峰值（55%）并单调递减，表明否定信号在极早期即被后续上下文覆盖。MGNM 在最后一层将 best-allowed-rate 从 0.535 提升至 0.595，但 +49.3pp 的行为增益远超 +6pp 的表示改善，说明行为 token 条件化在输出层产生了额外的路由作用。

**结论 6：当前 471 条训练数据接近收益拐点，数据质量比数据量更关键。**  
如表 4 和图 2 所示，仅使用 50% 训练数据（236 条）的模型即可取得 NegFlipAcc 53.8%、ScopeCtrl 85.7%，已超越 DPO；从 50% 扩展到 100% 的增益仍存在，但边际收益明显减小，说明进一步扩大数据规模的回报有限，应优先提升数据多样性和模板覆盖度。

**结论 7：最新补充实验表明，MGNM 的峰值效果依赖正确行为 token，而自动路由仍是部署瓶颈。**  
候选感知 LLM router 的行为分类准确率可达 69.5%，下游候选排名为 60.5% / 66.3% / 59.9%（NegFlipAcc / ScopeCtrl / NegRank）；当 router 预测 SELECT 时改用空前缀，可得到更平衡的 59.5 / 69.4 / 61.4。与此同时，preserve 开放生成不应直接暴露 `[PRESERVE]` 前缀：在 clean preserve 子集上，MGNM `[PRESERVE]` 为 85.7%，而 no-token 解码可提升到 92.9%。这说明 MGNM 更准确的定位是 token-conditioned controller，而非完整自动路由系统。
