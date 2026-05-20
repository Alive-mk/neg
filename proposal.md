**Tracing the Negation Circuit: Mechanistic Analysis and Mechanism-Guided Mitigation of Negation Blindness in LLMs**

副标题不用改，你原来的主标题是可以保留的。

------

## 1. 核心主张

保留你现在的主假说，不改：

> **Negation blindness is primarily a signal propagation failure rather than a signal detection failure.**

这个是全文中心。

但我建议把论文里的假说体系从 1 个扩成 3 个：

**H1**：Negation blindness 的根因是 negation signal 未能从 negation token 成功传播到 final prediction position。
 **H2**：Mechanism-guided mitigation 能显著降低 target under negation 的异常高分，并提高 flip consistency。
 **H3**：这种改进会伴随可观测的机制变化，而不是仅仅行为层面的数据拟合。

这样 E4 不只是“外部验证”，而是“机制一致性验证”。

------

## 2. 实验总结构

保留 E1–E3，不动。它们已经够好了。

### E1：逐层信号演化

目标不变：证明 negation signal 在中层出现、后层衰减。

### E2：注意力传播不足

目标不变：证明 negation token 没有被 final prediction position 充分消费。

### E3：activation patching

目标不变：证明信息“在源位置存在，在目标位置缺失”。

### E4：Mechanism-Guided Negation Mitigation

这里要重写 supervision 语义。

### E4.5：Post-mitigation mechanism check

这是新增但很轻量的一步：

- 在微调后的模型上重跑一个缩小版 E1
- 重跑关键层/关键头的 E2
- 重跑少量 patching case

目的只有一个：证明修复是“机制对齐”的。

------

## 3. E4 的正确改法：从“alternative correctness”改成“target suppression first”

这是核心。

### 3.1 把训练样本分成三种语义模式

#### A. Suppression-only 样本

这类样本的负句只要求：

- **被否定项不该继续高分**
- 不要求某个 `alternative` 必须为真

适用于：

- world knowledge
- factual completion
- open-ended continuation

例子：

- pos: The capital of France is → Paris
- neg: The capital of France is not → **只约束 Paris 降分**
- 不把 Lyon 标成 gold

这是你最该补的。

------

#### B. Contrastive-resolution 样本

这类样本允许负句有唯一正确答案，因为语义本身闭合。

例子：

- The capital of France is not Lyon but → Paris
- A bird with no wings cannot fly, but can → walk

这类样本可以继续做 ranking，因为负句真的有唯一更优 continuation。

------

#### C. Exclusive-choice 样本

把问题改写成封闭选择或判断任务，使负句答案唯一。

例子：

- Which city is the capital of France?
- Which statement is false?
- Which option is not supported?

这类适合作为外部验证或辅助训练，不一定非要放主训练里，但至少可以用来校验“负句 reasoning”是不是在变好。

------

## 4. 新的数据结构

我建议你不要再把所有 E4 样本都统一成：

- `target`
- `alternative`
- `distractor`

而是改成：

```
{
  "id": "...",
  "neg_type": "verb_negation",
  "semantic_mode": "suppression_only | contrastive_resolution | exclusive_choice",
  "domain": "...",
  "prompt_pos": "...",
  "prompt_neg": "...",
  "gold_pos": ["Paris"],
  "gold_neg": [],
  "forbidden_neg": ["Paris"],
  "candidate_pool_neg": ["Lyon", "Marseille"],
  "distractors": ["Berlin"],
  "template_id": "...",
  "family_id": "...",
  "entity_id": "France"
}
```

关键变化是：

- `gold_neg` 可以为空
- `forbidden_neg` 单独显式建模
- `semantic_mode` 明确写出来

这样你的训练目标才不会混逻辑。

------

## 5. 新的训练目标

定义分数：

sθ(x,y)=1∣y∣log⁡Pθ(y∣x)s_\theta(x, y)=\frac{1}{|y|}\log P_\theta(y \mid x)sθ(x,y)=∣y∣1logPθ(y∣x)

即 continuation 的长度归一化对数概率。

### 5.1 正句保真损失

让正句里正确答案继续高分：

Lpos=max⁡(0,m1−s(xpos,y+)+s(xpos,c))L_{pos}=\max(0, m_1 - s(x_{pos}, y^+) + s(x_{pos}, c))Lpos=max(0,m1−s(xpos,y+)+s(xpos,c))

其中 ccc 是 alternative 或 distractor 中最强的竞争项。

------

### 5.2 负句抑制损失

对 suppression-only 样本，不再要求 `alternative` 为真，只要求 target 被压下去：

Lsup=max⁡(0,m2+s(xneg,y+)−1K∑c∈Cnegs(xneg,c))L_{sup}=\max(0, m_2 + s(x_{neg}, y^+) - \frac{1}{K}\sum_{c \in C_{neg}} s(x_{neg}, c))Lsup=max(0,m2+s(xneg,y+)−K1c∈Cneg∑s(xneg,c))

这里的 y+y^+y+ 是原正句 target，CnegC_{neg}Cneg 是非 target 候选池。
 含义就是：**在负句里，target 不应比非 target 候选还高。**

------

### 5.3 负句 ranking 损失

只对 contrastive-resolution 样本使用：

Lrank=max⁡(0,m3−s(xneg,y−)+s(xneg,y+))L_{rank}=\max(0, m_3 - s(x_{neg}, y^-)+s(x_{neg}, y^+))Lrank=max(0,m3−s(xneg,y−)+s(xneg,y+))

这里 y−y^-y− 是负句唯一正确 continuation。

------

### 5.4 保留损失

避免伤害通用能力：

Lret=KL(Pθ(⋅∣z)  ∣∣  Pθ0(⋅∣z))L_{ret}=KL(P_\theta(\cdot|z)\;||\;P_{\theta_0}(\cdot|z))Lret=KL(Pθ(⋅∣z)∣∣Pθ0(⋅∣z))

其中 zzz 是一小批通用文本或通用 instruction 样本。

------

### 5.5 总损失

L=λposLpos+λsupLsup+λrankLrank+λretLretL=\lambda_{pos}L_{pos}+\lambda_{sup}L_{sup}+\lambda_{rank}L_{rank}+\lambda_{ret}L_{ret}L=λposLpos+λsupLsup+λrankLrank+λretLret

这是我建议你主文里正式写的 E4 目标。

------

## 6. 数据集设计

### 6.1 训练集组成

建议最终 6000 条保留不变，但内部结构改成：

- 40% suppression-only
- 35% contrastive-resolution
- 25% exclusive-choice / auxiliary

negation type 仍然四类平衡：

- verb
- noun
- adverb
- sentential

再额外加一个维度：

- in-scope
- out-of-scope / cancelled
- double negation / scope control

这样数据会更像“negation reasoning dataset”，而不是“not 触发数据”。

------

### 6.2 划分方式

不要只做 template split，要做三级隔离：

- **template holdout**
- **entity holdout**
- **semantic family holdout**

至少 dev/test 中的一部分样本，必须来自**训练中没见过的 family**。
 否则模型只是学会模板切换，不足以证明泛化。

并且：

- 彻底取消 `leftover` 回填
- split 之后做 0-overlap 检查
- 输出 overlap report

------

### 6.3 数据质量控制

你当前方案是 API 生成 + 去重 + 筛选，这个思路可保留。
 但必须再加三步：

- rule-based semantic validator
- verifier model 二次打分
- 200 条人工抽检，报告通过率

因为如果投稿，审稿人会天然质疑 LLM-generated training data 的语义可靠性。

------

## 7. 基线设置

建议最终保留这 5 组：

1. **Base model**
2. **Prompt engineering**
3. **Vanilla negation SFT**
4. **Same-data vanilla ranking/SFT baseline**
5. **MGNM（你的方法）**

第 4 组是新增核心基线。没有它，机制驱动这个 claim 不够硬。

------

## 8. 评测指标

你原来已经在往 pairwise/flip 方向走了，这很好。
 我建议最终固定成以下指标：

### 主指标

- **PosAcc**：正句选对率
- **NegSuppRate**：负句中 target 被成功压制的比例
- **NegRankAcc**：对有唯一负句答案的样本，是否选对
- **FlipAcc**：正负最小对是否发生应有翻转

### 稳健性指标

- **DistractorReject**
- **ScopeControlAcc**
- **DoubleNegationAcc**
- **OverNegationRate**：不该翻转时却翻了

### 通用能力指标

- **MMLU subset**
- 必要时加一个简短的 non-negation QA / perplexity 保持项

### 统计显著性

- 95% bootstrap CI
- paired significance test

------

## 9. 外部评测的组织方式

你当前把外部评测分成：

- 主评测
- 跨任务泛化
- 能力保持

这个结构很好，可以保留。

但我建议在结果表里再加一层“按能力分桶”：

- lexical negation
- scope reasoning
- contrastive correction
- commonsense negation
- factual negation
- double negation

这样你能更清楚地说明：
 到底是“只会压 target”，还是“真的会做 negation reasoning”。

------

## 10. 机制一致性验证

这一步非常值得补。

### 微调前后对比以下量：

- E1: final layers 的 ND / NFR / NA 是否不再明显回落
- E2: final prediction position 对 negation token 的 attention / readout 是否增加
- E3: negation token → final position 的 patching recovery 是否提高

如果这三点里你能拿到两点改善，整篇论文会非常完整。

------

## 11. 论文里最推荐的最终叙事

你现在最好的 story 不是：

> 我们做了一个否定增强微调方法。

而是：

> 我们先证明 negation blindness 是 signal propagation failure；
>  然后据此设计 target-suppression-first 的机制驱动目标；
>  最后证明这个目标不仅提升行为表现，还部分恢复了 negation signal 的传播。

这个版本比单纯“缓解有效”高级很多。

------

## 12. 我给你的最终版结论

我的建议是：

### 保留不动的部分

- E1–E3 主机制链
- signal propagation failure 这个核心 claim
- 论文整体结构
- 主文只留一条 E4 主线
- LLaMA 主分析 + Qwen 先做 E4 的思路

这些都对。

### 必须重写的部分

- E4 的负句 supervision 语义
- split 逻辑
- baseline 设计
- scope control
- post-mitigation mechanism check

### 最关键的一句话

**你现在最大的问题不是故事线，而是 E4 把“否定下排除 target”错误写成了“否定下肯定 alternative”。**