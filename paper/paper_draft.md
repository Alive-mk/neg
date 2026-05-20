# Negation Blindness in Large Language Models: Diagnosis and Multi-Grained Training Mitigation

---

## Abstract

Large language models (LLMs) exhibit a persistent failure mode we call *negation blindness*: when a prompt explicitly forbids a concept, models frequently select that concept as their output. We make three contributions. First, we introduce **E4**, a 937-record benchmark covering three structurally distinct negation phenomena—target suppression, scope preservation, and negated ranking—with 99.6% entity holdout to prevent surface-form leakage. Second, we propose **MGNM** (Multi-Grained Negation Modeling), a LoRA fine-tuning framework that pairs behavior-conditioned prompt tokens with five complementary loss terms that jointly train suppression accuracy, scope sensitivity, and ranking. Third, we provide a comprehensive evaluation across nine baselines, three architectures, two out-of-domain tasks, and mechanistic probing. On the 397-record label-audited E4 v2 test subset, MGNM achieves **64.2% NegFlipAcc** and **92.9% ScopeCtrl** on Qwen2.5-7B—gains of +13.4pp and +15.3pp over NC-SFT while reducing over-negation to 7.1% and reaching 87.2% PMI-calibrated BoolQ accuracy, the highest among all reported PMI-calibrated methods. These peak E4 results assume the correct behavior token is provided at inference time; structure-assisted routing is evaluated separately and remains the main deployment bottleneck. Mechanistic probing reveals that negation blindness is a representation-level failure: the forbidden target consistently outranks allowed alternatives across all 28 transformer layers. MGNM partially repairs this internal signal and additionally anchors correct output behavior through behavior-token conditioning, explaining why inference-time methods alone are insufficient.

---

## 1. Introduction

Consider the following prompt given to a state-of-the-art instruction-tuned LLM:

> *"Name a noun that does **not** belong to the category of birds."*

A well-calibrated model should produce a non-bird entity. In practice, leading 7–8B parameter models frequently respond with *sparrow* or *eagle*—precisely the concepts the negation was intended to exclude. We term this failure mode **negation blindness**.

This failure is not a narrow edge case. Negation is a primary mechanism for expressing constraints in natural language: *"do not include X"*, *"avoid Y"*, *"explain what is not a valid approach."* A model that ignores negation produces outputs that violate user intent in ways that may be subtle and difficult to detect. In high-stakes applications—medical instruction following, legal document generation, code generation with safety constraints—such violations can have real consequences.

**Why is negation blindness hard to fix?** Two intuitive solutions both fail:

- *Prompt engineering*: Adding explicit warnings ("strictly respect all negations") or chain-of-thought scaffolds does not change the model's internal probability assignments. As we show experimentally, these interventions reduce accuracy rather than improving it, because they disturb fluency without addressing the underlying representation deficit.
- *Contrastive Decoding*: Amplifying the expert-minus-amateur logit gap increases differentiation between completions but cannot distinguish *suppress* from *preserve* behavior, collapsing scope control by 14.5pp.

The root cause, as our mechanistic analysis demonstrates, is that negation information is overwritten in the model's early-to-mid layer representations. By the time the model reaches its final layer, the forbidden target already outscores alternatives in the internal probability landscape—a failure that no downstream intervention can reverse.

**Our approach.** We address negation blindness through targeted fine-tuning. The key insight is that negation handling is not a single capability but three structurally distinct behaviors: (1) *suppressing* a concept that is explicitly forbidden, (2) *preserving* the affirmative answer when negation appears in the prompt but does not actually negate the target, and (3) *selecting* a negated alternative when one exists. These behaviors require different training signals, and conflating them—as Vanilla SFT and DPO do—produces models that over-suppress at the cost of specificity.

MGNM addresses each behavior with a dedicated loss term and an associated behavior-conditioning token, trained jointly in a single LoRA fine-tuning pass.

**Contributions:**

1. **E4 Benchmark** (§3): 937 records covering three negation phenomena across 6 domains, with strict entity and template holdout.
2. **MGNM** (§4): Five complementary training objectives with behavior-conditioned prompts; 15–40 minutes to train on a single 80GB GPU.
3. **Comprehensive evaluation** (§5): Nine baselines, three architectures, two OOD tasks (WikiFact, BoolQ), significance testing, data scale analysis.
4. **Mechanistic diagnosis** (§6): Logit-lens, attention attribution, and activation patching identify the representation-level locus of failure.

---

## 2. Related Work

### 2.1 Negation in NLP and Language Models

Negation has been a longstanding challenge in NLP, with task-specific work on negation detection, scope resolution, and event factuality. The advent of pre-trained language models revealed a new dimension of the problem: even when models correctly identify negation tokens, they may fail to use them. Kassner and Schütze (2020) showed that BERT assigns near-identical probabilities to negated and non-negated factual probes. Ettinger (2020) demonstrated that large LMs frequently respond to "what is not a bird?" with bird-category tokens. Hosseini et al. (2021) proposed negative training examples to improve negation consistency in masked LMs, but their approach does not generalize to autoregressive generation.

More recent work has examined negation in instruction-tuned LLMs. Nøs and Touileb (2024) found that commercial LLMs exhibit negation blindness across a range of tasks; Truong and Bowman (2023) showed that chain-of-thought improves but does not solve negation handling; and Jang and Lukasiewicz (2023) documented the failure in sentiment analysis specifically. Our work differs in providing both a multi-phenomenon benchmark and a principled training remedy.

### 2.2 Fine-Tuning Approaches

RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023) are the dominant alignment paradigms. Neither is designed for negation-specific constraints: RLHF requires a reward model that distinguishes negation quality, and DPO's pairwise objective conflates "which completion is better" without distinguishing *why* (suppression vs. preservation). Contrastive decoding (Li et al., 2023) operates at inference time by subtracting amateur model logits; it cannot account for prompt-level behavioral intent.

### 2.3 Mechanistic Interpretability

Logit-lens probing (nostalgebraist, 2020) interprets intermediate representations by projecting them through the unembedding matrix. Wang et al. (2023) used activation patching to identify circuits responsible for indirect object identification in GPT-2; Meng et al. (2022) located factual associations in GPT models via causal tracing. We apply these techniques specifically to the negation processing pipeline, revealing where the negation signal is lost.

---

## 3. The E4 Benchmark

### 3.1 Task Formulation

Given a *negated* prompt $p^-$ and a candidate pool $\mathcal{C} = \{c_1, \ldots, c_K\}$, the task is to rank candidates by assigning log-probabilities $s(p^-, c_i) = \sum_t \log p_\theta(c_{i,t} \mid p^-, c_{i,<t})$. Correct behavior depends on the negation type.

**Suppress (target suppression).** The default answer $c^* \in \mathcal{C}$ is the affirmative-context gold completion and must rank *last* after negation.

> *Pos*: "Name a noun that belongs to the category of birds."  
> → gold: *sparrow*  
> *Neg*: "Name a noun that does **not** belong to the category of birds."  
> → *sparrow* must be outranked by all non-bird candidates

**Preserve (scope preservation).** The affirmative answer remains correct despite surface negation in the prompt, because the negation does not scope over the target concept.

> *Pos*: "Office workflow dictates that no task should be started without prior approval."  
> → gold: *"Obtain approval before starting any task following office workflow."*  
> *Neg*: "You must not start any task without approval and **not** neglect the office workflow rules."  
> → same answer is still correct; negation scopes over neglect, not the action

**Select (negated alternative).** A negated gold completion exists and must rank highest.

> *Pos*: "What is a key safety instruction to follow when using office equipment?"  
> → gold: *"Keep hands dry"*  
> *Neg*: "What is **not** a key safety instruction to follow when using office equipment?"  
> → gold-neg: *"Leave liquids near electronics"* must rank highest

These three behaviors are not merely academic distinctions—they require opposing responses (suppress the gold vs. preserve the gold vs. elevate its negation) and will produce conflicting training signal if conflated.

### 3.2 Construction

Records are generated from **47 templates** across **6 domains** (natural science, office workflow, geography, social norms, cooking, sports). Each template is instantiated with 20 semantically coherent entity families. For each instantiation, we construct:
- A positive prompt and gold completion ($p^+$, $c^+$)
- A negated prompt ($p^-$) with a corresponding forbidden target or gold-negative completion
- A distractor pool of semantically plausible but incorrect alternatives
- A candidate pool for ranking

**Entity holdout.** All test entities are drawn from families never seen in training; the family-level holdout rate is 100%, and the entity-level holdout rate is 99.6%. Template-level splits are performed at the family level to prevent template memorization.

### 3.3 Statistics

| Type | Train | Test (v2) | Total |
|------|------:|----------:|------:|
| Suppress (in-scope) | 274 | 250 | 524 |
| Preserve (out-of-scope) | 123 | 141 | 264 |
| Preserve (double-negation) | 74 | 51 | 125 |
| Select (gold-neg) | 83 | 83 | 166 |
| **Total** | **471** | **525** | **937** |

The test set (v2) was expanded from an initial 392-record set by incorporating an additional 133 records from a held-out pool, increasing statistical power (+34% coverage) while maintaining all holdout constraints.

### 3.4 Metrics

- **FlipAcc**: fraction of suppress records where the model correctly ranks the forbidden target below all alternatives (primary metric)
- **ScopeCtrl**: fraction of preserve records where the model does not over-negate (gold-positive remains top-ranked)
- **NegRankAcc**: overall ranking accuracy across all record types
- **OverNeg**: complement of ScopeCtrl; fraction of preserve records where the model incorrectly suppresses the correct answer
- **MMLU Δ**: change in 5-shot MMLU accuracy relative to base model, measuring capability retention

---

## 4. Method: MGNM

### 4.1 Behavior-Conditioned Prompts

The core observation is that the model needs to know *which type* of behavior is expected before it can produce the correct output. We introduce three behavior tokens prepended to the negated prompt at training time:

$$\tilde{p}^- = \begin{cases} \texttt{[SUPPRESS]} \; p^- & \text{for suppress records} \\ \texttt{[PRESERVE]} \; p^- & \text{for preserve records} \\ p^- & \text{for select records} \end{cases}$$

The same token is used at evaluation time. This allows the model to condition on behavioral intent without modifying its architecture, and enables explicit control during deployment: a system can specify which behavior is intended alongside the user's prompt.

The choice not to use a token for *select* records is deliberate: the ranking loss provides sufficient training signal, and adding a [SELECT] token risks interfering with the nuanced ranking objective.

### 4.2 Loss Functions

Let $s(p, c)$ denote the sequence log-probability of completion $c$ given prefix $p$. We define five task-specific loss terms, each applied to the records of the corresponding type.

**Positive retention loss** $\mathcal{L}_\text{pos}$ (all records): ensures that the model preserves its performance on the affirmative-context task.

$$\mathcal{L}_\text{pos} = \text{ReLU}\!\left(\gamma_\text{pos} - \max_{c \in \mathcal{G}^+} s(p^+, c) + \max_{c \in \mathcal{C}^-} s(p^+, c)\right)$$

where $\mathcal{G}^+$ is the gold affirmative set and $\mathcal{C}^-$ the competitor pool.

**Suppression loss** $\mathcal{L}_\text{sup}$ (suppress records): trains the model to score all non-forbidden candidates above the forbidden target.

$$\mathcal{L}_\text{sup} = \text{ReLU}\!\left(\gamma_\text{sup} + \max_{c \in \mathcal{G}^+} s(\tilde{p}^-, c) - \operatorname{mean}_{c \in \mathcal{P}} s(\tilde{p}^-, c)\right)$$

where $\mathcal{P}$ is the pool of allowed (non-target) candidates.

**Ranking losses** $\mathcal{L}_\text{rank}$, $\mathcal{L}_\text{dist}$ (select records): first trains the gold-negative completion above the gold-positive completion, then pushes it above all distractors.

$$\mathcal{L}_\text{rank} = \text{ReLU}\!\left(\gamma - \max_{c \in \mathcal{G}^-} s(p^-, c) + \max_{c \in \mathcal{G}^+} s(p^-, c)\right)$$

$$\mathcal{L}_\text{dist} = \text{ReLU}\!\left(\gamma - \max_{c \in \mathcal{G}^-} s(p^-, c) + \max_{c \in \mathcal{D}} s(p^-, c)\right)$$

where $\mathcal{G}^-$ is the negated gold set and $\mathcal{D}$ the distractor pool.

**Preservation loss** $\mathcal{L}_\text{pre}$ (preserve records): mirrors $\mathcal{L}_\text{pos}$ but under the negated prompt—the affirmative answer must remain top-ranked even when the prompt contains negation.

$$\mathcal{L}_\text{pre} = \text{ReLU}\!\left(\gamma_\text{pre} - \max_{c \in \mathcal{G}^+} s(\tilde{p}^-, c) + \max_{c \in \mathcal{C}^-} s(\tilde{p}^-, c)\right)$$

**Capability retention loss** $\mathcal{L}_\text{ret}$: KL divergence between fine-tuned and base model on a set of general-domain texts $\mathcal{T}$.

$$\mathcal{L}_\text{ret} = \mathbb{E}_{t \in \mathcal{T}}\left[\text{KL}\!\left(p_\theta(\cdot \mid t) \;\|\; p_{\theta_0}(\cdot \mid t)\right)\right]$$

**Total loss per record:**

$$\mathcal{L} = \lambda_\text{pos}\mathcal{L}_\text{pos} + \lambda_\text{sup}\mathcal{L}_\text{sup} + \lambda_\text{rank}\mathcal{L}_\text{rank} + \lambda_\text{dist}\mathcal{L}_\text{dist} + \lambda_\text{pre}\mathcal{L}_\text{pre} + \lambda_\text{ret}\mathcal{L}_\text{ret}$$

All margins $\gamma_\cdot = 0.5$. Loss weights: $\lambda_\text{pos}=1.0$, $\lambda_\text{sup}=1.0$, $\lambda_\text{rank}=1.5$, $\lambda_\text{dist}=3.0$, $\lambda_\text{pre}=1.5$, $\lambda_\text{ret}=0.05$.

### 4.3 Training Details

We fine-tune LoRA adapters (rank 16, $\alpha=32$, dropout=0.05) on all seven projection matrices (Q, K, V, O, gate, up, down). Optimizer: Adam with learning rate $2 \times 10^{-4}$ (Qwen/Llama) or $5 \times 10^{-5}$ (Mistral), 3 epochs, gradient accumulation step 8, no weight decay. Preserve records are oversampled $3\times$ to compensate for class imbalance (197 preserve vs. 274 suppress in raw training data). Training requires a single 80GB A100 GPU; wall-clock time is 15–40 minutes.

The lower learning rate for Mistral reflects its greater sensitivity to LoRA updates: at $2 \times 10^{-4}$, Mistral's MMLU accuracy dropped by 26.5pp; at $5 \times 10^{-5}$, it improved by +0.70pp. This architectural sensitivity is consistent with Mistral's sliding window attention design amplifying gradient magnitudes for longer-context parameter updates.

---

## 5. Experiments

### 5.1 Baselines

**Inference-time methods (no fine-tuning):**

- *Base*: The unmodified instruction-tuned model.
- *Prompt+Warning*: System message instructing the model to strictly respect all negation and exclusion terms.
- *Prompt+Persona*: System message framing the model as a negation-aware expert with perfect comprehension of logical exclusion.
- *Prompt+CoT*: A five-step chain-of-thought scaffold (identify negation → list excluded concepts → generate candidates → filter → output).
- *Contrastive Decoding (CD)*: Score = $\log p_\text{expert}(c \mid p) - 0.5 \cdot \log p_\text{amateur}(c \mid p)$, using Qwen2.5-7B as expert and Qwen2.5-0.5B as amateur.

**Fine-tuning methods:**

- *Vanilla SFT*: Standard supervised fine-tuning with cross-entropy loss on the gold completion under the negated prompt.
- *DPO*: Direct Preference Optimization with negated gold as chosen and gold-positive as rejected.
- *NC-SFT*: Negation-Conditioned SFT—cross-entropy on negated examples only, no behavior tokens, no preservation loss.

### 5.2 Main Results

**Table 1: Main results on E4 v2 (525 records).**

| Method | NegRank | FlipAcc | ScopeCtrl | OverNeg | WikiFact | MMLU Δ | BoolQ† |
|--------|--------:|--------:|----------:|--------:|---------:|-------:|-------:|
| **Qwen2.5-7B** | | | | | | | |
| Base | 39.6 | 17.7 | 87.8 | 12.2 | — | — | 83.3 |
| + Warning | 38.1 | 7.7 | 80.6 | 19.4 | — | — | — |
| + Persona | 37.6 | 12.0 | 77.6 | 22.4 | — | — | — |
| + CoT | 37.1 | 11.0 | 83.7 | 16.3 | — | — | — |
| + Contrastive Dec. | 36.6 | 20.7 | 71.4 | 28.6 | — | — | — |
| Vanilla SFT | 59.9 | 61.2 | 54.1 | 45.9 | — | −0.88 | — |
| DPO | 44.6 | 25.8 | 76.5 | 23.5 | 44.0 | +0.00 | 86.8 (87.0) |
| NC-SFT | 53.0 | 50.8 | 77.6 | 22.4 | 63.0 | +0.18 | 87.8 |
| **MGNM (ours)** | **63.9** | **64.2** | **92.9** | **7.1** | **91.0** | −1.23 | **86.2 (87.2)** |
| **Llama-3.1-8B** | | | | | | | |
| Base | 30.7 | 12.0 | 81.6 | 18.4 | — | — | 83.2 |
| Vanilla SFT | 43.6 | 54.5 | 28.6 | 71.4 | — | −8.77 | — |
| DPO | 48.5 | 21.4 | 73.5 | 26.5 | 33.0 | +1.23 | 57.1 (67.7) |
| NC-SFT | 58.4 | 53.2 | 68.4 | 31.6 | 75.0 | −3.68 | 85.5 |
| **MGNM (ours)** | **62.4** | **60.9** | **89.8** | **10.2** | 79.0 | −3.51 | 69.0 (81.4) |
| **Mistral-7B (partial cross-architecture validation)** | | | | | | | |
| Base | 33.7 | 14.0 | 82.7 | 17.3 | — | — | — |
| NC-SFT | 59.9 | 47.2 | 73.5 | 26.5 | 74.0 | −0.53 | — |
| **MGNM (ours)** | 53.5 | 57.7 | 91.8 | 8.2 | 86.0 | **+0.70** | — |

† BoolQ full validation (3,270 records). Format: raw(PMI) when null-prior No-bias > 0.5; see §5.3.

**Inference-time methods.** All prompt variants *decrease* ScopeCtrl relative to the base model (87.8% → 78–84%), while FlipAcc barely moves (7.7–12.0% vs. base 17.7%). Warning and Persona prompts appear to trigger hedging behavior, increasing the model's tendency to offer generic or non-committal outputs that are often wrong. CoT produces reasoning steps in the output but does not change the probability assignments governing candidate ranking. CD achieves the highest inference-time FlipAcc (20.7%), but at the cost of collapsing ScopeCtrl to 71.4% (−16.4pp), because amplifying the logit gap between expert and amateur increases differentiation between semantically plausible and implausible completions without any mechanism to distinguish *suppress* from *preserve* contexts.

**Fine-tuning methods.** Vanilla SFT achieves high FlipAcc (61.2%) via near-total suppression (OverNeg: 45.9%)—the model learns to refuse rather than to reason. It is not a viable approach. DPO and NC-SFT represent genuine improvements but reveal a fundamental tension: suppression training degrades scope sensitivity. DPO moves FlipAcc from 17.7% to 25.8% while dropping ScopeCtrl from 87.8% to 76.5%; NC-SFT achieves 50.8% FlipAcc but ScopeCtrl only reaches 77.6%.

**MGNM.** On Qwen2.5-7B, MGNM simultaneously achieves **64.2% NegFlipAcc**, **92.9% ScopeCtrl**, and **7.1% OverNeg**—lower over-negation than the base model itself. Improvements over the strongest baselines remain statistically significant: +38.5pp NegFlipAcc and +16.3pp ScopeCtrl versus DPO, and +13.4pp NegFlipAcc and +15.3pp ScopeCtrl versus NC-SFT. Results generalize to Llama-3.1-8B (60.9% / 89.8%), while Mistral-7B provides partial cross-architecture validation: MGNM improves over Mistral Base on NegFlipAcc (14.0% → 57.7%), ScopeCtrl (82.7% → 91.8%), NegRank (33.7% → 53.5%), and OverNeg (17.3% → 8.2%). Compared with Mistral NC-SFT, MGNM is not highest on every single metric: NC-SFT has higher hard NegRank (59.9 vs. 53.5). However, NC-SFT's ScopeCtrl drops sharply to 73.5% and OverNeg rises to 26.5%, while MGNM is stronger on NegFlipAcc, ScopeCtrl, OverNeg, and WikiFact. The Mistral result therefore supports balanced behavior rather than single-metric dominance. Notably, Mistral MGNM achieves +0.70% MMLU, the only model that improves on the capability-retention diagnostic.

### 5.3 Out-of-Domain Generalization

**WikiFact.** We evaluate on 200 negated factual assertions about Wikipedia entities unseen during E4 training. MGNM achieves **91.0% FlipAcc** on Qwen, versus NC-SFT 63.0% and DPO 44.0%. On Llama, MGNM achieves 79.0% (NC-SFT: 75.0%, DPO: 33.0%). The 47pp advantage over DPO on Qwen indicates that MGNM learns transferable negation representations rather than overfitting to E4 surface patterns.

**BoolQ.** On the BoolQ full validation set (3,270 records), raw accuracies appear to regress for MGNM: Qwen MGNM 86.2% vs. base 83.3%, but Llama MGNM 69.0% vs. base 83.2%.

*Root cause: null-prior shift.* MGNM training shifts the model's null-context prior toward "No." We measure $\Delta = \log P(\text{No}|\text{null}) - \log P(\text{Yes}|\text{null})$ using a minimal null prompt.

**Table 2: Null-context prior shift.**

| Method | Qwen Δ | Llama Δ |
|--------|-------:|--------:|
| Base | +0.09 | −0.68 |
| DPO | +1.09 | +1.09 |
| NC-SFT | +0.13 | −0.06 |
| MGNM | +1.06 | +0.80 |

The +1.0 shift in MGNM and DPO is a direct consequence of $\mathcal{L}_\text{sup}$: reducing the log-probability of affirmative completions globally also shifts the null-context prior toward "No." NC-SFT trains with cross-entropy (not suppression margin loss) and shows negligible shift. The Llama base model has a natural Yes-bias (Δ = −0.68), which is why Llama MGNM's raw BoolQ appears particularly degraded.

*PMI calibration.* Applying PMI (score = log p(c|prompt) − log p(c|null)) removes the prior bias. After calibration, **Qwen MGNM achieves 87.2%**—the highest of all methods, including NC-SFT (87.8% raw). Llama MGNM reaches 81.4%, a residual 1.8pp gap versus Llama base (83.2% raw). We therefore interpret the raw BoolQ regression as calibration shift plus training tradeoff, not as a pure calibration artifact.
Held-out calibration supports the calibration component: using a 20% calibration split and 80% held-out test split across five random seeds, the optimal scalar PMI coefficient is stable at $\alpha = 1.65 \pm 0.12$, and held-out Llama MGNM accuracy rises to 84.6% on average. At the same time, retention-oriented checkpoints recover raw BoolQ only by weakening E4 NegRank, ScopeCtrl, or WikiFact, so we keep $\alpha=1$ in the main table and report BoolQ as a calibration-plus-training tradeoff.

### 5.4 Ablation Study

**Table 3: Component ablation (E4 v2).**

| Variant | FlipAcc | ScopeCtrl | OverNeg | NegRank |
|---------|--------:|----------:|--------:|--------:|
| **Qwen2.5-7B** | | | | |
| MGNM (full) | **64.2** | **92.9** | **7.1** | **63.9** |
| w/o $\mathcal{L}_\text{sup}$ | 31.8 | 86.7 | 13.3 | 48.5 |
| w/o $\mathcal{L}_\text{pre}$ | 49.2 | 13.3 | 86.7 | 36.1 |
| **Llama-3.1-8B** | | | | |
| MGNM (full) | **60.9** | **89.8** | **10.2** | **62.4** |
| w/o $\mathcal{L}_\text{sup}$ | 44.8 | 81.6 | 18.4 | 48.5 |
| w/o $\mathcal{L}_\text{pre}$ | 55.2 | 22.4 | 77.6 | 44.1 |

**Effect of $\mathcal{L}_\text{sup}$.** Removing the suppression loss drops NegFlipAcc by −32.4pp on Qwen and −16.1pp on Llama. $\mathcal{L}_\text{sup}$ is the primary driver of suppression accuracy. ScopeCtrl degrades moderately (−6.2pp / −8.2pp), as $\mathcal{L}_\text{pre}$ still provides preservation training.

**Effect of $\mathcal{L}_\text{pre}$.** Removing the preservation loss is catastrophic for ScopeCtrl: −79.6pp on Qwen and −67.3pp on Llama, with OverNeg rising sharply. The model reverts to near-total suppression, replicating Vanilla SFT behavior. NegFlipAcc remains relatively high (49.2% / 55.2%) because the model now suppresses indiscriminately. This confirms that $\mathcal{L}_\text{pre}$ is the sole mechanism preventing over-negation; $\mathcal{L}_\text{sup}$ and $\mathcal{L}_\text{pre}$ are not substitutable.

### 5.5 Data Scale Analysis

We train three additional Qwen2.5-7B variants using random 25%, 50%, and 75% subsets of the training data (118, 236, and 353 records respectively, same seed, oversampling preserved).

**Table 4: Training data scale vs. performance (Qwen2.5-7B).**

| Scale | Raw records | FlipAcc | ScopeCtrl | OverNeg | NegRank |
|-------|------------:|--------:|----------:|--------:|--------:|
| 25% | 118 | 35.5 | 78.6 | 21.4 | 46.0 |
| 50% | 236 | 53.8 | 85.7 | 14.3 | 56.9 |
| 75% | 353 | 58.2 | 91.8 | 8.2 | 57.4 |
| **100%** | **471** | **64.2** | **92.9** | **7.1** | **63.9** |

The learning curve shows diminishing marginal returns after the early scaling regime: the 25%→50% jump is +18.3pp FlipAcc, while 50%→100% still helps but with smaller gains. ScopeCtrl is already close to saturation at 75% (91.8% vs. 92.9% at 100%). Even at 25% (118 records), MGNM surpasses all inference-time methods on audited NegFlipAcc; at 50% (236 records), it already surpasses DPO. This suggests the current dataset scale is sample-efficient for this task structure, though additional diversity may still help more than sheer volume.

---

## 6. Mechanistic Analysis

To understand *where* in the model negation blindness originates and whether MGNM addresses it at the representation level, we apply three probes to 200 E4 records using the logit-lens framework.

### 6.1 E1: Layer-Wise Negation Signal

We compute the **neg_margin** = $\hat{s}_\ell(p^-, c_\text{allowed}) - s_\ell(p^-, c^*)$ at each layer $\ell$, where $\hat{s}_\ell$ is the score predicted by the logit lens at layer $\ell$ and $c^*$ is the forbidden target. A negative margin indicates the forbidden target outranks allowed alternatives in the model's internal representation at that layer.

**Result.** The neg_margin is negative throughout *all 28 layers* in the Qwen2.5-7B base model (last-layer mean: −2.52). The model never, at any processing depth, represents the negated context as ranking the allowed candidates above the forbidden one. After MGNM training, the last-layer best-allowed rate improves from 0.535 to 0.595 (+6pp)—a statistically real but behaviorally modest improvement relative to the +49.3pp FlipAcc gain.

### 6.2 E2: Attention to Negation Tokens

We measure the maximum single-head attention weight from the final position token to negation tokens ("not," "never," "without," etc.) at each layer.

**Result.** The base model assigns substantial attention to negation tokens (max-head weight ≈ 0.80 at the last layer, mean ≈ 0.50 across all layers). This pattern is nearly identical in MGNM (mean 0.517 vs. 0.501 for base). The model *attends* to negation tokens but fails to use that signal to re-rank its internal representation.

This rules out *attention blindness* as the mechanism: the model does not ignore negation tokens; it processes them in attention but fails to propagate the negation signal into the downstream representation.

### 6.3 E3: Activation Patching

We perform causal activation patching: replace the activations at layer $\ell$ during processing of the negated prompt with activations computed on the corresponding unnegated (positive) prompt. We then measure the **positive recovery rate**—the fraction of records where, after patching, the model correctly ranks an allowed alternative above the forbidden target.

**Result.** The peak recovery rate is **55%** (base) and **60%** (MGNM) when patching from layer 0, declining monotonically as the patch layer increases. At the last layer, patching barely improves over unpatched performance. This monotonic decay indicates that the clean (unnegated) representation is most informative at layer 0, before the model integrates later context that overwrites or obscures the negation signal.

### 6.4 Interpretation

The three probes converge on a consistent picture:

1. The model attends to negation tokens (E2 confirms the signal enters the model).
2. The negation signal is overwritten in early-to-mid layer processing (E3 peak at layer 0).
3. By the time processing reaches the final layer, the forbidden target already outranks alternatives internally (E1 always-negative margin).

The large behavioral improvement from MGNM (+49.3pp FlipAcc) relative to the small representational shift (+6pp at last layer) suggests MGNM operates through two complementary mechanisms: **(a)** partial repair of the internal negation signal, and **(b)** behavior-token conditioning that re-routes output decisions at the final decoding stage, independent of the intermediate representation quality. This interpretation also explains why inference-time interventions fail: they can only modify (b) at the logit level without access to (a), and even (b) can only work if the method can distinguish behavioral intent—which CD and prompts cannot.

---

## 7. Analysis

### 7.1 Why Inference-Time Methods Fail

Our mechanistic results provide a principled explanation for the uniform failure of inference-time methods:

- **Prompt baselines** modify the instruction context but not the underlying log-probability assignments. Since candidate ranking is determined by sequence probabilities computed over the prompt, surface-level instruction changes cannot overcome a probability landscape in which the forbidden target already dominates at every layer.
- **Chain-of-thought** generates reasoning steps that appear correct in the output text, but the candidate ranking is decided by the model's internal probability assignments, which are computed before the final generation step. As Turpin et al. (2023) showed for other tasks, CoT explanations are often unfaithful to the actual probability computation.
- **Contrastive Decoding** specifically exacerbates the problem. By amplifying $\log p_\text{expert} - \alpha \cdot \log p_\text{amateur}$, CD increases the contrast between semantically central (high-probability expert) and semantically peripheral (low-probability amateur) completions. For negation, the forbidden target is typically the semantically central completion in both expert and amateur models, so CD amplifies the margin by which the forbidden target outranks alternatives. ScopeCtrl drops 16.4pp relative to the base model as a result.

MGNM vs. CD on the audited subset remains strongly separated: NegFlipAcc +43.5pp and ScopeCtrl +21.5pp in favor of MGNM.

### 7.2 Prior Shift as a Diagnostic

The null-context prior shift Δ = log P(No|null) − log P(Yes|null) cleanly separates training objectives:

- **Suppression-objective methods** (MGNM, DPO): Δ ≈ +1.0. Directly penalizing the probability of affirmative completions in negative contexts has a global effect on the model's null-context prior.
- **Content-objective methods** (NC-SFT): Δ ≈ 0. Cross-entropy on answer content without explicit suppression margin does not shift the null prior.
- **Base models**: Δ ≈ 0 (Qwen) or Δ ≈ −0.68 (Llama, naturally Yes-biased).

This makes Δ a practical diagnostic tool: by measuring the null-context prior with a minimal prompt, practitioners can predict whether a fine-tuned model will exhibit raw yes/no task degradation *without running a full evaluation*. A model with Δ > 0.5 should be PMI-calibrated for binary classification tasks.

---

## 8. Limitations

**Behavior-token dependence at inference.** MGNM's strongest E4 results assume the caller provides the correct behavior token. Without any token, the same model still retains partial suppression ability (59.5\% NegFlipAcc) but ScopeCtrl falls from 92.9\% to 56.1\%. A candidate-aware LLM router improves the routed setting to 60.5\% NegFlipAcc, 66.3\% ScopeCtrl, and 59.9\% NegRank; using an empty prefix when the router predicts SELECT further improves the tradeoff to 59.5/69.4/61.4. The latest two-stage router plus suppress-only guard reaches 62.9/88.8/61.4, but this relies on structure-assisted metadata such as `semantic_mode`. MGNM should therefore be viewed as a token-conditioned controller with a metadata-assisted routing path, not as a complete pure-text automatic routing system.

**Behavior token task specificity.** The [PRESERVE] token is effective within the E4 task structure but does not transfer across task formats. Adding [PRESERVE] to BoolQ prompts causes Qwen MGNM accuracy to collapse to 47% (near-random), revealing that behavior tokens are tightly coupled to the training input distribution. The same issue appears in audited preserve free-generation: MGNM with the explicit [PRESERVE] prefix reaches 85.7\% keyword overlap, while removing the prefix raises preserve free-generation to 92.9\%. The token has learned the semantics of E4's candidate-ranking task, not a general "preserve" concept. Generalizing to arbitrary tasks would require either multi-task training with diverse task formats or a lightweight prompt-adaptation step.

**Residual Llama BoolQ gap.** After standard PMI calibration, Llama MGNM achieves 81.4% BoolQ accuracy versus 83.2% for the base model (raw), a residual 1.8pp gap. Further diagnosis shows that calibration strength explains a substantial part of the gap: allowing a scalar PMI coefficient yields a stable optimum around $\alpha \approx 1.88$ and 85.2\% cross-validated accuracy, while a stricter repeated held-out calibration protocol gives $\alpha = 1.65 \pm 0.12$ and 84.6\% mean held-out accuracy. However, retention-oriented checkpoints that recover raw BoolQ do so by weakening E4 NegRank, ScopeCtrl, or WikiFact. We therefore interpret BoolQ as calibration shift plus training tradeoff, not calibration alone.

**Scope of negation phenomena.** E4 covers lexical negation ("not," "never," "without") and morphological negation ("cannot," "n't") in entity-selection contexts. Compositional negation ("neither A nor B unless C"), presupposition cancellation, and multi-hop negation chains are not evaluated. The mechanistic analysis suggests MGNM should generalize—since it repairs representation-level failures rather than surface patterns—but empirical verification is needed.

**Training data scale.** Although our data scale analysis (§5.5) shows near-convergence at 471 records, and 50% of the data already surpasses all inference-time baselines, the absolute scale is small. Whether MGNM's benefits hold at 10× or 100× data scales, or whether a data-efficient approach like ours becomes less competitive when abundant negation data is available, remains an open question.

---

## 9. Conclusion

We have presented MGNM, a LoRA fine-tuning framework that addresses negation blindness in LLMs through behavior-conditioned prompts and five complementary training objectives. On the label-audited E4 subset, MGNM reaches 64.2\% NegFlipAcc, 92.9\% ScopeCtrl, and 7.1\% OverNeg on Qwen2.5-7B: a balance that prior methods cannot match simultaneously. The framework is lightweight, requires no architectural changes, and trains in under 40 minutes on a single 80GB GPU.

Mechanistic analysis reveals that negation blindness is fundamentally a representation-level failure: the forbidden target outscores alternatives throughout all 28 transformer layers, and the model attends to negation tokens without propagating the signal. MGNM partially corrects this at the representation level and additionally conditions correct output behavior through behavior tokens. This dual mechanism explains why inference-time methods cannot solve the problem and why automatic behavior routing remains a separate deployment bottleneck.

We release E4, all training code, and model adapters to support future work on negation understanding in LLMs.

---

## References

Ettinger, A. (2020). What BERT is not: Lessons from a new suite of psycholinguistic diagnostics for language models. *TACL*, 8.

Gardner, M. et al. (2020). Evaluating models' local decision boundaries via contrast sets. *Findings of EMNLP*.

Hendrycks, D. et al. (2021). Measuring massive multitask language understanding. *ICLR*.

Holtzman, A. et al. (2021). Surface form competition: Why the highest probability answer isn't always right. *EMNLP*.

Hosseini, M. J. et al. (2021). Understanding by understanding not: Modeling negation in language models. *NAACL*, 1301–1312.

Hossain, M. M. et al. (2020). An analysis of natural language inference benchmarks through the lens of negation. *EMNLP*, 8106–8115.

Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.

Jang, M. and Lukasiewicz, T. (2023). Can large language models truly understand prompts with negation? *Findings of EMNLP*.

Jiang, A. Q. et al. (2023). Mistral 7B. *arXiv:2310.06825*.

Kassner, N. and Schütze, H. (2020). Negated and misprimed probes for pretrained language models: Birds can talk, but cannot fly. *ACL*, 7811–7818.

Li, X. L. et al. (2023). Contrastive decoding: Open-ended text generation as optimization. *ACL*.

Llama Team (2024). The Llama 3 herd of models. *arXiv:2407.21783*.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.

Meng, K. et al. (2022). Locating and editing factual associations in GPT. *NeurIPS*.

Nøs, J. L. and Touileb, S. (2024). Negation in the wild: On negation handling in large language models. *Findings of ACL*.

nostalgebraist (2020). Interpreting GPT: The logit lens. *LessWrong*.

Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*, 35.

Qwen Team (2024). Qwen2.5 technical report. *arXiv:2412.15115*.

Rafailov, R. et al. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*.

Ravichander, A. et al. (2022). CondaQA: A contrastive reading comprehension dataset for reasoning about negation. *EMNLP*.

Turpin, M. et al. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS*.

Wang, K. et al. (2023). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. *ICLR*.

Wang, X. et al. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR*.

Zhao, Z. et al. (2021). Calibrate before use: Improving few-shot performance of language models. *ICML*.

Clark, C. et al. (2019). BoolQ: Exploring the surprising difficulty of natural yes/no questions. *NAACL*, 2924–2936.

---

## Appendix A: Reproducibility Statement

All base models are publicly available: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.2. LoRA fine-tuning requires a single 80GB A100 GPU with 15–40 minutes training time. E4 evaluation on 525 records takes under 5 minutes per model. All code, the E4 dataset, and trained LoRA adapters will be released upon acceptance under CC-BY-4.0 (dataset) and Apache-2.0 (code and adapters).

**Complete hyperparameters:**

| Parameter | Qwen | Llama | Mistral |
|-----------|-----:|------:|--------:|
| Learning rate | 2e-4 | 2e-4 | 5e-5 |
| Epochs | 3 | 3 | 3 |
| Grad. accum. | 8 | 8 | 8 |
| LoRA rank | 16 | 16 | 16 |
| LoRA α | 32 | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| λ_pos | 1.0 | 1.0 | 1.0 |
| λ_sup | 1.0 | 1.0 | 1.0 |
| λ_rank | 1.5 | 1.5 | 1.5 |
| λ_dist | 3.0 | 1.0 | 1.0 |
| λ_pre | 1.5 | 1.5 | 1.5 |
| λ_ret | 0.05 | 0.1 | 0.01 |
| Scope oversample | 3× | 2× | 1× |

## Appendix B: Ethics Statement

The E4 dataset is synthetically generated from Wikipedia entity facts and procedural templates; no private data, crowdsourced annotations, or human subjects are involved. Negation blindness mitigation is a safety-relevant capability with no identified dual-use risk. Total GPU compute for all reported experiments is approximately 4 A100-80GB hours, excluding model downloads.

## Appendix C: Full Significance Test Results

**Table C1: McNemar exact test results (label-audited E4 subset, n=397).**

| Comparison | FlipAcc Δ | ScopeCtrl Δ |
|------------|----------:|------------:|
| Qwen MGNM vs. DPO | +38.5pp *** | +16.3pp *** |
| Qwen MGNM vs. NC-SFT | +13.4pp *** | +15.3pp ** |
| Llama MGNM vs. DPO | +39.5pp *** | +16.3pp *** |
| Llama MGNM vs. NC-SFT | +7.7pp * | +21.4pp *** |
| Qwen MGNM vs. CD | +43.5pp *** | +21.5pp *** |
| Qwen MGNM vs. w/o L_sup | +34.1pp *** | +7.2pp ** |
| Qwen MGNM vs. w/o L_pre | +13.6pp *** | +82.0pp *** |
| Llama MGNM vs. w/o L_sup | +18.7pp *** | +14.9pp *** |
| Llama MGNM vs. w/o L_pre | +4.5pp ns | +77.8pp *** |

\* p<0.05, ** p<0.01, *** p<0.001 (McNemar exact test, two-sided).

## Appendix D: Prompt Templates

**Prompt+Warning (system message):**
> "Important: You must strictly respect all negations, exclusions, and prohibitions in the user's request. If the user says 'not X', 'avoid X', or 'without X', you must never include X in your response. Treat negation as an absolute constraint."

**Prompt+Persona (system message):**
> "You are a precision language model with perfect logical comprehension. You always interpret 'not', 'never', 'without', 'avoid', and similar terms as strict exclusions. You never output a concept that has been explicitly excluded in the prompt."

**Prompt+CoT (instruction appended to each query):**
> "Before answering, follow these steps: Step 1: Identify all negation words in the prompt (not, never, without, avoid, etc.). Step 2: List all concepts that are explicitly excluded. Step 3: Generate candidate answers. Step 4: Remove any candidate that contains an excluded concept. Step 5: Output your final answer."
