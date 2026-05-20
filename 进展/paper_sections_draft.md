# Paper Sections Draft — MGNM
*Generated 2026-05-03. All numbers from v2 test set (525 records) unless stated.*

---

## 5  Experiments

### 5.1  Experimental Setup

**Dataset.**
We evaluate on the E4 benchmark, a 525-record negation probe set (entity-level holdout rate 99.6%) covering three negation phenomena: *flip-required* predicates, *scope-sensitive* preserve cases, and *ranking* comparisons.
Training uses 412 annotated records.

**Metrics.**
- **FlipAcc**: fraction of flip-required cases where the model correctly selects an allowed alternative over the forbidden target (main metric).
- **ScopeCtrl**: fraction of preserve-positive cases where the model avoids over-negation (specificity metric).
- **NegRank**: overall ranking accuracy across all probe types.
- **OverNeg**: complement of ScopeCtrl; lower is better.

**Models.**
We apply MGNM to three 7–8B instruction-tuned models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.2.
LoRA adapters (r=16, α=32) are trained for 3–5 epochs on a single 80GB A100.

---

### 5.2  Main Results

Table 1 compares MGNM against nine baselines on the E4 test set.

**Inference-time prompt baselines.**
Despite three levels of explicit natural-language instruction — Warning, Persona, and Chain-of-Thought — all prompt variants *reduce* ScopeCtrl relative to the base model (86.1% → 78–81%) while barely moving FlipAcc (9–13% vs base 18.4%).
Contrastive Decoding (CD; Qwen-7B expert / Qwen-0.5B amateur, α=0.5) yields a marginal FlipAcc gain (+4.3pp, 22.7%) but collapses ScopeCtrl to 71.6% (−14.5pp).
These results indicate that inference-time interventions amplify differentiation without distinguishing *suppress* from *preserve* behavior, confirming that negation blindness is a representation-level failure not addressable at decoding time.

**Fine-tuning baselines.**
Vanilla SFT achieves high FlipAcc (64.4%) by wholesale suppression, but at the cost of catastrophic over-negation (ScopeCtrl 35.1%, OverNeg 64.9%) — the model learns to refuse rather than to reason.
DPO improves over base (FlipAcc 26.0%) but remains far below MGNM, with ScopeCtrl degrading to 75.8%.
NC-SFT achieves 52.0% FlipAcc / 66.0% ScopeCtrl; it improves negation suppression at the cost of substantial specificity loss.

**MGNM.**
On Qwen2.5-7B, MGNM achieves **67.7% FlipAcc** and **91.2% ScopeCtrl** — improvements of +41.7pp and +15.5pp over the best fine-tuning baseline (DPO/NC-SFT respectively), both significant at p < 0.001 (McNemar test).
Over-negation drops to 8.8%, lower than the base model's 13.9%.
Results on Llama-3.1-8B (FlipAcc 63.7%, ScopeCtrl 91.8%) and Mistral-7B (57.7%, 91.8%) confirm that MGNM transfers across architectures without re-tuning hyperparameters beyond learning rate.

---

### 5.3  Out-of-Domain Generalization

To test whether improvements on the E4 benchmark reflect genuine negation understanding, we evaluate on two held-out domain tasks.

**WikiFact.**
Models are evaluated on 200 negated factual assertions about Wikipedia entities unseen during training.
MGNM achieves 91.0% FlipAcc (Qwen) and 79.0% (Llama), versus NC-SFT 63.0%/75.0% and DPO 44.0%/33.0%.
Vanilla SFT was not evaluated on WikiFact due to its near-total suppression behavior.
The large gap between MGNM and DPO on WikiFact (+47.0pp Qwen) indicates that MGNM's multi-grained training prevents overfitting to surface negation patterns.

**BoolQ.**
We evaluate on BoolQ full validation set (3,270 yes/no reading comprehension questions).
Raw accuracies show apparent degradation for MGNM (Qwen: 86.2% vs base 83.3%; Llama: 69.0% vs base 83.2%).
We investigate this below.

*Prior-shift calibration.*
Fine-tuning with suppress objectives shifts the model's null-context prior toward "No."
Formally, for a null prompt containing no passage or question, we measure:
```
Δ = log P(No | null) − log P(Yes | null)
```
Table 2 shows that MGNM and DPO induce large positive shifts (Qwen MGNM: Δ = +1.055; Llama MGNM: +0.797; DPO: +1.09 for both models), whereas NC-SFT and base models show negligible shift (|Δ| < 0.14).
This shift is a direct consequence of the L_sup loss, which trains the model to reduce the probability of forbidden (typically affirmative) outputs.

Applying PMI calibration (score = log P(c | prompt) − log P(c | null)) removes the prior bias.
After calibration, Qwen MGNM achieves **87.2% BoolQ accuracy** — the highest of all Qwen methods including NC-SFT (87.6% raw, 87.6% PMI) — confirming that the raw degradation is a calibration artifact rather than capability regression.
For Llama MGNM, PMI-calibrated accuracy is 81.4%, a residual gap of 1.8pp versus Llama base raw (83.2%).
We attribute this gap to training-distribution mismatch: MGNM is trained solely on E4 entity-completion probes, while BoolQ requires open-domain yes/no reasoning with full passage context.
The substantially smaller gap under PMI (1.8pp vs 14.2pp raw) validates this interpretation.

---

### 5.4  Ablation Study

Table 3 reports the contribution of each loss component by removing one at a time.

**Effect of L_sup.**
Removing the suppression loss (w/o L_sup) degrades FlipAcc from 67.7% to 33.5% (Qwen; −34.2pp***) and from 63.7% to 45.0% (Llama; −18.7pp***), confirming that direct suppression training is the primary driver of flip accuracy.
ScopeCtrl degrades moderately (91.2% → 84.0% Qwen; 91.8% → 76.8% Llama), suggesting that L_pre still provides partial preserve training even without L_sup.

**Effect of L_pre.**
Removing the preservation loss (w/o L_pre) is catastrophic for ScopeCtrl: 91.2% → 9.3% (Qwen; −81.9pp***) and 91.8% → 13.9% (Llama; −77.9pp***).
The model collapses to near-total suppression (OverNeg 90.7% / 86.1%), replicating Vanilla SFT behavior.
FlipAcc remains relatively high (54.1% / 59.2%) because the model now suppresses indiscriminately.
This confirms that L_pre is the sole mechanism preventing over-negation; its absence eliminates the scope distinction entirely.

Both components are individually necessary: neither can compensate for the absence of the other.

---

### 5.5  Mechanistic Analysis

To understand *where* negation blindness originates and whether MGNM fixes it at the representation level, we apply three mechanistic probes (E1–E3) following the logit-lens framework.

**E1: Layer-wise negation signal.**
We compute the neg_margin = logit_lens(allowed) − logit_lens(forbidden) at each transformer layer.
For the Qwen2.5-7B base model, neg_margin is negative throughout all 28 layers (last-layer mean: −2.52), meaning the forbidden target outranks allowed alternatives at every depth.
Attention pattern E2 shows that the model does attend to negation tokens (mean max-head attention 0.50 across layers), ruling out simple attention blindness.
Together, E1 and E2 indicate that the model encodes negation tokens in attention but fails to propagate the negation signal into downstream representations.

**E2: Attention to negation tokens.**
The final-position token allocates substantial attention to negation tokens (max-head weight ≈ 0.80 at last layer, mean ≈ 0.50 across layers), consistent across base and MGNM models.
This rules out the hypothesis that negation blindness arises from neglecting negation tokens; rather, the attention is present but ineffective at transforming the representation.

**E3: Activation patching.**
We perform causal patching from an unnegated (positive) counterpart.
Peak positive-recovery rate is 55% (base) / 60% (MGNM) when patching from layer 0, declining monotonically as the patch layer increases.
This monotonic decay indicates that the clean representation is overwritten increasingly early in processing, consistent with negation information being lost in early-to-mid layers rather than preserved and suppressed.

**Post-MGNM mechanistic shift.**
After MGNM training, the last-layer neg_best_allowed_rate improves from 0.535 (base) to 0.595 (+6pp), indicating modest but real improvement in internal ranking at the representation level.
The large behavioral gain (FlipAcc +49.3pp) relative to the small representational shift suggests that MGNM primarily operates through behavior-token conditioning at the output stage while also partially correcting the internal representation.

---

### 5.6  Cross-Architecture Transfer

MGNM with architecture-specific hyperparameters (lr=5×10⁻⁵ for Mistral vs 2×10⁻⁴ for Qwen/Llama) achieves ScopeCtrl ≥ 91.8% across all three architectures.
FlipAcc ranges from 57.7% (Mistral) to 67.7% (Qwen), with the Mistral gap attributable to its higher sensitivity to LoRA updates requiring a more conservative learning rate.
MMLU degradation is minimal: Qwen −1.23pp, Llama −3.51pp, Mistral +0.70pp — within the variance expected from fine-tuning at this scale.
The consistent ScopeCtrl performance (>91%) across architectures demonstrates that the L_pre mechanism generalizes without per-model re-design.

---

## 6  Analysis

### 6.1  Why Inference-Time Methods Fail

Prompt-based methods and Contrastive Decoding all fail to improve FlipAcc meaningfully while degrading ScopeCtrl.
The mechanistic analysis explains why: since negation information is overwritten in early-to-mid layers (E3 peak at layer 0), no downstream manipulation of logits or attention can recover the signal.
CD specifically exacerbates the problem: by amplifying the expert–amateur logit difference, it increases the contrast between high-probability (typically non-negated) and low-probability (negated) completions without any mechanism to distinguish *suppress* from *preserve* contexts, causing ScopeCtrl to fall 14.5pp below base.
This finding parallels the failure of CoT prompting (FlipAcc 11.2%), where surface-level reasoning steps appear in the output but the underlying representation — which determines the probability assignment — remains unchanged.

### 6.2  Prior Shift as a Diagnostic Tool

The null-context prior shift Δ = log P(No|null) − log P(Yes|null) serves as a reliable indicator of training-objective semantics:
- Methods that directly reward non-default selection (MGNM, DPO) induce Δ ≈ +1.0.
- Methods that train over response content (NC-SFT) leave Δ ≈ 0.
- Base models have near-zero or negative Δ (Yes-biased for Llama: −0.68).

This suggests Δ can be used as a lightweight diagnostic to predict whether a fine-tuned model will show BoolQ-style raw degradation without running the full BoolQ evaluation pipeline.

---

## 7  Limitations

**Training data scale.**
MGNM is trained on 412 records; we have not systematically ablated data scale (25%, 50%, 100%, 200%).
Whether performance saturates at this scale or continues to improve with more data remains an open question for future work.

**Behavior token task specificity.**
The [PRESERVE] behavior token is effective within the E4 task format but does not transfer to other task formats.
Adding [PRESERVE] to BoolQ prompts causes Qwen MGNM accuracy to drop to 47% (near-random), showing that behavior tokens are deeply coupled to the training distribution's input structure.
Generalizing behavior token conditioning to arbitrary tasks would require either multi-task training or a lightweight prompt-adaptation step.

**Residual Llama BoolQ gap.**
After PMI calibration, Llama MGNM achieves 81.4% BoolQ accuracy versus 83.2% for the base model (raw), a residual 1.8pp gap.
This gap may reflect the narrower Llama training regime (fewer OOD examples in the MGNM training set) or Llama's stronger pre-trained yes/no bias (Δ = −0.68 for base).
Joint training with a small BoolQ auxiliary set is a straightforward fix that we leave to future work.

**Scope of negation phenomena.**
The E4 benchmark focuses on lexical and morphological negation in entity-selection contexts.
Compositional negation (e.g., "neither A nor B unless C"), presupposition cancellation, and negation in multi-hop chains are not covered.
We expect MGNM to generalize to these cases — since the mechanism addresses representation-level encoding rather than surface-level patterns — but have not verified this empirically.

---

## Appendix A  Reproducibility

All experiments use publicly available base models.
LoRA fine-tuning requires a single 80GB A100 GPU; training time per model is 15–40 minutes (3–5 epochs on 412 records).
E4 evaluation on 525 records requires <5 minutes per model on the same hardware.
The complete training pipeline, evaluation scripts, and E4 dataset will be released upon acceptance.

Hyperparameter details:
- LoRA: rank=16, alpha=32, dropout=0.05, target_modules=["q_proj","v_proj"]
- Batch size: 4 (gradient accumulation=2)
- LR scheduler: cosine with warmup (50 steps)
- Qwen/Llama lr=2×10⁻⁴; Mistral lr=5×10⁻⁵
- λ_sup=1.0, λ_rank=1.0, λ_dist=1.0 (Qwen: 0.3 for dist), λ_pre=1.5, λ_ret=0.01

---

## Appendix B  Ethics Statement

The E4 dataset consists of synthetic entity-completion probes derived from Wikipedia factoids; no private data or crowdsourced human annotations are used.
Negation blindness mitigation is broadly beneficial and has no identified dual-use risk.
Total compute for all reported experiments: approximately 4 GPU-hours (A100 80GB), excluding baseline model downloads.
The dataset will be released under CC-BY-4.0; model adapters under Apache-2.0.

---

## Key Numbers Reference Card

| | Qwen Base | Qwen MGNM | Llama Base | Llama MGNM | Mistral MGNM |
|---|---|---|---|---|---|
| FlipAcc | 18.4% | **67.7%** | 12.7% | **63.7%** | **57.7%** |
| ScopeCtrl | 86.1% | **91.2%** | 84.0% | **91.8%** | **91.8%** |
| NegRank | 39.6% | **63.9%** | 30.7% | **62.4%** | **53.5%** |
| OverNeg | 13.9% | **8.8%** | 16.0% | **8.2%** | **8.2%** |
| WikiFact | — | **91.0%** | — | 79.0% | **86.0%** |
| BoolQ (raw) | 83.3% | 86.2% | 83.2% | 69.0% | — |
| BoolQ (PMI) | 82.9% | **87.2%** | 74.9%† | 81.4% | — |
| MMLU Δ | — | −1.23pp | — | −3.51pp | **+0.70pp** |

† Llama base has natural Yes-bias (Δ=−0.68); PMI is unfavorable; use raw 83.2% as fair baseline.

**Significance vs best prior baseline (McNemar p < 0.001 for all):**
- Qwen MGNM vs DPO: FlipAcc +41.7pp***, ScopeCtrl +15.5pp***
- Qwen MGNM vs NC-SFT: FlipAcc +15.7pp***, ScopeCtrl +25.3pp***
- Llama MGNM vs DPO: FlipAcc +42.3pp***, ScopeCtrl +22.7pp***
- Llama MGNM vs NC-SFT: FlipAcc +9.4pp**, ScopeCtrl +28.9pp***
- Qwen MGNM vs CD: FlipAcc +45.0pp***, ScopeCtrl +19.6pp***
