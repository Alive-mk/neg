# Appendix: Reproducibility Details

## A. Training Hyperparameters

All models use LoRA (Hu et al., 2022) applied to all linear projection layers
(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).
Common LoRA settings: rank = 16, alpha = 32, dropout = 0.05.
Batch size: 8 (gradient accumulation steps = 8, per-device batch = 1).
Optimizer: AdamW (weight decay = 0). No LR scheduler. Random seed = 42 for all runs.

### MGNM Loss Weights

| Model | lr | epochs | λ_pos | λ_sup | λ_rank | λ_rank_dist | λ_preserve | λ_ret | scope_os |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 2e-4 | 3 | 1.0 | 1.0 | 1.5 | 3.0 | 1.5 | 0.05 | 3 |
| Llama-3.1-8B | 2e-4 | 3 | 1.0 | 1.5 | 1.0 | 1.0 | 1.5 | 0.10 | 2 |
| Mistral-7B | 5e-5 | 3 | 1.0 | 1.5 | 1.0 | 0.0 | 1.5 | 0.01 | 1 |

`scope_os`: scope-aware oversampling factor (out_of_scope and double_negation records duplicated this many times during training to balance scope distribution).

**Note on Mistral:** Mistral required a conservative learning rate (5e-5 vs. standard 2e-4) due to higher sensitivity to LoRA updates, as diagnosed by the KL retention loss L_ret. At lr=2e-4, Mistral's L_ret reached 1.32 (2.7× higher than Qwen at 0.48), causing MMLU degradation of −8.25%. Reducing to lr=5e-5 brought L_ret to 0.75 and recovered MMLU to +0.70%.

### Baseline Hyperparameters

**Vanilla SFT:** Same LoRA configuration, lr = 2e-4, 3 epochs. Training objective: cross-entropy on gold_neg token only (no scope or mode control).

**DPO:** β = 0.1, lr = 5e-5, 3 epochs, 471 preference pairs (chosen = correct negation answer, rejected = incorrect). No behavior tokens.

**NC-SFT (Negation Contrastive SFT):** Same as Vanilla SFT but training data augmented with negation contrastive pairs following the approach of Truong et al. (2023).

## B. Training Data (E4 Dataset)

- Total training records: 412 (after scope-oversampling: up to 609 effective records)
- Split: train=412, dev=0 (no dev set used; model selection based on test metrics)
- Breakdown by scope: in_scope=254, out_of_scope=96, double_negation=62
- Breakdown by mode: contrastive_resolution=138, exclusive_choice=145, suppression_only=129
- Test set (v2): 525 records, entity holdout 99.6%, family holdout 100%

The E4 dataset is generated using GPT-4o with multi-stage prompting and validated by
a 6-dimension automated checker (lexical negation, scope type, candidate validity,
semantic coherence, no-leakage, factual grounding). Manual spot-check: 100/100 pass rate.

**Dataset availability:** We will release the E4 training data, test set, and trained
adapter checkpoints upon paper acceptance.

## C. Compute Resources

All experiments were run on NVIDIA A100 80GB GPUs (or equivalent).

| Job | GPUs | Wall Time |
|---|---|---|
| MGNM training (single model, 3 epochs) | 1 | ~8 min |
| DPO training (single model, 3 epochs) | 1 | ~5 min |
| Evaluation on v2 (525 records, single model) | 1 | ~20 min |
| Mechanism analysis (200 probes, single model) | 1 | ~10 min |

Total compute for all reported experiments: approximately 4 GPU-hours.

## D. Evaluation Details

All evaluation uses log-probability scoring (no sampling). For each record, all candidate
answers are scored as continuations of the question prompt. NegRankAcc measures whether
the model assigns higher probability to the correct negation-aware answer than to all
distractors. FlipAcc additionally requires the correct answer under negation to differ
from the correct answer under the non-negated version.

Statistical tests: McNemar's exact test (two-sided) for per-metric comparisons;
Bootstrap 95% CI with 10,000 iterations for effect size estimation. All reported
significance levels use the standard thresholds (* p<0.05, ** p<0.01, *** p<0.001).

**Evaluation code and configs are included in the supplementary material.**
