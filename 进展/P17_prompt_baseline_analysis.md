# Analysis: Why Prompt Engineering Baselines Fail on Negation

## Results (v2 test set, 525 records)

| Method | NegRankAcc | FlipAcc | ScopeCtrl | OverNeg |
|---|---|---|---|---|
| Qwen Base (no prompt) | 36.6% | 18.1% | 83.0% | 17.0% |
| Warning prompt | — | ~9% | — | — |
| Persona prompt | — | ~12.7% | — | — |
| CoT prompt | — | ~11.2% | — | — |
| **MGNM (ours)** | **63.9%** | **67.7%** | **91.2%** | **8.8%** |

(Full results in `outputs/eval_prompt_baseline_v2.json`)

**Key finding:** All three prompt strategies not only fail to improve FlipAcc but actively
reduce it below the no-prompt baseline (18.1% → 9-12%). Warning and CoT prompts are
particularly harmful.

## Why Prompts Fail: Proposed Analysis

**1. Contradiction between prompt intent and model prior.**
Prompts like "Pay careful attention to negation words" implicitly signal that negation is
difficult or potentially misleading. For a model already predisposed to assign high probability
to plausible-sounding completions, this caution cue activates a *hedging strategy*: the model
becomes more conservative and defaults to the most statistically common completion, which is
usually the non-negated (incorrect) answer.

This is consistent with the "representation-level" hypothesis: the model's internal representation
of negation-constrained prompts already discards the negation signal by mid-layers (see E1 probe,
Section 5). No surface-level instruction can override this failure if the model never encodes the
negation semantics in its intermediate representations.

**2. CoT reasoning without negation-aware training is circular.**
The CoT prompt instructs the model to "locate negation words" and "invert the expected answer."
However, for a model that is negation-blind at the representation level, the CoT reasoning steps
are nominal: the model generates step 1 (locate negation) and step 2 (invert), but the final
answer selection still relies on the corrupted intermediate representations. The reasoning is
present in surface output but not causally connected to the answer. This mirrors findings by
Golovneva et al. (2022) showing that chain-of-thought prompting can fail when the underlying
computation is unsupported by the model's parametric knowledge.

**3. Persona prompts increase false confidence.**
Persona-style prompts ("You are a precise reasoner who correctly handles negation") increase the
model's confidence score for whatever answer it was already going to generate, rather than changing
which answer is selected. This false confidence can *suppress* the marginal probability mass that
would otherwise land on correct alternatives, reducing FlipAcc below baseline.

**4. Structural distinction from MGNM.**
These failure patterns collectively support the core claim of our paper: negation blindness is a
**representation-level deficiency** (visible in E1 probes, where the model's intermediate states
consistently rank negation-target tokens above allowed alternatives), not a *generation-level* or
*instruction-following* failure. MGNM addresses it at the correct level — directly reshaping the
model's scoring behavior via discriminative training — whereas prompts operate at the surface level
and cannot reorganize internal representations.

**Citation targets:**
- Golovneva et al. (2022): "Roscoe: A suite of metrics for scoring step-by-step reasoning" (CoT failure modes)
- Jang & Lukasiewicz (2023): "Can Large Language Models Truly Understand Prompts?" (prompt instruction limitations)
- This work's E1/E3 mechanism probes (Sections 5.1-5.2): causal evidence that the failure is pre-linguistic
