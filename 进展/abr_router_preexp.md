# ABR 自动行为路由预实验

## 目标

测试 Agentic Behavior Router 是否能在不重新训练 MGNM 的前提下，为每条否定 prompt 自动选择 `SUPPRESS` / `PRESERVE` / `SELECT`。本轮先完成可复现实验骨架、dry-run 规则路由、E4 下游接入和 WikiFact suppress-only 诊断。

## 数据与输入

- E4 clean strict: `data/processed/validated_largetest_v2_clean_strict.jsonl`，397 条。
- WikiFact patched: `data/external/wikifact_neg_patched.jsonl`，100 条，全部按 `SUPPRESS` 离线评估。
- 构造输出：
  - `outputs/abr_router_preexp/e4_abr_prompt_only.jsonl`
  - `outputs/abr_router_preexp/e4_abr_prompt_candidates.jsonl`
  - `outputs/abr_router_preexp/wikifact_abr_inputs.jsonl`
  - `outputs/abr_router_preexp/input_summary.json`

E4 类别分布：`PRESERVE=98`，`SELECT=202`，`SUPPRESS=97`。

## 实现文件

- `scripts/build_abr_router_inputs.py`
- `scripts/run_abr_agent_router.py`
- `scripts/eval_abr_router.py`
- `scripts/eval_mgnm_with_abr_routes.py`

`gold_behavior` 只写入离线评估文件，不进入 ABR prompt。`SELECT` 在 downstream 中映射为空前缀。

## 运行命令

```bash
python scripts/build_abr_router_inputs.py

python scripts/run_abr_agent_router.py \
  --input outputs/abr_router_preexp/e4_abr_prompt_only.jsonl \
  --output outputs/abr_router_preexp/e4_abr_prompt_only_routes.jsonl \
  --mode prompt_only \
  --dry-run-rule-router

python scripts/run_abr_agent_router.py \
  --input outputs/abr_router_preexp/e4_abr_prompt_candidates.jsonl \
  --output outputs/abr_router_preexp/e4_abr_prompt_candidates_routes.jsonl \
  --mode prompt_candidates \
  --dry-run-rule-router

python scripts/eval_abr_router.py \
  --routes outputs/abr_router_preexp/e4_abr_prompt_only_routes.jsonl \
  --output outputs/abr_router_preexp/e4_abr_prompt_only_summary.json

python scripts/eval_abr_router.py \
  --routes outputs/abr_router_preexp/e4_abr_prompt_candidates_routes.jsonl \
  --output outputs/abr_router_preexp/e4_abr_prompt_candidates_summary.json
```

下游评测使用 `outputs/e4_qwen_r3v5_lp15/model_config_with_adapter.json` 中的 `qwen2_5_7b_e4`，并复用 `outputs/score_cache_router`：

```bash
python scripts/eval_mgnm_with_abr_routes.py \
  --models outputs/e4_qwen_r3v5_lp15/model_config_with_adapter.json \
  --input data/processed/validated_largetest_v2_clean_strict.jsonl \
  --routes outputs/abr_router_preexp/e4_abr_prompt_only_routes.jsonl \
  --output outputs/abr_router_preexp/e4_downstream_prompt_only.json \
  --cache-dir outputs/score_cache_router \
  --model-names qwen2_5_7b_e4 \
  --two-stage-summary outputs/eval_supervised_router_twostage_rule_textguard_empty_select_strict.json

python scripts/eval_mgnm_with_abr_routes.py \
  --models outputs/e4_qwen_r3v5_lp15/model_config_with_adapter.json \
  --input data/processed/validated_largetest_v2_clean_strict.jsonl \
  --routes outputs/abr_router_preexp/e4_abr_prompt_candidates_routes.jsonl \
  --output outputs/abr_router_preexp/e4_downstream_prompt_candidates.json \
  --cache-dir outputs/score_cache_router \
  --model-names qwen2_5_7b_e4 \
  --two-stage-summary outputs/eval_supervised_router_twostage_rule_textguard_empty_select_strict.json
```

WikiFact:

```bash
python scripts/run_abr_agent_router.py \
  --input outputs/abr_router_preexp/wikifact_abr_inputs.jsonl \
  --output outputs/abr_router_preexp/wikifact_abr_routes.jsonl \
  --mode prompt_candidates \
  --dry-run-rule-router

python scripts/eval_abr_router.py \
  --routes outputs/abr_router_preexp/wikifact_abr_routes.jsonl \
  --output outputs/abr_router_preexp/wikifact_abr_summary.json
```

## Routing 结果

当前结果来自 `--dry-run-rule-router`，不是 LLM agent ABR。

| Router | Accuracy | Macro-F1 | Suppress Recall | Preserve Recall | Select Recall |
| ------ | -------: | -------: | --------------: | --------------: | ------------: |
| ABR prompt-only dry-run | 62.7 | 62.4 | 88.7 | 48.0 | 57.4 |
| ABR prompt+candidates dry-run | 64.7 | 63.8 | 87.6 | 48.0 | 61.9 |
| Two-stage supervised | 92.9 | 93.4 | 94.8 | 93.9 | 91.6 |

dry-run 的主要失败是把 preserve 错分为 `SUPPRESS/SELECT`，并把很多 select/suppress 边界样本混淆。

## Downstream 结果

| Setting | NegFlipAcc | ScopeCtrl | OverNeg | NegRank |
| ------- | ---------: | --------: | ------: | ------: |
| Two-stage + guard | 61.9 | 88.8 | 11.2 | 61.4 |
| ABR prompt-only dry-run | 51.2 | 45.9 | 54.1 | 40.1 |
| ABR prompt+candidates dry-run | 52.8 | 46.9 | 53.1 | 42.6 |

`No token` 和 `Oracle token` 没有在本轮表内报告，因为可用本地 summary 与本次 397 条 clean-strict 输入或 adapter 口径不完全一致。

## WikiFact 结果

| Router | WikiFact Suppress Recall |
| ------ | -----------------------: |
| two-stage no guard | 29.0 |
| metadata/text guard | 100.0 |
| ABR prompt-only dry-run | 0.0 |
| ABR prompt+candidates dry-run | 0.0 |

dry-run 规则把 `What is NOT ...` 类 WikiFact prompt 全部路由到 `SELECT`，因此 suppress recall 为 0。这复现了 suppress-only 域外失败模式，但不能代表真实 LLM ABR。

## 错误分析

已生成：`outputs/abr_router_preexp/e4_abr_error_analysis.md`。

主要错误：

- `PRESERVE -> SUPPRESS/SELECT`：规则看到否定后误伤 out-of-scope 或 double-negation 样本，导致 ScopeCtrl 崩。
- `SUPPRESS -> SELECT`：`What/Which ... not ...` 被过度解释成 invalid-option selection。
- `SELECT -> SUPPRESS`：比较/反向选择类样本被当成简单目标压制。

## 结论

本轮完成了 ABR 预实验的可复现 pipeline，但真实 agent ABR 未运行。原因是当前环境缺少 `NEG_OPENAI_API_KEY` 和 `/tmp/.neg_api_key`。dry-run 规则路由结果低于 two-stage supervised router，且 downstream ScopeCtrl 明显低，不应作为 ABR 能力结论。

## 是否建议进入论文

不建议进入主表。当前可进入进展记录或方法附录的工程准备部分；真实 LLM ABR 跑完后，再判断是否放入 router 附录。

## 风险

- dry-run 规则不是 agent ABR，不能用于声称 agentic routing 失败或成功。
- downstream 只跑了 Qwen `e4_qwen_r3v5_lp15` adapter，未覆盖 Llama/Mistral。
- ABR prompt 固定但未用真实 LLM 输出验证 rationale 质量。
- WikiFact 当前只有 suppress recall，没有 ABR downstream NegFlipAcc。
