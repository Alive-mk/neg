# ABR 自动行为路由预实验

## 目标

测试 Agentic Behavior Router 是否能在不重新训练 MGNM 的前提下，为每条否定 prompt 自动选择 `SUPPRESS` / `PRESERVE` / `SELECT`，并评估接入已有 MGNM adapter 后的下游候选排名表现。

## 数据与输入

- E4 clean strict: `data/processed/validated_largetest_v2_clean_strict.jsonl`，397 条。
- WikiFact patched: `data/external/wikifact_neg_patched.jsonl`，100 条，全部按 `SUPPRESS` 离线评估。
- E4 类别分布：`PRESERVE=98`，`SELECT=202`，`SUPPRESS=97`。
- 构造输出：
  - `outputs/abr_router_preexp/e4_abr_prompt_only.jsonl`
  - `outputs/abr_router_preexp/e4_abr_prompt_candidates.jsonl`
  - `outputs/abr_router_preexp/wikifact_abr_inputs.jsonl`
  - `outputs/abr_router_preexp/input_summary.json`

`gold_behavior` 只写入离线评估文件，不进入 ABR prompt。ABR 只输出 route，不回答问题、不排序候选。`SELECT` 在 downstream 中映射为空前缀。

## 实现文件

- `scripts/build_abr_router_inputs.py`
- `scripts/run_abr_agent_router.py`
- `scripts/eval_abr_router.py`
- `scripts/eval_mgnm_with_abr_routes.py`

本轮追加了 `run_abr_agent_router.py --workers`，用于并发调用 OpenAI-compatible chat completions API；cache 写入使用线程锁，避免并发写坏 JSONL cache。

## 运行命令

```bash
python scripts/build_abr_router_inputs.py

python scripts/run_abr_agent_router.py \
  --input outputs/abr_router_preexp/e4_abr_prompt_only.jsonl \
  --output outputs/abr_router_preexp/e4_abr_prompt_only_routes.jsonl \
  --mode prompt_only \
  --api-key-file /tmp/.neg_api_key \
  --endpoint https://api.ai-gaochao.cn/v1/chat/completions \
  --model gpt-4.1-mini \
  --cache outputs/abr_router_preexp/abr_llm_cache.jsonl \
  --workers 4

python scripts/run_abr_agent_router.py \
  --input outputs/abr_router_preexp/e4_abr_prompt_candidates.jsonl \
  --output outputs/abr_router_preexp/e4_abr_prompt_candidates_routes.jsonl \
  --mode prompt_candidates \
  --api-key-file /tmp/.neg_api_key \
  --endpoint https://api.ai-gaochao.cn/v1/chat/completions \
  --model gpt-4.1-mini \
  --cache outputs/abr_router_preexp/abr_llm_cache.jsonl \
  --workers 4

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
  --output outputs/abr_router_preexp/wikifact_abr_prompt_only_routes.jsonl \
  --mode prompt_only \
  --api-key-file /tmp/.neg_api_key \
  --endpoint https://api.ai-gaochao.cn/v1/chat/completions \
  --model gpt-4.1-mini \
  --cache outputs/abr_router_preexp/abr_llm_cache.jsonl \
  --workers 4

python scripts/run_abr_agent_router.py \
  --input outputs/abr_router_preexp/wikifact_abr_inputs.jsonl \
  --output outputs/abr_router_preexp/wikifact_abr_routes.jsonl \
  --mode prompt_candidates \
  --api-key-file /tmp/.neg_api_key \
  --endpoint https://api.ai-gaochao.cn/v1/chat/completions \
  --model gpt-4.1-mini \
  --cache outputs/abr_router_preexp/abr_llm_cache.jsonl \
  --workers 4
```

验证命令：

```bash
python -m py_compile scripts/build_abr_router_inputs.py \
  scripts/run_abr_agent_router.py \
  scripts/eval_abr_router.py \
  scripts/eval_mgnm_with_abr_routes.py
```

## Routing 结果

结果来自固定 ABR prompt + `gpt-4.1-mini`，不是 dry-run 规则路由。四个真实 LLM 路由文件均为 0 个 parse/router error。

| Router | Accuracy | Macro-F1 | Suppress Recall | Preserve Recall | Select Recall |
| ------ | -------: | -------: | --------------: | --------------: | ------------: |
| ABR prompt-only | 51.1 | 50.6 | 61.9 | 54.1 | 44.6 |
| ABR prompt+candidates | 53.9 | 50.8 | 58.8 | 37.8 | 59.4 |
| Two-stage supervised | 92.9 | 93.4 | 94.8 | 93.9 | 91.6 |
| Two-stage + text guard | 92.9 | 93.4 | 94.8 | 93.9 | 91.6 |

Confusion matrix, prompt-only:

| Gold | SUPPRESS | PRESERVE | SELECT | PARSE_ERROR |
| ---- | -------: | -------: | -----: | ----------: |
| SUPPRESS | 60 | 35 | 2 | 0 |
| PRESERVE | 43 | 53 | 2 | 0 |
| SELECT | 69 | 43 | 90 | 0 |

Confusion matrix, prompt+candidates:

| Gold | SUPPRESS | PRESERVE | SELECT | PARSE_ERROR |
| ---- | -------: | -------: | -----: | ----------: |
| SUPPRESS | 57 | 24 | 16 | 0 |
| PRESERVE | 32 | 37 | 29 | 0 |
| SELECT | 54 | 28 | 120 | 0 |

候选池提高了 SELECT recall，但压低 preserve recall；总体 accuracy 仍只有约 54%，明显低于 supervised router。

## Downstream 结果

| Setting | NegFlipAcc | ScopeCtrl | OverNeg | NegRank |
| ------- | ---------: | --------: | ------: | ------: |
| Two-stage + guard | 61.9 | 88.8 | 11.2 | 61.4 |
| ABR prompt-only | 42.5 | 50.0 | 50.0 | 41.1 |
| ABR prompt+candidates | 47.5 | 51.0 | 49.0 | 45.5 |

按 gold_behavior 分组的下游 `neg_correct_accuracy`：

| Setting | SUPPRESS | PRESERVE | SELECT |
| ------- | -------: | -------: | -----: |
| ABR prompt-only | 73.2 | 50.0 | 41.1 |
| ABR prompt+candidates | 78.4 | 51.0 | 45.5 |

`No token` 和 `Oracle token` 没有在本轮表内自动填入，因为可用本地 summary 与本次 397 条 clean-strict 输入或 adapter 口径不完全一致。`outputs/abr_router_preexp/e4_downstream_summary.md` 中保留为 `NA`，避免混用口径。

## WikiFact 结果

| Router | WikiFact Suppress Recall | Downstream NegFlipAcc |
| ------ | -----------------------: | --------------------: |
| two-stage no guard | 29.0 | NA |
| metadata/text guard | 100.0 | NA |
| ABR prompt-only | 21.0 | NA |
| ABR prompt+candidates | 11.0 | NA |

WikiFact prompt-only 预测分布：`SELECT=79`，`SUPPRESS=21`。prompt+candidates 预测分布：`SELECT=89`，`SUPPRESS=11`。ABR 没有缓解 suppress-only 域外样本被错分为 SELECT 的问题，反而低于 two-stage no guard。

## 错误分析

已生成：`outputs/abr_router_preexp/e4_abr_error_analysis.md`。

主要错误：

- `SELECT -> SUPPRESS/PRESERVE`：E4 中大量 invalid/false/incorrect 选择任务被 ABR 当作一般否定压制或 out-of-scope preserve，导致 NegRank 低。
- `PRESERVE -> SUPPRESS/SELECT`：ABR 对 scope 的判断不稳定，候选池模式下 preserve recall 从 54.1 降到 37.8，带来高 OverNeg。
- `SUPPRESS -> PRESERVE/SELECT`：避免、排除、not used for 等 suppress prompt 与 preserve/selection cue 混淆。
- rationale 多数能复述 route 规则，但对真实 scope 边界解释不够具体，不能作为可靠可解释证据。

## 结论

按预设标准，这是失败但有价值的 ABR 预实验：E4 accuracy 低于 80%，Preserve recall 差，WikiFact suppress recall 也低于 two-stage no guard；接入 MGNM 后 ScopeCtrl 只有约 50%，明显低于 two-stage + guard 的 88.8。

结论应写为：agentic routing 在当前固定 prompt 和 `gpt-4.1-mini` 设置下不能替代 oracle behavior token 或 supervised/text-derived router；行为路由本身仍是独立难题。

## 是否建议进入论文

不建议进入主表。可以进入附录或 future work，作为 negative result 支持“自动行为路由不是由通用 LLM prompt 直接解决”的论点。主文仍应保留 supervised router 和 text-derived guard。

## 风险

- 只测试了一个 LLM router prompt、一个模型和一个 temperature=0 设置，不能排除更强模型或更好 prompt 改善结果。
- ABR prompt 固定后才跑完整评测，没有用测试标签调 prompt；但 prompt 本身可能不适配 E4 的具体三分类边界。
- downstream 只跑了 Qwen `e4_qwen_r3v5_lp15` adapter，未覆盖 Llama/Mistral。
- WikiFact 当前只报告 ABR suppress recall，没有 ABR downstream NegFlipAcc。
- API key 仅通过 `/tmp/.neg_api_key` 使用，未写入仓库。
