<claude-mem-context>
# Memory Context

# [neg] recent context, 2026-05-19 1:19pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 26 obs (10,433t read) | 100,136t work | 90% savings

### Apr 24, 2026
3 8:06a 🔵 否定失明实验 4.23 日报：行为 token 机制分析与 R3v4 最优方案确定
6 8:08a 🔵 April 23 ML Training and Evaluation Runs in /data/mingkai/neg
7 8:09a 🔵 r3v4 Phase 2 Training: Adapter Chaining on 90-Record select_gold_neg Dataset
8 8:10a 🔵 r3v4_p2 Evaluation Results: qwen2_5_7b_e4 on largetest (n=392)
9 " 🔵 r3v4 Phase 1 vs Phase 2 Metric Comparison: Phase 2 Regresses on Most Scores
10 8:11a 🔵 Three-Model Comparison on largetest r3v4: llama leads PosAcc/FlipAcc, mgnm leads ScopeCtrl
11 " 🔵 Phase 2 Adapter Chaining Implemented via PeftModel.from_pretrained with is_trainable=True
12 " 🔵 r3v4 WikiFact vs largetest: Both Models Score Much Higher on WikiFact, FlipAcc Jumps ~20pp
13 8:14a 🔵 train_mgnm.py Added --behavior-filter and --init_adapter_path Arguments Post-4.23 Progress Note
14 8:16a ✅ 进展/4.23.md Updated with R3v4 Phase 2 Experiment Results and Decision
S9 MGNM Negation Model: Root-cause analysis and targeted fixes for over-negation and NegRank failures in R3v4 (Apr 24, 8:16 AM)
S3 进展/4.23.md Updated with R3v4 Phase 2 Experiment Results and Decision (Apr 24, 8:16 AM)
16 8:23a 🔵 MGNM Training Run r3v4 Shows Overfitting Pattern Across Both Qwen and Llama Models
17 " 🔵 MGNM R3v4 Eval Breakdown Reveals Over-Negation on Out-of-Scope Items as Primary Failure Mode
21 8:25a 🔵 MGNM P2 Model Has Worse Over-Negation Than R3v4 — 31.3% vs 20.9% on Out-of-Scope Items
22 8:26a 🔵 Training Data Scope Imbalance (3:1 in_scope vs out_of_scope) Root Cause of Over-Negation Failures
24 8:35a 🔵 All MGNM Training Splits Have In-Scope Majority — Confirmed Root Cause of Scope Boundary Failure
27 " 🔵 MGNM Training Loop Processes One Record at a Time with Gradient Accumulation — No True Batching
29 8:36a 🔵 MGNM Experiment Infrastructure: 29 Active Tmux Sessions Running Concurrent Ablations and Evals
30 8:37a 🔵 MGNM train_mgnm.py Full Hyperparameter Defaults and Phase-2 Fine-Tuning Capability Confirmed
31 8:38a 🟣 Added --scope-oversample Flag to train_mgnm.py to Fix Out-of-Scope Training Imbalance
33 " 🔴 Scope Oversampling Logic Implemented in train_mgnm.py to Fix Out-of-Scope Training Underrepresentation
34 " 🔵 Merged Training Split Lacks dev.jsonl — Only balanced Split Has Full Train/Dev/Test
35 " 🟣 R3v5 Qwen Training Launched with Scope Oversampling and Boosted Preserve/Rank Lambda Weights
36 8:39a ✅ R3v5 Training Confirmed Started — Scope Oversampling Added 197 Records, Total 609
### Apr 26, 2026
S10 用户发送简单问候"你好" (Apr 26, 12:47 PM)
37 12:48p ✅ 4.25进展文档创建 — neg-blindness过度否定根因分析总结
38 12:49p 🔵 4月25日实际工作内容确认 — llama_r3v5_noos模型训练与评估
40 12:50p 🔵 neg项目tmux会话全景 — 4月24日密集实验轨迹确认

Access 100k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>

# Neg 项目 Agent 协作协议

## 语言和沟通风格

- 使用中文解释。
- 使用 Linux bash 命令。
- 给出命令时，尽量给可直接复制执行的版本。
- 先解释原因，再给修复方案或下一步动作。
- 保持简洁，重点说明改了哪些文件、运行了哪些命令、结果是什么、还有什么风险。

## 项目使命

本仓库研究生成式大语言模型中的 **negation blindness**：模型在否定语境中仍然偏向原正例、错误选择被否定目标，或无法稳定控制否定作用域。

本项目的核心目标不是单纯提高某一个否定指标，而是系统回答：

> 否定处理是否是一种单一能力？如果不是，模型在 suppress、preserve、select 三类否定行为上的失败机制是否不同？MGNM 是否能够在提升否定控制能力的同时避免 over-negation，并且这种提升是否可复现、可泛化、可解释？

每次代码、数据或实验改动，都必须服务于至少一个明确研究问题，并能产出可进入论文、附录或 release artifact 的结果。

## 研究问题

### RQ1：否定盲目性能否被拆解成不同类型的行为失败？

本项目不把所有否定样本都视为“答案翻转”。E4/E4v3 应当区分三类行为：

1. `suppress_target`：否定后应压制原正例目标；
2. `select_gold_neg`：否定后应选择正确负例目标；
3. `preserve_positive`：否定不作用于目标概念时，应保留正例答案。

需要回答的问题：

- 模型是否在 suppress、select、preserve 三类行为上呈现不同失败模式？
- NegFlipAcc 的提升是否掩盖了 ScopeCtrl 的下降？
- 普通 SFT、DPO、NC-SFT 是否会导致 over-negation？
- SELECT 样本是否存在 single-gold 标注过严的问题？

判断依据：

- NegFlipAcc
- NegSuppRate
- Hard NegRank
- Multi-answer NegRank
- ScopeCtrl
- OverNeg

### RQ2：MGNM 是否能在提升否定控制的同时避免 over-negation？

MGNM 的目标不是简单提高 suppress，而是同时实现：

1. 被否定目标能够下降；
2. 正确负例能够上升；
3. preserve 样本中的正例答案不被误伤。

需要回答的问题：

- MGNM 是否优于 Base、Vanilla SFT、DPO、NC-SFT、Prompt、CoT、Contrastive Decoding？
- MGNM 的优势是否来自真正的多粒度建模，而不是简单压制所有正例？
- 去掉 `L_sup` 或 `L_pre` 后，是否会出现预期退化？
- 行为 token 是否是 MGNM 条件控制能力的关键来源？

判断依据：

- 主表：NegFlipAcc、ScopeCtrl、OverNeg、NegRank
- 消融表：w/o `L_sup`、w/o `L_pre`
- behavior token ablation
- no-token / wrong-token / oracle-token 对照
- McNemar 显著性检验

### RQ3：MGNM 的提升是否具有泛化性，而不是模板记忆？

需要回答的问题：

- 模型是否只记住 E4 模板？
- 在 unseen entity、unseen family、unseen template 或 WikiFact 上是否仍然有效？
- MGNM 是否能迁移到不同模型架构，例如 Qwen、Llama、Mistral？
- 在公开否定 benchmark 或外部构造任务上是否仍有相对优势？

判断依据：

- largetest / clean strict test
- WikiFact
- unseen-template split
- public negation benchmark
- Qwen/Llama/Mistral 跨架构结果
- MMLU sanity check
- BoolQ retention / calibration 诊断

### RQ4：自动行为路由是否可行？它的边界在哪里？

MGNM 的最强结果通常依赖 oracle behavior token，因此必须单独评估 router。

需要回答的问题：

- router 能否从 prompt 或候选池中判断 suppress / preserve / select？
- two-stage router 是否优于 single-stage router？
- metadata guard、text-derived suppression guard 是否能修复 suppress-only 域外失败？
- router 的提升是否依赖数据集内部字段，还是可以从纯文本规则中推断？
- 自动路由失败主要来自哪一类错误，例如 preserve 被误分为 select，或 suppress 被误分为 select？

判断依据：

- router classification accuracy
- suppress recall
- preserve recall
- select accuracy
- downstream NegFlipAcc / ScopeCtrl / NegRank
- WikiFact suppress-only recall
- metadata guard vs text-derived guard 对比

### RQ5：否定盲目性的内部机制是什么？

需要回答的问题：

- 模型是否真的没有注意到否定词？
- 否定信号在哪些层被保留或丢失？
- MGNM 的行为提升是否对应内部表示改善？
- behavior token 是否主要影响最终解码决策，而不是完全修复中间表示？
- correct token、wrong token、no token 的机制差异是什么？

判断依据：

- logit lens
- neg_margin
- attention attribution
- activation patching
- correct-token vs wrong-token probe
- Base vs MGNM 中间层对比

## 实验方法与执行规范

### 1. 数据实验

数据相关实验必须明确说明：

- 使用哪个数据版本：E4 v2、clean strict、E4v3、WikiFact、BoolQ、MMLU；
- 使用哪个 split：train、dev、test、audited test、unseen-template test；
- 是否使用 multi-answer labels；
- 是否存在 entity / family / template leakage；
- 是否改变了 candidate pool 或 prompt 格式；
- 是否改变了评价指标。

每次数据改动必须输出：

```text
data/...jsonl
outputs/...summary.json
outputs/...audit.tsv
进展/YYYY-MM-DD.md
```

数据实验必须检查：

```text
1. train/test entity 是否重叠
2. family/template 是否泄漏
3. candidate pool 是否包含 gold
4. preserve 样本是否被误标为 suppress/select
5. select 样本是否存在多个合理 negative answer
6. multi-answer labels 是否和 hard single-gold 同时保留
```

### 2. 主模型训练实验

每次训练实验必须记录：

- base model
- adapter 名称
- train file
- epoch
- learning rate
- LoRA config
- loss weights
- random seed
- 是否使用 scope oversampling
- 是否使用 behavior token
- 是否从已有 adapter 继续训练

训练输出至少包括：

```text
outputs/<run_name>/training_summary.json
outputs/<run_name>/config.json
outputs/<run_name>/eval_clean_strict.json
outputs/<run_name>/eval_wikifact.json
outputs/<run_name>/mmlu.json 或 boolq.json
```

每次训练完成后，至少运行项目内对应评测脚本。若当前标准脚本尚未拆分为下面这些文件，应使用现有等价脚本，并在进展记录中写清楚映射关系：

```bash
python scripts/eval_clean_strict.py ...
python scripts/eval_wikifact.py ...
python scripts/eval_boolq.py ...   # 如果该模型涉及 calibration/retention
python scripts/eval_mmlu.py ...    # 如果该实验声称能力保留
```

若实验目标是论文主结果，必须同时跑对应 baseline 或引用已有 baseline 文件，不能只报告 MGNM 单独结果。

### 3. 主评测实验

主评测必须同时报告：

```text
NegRank
NegFlipAcc
ScopeCtrl
OverNeg
NegSuppRate
Hard NegRank
Multi-answer NegRank
WikiFact
MMLU delta
BoolQ raw / PMI
```

其中：

- `Hard NegRank` 是 strict single-gold 指标；
- `Multi-answer NegRank` 是人工或规则验证的多答案指标；
- `ScopeCtrl` 必须和 `OverNeg` 一起报告；
- `NegFlipAcc` 不能单独作为结论依据；
- BoolQ 需要区分 raw、standard PMI、scalar PMI；
- MMLU 如果只是 570 题子集，必须标注为 sanity check。

主表输出建议：

```text
outputs/main_table_clean_strict.csv
outputs/main_table_wikifact.csv
outputs/main_table_boolq.csv
outputs/main_table_mmlu.csv
outputs/main_table_combined.md
```

### 4. Router / Reaggregation 实验

router 实验必须明确区分：

| 路由类型 | 说明 |
| --- | --- |
| oracle token | 已知正确行为标签 |
| no token | 不提供行为 token |
| LLM router | 用 LLM 根据 prompt/candidates 判断 |
| supervised router | 训练分类器预测行为 |
| metadata guard | 使用数据字段，如 semantic_mode |
| text-derived guard | 使用 prompt 文本规则判断 explicit exclusion |
| empty SELECT | router 预测 select 时不加 token |

router 实验必须报告：

```text
classification accuracy
suppress recall
preserve recall
select accuracy
downstream NegFlipAcc
downstream ScopeCtrl
downstream NegRank
WikiFact suppress recall
WikiFact downstream NegFlipAcc
```

必须检查：

```text
1. router 是否使用了 test label
2. 阈值是否在 dev/calibration set 上选择
3. 是否把 metadata guard 伪装成纯文本 router
4. 是否在 WikiFact 上单独验证 suppress-only 泛化
5. SELECT empty prefix 是否改变评测口径
```

router 结果不能直接宣称“自动路由已解决”。若依赖 metadata 或规则，应写成：

```text
structure-assisted routing
metadata-assisted routing
text-derived suppression guard
```

### 5. Free-generation 实验

free-generation 只是候选排名之外的补充评测，不作为主 benchmark。

必须区分：

| 类型 | 推荐设置 |
| --- | --- |
| suppress free-gen | MGNM oracle `[SUPPRESS]` |
| preserve free-gen | MGNM no-token |
| visible `[PRESERVE]` | 仅作为 ablation |
| hidden-control | 仅作为失败诊断或附录 |
| generate-5 pass@5 keyword | 仅作为 upper-bound diagnostic |

free-generation 自动指标需要说明：

```text
keyword coverage is an automatic proxy, not full semantic evaluation
```

若进行人工审计，需要输出：

```text
outputs/freegen_human_audit_sample_100.tsv
outputs/freegen_human_audit_summary.json
```

人工审计字段：

```text
prompt
model_output
keyword_result
human_label
error_type
```

人工标签标准：

```text
correct: 输出语义上满足 prompt，并正确压制/保留目标概念
wrong: 输出违反 prompt、包含被禁概念、误伤应保留概念或跑题
```

### 6. BoolQ / Calibration / Retention 实验

BoolQ 实验必须区分：

```text
raw accuracy
standard PMI, alpha=1
scalar PMI, alpha tuned on calibration set
held-out scalar PMI
null-prior delta
```

不允许只报告最优 calibration 值而不说明 calibration protocol。

每次 BoolQ 实验必须输出：

```text
outputs/boolq_<run_name>_raw.json
outputs/boolq_<run_name>_pmi.json
outputs/boolq_<run_name>_calibration.json
```

如果某 checkpoint 恢复 BoolQ raw，但 E4 / WikiFact 下降，应归类为：

```text
training tradeoff
```

不要把它写成主模型改进。

### 7. MMLU / 能力保留实验

MMLU 当前为轻量 sanity check，不是完整能力评测。

报告时必须写：

```text
MMLU 570-question diagnostic subset
```

不能写成 full MMLU。

若模型在 MMLU 上提升小于 1pp，只能写：

```text
no observable degradation
```

不要写成明显能力提升。

### 8. 机制分析实验

机制分析必须说明：

- probe 数量
- probe 来源
- 是否为 candidate-ranking setting
- 分析对象是 Base 还是 MGNM
- 是否使用 behavior token
- 结论是否只限于当前 setting

推荐机制实验包括：

```text
E1 logit lens
E2 attention to negation tokens
E3 activation patching
correct-token vs wrong-token comparison
suppress / preserve / select 分类型分析
```

机制结论必须避免过度外推：

```text
The finding is limited to E4 candidate-ranking probes.
```

## 最终预期产出

项目最终应收敛到以下可复现产物。

### 1. 代码仓库

必须包含或逐步补齐：

```text
scripts/train_mgnm.py
scripts/eval_clean_strict.py
scripts/eval_wikifact.py
scripts/eval_boolq.py
scripts/eval_mmlu.py
scripts/eval_freegen.py
scripts/eval_router.py
scripts/aggregate_main_tables.py
scripts/check_leakage.py
```

要求：

- 无硬编码绝对路径；
- 支持 config 或 CLI 参数；
- 支持 random seed；
- 输出 JSON/CSV/Markdown 三类结果；
- 所有主表可由脚本重现。

### 2. 数据 release

至少包含：

```text
data/release/e4v2_clean_strict.jsonl
data/release/e4v3_train.jsonl
data/release/e4v3_test.jsonl
data/release/e4v3_multianswer.jsonl
data/release/wikifact_eval.jsonl
```

每条样本应包含：

```text
id
behavior
domain
template_id
entity_family
positive_prompt
negative_prompt
candidate_pool
positive_gold
negative_gold_single
negative_gold_multi
valid_negative_candidates
split
holdout_group
```

### 3. 主结果表

论文主表应能由脚本自动生成：

```text
paper/tables/table_main_e4.tex
paper/tables/table_wikifact_boolq_mmlu.tex
paper/tables/table_ablation.tex
paper/tables/table_router.tex
paper/tables/table_multianswer.tex
paper/tables/table_freegen.tex
```

每张表必须有对应 source 文件：

```text
outputs/tables/table_main_e4.csv
outputs/tables/table_router.csv
...
```

### 4. 实验日志

每个重要实验必须在 `进展/` 中留下记录：

```text
进展/YYYY-MM-DD.md
```

记录格式：

```text
## 今日目标

## 修改文件

## 运行命令

## 关键结果

## 是否进入论文

## 风险和下一步
```

### 5. 论文材料

最终应准备：

```text
paper/main.tex
paper/appendix.tex
paper/figures/
paper/tables/
paper/references.bib
paper/checklist.tex
```

论文材料必须包含：

- method
- E4/E4v3 数据说明
- main results
- ablations
- router
- free-generation
- BoolQ calibration
- SELECT multi-answer audit
- mechanism
- limitations
- ethics
- reproducibility appendix

## 基于 Git 的 Worker/Judge 工作流

两个 Codex 终端通过 Git 通信。

### Worker 角色

Worker 负责实现改动、运行实验、记录结果、提交 commit 并 push 分支。

Worker 默认流程：

```bash
cd /data/mingkai/neg
git pull --ff-only
git switch -c worker/task-name-$(date +%Y%m%d-%H%M)
```

完成实现后：

```bash
git status --short
git diff
git add <changed-files>
git commit -m "Short factual message"
git push -u origin HEAD
```

Worker 规则：

- 不要使用 `git add .`，除非任务明确要求提交整个项目。
- 只提交和当前任务相关的文件。
- push 前运行最小必要验证。
- push 后告诉 judge 分支名、验证命令和验证结果。
- 如果 judge 指出问题，在同一个 worker 分支上修复并再次 push。

每个 Worker 分支结束时必须提供：

```text
分支名：
修改文件：
新增文件：
运行命令：
核心结果：
是否改变评测口径：
是否存在风险：
建议是否进入论文：
```

### Judge 角色

Judge 负责评审 worker 分支，并给出具有科研判断的下一步建议。默认不直接改代码。

Judge 默认流程：

```bash
cd /data/mingkai/neg
git fetch origin
git branch -r
```

在独立 worktree 中评审 worker 分支：

```bash
cd /data/mingkai/neg
git fetch origin
BRANCH="worker/replace-with-branch-name"
git worktree add ../neg-judge "origin/$BRANCH"
cd ../neg-judge
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
```

Judge 规则：

- 默认不要修改 worker 的工作区。
- 除非用户明确要求，不要 reset、delete 或 rewrite 分支。
- 同时评审代码正确性和研究有效性。
- 优先检查 bug、数据泄漏、评测口径漂移、缺少对照、不可复现路径、大文件误入库等问题。
- 条件允许时运行轻量验证。
- 每次评审最后必须给出一个明确结论：`通过`、`需要修改` 或 `阻塞`。

每个 Judge 评审必须回答：

```text
这个改动回答了哪个研究问题？
结果是否可信？
是否存在数据泄漏？
是否改变 metric 定义？
是否需要补 baseline？
是否能进入论文主表/附录/release？
结论：通过 / 需要修改 / 阻塞
```

## Judge 评审模板

评审建议使用以下格式：

```text
结论：通过 / 需要修改 / 阻塞

主要问题：
- 文件路径：问题原因；为什么影响研究可信度；建议怎么改。

实验判断：
- 这个改动回答了什么研究问题？
- 是否改变了评测口径？
- 是否可能数据泄漏或指标虚高？
- 是否需要补对照实验、统计显著性或人工质检？
- 是否影响最终论文表格或数据 release？

验证：
- 已运行：<command>
- 结果：<pass/fail/blocked>

下一步：
- 给 worker 1-3 条具体、可执行任务。
```

## 当前优先级

### P0：必须保持稳定

- clean strict 主评测
- Qwen/Llama/Mistral 主结果
- SELECT Hard + Multi-answer 指标
- router oracle / no-token / text-derived guard
- BoolQ raw / PMI / tradeoff
- WikiFact
- MMLU sanity check

### P1：建议补强

- free-generation 100 条人工审计
- unseen-template split
- public negation benchmark subset
- Mistral loss ablation
- E4v3 扩展版

### P2：可选探索

- pure-text router
- correct-token vs wrong-token mechanism
- larger MMLU
- more public benchmarks
- larger-scale release dataset

## 接受实验结果的最低标准

一个实验要进入论文主表或附录，至少满足：

```text
1. 有明确研究问题
2. 有对应 baseline
3. 有固定数据 split
4. 有可复现命令
5. 有 JSON/CSV 输出
6. 没有数据泄漏
7. 没有 metric drift
8. 结论不依赖单个 cherry-picked example
```

如果不满足这些条件，只能进入进展记录，不能进入论文主结果。

## 研究风险检查清单

接受 worker 分支前，检查：

- Evaluation split leakage：family/entity/template 或 held-out group 是否污染。
- Metric drift：metric 定义、candidate pool、prompt 或 aggregation 逻辑是否变化。
- Over-negation：否定指标提高的同时，`preserve_positive` 或 scope control 是否变差。
- Prompt/token dependence：提升是否只来自显式 behavior token 或 prompt 格式。
- Incomplete baselines：相关场景下是否缺少 vanilla SFT、base model、DPO、contrastive 或 router 对照。
- Missing sanity checks：是否缺少 MMLU 或非否定能力 retention 检查。
- Irreproducibility：是否存在硬编码绝对路径、缺配置、缺随机种子、模型路径未说明。
- Large artifacts：是否误提交了 `data/`、`outputs/`、`model/`、cache、checkpoint 或大量 log。

## 最重要的原则

不要为了提高单个指标牺牲研究可信度。

特别注意：

- NegFlipAcc 提升但 ScopeCtrl 崩，不能算成功；
- NegRank 提升但 SELECT 标签有歧义，必须报告 Multi-answer NegRank；
- BoolQ raw 恢复但 E4/WikiFact 崩，属于 training tradeoff；
- router 提升但依赖 metadata，必须明确写成 structure-assisted；
- free-generation 提升但使用 keyword-aware rerank，只能作为 upper-bound diagnostic；
- MMLU 小幅上升不能写成通用能力提升。

## 当前仓库备注

- Remote: `git@github.com:Alive-mk/neg.git`
- 主分支：`main`
- 本地大产物默认不要进入 Git，除非明确选择为 release 内容。
- `data/`、`outputs/` 和 `model/` 是本地大目录，通常应保持 ignored。
- 现有进展记录在 `进展/`。
- 论文材料在 `paper/`。
