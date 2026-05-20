# Negation Blindness

这个仓库用于研究和复现实验中的 negation blindness：模型在看到否定提示时，仍然偏向原正例、过度否定或无法稳定控制否定作用域的问题。

当前代码覆盖四类工作：

1. 构造 E4 / E4v3 否定样本。
2. 校验、清洗并切分数据。
3. 训练 MGNM、SFT、DPO、contrastive、rank-distillation 等模型变体。
4. 在 largetest、WikiFact、BoolQ、MMLU、free-generation 和 router 场景下评测。

## 仓库结构

- `configs/`: 数据计划和模型配置。
- `resources/`: 生成数据使用的种子主题。
- `src/neg_blindness/`: schema、IO、prompt、split、API、metric 和 evaluation 核心逻辑。
- `scripts/`: 数据构造、训练、评测、路由、统计显著性和导出脚本。
- `paper/`: 论文草稿、LaTeX 源文件和参考文献。
- `进展/`: 实验日志、阶段总结和待解决问题。
- `data/`: 本地数据，默认不进 git。
- `outputs/`: 本地实验输出，默认不进 git。
- `model/`: 本地模型或 adapter，默认不进 git。

## 环境准备

建议使用项目内虚拟环境：

```bash
cd /data/mingkai/neg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要调用 OpenAI-compatible API，先设置环境变量：

```bash
export NEG_OPENAI_API_KEY='你的key'
```

本地 Hugging Face 评测需要在 `configs/model_config.json` 或对应配置文件中填写真实 `model_path`。

## 基础流水线

生成候选数据：

```bash
python scripts/generate_dataset.py \
  --plan configs/dataset_plan.json \
  --models configs/model_config.json \
  --output data/raw/generated_candidates.jsonl
```

规则校验：

```bash
python scripts/validate_dataset.py \
  --input data/raw/generated_candidates.jsonl \
  --output data/processed/validated.jsonl \
  --report outputs/validation_report.json
```

切分数据：

```bash
python scripts/split_dataset.py \
  --input data/processed/validated.jsonl \
  --output-dir data/processed/splits \
  --report outputs/split_report.json
```

评测模型：

```bash
python scripts/evaluate_models.py \
  --models configs/model_config.json \
  --input data/processed/splits/test.jsonl \
  --output outputs/eval_test.json \
  --cache-dir outputs/score_cache
```

导出 E4v3 release 格式：

```bash
python scripts/export_e4v3_release.py \
  --split-input train=data/processed/splits/train.jsonl \
  --split-input dev=data/processed/splits/dev.jsonl \
  --split-input test=data/processed/splits/test.jsonl \
  --output-dir outputs/e4v3_release
```

## 常用实验入口

训练 MGNM：

```bash
python scripts/train_mgnm.py --help
```

BoolQ 否定评测：

```bash
python scripts/eval_boolq_negation.py --help
```

Router 聚合评测：

```bash
python scripts/eval_with_router.py --help
```

Free-generation preserve 评测：

```bash
python scripts/eval_freegen_preserve.py --help
```

统计显著性：

```bash
python scripts/stats_significance.py --help
```

## 数据记录核心字段

E4 记录的核心字段包括：

- `semantic_mode`: `suppression_only`、`contrastive_resolution`、`exclusive_choice`
- `scope_type`: `in_scope`、`out_of_scope`、`double_negation`
- `expected_neg_behavior`: `suppress_target`、`select_gold_neg`、`preserve_positive`
- `gold_pos`
- `gold_neg`
- `forbidden_neg`
- `candidate_pool_neg`
- `distractors`

其中 `preserve_positive` 用于显式表示 scope control 和 double negation，否则评测无法区分“应该翻转”和“应该保留正例”。

## Git 协作建议

这个仓库适合用一个 worker 分支和一个 judge worktree 做 Codex 双角色协作。

worker 创建任务分支并提交：

```bash
cd /data/mingkai/neg
git switch -c codex/worker-task-$(date +%Y%m%d-%H%M)

# 修改代码、跑测试后：
git status --short
git diff
git add README.md
git commit -m "Update project README"
git push -u origin HEAD
```

judge 在独立 worktree 评审，不直接改 worker 工作区：

```bash
cd /data/mingkai/neg
git fetch origin

BRANCH="codex/worker-task-替换成实际分支名"
git worktree add ../neg-judge "origin/$BRANCH"
cd ../neg-judge

git diff --stat origin/main...HEAD
git diff origin/main...HEAD
```

推荐规则：

- worker 负责实现、测试和提交。
- judge 负责审查 diff、运行验证命令、给出下一步。
- judge 默认不改代码；需要修复时，把具体问题写清楚交回 worker。

## 当前状态

- 远端仓库：`git@github.com:Alive-mk/neg.git`
- 当前主分支：`main`
- 大型本地产物通过 `.gitignore` 排除：`data/`、`outputs/`、`model/`
- 论文材料在 `paper/` 和 `进展/`

首次正式入库前建议先确认哪些小数据、配置和结果摘要需要 force-add，避免把大模型或完整输出误传到 GitHub。
