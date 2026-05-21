#!/usr/bin/env bash
set -euo pipefail

REPO="${NEG_AGENT_REPO:-/data/mingkai/neg}"
STATE="${NEG_AGENT_STATE:-/tmp/neg-agent-duo}"
ITERATIONS="${NEG_AGENT_ITERATIONS:-forever}"
SLEEP_SECONDS="${NEG_AGENT_SLEEP_SECONDS:-30}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_TIMEOUT_SECONDS="${NEG_AGENT_CODEX_TIMEOUT_SECONDS:-7200}"
CODEX_SANDBOX="${NEG_CODEX_SANDBOX:-danger-full-access}"
CODEX_APPROVAL="${NEG_CODEX_APPROVAL:-never}"

JUDGE_WORKTREES="$STATE/judge-worktrees"
LOGS="$STATE/logs"
PENDING_FILE="$STATE/pending_review.env"

mkdir -p "$STATE" "$JUDGE_WORKTREES" "$LOGS"

log() {
  printf '[judge %s] %s\n' "$(date '+%F %T')" "$*"
}

field_from_pending() {
  local field="$1"
  sed -n "s/^${field}=//p" "$PENDING_FILE" 2>/dev/null | tail -n 1
}

codex_exec() {
  local worktree="$1"
  local prompt="$2"
  local log_file="$3"
  local top_args=()
  local extra_args=()

  if [[ -n "$CODEX_APPROVAL" ]]; then
    top_args+=(--ask-for-approval "$CODEX_APPROVAL")
  fi

  if [[ -n "${NEG_CODEX_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${NEG_CODEX_EXTRA_ARGS})
  fi

  timeout "$CODEX_TIMEOUT_SECONDS" "$CODEX_BIN" "${top_args[@]}" exec \
    -C "$worktree" \
    --sandbox "$CODEX_SANDBOX" \
    "${extra_args[@]}" \
    "$prompt" 2>&1 | tee "$log_file"
}

iteration=0
while [[ "$ITERATIONS" == "forever" || "$iteration" -lt "$ITERATIONS" ]]; do
  iteration=$((iteration + 1))

  state="$(field_from_pending state || true)"
  branch="$(field_from_pending branch || true)"
  commit="$(field_from_pending commit || true)"
  safe="$(field_from_pending safe || true)"
  artifact_dir="$(field_from_pending artifact_dir || true)"
  review_file="$(field_from_pending review_file || true)"

  if [[ "$state" != "pending" || -z "$branch" || -z "$commit" || -z "$safe" || -z "$review_file" ]]; then
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if [[ -s "$review_file" ]]; then
    sleep "$SLEEP_SECONDS"
    continue
  fi

  log "reviewing $branch at $commit"

  git -C "$REPO" fetch origin
  judge_worktree="$JUDGE_WORKTREES/${safe}_${commit:0:12}"
  git -C "$REPO" worktree add --detach "$judge_worktree" "$commit"

  judge_log="$LOGS/${safe}_${commit:0:12}-judge.log"
  tmp_review="$review_file.tmp"

  judge_prompt="$(
    cat <<'PROMPT'
你是 judge。先读取 AGENTS.md，并严格按 Judge 角色工作。

请评审 worker 分支：
PROMPT
    printf -- '- branch: %s\n' "$branch"
    printf -- '- commit: %s\n' "$commit"
    printf -- '- artifact_dir: %s\n' "${artifact_dir:-未提供}"
    printf -- '- canonical_repo: %s\n' "$REPO"
    cat <<'PROMPT'

要求：
1. 默认只评审，不直接修改代码。
2. 重点检查这个改动回答了哪个研究问题、是否可信、是否存在数据泄漏、metric drift、缺 baseline、不可复现路径、大文件误入库。
3. 对“完整实验包”按真实实验标准验收：检查是否实际运行了数据/模型/GPU评测或训练，是否有 JSON/CSV/日志产物，是否记录了 split、token 口径、模型名、命令和指标。
4. 如 artifact_dir 存在，检查其中的关键产物；不要只看 git diff。真实数据和历史 outputs/model 可能只存在于 canonical_repo，不在临时 judge worktree 中；需要复查时使用绝对路径。
PROMPT
    printf -- '   - `%s/data`\n' "$REPO"
    printf -- '   - `%s/outputs`\n' "$REPO"
    printf -- '   - `%s/model`\n' "$REPO"
    cat <<'PROMPT'
5. 运行必要验证；如果验证因依赖、GPU 或数据缺失无法运行，要写清楚 blocked 原因。
6. 使用 origin/main...HEAD 查看 diff。
7. 最终必须输出下面模板，第一行必须是结论。

模板：
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
PROMPT
  )"

  if codex_exec "$judge_worktree" "$judge_prompt" "$judge_log" > "$tmp_review"; then
    mv "$tmp_review" "$review_file"
  else
    {
      printf '结论：阻塞\n\n'
      printf '主要问题：\n'
      printf -- '- judge codex exec failed；请查看日志：%s\n\n' "$judge_log"
      printf '验证：\n'
      printf -- '- 已运行：codex exec judge review\n'
      printf -- '- 结果：blocked\n\n'
      printf '下一步：\n'
      printf -- '- worker 暂停该分支，等待日志排查。\n'
    } > "$review_file"
    rm -f "$tmp_review"
  fi

  sleep "$SLEEP_SECONDS"
done
