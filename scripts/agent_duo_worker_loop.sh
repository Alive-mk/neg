#!/usr/bin/env bash
set -euo pipefail

REPO="${NEG_AGENT_REPO:-/data/mingkai/neg}"
STATE="${NEG_AGENT_STATE:-/tmp/neg-agent-duo}"
TASK_FILE="${NEG_AGENT_TASK_FILE:-$STATE/task.md}"
ITERATIONS="${NEG_AGENT_ITERATIONS:-1}"
MAX_REVISIONS="${NEG_AGENT_MAX_REVISIONS:-2}"
SLEEP_SECONDS="${NEG_AGENT_SLEEP_SECONDS:-30}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_TIMEOUT_SECONDS="${NEG_AGENT_CODEX_TIMEOUT_SECONDS:-900}"
CODEX_SANDBOX="${NEG_CODEX_SANDBOX:-workspace-write}"
CODEX_APPROVAL="${NEG_CODEX_APPROVAL:-}"
ARTIFACTS_DIR="${NEG_AGENT_ARTIFACTS_DIR:-$REPO/outputs/agent_duo}"

WORKTREES="$STATE/worker-worktrees"
REVIEWS="$STATE/reviews"
LOGS="$STATE/logs"
PENDING_FILE="$STATE/pending_review.env"
DONE_FILE="$STATE/done.md"

mkdir -p "$STATE" "$WORKTREES" "$REVIEWS" "$LOGS"

if [[ ! -f "$TASK_FILE" ]]; then
  cat > "$TASK_FILE" <<'TASK'
按照 AGENTS.md 的当前优先级，自动选择一个完整、可验证、可提交的科研实验包。
优先从 P0 开始；每轮必须尽量完成“实验目标 -> 真实运行 -> 指标汇总 -> 风险判断 -> 下一步”的闭环。
如果数据和模型存在，不要停留在 --help、py_compile 或文档整理；必要时使用 GPU 跑真实评测或训练。
TASK
fi

log() {
  printf '[worker %s] %s\n' "$(date '+%F %T')" "$*"
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

write_pending_review() {
  local branch="$1"
  local commit="$2"
  local safe="$3"
  local artifact_dir="$4"
  local review_file="$REVIEWS/${safe}_${commit:0:12}.md"

  {
    printf 'state=pending\n'
    printf 'branch=%s\n' "$branch"
    printf 'commit=%s\n' "$commit"
    printf 'safe=%s\n' "$safe"
    printf 'artifact_dir=%s\n' "$artifact_dir"
    printf 'review_file=%s\n' "$review_file"
  } > "$PENDING_FILE.tmp"
  mv "$PENDING_FILE.tmp" "$PENDING_FILE"
  printf '%s\n' "$review_file"
}

wait_for_review() {
  local review_file="$1"
  log "waiting for judge review: $review_file"
  while [[ ! -s "$review_file" ]]; do
    sleep "$SLEEP_SECONDS"
  done
}

mark_idle() {
  printf 'state=idle\n' > "$PENDING_FILE"
}

stage_and_commit_if_needed() {
  local worktree="$1"
  local message="$2"

  git -C "$worktree" add -A -- \
    . \
    ':!data/**' \
    ':!outputs/**' \
    ':!model/**' \
    ':!**/__pycache__/**' \
    ':!*.log'

  if git -C "$worktree" diff --cached --quiet; then
    return 1
  fi

  git -C "$worktree" commit -m "$message"
}

iteration=0
while [[ "$ITERATIONS" == "forever" || "$iteration" -lt "$ITERATIONS" ]]; do
  iteration=$((iteration + 1))

  timestamp="$(date '+%Y%m%d-%H%M%S')"
  branch="worker/auto-$timestamp"
  safe="${branch//\//__}"
  worktree="$WORKTREES/$safe"
  worker_log="$LOGS/$safe-worker.log"
  artifact_dir="$ARTIFACTS_DIR/$safe"

  log "starting iteration $iteration on $branch"

  git -C "$REPO" fetch origin
  git -C "$REPO" worktree add -b "$branch" "$worktree" origin/main
  mkdir -p "$artifact_dir"

  task_text="$(sed -n '1,240p' "$TASK_FILE" 2>/dev/null || true)"
  done_text="$(tail -n 260 "$DONE_FILE" 2>/dev/null || true)"

  worker_prompt="$(cat <<PROMPT
你是 worker。先读取 AGENTS.md，并严格按 Worker 角色工作。

当前任务池：
$task_text

已完成/已阻塞记录：
$done_text

持久实验产物目录：
$artifact_dir

请自动选择一个完整、可验证、可提交的科研实验包。要求：
1. 优先服务 P0；每轮围绕一个明确 RQ 完成“实验目标 -> 真实运行 -> 指标汇总 -> 风险判断 -> 下一步”闭环。
2. 如果数据和模型存在，不要停留在 --help、py_compile 或文档整理；必须运行真实评测。必要时使用 GPU。若确实需要训练，可以启动有边界的训练并记录配置。
3. GPU/PyTorch 命令可以直接运行；本轮 supervisor 已按需要提供非 sandbox 权限。优先使用已存在脚本，例如 eval_clean_strict.py、eval_with_router.py、check_leakage.py、eval_boolq_pmi.py、evaluate_mmlu.py。
4. 实验输出、cache、模型产物不要提交到 git。需要保留的 JSON/CSV/日志写入或复制到上面的持久实验产物目录，并在进展日志中记录路径。
5. 只提交和本任务相关的代码、配置、轻量测试、进展日志；不要使用 git add .。
6. 如果完整实验因数据、模型、GPU、依赖或时间阻塞，必须先尝试可行替代实验，并把 blocked 原因写清楚。
7. 不要自己 commit 或 push；外层 supervisor 会统一提交和推送。

结束时必须输出：
分支名：
修改文件：
新增文件：
运行命令：
核心结果：
是否改变评测口径：
是否存在风险：
建议是否进入论文：
PROMPT
)"

  if ! codex_exec "$worktree" "$worker_prompt" "$worker_log"; then
    log "codex worker failed; see $worker_log"
    {
      printf '\n## %s\n\n' "$branch"
      printf '状态：worker 执行失败\n\n'
      printf '日志：%s\n' "$worker_log"
    } >> "$DONE_FILE"
    continue
  fi

  stage_and_commit_if_needed "$worktree" "Auto worker update $timestamp" || true

  commits="$(git -C "$worktree" rev-list --count origin/main..HEAD)"
  if [[ "$commits" == "0" ]]; then
    log "no commit produced on $branch"
    {
      printf '\n## %s\n\n' "$branch"
      printf '状态：未产生 commit\n\n'
      printf '日志：%s\n' "$worker_log"
    } >> "$DONE_FILE"
    continue
  fi

  git -C "$worktree" push -u origin HEAD
  commit="$(git -C "$worktree" rev-parse HEAD)"
  review_file="$(write_pending_review "$branch" "$commit" "$safe" "$artifact_dir")"

  revision=0
  while [[ "$revision" -le "$MAX_REVISIONS" ]]; do
    wait_for_review "$review_file"

    if grep -q '^结论：通过' "$review_file"; then
      log "judge passed $branch at $commit"
      {
        printf '\n## %s\n\n' "$branch"
        printf '状态：通过\n\n'
        printf 'commit：%s\n' "$commit"
        printf 'review：%s\n' "$review_file"
      } >> "$DONE_FILE"
      mark_idle
      break
    fi

    if [[ "$revision" -ge "$MAX_REVISIONS" ]]; then
      log "max revisions reached for $branch"
      {
        printf '\n## %s\n\n' "$branch"
        printf '状态：达到最大修订次数，等待人工查看\n\n'
        printf 'commit：%s\n' "$commit"
        printf 'review：%s\n' "$review_file"
      } >> "$DONE_FILE"
      mark_idle
      break
    fi

    revision=$((revision + 1))
    fix_log="$LOGS/$safe-fix-$revision.log"
    review_text="$(sed -n '1,260p' "$review_file")"

    fix_prompt="$(cat <<PROMPT
你是 worker。judge 已经评审当前分支 $branch，并要求修改。请读取 AGENTS.md 和下面的 review，只做必要修复。

Judge review：
$review_text

要求：
1. 保持在当前分支 $branch。
2. 只修复 judge 指出的具体问题。
3. 不要重写无关文件，不要使用 git add .。
4. 运行最小相关验证。
5. 不要自己 commit 或 push；外层 supervisor 会统一提交和推送。
6. 最后按 Worker 产出模板说明结果。
PROMPT
)"

    old_commit="$commit"
    if ! codex_exec "$worktree" "$fix_prompt" "$fix_log"; then
      log "codex fix failed; see $fix_log"
      break
    fi

    stage_and_commit_if_needed "$worktree" "Auto worker revision $timestamp-$revision" || true

    commit="$(git -C "$worktree" rev-parse HEAD)"
    if [[ "$commit" == "$old_commit" ]]; then
      log "fix round produced no new commit"
      break
    fi

    git -C "$worktree" push -u origin HEAD
    review_file="$(write_pending_review "$branch" "$commit" "$safe" "$artifact_dir")"
  done

  sleep "$SLEEP_SECONDS"
done
