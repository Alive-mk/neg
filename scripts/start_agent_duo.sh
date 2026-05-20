#!/usr/bin/env bash
set -euo pipefail

REPO="${NEG_AGENT_REPO:-/data/mingkai/neg}"
STATE="${NEG_AGENT_STATE:-/tmp/neg-agent-duo}"
TASK_FILE="${NEG_AGENT_TASK_FILE:-$STATE/task.md}"
WORKER_SESSION="${NEG_WORKER_SESSION:-neg-worker}"
JUDGE_SESSION="${NEG_JUDGE_SESSION:-neg-judge}"

mkdir -p "$STATE"

if [[ $# -gt 0 ]]; then
  printf '%s\n' "$*" > "$TASK_FILE"
elif [[ ! -f "$TASK_FILE" ]]; then
  cat > "$TASK_FILE" <<'TASK'
按照 AGENTS.md 的当前优先级，自动选择一个最小、可验证、可提交的科研推进任务。
优先从 P0 开始；一次只做一个 bounded step；不要为了扩大范围而重构无关文件。
TASK
fi

if tmux has-session -t "$WORKER_SESSION" 2>/dev/null; then
  printf 'worker session already exists: %s\n' "$WORKER_SESSION"
else
  printf -v worker_cmd 'cd %q && NEG_AGENT_ITERATIONS=forever bash scripts/agent_duo_worker_loop.sh' "$REPO"
  tmux new-session -d -s "$WORKER_SESSION" "$worker_cmd"
  printf 'started worker session: %s\n' "$WORKER_SESSION"
fi

if tmux has-session -t "$JUDGE_SESSION" 2>/dev/null; then
  printf 'judge session already exists: %s\n' "$JUDGE_SESSION"
else
  printf -v judge_cmd 'cd %q && NEG_AGENT_ITERATIONS=forever bash scripts/agent_duo_judge_loop.sh' "$REPO"
  tmux new-session -d -s "$JUDGE_SESSION" "$judge_cmd"
  printf 'started judge session: %s\n' "$JUDGE_SESSION"
fi

cat <<INFO

state dir:
  $STATE

task file:
  $TASK_FILE

monitor:
  tmux capture-pane -pt $WORKER_SESSION
  tmux capture-pane -pt $JUDGE_SESSION

attach:
  tmux attach -t $WORKER_SESSION
  tmux attach -t $JUDGE_SESSION

stop:
  tmux kill-session -t $WORKER_SESSION
  tmux kill-session -t $JUDGE_SESSION
INFO
