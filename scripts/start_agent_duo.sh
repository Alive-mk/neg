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
按照 AGENTS.md 的当前优先级，自动选择一个完整、可验证、可提交的科研实验包。
优先从 P0 开始；每轮必须尽量完成“实验目标 -> 真实运行 -> 指标汇总 -> 风险判断 -> 下一步”的闭环。
如果数据和模型存在，不要停留在 --help、py_compile 或文档整理；必要时使用 GPU 跑真实评测或训练。
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
