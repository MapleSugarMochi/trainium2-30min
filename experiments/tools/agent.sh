#!/usr/bin/env bash
# Start once on the instance inside tmux:  bash ~/agent.sh
# Every 60s: fetch origin, take the latest runner from the ops branch, run at most one job.
REPO=/home/ubuntu/workspace/trn-r1-test
while true; do
  cd "$REPO" && git fetch -q origin 2>/dev/null
  if git show origin/ops:runner/run_once.sh > "$HOME/run_once.sh.new" 2>/dev/null; then
    mv "$HOME/run_once.sh.new" "$HOME/run_once.sh"
  fi
  bash "$HOME/run_once.sh" 2>&1 | tee -a "$HOME/runner.log" | tail -3
  sleep 60
done
