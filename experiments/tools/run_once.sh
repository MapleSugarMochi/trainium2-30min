#!/usr/bin/env bash
# Runs at most ONE queued experiment, then exits. Called in a loop by agent.sh.
# The latest copy is pulled from origin/ops before every call, so the runner
# can be updated remotely by pushing to the ops branch.
#
# Queue: origin/ops:runner/queue.tsv, tab-separated, header line first:
#   id  ref  nc  max_train_seconds  args  env  desc
#   ref  = branch name on origin (e.g. exp/warmdown) or a commit sha
#   nc   = 1 (8 procs) or 2 (4 procs)
#   args = extra train.py args, space-separated, or "-"
#   env  = extra env vars "K=V K2=V2", or "-" (values must not contain spaces)
# Results: pushed to origin/results (runs/<id>/..., all_runs.tsv).
# The status column is only crash / keep_pending here; keep vs discard was decided
# by hand afterwards and is applied by tools/make_results.py (its KEEP set).
set -uo pipefail

WS=/home/ubuntu/workspace
REPO=$WS/trn-r1-test
EXP=$WS/exp
RES=$WS/results
CKPT=$WS/ckpt
VENV=$WS/native_venv/bin/activate
# Latest time every job must finish by (the instance was reclaimed 2026-11-11 11:30 UTC).
# Override with RUNNER_DEADLINE_UTC=YYYY-MM-DDTHH:MM:SSZ.
DEADLINE_UTC=${RUNNER_DEADLINE_UTC:-2026-11-11T11:00:00Z}
# Wall time beyond max_train_seconds: startup ~90 s + eval ~150 s + pushes (measured 1887-1891 s wall for 1800 s jobs, excl. eval).
OVERHEAD_S=450

log() { echo "[$(date -u +%FT%TZ)] $*"; }
DEADLINE_EPOCH=$(date -u -d "$DEADLINE_UTC" +%s) || { log "bad RUNNER_DEADLINE_UTC: $DEADLINE_UTC"; exit 1; }

cd "$REPO" || { log "repo missing"; exit 1; }
git config user.name >/dev/null || git config user.name "runner"
git config user.email >/dev/null || git config user.email "runner@example.com"
git fetch -q origin || { log "fetch failed"; exit 0; }

# --- one-time worktree setup ---------------------------------------------------
if [ ! -d "$EXP/.git" ] && [ ! -f "$EXP/.git" ]; then
  git worktree add --detach "$EXP" origin/main || exit 1
fi
if [ ! -f "$RES/.git" ] && [ ! -d "$RES/.git" ]; then
  git worktree add -B results "$RES" origin/results || exit 1
fi
mkdir -p "$CKPT"
git -C "$RES" config user.name >/dev/null || git -C "$RES" config user.name "$(git config user.name)"

push_results() {
  local msg="$1"
  git -C "$RES" add -A
  git -C "$RES" commit -q -m "$msg" || true
  for i in 1 2 3 4 5; do
    git -C "$RES" pull -q --rebase origin results && git -C "$RES" push -q origin HEAD:results && return 0
    sleep 5
  done
  log "push failed"
}

# --- heartbeat (so the remote side can see the runner is alive) ---------------
heartbeat() {
  {
    echo "time_utc: $(date -u +%FT%TZ)"
    echo "state: $1"
    echo "disk: $(df -h $WS | tail -1)"
    echo "$BACKUP_LINE"
    echo "busy_procs: $(pgrep -fa 'train.py|prepare.py eval-public|run_baseline.sh' | grep -v pgrep | head -5 | tr '\n' ';')"
  } > "$RES/HEARTBEAT"
}

hb_push() {
  local f=$HOME/.hb_last now; now=$(date +%s)
  if [ ! -f "$f" ] || [ $((now - $(cat "$f"))) -ge 600 ]; then
    echo "$now" > "$f"; push_results "heartbeat" >/dev/null 2>&1
  fi
}

# --- mirror every origin branch to the backup repo (if configured), every 30 min
backup_sync() {
  git -C "$REPO" remote get-url backup >/dev/null 2>&1 || { echo "backup: not configured"; return; }
  local f=$HOME/.backup_last now; now=$(date +%s)
  if [ -f "$f" ] && [ $((now - $(cat "$f"))) -lt 1800 ]; then cat "$HOME/.backup_status" 2>/dev/null; return; fi
  local refs=""
  for r in $(git -C "$REPO" for-each-ref --format='%(refname:strip=3)' refs/remotes/origin | grep -v '^HEAD$'); do
    refs="$refs +refs/remotes/origin/$r:refs/heads/$r"
  done
  if timeout 300 git -C "$REPO" push -q backup $refs 2>"$HOME/.backup_err"; then
    echo "$now" > "$f"; echo "backup: ok $(date -u +%FT%TZ) ($(echo $refs | wc -w) branches)" > "$HOME/.backup_status"
  else
    echo "backup: FAILED $(date -u +%FT%TZ): $(tail -1 $HOME/.backup_err)" > "$HOME/.backup_status"
  fi
  cat "$HOME/.backup_status"
}
BACKUP_LINE=$(backup_sync)

# --- import finished manual runs from run_baseline.sh --------------------------
for ev in "$REPO"/logs/*.eval; do
  [ -f "$ev" ] || continue
  name=$(basename "$ev" .eval)
  [ -d "$RES/runs/manual_$name" ] && continue
  [ -f "$REPO/logs/$name.end" ] || continue
  grep -q '"val_bpb"' "$ev" || continue
  d="$RES/runs/manual_$name"; mkdir -p "$d"
  cp "$REPO/logs/$name".* "$d/" 2>/dev/null
  log "imported manual run $name"
  push_results "import manual run $name"
done

# --- busy? (a manual run is going) --------------------------------------------
if pgrep -f 'train.py|prepare.py eval-public|run_baseline.sh' >/dev/null; then
  heartbeat "waiting: manual run in progress"; hb_push; exit 0
fi

# --- stop switch ---------------------------------------------------------------
if git cat-file -e origin/ops:runner/STOP 2>/dev/null; then
  heartbeat "stopped (runner/STOP present)"; hb_push; exit 0
fi

# --- pick the next job ----------------------------------------------------------
QUEUE=$(git show origin/ops:runner/queue.tsv 2>/dev/null | tail -n +2)
JOB=""
while IFS=$'\t' read -r id ref nc maxs args envs desc; do
  [ -z "${id:-}" ] && continue
  case "$id" in \#*) continue;; esac
  [ -d "$RES/runs/$id" ] && continue
  # skip queued jobs that can no longer finish before the deadline
  [ $(( $(date -u +%s) + maxs + OVERHEAD_S )) -gt "$DEADLINE_EPOCH" ] && continue
  JOB="$id"; break
done <<< "$QUEUE"

# --- queue empty: idle (no seed fillers, by request) ---------------------------
if [ -z "$JOB" ]; then
  heartbeat "idle: queue empty"; hb_push; exit 0
fi

now=$(date -u +%s)
need=$OVERHEAD_S
if [ $((now + maxs + need)) -gt "$DEADLINE_EPOCH" ]; then
  heartbeat "refusing $id: would not finish before deadline"; hb_push; exit 0
fi

# --- run it ----------------------------------------------------------------------
if git rev-parse -q --verify "origin/$ref^{commit}" >/dev/null; then target="origin/$ref"; else target="$ref"; fi
git -C "$EXP" checkout -q --detach --force "$target" || {
  d="$RES/runs/$id"; mkdir -p "$d"; echo "bad ref $ref" > "$d/error.txt"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "-" "$ref" "$nc" "$maxs" "$args" "$envs" "-" "-" "-" "-" "-" "-" "crash" "$desc" >> "$RES/all_runs.tsv"
  push_results "run $id: bad ref"; exit 0; }
commit=$(git -C "$EXP" rev-parse --short HEAD)
[ "$nc" = "2" ] && np=4 || np=8
[ "$args" = "-" ] && args=""
[ "$envs" = "-" ] && envs=""
# Split into arrays without glob expansion, and reject env entries that are not K=V
# (a value with a space would otherwise silently become a separate argument).
read -ra args_arr <<< "$args"
read -ra env_arr <<< "$envs"
for kv in "${env_arr[@]}"; do
  if [[ ! "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
    d="$RES/runs/$id"; mkdir -p "$d"; echo "bad env entry: $kv" > "$d/error.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$commit" "$ref" "$nc" "$maxs" "${args:--}" "${envs:--}" "-" "-" "-" "-" "-" "-" "crash" "$desc" >> "$RES/all_runs.tsv"
    push_results "run $id: bad env"; exit 0
  fi
done

d="$RES/runs/$id"; mkdir -p "$d"
out="$CKPT/$id"; mkdir -p "$out"
cmd="NEURON_LOGICAL_NC_CONFIG=$nc torchrun --standalone --nproc_per_node=$np train.py"
{
  echo "id: $id"; echo "ref: $ref"; echo "commit: $commit"; echo "launch: $cmd"
  echo "args: $args"; echo "env: $envs"; echo "max_train_seconds: $maxs"; echo "desc: $desc"
  echo "start_utc: $(date -u +%FT%TZ)"
} > "$d/meta.txt"
heartbeat "running $id ($commit)"; push_results "start $id ($commit)"

source "$VENV"
cd "$EXP"
t0=$(date +%s)
# Hardware telemetry sidecar: 10 min into the run (steady state), sample
# neuron-monitor for 60 s (power / utilization if the tool reports them).
( sleep 600; command -v neuron-monitor >/dev/null && timeout 60 neuron-monitor > "$d/neuron_monitor.jsonl" 2>"$d/neuron_monitor.err"; \
  command -v neuron-ls >/dev/null && neuron-ls > "$d/neuron_ls.txt" 2>&1 ) &
env "${env_arr[@]}" NEURON_LOGICAL_NC_CONFIG="$nc" timeout $((maxs + 2400)) \
  torchrun --standalone --nproc_per_node="$np" train.py \
  --max-train-seconds "$maxs" --out-dir "$out" --no-eval-public "${args_arr[@]}" \
  > "$d/train.log" 2>&1
rc=$?
t1=$(date +%s)

status=keep_pending
bpb="-"; steps="-"; sps="-"; startup="-"; trains="-"; evals="-"
if [ $rc -ne 0 ] || [ ! -f "$out/final.pt" ]; then
  status=crash
  tail -50 "$d/train.log" > "$d/error_tail.txt"
else
  timeout 1500 python prepare.py eval-public --train-py train.py --checkpoint "$out/final.pt" > "$d/eval.log" 2>&1
  bpb=$(python - "$d/eval.log" <<'PY'
import json,sys,re
t=open(sys.argv[1]).read()
m=re.search(r"\{.*\}", t, re.S)
try: print("%.6f"%json.loads(m.group(0))["val_bpb"])
except Exception: print("-")
PY
)
  evals=$(grep -o '"eval_seconds": [0-9.]*' "$d/eval.log" | awk '{printf "%.0f",$2}')
  [ "$bpb" = "-" ] && status=crash
fi
steps=$(grep -E '^num_steps:' "$d/train.log" | awk '{print $2}')
trains=$(grep -E '^training_seconds:' "$d/train.log" | awk '{print $2}')
startup=$(grep -E '^startup_allowance_est:' "$d/train.log" | awk '{print $2}')
if [ -n "${steps:-}" ] && [ -n "${trains:-}" ] && [ "$steps" -gt 5 ]; then
  sps=$(python -c "print('%.3f'%($trains/($steps-5)))")
fi
echo "end_utc: $(date -u +%FT%TZ)" >> "$d/meta.txt"
echo "wall_seconds: $((t1 - t0))" >> "$d/meta.txt"
echo "exit_code: $rc" >> "$d/meta.txt"

[ -f "$RES/all_runs.tsv" ] || printf 'id\tcommit\tref\tnc\tmax_s\targs\tenv\tval_bpb\tsteps\ts_per_step\tstartup_s\ttrain_s\teval_s\tstatus\tdesc\n' > "$RES/all_runs.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$id" "$commit" "$ref" "$nc" "$maxs" "${args:--}" "${envs:--}" "$bpb" "${steps:--}" "$sps" "${startup:--}" "${trains:--}" "${evals:--}" "$status" "$desc" \
  >> "$RES/all_runs.tsv"
heartbeat "finished $id: bpb=$bpb status=$status"
push_results "result $id ($commit): val_bpb=$bpb steps=${steps:--} status=$status"
log "done $id bpb=$bpb"
