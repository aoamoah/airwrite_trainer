#!/usr/bin/env bash
# Re-runnable ablation chain for the `dataset` corpus.
#
# Each ablation records a marker in reports/.ablation_state/ only once its run
# directory holds both results.json and report.md. Re-running this script skips
# whatever already finished, so an interruption (Windows Update tearing down
# the WSL VM, as on 2026-09-20) costs one run rather than the whole queue.
set -u

cd /home/kwame/airwrite_trainer || exit 1
PY=venv/bin/python
STATE=reports/.ablation_state
mkdir -p "$STATE"

# name|flags passed to train.py alongside --datasets dataset
ABLATIONS=(
  "nomotion|--no-motion"
  "motiononly|--motion-only"
  "world|--pose-source world"
  "velocity|--velocity"
)

complete() {           # $1 = marker file; 0 when its run dir is intact
  local marker=$1 dir
  [ -f "$marker" ] || return 1
  dir=$(cat "$marker")
  [ -n "$dir" ] && [ -f "$dir/results.json" ] && [ -f "$dir/report.md" ]
}

for entry in "${ABLATIONS[@]}"; do
  name=${entry%%|*}
  flags=${entry#*|}
  marker="$STATE/$name.done"
  log="logs_abl_$name.txt"

  if complete "$marker"; then
    echo "[skip] $name -> $(cat "$marker")"
    continue
  fi

  echo "[run ] $name ($flags) started $(date '+%F %T')"
  # shellcheck disable=SC2086
  $PY train.py --datasets dataset $flags > "$log" 2>&1
  rc=$?

  dir=$(sed -n 's|^Models + results\.json in: ||p' "$log" | tail -1)
  if [ $rc -eq 0 ] && [ -n "$dir" ] \
     && [ -f "$dir/results.json" ] && [ -f "$dir/report.md" ]; then
    echo "$dir" > "$marker"
    echo "[done] $name -> $dir  $(date '+%F %T')"
  else
    echo "[FAIL] $name rc=$rc dir='${dir:-none}' — not marked; rerun to retry"
  fi
done

echo "[chain] finished $(date '+%F %T')"
