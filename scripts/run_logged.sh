#!/bin/bash
# Unique-path command logger for the M0 candidate-2 evidence ledger.
# Usage: bash scripts/run_logged.sh <command_id> <relative_log_path> -- <argv...>
# Refuses to reuse a log path (exit 99); appends one JSONL record per command
# with the real exit code; the log itself ends with an EXIT_CODE=<n> line.
set -u
WT=/root/workspace/Code/DualTimesField-worktrees/agentic-asof-abs-predispatch-c2-20261029-r4
EVIDENCE_ROOT="$WT/logs/forecasting/m0_predispatch_td_abs_c2_20261029"
ID=$1; LOGREL=$2; shift 2
if [ "${1:-}" != "--" ]; then echo "usage: run_logged.sh <id> <log_path> -- <argv...>" >&2; exit 98; fi
shift
LOG="$EVIDENCE_ROOT/$LOGREL"
LEDGER="$EVIDENCE_ROOT/commands.jsonl"
if [ -e "$LOG" ]; then echo "run_logged: log path already exists: $LOG" >&2; exit 99; fi
mkdir -p "$(dirname "$LOG")" "$(dirname "$LEDGER")"
cd "$WT" || exit 97
START_UTC=$(date -u +%FT%T.%6NZ)
TMP="$LOG.raw.$$"
"$@" >"$TMP" 2>&1
CODE=$?
END_UTC=$(date -u +%FT%T.%6NZ)
{
  echo "=== run_logged id=$ID ==="
  echo "argv: $*"
  echo "git_head: $(git rev-parse HEAD)"
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "start_utc: $START_UTC"
  echo "end_utc: $END_UTC"
  echo "--- output begin ---"
  cat "$TMP"
  echo "--- output end ---"
  echo "EXIT_CODE=$CODE"
} >"$LOG"
rm -f "$TMP"
printf '%s\n' "{\"id\": \"$ID\", \"argv\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"), \"log_path\": \"$LOGREL\", \"git_head\": \"$(git rev-parse HEAD)\", \"start_utc\": \"$START_UTC\", \"end_utc\": \"$END_UTC\", \"exit_code\": $CODE}" >>"$LEDGER"
exit $CODE
