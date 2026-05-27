#!/usr/bin/env bash
# Run OPD stages 1 -> 2 -> 3 sequentially.
# Usage: run_opd_curriculum.sh  [stage1|stage2|stage3|all]
set -euo pipefail

PROJECT_ROOT="."
PYBIN="python"
TARGET="${1:-all}"

cd "$PROJECT_ROOT"
export TRL_EXPERIMENTAL_SILENCE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

run_stage() {
    local stage="$1"
    echo "=== Running OPD $stage ==="
    "$PYBIN" -m training.train_opd \
        --config "$PROJECT_ROOT/training/configs/opd_$stage.yaml"
}

case "$TARGET" in
    stage1) run_stage stage1 ;;
    stage2) run_stage stage2 ;;
    stage3) run_stage stage3 ;;
    all)    run_stage stage1; run_stage stage2; run_stage stage3 ;;
    *) echo "Unknown stage: $TARGET" >&2; exit 1 ;;
esac
