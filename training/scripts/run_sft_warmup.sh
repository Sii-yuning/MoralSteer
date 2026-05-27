#!/usr/bin/env bash
# SFT warmup before OPD. Single-node, defaults sized for Qwen3.5-0.8B smoke run.
set -euo pipefail

PROJECT_ROOT="."
PYBIN="python"
MODEL="${1:-/path/to/base_models/Qwen3.5-0.8B}"
OUTDIR="${2:-$PROJECT_ROOT/training/artifacts/sft_warmup}"

mkdir -p "$OUTDIR"
cd "$PROJECT_ROOT"

export TRL_EXPERIMENTAL_SILENCE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

"$PYBIN" -m training.train_sft_warmup \
    --model "$MODEL" \
    --output-dir "$OUTDIR" \
    --max-seq-length 2048 \
    --per-device-bsz 2 \
    --grad-accum 8 \
    --lr 2e-5 \
    --epochs 1 \
    --bf16
