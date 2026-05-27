#!/usr/bin/env bash
# MoralSteer two-model serial pipeline:
#   1. Train Qwen3.5-4B  (SFT warmup -> OPD stage 1 -> stage 2 -> stage 3)   ← small-first, validates pipeline + loss shape
#   2. Clean GPU processes
#   3. Train Qwen3.5-9B  (SFT warmup -> OPD stage 1 -> stage 2 -> stage 3)
#   4. Clean GPU processes
#
# For each model only the *final* checkpoint (opd_stage3) is kept; SFT warmup
# and intermediate stages are removed once they have served as input to the next
# stage. TensorBoard logs land under `<run>/<stage>/tb`.
#
# Usage:
#   bash training/scripts/train_two_models.sh
#       optional env:
#           ONLY_MODEL=4b|9b     # run a single model (default: run both, 4b first)
#           DRY_RUN=1            # echo commands without running
#           USE_ACCELERATE=1     # wrap python with `accelerate launch` (multi-GPU FSDP)
#           ACCEL_CONFIG=path    # override accelerate config (default: configs/accelerate_fsdp.yaml)
#           NUM_PROCESSES=N      # only used with USE_ACCELERATE; default 8, set 4 for 4-GPU box

set -euo pipefail

PROJECT_ROOT="."
PYBIN="python"
TIER1_JSONL="$PROJECT_ROOT/training/artifacts/data/pilot5k/tier1.jsonl"
RUN_ROOT="$PROJECT_ROOT/training/artifacts/runs"

QWEN_4B="/path/to/base_models/Qwen3.5-4B"
QWEN_9B="/path/to/base_models/Qwen3.5-9B"

ONLY_MODEL="${ONLY_MODEL:-all}"
DRY_RUN="${DRY_RUN:-0}"
USE_ACCELERATE="${USE_ACCELERATE:-0}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$PROJECT_ROOT/training/configs/accelerate_fsdp.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

# Build the launcher prefix once: either plain python (-m) or `accelerate launch ... -m`.
# When USE_ACCELERATE=1 we keep `-m <module>` so command shape stays uniform.
if [[ "$USE_ACCELERATE" == 1 ]]; then
    if ! command -v accelerate >/dev/null 2>&1 && ! "$PYBIN" -c 'import accelerate' >/dev/null 2>&1; then
        echo "USE_ACCELERATE=1 but accelerate not importable in $PYBIN" >&2
        exit 2
    fi
    LAUNCH=("$PYBIN" -m accelerate.commands.launch --config_file "$ACCEL_CONFIG" --num_processes "$NUM_PROCESSES")
else
    LAUNCH=("$PYBIN")
fi

mkdir -p "$RUN_ROOT"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TRL_EXPERIMENTAL_SILENCE=1

# ---------- helpers ----------
log()       { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_ROOT/orchestration.log"; }
section()   { echo; echo "============================================================"; log "$*"; echo "============================================================"; }
do_run()    { if [[ "$DRY_RUN" == 1 ]]; then echo "+ $*"; else "$@"; fi; }

prune_intermediate_ckpts() {
    # Remove HF Trainer `checkpoint-XXX` sub-dirs once the run has flushed
    # the canonical weights into `<output_dir>/model.safetensors` via save_model().
    local OUTDIR="$1"
    if [[ -d "$OUTDIR" ]]; then
        find "$OUTDIR" -maxdepth 1 -type d -name 'checkpoint-*' -print -exec rm -rf {} +
    fi
}

cleanup_gpu() {
    section "Cleanup GPU processes"
    do_run pkill -f 'training.train_' || true
    sleep 5
    do_run pkill -9 -f 'training.train_' || true
    sleep 3
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$RUN_ROOT/orchestration.log"
}

# ---------- per-model pipeline ----------
train_model() {
    local MODEL_PATH="$1"
    local TAG="$2"
    local SEQ_LEN="${3:-1536}"

    if [[ ! -d "$MODEL_PATH" ]]; then
        log "!! model path missing: $MODEL_PATH — skipping $TAG"
        return 0
    fi

    local RUN_DIR="$RUN_ROOT/$TAG"
    mkdir -p "$RUN_DIR"

    local SFT_DIR="$RUN_DIR/sft_warmup"
    local S1_DIR="$RUN_DIR/opd_stage1"
    local S2_DIR="$RUN_DIR/opd_stage2"
    local S3_DIR="$RUN_DIR/opd_stage3"

    section "[$TAG] step 1/4: SFT warmup ($MODEL_PATH)"
    # NOTE: full RolePlay_Villain.json -> ~273k turn SFT samples (50x bigger than
    # the 5k tier-1 pilot). That defeats the "small + complex loss" thesis and
    # overfits role-play dialogue. Cap to a few hundred scenarios -> ~5-6k turns,
    # matching tier-1 scale.
    if [[ -f "$SFT_DIR/model.safetensors" && -f "$SFT_DIR/preprocessor_config.json" ]]; then
        log "  SFT warmup ckpt already present ($SFT_DIR) — skipping. Set FORCE_SFT=1 to retrain."
        if [[ "${FORCE_SFT:-0}" == "1" ]]; then
            log "  FORCE_SFT=1 -> retraining SFT warmup."
        else
            :  # noop, skip the launch
        fi
    fi
    if [[ ! -f "$SFT_DIR/model.safetensors" || ! -f "$SFT_DIR/preprocessor_config.json" || "${FORCE_SFT:-0}" == "1" ]]; then
        do_run "${LAUNCH[@]}" -m training.train_sft_warmup \
            --model "$MODEL_PATH" \
            --output-dir "$SFT_DIR" \
            --limit-scenarios "${SFT_LIMIT_SCENARIOS:-500}" \
            --max-seq-length "$SEQ_LEN" \
            --per-device-bsz 2 \
            --grad-accum 8 \
            --lr 2e-5 \
            --epochs 1 \
            --bf16 \
            --report-to tensorboard \
            --logging-dir "$SFT_DIR/tb"
        prune_intermediate_ckpts "$SFT_DIR"
    fi

    section "[$TAG] step 2/4: OPD stage1 (M(c)<0.3)"
    do_run "${LAUNCH[@]}" -m training.train_opd \
        --config "$PROJECT_ROOT/training/configs/opd_stage1.yaml" \
        --override-output-dir   "$S1_DIR" \
        --override-student-model "$SFT_DIR" \
        --override-teacher-model "$SFT_DIR"
    prune_intermediate_ckpts "$S1_DIR"

    section "[$TAG] step 3/4: OPD stage2 (M(c)<0.6)"
    do_run "${LAUNCH[@]}" -m training.train_opd \
        --config "$PROJECT_ROOT/training/configs/opd_stage2.yaml" \
        --override-output-dir   "$S2_DIR" \
        --override-student-model "$S1_DIR" \
        --override-teacher-model "$SFT_DIR"
    prune_intermediate_ckpts "$S2_DIR"
    # stage1 weights no longer needed (stage2 supersedes), drop the heavy files.
    rm -f "$S1_DIR/model.safetensors" "$S1_DIR"/model-*.safetensors 2>/dev/null || true

    section "[$TAG] step 4/4: OPD stage3 (full + L4 oversample=8)"
    do_run "${LAUNCH[@]}" -m training.train_opd \
        --config "$PROJECT_ROOT/training/configs/opd_stage3.yaml" \
        --override-output-dir   "$S3_DIR" \
        --override-student-model "$S2_DIR" \
        --override-teacher-model "$SFT_DIR"
    prune_intermediate_ckpts "$S3_DIR"
    # All prior stages are now subsumed by stage3. Keep only stage3 weights + tb.
    rm -f "$S2_DIR/model.safetensors" "$S2_DIR"/model-*.safetensors 2>/dev/null || true
    rm -f "$SFT_DIR/model.safetensors" "$SFT_DIR"/model-*.safetensors 2>/dev/null || true

    log "[$TAG] done. final ckpt at $S3_DIR"
    du -sh "$RUN_DIR"/* 2>/dev/null | tee -a "$RUN_ROOT/orchestration.log" || true
}

# ---------- preflight ----------
section "Preflight"
log "PROJECT_ROOT = $PROJECT_ROOT"
log "TIER1_JSONL  = $TIER1_JSONL"
log "RUN_ROOT     = $RUN_ROOT"
log "ONLY_MODEL   = $ONLY_MODEL"
log "DRY_RUN      = $DRY_RUN"
log "USE_ACCELERATE = $USE_ACCELERATE  (NUM_PROCESSES=$NUM_PROCESSES, config=$ACCEL_CONFIG)"
log "GPU snapshot:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | tee -a "$RUN_ROOT/orchestration.log"

if [[ ! -f "$TIER1_JSONL" ]]; then
    log "!! tier1.jsonl missing: $TIER1_JSONL"
    log "   run training/scripts/build_pilot_5k.sh first."
    exit 2
fi

# ---------- run ----------
if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "4b" ]]; then
    train_model "$QWEN_4B"  "qwen35_4b"  1536
    cleanup_gpu
fi

if [[ "$ONLY_MODEL" == "all" || "$ONLY_MODEL" == "9b" ]]; then
    train_model "$QWEN_9B"  "qwen35_9b"  1536
    cleanup_gpu
fi

section "All done"
log "Tensorboard:"
log "   tensorboard --logdir $RUN_ROOT --port 6006"
log "Final checkpoints:"
ls -d "$RUN_ROOT"/*/opd_stage3 2>/dev/null | tee -a "$RUN_ROOT/orchestration.log" || true
