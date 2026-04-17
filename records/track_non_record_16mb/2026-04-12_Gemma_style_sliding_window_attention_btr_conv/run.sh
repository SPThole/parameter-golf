#!/bin/bash
# ============================================================
# 2026-04-11_newSota_copy_gl_att
# Global/Local Attention (Gemma-4 style) on Parallel Residuals base
#
# Branched from 2026-04-11_newSota (PR #1523 parallel residuals).
# Adds Gemma-4-style alternating global/local attention:
#   - Global layers: full causal attention (attend to all previous tokens)
#   - Local layers: sliding window attention (attend to last N tokens only)
#   - Pattern alternates: G,L,G,L,... (configurable via GL_ATT_MODE/STRIDE)
#   - ~30% faster per-step on local layers (less memory, fewer FLOPs)
#
# Usage:
#   MODE=1H100 ./run.sh          # Single H100 (long wallclock, grad accum=8)
#   MODE=8H100 ./run.sh          # 8×H100 cluster (600s wallclock, distributed)
#   SEED=42 MODE=8H100 ./run.sh  # Specify seed
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="2026-04-11_newSota_copy_gl_att"
cd /workspace/parameter-golf


# ---------- 0. Mode selection ----------
MODE="${MODE:-1H100}"
echo "=== Mode: ${MODE} ==="

# ---------- 1. Dependencies ----------
echo "=== Installing dependencies ==="
pip install -q brotli sentencepiece numpy tqdm huggingface-hub datasets tiktoken typing-extensions==4.15.0 setuptools kernels
# Flash Attention 3 (required for sliding window support)
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ \
  2>/dev/null || echo "WARN: flash_attn_3 wheel install failed — trying flash-attn-interface..."

# CUTLASS EVT fusion (build from source if .so not present)
if [ ! -f "${SCRIPT_DIR}/cutlass_evt_fusion/cutlass_evt_fusion.cpython-312-x86_64-linux-gnu.so" ]; then
    echo "=== Building CUTLASS EVT fusion from source ==="
    if [ ! -d "/opt/cutlass" ]; then
        git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
        cd /opt/cutlass
        git checkout 08185b9c3e90510ee2b656662ed0d53b06d28157
        cd -
    fi
    pip install --no-build-isolation "${SCRIPT_DIR}/cutlass_evt_fusion"
fi

# ---------- 2. Download data & tokenizer ----------
echo "=== Downloading SP8192 tokenizer & FineWeb shards ==="
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# ---------- 3. Shared hyperparameters ----------
export SEED="${SEED:-1337}"
export VOCAB_SIZE=8192
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export EVAL_STRIDE=64
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"

# --- Architecture (same as newSota base) ---
export NUM_LAYERS=11
export XSA_LAST_N=11
export MODEL_DIM=512
export EMBEDDING_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4.0
export ROPE_DIMS=16
export ROPE_BASE=10000.0
export LN_SCALE=1
export QK_GAIN_INIT=5.0
export LOGIT_SOFTCAP=30.0
export TIE_EMBEDDINGS=1
export SKIP_GATES_ENABLED=1

# --- Parallel Residuals ---
export PARALLEL_RESIDUAL=1
export PARALLEL_RESIDUAL_START=8
export PARALLEL_START_LAYER_IS_PHYSICAL=1
export PARALLEL_FINAL_LANE=mlp
export PARALLEL_FREEZE_LANE0=0

# --- Layer Looping ---
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35

# === NEW: Global/Local Attention (Gemma-4 style) ===
export GL_ATT_ENABLED=1
export GL_ATT_MODE=glg          # Global-Local-Global alternating
export GL_ATT_STRIDE=1          # 1 = alternate every layer: G,L,G,L,G,L,...
export GL_ATT_WINDOW_SIZE=256   # Sliding window context for local layers

# --- Learning rates ---
export MATRIX_LR=0.022
export SCALAR_LR=0.02
export TIED_EMBED_LR=0.03
export TIED_EMBED_INIT_STD=0.005
export EMBED_LR=0.6
export HEAD_LR=0.008
export MIN_LR=0.0

# --- Weight decay ---
export MUON_WD=0.095
export ADAM_WD=0.02
export EMBED_WD=0.085

# --- Muon optimizer ---
export MUON_MOMENTUM=0.97
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_BACKEND_STEPS=5
export MUON_ROW_NORMALIZE=1

# --- Adam ---
export BETA1=0.9
export BETA2=0.95
export ADAM_EPS=1e-8

# --- Grad clipping ---
export GRAD_CLIP_NORM=0.3

# --- EMA ---
export EMA_DECAY=0.9965

# --- GPTQ Quantization ---
export GPTQ_CALIBRATION_BATCHES=64
export MATRIX_BITS=6
export EMBED_BITS=8
export MATRIX_CLIP_SIGMAS=12.85
export EMBED_CLIP_SIGMAS=20.0
export COMPRESSOR=brotli

# --- Sliding window eval ---
export SLIDING_WINDOW_ENABLED=1

# --- TTT (Test-Time Training) ---
export TTT_ENABLED=1
export TTT_LR=0.01
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export TTT_CHUNK_TOKENS=32768

# ---------- 4. Mode-specific configuration ----------
if [ "${MODE}" = "8H100" ]; then
    # ===== 8×H100 SXM cluster (competition setting) =====
    echo "=== 8×H100: 600s wallclock, distributed training ==="
    export MAX_WALLCLOCK_SECONDS=600
    export TRAIN_BATCH_TOKENS=786432
    export ITERATIONS=20000
    export WARMDOWN_FRAC=0.72
    export WARMUP_STEPS=20
    export GPTQ_RESERVE_SECONDS=13

    export RUN_ID="${EXP_NAME}_8xH100_seed${SEED}"
    echo "=== ${EXP_NAME} seed=${SEED} 8×H100 ==="
    echo "=== GL_ATT: ${GL_ATT_MODE} stride=${GL_ATT_STRIDE} window=${GL_ATT_WINDOW_SIZE} ==="
    torchrun --standalone --nproc_per_node=8 "${SCRIPT_DIR}/train_gpt_human.py" 2>&1 \
        | tee "${SCRIPT_DIR}/logs_8xH100_seed${SEED}.txt"

elif [ "${MODE}" = "1H100" ]; then
    # ===== 1×H100 (longer wallclock, grad accum compensates) =====
    echo "=== 1×H100: 4800s wallclock, single GPU ==="
    export MAX_WALLCLOCK_SECONDS=4800
    export TRAIN_BATCH_TOKENS=786432
    export ITERATIONS=20000
    export WARMDOWN_FRAC=0.72
    export WARMUP_STEPS=20
    export GPTQ_RESERVE_SECONDS=60

    export RUN_ID="${EXP_NAME}_1xH100_seed${SEED}"
    echo "=== ${EXP_NAME} seed=${SEED} 1×H100 ==="
    echo "=== GL_ATT: ${GL_ATT_MODE} stride=${GL_ATT_STRIDE} window=${GL_ATT_WINDOW_SIZE} ==="
    python3 "${SCRIPT_DIR}/train_gpt_human.py" 2>&1 \
        | tee "${SCRIPT_DIR}/logs_1xH100_seed${SEED}.txt"

else
    echo "ERROR: Unknown MODE='${MODE}'. Use MODE=1H100 or MODE=8H100."
    exit 1
fi

echo "=== ${EXP_NAME} COMPLETE ==="
