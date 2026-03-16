#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-240}
AMP=${2:-true}
NUM_EPOCHS=${3:-100}
LEARNING_RATE=${4:-0.002}
WEIGHT_DECAY=${5:-0.1}

# choices: 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
#          'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'
TASK=homo

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
  if [[ "$GPU_CC" == 12.* ]]; then
    echo "[WARN] Detected GPU compute capability $GPU_CC (sm_120 class)." >&2
    echo "[WARN] Your current PyTorch/CUDA build may not support this architecture." >&2
    echo "[WARN] If training fails with nvrtc -arch errors, update PyTorch/CUDA to a build that supports sm_120." >&2
  fi
fi

python -m se3_transformer.runtime.training \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --use_layer_norm \
  --norm \
  --save_ckpt_path model_qm9.pth \
  --precompute_bases \
  --seed 42 \
  --task "$TASK"