#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Step 2 - Train a referring-LiDAR model (multi-GPU DDP) from the feature cache.
#
# META_ARCH selects the architecture:
#   refer_model           base (per-proposal query init + rotated deformable attn)
#   refer_model_lang_dec  + per-decoder-layer language cross-attention
#   refer_model_angle     + SEED-style iterative orientation refinement
#
# REFER_DATA_DIR selects the negative-sampling ablation set (sampled_no_neg /
# sampled_baseline / sampled_balanced / sampled_all_neg). All hyper-parameters
# below are the thesis defaults; override any via environment or trailing flags.
#
# Example (language-decoder model, 2 GPUs):
#   META_ARCH=refer_model_lang_dec \
#   REFER_DATA_DIR=/data/nuscenes/ablation_fixed/sampled_baseline \
#   FEATURE_CACHE_DIR=/data/cache/pointpillars_fp16 \
#   OUTPUT_DIR=/data/checkpoints/langdec_baseline \
#   bash scripts/train.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
REPO_ROOT="$(cd "$HERE/.." && pwd)"; export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

META_ARCH="${META_ARCH:-refer_model}"
REFER_DATA_DIR="${REFER_DATA_DIR:?set REFER_DATA_DIR to a prepared ablation set}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for checkpoints}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:?set FEATURE_CACHE_DIR (from extract_cache.sh)}"
NUSCENES_DATAROOT="${NUSCENES_DATAROOT:-/data/nuscenes}"
POINTPILLARS_CKPT="${POINTPILLARS_CKPT:-/data/ckpts/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth}"
NPROC="${NPROC:-2}"                 # GPUs
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-3}"       # per GPU
QUERIES_PER_FRAME="${QUERIES_PER_FRAME:-2}"

mkdir -p "$OUTPUT_DIR"
echo "[train] meta_arch=$META_ARCH  gpus=$NPROC  epochs=$EPOCHS  set=$(basename "$REFER_DATA_DIR")"
torchrun --standalone --nproc_per_node="$NPROC" main_simple_ddp.py \
  --meta-arch "$META_ARCH" \
  --nuscenes-dataroot "$NUSCENES_DATAROOT" \
  --refer-data-dir "$REFER_DATA_DIR" \
  --pointpillars-ckpt "$POINTPILLARS_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --feature-cache-dir "$FEATURE_CACHE_DIR" --feature-cache-strict \
  --question-types-json configs/question_types_det.json \
  --sweeps-num 10 --point-cloud-range -50 -50 -5 50 50 3 \
  --set-cost-class 2 --set-cost-bbox 5 --set-cost-center 5 --set-cost-refer 2 --set-cost-refer-beta 0.35 \
  --cls-loss-coef 2 --bbox-loss-coef 5 --giou-loss-coef 2 --refer-loss-coef 2 --loss-3d-coef 2 --quality-loss-coef 1.0 \
  --focal-alpha 0.25 --lr 1e-4 --lr-drop 8 --weight-decay 1e-3 \
  --batch-size "$BATCH_SIZE" --num-workers 4 --queries-per-frame "$QUERIES_PER_FRAME" \
  --proposal-queries 150 --proposal-w-from dy \
  --hidden-dim 256 --dim-feedforward 1024 --nheads 8 --enc-layers 6 --dec-layers 6 --dropout 0.2 \
  --num-feature-levels 3 --dec-n-points 4 --enc-n-points 4 --dga-grid-size 5 \
  --epochs "$EPOCHS" \
  "$@"
