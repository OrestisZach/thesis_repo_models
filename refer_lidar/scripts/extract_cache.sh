#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Step 1 - Build the frozen-detector feature cache.
#
# Runs the frozen PointPillars detector over every frame and stores, per frame:
#   * the 3 BEV FPN feature maps,
#   * the top-K proposal boxes + objectness scores,
#   * the per-proposal heading (yaw), used by the +angle model.
# Training and evaluation read this cache instead of re-running the detector.
#
# SPLIT=all writes both train and val.
#
# Example:
#   REFER_DATA_DIR=/data/nuscenes/ablation_fixed/sampled_all_neg \
#   OUTPUT_DIR=/data/cache/pointpillars_fp16 \
#   bash scripts/extract_cache.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
REPO_ROOT="$(cd "$HERE/.." && pwd)"; export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

NUSCENES_DATAROOT="${NUSCENES_DATAROOT:-/data/nuscenes}"
REFER_DATA_DIR="${REFER_DATA_DIR:?set REFER_DATA_DIR to the prepared refer-query dir}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for the feature cache}"
POINTPILLARS_CKPT="${POINTPILLARS_CKPT:-/data/ckpts/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth}"
POINTPILLARS_CONFIG="${POINTPILLARS_CONFIG:-$REPO_ROOT/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py}"
SPLIT="${SPLIT:-all}"                       # train | val | all
SWEEPS_NUM="${SWEEPS_NUM:-10}"              # 10 past sweeps + current = 11 frames aggregated
PROPOSAL_QUERIES="${PROPOSAL_QUERIES:-150}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"         # float16 halves cache size with no measurable effect
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$OUTPUT_DIR"
echo "[extract_cache] split=$SPLIT  dtype=$SAVE_DTYPE  out=$OUTPUT_DIR"
python -u extract_pointpillars_feature_maps_predictions.py \
  --nuscenes-dataroot "$NUSCENES_DATAROOT" \
  --refer-data-dir "$REFER_DATA_DIR" \
  --pointpillars-config "$POINTPILLARS_CONFIG" \
  --pointpillars-ckpt "$POINTPILLARS_CKPT" \
  --split "$SPLIT" \
  --sweeps-num "$SWEEPS_NUM" \
  --proposal-queries "$PROPOSAL_QUERIES" \
  --num-point-features 4 \
  --point-cloud-range -50 -50 -5 50 50 3 \
  --save-dtype "$SAVE_DTYPE" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "$@"
