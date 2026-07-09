#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Step 3 - Resumable, sharded evaluation on the all_neg val split.
#
# The val set is split into NUM_SHARDS contiguous chunks; each shard runs on one
# GPU and writes its own predictions pkl. Re-running the SAME command skips any
# shard whose pkl already exists, so the job can be interrupted and resumed with
# at most one shard of lost work. When all shards exist, it merges them into the
# per-type / per-class metrics json on CPU (no GPU needed).
#
# Metric: nuScenes-style center-distance mAP, reported as mean-over-question-types.
#
# Protocol (thesis defaults, faithful to the nuScenes devkit):
#   MAP_MIN_CONF=0           no confidence floor
#   EGO_RANGE_TRANSFORM      per-class ego-distance range filtering of GT AND
#                            predictions; auto-discovered at
#                            $DATA_ROOT/ego_range_transform.json if present
#                            (build it with build_ego_range_transform.py).
#   GT_BLACKLIST (optional)  exclude annotation tokens the devkit never scores;
#                            needed ONLY for datasets generated before the
#                            generator applied devkit eligibility at source.
#
# Example (final model = +angle, on the winning baseline r=0.3):
#   META_ARCH=refer_model_angle \
#   MODEL_CKPT=/data/checkpoints/angle_baseline/checkpoint_best.pth \
#   FEATURE_CACHE_DIR=/data/cache/pointpillars_fp16 \
#   OUT_DIR=/data/outputs/eval_angle_baseline \
#   bash scripts/eval.sh
#
# Parallelise across GPUs with disjoint shard ranges into the SAME OUT_DIR:
#   SHARD_MIN=0  SHARD_MAX=11 DEVICE=cuda:0 bash scripts/eval.sh &
#   SHARD_MIN=12 SHARD_MAX=23 DEVICE=cuda:1 bash scripts/eval.sh &
# The merge runs once all shards exist.
#
# The tight 0.2 m center-distance NMS that defines the FINAL model is applied at
# merge time with MERGE_NMS_RADIUS=0.2 (default 0 = the NMS-free ablation number).
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
REPO_ROOT="$(cd "$HERE/.." && pwd)"; export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

META_ARCH="${META_ARCH:-refer_model}"
MODEL_CKPT="${MODEL_CKPT:?set MODEL_CKPT to checkpoint_best.pth}"
EVAL_REFER_DIR="${EVAL_REFER_DIR:-/data/nuscenes/ablation_fixed/sampled_all_neg}"
DATA_ROOT="${DATA_ROOT:-/data/nuscenes}"
POINTPILLARS_CKPT="${POINTPILLARS_CKPT:-/data/ckpts/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-}"
QUESTION_TYPES_JSON="${QUESTION_TYPES_JSON:-configs/question_types_det.json}"
MAP_CONF_MODE="${MAP_CONF_MODE:-refer}"
MAP_MIN_CONF="${MAP_MIN_CONF:-0}"           # 0 = no confidence floor (thesis protocol)
MERGE_NMS_RADIUS="${MERGE_NMS_RADIUS:-0}"   # 0.2 => the final-model tight center-distance NMS
DEVICE="${DEVICE:-cuda:0}"
NUM_SHARDS="${NUM_SHARDS:-24}"              # MUST stay constant across resumes
SHARD_MIN="${SHARD_MIN:-0}"
SHARD_MAX="${SHARD_MAX:-$((NUM_SHARDS-1))}"
OUT_DIR="${OUT_DIR:?set OUT_DIR for shard pkls + metrics json}"

# Optional devkit-protocol files.
GT_BLACKLIST="${GT_BLACKLIST:-}"           # legacy datasets only; empty = skip
EGO_RANGE_TRANSFORM="${EGO_RANGE_TRANSFORM:-$DATA_ROOT/ego_range_transform.json}"

mkdir -p "$OUT_DIR"; LOG="$OUT_DIR/eval.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
[ -f "$MODEL_CKPT" ] || { log "FATAL: model ckpt not found: $MODEL_CKPT"; exit 1; }

# Assemble optional protocol args.
PROTO_ARGS=()
if [ -n "$GT_BLACKLIST" ]; then
  [ -f "$GT_BLACKLIST" ] || { log "FATAL: GT_BLACKLIST set but not found: $GT_BLACKLIST"; exit 1; }
  PROTO_ARGS+=(--gt-blacklist "$GT_BLACKLIST")
fi
if [ -f "$EGO_RANGE_TRANSFORM" ]; then
  PROTO_ARGS+=(--ego-range-transform "$EGO_RANGE_TRANSFORM")
else
  log "NOTE: ego-range transform not found at $EGO_RANGE_TRANSFORM — running without devkit range filtering."
fi
CACHE_ARGS=(); [ -n "$FEATURE_CACHE_DIR" ] && CACHE_ARGS=(--feature-cache-dir "$FEATURE_CACHE_DIR")

log "resumable eval: meta_arch=$META_ARCH shards=${SHARD_MIN}..${SHARD_MAX}/$NUM_SHARDS ckpt=$MODEL_CKPT out=$OUT_DIR"

for i in $(seq "$SHARD_MIN" "$SHARD_MAX"); do
  pkl="$OUT_DIR/shard_${i}_of_${NUM_SHARDS}.pkl"
  if [ -f "$pkl" ]; then log "shard $i/$NUM_SHARDS: exists -> skip"; continue; fi
  log "shard $i/$NUM_SHARDS: START"
  python -u inference_lidar_simple.py \
    --mode eval --meta-arch "$META_ARCH" \
    --data-root "$DATA_ROOT" --ann-file nuscenes_infos_val.pkl \
    --refer-data-dir "$EVAL_REFER_DIR" --eval-split val \
    --pointpillars-ckpt "$POINTPILLARS_CKPT" --model-ckpt "$MODEL_CKPT" \
    --output-dir "$OUT_DIR" --question-types-json "$QUESTION_TYPES_JSON" \
    --map-conf-mode "$MAP_CONF_MODE" --map-min-conf "$MAP_MIN_CONF" --eval-chunk-size 32 \
    --dist-threshold 2.0 --num-feature-levels 3 --proposal-queries 150 --proposal-w-from dy \
    --point-cloud-range -50 -50 -5 50 50 3 --eval-use-all-proposals --device "$DEVICE" \
    --num-shards "$NUM_SHARDS" --shard-id "$i" --save-predictions "$pkl" \
    "${PROTO_ARGS[@]}" "${CACHE_ARGS[@]}" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$pkl" ]; then
    log "shard $i/$NUM_SHARDS: INCOMPLETE (rc=$rc). Re-run the same command to resume from shard $i."
    exit 1
  fi
  log "shard $i/$NUM_SHARDS: DONE"
done

n_done=$(ls "$OUT_DIR"/shard_*_of_${NUM_SHARDS}.pkl 2>/dev/null | wc -l)
if [ "$n_done" -lt "$NUM_SHARDS" ]; then
  log "range ${SHARD_MIN}..${SHARD_MAX} done; ${n_done}/${NUM_SHARDS} shards exist overall — merge deferred."
  exit 0
fi

log "all $NUM_SHARDS shards saved. MERGING -> metrics json (CPU-only)..."
python -u inference_lidar_simple.py \
  --mode eval --meta-arch "$META_ARCH" --map-conf-mode "$MAP_CONF_MODE" \
  --map-min-conf "$MAP_MIN_CONF" --dist-threshold 2.0 --merge-nms-radius "$MERGE_NMS_RADIUS" \
  --refer-data-dir "$EVAL_REFER_DIR" --pointpillars-ckpt "$POINTPILLARS_CKPT" \
  --output-dir "$OUT_DIR" \
  --merge-predictions "$OUT_DIR"/shard_*_of_${NUM_SHARDS}.pkl >> "$LOG" 2>&1
[ $? -eq 0 ] && log "MERGE DONE -> $OUT_DIR/eval_val_all_types.json" || log "MERGE FAILED"
