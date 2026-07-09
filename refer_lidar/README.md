# Referring 3D Detection on LiDAR

Language-guided 3D object detection on nuScenes LiDAR. A frozen **PointPillars**
detector supplies BEV features and oriented box proposals; a Deformable-DETR–style
decoder fuses them with **RoBERTa** language embeddings to ground a natural-language
query ("the closest car on the left", "find all pedestrians in front") to 3D boxes —
including correctly answering **null-target** queries ("there is no such object").

This directory is the entire contribution. It sits inside an unmodified
[mmdetection3d v1.4.0](../README.mmdet3d.md) tree, which it uses purely as a library
(frozen PointPillars, box structures, BEV NMS). Nothing in `../mmdet3d/` is changed.

---

## Setup (Docker)

The supported way to run everything is the prebuilt container: PyTorch 2.4 / CUDA 12.4,
the mmdet3d fork installed as a package, the MSDeformAttn CUDA op, `mmcv`/`mmengine`/
`mmdet`, transformers, and the nuScenes devkit are all baked in — **no manual build step**.
From the repository root:

```bash
REFER_DATA_ROOT=/path/to/nuscenes docker compose -f refer_lidar/docker-compose.yml build refer-lidar
REFER_DATA_ROOT=/path/to/nuscenes docker compose -f refer_lidar/docker-compose.yml run  --rm refer-lidar
```

This drops you in `/workspace/refer_lidar` with the pipeline ready. `REFER_DATA_ROOT` (your
nuScenes + prepared queries + caches) is mounted at `/data`; RoBERTa weights are pulled from
Hugging Face on first run. A bare-metal fallback is described under [Setup without Docker](#setup-without-docker).

---

## Architectures

`--meta-arch` selects one of three models forming a **cumulative ablation chain** — all
share the frozen detector, query initialisation, Hungarian matcher, and core losses, and
each row adds exactly one thing on top of the previous:

| `--meta-arch` | Adds |
|---|---|
| `refer_model` | Base: per-proposal query init + **rotated** deformable cross-attention (the k×k sampling grid is oriented by each proposal's yaw). |
| `refer_model_lang_dec` | + per-decoder-layer **language cross-attention** (each object query attends the full RoBERTa token sequence at every layer). |
| `refer_model_angle` | + SEED-style **iterative orientation refinement**: each decoder layer samples with the current heading estimate, predicts a residual Δθ (zero-init head), detaches, and the next layer samples with the refined heading; supervised by a wrapped-L1 `loss_rad`. |

The **final model** of the thesis is `refer_model_angle` followed by a tight **0.2 m
center-distance NMS** applied at evaluation-merge time (a cheap post-processing step, not a
fourth architecture). It removes the near-coincident duplicate boxes that two coincident
frozen-detector proposals occasionally produce, and lifts the mean-of-types mAP from
0.425 (NMS-free) to **0.452**. The cumulative chain itself is kept NMS-free so that each
row's gain is attributable purely to its one architectural change.

## Pipeline

Everything runs off a **feature cache**: the frozen detector is run once over the dataset
and its BEV feature maps + top-150 proposals (boxes, objectness scores, per-proposal yaw)
are stored to disk; training and evaluation read the cache instead of re-running the
detector.

```
scripts/extract_cache.sh   # 1. frozen PointPillars -> feature + proposal cache
scripts/train.sh           # 2. train a model (DDP) from the cache
scripts/eval.sh            # 3. resumable sharded evaluation -> per-type / per-class mAP
```

All three are parametrised by environment variables with thesis defaults baked in.

### 1. Build the cache
```bash
REFER_DATA_DIR=/data/nuscenes/ablation_fixed/sampled_all_neg \
OUTPUT_DIR=/data/cache/pointpillars_fp16 \
bash scripts/extract_cache.sh          # SPLIT=all -> train+val
```
Produces one `.pt` per frame under `{train,val}/features/` plus `index_{train,val,all}.json`.
Each payload holds `feature_maps` — the 3 FPN levels at their native (different)
resolutions: `256×200×200`, `256×100×100`, `256×50×50` — plus `proposal_boxes_8d`
(150 × 4, normalised `[cx,cy,w,l]`), `proposal_scores` (150), `proposal_yaw` (150, radians),
and the tag `detector: "pointpillars"`.

### 2. Train
```bash
META_ARCH=refer_model_lang_dec \
REFER_DATA_DIR=/data/nuscenes/ablation_fixed/sampled_baseline \
FEATURE_CACHE_DIR=/data/cache/pointpillars_fp16 \
OUTPUT_DIR=/data/checkpoints/langdec_baseline \
bash scripts/train.sh                  # NPROC=2 GPUs, EPOCHS=15 by default
```
`REFER_DATA_DIR` selects the negative-sampling ablation set (`sampled_no_neg` /
`sampled_baseline` r=0.3 / `sampled_balanced` r=1 / `sampled_all_neg`). Writes
`checkpoint_best.pth` + `metrics.csv`.

### 3. Evaluate
```bash
META_ARCH=refer_model_angle \
MODEL_CKPT=/data/checkpoints/angle_baseline/checkpoint_best.pth \
FEATURE_CACHE_DIR=/data/cache/pointpillars_fp16 \
MERGE_NMS_RADIUS=0.2 \
OUT_DIR=/data/outputs/eval_angle_baseline \
bash scripts/eval.sh
```
The exhaustive **all_neg val** split is split into `NUM_SHARDS=24` resumable chunks (each
writes its own pkl; re-run to resume; merged to a metrics json on CPU). Metric: nuScenes
center-distance mAP, reported as **mean-over-question-types**. `MERGE_NMS_RADIUS=0.2` gives
the final model; leave it at 0 for the NMS-free ablation number.

---

## Reproducing the thesis experiments

Each experiment below maps to a chapter of the results and to concrete commands. They all
assume the feature cache from step 1 and a prepared query dataset (see [Data](#data)). Set
`CACHE=/data/cache/pointpillars_fp16` and `SETS=/data/nuscenes/ablation_fixed` first.

### 1 — Negative-ratio ablation → picks the training set (§ "negative-sampling ratio")
Train the **base** model on all four negative-sampling sets and evaluate each on the common
`all_neg` val. Precision rises and recall falls monotonically with the ratio; mean-of-types
mAP peaks at the intermediate `baseline` (r = 0.3), which is adopted for every later model.
```bash
for SET in no_neg baseline balanced all_neg; do
  META_ARCH=refer_model REFER_DATA_DIR=$SETS/sampled_$SET \
    FEATURE_CACHE_DIR=$CACHE OUTPUT_DIR=/data/ckpts/base_$SET bash scripts/train.sh
  META_ARCH=refer_model MODEL_CKPT=/data/ckpts/base_$SET/checkpoint_best.pth \
    FEATURE_CACHE_DIR=$CACHE OUT_DIR=/data/out/base_$SET bash scripts/eval.sh
done
# figures (precision/recall/mAP vs r; per-type mAP): make_thesis_figures.py
```

### 2 — Language in the decoder (§ "language in the decoder")
Same recipe, `--meta-arch refer_model_lang_dec`, trained on `baseline`. Adding per-layer
language cross-attention improves the mean-of-types mAP (0.403 → 0.419), with the largest
gain on the hardest spatial type (`closest_in_sector`).

### 3 — Orientation refinement (§ "orientation and final training")
Train `--meta-arch refer_model_angle` on `baseline`. The added iterative Δθ tightens the
heading *axis* but leaves the ~180° direction flips inherited from the frozen proposals — a
structural finding analysed by:
```bash
python diagnose_yaw_flips.py <eval_shard.pkl>     # axis-vs-direction error split
# write-up: docs/yaw_flip_diagnosis.md
```

### 4 — Duplicate predictions and NMS → the final model (§ "duplicate predictions and NMS")
Two coincident frozen-detector proposals produce identical queries that decoder
self-attention cannot separate, yielding duplicate boxes. A tight 0.2 m center-distance NMS
removes exactly these and defines the final model (re-run step 3's eval with
`MERGE_NMS_RADIUS=0.2`). The radius study is documented in `nms_dedup.py`.

### 5 — Final per-type and per-class results (§ "main results", § "per-class detection")
The merged metrics json from the final eval already contains the per-question-type table and
the nuScenes 10-class macro. Because the `object_detection_all_category` type is one
"find all X" query per class, its per-class breakdown *is* a classic per-class detection
evaluation:
```bash
python analyze_object_detection_all_category.py <eval_shard.pkl>   # per-class AP / errors
```

### 6 — Class confusion (§ "class confusion")
Compare the confusion matrix of the full pipeline against the standalone detector (the
upper bound on class discrimination):
```bash
python confusion_from_shards.py '<OUT_DIR>/shard_*_of_24.pkl'   # pipeline confusion (no GPU)
python pointpillars_confusion.py --feature-cache-dir $CACHE ... # standalone-detector confusion
python plot_confusion_heatmap.py ...                            # render the heatmaps
```

### 7 — Proposal-quality ceiling (§ "proposal quality analysis")
The recall of the frozen proposals upper-bounds everything downstream:
```bash
python analyze_proposal_threshold.py --point-cloud-range -50 ...  # recall vs score-thr and top-K
python analyze_proposal_3d_noise.py ...                           # proposal center / yaw error
```

---

## Evaluation protocol

Evaluation replicates the official nuScenes devkit protocol (101-point AP with
min-recall/min-precision 0.1, TP metrics at 2 m with the devkit cummean, cone/barrier
orientation rules, per-class `ego_dist < range` filtering of GT **and** predictions before
matching, no confidence floor) and reports both per-question-type task metrics and the
nuScenes 10-class macro (`nuscenes_macro` in the output JSON) for direct comparison with
published detectors. Declared deviations: no NDS/mAVE/mAAE (no velocity/attribute
predictions); predictions are additionally clipped to the generation-time sensor-origin
range (the ego-side annulus has no GT by construction).

The exact per-class ego-distance range filtering needs a small per-keyframe transform,
auto-discovered by `eval.sh` at `$DATA_ROOT/ego_range_transform.json`:
```bash
python build_ego_range_transform.py --table-dir $NUSC/v1.0-trainval --out ego_range_transform.json
```
Datasets from the current generator already satisfy the devkit eligibility at source and
need nothing further. For a dataset generated **before** that (legacy), also pass a
GT blacklist (`GT_BLACKLIST=...` for `eval.sh`) so annotation tokens the devkit never scores
are excluded:
```bash
python build_gt_blacklist.py --table-dir $NUSC/v1.0-trainval --out gt_blacklist_trainval.json
```

---

## Layout

```
main_simple_ddp.py                        training entry point (DDP)
inference_lidar_simple.py                 inference / evaluation entry point
nuscenes_lidar_simple.py                  referring dataset (reads the feature cache)
extract_pointpillars_feature_maps_predictions.py   cache builder (frozen detector)
configs/question_types_det.json           the 6 query types
models/                                    the three architectures + shared components
  refer_model.py            base model + PointPillarsDetectorBridge
  refer_model_lang_dec.py   + per-layer language cross-attention
  refer_model_angle.py      + iterative orientation refinement
  deformable_transformer_plus.py / _final.py   decoder (rotated deformable attention)
  matcher.py                language-quality-aware Hungarian matcher
  ops/                      MSDeformAttn CUDA op (prebuilt in the container)
  structures/               box / Instances helpers
util/                                      box ops + misc (NestedTensor, inverse_sigmoid, ...)
scripts/                                   extract_cache.sh · train.sh · eval.sh
Dockerfile.fast · docker-compose.yml       the container (see Setup)
# analysis / thesis-reproduction:
analyze_proposal_threshold.py             proposal recall vs score-threshold and top-K
analyze_proposal_3d_noise.py              proposal center / yaw error analysis
analyze_object_detection_all_category.py  per-class AP on the "find all <class>" query type
pointpillars_confusion.py                 standalone-detector confusion matrix
class_confusion_from_pkl.py               pipeline confusion matrix from a single eval pkl
confusion_from_shards.py                  pipeline confusion matrix over eval shards (no GPU)
plot_confusion_heatmap.py                 confusion-matrix heatmap
nms_dedup.py                              center-distance NMS de-duplication study (the 0.2 m radius)
diagnose_yaw_flips.py                     yaw axis-vs-direction error analysis
make_thesis_figures.py                    regenerate the thesis figures
docs/yaw_flip_diagnosis.md                write-up: why +angle helps mAP but not mAOE
docs/val_subset_stability.md              write-up: the fixed 20% val subset is safe for model selection
```

## Setup without Docker

Only if you cannot use the container. Install mmdet3d's dependencies (`mmcv`, `mmengine`,
`mmdet`) per the [mmdetection3d v1.4.0 install guide](../docs), then compile the CUDA op once
(needs CUDA + a GPU-enabled PyTorch):
```bash
cd models/ops && bash make.sh        # or: python setup.py build_ext --inplace
```
The scripts add the repository root to `PYTHONPATH` so `import mmdet3d` resolves against
`../mmdet3d/`.

## Data

The referring-query dataset (the `ablation_fixed/sampled_*` directories) is produced by a
**separate** generation repository
([thesis_repo_dataset](https://github.com/OrestisZach/thesis_repo_dataset)) and consumed
here as prepared files via `--refer-data-dir`; nuScenes (`--nuscenes-dataroot`) and the
PointPillars checkpoint (`--pointpillars-ckpt`) are the other inputs.
