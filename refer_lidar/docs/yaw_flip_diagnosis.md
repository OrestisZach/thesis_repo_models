# Yaw-flip diagnosis: why +angle improves mAP but worsens mAOE

**Date:** 2026-07-06 · **Script:** [`diagnose_yaw_flips.py`](../diagnose_yaw_flips.py) ·
**Data:** ghost-clean 80 % val eval (`eval_ghostclean_wave2_{langdec,angle}_baseline`, shard 0 of 10)

## Observation (ghost-clean 80 % val, mean over 6 question types)

| model | mean-of-types mAP | macro mAOE (rad, ↓) |
|---|---|---|
| `refer_model_lang_dec` (+lang, free sin/cos heading) | 0.4189 | **0.4797** |
| `refer_model_angle` (+lang+yaw, `prop_yaw + Σ Δθ` heading) | **0.4256** | 0.5057 |

The angle model wins mAP on **all six** question types yet loses mAOE by 0.026 —
counterintuitive, since it adds a *dedicated* heading loss (`loss_rad`,
wrapped-L1, deep supervision on all 6 decoder layers) that the +lang model
does not have.

## Experiment

Both models were evaluated on the **identical** frames, queries and cached
proposals (`shard_0_of_10.pkl`, 108,578 query records each). For every query
record we replicate the devkit-style TP matching (confidence-sorted greedy
nearest-unmatched-GT, 2.0 m center distance) and compute the per-TP
orientation error with the nuScenes class rules (traffic_cone excluded,
barrier mod π). Because the matching input is identical, the two models'
TPs can be **paired** on `(record, matched GT)` — isolating the heading
difference from any detection difference.

Run: `python diagnose_yaw_flips.py` (CPU, ~4 min; paths at top of script).

## Results

45,647 TPs per model; 45,582 paired.

| statistic | +lang (free sin/cos) | +angle (prop+Δθ) |
|---|---|---|
| median yaw error | 0.0868 | **0.0656** |
| TPs with error < 0.1 rad | 53.3 % | **59.8 %** |
| paired mean error, non-flipped boxes | — | **−0.0154** (angle better) |
| flip rate (error > π/2) | **12.32 %** | 13.52 % |
| paired mean gap (angle − langdec) | — | +0.0210 |

Decomposition of the paired gap:

| component | contribution |
|---|---|
| direction flips (π-errors) | **+0.0256** |
| fine alignment (non-flipped) | **−0.0046** |

**The angle head is *better* at orientation (24 % lower median error) —
the entire mAOE regression is front/back direction flips.** One flip costs
≈ π ≈ 48 median-quality alignments in the mAOE mean. "Only-angle-flipped"
paired TPs: 2,153 vs 1,604 "only-langdec-flipped".

Per-class: langdec flips less on car/bus/truck/pedestrian; angle flips less
on bicycle/motorcycle/construction_vehicle. Net +1.2 pp flips for angle.

## Why — mechanism

1. **mAP is orientation-blind** (center-distance matching), and the rotated
   deformable-sampling grid is symmetric under a 180° rotation — a flipped
   yaw produces the *identical* sampling footprint. So flips cannot hurt
   mAP, while the (genuinely better) fine alignment improves feature
   aggregation → mAP up everywhere.
2. **The output heading is anchored to the frozen PointPillars proposal yaw**
   (`pred_yaw = prop_yaw + Σ Δθ`), and PointPillars' own heading quality is
   mAOE ≈ 0.53 — including its direction-classifier mistakes. To repair an
   inherited flip the residual head must output Δ ≈ π; wrapped-L1 on a
   bimodal target population (flipped / not flipped, indistinguishable
   features) has cancelling gradients, so flips survive training. The free
   sin/cos head decides direction per-feature and is right more often.
3. **SEED does not have this failure mode** (reference: `seed files/SEED-main`):
   its per-layer heads *re-predict* the full box including a **bounded**
   angle `(θ+π)/2π` refined in inverse-sigmoid space
   (`Det3DHead.forward`, `seed_head.py:655`), its anchor is the previous
   layer's own jointly-trained prediction (not a frozen detector), and its
   Hungarian matcher includes an angle cost (`cost_rad`,
   `hungarian_assigner.py:168`) keeping assignments angle-consistent.

## A proposed remedy (future work)

The final model keeps the +angle model with the surviving direction flips as a
conscious mAOE trade-off; a proper fix is left as future work. The natural
remedy is to decompose direction from the fine angle — the standard
PointPillars/SECOND dir-classifier pattern, adapted to the residual head:

- `angle_embed` (unchanged) refines the heading **axis**; `loss_rad` becomes
  the **mod-π (axial) wrapped-L1** `|½·atan2(sin 2δ, cos 2δ)|`, the regime
  where the head already beats free regression;
- a new per-layer binary `flip_embed` picks the semicircle, BCE-supervised
  with label `|wrap(pred_yaw_detached − gt_yaw)| > π/2`;
- reported heading = `pred_yaw + π·[σ(pred_flip) > 0.5]`;
- zero-init ⇒ σ = 0.5 ⇒ strict `>` keeps flips off at step 0 — training
  starts exactly at the angle model.

Expected: flip rate drops to ≤ the free-head level while keeping the
−0.015 fine-angle advantage ⇒ mAOE ≈ 0.46–0.47 (beating +lang's 0.4797),
mAP unchanged (rotation path identical, matching orientation-blind).
