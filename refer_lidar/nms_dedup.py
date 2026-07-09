"""
Center-distance NMS (de-duplication) for Refer-LiDAR predictions.
================================================================

WHY THIS EXISTS
---------------
The frozen PointPillars detector occasionally emits *near-coincident* proposals
for the same object. Those become bit-identical decoder queries (same reference
box, same sampling angle), and the decoder self-attention -- being permutation
symmetric -- cannot break the tie, so the model returns two overlapping boxes
with (nearly) identical refer score. See the "closest truck" case: two `pred 0.72`
boxes stacked on one truck.

This is a real model output, not a rendering bug. Under nuScenes AP (greedy
center-distance matching, one GT once) a duplicate is a false positive, so the
NMS-free metric is a *conservative lower bound*. A tight center-distance NMS
removes exactly these duplicates.

RADIUS
------
The boxes we want to drop are essentially COINCIDENT (center distance ~0). Use a
SMALL radius (0.2 m) so we suppress only true duplicates and never merge genuine
dense neighbours (pedestrians / cones cluster within ~1-2 m and must survive).
Measured mean-of-types mAP on the final (+angle) model, all_neg val:
    r = 0.2 m  -> safe: removes duplicates, keeps dense objects  (recommended)
    r = 0.5 m  -> still a net gain
    r = 2.0 m  -> BACKFIRES: merges real neighbours, pedestrians collapse

WHERE THIS IS APPLIED IN THE EVAL PIPELINE (this repo)
------------------------------------------------------
    refer_lidar/inference_lidar_simple.py
      * `_nms_records_by_center(query_records, radius)`   (the merge-time version)
      * CLI flag `--merge-nms-radius <m>` (0 = off, default) on the eval/merge path
      * injection point: right before AP computation in the `--merge-predictions`
        branch (search for "DIAGNOSTIC center-distance NMS").
    Reproduce:  python -u inference_lidar_simple.py --mode eval \
                  --meta-arch refer_model_angle --merge-predictions <shards> \
                  --merge-nms-radius 0.2 --output-dir <out>

FOR THE VISUALIZATION APP
-------------------------
The service returns the raw top-k object queries with no dedup. Apply
`nms_keep_indices` on the BEV centers (x, y) of the predicted boxes, ranked by
refer score, then keep only the returned indices across ALL parallel arrays
(boxes, scores, labels, ...). Do the confidence threshold FIRST, then dedup.
"""
from __future__ import annotations
import numpy as np


def nms_keep_indices(centers_xy, scores, radius=0.2):
    """Greedy center-distance NMS on BEV centers.

    Args:
        centers_xy: (N, 2) array-like of BEV (x, y) box centers, in metres.
        scores:     (N,) array-like ranking score (use the *refer* score).
        radius:     suppression radius in metres (0.2 recommended; <=0 = no-op).

    Returns:
        np.ndarray of kept row indices (int), in descending-score order.
        Boxes kept greedily best-first; any box whose center lies within
        `radius` of an already-kept box is dropped.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    n = min(centers.shape[0], scores.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=int)
    if radius is None or float(radius) <= 0.0 or n == 1:
        return np.argsort(-scores[:n], kind="stable")

    centers = centers[:n]
    scores = scores[:n]
    r2 = float(radius) ** 2
    # pairwise squared distance, then walk best-first suppressing neighbours
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)          # (n, n)
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in np.argsort(-scores, kind="stable"):
        if suppressed[i]:
            continue
        keep.append(int(i))
        nb = d2[i] < r2
        nb[i] = False
        suppressed |= nb
    return np.asarray(keep, dtype=int)


def dedup_predictions(boxes, scores, radius=0.2, score_threshold=None):
    """Convenience wrapper for the viz app.

    Args:
        boxes:  (N, >=2) predicted boxes; columns 0,1 must be BEV center (x, y).
        scores: (N,) refer score.
        radius: suppression radius (m), default 0.2.
        score_threshold: if given, drop boxes below it BEFORE dedup.

    Returns:
        keep_idx: indices into the ORIGINAL arrays to retain (desc score order).
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    idx = np.arange(min(boxes.shape[0], scores.shape[0]))
    if score_threshold is not None:
        idx = idx[scores[idx] >= float(score_threshold)]
    if idx.size == 0:
        return idx
    local = nms_keep_indices(boxes[idx, :2], scores[idx], radius=radius)
    return idx[local]


if __name__ == "__main__":
    # tiny self-test: two identical boxes + one distinct neighbour
    boxes = np.array([[10.0, 5.0, 4.0, 2.0],
                      [10.0, 5.0, 4.0, 2.0],   # exact duplicate of row 0
                      [11.2, 5.0, 4.0, 2.0]])  # 1.2 m away -> kept at r=0.2
    scores = np.array([0.72, 0.72, 0.55])
    keep = dedup_predictions(boxes, scores, radius=0.2)
    print("kept indices:", keep.tolist(), "(expected [0, 2] -> duplicate dropped)")
