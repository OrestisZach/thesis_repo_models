#!/usr/bin/env python3
"""Sweep proposal score thresholds on the training set and compute recall.

For each frame with cached CenterPoint proposals, this script:
  1. Loads proposals and GT objects
  2. For a range of score thresholds, filters proposals
  3. Runs Hungarian matching (BEV IoU cost) between filtered proposals and GT
  4. Records recall, avg proposals kept, and per-class breakdown

Usage (inside docker):
    python analyze_proposal_threshold.py \
        --feature-cache-dir /data/feature_maps_predictions \
        --refer-data-dir /data/nuscenes/refer_detection_with_negatives \
        --data-root /data/nuscenes \
        --split train
"""
import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet3d.utils import register_all_modules
register_all_modules(init_default_scope=True)

from nuscenes_lidar_simple import (DETECTION_RANGE, build as build_refer_dataset,
                                   metres_to_norm)


def gt_classes_for_frame(dataset, flat_idx: int) -> list:
    """Class names positionally aligned with dataset._build_targets() output.

    Replaces the old obj_id-matching loop, which silently failed for ~half the
    GT: obj_id = scene_idx * 1_000_000 + local_id exceeds float32's exact
    integer range (2^24) for scene_idx >= 17, and targets['obj_ids'] is a
    float32 tensor — so .item() != float(obj['obj_id']) and the class fell
    back to '?'. Here we simply replicate _build_targets' filter conditions
    and collect the class name of every KEPT object, in order.
    """
    objects = dataset.frame_objects[flat_idx]
    pc_range = dataset.point_cloud_range
    out = []
    for obj in objects:
        cls_id = obj['class_id']
        if cls_id < 0:
            continue
        x, y, _ = obj['center_sensor']
        if math.sqrt(x * x + y * y) > DETECTION_RANGE.get(cls_id, 50.0):
            continue
        cx_norm, cy_norm = metres_to_norm(x, y, pc_range)
        if not (0.0 <= cx_norm <= 1.0 and 0.0 <= cy_norm <= 1.0):
            continue
        out.append(obj.get('class', '?'))
    return out


# ---------------------------------------------------------------------------
# IoU helpers  (axis-aligned BEV, fast vectorized)
# ---------------------------------------------------------------------------

def _bev_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Axis-aligned BEV IoU between proposals and GT.

    Both inputs: (N, 4) in [cx, cy, w, l] normalized coords.
    Returns: (N, M) IoU matrix.
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    # Convert [cx, cy, w, l] → [x1, y1, x2, y2]
    # Note: w = perpendicular (y-dir), l = along heading (x-dir) for GT
    # For axis-aligned IoU, dimension assignment doesn't matter as long as
    # proposals and GT use the same convention consistently.
    a_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    a_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    a_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    a_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    b_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
    b_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    b_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
    b_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    # Vectorized pairwise IoU
    # (N, 1) vs (1, M) broadcasting
    xx1 = np.maximum(a_x1[:, None], b_x1[None, :])
    yy1 = np.maximum(a_y1[:, None], b_y1[None, :])
    xx2 = np.minimum(a_x2[:, None], b_x2[None, :])
    yy2 = np.minimum(a_y2[:, None], b_y2[None, :])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)  # (N,)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)  # (M,)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-8

    return (inter / union).astype(np.float32)


def _center_dist_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Euclidean center distance between (N, 4) and (M, 4) boxes."""
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)
    dx = pred_boxes[:, 0:1] - gt_boxes[:, 0:1].T  # (N, M)
    dy = pred_boxes[:, 1:2] - gt_boxes[:, 1:2].T
    return np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Hungarian matching (mirrors BEVHungarianMatcherV2 cost structure)
# ---------------------------------------------------------------------------

def hungarian_match(
    prop_boxes: np.ndarray,    # (Q, 4) [cx, cy, w, l] normalized
    gt_boxes: np.ndarray,      # (G, 4) [cx, cy, w, l] normalized
    iou_threshold: float = 0.0,
):
    """Run Hungarian matching and return per-GT best IoU.

    Cost = center_dist * 5.0 + L1_box * 5.0 - IoU * 2.0

    Returns:
        matched_ious: (G,) IoU of each GT with its matched proposal (0 if unmatched)
        n_matched_at_iou: dict {iou_thresh: count} for IoU thresholds [0.1, 0.2, 0.3, 0.5]
    """
    Q, G = prop_boxes.shape[0], gt_boxes.shape[0]
    if G == 0:
        return np.array([], dtype=np.float32), {}
    if Q == 0:
        return np.zeros(G, dtype=np.float32), {}

    # Compute cost components
    iou_mat = _bev_iou_matrix(prop_boxes, gt_boxes)        # (Q, G)
    center_dist = _center_dist_matrix(prop_boxes, gt_boxes) # (Q, G)

    # L1 on all 4 box dims
    l1_box = np.abs(prop_boxes[:, None, :] - gt_boxes[None, :, :]).sum(axis=2)  # (Q, G)

    cost = center_dist * 5.0 + l1_box * 5.0 - iou_mat * 2.0

    # Optional IoU threshold: set infinite cost for pairs below threshold
    if iou_threshold > 0:
        cost[iou_mat < iou_threshold] = 1e6

    # Solve assignment
    row_idx, col_idx = linear_sum_assignment(cost)

    matched_ious = np.zeros(G, dtype=np.float32)
    for r, c in zip(row_idx, col_idx):
        matched_ious[c] = iou_mat[r, c]

    return matched_ious


# ---------------------------------------------------------------------------
# Proposal box conversion (normalized 8D → 4D BEV [cx, cy, w, l])
# ---------------------------------------------------------------------------

def proposals_to_bev4(raw_proposals: torch.Tensor) -> np.ndarray:
    """Convert proposals to [cx, cy, w, l] in normalized space.

    Works for both 8D CenterPoint proposals (first 4 dims used) and
    4D PointPillars proposals (returned as-is).
    """
    return raw_proposals[:, :4].numpy()


_fallback_warned = False


def load_cached_payload(dataset, entry: dict, proposal_key: str):
    """Load a cache payload directly, supporting any proposal_key.

    Bypasses _load_cached_features (which hardcodes 'proposal_boxes_8d').
    Falls back to 'proposal_boxes_8d' if the requested key is absent
    (PointPillars extractor saves 4-D proposals under that name).
    Returns None if cache miss.
    """
    global _fallback_warned
    rec = dataset._resolve_feature_cache_record(entry)
    if rec is None:
        return None
    payload = torch.load(rec['feature_file'], map_location='cpu', weights_only=False)
    srcs = payload.get('feature_maps')
    props = payload.get(proposal_key)
    # Fallback: PointPillars extractor stores 4-D proposals under 'proposal_boxes_8d'
    if props is None and proposal_key != 'proposal_boxes_8d':
        props = payload.get('proposal_boxes_8d')
        if props is not None and not _fallback_warned:
            print(f"[analyze] WARNING: key '{proposal_key}' not in payload; "
                  f"falling back to 'proposal_boxes_8d' (shape {tuple(props.shape)})",
                  flush=True)
            _fallback_warned = True
    scores = payload.get('proposal_scores')
    if srcs is None or props is None or scores is None:
        raise RuntimeError(
            f"Invalid cached feature payload (key={proposal_key!r}): {rec['feature_file']}")
    if not isinstance(srcs, list):
        raise RuntimeError(
            f"feature_maps must be a list in cached payload: {rec['feature_file']}")
    return {
        'srcs': [s.float().contiguous() for s in srcs],
        'props': props.float().contiguous(),
        'scores': scores.float().contiguous(),
        'record': rec,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze proposal score thresholds')
    parser.add_argument('--feature-cache-dir', default='/data/feature_maps_predictions')
    parser.add_argument('--proposal-key', default='proposal_boxes_8d',
                        help='Payload key for proposals: proposal_boxes_8d (CenterPoint) '
                             'or proposal_boxes_4d (PointPillars).')
    parser.add_argument('--refer-data-dir', default='/data/nuscenes/refer_detection_with_negatives')
    parser.add_argument('--data-root', default='/data/nuscenes')
    parser.add_argument('--ann-file', default=None,
                        help='Override NuScenes ann file path.')
    parser.add_argument('--sweeps-num', type=int, default=10)
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit frames to process (for quick testing)')
    parser.add_argument('--iou-thresholds', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.5],
                        help='IoU thresholds to report recall at.')
    parser.add_argument('--topk-list', type=int, nargs='+',
                        default=[10, 25, 50, 75, 100, 125, 150],
                        help='Top-K-by-rank sweep (query-count ablation ceiling). '
                             'Padded/degenerate cache slots are excluded, so K is '
                             'capped by the real proposals available per frame.')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to this file (CSV).')
    args = parser.parse_args()

    if args.ann_file is None:
        if args.split == 'train':
            args.ann_file = os.path.join(args.data_root, 'nuscenes_infos_train.pkl')
        else:
            args.ann_file = os.path.join(args.data_root, 'nuscenes_infos_val.pkl')

    # Score thresholds to sweep
    score_thresholds = np.arange(0.0, 0.55, 0.025).tolist()

    # ── 1. Load dataset ──
    print(f'Loading {args.split} dataset...')
    ds_args = SimpleNamespace(
        refer_data_dir=args.refer_data_dir,
        nuscenes_dataroot=args.data_root,
        nuscenes_ann_file=args.ann_file,
        sweeps_num=args.sweeps_num,
        point_cloud_range=args.point_cloud_range,
        question_types_json=None,
        question_types=None,
        queries_per_frame=1,
        precompute_bev=False,
        num_point_features=5,
        voxel_size=[0.08, 0.08, 4.0],
        lidar_root=None,
        backend_args=None,
        feature_cache_dir=args.feature_cache_dir,
        feature_cache_strict=False,  # don't crash on cache misses
    )
    dataset = build_refer_dataset(args.split, ds_args)
    total_frames = len(dataset.frames)
    print(f'  Total frames: {total_frames}')
    print(f'  Frames with queries (valid_indices): {len(dataset.valid_indices)}')

    # We iterate ALL frames (not just valid_indices) since we want to match
    # proposals against all GT objects regardless of referring queries.
    frame_indices = list(range(total_frames))
    if args.max_frames is not None:
        frame_indices = frame_indices[:args.max_frames]
        print(f'  Limiting to {args.max_frames} frames')

    # ── 2. Per-threshold accumulators ──
    # For each score threshold, accumulate:
    #   - total GT objects
    #   - matched GT per IoU threshold (recall numerator)
    #   - total proposals kept
    #   - per-class GT and recalls
    stats = {}
    for st in score_thresholds:
        stats[st] = {
            'n_frames': 0,
            'n_gt_total': 0,
            'n_proposals_kept_total': 0,
            'matched_at_iou': {iou_t: 0 for iou_t in args.iou_thresholds},
            'per_class_gt': defaultdict(int),
            'per_class_matched': {iou_t: defaultdict(int) for iou_t in args.iou_thresholds},
        }

    # ── 2b. Per-top-K accumulators (query-count ablation ceiling) ──
    topk_list = sorted(set(args.topk_list))
    stats_k = {}
    for k in topk_list:
        stats_k[k] = {
            'n_frames': 0,
            'n_gt_total': 0,
            'n_proposals_kept_total': 0,
            'matched_at_iou': {iou_t: 0 for iou_t in args.iou_thresholds},
            'per_class_gt': defaultdict(int),
            'per_class_matched': {iou_t: defaultdict(int) for iou_t in args.iou_thresholds},
        }

    n_skipped = 0
    n_no_gt = 0
    t0 = time.time()

    # ── 3. Iterate frames ──
    for fi, flat_idx in enumerate(frame_indices):
        if (fi + 1) % 500 == 0 or fi == 0:
            elapsed = time.time() - t0
            fps = (fi + 1) / max(elapsed, 1e-3)
            print(f'  [{fi+1}/{len(frame_indices)}] {fps:.1f} frames/s ...', flush=True)

        entry = dataset.frames[flat_idx]

        # Load cached proposals
        cached = load_cached_payload(dataset, entry, args.proposal_key)
        if cached is None:
            n_skipped += 1
            continue

        raw_proposals = cached['props']   # (Q, 8)
        scores = cached['scores']         # (Q,)

        # Build GT for this frame — use empty ref_tokens to get all objects
        targets = dataset._build_targets(flat_idx, set())
        n_gt = len(targets['labels'])
        if n_gt == 0:
            n_no_gt += 1
            continue

        gt_boxes = targets['boxes'].numpy()  # (G, 4) [cx, cy, w, l]

        # Per-GT class names, positionally aligned with _build_targets
        # (fixes the float32 obj_id bug that put ~51% of GT in class '?')
        gt_classes = gt_classes_for_frame(dataset, flat_idx)
        assert len(gt_classes) == n_gt, \
            f'class/target misalignment: {len(gt_classes)} vs {n_gt}'

        # Proposals in BEV
        prop_bev = proposals_to_bev4(raw_proposals)  # (Q, 4)
        scores_np = scores.numpy()

        # ── Sweep thresholds ──
        for st in score_thresholds:
            mask = scores_np >= st
            n_kept = int(mask.sum())

            s = stats[st]
            s['n_frames'] += 1
            s['n_gt_total'] += n_gt
            s['n_proposals_kept_total'] += n_kept

            for cls in gt_classes:
                s['per_class_gt'][cls] += 1

            if n_kept == 0:
                # No proposals pass — everything is unmatched
                continue

            filtered_props = prop_bev[mask]

            # Run Hungarian matching
            matched_ious = hungarian_match(filtered_props, gt_boxes)

            # Count recalls at different IoU thresholds
            for iou_t in args.iou_thresholds:
                matched_mask = matched_ious >= iou_t
                n_matched = int(matched_mask.sum())
                s['matched_at_iou'][iou_t] += n_matched

                # Per-class
                for gi in range(n_gt):
                    if matched_mask[gi]:
                        s['per_class_matched'][iou_t][gt_classes[gi]] += 1

        # ── Sweep top-K by rank (query-count ablation ceiling) ──
        # Exclude padded slots (score==0, zero-area boxes) so K counts only
        # REAL proposals; rank by detector confidence.
        real = (scores_np > 0.0) & (prop_bev[:, 2] * prop_bev[:, 3] > 0.0)
        order = np.argsort(-scores_np[real])
        ranked_props = prop_bev[real][order]
        for k in topk_list:
            sk = stats_k[k]
            sk['n_frames'] += 1
            sk['n_gt_total'] += n_gt
            kept = ranked_props[:k]
            sk['n_proposals_kept_total'] += kept.shape[0]
            for cls in gt_classes:
                sk['per_class_gt'][cls] += 1
            if kept.shape[0] == 0:
                continue
            matched_ious = hungarian_match(kept, gt_boxes)
            for iou_t in args.iou_thresholds:
                matched_mask = matched_ious >= iou_t
                sk['matched_at_iou'][iou_t] += int(matched_mask.sum())
                for gi in range(n_gt):
                    if matched_mask[gi]:
                        sk['per_class_matched'][iou_t][gt_classes[gi]] += 1

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s. Skipped {n_skipped} frames (no cache), '
          f'{n_no_gt} frames (no GT).')

    # ── 4. Report ──
    print('\n' + '=' * 100)
    print(' PROPOSAL SCORE THRESHOLD ANALYSIS')
    print(f' Split: {args.split} | Frames processed: {stats[score_thresholds[0]]["n_frames"]}')
    print('=' * 100)

    # Header
    iou_cols = ''.join(f' | Recall@{t:.1f}' for t in args.iou_thresholds)
    header = f'{"Thresh":>8} | {"Avg Props":>10} | {"Total GT":>10}{iou_cols}'
    print(header)
    print('-' * len(header))

    rows = []
    for st in score_thresholds:
        s = stats[st]
        n_frames = max(s['n_frames'], 1)
        avg_props = s['n_proposals_kept_total'] / n_frames
        n_gt = s['n_gt_total']

        cols = [f'{st:8.3f}', f'{avg_props:10.1f}', f'{n_gt:10d}']
        for iou_t in args.iou_thresholds:
            n_matched = s['matched_at_iou'][iou_t]
            recall = n_matched / max(n_gt, 1) * 100
            cols.append(f'{recall:9.2f}%')
        row_str = ' | '.join(cols)
        print(row_str)
        rows.append([st, avg_props, n_gt] +
                    [s['matched_at_iou'][t] / max(s['n_gt_total'], 1) * 100
                     for t in args.iou_thresholds])

    # ── 5. Per-class breakdown at a few key thresholds ──
    key_score_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    key_score_thresholds = [t for t in key_score_thresholds if t in [round(x, 3) for x in score_thresholds]]

    for st in key_score_thresholds:
        # Find closest threshold
        st_key = min(score_thresholds, key=lambda x: abs(x - st))
        s = stats[st_key]
        if s['n_gt_total'] == 0:
            continue

        print(f'\n{"─" * 80}')
        print(f'  Per-class recall at score threshold = {st_key:.3f} '
              f'(avg {s["n_proposals_kept_total"] / max(s["n_frames"], 1):.0f} proposals/frame)')
        print(f'{"─" * 80}')

        all_classes = sorted(s['per_class_gt'].keys())
        cls_header = f'  {"Class":>20} | {"GT Count":>10}'
        for iou_t in args.iou_thresholds:
            cls_header += f' | {"R@" + str(iou_t):>10}'
        print(cls_header)
        print('  ' + '-' * (len(cls_header) - 2))

        for cls in all_classes:
            gt_count = s['per_class_gt'][cls]
            parts = [f'  {cls:>20}', f'{gt_count:10d}']
            for iou_t in args.iou_thresholds:
                matched = s['per_class_matched'][iou_t].get(cls, 0)
                recall = matched / max(gt_count, 1) * 100
                parts.append(f'{recall:9.1f}%')
            print(' | '.join(parts))

    # ── 6. Top-K-by-rank sweep (query-count ablation ceiling) ──
    print(f'\n{"=" * 100}')
    print(' TOP-K BY RANK — proposal-recall ceiling per query count')
    print(' (use this to pick the query-count ablation values; trained refer-mAP')
    print('  at each Q sits underneath the corresponding ceiling)')
    print(f'{"=" * 100}')
    header_k = f'{"K":>6} | {"Avg Props":>10} | {"Total GT":>10}{iou_cols}'
    print(header_k)
    print('-' * len(header_k))
    rows_k = []
    for k in topk_list:
        sk = stats_k[k]
        n_frames = max(sk['n_frames'], 1)
        avg_props = sk['n_proposals_kept_total'] / n_frames
        n_gt = sk['n_gt_total']
        cols = [f'{k:6d}', f'{avg_props:10.1f}', f'{n_gt:10d}']
        for iou_t in args.iou_thresholds:
            recall = sk['matched_at_iou'][iou_t] / max(n_gt, 1) * 100
            cols.append(f'{recall:9.2f}%')
        print(' | '.join(cols))
        rows_k.append([k, avg_props, n_gt] +
                      [sk['matched_at_iou'][t] / max(n_gt, 1) * 100
                       for t in args.iou_thresholds])

    # Per-class breakdown at each K (the small-object story lives here)
    for k in topk_list:
        sk = stats_k[k]
        if sk['n_gt_total'] == 0:
            continue
        print(f'\n{"─" * 80}')
        print(f'  Per-class recall at top-K = {k}')
        print(f'{"─" * 80}')
        all_classes = sorted(sk['per_class_gt'].keys())
        cls_header = f'  {"Class":>20} | {"GT Count":>10}'
        for iou_t in args.iou_thresholds:
            cls_header += f' | {"R@" + str(iou_t):>10}'
        print(cls_header)
        print('  ' + '-' * (len(cls_header) - 2))
        for cls in all_classes:
            gt_count = sk['per_class_gt'][cls]
            parts = [f'  {cls:>20}', f'{gt_count:10d}']
            for iou_t in args.iou_thresholds:
                matched = sk['per_class_matched'][iou_t].get(cls, 0)
                parts.append(f'{matched / max(gt_count, 1) * 100:9.1f}%')
            print(' | '.join(parts))

    # ── 7. Recommendation (relative to the achievable ceiling) ──
    # The old block asked for an absolute 95% recall that no setting reaches
    # and silently fell back to the floor; recommendations are now expressed
    # as a fraction of the K_max ceiling.
    print(f'\n{"=" * 100}')
    print(' RECOMMENDATION — smallest K reaching a fraction of the K_max ceiling')
    print(f'{"=" * 100}')
    target_iou = args.iou_thresholds[0]  # typically 0.1
    k_max = topk_list[-1]
    ceil_gt = max(stats_k[k_max]['n_gt_total'], 1)
    ceiling = stats_k[k_max]['matched_at_iou'][target_iou] / ceil_gt * 100
    print(f'  Ceiling: Recall@{target_iou:.1f} = {ceiling:.2f}% at K = {k_max}\n')
    for frac in [0.99, 0.975, 0.95, 0.90]:
        target = ceiling * frac
        pick = None
        for k in topk_list:
            sk = stats_k[k]
            r = sk['matched_at_iou'][target_iou] / max(sk['n_gt_total'], 1) * 100
            if r >= target:
                pick = (k, r)
                break
        if pick:
            print(f'  >= {frac * 100:5.1f}% of ceiling ({target:5.2f}%):  '
                  f'K = {pick[0]:4d}  (Recall@{target_iou:.1f} = {pick[1]:.2f}%)')

    # ── 8. Optional CSV output (threshold sweep + top-K sweep) ──
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        iou_headers = ','.join(f'recall@{t:.1f}' for t in args.iou_thresholds)
        with open(args.output, 'w') as f:
            f.write(f'score_threshold,avg_proposals,total_gt,{iou_headers}\n')
            for row in rows:
                f.write(','.join(str(round(v, 4)) for v in row) + '\n')
        topk_path = (os.path.splitext(args.output)[0] + '_topk.csv')
        with open(topk_path, 'w') as f:
            f.write(f'top_k,avg_proposals,total_gt,{iou_headers}\n')
            for row in rows_k:
                f.write(','.join(str(round(v, 4)) for v in row) + '\n')
        print(f'\nCSV saved to {args.output} and {topk_path}')


if __name__ == '__main__':
    main()
