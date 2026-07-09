#!/usr/bin/env python3
"""Analyze CenterPoint proposal noise in 3D attributes (z, h, yaw) vs GT.

Computes per-class statistics of how well CenterPoint proposals match GT
in the BEV dimensions (cx, cy, w, l) vs the 3D dimensions (z, h, yaw).

This answers the question: does including z/h/yaw in query_pos_mlp hurt
small objects like pedestrians because CenterPoint produces noisy estimates?

Usage:
    python analyze_proposal_3d_noise.py \
        --feature-cache-dir /data/feature_maps_predictions \
        --refer-data-dir /data/nuscenes/refer_detection_with_negatives \
        --data-root /data/nuscenes \
        --n-frames 200 \
        --output /data/outputs/proposal_3d_noise_analysis.txt
"""
import argparse
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet3d.utils import register_all_modules
register_all_modules(init_default_scope=True)

from nuscenes_lidar_simple import build as build_refer_dataset


def _proposals_to_metres(raw_proposals, scores, pc_range):
    """Convert normalized proposals to metres. Returns (N, 8) array."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = max(float(z_max - z_min), 1e-6)

    n = raw_proposals.shape[0]
    dim = raw_proposals.shape[1]
    out = np.zeros((n, 8), dtype=np.float32)

    out[:, 0] = raw_proposals[:, 0] * x_span + x_min   # cx (m)
    out[:, 1] = raw_proposals[:, 1] * y_span + y_min   # cy (m)
    # Bridge convention (proposal_w_from='dy'):
    #   index 2 = w_norm (perpendicular), index 3 = l_norm (along heading)
    out[:, 2] = raw_proposals[:, 2] * y_span  # width (m)
    out[:, 3] = raw_proposals[:, 3] * x_span  # length (m)
    out[:, 4] = raw_proposals[:, 4] * z_span + z_min   # z (m)
    out[:, 5] = raw_proposals[:, 5] * 10.0              # h (m)

    if dim >= 8:
        dim6 = raw_proposals[:, 6]
        dim7 = raw_proposals[:, 7]
        unique_d7 = np.unique(dim7)
        if len(unique_d7) <= 2 and all(v in (0.0, 1.0) for v in unique_d7):
            yaw_half = dim6 * math.pi - (math.pi / 2.0)
            out[:, 6] = yaw_half + dim7 * math.pi
        else:
            out[:, 6] = np.arctan2(dim6, dim7)
    elif dim == 7:
        out[:, 6] = raw_proposals[:, 6] * math.pi - (math.pi / 2.0)

    out[:, 7] = scores
    return out


def _gt_to_metres(targets, frame_objects, pc_range):
    """Convert GT targets to metres. Returns list of dicts with class info."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    x_span = x_max - x_min
    y_span = y_max - y_min

    obj_id_to_class = {}
    for obj in frame_objects:
        obj_id_to_class[float(obj['obj_id'])] = obj.get('class', '?')

    gt_list = []
    n_gt = len(targets['labels'])
    for i in range(n_gt):
        bev = targets['boxes'][i].numpy()
        a3d = targets['attrs_3d'][i].numpy()
        obj_id = targets['obj_ids'][i].item()

        gt_list.append({
            'cx': bev[0] * x_span + x_min,
            'cy': bev[1] * y_span + y_min,
            'w': bev[2] * x_span,     # GT: w_norm = width / x_span
            'l': bev[3] * y_span,     # GT: l_norm = length / y_span
            'z': a3d[0],              # z in metres (already absolute from attrs_3d)
            'h': a3d[1],              # h in metres
            'yaw': math.atan2(a3d[2], a3d[3]),
            'class': obj_id_to_class.get(obj_id, '?'),
        })
    return gt_list


def _match_proposals_to_gt(proposals_m, gt_list, iou_thresh=0.1):
    """Greedy BEV center-distance matching of proposals to GT.
    Returns list of (gt_idx, prop_idx, center_dist) tuples."""
    if len(gt_list) == 0 or len(proposals_m) == 0:
        return []

    # Compute center distances
    gt_centers = np.array([[g['cx'], g['cy']] for g in gt_list])
    prop_centers = proposals_m[:, :2]

    # For each GT, find closest proposal
    matches = []
    used_props = set()
    for gi in range(len(gt_list)):
        dists = np.sqrt(((prop_centers - gt_centers[gi]) ** 2).sum(axis=1))
        # Also check rough BEV overlap: proposal center must be within
        # a reasonable distance (e.g., max of GT dimensions)
        gt_size = max(gt_list[gi]['w'], gt_list[gi]['l'], 1.0)
        max_dist = gt_size * 1.5

        order = np.argsort(dists)
        for pi in order:
            if pi in used_props:
                continue
            if dists[pi] > max_dist:
                break
            matches.append((gi, int(pi), float(dists[pi])))
            used_props.add(pi)
            break

    return matches


def _angular_diff(a, b):
    """Signed angular difference, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser(description='Analyze CenterPoint 3D proposal noise vs GT')
    parser.add_argument('--feature-cache-dir', default='/data/feature_maps_predictions')
    parser.add_argument('--refer-data-dir', default='/data/nuscenes/refer_detection_with_negatives')
    parser.add_argument('--data-root', default='/data/nuscenes')
    parser.add_argument('--ann-file', default='nuscenes_infos_val.pkl')
    parser.add_argument('--sweeps-num', type=int, default=10)
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    parser.add_argument('--n-frames', type=int, default=500,
                        help='Number of frames to analyze (0 = all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--use-pp-head', action='store_true',
                        help='Run the PointPillars head on cached feature maps to get FULL 3D boxes '
                             '(z/h/yaw) instead of reading the 4D cached proposals (which have no yaw).')
    parser.add_argument('--pointpillars-config',
                        default='/workspace/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py')
    parser.add_argument('--pointpillars-ckpt',
                        default='/data/ckpts/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth')
    parser.add_argument('--score-thr', type=float, default=0.1)
    args = parser.parse_args()
    args.ann_file = os.path.join(args.data_root, args.ann_file)

    random.seed(args.seed)
    np.random.seed(args.seed)
    pc_range = args.point_cloud_range

    # Load dataset
    print('Loading dataset...')
    ds_args = SimpleNamespace(
        refer_data_dir=args.refer_data_dir,
        nuscenes_dataroot=args.data_root,
        nuscenes_ann_file=args.ann_file,
        sweeps_num=args.sweeps_num,
        point_cloud_range=pc_range,
        question_types_json=None,
        question_types=None,
        queries_per_frame=1,
        precompute_bev=False,
        num_point_features=5,
        voxel_size=[0.08, 0.08, 4.0],
        lidar_root=None,
        backend_args=None,
        feature_cache_dir=args.feature_cache_dir,
        feature_cache_strict=True,
    )
    dataset = build_refer_dataset(args.split, ds_args)
    n_total = len(dataset.valid_indices)
    print(f'  Total frames with queries ({args.split}): {n_total}')

    # Optionally build the PointPillars head to recover FULL 3D boxes from cached feats.
    pp_head = None
    if args.use_pp_head:
        from models.refer_model_lang_dec import PointPillarsDetectorBridge
        from mmdet3d.structures import LiDARInstance3DBoxes
        pp_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'  [use-pp-head] building PointPillars on {pp_device} ...')
        _bridge = PointPillarsDetectorBridge(
            config_path=args.pointpillars_config, checkpoint_path=args.pointpillars_ckpt,
            point_cloud_range=pc_range, proposal_count=150)
        _bridge.to(pp_device).eval()
        pp_head = _bridge.detector.pts_bbox_head
        pp_metas = [{'box_type_3d': LiDARInstance3DBoxes}]

    # Sample frames
    frame_indices = list(range(n_total))
    if args.n_frames > 0 and args.n_frames < n_total:
        frame_indices = random.sample(frame_indices, args.n_frames)
    n_frames = len(frame_indices)
    print(f'  Analyzing {n_frames} frames...')

    # Per-class error accumulators
    # For each matched GT-proposal pair, store errors in each dimension
    class_errors = defaultdict(lambda: {
        'cx_err': [], 'cy_err': [],    # BEV center errors (m)
        'w_err': [], 'l_err': [],      # BEV size errors (m)
        'z_err': [], 'h_err': [],      # 3D attribute errors (m)
        'yaw_err': [],                 # yaw error (rad)
        'gt_w': [], 'gt_l': [],        # GT sizes for context
        'gt_h': [],
        'score': [],                   # proposal confidence
        'center_dist': [],             # matching distance
        'n_matched': 0,
        'n_gt': 0,
    })

    for fi, frame_i in enumerate(frame_indices):
        if fi % 100 == 0:
            print(f'  Processing frame {fi}/{n_frames}...')

        flat_idx = dataset.valid_indices[frame_i]
        entry = dataset.frames[flat_idx]
        frame_objects = dataset.frame_objects[flat_idx]

        # Load cached proposals
        cached = dataset._load_cached_features(entry)
        if cached is None:
            continue

        if pp_head is not None:
            # Run the PointPillars head on cached FPN feats -> full 3D detections.
            feats = [s.unsqueeze(0).to(pp_device) if s.dim() == 3 else s.to(pp_device)
                     for s in cached['srcs']]
            with torch.no_grad():
                cs, bp, dp = pp_head(feats)
                res = pp_head.predict_by_feat(cs, bp, dp, batch_input_metas=pp_metas,
                                              cfg=pp_head.test_cfg, rescale=False)
            det = res[0]
            pb = det.bboxes_3d.tensor.detach().cpu().numpy()   # (N, 9): x,y,z,dx,dy,dz,yaw,...
            ps = det.scores_3d.detach().cpu().numpy()
            keep = ps >= args.score_thr
            pb, ps = pb[keep], ps[keep]
            if pb.shape[0] == 0:
                for gt in _gt_to_metres(dataset._build_targets(flat_idx, set()), frame_objects, pc_range):
                    class_errors[gt['class']]['n_gt'] += 1
                continue
            proposals_m = np.zeros((pb.shape[0], 8), dtype=np.float32)
            proposals_m[:, 0] = pb[:, 0]; proposals_m[:, 1] = pb[:, 1]   # cx, cy
            proposals_m[:, 2] = pb[:, 4]; proposals_m[:, 3] = pb[:, 3]   # w=dy, l=dx
            proposals_m[:, 4] = pb[:, 2]; proposals_m[:, 5] = pb[:, 5]   # z, h=dz
            proposals_m[:, 6] = pb[:, 6]; proposals_m[:, 7] = ps         # yaw, score
        else:
            raw_proposals = cached['props'].numpy()
            scores = cached['scores'].numpy()
            proposals_m = _proposals_to_metres(raw_proposals, scores, pc_range)

        # Build GT for all objects in this frame (use empty ref_tokens to get all)
        ref_tokens = set()
        targets = dataset._build_targets(flat_idx, ref_tokens)
        gt_list = _gt_to_metres(targets, frame_objects, pc_range)

        if len(gt_list) == 0:
            continue

        # Match proposals to GT
        matches = _match_proposals_to_gt(proposals_m, gt_list)

        for gi, pi, cdist in matches:
            gt = gt_list[gi]
            prop = proposals_m[pi]
            cls = gt['class']

            errs = class_errors[cls]
            errs['n_matched'] += 1
            errs['cx_err'].append(abs(prop[0] - gt['cx']))
            errs['cy_err'].append(abs(prop[1] - gt['cy']))
            errs['w_err'].append(abs(prop[2] - gt['w']))
            errs['l_err'].append(abs(prop[3] - gt['l']))
            errs['z_err'].append(abs(prop[4] - gt['z']))
            errs['h_err'].append(abs(prop[5] - gt['h']))
            errs['yaw_err'].append(abs(_angular_diff(prop[6], gt['yaw'])))
            errs['gt_w'].append(gt['w'])
            errs['gt_l'].append(gt['l'])
            errs['gt_h'].append(gt['h'])
            errs['score'].append(prop[7])
            errs['center_dist'].append(cdist)

        for gt in gt_list:
            class_errors[gt['class']]['n_gt'] += 1

    # Print analysis
    lines = []
    def p(s=''):
        lines.append(s)
        print(s)

    p()
    p('=' * 100)
    p(f' {"POINTPILLARS (head)" if args.use_pp_head else "CACHED-PROPOSAL"} 3D NOISE ANALYSIS')
    p(f' {n_frames} frames analyzed ({args.split} split)')
    p('=' * 100)
    p()

    # Sort classes by typical size (smallest first)
    size_order = sorted(
        class_errors.keys(),
        key=lambda c: np.median(class_errors[c]['gt_l']) if class_errors[c]['gt_l'] else 999
    )

    p(f'{"Class":>30s} | {"N_GT":>6s} {"Match":>6s} {"Rate":>6s} | '
      f'{"cx_err":>7s} {"cy_err":>7s} {"w_err":>7s} {"l_err":>7s} | '
      f'{"z_err":>7s} {"h_err":>7s} {"yaw_err":>8s} | '
      f'{"med_W":>6s} {"med_L":>6s} {"med_H":>6s} | '
      f'{"z/L":>6s} {"h/H":>6s} {"yaw°":>6s}')
    p('-' * 160)

    for cls in size_order:
        errs = class_errors[cls]
        n_gt = errs['n_gt']
        n_m = errs['n_matched']
        if n_m == 0:
            p(f'{cls:>30s} | {n_gt:6d} {n_m:6d} {"0.0%":>6s} | (no matches)')
            continue

        cx_e = np.median(errs['cx_err'])
        cy_e = np.median(errs['cy_err'])
        w_e = np.median(errs['w_err'])
        l_e = np.median(errs['l_err'])
        z_e = np.median(errs['z_err'])
        h_e = np.median(errs['h_err'])
        yaw_e = np.median(errs['yaw_err'])

        med_w = np.median(errs['gt_w'])
        med_l = np.median(errs['gt_l'])
        med_h = np.median(errs['gt_h'])

        # Relative errors: z_err / length, h_err / height, yaw in degrees
        z_rel = z_e / max(med_l, 0.01)
        h_rel = h_e / max(med_h, 0.01)
        yaw_deg = math.degrees(yaw_e)

        rate = f'{100*n_m/n_gt:.1f}%' if n_gt > 0 else 'N/A'

        p(f'{cls:>30s} | {n_gt:6d} {n_m:6d} {rate:>6s} | '
          f'{cx_e:7.3f} {cy_e:7.3f} {w_e:7.3f} {l_e:7.3f} | '
          f'{z_e:7.3f} {h_e:7.3f} {yaw_e:8.4f} | '
          f'{med_w:6.2f} {med_l:6.2f} {med_h:6.2f} | '
          f'{z_rel:6.3f} {h_rel:6.3f} {yaw_deg:6.1f}')

    p()
    p('Legend:')
    p('  cx/cy/w/l_err: median absolute error in BEV attributes (metres)')
    p('  z/h_err: median absolute error in 3D attributes (metres)')
    p('  yaw_err: median absolute yaw error (radians)')
    p('  z/L: z_err relative to object length')
    p('  h/H: h_err relative to object height')
    p('  yaw°: median yaw error in degrees')
    p()

    # Summary: aggregate small vs large
    p('=' * 100)
    p(' AGGREGATE: SMALL OBJECTS vs LARGE OBJECTS')
    p('=' * 100)

    small_classes = {'pedestrian', 'traffic_cone', 'bicycle', 'motorcycle'}
    large_classes = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle'}

    for group_name, group_set in [('SMALL (ped/cone/bike/moto)', small_classes),
                                   ('LARGE (car/truck/bus/trailer/constr)', large_classes)]:
        all_z = []
        all_h = []
        all_yaw = []
        all_cx = []
        all_cy = []
        all_w = []
        all_l = []
        n_m_total = 0
        n_gt_total = 0
        for cls in class_errors:
            if cls.lower().replace(' ', '_') in group_set or cls.lower() in group_set:
                e = class_errors[cls]
                all_z.extend(e['z_err'])
                all_h.extend(e['h_err'])
                all_yaw.extend(e['yaw_err'])
                all_cx.extend(e['cx_err'])
                all_cy.extend(e['cy_err'])
                all_w.extend(e['w_err'])
                all_l.extend(e['l_err'])
                n_m_total += e['n_matched']
                n_gt_total += e['n_gt']

        if len(all_z) == 0:
            p(f'  {group_name}: no matches')
            continue

        p(f'  {group_name}:  n_gt={n_gt_total}  matched={n_m_total}')
        p(f'    BEV errors (median): cx={np.median(all_cx):.3f}m  cy={np.median(all_cy):.3f}m  '
          f'w={np.median(all_w):.3f}m  l={np.median(all_l):.3f}m')
        p(f'    3D errors  (median): z={np.median(all_z):.3f}m  h={np.median(all_h):.3f}m  '
          f'yaw={math.degrees(np.median(all_yaw)):.1f}°')
        p(f'    3D errors  (p90):    z={np.percentile(all_z, 90):.3f}m  '
          f'h={np.percentile(all_h, 90):.3f}m  yaw={math.degrees(np.percentile(all_yaw, 90)):.1f}°')
        p()

    # Key question: is the noise in z/h/yaw significantly worse for small objects?
    p('=' * 100)
    p(' VERDICT: Should query_pos use 4D (BEV only) or 8D (BEV + 3D)?')
    p('=' * 100)
    p()
    p('  If 3D errors are large relative to object size for small objects,')
    p('  including z/h/yaw in query_pos will inject noise that confuses the')
    p('  position encoding, especially for pedestrians and traffic cones.')
    p()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines))
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
