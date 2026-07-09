#!/usr/bin/env python3
"""Extract PointPillars feature maps + top-K proposals for all samples in a split.

Runs the frozen PointPillars detector once over a split and caches, per frame,
the FPN feature maps and the top-K proposals that seed the referring decoder.
Training and evaluation read this cache instead of re-running the detector.

Payload saved per sample:
  feature_maps        : list of 3 float16/float32 tensors (one per FPN level)
  proposal_boxes_8d   : float32 tensor (K, 4) — normalized [cx, cy, w, l]
  proposal_scores     : float32 tensor (K,)
  proposal_yaw        : float32 tensor (K,) — per-proposal heading in radians
                        (used by the orientation-refinement +angle model)
"""
import argparse
import json
import os
import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nuscenes_lidar_simple import build as build_refer_dataset  # noqa: E402
from models.refer_model_lang_dec import PointPillarsDetectorBridge  # noqa: E402


def _cache_key(sample_token: str, lidar_path: str) -> str:
    raw = f"{sample_token}|{lidar_path}"
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract per-sample PointPillars feature maps and top-K proposals for caching.'
    )
    parser.add_argument('--nuscenes-dataroot', required=True)
    parser.add_argument('--refer-data-dir', required=True)
    parser.add_argument('--pointpillars-config', required=True)
    parser.add_argument('--pointpillars-ckpt', required=True)
    parser.add_argument('--nuscenes-ann-file', type=str, default=None)
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test', 'all'],
        default='train',
        help='Dataset split to extract; all means train+val.',
    )
    parser.add_argument('--question-types-json', type=str, default='configs/question_types_det.json')
    parser.add_argument('--sweeps-num', type=int, default=9, help='Past sweeps count (9 => 10 total frames)')
    parser.add_argument('--proposal-queries', type=int, default=150)
    parser.add_argument('--proposal-w-from', choices=['dx', 'dy'], default='dy')
    parser.add_argument('--point-cloud-range', type=float, nargs=6, default=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
    parser.add_argument(
        '--num-point-features', type=int, default=4,
        help='4 = [x, y, z, dt] (stock nuScenes PointPillars layout; recommended). '
             '5 = [x, y, z, intensity, dt] is converted to 4-D by '
             'PointPillarsDetectorBridge._ensure_pointpillars_input, so the cache '
             'is equivalent but slower. Older revisions of the bridge would '
             'silently drop dt and feed intensity to the dt slot for 5-D input.',
    )
    parser.add_argument('--output-dir', type=str, default='/data/feature_maps_predictions_pointpillars')
    parser.add_argument('--save-dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--max-samples', type=int, default=0, help='0 means all valid frames')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def _to_save_dtype(t: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == 'float16':
        return t.half().contiguous().cpu()
    return t.float().contiguous().cpu()


def _extract_single_split(
    args: argparse.Namespace,
    split_name: str,
    out_root: Path,
    bridge: PointPillarsDetectorBridge,
    device: torch.device,
) -> dict:
    split_root = out_root / split_name
    feature_dir = split_root / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)

    ds_args = SimpleNamespace(
        refer_data_dir=args.refer_data_dir,
        nuscenes_dataroot=args.nuscenes_dataroot,
        nuscenes_ann_file=args.nuscenes_ann_file,
        sweeps_num=args.sweeps_num,
        point_cloud_range=args.point_cloud_range,
        question_types_json=args.question_types_json,
        question_types=None,
        queries_per_frame=1,
        precompute_bev=False,
        num_point_features=args.num_point_features,
        voxel_size=[0.16, 0.16, 4.0],
        lidar_root=None,
        backend_args=None,
        feature_cache_dir=None,
        feature_cache_strict=False,
    )

    print(f'[Extract] Building dataset split={split_name} ...', flush=True)
    print(f'[Extract] num_point_features={args.num_point_features}', flush=True)
    dataset = build_refer_dataset(split_name, ds_args)
    valid_indices = list(dataset.valid_indices)
    if args.max_samples > 0:
        valid_indices = valid_indices[:args.max_samples]
    print(f'[Extract] Valid frames to process: {len(valid_indices)}', flush=True)

    records = []
    sorted_failures = 0
    wrong_count = 0

    for i, flat_idx in enumerate(valid_indices):
        entry = dataset.frames[flat_idx]
        sample_token = str(entry.get('sample_token', f'flat_{flat_idx}'))
        lidar_path = str(entry.get('lidar_path', ''))
        lidar_basename = os.path.basename(lidar_path)
        cache_key = _cache_key(sample_token, lidar_path)

        points = dataset._load_points_with_sweeps(entry).to(device, non_blocking=True)

        with torch.no_grad():
            srcs, proposal_boxes, proposal_scores = bridge([points])
        # Per-proposal heading (radians), captured inside bridge.forward() and used
        # by the orientation-refinement (+angle) model. Shape (1, K).
        proposal_yaw = bridge.last_prop_yaw

        srcs = [s[0] for s in srcs]
        boxes = proposal_boxes[0]
        scores = proposal_scores[0]

        if boxes.shape[0] != args.proposal_queries:
            wrong_count += 1
        if scores.numel() > 1 and not bool(torch.all(scores[:-1] >= scores[1:])):
            sorted_failures += 1

        feature_file = feature_dir / f'{cache_key}.pt'
        payload = {
            'sample_token': sample_token,
            'lidar_path': lidar_path,
            'lidar_basename': lidar_basename,
            'cache_key': cache_key,
            'detector': 'pointpillars',
            'num_feature_levels': len(srcs),
            'feature_maps': [_to_save_dtype(s, args.save_dtype) for s in srcs],
            'proposal_boxes_8d': _to_save_dtype(boxes, 'float32'),
            'proposal_scores': _to_save_dtype(scores, 'float32'),
            'proposal_count': int(args.proposal_queries),
            'sweeps_num': int(args.sweeps_num),
        }
        payload['proposal_yaw'] = _to_save_dtype(proposal_yaw[0], 'float32')
        torch.save(payload, feature_file)

        queries = dataset.frame_refer_queries[flat_idx]
        qa_records = []
        for q in queries:
            qa_records.append({
                'qa_token': q.get('qa_token'),
                'query_type': q.get('query_type'),
                'query': q.get('query'),
                'target_tokens': [t.get('token') for t in q.get('targets', [])],
                'targets': q.get('targets', []),
            })

        pred_targets = []
        boxes_cpu = boxes.detach().cpu()
        scores_cpu = scores.detach().cpu()
        for rank in range(boxes_cpu.shape[0]):
            pred_targets.append({
                'rank': rank,
                'score': float(scores_cpu[rank].item()),
                'box_8d': [float(x) for x in boxes_cpu[rank].tolist()],
            })

        records.append({
            'split': split_name,
            'flat_idx': int(flat_idx),
            'cache_key': cache_key,
            'sample_token': sample_token,
            'lidar_path': lidar_path,
            'lidar_basename': lidar_basename,
            'feature_file': str(feature_file.relative_to(out_root)),
            'num_proposals': int(boxes_cpu.shape[0]),
            'qa_tokens': [q.get('qa_token') for q in queries],
            'qa_records': qa_records,
            'prediction_targets': pred_targets,
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f'[Extract] {i + 1}/{len(valid_indices)} processed | '
                f'sorted_failures={sorted_failures} wrong_count={wrong_count}',
                flush=True,
            )

    index = {
        'version': 1,
        'split': split_name,
        'output_root': str(out_root),
        'feature_subdir': str((split_root / 'features').relative_to(out_root)),
        'detector': 'pointpillars',
        'pointpillars_config': args.pointpillars_config,
        'pointpillars_ckpt': args.pointpillars_ckpt,
        'proposal_queries': int(args.proposal_queries),
        'num_feature_levels': 3,
        'sweeps_num': int(args.sweeps_num),
        'num_point_features': int(args.num_point_features),
        'question_types_json': args.question_types_json,
        'check_topk_count_mismatch': int(wrong_count),
        'check_score_sorted_failures': int(sorted_failures),
        'samples': records,
    }

    out_json_split = split_root / f'index_{split_name}.json'
    with open(out_json_split, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)

    out_json_root = out_root / f'index_{split_name}.json'
    with open(out_json_root, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)

    out_all = out_root / 'index_all.json'
    prev_samples = []
    prev_summaries = {}
    if out_all.exists():
        try:
            with open(out_all, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                if isinstance(prev.get('samples'), list):
                    prev_samples = prev['samples']
                if isinstance(prev.get('split_summaries'), dict):
                    prev_summaries = prev['split_summaries']
            elif isinstance(prev, list):
                prev_samples = prev
        except Exception:
            prev_samples = []
            prev_summaries = {}

    merged_samples = [
        r for r in prev_samples
        if not (isinstance(r, dict) and str(r.get('split')) == str(split_name))
    ]
    merged_samples.extend(records)

    prev_summaries[str(split_name)] = {
        'index_file': str(out_json_root.relative_to(out_root)),
        'num_samples': len(records),
        'proposal_queries': int(args.proposal_queries),
        'sweeps_num': int(args.sweeps_num),
        'num_point_features': int(args.num_point_features),
        'num_feature_levels': 3,
    }

    merged_index = {
        'version': 2,
        'detector': 'pointpillars',
        'output_root': str(out_root),
        'split_summaries': prev_summaries,
        'samples': merged_samples,
    }
    with open(out_all, 'w', encoding='utf-8') as f:
        json.dump(merged_index, f, indent=2)

    print(f'[Extract] Done split={split_name}', flush=True)
    print(f'[Extract] Saved split index: {out_json_split}', flush=True)
    print(f'[Extract] Saved root split index: {out_json_root}', flush=True)
    print(f'[Extract] Saved alias: {out_all}', flush=True)
    print(
        f'[Extract] top-k checks: count_mismatch={wrong_count}, score_sorted_failures={sorted_failures}',
        flush=True,
    )
    return {
        'split': split_name,
        'num_samples': len(records),
        'count_mismatch': int(wrong_count),
        'score_sorted_failures': int(sorted_failures),
    }


def main() -> None:
    args = parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    bridge = PointPillarsDetectorBridge(
        config_path=args.pointpillars_config,
        checkpoint_path=args.pointpillars_ckpt,
        point_cloud_range=args.point_cloud_range,
        proposal_count=args.proposal_queries,
        proposal_w_from=args.proposal_w_from,
        freeze=True,
    )

    device = torch.device(args.device)
    bridge.to(device)
    bridge.eval()

    requested_splits: List[str]
    if args.split == 'all':
        requested_splits = ['train', 'val']
    else:
        requested_splits = [args.split]

    summaries = []
    for split_name in requested_splits:
        summary = _extract_single_split(args, split_name, out_root, bridge, device)
        summaries.append(summary)

    if not summaries:
        raise RuntimeError('No splits were extracted successfully. Check split names and data paths.')

    total_samples = sum(s['num_samples'] for s in summaries)
    total_count_mismatch = sum(s['count_mismatch'] for s in summaries)
    total_sorted_failures = sum(s['score_sorted_failures'] for s in summaries)

    print('[Extract] Finished', flush=True)
    print(f'[Extract] completed_splits={[s["split"] for s in summaries]}', flush=True)
    print(f'[Extract] total_samples={total_samples}', flush=True)
    print(
        f'[Extract] total_top-k checks: count_mismatch={total_count_mismatch}, '
        f'score_sorted_failures={total_sorted_failures}',
        flush=True,
    )


if __name__ == '__main__':
    main()
