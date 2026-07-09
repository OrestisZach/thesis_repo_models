#!/usr/bin/env python3
"""Native PointPillars class-confusion matrix on the val set.

The refer pipeline collapses PointPillars to *class-agnostic* proposals (boxes +
one scalar score). Here we run PointPillars' OWN detection head on the cached FPN
features (exact: the cache stores extract_pts_feat output) to recover its native
multi-class predictions (labels_3d), then match them to GT (center distance <= 2m)
and build confusion[pred_class][true_class]. Compared with the refer model's
confusion (class_confusion_from_pkl.py), this tells us whether the downstream
refer model *disambiguates* or *ambiguates* classes relative to the detector.
"""
import sys, os, time, argparse
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmdet3d.utils import register_all_modules
from mmdet3d.structures import LiDARInstance3DBoxes
from nuscenes_lidar_simple import build as build_refer_dataset
from analyze_proposal_threshold import load_cached_payload, gt_classes_for_frame
from models.refer_model_lang_dec import PointPillarsDetectorBridge

# PointPillars label order (configs/_base_/datasets/nus-3d.py)
PP_NAMES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
# Display order (matches the refer-model confusion table)
CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
CIDX = {c: i for i, c in enumerate(CLASSES)}
SHORT = {'car': 'car', 'truck': 'truck', 'bus': 'bus', 'trailer': 'trail',
         'construction_vehicle': 'const', 'pedestrian': 'ped', 'motorcycle': 'moto',
         'bicycle': 'bike', 'traffic_cone': 'cone', 'barrier': 'barr'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature-cache-dir', default='/data/cache/pointpillars_fp16')
    ap.add_argument('--refer-data-dir', default='/data/nuscenes/all_neg')
    ap.add_argument('--data-root', default='/data/nuscenes')
    ap.add_argument('--pointpillars-config', default='/workspace/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py')
    ap.add_argument('--pointpillars-ckpt', default='/data/ckpts/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth')
    ap.add_argument('--point-cloud-range', type=float, nargs=6, default=[-50, -50, -5, 50, 50, 3])
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--dist-m', type=float, default=2.0)
    ap.add_argument('--max-frames', type=int, default=None)
    args = ap.parse_args()

    register_all_modules(init_default_scope=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    span_m = float(args.point_cloud_range[3] - args.point_cloud_range[0])

    bridge = PointPillarsDetectorBridge(
        config_path=args.pointpillars_config, checkpoint_path=args.pointpillars_ckpt,
        point_cloud_range=args.point_cloud_range, proposal_count=150)
    bridge.to(device).eval()
    head = bridge.detector.pts_bbox_head
    metas = [{'box_type_3d': LiDARInstance3DBoxes}]

    ds_args = SimpleNamespace(
        refer_data_dir=args.refer_data_dir, nuscenes_dataroot=args.data_root,
        nuscenes_ann_file=os.path.join(args.data_root, 'nuscenes_infos_val.pkl'),
        sweeps_num=10, point_cloud_range=args.point_cloud_range, question_types_json=None,
        question_types=None, queries_per_frame=1, precompute_bev=False,
        num_point_features=5, voxel_size=[0.08, 0.08, 4.0], lidar_root=None,
        backend_args=None, feature_cache_dir=args.feature_cache_dir, feature_cache_strict=False)
    dataset = build_refer_dataset('val', ds_args)
    idxs = list(range(len(dataset.frames)))
    if args.max_frames:
        idxs = idxs[:args.max_frames]
    print(f'Frames: {len(idxs)}  score_thr={args.score_thr} dist<= {args.dist_m}m', flush=True)

    conf = np.zeros((10, 11), dtype=np.int64)   # [pred_class][true_class | background]
    n_skip = 0
    t0 = time.time()
    for fi, flat_idx in enumerate(idxs):
        if (fi + 1) % 500 == 0 or fi == 0:
            print(f'  [{fi+1}/{len(idxs)}] {(fi+1)/max(time.time()-t0,1e-3):.1f} fps', flush=True)
        entry = dataset.frames[flat_idx]
        cached = load_cached_payload(dataset, entry, 'proposal_boxes_8d')
        if cached is None:
            n_skip += 1
            continue
        feats = [s.unsqueeze(0).to(device) if s.dim() == 3 else s.to(device) for s in cached['srcs']]
        with torch.no_grad():
            cls_scores, bbox_preds, dir_cls_preds = head(feats)
            results = head.predict_by_feat(cls_scores, bbox_preds, dir_cls_preds,
                                           batch_input_metas=metas, cfg=head.test_cfg, rescale=False)
        det = results[0]
        pb = det.bboxes_3d.tensor.detach().cpu()
        ps = det.scores_3d.detach().cpu().numpy()
        pl = det.labels_3d.detach().cpu().numpy()
        keep = ps >= args.score_thr
        if keep.sum() == 0:
            continue
        pb, ps, pl = pb[keep], ps[keep], pl[keep]
        pxy = bridge._normalize_proposals(pb).numpy()[:, :2]   # normalized centers

        targets = dataset._build_targets(flat_idx, set())
        gt_boxes = targets['boxes'].numpy()
        gt_names = gt_classes_for_frame(dataset, flat_idx)
        gt_xy = gt_boxes[:, :2] if gt_boxes.shape[0] else np.zeros((0, 2), np.float32)

        order = np.argsort(-ps)
        used = np.zeros(gt_xy.shape[0], dtype=bool)
        for pi in order:
            name = PP_NAMES[int(pl[pi])]
            row = CIDX[name]
            if gt_xy.shape[0] == 0:
                conf[row, 10] += 1; continue
            d = np.sqrt(((pxy[pi] - gt_xy) ** 2).sum(1)) * span_m
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= args.dist_m and gt_names[j] in CIDX:
                used[j] = True; conf[row, CIDX[gt_names[j]]] += 1
            else:
                conf[row, 10] += 1
    print(f'Done {time.time()-t0:.1f}s  skipped={n_skip}', flush=True)

    hdr = [SHORT[c] for c in CLASSES] + ['bg']
    print("\nPointPillars native confusion (row-normalized %, pred class -> true class):")
    print("pred\\true   " + " ".join(f"{h:>5}" for h in hdr) + "   |   #det")
    for x in range(10):
        tot = conf[x].sum()
        if tot == 0:
            print(f"{SHORT[CLASSES[x]]:>9}   " + " ".join("    -" for _ in hdr) + "   |   0"); continue
        row = conf[x] / tot * 100
        print(f"{SHORT[CLASSES[x]]:>9}   " + " ".join(f"{row[k]:5.1f}" for k in range(11)) + f"   |   {tot}")
    print("\nPer predicted class: correct / other-class / background:")
    for x in range(10):
        tot = conf[x].sum()
        if tot == 0:
            print(f"  {CLASSES[x]:22s} no detections"); continue
        correct = conf[x, x] / tot * 100
        bg = conf[x, 10] / tot * 100
        off = sorted([(conf[x, y], CLASSES[y]) for y in range(10) if y != x], reverse=True)
        top = ", ".join(f"{nm}={c/tot*100:.1f}%" for c, nm in off[:3] if c > 0)
        print(f"  {CLASSES[x]:22s} correct={correct:5.1f}  other={100-correct-bg:5.1f}  bg={bg:5.1f}   | top: {top}")

    out_csv = os.environ.get('CONF_OUT', '/data/outputs/confusion_pp.csv')
    with open(out_csv, 'w') as f:
        f.write('pred,' + ','.join(CLASSES) + ',bg\n')
        for x in range(10):
            tot = max(int(conf[x].sum()), 1)
            f.write(CLASSES[x] + ',' + ','.join(f'{conf[x,k]/tot*100:.2f}' for k in range(11)) + '\n')
    print(f'[csv] {out_csv}')


if __name__ == '__main__':
    main()
