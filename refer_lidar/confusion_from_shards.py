#!/usr/bin/env python3
"""Class-confusion matrix for object_detection_all_category, summed over one or
more eval shard pkls (each holding per_type_records). No GPU / no model run.

Records are 10-per-frame (one "find all <class>" query per class). For each frame
we form the full-frame multi-class GT = union of that frame's 10 records' gt_boxes,
each labelled by the record's own `cname` (robust to save-order). For query X we
take confident preds (refer-score >= THR), greedily match by center distance
(<= DIST m, meters) to the full-frame GT, and tally confusion[X][trueclass];
column 'bg' = confident boxes matching no GT within DIST.

Usage:
  python confusion_from_shards.py 'OUT/shard_*_of_8.pkl' [THR] [OUT_CSV]
  python confusion_from_shards.py OUT/shard_0_of_8.pkl OUT/shard_1_of_8.pkl ... [THR] [OUT_CSV]
"""
import sys, os, glob, pickle
import numpy as np

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
CIDX = {c: i for i, c in enumerate(CLASSES)}
SHORT = {'car': 'car', 'truck': 'truck', 'bus': 'bus', 'trailer': 'trail',
         'construction_vehicle': 'const', 'pedestrian': 'ped', 'motorcycle': 'moto',
         'bicycle': 'bike', 'traffic_cone': 'cone', 'barrier': 'barr'}

_ALIAS = {'constructionvehicle': 'construction_vehicle', 'construction vehicle': 'construction_vehicle',
          'trafficcone': 'traffic_cone', 'traffic cone': 'traffic_cone'}


def _canon(name):
    s = str(name).strip().lower().replace('-', '_')
    s = _ALIAS.get(s.replace('_', ''), _ALIAS.get(s, s))
    return s if s in CIDX else None


def _nms_center_keep(pred_boxes, pred_scores, radius):
    """Greedy center-distance NMS keep-mask, identical to the eval merge's
    _nms_records_by_center (inference_lidar_simple.py): keep boxes best-first by
    refer score, drop any whose BEV center lies within `radius` m of a kept box."""
    n = min(pred_boxes.shape[0], pred_scores.shape[0])
    keep = np.ones(n, dtype=bool)
    if radius is None or float(radius) <= 0.0 or n <= 1:
        return keep
    r2 = float(radius) ** 2
    centers = pred_boxes[:n, :2]
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    suppressed = np.zeros(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for i in np.argsort(-pred_scores[:n], kind='stable'):
        if suppressed[i]:
            continue
        keep[i] = True
        nb = d2[i] < r2
        nb[i] = False
        suppressed |= nb
    return keep


def _collect_args(argv):
    thr, out_csv, pkls, nms = 0.5, '/data/outputs/confusion_refer_angle.csv', [], 0.0
    for a in argv:
        if a.endswith('.csv'):
            out_csv = a
        elif a.endswith('.pkl') or '*' in a:
            pkls.extend(sorted(glob.glob(a)) if '*' in a else [a])
        elif a.startswith('nms='):
            nms = float(a.split('=', 1)[1])
        else:
            try:
                thr = float(a)
            except ValueError:
                pass
    return thr, out_csv, pkls, nms


def main():
    thr, out_csv, pkls, nms = _collect_args(sys.argv[1:])
    if not pkls:
        print('no pkls given'); sys.exit(1)
    dist = 2.0
    print(f'[confusion] {len(pkls)} shard(s), refer-score thr={thr}, dist<={dist}m, nms={nms}m', flush=True)

    conf = np.zeros((10, 11), dtype=np.int64)   # [query_class][true_class | bg]
    gt_total = np.zeros(10, dtype=np.int64)
    n_frames = 0          # complete-10 frames used
    n_partial = 0         # frames dropped (missing some classes)
    for p in pkls:
        recs = pickle.load(open(p, 'rb'))['per_type_records']['object_detection_all_category']
        # Records carry no frame id, but each frame emits its class queries in a
        # FIXED order with each class at most once, so a frame boundary is exactly
        # where a cname repeats. (Sharding is by whole frames, so no frame spans two
        # shards.) We keep only frames that have all 10 classes -> the full-frame GT
        # union is complete and every returned box can be labelled by its true class.
        blocks = []
        cur, seen = [], set()
        for r in recs:
            c = _canon(r['cname'])
            if c in seen:
                blocks.append(cur); cur, seen = [], set()
            cur.append(r); seen.add(c)
        if cur:
            blocks.append(cur)
        for block in blocks:
            names = {_canon(r['cname']) for r in block}
            if len(block) != 10 or len(names) != 10 or None in names:
                n_partial += 1
                continue
            n_frames += 1
            gt_xy, gt_lab = [], []
            for rec in block:
                gb = rec['gt_boxes']
                ci = CIDX.get(_canon(rec['cname']))
                if gb.shape[0] and ci is not None:
                    gt_xy.append(gb[:, :2]); gt_lab += [ci] * gb.shape[0]
                    gt_total[ci] += gb.shape[0]
            gt_xy = np.concatenate(gt_xy, 0) if gt_xy else np.zeros((0, 2), np.float32)
            gt_lab = np.array(gt_lab, dtype=np.int64)
            for rec in block:
                x = CIDX.get(_canon(rec['cname']))
                if x is None:
                    continue
                ps = np.asarray(rec['pred_scores'], np.float32)
                pb = np.asarray(rec['pred_boxes'], np.float32)
                if nms > 0.0:                       # de-dup exactly as the eval merge does
                    nkeep = _nms_center_keep(pb, ps, nms)
                    n = nkeep.shape[0]
                    ps = ps[:n][nkeep]; pb = pb[:n][nkeep]
                keep = ps >= thr
                if not keep.any():
                    continue
                pxy = pb[keep, :2]
                order = np.argsort(-ps[keep])
                used = np.zeros(gt_xy.shape[0], dtype=bool)
                for pi in order:
                    if gt_xy.shape[0] == 0:
                        conf[x, 10] += 1; continue
                    d = np.sqrt(((pxy[pi] - gt_xy) ** 2).sum(1))
                    d[used] = np.inf
                    j = int(np.argmin(d))
                    if d[j] <= dist:
                        used[j] = True; conf[x, gt_lab[j]] += 1
                    else:
                        conf[x, 10] += 1
    print(f'[confusion] complete-10 frames used={n_frames}  partial frames dropped={n_partial}', flush=True)

    hdr = [SHORT[c] for c in CLASSES] + ['bg']
    print('\nRow-normalized confusion (% of confident boxes for "find all <row>"):')
    print('query\\true ' + ' '.join(f'{h:>5}' for h in hdr) + '   | #boxes  GT')
    for x in range(10):
        tot = conf[x].sum()
        if tot == 0:
            print(f'{SHORT[CLASSES[x]]:>10} ' + ' '.join('    -' for _ in hdr) + f'   | 0  {gt_total[x]}'); continue
        row = conf[x] / tot * 100
        print(f'{SHORT[CLASSES[x]]:>10} ' + ' '.join(f'{row[k]:5.1f}' for k in range(11)) + f'   | {tot}  {gt_total[x]}')

    print('\nPer query class: correct / other / bg  (top confusions):')
    for x in range(10):
        tot = conf[x].sum()
        if tot == 0:
            print(f'  {CLASSES[x]:22s} no confident boxes'); continue
        correct = conf[x, x] / tot * 100; bg = conf[x, 10] / tot * 100
        off = sorted([(conf[x, y], CLASSES[y]) for y in range(10) if y != x], reverse=True)
        top = ', '.join(f'{nm}={c/tot*100:.1f}%' for c, nm in off[:3] if c > 0)
        print(f'  {CLASSES[x]:22s} correct={correct:5.1f}  other={100-correct-bg:5.1f}  bg={bg:5.1f}   | {top}')

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w') as fo:
        fo.write('pred,' + ','.join(CLASSES) + ',bg\n')
        for x in range(10):
            tot = max(int(conf[x].sum()), 1)
            fo.write(CLASSES[x] + ',' + ','.join(f'{conf[x,k]/tot*100:.2f}' for k in range(11)) + '\n')
    print(f'[csv] {out_csv}', flush=True)


if __name__ == '__main__':
    main()
