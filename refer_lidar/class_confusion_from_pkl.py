#!/usr/bin/env python3
"""Class-confusion matrix for the object_detection_all_category query type, built
from a single saved eval pkl. No GPU / no model run. (For a run split across
several eval shards, use confusion_from_shards.py instead.)

Records are stored strictly 10-per-frame (one query per class, fixed order), so the
full-frame multi-class GT for a frame = union of that frame's 10 records' gt_boxes,
each labelled by its query class (cname). For each query "find all X", we take the
confident predictions (refer-score >= thr) and greedily match them (center distance
<= 2 m) to the full-frame GT. confusion[X][Y] = boxes returned for "find all X" that
are truly class Y; column 'background' = confident boxes matching no GT within 2 m.

Usage:  python class_confusion_from_pkl.py <eval_shard.pkl> [refer_score_thr]
"""
import sys, os, pickle, numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: python class_confusion_from_pkl.py <eval_shard.pkl> [refer_score_thr]")
PKL = sys.argv[1]
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # refer-score operating point
DIST = 2.0
CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
CIDX = {c: i for i, c in enumerate(CLASSES)}

print(f"[load] {PKL}  (refer-score thr={THR}, dist<= {DIST}m)", flush=True)
recs = pickle.load(open(PKL, "rb"))["per_type_records"]["object_detection_all_category"]
assert len(recs) % 10 == 0, len(recs)
nframes = len(recs) // 10

# confusion[query_class][true_class or 'background'] = count of confident returned boxes
conf = np.zeros((10, 11), dtype=np.int64)   # last col = background
gt_total = np.zeros(10, dtype=np.int64)      # GTs per class (for context)

for f in range(nframes):
    block = recs[f * 10:(f + 1) * 10]
    # full-frame GT: union of all classes' gt_boxes (meters), labelled by class
    gt_boxes, gt_lab = [], []
    for j, rec in enumerate(block):
        gb = rec['gt_boxes']
        if gb.shape[0]:
            gt_boxes.append(gb[:, :2]); gt_lab += [j] * gb.shape[0]
            gt_total[j] += gb.shape[0]
    gt_xy = np.concatenate(gt_boxes, 0) if gt_boxes else np.zeros((0, 2), np.float32)
    gt_lab = np.array(gt_lab, dtype=np.int64)

    for x, rec in enumerate(block):                 # query class x = "find all CLASSES[x]"
        pb, ps = rec['pred_boxes'], rec['pred_scores']
        keep = ps >= THR
        if not keep.any():
            continue
        pxy = pb[keep, :2]
        order = np.argsort(-ps[keep])               # confident-first (nuScenes-style)
        used = np.zeros(gt_xy.shape[0], dtype=bool)
        for pi in order:
            if gt_xy.shape[0] == 0:
                conf[x, 10] += 1; continue
            d = np.sqrt(((pxy[pi] - gt_xy) ** 2).sum(1))
            d[used] = np.inf
            j = int(np.argmin(d)) if gt_xy.shape[0] else -1
            if j >= 0 and d[j] <= DIST:
                used[j] = True; conf[x, gt_lab[j]] += 1
            else:
                conf[x, 10] += 1                    # background / hallucination

# ---- report ----
short = {'car': 'car', 'truck': 'truck', 'bus': 'bus', 'trailer': 'trail', 'construction_vehicle': 'const',
         'pedestrian': 'ped', 'motorcycle': 'moto', 'bicycle': 'bike', 'traffic_cone': 'cone', 'barrier': 'barr'}
hdr = [short[c] for c in CLASSES] + ['bg']
print("\nRow-normalized confusion (%% of confident boxes returned for 'find all <row>'):")
print("query\\true   " + " ".join(f"{h:>5}" for h in hdr) + "   |   #boxes")
for x in range(10):
    tot = conf[x].sum()
    if tot == 0:
        print(f"{short[CLASSES[x]]:>10}   " + " ".join("    -" for _ in hdr) + "   |   0")
        continue
    row = conf[x] / tot * 100
    cells = []
    for k in range(11):
        s = f"{row[k]:5.1f}"
        cells.append(s)
    print(f"{short[CLASSES[x]]:>10}   " + " ".join(cells) + f"   |   {tot}")

print("\nPer query class: correct / confused-to-other / background  (and top confusion):")
for x in range(10):
    tot = conf[x].sum()
    if tot == 0:
        print(f"  {CLASSES[x]:22s} no confident boxes"); continue
    correct = conf[x, x] / tot * 100
    bg = conf[x, 10] / tot * 100
    other = 100 - correct - bg
    off = [(conf[x, y], CLASSES[y]) for y in range(10) if y != x]
    off.sort(reverse=True)
    top = ", ".join(f"{nm}={c/tot*100:.1f}%" for c, nm in off[:3] if c > 0)
    print(f"  {CLASSES[x]:22s} correct={correct:5.1f}  other={other:5.1f}  bg={bg:5.1f}   | top: {top}")

# dump row-normalized matrix for the heatmap figure
_out = os.environ.get('CONF_OUT', '/data/outputs/confusion_refer.csv')
with open(_out, 'w') as _f:
    _f.write('pred,' + ','.join(CLASSES) + ',bg\n')
    for _x in range(10):
        _tot = max(int(conf[_x].sum()), 1)
        _f.write(CLASSES[_x] + ',' + ','.join(f'{conf[_x,_k]/_tot*100:.2f}' for _k in range(11)) + '\n')
print(f'[csv] {_out}')
