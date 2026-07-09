"""Diagnose WHY angle (+lang+yaw) has worse mAOE than langdec (+lang).

Loads shard_0 of both models (identical frames & queries), replicates the
devkit-style conf-sorted greedy 2.0 m TP matching per query record, and
compares the TP yaw-error DISTRIBUTIONS. Key question: does the angle model
inherit PointPillars direction flips (error spike near pi) that the free
sin/cos head avoids?
"""
import pickle, sys
import numpy as np

SH = {
    'langdec (+lang, free sin/cos)': '/data/outputs/eval_ghostclean_wave2_langdec_baseline/shard_0_of_10.pkl',
    'angle (+lang+yaw, prop+delta)': '/data/outputs/eval_ghostclean_wave2_angle_baseline/shard_0_of_10.pkl',
}

PERIOD_PI = {'barrier'}          # nuScenes: barrier orientation mod pi
EXCLUDED  = {'traffic_cone'}     # cone orientation excluded


def yaw_err(pred_attr, gt_attr, cname):
    py = np.arctan2(pred_attr[2], pred_attr[3])
    gy = np.arctan2(gt_attr[2], gt_attr[3])
    d = abs(py - gy) % (2 * np.pi)
    if d > np.pi:
        d = 2 * np.pi - d
    if cname in PERIOD_PI and d > np.pi / 2:
        d = np.pi - d
    return d


def tp_yaw_errors(records):
    """Greedy conf-sorted 2.0m matching inside each query record -> yaw errors.

    Returns dict: cname -> list[(qkey, gt_idx, err)] so models can be paired.
    """
    out = {}
    for ri, rec in enumerate(records):
        boxes, scores = rec['pred_boxes'], rec['pred_scores']
        attrs, g_attrs = rec.get('pred_attrs_3d'), rec.get('gt_attrs_3d')
        g_boxes, g_names = rec['gt_boxes'], rec.get('gt_class_names')
        if (attrs is None or g_attrs is None or g_boxes.shape[0] == 0
                or boxes.shape[0] == 0):
            continue
        order = np.argsort(-scores)
        used = np.zeros(g_boxes.shape[0], dtype=bool)
        for pi in order:
            d = np.sqrt((boxes[pi, 0] - g_boxes[:, 0]) ** 2
                        + (boxes[pi, 1] - g_boxes[:, 1]) ** 2)
            d[used] = np.inf
            gi = int(np.argmin(d))
            if d[gi] < 2.0:
                used[gi] = True
                cn = (g_names[gi] if g_names is not None and len(g_names) > gi
                      else 'unknown')
                if cn in EXCLUDED:
                    continue
                if g_attrs.shape[0] > gi:
                    e = yaw_err(attrs[pi], g_attrs[gi], cn)
                    out.setdefault(cn, []).append(((ri, gi), float(e)))
    return out


def summarize(name, errs_by_class):
    all_e = np.array([e for v in errs_by_class.values() for _, e in v])
    print(f"\n=== {name} ===  TPs(with yaw)={len(all_e)}")
    print(f"  mean={all_e.mean():.4f}  median={np.median(all_e):.4f}")
    bins = [(0, 0.1), (0.1, 0.5), (0.5, np.pi / 2), (np.pi / 2, np.pi - 0.3),
            (np.pi - 0.3, np.pi + 1e-6)]
    labels = ['<0.1 (good)', '0.1-0.5', '0.5-pi/2', 'pi/2-2.84', '>2.84 (FLIP)']
    for (lo, hi), lb in zip(bins, labels):
        f = ((all_e >= lo) & (all_e < hi)).mean()
        print(f"    {lb:16s} {100 * f:5.1f}%")
    print(f"  flip-rate (err>pi/2): {100 * (all_e > np.pi / 2).mean():.2f}%")
    print("  per-class mean err (n>=50):")
    for cn in sorted(errs_by_class):
        es = np.array([e for _, e in errs_by_class[cn]])
        if len(es) >= 50:
            print(f"    {cn:22s} n={len(es):6d} mean={es.mean():.4f} "
                  f"flip%={100 * (es > np.pi / 2).mean():5.1f}")
    return {cn: dict(v) for cn, v in errs_by_class.items()}


res = {}
for name, path in SH.items():
    print(f"loading {path} ...", flush=True)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    recs = [r for v in data['per_type_records'].values() for r in v]
    print(f"  {len(recs)} query records")
    res[name] = summarize(name, tp_yaw_errors(recs))

# ---- PAIRED comparison on identical (record, gt) matches ----
names = list(SH)
a, b = res[names[0]], res[names[1]]
print("\n=== PAIRED (same record idx + same GT matched by both) ===")
pairs = []
for cn in a:
    if cn not in b:
        continue
    common = set(a[cn]) & set(b[cn])
    pairs += [(a[cn][k], b[cn][k]) for k in common]
pa = np.array([p for p, _ in pairs]); pb = np.array([q for _, q in pairs])
print(f"  paired TPs: {len(pairs)}")
print(f"  {names[0]}: mean={pa.mean():.4f} flip%={100*(pa>np.pi/2).mean():.2f}")
print(f"  {names[1]}: mean={pb.mean():.4f} flip%={100*(pb>np.pi/2).mean():.2f}")
# where does angle lose? decompose the paired delta
d = pb - pa
print(f"  mean paired delta (angle - langdec): {d.mean():+.4f}")
both_ok  = ((pa <= np.pi/2) & (pb <= np.pi/2))
a_flip   = ((pa >  np.pi/2) & (pb <= np.pi/2))
b_flip   = ((pa <= np.pi/2) & (pb >  np.pi/2))
bothflip = ((pa >  np.pi/2) & (pb >  np.pi/2))
print(f"  neither flipped: {both_ok.sum():6d}  mean delta there {d[both_ok].mean():+.4f}")
print(f"  ONLY angle flipped:   {b_flip.sum():6d}  (angle inherits/creates flips)")
print(f"  ONLY langdec flipped: {a_flip.sum():6d}")
print(f"  both flipped:         {bothflip.sum():6d}")
# contribution of flip mismatch to the mean gap
contrib_flip = (d[b_flip].sum() - (-d[a_flip]).sum()) / len(d)
contrib_small = d[both_ok | bothflip].sum() / len(d)
print(f"  gap contribution: flip-mismatch {contrib_flip:+.4f} | small-angle {contrib_small:+.4f}")
