#!/usr/bin/env python3
"""Generate the result figures for the thesis results chapter.

Proposal-recall figures use the CORRECT pc_range (-50..50, matching the cached
proposals). They read the CSVs written by analyze_proposal_recall.py, which
reports BOTH:
  * IoU recall  @ {0.1,0.2,0.3,0.5}      (box-overlap / coverage view)
  * center-distance recall @ {0.5,1,2,4}m (the nuScenes-AP ceiling view)

Figures:
  1. topk_ceiling_iou.png   -- IoU recall vs K
  2. topk_ceiling_dist.png  -- center-distance recall vs K (AP ceiling)
  3. perclass_recall_dist.png -- per-class center-distance recall @K=150
  4. ceiling_vs_ap.png      -- per-class proposal ceiling (mean dist-recall)
                               vs achieved detection mAP  (the decoder gap)
  5. negatives_tradeoff.png -- P/R@0.5 and mAP vs negative ratio
  6. neg_ablation_types.png -- per-type mAP across the negative-ratio ablation

Usage (inside container):
    python make_thesis_figures.py \
        --recall-dir /data/outputs/prop_recall_val_HHMMSS --split val \
        --out-dir /data/outputs/thesis_figures
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --- Negative-ratio ablation (Table tab:neg_ablation, all_neg val) ----------
# mAP = mean over the 6 question types (mean-of-types), matching tab:neg_ablation.
NEG_ABLATION = {
    "no_neg":   dict(r=0.0, mAP=0.363, P=0.473, R=0.610,
                     types=dict(od=0.449, closest_sec=0.146, all_cat=0.307, closest=0.195, od_all=0.675, ped_road=0.408)),
    "baseline": dict(r=0.3, mAP=0.403, P=0.686, R=0.603,
                     types=dict(od=0.440, closest_sec=0.213, all_cat=0.347, closest=0.247, od_all=0.674, ped_road=0.494)),
    "balanced": dict(r=1.0, mAP=0.394, P=0.691, R=0.580,
                     types=dict(od=0.405, closest_sec=0.195, all_cat=0.348, closest=0.241, od_all=0.673, ped_road=0.498)),
    "all_neg":  dict(r=np.inf, mAP=0.342, P=0.694, R=0.378,
                     types=dict(od=0.247, closest_sec=0.073, all_cat=0.349, closest=0.195, od_all=0.676, ped_road=0.514)),
}
NEG_ORDER = ["no_neg", "baseline", "balanced", "all_neg"]
NEG_XLABELS = {"no_neg": "0", "baseline": "0.3", "balanced": "1", "all_neg": r"$\infty$"}
QTYPE_ORDER = ["od", "closest_sec", "all_cat", "closest", "od_all", "ped_road"]
QTYPE_LABELS = {"od": "object_detection", "closest_sec": "closest_in_sector", "all_cat": "all_category",
                "closest": "closest", "od_all": "object_detection_all", "ped_road": "pedestrians_on_road"}

# --- Per-class detection AP (Table tab:det_eval, val). AP@{0.5,1,2,4}m, mAP. --
# keys match analyze_proposal_recall class names.
# Per-class AP for the _final (+lang) model, object_detection_all_category, all_neg val.
# Recomputed from the saved eval pkl (per-class mean == type aggregate 0.343). See tab:det_eval.
DET_EVAL = {
    "car":                  [0.595, 0.712, 0.756, 0.789, 0.713],
    "truck":                [0.138, 0.274, 0.341, 0.385, 0.285],
    "bus":                  [0.099, 0.275, 0.376, 0.405, 0.289],
    "trailer":              [0.016, 0.143, 0.311, 0.422, 0.223],
    "construction_vehicle": [0.001, 0.037, 0.090, 0.120, 0.062],
    "pedestrian":           [0.645, 0.660, 0.681, 0.707, 0.673],
    "motorcycle":           [0.213, 0.296, 0.311, 0.322, 0.286],
    "bicycle":              [0.057, 0.074, 0.081, 0.090, 0.076],
    "barrier":              [0.285, 0.474, 0.557, 0.597, 0.478],
    "traffic_cone":         [0.297, 0.317, 0.350, 0.405, 0.342],
}
DIST_TS = [0.5, 1.0, 2.0, 4.0]


def _read_topk(path):
    rows = list(csv.DictReader(open(path)))
    ks = [int(float(r["top_k"])) for r in rows]
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return ks, cols


def plot_topk_ceiling(recall_csv, out_path, kind):
    if not os.path.isfile(recall_csv):
        print(f"[skip] {out_path}: {recall_csv} missing")
        return
    ks, cols = _read_topk(recall_csv)
    if kind == "iou":
        series = [("iou@0.1", "R@0.1", "o"), ("iou@0.3", "R@0.3", "s"), ("iou@0.5", "R@0.5", "^")]
        title, ylab = "Proposal recall ceiling (IoU) vs K", "IoU recall (%)"
    else:
        series = [("dist@0.5m", r"d$\leq$0.5m", "o"), ("dist@1.0m", r"d$\leq$1m", "s"),
                  ("dist@2.0m", r"d$\leq$2m", "^"), ("dist@4.0m", r"d$\leq$4m", "D")]
        title, ylab = "Proposal recall ceiling (center distance) vs K  =  nuScenes-AP ceiling", "Center-distance recall (%)"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for col, lbl, mk in series:
        if col not in cols:
            continue
        ys = cols[col]
        ax.plot(ks, ys, marker=mk, lw=2, ms=6, label=lbl)
        ax.annotate(f"{ys[-1]:.1f}", (ks[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), fontsize=9)
    ax.set_xlabel("Top-K proposals (ranked by score)"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.set_xticks(ks); ax.grid(True, alpha=0.3); ax.legend(loc="lower right"); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")


def _read_perclass(path, k=150):
    """Return {class: {col: value}} for the rows with top_k == k."""
    out = {}
    if not os.path.isfile(path):
        return out
    for r in csv.DictReader(open(path)):
        if int(float(r["top_k"])) != k:
            continue
        out[r["class"]] = {kk: float(v) for kk, v in r.items() if kk not in ("top_k", "class")}
    return out


def plot_perclass_iou_bars(recall_dir, split, out_path, ks=(50, 100, 150), col="iou@0.1"):
    path = os.path.join(recall_dir, f"recall_perclass_iou_{split}.csv")
    perk = {k: _read_perclass(path, k) for k in ks}
    if not all(perk[k] for k in ks):
        print(f"[skip] {out_path}: per-class iou csv missing")
        return
    classes = sorted(perk[ks[-1]], key=lambda c: perk[ks[-1]][c][col], reverse=True)
    x = np.arange(len(classes)); w = 0.8 / len(ks)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, k in enumerate(ks):
        ax.bar(x + i * w, [perk[k][c][col] for c in classes], w, label=f"K={k}")
    ax.set_xticks(x + w * (len(ks) - 1) / 2); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel(f"IoU recall {col.split('@')[1]} (%)"); ax.set_title(f"Per-class proposal recall (IoU>={col.split('@')[1]}) at K=50/100/150")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")


def plot_iou_vs_dist(recall_dir, split, out_path):
    """Per class at K=150: IoU>=0.1 recall vs center-distance d<=2m recall.
    Demonstrates that IoU under-counts SMALL objects (cone/ped/bike): their
    boxes are tiny, so a well-centred proposal still scores ~0 IoU."""
    pc_i = _read_perclass(os.path.join(recall_dir, f"recall_perclass_iou_{split}.csv"))
    pc_d = _read_perclass(os.path.join(recall_dir, f"recall_perclass_dist_{split}.csv"))
    if not pc_i or not pc_d:
        print(f"[skip] {out_path}: per-class csv missing")
        return
    classes = sorted(pc_i, key=lambda c: pc_d[c]["dist@2.0m"] - pc_i[c]["iou@0.1"], reverse=True)
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, [pc_d[c]["dist@2.0m"] for c in classes], w, label=r"center-distance recall d$\leq$2m", color="#55A868")
    ax.bar(x + w / 2, [pc_i[c]["iou@0.1"] for c in classes], w, label=r"IoU recall IoU$\geq$0.1", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("recall (%)"); ax.set_title(f"Center-distance vs IoU recall per class (K=150, split={split})\n(large gap = small object: present but tiny box => low IoU)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")


def plot_score_threshold(score_csv, out_path):
    """Recall (IoU) vs detector score threshold, from analyze_proposal_threshold prop.csv."""
    if not os.path.isfile(score_csv):
        print(f"[skip] {out_path}: {score_csv} missing")
        return
    rows = list(csv.DictReader(open(score_csv)))
    th = np.array([float(r["score_threshold"]) for r in rows])
    avg = np.array([float(r["avg_proposals"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for col, lbl, mk in [("recall@0.1", "R@0.1", "o"), ("recall@0.3", "R@0.3", "s"), ("recall@0.5", "R@0.5", "^")]:
        if col in rows[0]:
            ax.plot(th, [float(r[col]) for r in rows], marker=mk, lw=2, ms=5, label=lbl)
    ax.set_xlabel("Detector score threshold"); ax.set_ylabel("IoU recall (%)")
    ax.set_title("Proposal recall vs detector score threshold (train, pc_range -50)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left"); ax.set_ylim(0, 100)
    ax2 = ax.twinx(); ax2.plot(th, avg, "k--", alpha=0.5, label="avg proposals kept")
    ax2.set_ylabel("avg proposals kept"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")


def plot_perclass_recall_dist(recall_dir, split, out_path):
    pc = _read_perclass(os.path.join(recall_dir, f"recall_perclass_dist_{split}.csv"))
    if not pc:
        print(f"[skip] {out_path}: per-class dist csv missing")
        return
    classes = sorted(pc, key=lambda c: pc[c]["dist@4.0m"], reverse=True)
    cols = [("dist@0.5m", r"d$\leq$0.5m"), ("dist@1.0m", r"d$\leq$1m"),
            ("dist@2.0m", r"d$\leq$2m"), ("dist@4.0m", r"d$\leq$4m")]
    x = np.arange(len(classes)); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (c, lbl) in enumerate(cols):
        ax.bar(x + i * w, [pc[cl][c] for cl in classes], w, label=lbl)
    ax.set_xticks(x + w * (len(cols) - 1) / 2); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Center-distance recall (%)"); ax.set_title("Per-class proposal recall by center distance (K=150)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(ncol=4); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")


def _load_det_map(det_eval_json):
    """class -> mAP from a det_eval_{split}.json (per_class). None on failure."""
    if not det_eval_json or not os.path.isfile(det_eval_json):
        return None
    d = json.load(open(det_eval_json))
    pc = d.get("per_class", {})
    return {c: float(v["mAP"]) for c, v in pc.items() if "mAP" in v}


def plot_ceiling_vs_ap(recall_dir, split, out_path, out_csv, det_map=None):
    """Per-class: proposal-recall ceiling mAP-equiv (mean of dist-recalls at the
    4 thresholds) vs achieved detection mAP. The gap = decoder/referring + precision loss.

    det_map: optional {class: mAP} (split-matched, e.g. from det_eval_train.json).
    Falls back to the embedded val DET_EVAL table when None."""
    pc = _read_perclass(os.path.join(recall_dir, f"recall_perclass_dist_{split}.csv"))
    if not pc:
        print(f"[skip] {out_path}: per-class dist csv missing")
        return
    ap_map = det_map if det_map is not None else {c: v[4] for c, v in DET_EVAL.items()}
    classes = [c for c in ap_map if c in pc]
    ceil = {c: np.mean([pc[c][f"dist@{t}m"] for t in DIST_TS]) / 100.0 for c in classes}
    achieved = {c: ap_map[c] for c in classes}
    classes.sort(key=lambda c: ceil[c] - achieved[c], reverse=True)
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, [ceil[c] for c in classes], w, label="Proposal ceiling (mean dist-recall)", color="#4C72B0")
    ax.bar(x + w / 2, [achieved[c] for c in classes], w, label="Achieved detection mAP", color="#DD8452")
    for i, c in enumerate(classes):
        ax.annotate(f"{ceil[c]-achieved[c]:.2f}", (i, max(ceil[c], achieved[c]) + 0.02), ha="center", fontsize=8, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("score"); ax.set_title(f"Proposal-recall ceiling vs achieved mAP per class (split={split})\n(gap above each pair = decoder/referring + precision loss)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(); ax.set_ylim(0, 1.0)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")
    with open(out_csv, "w") as f:
        f.write("class,proposal_ceiling_meanDistRecall,achieved_mAP,gap\n")
        for c in classes:
            f.write(f"{c},{ceil[c]:.3f},{achieved[c]:.3f},{ceil[c]-achieved[c]:.3f}\n")
    print(f"[ok] {out_csv}")


def plot_negatives_tradeoff(out_path):
    xs = np.arange(len(NEG_ORDER))
    P = [NEG_ABLATION[m]["P"] for m in NEG_ORDER]; R = [NEG_ABLATION[m]["R"] for m in NEG_ORDER]
    mAP = [NEG_ABLATION[m]["mAP"] for m in NEG_ORDER]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, P, "o-", lw=2, ms=7, label="Precision@0.5")
    ax.plot(xs, R, "s-", lw=2, ms=7, label="Recall@0.5")
    ax.plot(xs, mAP, "^-", lw=2, ms=7, label="mAP")
    bi = NEG_ORDER.index("baseline"); ax.axvline(bi, color="grey", ls="--", alpha=0.5)
    ax.annotate("chosen (r=0.3)", (bi, 0.66), ha="center", fontsize=9, color="grey")
    ax.set_xticks(xs); ax.set_xticklabels([NEG_XLABELS[m] for m in NEG_ORDER])
    ax.set_xlabel("Negative ratio r"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall@0.5 and mAP vs negative ratio")
    ax.grid(True, alpha=0.3); ax.legend(loc="center left"); ax.set_ylim(0, 0.75)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")


def plot_neg_types(out_path):
    x = np.arange(len(QTYPE_ORDER)); w = 0.8 / len(NEG_ORDER)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(NEG_ORDER):
        ax.bar(x + i * w, [NEG_ABLATION[m]["types"][t] for t in QTYPE_ORDER], w, label=f"{m} (r={NEG_XLABELS[m]})")
    ax.set_xticks(x + w * (len(NEG_ORDER) - 1) / 2)
    ax.set_xticklabels([QTYPE_LABELS[t] for t in QTYPE_ORDER], rotation=25, ha="right")
    ax.set_ylabel("mAP"); ax.set_title("Per-question-type mAP across negative-ratio ablation")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[ok] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall-dir", required=True, help="analyze_proposal_recall.py output dir")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out-dir", default="/data/outputs/thesis_figures")
    ap.add_argument("--score-csv", default=None,
                    help="optional analyze_proposal_threshold prop.csv (score-threshold sweep, -50)")
    ap.add_argument("--det-eval-json", default=None,
                    help="optional det_eval_{split}.json for split-matched per-class mAP in ceiling_vs_ap")
    ap.add_argument("--no-neg", action="store_true", help="skip the negative-ratio figures")
    ap.add_argument("--no-ceiling-vs-ap", action="store_true",
                    help="skip ceiling_vs_ap (use when no split-matched detection mAP is available)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    topk_csv = os.path.join(args.recall_dir, f"recall_topk_{args.split}.csv")
    p = lambda n: os.path.join(args.out_dir, n)

    plot_topk_ceiling(topk_csv, p("topk_ceiling_iou.png"), "iou")
    plot_topk_ceiling(topk_csv, p("topk_ceiling_dist.png"), "dist")
    plot_perclass_recall_dist(args.recall_dir, args.split, p("perclass_recall_dist.png"))
    plot_perclass_iou_bars(args.recall_dir, args.split, p("perclass_recall_iou.png"))
    plot_iou_vs_dist(args.recall_dir, args.split, p("iou_vs_dist_perclass.png"))
    if not args.no_ceiling_vs_ap:
        det_map = _load_det_map(args.det_eval_json)
        if args.det_eval_json and det_map is None:
            print(f"[warn] could not read det_eval json {args.det_eval_json}; using embedded val mAP")
        plot_ceiling_vs_ap(args.recall_dir, args.split, p("ceiling_vs_ap.png"), p("ceiling_vs_ap.csv"), det_map=det_map)
    if args.score_csv:
        plot_score_threshold(args.score_csv, p("score_threshold_recall.png"))
    if not args.no_neg:
        plot_negatives_tradeoff(p("negatives_tradeoff.png"))
        plot_neg_types(p("neg_ablation_types.png"))
    print(f"\nFigures -> {args.out_dir}")


if __name__ == "__main__":
    main()
