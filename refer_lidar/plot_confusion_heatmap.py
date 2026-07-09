#!/usr/bin/env python3
"""Three-panel class-confusion on val object_detection_all_category (row-normalized %):
native PointPillars, the full pipeline (+angle), and their per-cell difference
(pipeline - PP, percentage points). Reads confusion_pp.csv and confusion_refer*.csv
(cols: pred, 10 classes, bg)."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PP_CSV = sys.argv[1] if len(sys.argv) > 1 else '/data/outputs/confusion_pp.csv'
REF_CSV = sys.argv[2] if len(sys.argv) > 2 else '/data/outputs/confusion_refer.csv'
OUT = sys.argv[3] if len(sys.argv) > 3 else '/data/outputs/thesis_figures/FINAL_for_thesis/class_confusion.png'
SHORT = ['car', 'truck', 'bus', 'trail', 'const', 'ped', 'moto', 'bike', 'cone', 'barr', 'bg']


def load(path):
    rows = list(csv.reader(open(path)))[1:]
    return np.array([[float(v) for v in r[1:]] for r in rows])   # (10, 11)


def round_rows_to_100(M):
    """Round each row to integer percentages summing to exactly 100
    (largest-remainder / Hamilton). This guarantees the displayed cells of a
    row add up to 100 and that the difference panel equals the cell-by-cell
    difference of the two rounded panels (so each diff row sums to 0)."""
    M = np.asarray(M, dtype=float)
    out = np.floor(M).astype(int)
    for i in range(M.shape[0]):
        target = int(round(M[i].sum()))            # each row is a distribution -> 100
        deficit = target - int(out[i].sum())
        rem = M[i] - np.floor(M[i])
        if deficit > 0:                            # give +1 to the largest remainders
            for k in np.argsort(-rem)[:deficit]:
                out[i, k] += 1
        elif deficit < 0:                          # take -1 from the smallest remainders
            for k in np.argsort(rem)[:int(-deficit)]:
                out[i, k] -= 1
    return out


pp, ref = round_rows_to_100(load(PP_CSV)), round_rows_to_100(load(REF_CSV))
delta = ref - pp   # integer panels -> diff = cell-by-cell difference, rows sum to 0

fig, axes = plt.subplots(1, 3, figsize=(23, 6.6))

# ---- panels 0,1: sequential Blues (0..100) ----
for ax, M, title in [(axes[0], pp, 'PointPillars (native detector)'),
                     (axes[1], ref, 'Full pipeline (final, +angle +NMS)')]:
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(11)); ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(10)); ax.set_yticklabels(SHORT[:10], fontsize=9)
    ax.set_xlabel('true class  (bg = no object within 2 m)')
    if ax is axes[0]:
        ax.set_ylabel('asserted class')
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i in range(10):
        for j in range(11):
            v = M[i, j]
            if v >= 1.0:
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7.5,
                        color='white' if v >= 50 else 'black')
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor='#d62728', lw=1.6))
cb1 = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
cb1.set_label('% of asserted-class boxes')

# ---- panel 2: difference (pipeline - PP), diverging, centred at 0 ----
vmax = max(5.0, np.ceil(np.abs(delta).max() / 5.0) * 5.0)
axd = axes[2]
imd = axd.imshow(delta, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
axd.set_xticks(range(11)); axd.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
axd.set_yticks(range(10)); axd.set_yticklabels(SHORT[:10], fontsize=9)
axd.set_xlabel('true class  (bg = no object within 2 m)')
axd.set_title('Difference  (full pipeline − PointPillars, pp)', fontsize=12, fontweight='bold')
for i in range(10):
    for j in range(11):
        v = delta[i, j]
        if abs(v) >= 1.0:
            axd.text(j, i, f'{v:+.0f}', ha='center', va='center', fontsize=7.5,
                     color='white' if abs(v) >= 0.65 * vmax else 'black')
    axd.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor='#111111', lw=1.6))
cb2 = fig.colorbar(imd, ax=axd, fraction=0.045, pad=0.02)
cb2.set_label('Δ percentage points  (red: higher in pipeline)')

fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'[ok] {OUT}')
