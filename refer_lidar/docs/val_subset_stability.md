# Validation-Subset Stability: is a fixed 20% val set safe for model selection?

**TL;DR.** For early stopping and `checkpoint_best` selection we validate on a
**fixed 20% subset** of the 6,019-frame nuScenes val split (`--val-fraction 0.2`,
`--val-subset-seed 42`), reused identically every epoch and every run. A
bootstrap over 4,000 random 20% draws at the real data scale shows the subset
mean is stable to **~0.1–0.15 val_loss units** — the *same order* as the
epoch-to-epoch wobble already present on the full 6,019-frame val set, and
**10–20× smaller** than the descent signal that early stopping keys off. The
best epoch is a flat basin of near-equivalent checkpoints; a 20% subset picks
inside that basin reliably. Model *ranking* across ablations is decided by a
separate held-out shard evaluation, not this quantity.

---

## 1. What the 20% subset is used for

| Consumer | Uses the 20% subset? | Notes |
|---|---|---|
| Per-epoch `val_loss` (early stopping, patience 3) | **yes** | fixed indices, seed 42 |
| `checkpoint_best.pth` selection | **yes** | argmin val_loss over epochs |
| Ablation **winner** (neg-ratio / arch) | **no** | dedicated held-out shard eval (mAP / TP metrics) |

The subset is drawn **once** with a fixed seed and reused across all epochs and
all runs (`main_simple_ddp.py`, `[Data] val subset: 1204/6019 items
(fraction=0.2, seed=42, fixed across epochs/runs)`). This matters: because the
same items are scored every epoch, the subset's sampling bias is a *constant*
that largely cancels when comparing one epoch to the next.

## 2. Why this needs checking

Early stopping / `checkpoint_best` compare val_loss across epochs. If a 20% draw
were a noisy estimate of the full-val loss, it could (a) stop at the wrong epoch
or (b) select a materially worse checkpoint. We quantify the sampling noise of a
20% aggregate at this data scale and compare it to the real signal it must
resolve.

## 3. Methodology

**Data.** Real per-query records from a held-out evaluation shard
(`eval_seedrot_all_neg/shard_0_of_24.pkl`, 7,711 positive queries), i.e. actual
model predictions vs ground truth, not synthetic data.

**Per-item val-signal proxies.** We do not log per-item Hungarian val_loss, so we
bootstrap two per-query proxies that bracket its behaviour:

- **`recall@2m`** — fraction of a query's GT boxes with a prediction within 2 m
  (bounded [0,1], per-item CV = 29.5%). Representative of a *smooth* aggregate
  loss.
- **`loc-error`** — mean nearest-prediction centre distance per query, capped at
  10 m (heavy-tailed, per-item CV = 184%). A deliberately **pessimistic**
  stress test; far more dispersed than a Hungarian loss.

The subset-mean sampling noise is governed by the *per-item variance* and the
*absolute subset size* (CLT: SE ∝ σ/√n, with a finite-population correction for
sampling without replacement). Any bounded per-item quantity therefore obeys the
same 1/√n shrinkage; the two proxies calibrate the coefficient.

**Bootstrap.** For each proxy we draw **4,000** subsets *without replacement* at
the absolute sizes the training uses — n = 1,204 (frame count) and n = 2,408
(≈ frames × `queries_per_frame` = 2, the effective query-signal count) — and
report the distribution of the subset mean, its coefficient of variation (CV),
and the maximum deviation from the full-population mean. We also compute the
probability that a fixed subset *reverses* the ordering of two epochs whose
full-val losses differ by a gap `g`, modelling the fixed-subset correlation
across epochs (ρ) that cancels shared bias.

**Real-signal calibration.** We read the full 6,019-frame val_loss trajectories
of two completed runs (`seedrot_baseline_20260630`,
`pointpillars_3_...20260530`) to measure the actual epoch-to-epoch differences
the subset must resolve.

Reproduce: `python /data/tmp_valsubset_boot.py` (CPU-only).

## 4. Results

### 4.1 Sampling noise of a 20% draw (4,000 bootstraps)

| proxy | n | subset-mean CV | max dev / 4,000 | ≈ val_loss units (scale ≈ 20) |
|---|---|---|---|---|
| recall@2m (realistic) | 1,204 | **0.77%** | 2.4% | ~0.15 |
| recall@2m (realistic) | 2,408 | **0.49%** | 1.7% | ~0.10 |
| loc-error (pessimistic) | 1,204 | 4.86% | 16.8% | ~1.0 |
| loc-error (pessimistic) | 2,408 | 3.10% | 10.0% | ~0.6 |

At the effective query-signal count (n ≈ 2,408) the realistic proxy puts the 20%
sampling noise at **~0.10 val_loss units**. The heavy-tailed proxy (~0.6) is an
over-estimate because the true Hungarian loss is far smoother than raw
localisation error. Finite-population correction (20% of 6,019, factor √0.8) and
per-frame pre-averaging both push the true figure *below* these numbers.

### 4.2 The signal it must resolve (real full-val curves)

`seedrot_baseline_20260630`, full 6,019-frame val_loss:

```
epoch:   1     4     8     9    10    11    12    13
val:   23.76 21.27 20.88 20.42 20.47 20.47 20.34 20.48
       └──── descent ~2.9 ────┘ └──── plateau basin, wobble ~0.05–0.15 ────┘
```

- **Descent** (epochs 1→8): **~1–3** val_loss units.
- **Plateau wobble** (epochs 8–13, *on the full set*): **~0.1–0.2** units — the
  best epoch is already fuzzy at full resolution.

### 4.3 Epoch-reversal probability

For two epochs separated by a real ≥1%-of-mean gap (≈0.2 val_loss units), a fixed
20% subset (ρ≈0.7) reverses their order with probability **~0.4%** (realistic
proxy, n=2,408). Reversals only become likely for gaps ≲0.1 val_loss — i.e.
*within* the plateau, where the checkpoints are near-identical.

## 5. Interpretation

1. **20% noise ≈ full-val noise.** The subset adds ~0.10–0.15 val_loss units on
   top of a metric that already wobbles ~0.1–0.2 at full size. It does not change
   the qualitative picture: the best epoch is a flat basin, not a sharp peak.
2. **Plateau-onset detection is trivially robust.** Early stopping keys off the
   ~1–3 unit descent, which is 10–20× the sampling noise, so it fires in the
   right region essentially always.
3. **Fixed subset cancels bias across epochs.** Because seed-42 indices are
   reused every epoch, the shared sampling bias subtracts out in epoch-to-epoch
   comparisons, making the *ranking* noise smaller than the raw CV suggests.

**Worst realistic case:** `checkpoint_best` lands ±1–2 epochs inside the flat
basin — a model within ~0.1 val_loss of the true full-val optimum, i.e. within
full-val noise of it.

## 6. Threats to validity

- Proxies stand in for the un-logged per-item Hungarian loss; we mitigate by
  bracketing with a bounded and a heavy-tailed quantity, and the 1/√n argument is
  proxy-agnostic.
- Bootstrap samples query-level units; real val averages over frames
  (≥ as many units, with intra-frame pre-averaging) → true noise ≤ reported.
- Rare classes contribute few val instances; this affects rare-class-sensitive
  *selection*, which is handled by the (larger) ablation shard eval, not the
  early-stopping val_loss.

## 7. Conclusion (for the experimental section)

A fixed 20% validation subset (seed 42) is a statistically sound choice for
early stopping and checkpoint selection at this data scale: its sampling noise
(~0.10–0.15 val_loss units) is dominated by the ~1–3 unit training signal and is
comparable to the residual noise of the full validation set itself. It cuts
per-epoch validation cost ~5× while leaving best-epoch selection inside a basin
of near-equivalent checkpoints. Cross-ablation model ranking is decided
separately on a held-out shard evaluation, so absolute-metric fidelity is never
sourced from the subsampled quantity.
