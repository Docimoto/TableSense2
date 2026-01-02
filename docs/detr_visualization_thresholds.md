# DocAI — Training Visualization: Best/Worst + Miss/Extra Buckets (Pixel Coordinates)


- **Boxes are in pixel/sheet coordinates** (not normalized).
- **Targets per sheet:** typically **1–3 tables**.
- We want to start printing after 25 epochs and then dump **best 5%** and **worst 5%**, plus **high miss_rate** and **high extra_rate** examples, with a **cap**.

This document proposes **practical thresholds** and a **repeatable selection policy** that stays stable even when sheet sizes differ.

---

## 1) Always normalize *scoring*, even if we *print* legends in excel sheet coordinates


Let the sheet size be `(W, H)` in excel coordinates.

- **Diagonal:** `D = sqrt(W^2 + H^2)`
- For each matched GT/pred pair:
  - `IoU` (or `GIoU`) computed in pixel space (dimensionless).
  - **Normalized L1:**  
    `nL1 = (|x1p-x1g| + |y1p-y1g| + |x2p-x2g| + |y2p-y2g|) / D`  
    (If you store boxes as `(x1,y1,x2,y2)`.)

This avoids thresholds breaking when one sheet is 1200×800 and another is 5000×3000.

---

## 2) Matching & per-image bookkeeping (DETR-style)

After Hungarian matching, for each image:

- `T = #targets` (GT tables)
- `P = #predictions_considered` (e.g., top-K after score threshold)
- `M = #matched_pairs`
- `U_t = T - M`  (unmatched targets → **misses**)
- `U_p = P - M`  (unused preds → **extras**)

Define:
- `miss_rate = U_t / max(T, 1)`
- `extra_rate = U_p / max(T, 1)`  *(relative to targets; stable when T is small)*

And matched-summary stats:
- `mean_iou = mean(IoU over matched pairs)` (if `M>0`, else 0)
- `p10_iou = 10th percentile IoU over matched pairs` (for “one bad table drags the sheet”)
- `mean_nL1 = mean(nL1 over matched pairs)` (if `M>0`, else large sentinel)

---

## 3) Suggested buckets & thresholds (Great / Bad / Miss / Extra)

These work well for **1–3 targets per image** and pixel coords with normalization as above.

### 3.1 “Great” images (high-quality, stable)
Mark an image as **GREAT** if:

- `miss_rate == 0`
- `extra_rate <= 0.25`  
  *(with T=1, that allows at most 0 extras; with T=2–3, it tolerates 0–1 extra)*
- `mean_iou >= 0.85`
- `p10_iou >= 0.75`
- `mean_nL1 <= 0.020`

**Interpretation:**  
- Mean overlap is very high.
- Even the worst matched table is still good.
- Corners are typically within ~2% of the sheet diagonal in aggregate L1.

> If your tables are extremely regular and you want a stricter bar, move `mean_iou` to **0.90** and `mean_nL1` to **0.015**.

---

### 3.2 “Bad” images (geometrically poor)
Mark an image as **BAD** if **any** of the following is true:

- `miss_rate >= 0.34`  
  *(with T=3, missing 1 table triggers; with T=1, missing 1 table triggers as 1.0)*
- `mean_iou <= 0.35`  *(when M>0)*
- `p10_iou <= 0.20`   *(one matched table is disastrously off)*
- `mean_nL1 >= 0.080` *(corners are far off relative to sheet size)*

**Interpretation:**  
- Either you missed at least one table (common failure mode),
- or overlaps are poor,
- or at least one match is catastrophically wrong,
- or boxes are far off in normalized pixel distance.

---

### 3.3 “High miss_rate” images (model failed to find GT tables)
You’ll get the most actionable insights here.

Mark as **HIGH_MISS** if:

- `miss_rate >= 0.50`  
  *(missed ≥ half of GT tables)*
- OR `U_t >= 1` **and** `mean_iou < 0.60`  
  *(misses + weak remaining matches)*

Recommended sampling priority:
1. `miss_rate` descending
2. then `mean_iou` ascending
3. then `mean_nL1` descending

---

### 3.4 “High extra_rate” images (hallucinated tables / duplicates)
Mark as **HIGH_EXTRA** if:

- `extra_rate >= 1.0`  
  *(extras >= targets; with T=1, that means ≥1 extra)*
- OR `U_p >= 2`  
  *(regardless of T)*

Recommended sampling priority:
1. `extra_rate` descending
2. then `U_p` descending
3. then `mean_iou` ascending (extras + poor geometry is especially bad)

---

## 4) Ranking score for best 5% / worst 5%

Even with thresholds, the most useful workflow is **ranking** and taking the tails.

A simple **badness** score (lower is better):

```
badness =
  (1 - mean_iou) +
  0.70 * miss_rate +
  0.35 * min(extra_rate, 2.0) +
  0.25 * min(mean_nL1 / 0.05, 2.0)
```

Notes:
- Misses are weighted highest (often the most painful).
- Extras matter, but less than misses.
- nL1 contributes, but is capped so it doesn’t overwhelm.

**Best 5%:** lowest `badness`  
**Worst 5%:** highest `badness`

If you prefer IoU-centric ranking, increase `(1 - mean_iou)` weight and reduce `mean_nL1` weight.

---

## 5) Sampling policy with caps (recommended)

Per epoch (after your chosen start epoch), select up to **N_total** images. Example:

- `N_total = 80` per epoch
  - `N_best = 20`
  - `N_worst = 20`
  - `N_high_miss = 20`
  - `N_high_extra = 20`

If a bucket has fewer candidates, **spill over** into the worst bucket (or the next-most-informative bucket).

Also strongly recommended:
- **Deduplicate** so the same image doesn’t appear in multiple buckets unless it’s truly informative.
- Track a small **“recently printed” cache** (e.g., last 3 epochs) to avoid printing the same sheet repeatedly.

---

## 6) Practical defaults for “when to start printing”

Common good defaults:
- Start at `epoch >= 10` **or** after you see the first clear drop in validation loss.
- Then print every epoch **or** every `k=2` epochs depending on disk usage.

If you want a more adaptive trigger:
- Start when the EMA of `val_mean_iou` improves by less than `Δ=0.01` for 2 consecutive epochs.

---

## 7) Visualization tips that make debugging faster

For each saved overlay image, write a small caption block in the corner:

- `epoch`, `global_step`, `image_id`
- `T,P,M`, `miss_rate`, `extra_rate`
- `mean_iou`, `p10_iou`, `mean_nL1`
- top-3 matched pairs: `IoU` + `nL1` each

Color suggestion:
- GT boxes: green
- Pred boxes: red dashed lines
- Matched overlap: optional yellow fill or hatch
- Unmatched GT: blue outline (misses)
- Unused preds: orange outline (extras)

---


