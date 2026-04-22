# Pipeline

End-to-end view of how a raw HuggingFace download becomes the numbers in
[`overview`](overview.md) and the figures in
[`../notebooks/analysis.ipynb`](../notebooks/analysis.ipynb). This page
explains *why* each step exists. For a module-level tour of the code, see
[`code-map`](code-map.md). For the commands, see [`reproduce`](reproduce.md).

## Shape of the flow

```
raw HF download
     v
dedupe (same-label collapse, conflicting-label drop)
     v
stratified 80/10/10 split (seeded, committed)
     v
per-model LR sweep on val (seed 13, 3 LRs x 3 models = 9 runs)
     v
10-seed ablation on test (3 models x 10 seeds = 30 runs, per-model LR)
     v
pooled paired bootstrap (macro-F1 + PR-AUC, 1000 iters, 95% CI)
     v
pairwise_gaps.csv  +  figures from analysis.ipynb
```

Steps 1-4 happen in Colab (GPU-bound). Steps 5-6 run locally in seconds from
committed artifacts.

## 1. Raw download

`pv_ade.data.load_ade_corpus()` wraps a single `load_dataset(...)` call. No
preprocessing beyond what HuggingFace ships. The function exists so the rest
of the pipeline can depend on a stable signature rather than an inline call.

## 2. Dedupe

The corpus has exact-text duplicates and could in principle have conflicting
labels on the same text. Either would bias the test set: same-label duplicates
silently reweight loss; conflicting labels cap achievable accuracy below 100%.
The [`dataset`](dataset.md) page has the full policy and the actual counts
(2,621 same-label rows dropped, 0 conflicting groups on this release).

Why do it before splitting: if we deduped per split, the same sentence could
land in both train and test, which would leak trivially.

## 3. Stratified split

One 80/10/10 split, `split_seed = 42`, stratified on label, committed to
`data/splits/` as integer indices into the original HF dataset. Indices are
sorted within each split so git diffs are stable.

Why commit the indices instead of the sentences: the indices are tiny, the
dataset is large, and committing the mapping lets every Colab runtime
reconstruct the exact same train/val/test without redownloading a different
corpus version. Every model in the ablation sees identical data.

Why one split and not cross-validation: the ablation is cheap enough that a
single held-out test is defensible, and the paired bootstrap on pooled
predictions (step 5) quantifies the test-sampling noise directly. CV would
have conflated training-run noise with sampling noise.

## 4a. Per-model LR sweep

For each model x each candidate LR in `{2e-5, 5e-5, 1e-4}` the Colab notebook
runs one training run at `seeds[0] = 13`, evaluates on val, and picks the LR
with the highest val macro-F1. Winners are written back to
`configs/ablation.yaml` and checked in.

Why per-model rather than one shared LR: different pretraining backbones have
different sensitivities to learning rate. Using a single LR tuned on one model
would silently penalize the others and bias the between-model comparison. The
sweep pinned `bert-base` at 1e-4 and both biomedical models at 5e-5 — a real
gap, not a rounding difference.

Why one seed: the sweep is a cheap hyperparameter check, not a claim about
which LR is best in expectation. Nine runs (3 x 3) is the budget; adding
seeds would multiply it without meaningfully tightening the winner selection.
The honest framing is "picked a reasonable LR per model on a single val
split" — not "extensively tuned."

Why only three LRs: they span the conventional fine-tuning range for
BERT-family models. A wider grid would burn Colab compute without changing
the winner.

## 4b. 10-seed ablation

For each (model, seed) in `{bert-base, biobert, pubmedbert} x {10 seeds}`,
train with the model's sweep-winner LR and evaluate on **test**. Each run
writes:

- `results/metrics/<model>_seed<S>.json` — config + all metrics for that run
- `results/predictions/<model>_seed<S>.npz` — `y_true`, `y_pred`, `y_prob_pos`

Why ten seeds: enough to see cross-seed std shape (~0.003-0.007 macro-F1 on
this task) and make pooled-prediction bootstrap CIs meaningful, without
running out of Colab CU. Ten is the floor for this kind of ablation, not a
thorough sampling — a larger replication would tighten every CI.

Why save raw predictions, not just metrics: the bootstrap and any future
threshold-tuning or calibration analysis needs per-example probabilities.
Losing them would mean retraining to re-evaluate, which defeats the point of
committing the evidence.

## 5. Pooled paired bootstrap

Per-model predictions are averaged across seeds into one pooled probability
vector per model; labels are thresholded at 0.5 (argmax). For each model pair
`(a, b)` the bootstrap resamples test indices with replacement 1,000 times
and computes `metric(a) - metric(b)` on each resample, reporting the 2.5/97.5
percentiles as the 95% CI.

Why pool across seeds: this isolates test-set sampling noise. Seed noise is
captured separately by the per-model mean-plus-std in the analysis notebook.
The two sources of variance answer different questions and should not be
collapsed together.

Why paired bootstrap: the two models are evaluated on the same test indices
each iteration, so the resample cancels out "this iteration happened to be
an easy test fold" as a confound. Unpaired would waste most of the test-set
sampling variance on noise.

Why 1,000 iterations: enough to stabilize quantiles at 2.5/97.5 without
burning time on what is already a cheap computation. Committed CIs are
deterministic given `seed=42` on the bootstrap RNG.

Why macro-F1 and PR-AUC: macro-F1 is the headline (accuracy on minority
class weighted equally with majority); PR-AUC is threshold-independent and
answers "is the probability ordering any good" separately from "is the 0.5
cut a good decision." Both are reported so an improvement in ranking doesn't
get conflated with an improvement in calibration at the default cut.

The committed values in
[`results/pairwise_gaps.csv`](../results/pairwise_gaps.csv) are regenerated
by [`scripts/compute_pairwise_gaps.py`](../scripts/compute_pairwise_gaps.py).

## 6. Figures and threshold sensitivity

`notebooks/analysis.ipynb` loads the committed run outputs and produces every
figure in the write-up: seed-variance scatter, pairwise gap forest plot, PR
curves, and a tuned-threshold sensitivity table.

The threshold-tuning pass (pick the per-model threshold that maximizes val
macro-F1 on the sweep-winner val run) is deliberately scoped as a
**sensitivity check**, not a headline result. Tuned thresholds come from a
single val run (seed 13); applying them to pooled (seed-averaged) test
probabilities introduces an interaction between pooling and the val cutpoint
that the per-seed evaluation does not exhibit. The argmax-0.5 bootstrap is
the defensible headline; the tuned-threshold pass shows how much the 0.5
choice moves the picture.

## Where each step runs

| Step | Runtime | GPU? |
|---|---|---|
| Load + dedupe + split | Colab (on first run; split then committed) | no |
| LR sweep | Colab | yes |
| 10-seed ablation | Colab | yes |
| Bootstrap / `pairwise_gaps.csv` | Local | no |
| Figures / analysis notebook | Local | no |

After training finishes, all downstream analysis runs locally in seconds from
`results/`. That's intentional: training expense should be paid once and
cached; analysis iteration should be free.
