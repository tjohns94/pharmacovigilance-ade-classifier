# Code map

Module-by-module tour of the source tree. Use this as an index — jump into
the file when you need the details. For the end-to-end flow these modules
implement, see [`pipeline`](pipeline.md).

## Layout

```
src/pv_ade/
  __init__.py       # version marker
  data.py           # load + dedupe + stratified split
  model.py          # checkpoint -> (model, tokenizer)
  train.py          # single-run training loop + per-run artifact writer
  evaluate.py       # metrics + paired bootstrap CI
  analysis.py       # aggregate runs, pool predictions, pairwise gaps, threshold tuning
scripts/
  compute_pairwise_gaps.py   # refresh results/pairwise_gaps.csv headlessly
```

Tests in `tests/` mirror the module names (`test_data.py`, `test_evaluate.py`,
`test_analysis.py`).

## `pv_ade.data`

One-liner: load ADE Corpus v2, dedupe on normalized text, write a seeded
stratified 80/10/10 split plus a data card.

**Key public surface**

- `load_ade_corpus() -> Dataset` — thin wrapper around
  `load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification", split="train")`.
- `dedupe_and_audit(texts, labels) -> (kept_indices, DedupeAudit)` — the
  same-label-collapse / conflicting-drop policy. Returns kept indices into
  the originals plus a counted audit.
- `DedupeAudit` — frozen dataclass with the six counts persisted in the
  data card (`n_total_original`, `n_unique_normalized`,
  `n_same_label_dupes_dropped`, `n_conflicting_groups`,
  `n_conflicting_instances_dropped`, `n_after_dedupe`).
- `generate_splits(dataset, seed, ratios, out_path, data_card_path=None)` —
  dedupes then stratified-splits; writes both a splits JSON and a data-card
  JSON. Only called when the splits file is missing.
- `load_splits(path) -> dict[str, list[int]]` — read committed splits.
- `apply_splits(dataset, splits) -> dict[str, Dataset]` — slice the raw
  dataset into train/val/test partitions.

**Depends on:** `sklearn.model_selection.train_test_split`, HuggingFace
`datasets` (import deferred).

**Depended on by:** `pv_ade.train` (uses `load_ade_corpus`, `load_splits`,
`apply_splits`), the Colab notebook's split cell (uses `generate_splits`
when the file is missing), `tests/test_data.py`.

## `pv_ade.model`

One-liner: load a pretrained HF checkpoint and wrap it with a
sequence-classification head.

**Key public surface**

- `build_model(checkpoint, num_labels=2) -> (model, tokenizer)` — pairs
  `AutoTokenizer.from_pretrained` with `AutoModelForSequenceClassification.from_pretrained`.

That's it. Each backbone uses its own tokenizer (they are not interchangeable
across checkpoints), but the classification head above the encoder is
identical for all three models in the ablation. Keeping the wrapper trivial
keeps the ablation honest: the only moving part is the pretraining corpus.

**Depends on:** `transformers`.

**Depended on by:** `pv_ade.train`.

## `pv_ade.train`

One-liner: run one (model, seed) fine-tune, evaluate on the named split,
persist per-run metrics + raw predictions.

**Key public surface**

- `TrainConfig` — dataclass holding `max_length`, `batch_size`,
  `learning_rate`, `num_epochs`, `weight_decay`, `warmup_ratio`. Defaults
  are placeholders; `train_one_run` rejects configs where `learning_rate`
  or `num_epochs` is unset.
- `train_one_run(model_name, checkpoint, seed, config, splits_path,
  metrics_dir, predictions_dir, eval_split="test") -> dict` — the
  single-run entrypoint the Colab notebook calls. Loads the corpus, applies
  the committed split, builds the model, fine-tunes with
  `transformers.Trainer`, evaluates on `eval_split`, writes
  `<run_id>.json` to `metrics_dir` and `<run_id>.npz` (with `y_true`,
  `y_pred`, `y_prob_pos`) to `predictions_dir`. `run_id = f"{model_name}_seed{seed}"`.

**Training defaults in the ablation config** (from `configs/ablation.yaml`):
`max_length=128`, `batch_size=32`, `num_epochs=4`, `weight_decay=0.01`,
`warmup_ratio=0.1`, `fp16` enabled when CUDA is available. Per-model LRs
are written back from the sweep: `bert-base 1e-4`, `biobert 5e-5`,
`pubmedbert 5e-5`.

**What gets logged:** the metrics JSON stores `model`, `checkpoint`, `seed`,
full resolved `TrainConfig`, `eval_split`, and every metric from
`classification_metrics` (accuracy, per-class P/R/F1, macro-F1, PR-AUC).
The NPZ stores the three prediction arrays the bootstrap and any future
threshold analysis need.

**Depends on:** `pv_ade.data`, `pv_ade.model`, `pv_ade.evaluate`,
`transformers`, `torch`, `numpy`.

**Depended on by:** the sweep and ablation cells in
`notebooks/colab_train.ipynb`.

## `pv_ade.evaluate`

One-liner: compute classification metrics for one run; run the paired
bootstrap on two prediction arrays.

**Key public surface**

- `classification_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]` —
  accuracy, per-class precision/recall/F1, macro-F1, and (if `y_prob` is
  given) PR-AUC. Zero-division returns 0 so the function never raises on
  degenerate predictions.
- `RANKING_METRICS: frozenset[str]` — `{"pr_auc"}`. Callers must pass
  probabilities (not thresholded labels) when `metric` is in this set.
- `bootstrap_gap_ci(y_true, preds_a, preds_b, n_iter=1000, ci=0.95,
  metric="macro_f1", seed=0) -> (point, lo, hi)` — paired bootstrap on
  test indices; returns point estimate on the full test set plus CI
  endpoints from the 2.5/97.5 percentiles of `metric(a) - metric(b)` across
  resamples. Drops degenerate single-class resamples for ranking metrics;
  threshold metrics rely on sklearn's `zero_division=0` to return a finite
  value.

Supported metrics are listed in `_METRIC_FNS`: `macro_f1`, `accuracy`,
`f1_pos`, `pr_auc`.

**Depends on:** `sklearn.metrics`, `numpy`.

**Depended on by:** `pv_ade.train` (calls `classification_metrics`),
`pv_ade.analysis` (calls both), `tests/test_evaluate.py`.

## `pv_ade.analysis`

One-liner: aggregate committed run outputs into tables, figures, and the
pairwise-gap CSV. Runs locally, no GPU.

**Key public surface**

- `load_run_results(metrics_dir) -> DataFrame` — reads every
  `metrics/*.json` into one long-form DataFrame keyed by `(model, seed)`.
- `summarize_by_model(results, metric="macro_f1") -> DataFrame` — per-model
  mean / std / seed count for a given metric.
- `pool_predictions(predictions_dir, model, threshold=0.5) ->
  (y_true, y_pred, y_prob)` — seed-average the positive-class
  probabilities for one model, threshold to labels. Validates `y_true`
  matches across seeds.
- `tune_threshold(y_true, y_prob, metric="macro_f1", grid=None) -> float` —
  sweeps thresholds on `[0.05, 0.95]` step 0.01, picks the one that
  maximizes macro-F1. Ties broken toward 0.5.
- `per_model_val_thresholds(sweep_predictions_dir, winner_lr_by_model,
  sweep_seed, metric="macro_f1") -> dict[str, float]` — one tuned
  threshold per model from the sweep-winner val predictions.
- `per_seed_metrics_at_threshold(predictions_dir, thresholds_by_model) ->
  DataFrame` — recompute classification metrics per (model, seed) at a
  tuned threshold.
- `pairwise_gaps(results, predictions_dir, metric="macro_f1", n_iter=1000,
  ci=0.95, seed=0, thresholds_by_model=None) -> DataFrame` — the paired
  bootstrap over model pairs, emitted in alphabetical `model_a < model_b`
  order. Gap is `metric(a) - metric(b)`. Ranking metrics use probabilities
  directly; threshold metrics use the pooled thresholded labels.

**Depends on:** `pv_ade.evaluate`, `numpy`, `sklearn.metrics`, `pandas`
(import deferred).

**Depended on by:** `notebooks/analysis.ipynb`,
`scripts/compute_pairwise_gaps.py`, `tests/test_analysis.py`.

## `scripts/compute_pairwise_gaps.py`

One-liner: re-derive `results/pairwise_gaps.csv` from committed metrics +
predictions, without opening the notebook.

Runs the paired bootstrap for both `macro_f1` and `pr_auc`, combines the
two tables into a single long-format CSV with a `crosses_zero` flag
(`ci_lo <= 0 <= ci_hi`), and prints the result. No retraining involved.

**Depends on:** `pv_ade.analysis`, `pandas`.

**Depended on by:** ops workflow after a fresh training pass — the
notebook reproduces the same numbers in-line, but this script is the
headless path.

## Note on what's not a script

The sweep and final ablation are not standalone scripts: they live in cells
4 and 5 of `notebooks/colab_train.ipynb`, which is the Colab-facing entry
point. Figure generation lives in `notebooks/analysis.ipynb`. The only
committed script is the bootstrap regeneration above. If you expect a
`scripts/run_sweep.py` or `scripts/make_figures.py`, look in the notebooks
instead.
