# Reproduce

Exact steps to go from a clean checkout to the figures and CI numbers in
[`overview`](overview.md). Training is Colab-first since that's where the
GPU time lives; everything downstream runs locally in seconds.

Every step reads from `configs/ablation.yaml`. That file is the single source
of truth for the experiment — the split seed, per-model LRs, seed list,
bootstrap iteration count, and primary metric all live there.

## 0. Clone and pick a lane

```bash
git clone https://github.com/tjohns94/pharmacovigilance-ade-classifier.git
cd pharmacovigilance-ade-classifier
```

Two use cases:

- **Just the analysis / figures.** No GPU needed. Install locally and run the
  analysis notebook (section 5). Takes ~10 seconds end to end.
- **Retrain from scratch.** Run the Colab notebook (sections 2-4), download
  `results/`, commit, then rerun the local analysis.

## 1. Local environment (analysis only)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-colab.txt
pip install -e .
```

Python 3.10+ required (see `pyproject.toml`). The `-e .` install puts
`pv_ade` on the path so the notebook and scripts can import it.

Quick sanity check:

```bash
pytest
```

Should pass all smoke tests in `tests/` (data splits, metrics, analysis
aggregation).

## 2. Colab — open the training notebook (GPU required)

Upload or open `notebooks/colab_train.ipynb` in Colab Pro. At the top of the
notebook:

- `SETUP_MODE = "github"` — clone this repo fresh into `/content/portfolio`.
- `SETUP_MODE = "drive"` — use a Drive folder at
  `MyDrive/projects/pharmacovigilance-ade-classifier/`. Pick this if you
  want your local working copy to be the one Colab executes against.

Turn on a GPU runtime (Runtime -> Change runtime type -> GPU). Any T4-class
accelerator is enough; a single epoch on the training split is a few minutes.

Cells 1-2 handle clone + `pip install -q -r requirements-colab.txt` + path
wiring. Expected output: a list of lengths for the three splits after cell 3
once the committed split loads:

```
{'train': 16716, 'val': 2089, 'test': 2090}
```

If the split file is missing (first-ever run) cell 3 regenerates it from the
raw HuggingFace download before returning those same counts.

## 3. Colab — LR sweep (GPU, cell 4)

Runs only when any entry in `training.learning_rate_per_model` in
`configs/ablation.yaml` is null. For a fresh repo with those values already
committed, this cell prints:

```
Sweep already locked - skipping. learning_rate_per_model={...}.
```

To force a re-sweep: null out one or more entries in `ablation.yaml` and
rerun the cell. The sweep trains 3 models x 3 LRs = **9 runs at seed 13**,
evaluates each on val, picks the per-model winner on `evaluation.primary_metric`
(default `macro_f1`), and writes the winners back to `ablation.yaml`. Sweep
artifacts go to `results/sweep/metrics/` and `results/sweep/predictions/`.

Expected winners on the committed split (`split_seed=42`):

| Model | Winner LR |
|---|---|
| `bert-base` | 1.0e-4 |
| `biobert` | 5.0e-5 |
| `pubmedbert` | 5.0e-5 |

The single-seed sweep is a hyperparameter check, not a rigorous HP search.
See [`pipeline`](pipeline.md#4a-per-model-lr-sweep) for the reasoning.

## 4. Colab — 10-seed ablation (GPU, cell 5)

Runs when every LR is set. Trains 3 models x 10 seeds = **30 runs**, each
4 epochs at `batch_size=32`, `max_length=128`, `fp16` on. Seeds are
`[13, 42, 123, 2024, 7777, 1000, 2500, 4242, 8675, 31415]`.

Writes to `results/metrics/<model>_seed<S>.json` (config + metrics) and
`results/predictions/<model>_seed<S>.npz` (y_true, y_pred, y_prob_pos).

Expected wall time: roughly 1.5-2.5 hours on a T4, depending on queuing.
Colab CU budget: target under 10 CU, hard cap 15 CU (see `PLAN.md`).

Cell 6 zips `results/` and triggers a download. Commit the zip contents to
`results/` in your local checkout.

Sanity check after download:

```bash
ls results/metrics   | wc -l   # should be 30
ls results/predictions | wc -l # should be 30
```

## 5. Local — regenerate pairwise_gaps.csv

Run the bootstrap script from the repo root:

```bash
python scripts/compute_pairwise_gaps.py
```

Rewrites `results/pairwise_gaps.csv`. Deterministic given `seed=42` on the
bootstrap RNG (hardcoded in the script) and the committed predictions.
Expected output (printed to stdout and written to CSV) matches the committed
values — six rows, one per `(metric, model_a, model_b)` tuple:

| metric | pair | point_gap | CI lo | CI hi | crosses_zero |
|---|---|---:|---:|---:|---|
| macro_f1 | bert-base vs biobert | -0.0036 | -0.0141 | +0.0070 | True |
| macro_f1 | bert-base vs pubmedbert | -0.0121 | -0.0236 | -0.0025 | False |
| macro_f1 | biobert vs pubmedbert | -0.0085 | -0.0176 | -0.0002 | False |
| pr_auc | bert-base vs biobert | -0.0117 | -0.0202 | -0.0052 | False |
| pr_auc | bert-base vs pubmedbert | -0.0177 | -0.0311 | -0.0067 | False |
| pr_auc | biobert vs pubmedbert | -0.0060 | -0.0165 | +0.0026 | True |

Sign convention: gap is `metric(model_a) - metric(model_b)`, so negative
means `model_b` is ahead. `crosses_zero = True` means the 95% CI includes
zero (not statistically distinguishable).

## 6. Local — regenerate figures

```bash
jupyter lab notebooks/analysis.ipynb
# Kernel -> Restart & Run All
```

Writes three PNGs to `figures/`:

- `seed_variance_macro_f1.png` — per-seed macro-F1 scatter with cross-seed
  means.
- `pairwise_gaps_forest.png` — six-row forest plot of bootstrap CIs.
- `pr_curves.png` — PR curves on pooled predictions.

Expected wall time: under 10 seconds on a modern laptop, no GPU. The
notebook also renders the argmax-vs-tuned-threshold sensitivity table — see
[`pipeline`](pipeline.md#6-figures-and-threshold-sensitivity) for why that's
a sensitivity check, not a headline.

## Summary of what needs a GPU

| Step | GPU? |
|---|---|
| Env setup (section 1) | no |
| Colab clone + split load (section 2) | no |
| LR sweep (section 3) | **yes** |
| 10-seed ablation (section 4) | **yes** |
| Pairwise gaps CSV (section 5) | no |
| Figures (section 6) | no |

If you only need the results, skip to sections 5 and 6 — the committed
artifacts in `results/` let every claim in
[`overview`](overview.md) and the analysis notebook reproduce without
retraining.
