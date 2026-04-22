# Overview

This project is a small, self-contained ablation study: does biomedical-domain
pretraining measurably improve transformer performance on a binary adverse-drug-event
(ADE) sentence-classification task, and — if so — does from-scratch biomedical
pretraining do better than continued pretraining on top of a general-domain base?

Three encoders are fine-tuned on [ADE Corpus v2][corpus]:

- `bert-base-uncased` — general-domain pretraining; the baseline ([Devlin et al., 2019][bert]).
- `BioBERT` — general-domain BERT with continued pretraining on PubMed + PMC ([Lee et al., 2020][biobert]).
- `PubMedBERT` — pretrained from scratch on PubMed abstracts ([Gu et al., 2021][pubmedbert]).

Each model is trained with its own sweep-winner learning rate across ten random
seeds. Pairwise gaps are reported with 95% paired-bootstrap confidence intervals
on pooled test predictions, following the significance-testing protocol from
[Koehn (2004)][koehn].

## Headline result

At argmax threshold 0.5, PubMedBERT beats BERT on both macro-F1
(−0.012, 95% CI [−0.024, −0.003]) and PR-AUC (−0.018, 95% CI [−0.031, −0.007]).
BioBERT improves probability ranking over BERT (PR-AUC gap −0.012, CI
[−0.020, −0.005]) but its macro-F1 improvement is not distinguishable from zero.
Effect sizes are modest (≈1pp macro-F1) and the task is narrow. See
[`paper.md`](paper.md) for the full writeup.

## How to navigate this repo

| If you want to… | Read |
|---|---|
| Understand the scientific result | [`paper.md`](paper.md) |
| Run the analysis yourself | [`reproduce.md`](reproduce.md) |
| Understand what's in the data | [`dataset.md`](dataset.md) |
| See how Colab training and local analysis fit together | [`pipeline.md`](pipeline.md) |
| Find a specific function in `src/` | [`code-map.md`](code-map.md) |
| Read the analysis with figures inline | [`../notebooks/analysis.ipynb`](../notebooks/analysis.ipynb) |

## Repo layout

```
pharmacovigilance-classifier/
├── configs/ablation.yaml             # Single source of truth for the experiment
├── data/splits/                      # Committed train/val/test indices + data card
├── notebooks/
│   ├── colab_train.ipynb             # GPU training runs (Colab)
│   └── analysis.ipynb                # Local analysis + figures, runs in seconds
├── scripts/
│   └── compute_pairwise_gaps.py   # Refresh results/pairwise_gaps.csv
├── src/pv_ade/                       # Package: data / model / train / evaluate / analysis
├── results/                          # Per-run metrics + predictions (committed)
├── figures/                          # Rendered PNGs from analysis.ipynb
├── tests/                            # pytest unit tests
└── docs/                             # You are here
```

## Quickstart (analysis only, no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/analysis.ipynb                # runs in ~10 seconds
```

The notebook regenerates every figure and table in [`paper.md`](paper.md) from
committed run outputs. To re-run training itself, see [`reproduce.md`](reproduce.md).

## Design philosophy

- **Config over flags.** `configs/ablation.yaml` is the only place ablation knobs live; notebooks read it, don't override it.
- **Commit the evidence.** Splits, per-run metrics, and raw predictions are all in `results/`. Every claim in [`paper.md`](paper.md) is reproducible from the committed artifacts without retraining.
- **Separate the two sources of noise.** Seed variance (cross-seed std) and test-set sampling variance (paired bootstrap on pooled predictions) answer different questions; the analysis keeps them apart.

[corpus]: https://doi.org/10.1016/j.jbi.2012.04.008
[bert]: https://arxiv.org/abs/1810.04805
[biobert]: https://arxiv.org/abs/1901.08746
[pubmedbert]: https://arxiv.org/abs/2007.15779
[koehn]: https://aclanthology.org/W04-3250/
