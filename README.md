[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![한국어](https://img.shields.io/badge/lang-한국어-red.svg)](README.ko.md)

# Pharmacovigilance ADE Classifier

A paired-bootstrap ablation study of biomedical-domain pretraining on a binary
adverse-drug-event (ADE) sentence-classification task.

## What this is

This project fine-tunes three transformer encoders — `bert-base-uncased`,
BioBERT, and PubMedBERT — on the classification split of ADE Corpus v2 and
asks two narrow questions. First, does biomedical-domain pretraining
measurably improve downstream performance over a general-domain baseline?
Second, does from-scratch biomedical pretraining (PubMedBERT) improve over
continued pretraining from general-domain weights (BioBERT)? The study is
best read as a methodology demonstration of honest significance testing on a
small corpus, not as a general claim about pretraining recipes.

## Headline result

At argmax threshold 0.5, PubMedBERT beats base BERT on both macro-F1
(gap -0.012, 95% CI [-0.024, -0.003]) and PR-AUC (-0.018, CI [-0.031, -0.007]).
BioBERT improves probability ranking over base BERT (PR-AUC gap -0.012, CI
[-0.020, -0.005]) but its macro-F1 improvement is not distinguishable from
zero. Effect sizes are modest — roughly one percentage point macro-F1 — and
the evaluation is confined to one corpus and one test split. See
[`docs/paper.md`](docs/paper.md) for the full writeup and
[`results/pairwise_gaps.csv`](results/pairwise_gaps.csv) for the raw numbers.

## Quickstart

The fastest path to reproducing a result is to run the analysis notebook on
the committed run outputs. No GPU needed, takes about ten seconds end to end.

```bash
git clone https://github.com/tjohns94/pharmacovigilance-ade-classifier.git
cd pharmacovigilance-ade-classifier
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-colab.txt
pip install -e .
jupyter lab notebooks/analysis.ipynb
```

Requires Python 3.10+. Every figure and table in `paper.md` regenerates from
committed artifacts in `results/` without retraining. To retrain the 30-run
ablation (3 models x 10 seeds) on Colab, see
[`docs/reproduce.md`](docs/reproduce.md).

## Repo layout

```
configs/ablation.yaml     Single source of truth for the experiment
data/splits/              Committed train/val/test indices + data card
notebooks/
  colab_train.ipynb       GPU training runs (Colab)
  analysis.ipynb          Local analysis + figures, seconds to run
scripts/                  Headless bootstrap regeneration
src/pv_ade/               Package: data, model, train, evaluate, analysis
results/                  Per-seed metrics, raw predictions, pairwise gaps
figures/                  Rendered PNGs from analysis.ipynb
tests/                    pytest smoke tests
docs/                     Project documentation
```

## Where to read more

| If you want to | Read |
|---|---|
| Get oriented quickly | [`docs/overview.md`](docs/overview.md) |
| Understand the scientific result | [`docs/paper.md`](docs/paper.md) |
| Run the analysis yourself | [`docs/reproduce.md`](docs/reproduce.md) |
| Understand the data and dedupe policy | [`docs/dataset.md`](docs/dataset.md) |
| See how Colab training and local analysis fit together | [`docs/pipeline.md`](docs/pipeline.md) |
| Find a specific function in `src/` | [`docs/code-map.md`](docs/code-map.md) |
| Read the analysis with figures inline | [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) |

## Author

Tyson Johnson, Department of Computational and Data Sciences,
George Mason University.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgment

Planning, implementation, and drafting were assisted by Anthropic's Claude.
All analysis, interpretations, and conclusions are the author's own.
