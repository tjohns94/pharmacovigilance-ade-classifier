# Dataset

Everything in the ablation runs on [ADE Corpus v2][corpus] — a sentence-level
benchmark for detecting adverse drug events (ADEs) in medical case reports.
This page covers what's in the corpus, how it's cleaned, and what the
committed split actually looks like.

## Source

- **Name:** ADE Corpus v2, classification subset (`Ade_corpus_v2_classification`).
- **Access:** HuggingFace Datasets — `load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification")`.
- **Citation:** Gurulingappa, H., Rajput, A. M., Roberts, A., Fluck, J.,
  Hofmann-Apitius, M., & Toldo, L. (2012). *Development of a benchmark corpus
  to support the automatic extraction of drug-related adverse effects from
  medical case reports.* Journal of Biomedical Informatics, 45(5), 885-892.
  [doi:10.1016/j.jbi.2012.04.008][corpus].
- **License:** set by the dataset provider. Check the HuggingFace dataset card
  before redistribution.

The corpus only ships a single `train` split; building train/val/test is this
project's responsibility — see [`pipeline`](pipeline.md) for how that plays out
end to end.

## What a row looks like

Two columns, one row per sentence:

| Column | Type | Meaning |
|---|---|---|
| `text` | `str` | A sentence extracted from a medical case report. |
| `label` | `int` | `1` if the sentence describes an ADE, `0` otherwise. |

The positive class is "this sentence asserts an adverse drug event." Class
balance is skewed toward negatives — about one in five sentences is positive
in every split (see the table below).

## Raw size and class balance

The raw HuggingFace `train` split ships **23,516** rows. It is not deduplicated
on the provider side — the same sentence can appear multiple times, sometimes
with the same label and sometimes with disagreeing labels. The project applies
its own dedupe pass before splitting.

## Dedupe policy

Implemented in [`src/pv_ade/data.py`](../src/pv_ade/data.py). Two rules, applied
after normalizing each row's text (lowercase, strip, collapse internal
whitespace) so trivial spacing or casing differences don't dodge the check:

1. **Same text, same label.** Keep the first occurrence; drop the rest.
   Duplicate signal does not help a classifier — it just reweights the loss
   toward whichever sentence happened to be copied more times.
2. **Same text, conflicting labels.** Drop every copy. No model can predict
   two different labels for the same input, so leaving these in the test set
   would cap measurable metrics at the conflict rate rather than the model's
   actual quality.

Single-instance label noise (a lone row that's probably mislabeled) is
preserved. That's real annotation imperfection and is part of what a deployed
model has to handle; the dedupe pass only resolves the specific pathology of
"same text, contradictory truth."

### What dedupe actually removed

Numbers pulled from [`data/splits/ade_corpus_v2_80_10_10.datacard.json`](../data/splits/ade_corpus_v2_80_10_10.datacard.json):

| Counter | Value |
|---|---:|
| `n_total_original` | 23,516 |
| `n_unique_normalized` | 20,895 |
| `n_same_label_dupes_dropped` | 2,621 |
| `n_conflicting_groups` | 0 |
| `n_conflicting_instances_dropped` | 0 |
| `n_after_dedupe` | 20,895 |

Interpretation: this release of ADE Corpus v2 has exact-text duplicates but no
conflicting-label groups. All 2,621 removed rows were redundant same-label
copies. The "drop conflicting groups" rule is still enforced — it's a policy
decision, not a no-op — it just didn't fire on this data.

## Final per-split sizes

Stratified 80/10/10 split on label, `split_seed = 42`, committed to
[`data/splits/ade_corpus_v2_80_10_10.json`](../data/splits/ade_corpus_v2_80_10_10.json)
as integer indices into the *original* (pre-dedupe) HuggingFace dataset.
Dedupe-dropped rows are excluded from every split.

| Split | n | n_positive | n_negative | positive_rate |
|---|---:|---:|---:|---:|
| train | 16,716 | 3,417 | 13,299 | 0.2044 |
| val | 2,089 | 427 | 1,662 | 0.2044 |
| test | 2,090 | 427 | 1,663 | 0.2043 |

Positive rate is constant across splits to four decimal places — the
stratification is doing its job. The test split is ~2.1k sentences, which is
the noise floor the paired bootstrap is quantifying in [`pipeline`](pipeline.md).

## Regenerating the split

Deterministic given the raw corpus and `configs/ablation.yaml`. The Colab
notebook calls `generate_splits(...)` only when the JSON file is missing, so
the committed file is the source of truth in normal operation. See
[`reproduce`](reproduce.md) for the command.

[corpus]: https://doi.org/10.1016/j.jbi.2012.04.008
