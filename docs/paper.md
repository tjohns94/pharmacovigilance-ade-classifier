# Biomedical Pretraining on a Binary Adverse-Drug-Event Sentence Classifier: A Paired-Bootstrap Ablation

**Author:** Tyson Johnson  
**Affiliation:** Department of Computational and Data Sciences, George Mason University  
**Status:** Portfolio technical writeup

---

## Abstract

Three transformer encoders — `bert-base-uncased`, BioBERT, and PubMedBERT — are fine-tuned on the binary adverse-drug-event (ADE) sentence-classification split of ADE Corpus v2, and this study tests whether biomedical-domain pretraining is associated with a measurable change in downstream performance on a small, self-contained ablation. Each model is trained across ten random seeds at a per-model learning rate selected by a single-seed validation sweep; pairwise gaps are reported with 95% paired-bootstrap confidence intervals on seed-averaged test predictions, following the protocol of Koehn [1]. At an argmax threshold of 0.5, PubMedBERT shows higher macro-F1 than base BERT (gap −0.012, 95% CI [−0.024, −0.003]) and higher PR-AUC (−0.018, 95% CI [−0.031, −0.007]); BioBERT shows a higher PR-AUC than base BERT (gap −0.012, 95% CI [−0.020, −0.005]) but its macro-F1 gap is not distinguishable from zero. Observed effect sizes are small — of order one percentage point macro-F1 — and the evaluation is confined to one corpus and one test split, so these results are best read as a methodology demonstration rather than a claim about the relative merits of the pretraining recipes in general.

## I. Introduction

Pharmacovigilance is the monitoring of drug safety signals after approval, and free-text case reports, narrative electronic-health-record sections, and clinical-study source documents are an important input to that process. A common first-pass triage task is deciding, at the sentence level, whether a passage describes an adverse drug event — a binary screening decision that filters candidate spans before downstream extraction. The ADE Corpus v2 of Gurulingappa et al. [2] is a widely used open benchmark for this screening step.

Two methodological questions motivate this study. First, is biomedical-domain pretraining associated with a measurable change in downstream classification on a narrow binary ADE task, relative to a general-domain BERT baseline [3]? Second, conditional on biomedical pretraining showing a gap, does *from-scratch* biomedical pretraining — the PubMedBERT recipe of Gu et al. [4] — show an additional gap over *continued* biomedical pretraining initialized from general-domain BERT, as in BioBERT [5]? The second question is of interest because the two recipes impose different inductive biases on vocabulary and representation while converging on the same nominal target corpus (PubMed).

The contribution of this note is not methodological. It is an attempt to answer those two questions on one public corpus, with a replication protocol that keeps training-run noise and test-set sampling noise separate, and with honest confidence intervals that flag effects too small to distinguish from either source of variance. Results are reported exactly as measured; no claim is made about transfer to entity extraction, relation linking, or clinical-document tasks.

## II. Related Work

BERT [3] established the masked-language-modeling pretraining plus task-specific fine-tuning template used throughout this study. BioBERT [5] adapted that template to the biomedical domain by continuing pretraining on PubMed abstracts and PMC full-text articles starting from the released BERT weights, and reported consistent downstream gains on biomedical NER, relation extraction, and QA. PubMedBERT [4] argued that, when a sufficiently large in-domain corpus is available, pretraining from scratch on PubMed — with a domain-specific WordPiece vocabulary — outperforms continued pretraining; the BLURB benchmark results reported in that paper motivate the two-question framing used here. For the statistical-significance protocol, this study follows Koehn [1], whose paired-bootstrap recipe for MT evaluation transfers directly to any paired comparison of systems on a common test set.

## III. Methods

### A. Dataset

The `Ade_corpus_v2_classification` subset of ADE Corpus v2, as distributed through the HuggingFace Datasets hub, is used throughout. The raw corpus contains 23,516 (text, label) rows. A single stratified 80/10/10 split, generated once with seed 42 and committed to the repository, is shared across all models. The committed split uses deduplicated-row indices and yields 16,716 training, 2,089 validation, and 2,090 test instances. The positive-class (ADE-related) rate is 0.2044 in train, 0.2044 in validation, and 0.2043 in test — a 20% positive prevalence that is preserved by stratification.

### B. Preprocessing

Before splitting, a deterministic deduplication pass is applied, keyed on a normalized text form (lowercase, stripped, internal whitespace collapsed). Two cases arise:

1. Same normalized text, matching labels across all copies: collapse to the first occurrence.
2. Same normalized text, *conflicting* labels across copies: drop every copy in the group.

The second policy is the load-bearing one. Leaving label-conflicting duplicates in the test set would cap any classifier's achievable metric at the non-conflict rate rather than measuring model quality, because no classifier can emit two different predictions for the same input. The first policy removes a source of train/test leakage that would otherwise inflate reported numbers.

On ADE Corpus v2 this pass removes 2,621 same-label duplicate rows and zero conflicting-label rows, leaving 20,895 instances to split. Single-instance label noise (sentences that appear once and may be mislabeled) is preserved — that is a real property of the annotation pipeline, and a deployed screener must tolerate it.

Table I summarizes the resulting data card.

**Table I. Dataset after dedupe.**

| Split | N | N positive | N negative | Positive rate |
|---|---:|---:|---:|---:|
| Train | 16,716 | 3,417 | 13,299 | 0.2044 |
| Val | 2,089 | 427 | 1,662 | 0.2044 |
| Test | 2,090 | 427 | 1,663 | 0.2043 |
| **Total after dedupe** | **20,895** | — | — | — |
| Original (pre-dedupe) | 23,516 | — | — | — |

### C. Models

Three encoders are fine-tuned with a shared two-class classification head (a linear layer over the pooled `[CLS]` representation, as supplied by the HuggingFace `AutoModelForSequenceClassification` wrapper):

- **bert-base** — `bert-base-uncased` [3], general-domain pretraining. Baseline.
- **BioBERT** — `dmis-lab/biobert-base-cased-v1.2` [5], continued pretraining on PubMed + PMC initialized from base BERT.
- **PubMedBERT** — `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` [4], from-scratch pretraining on PubMed abstracts and full-text articles with a domain-specific vocabulary.

All three share encoder depth, hidden size, and head architecture; the only intentional differences are pretraining corpus and vocabulary.

### D. Training

All runs use batch size 32, maximum sequence length 128 tokens, weight decay 0.01, warmup ratio 0.1, and four training epochs, with fp16 enabled when a GPU is available. Each model is fine-tuned across ten random seeds: {13, 42, 123, 2024, 7777, 1000, 2500, 4242, 8675, 31415}. Learning rate is set per model from a 3×3 validation sweep (three learning rates — 2e-5, 5e-5, 1e-4 — times three backbones, four epochs each) run at the first seed only; the macro-F1-on-validation winner is then used for all ten final-ablation seeds of that backbone. The winners are 1e-4 for bert-base and 5e-5 for both biomedical models.

Running the sweep at a single seed is a deliberate cost compromise, not a methodological claim. It is flagged explicitly as a limitation in §VI rather than as a strength: a more thorough protocol would repeat the sweep across seeds. In practice the validation differences between the three learning rates per backbone were large enough that single-seed selection picked a plausible winner; the alternative would have been a 90-run (3 models × 3 LRs × 10 seeds) sweep, which is outside the scope of a portfolio ablation.

### E. Evaluation

Each training run writes to disk the per-seed metrics JSON and a raw `(y_true, y_pred, y_prob_positive)` array on the test split, so all downstream analysis runs without retraining. Two sources of variance are reported separately, because they answer different questions:

- **Seed variance** — cross-seed mean and standard deviation of each per-run metric, capturing training-run noise under the shared split.
- **Test-set sampling variance** — paired bootstrap on pooled (seed-averaged) test predictions, 1000 iterations, 95% percentile CI, with the same bootstrap seed across metrics.

Pooling predictions before bootstrapping places the bootstrap on a per-model average prediction vector, which isolates test-set sampling from seed variance. For threshold metrics (macro-F1, accuracy, F1 on the positive class) the resamples use thresholded labels; for the ranking metric (PR-AUC) they use positive-class probabilities. Resamples with only one class present in the resampled `y_true` are dropped for ranking metrics (average-precision is undefined there) and retained for threshold metrics (scikit-learn's `zero_division=0` returns a finite value, which the bootstrap accepts).

The primary protocol is argmax at threshold 0.5 on seed-averaged probabilities. A secondary, per-model threshold tuned on validation macro-F1 (sweep in [0.05, 0.95] at step 0.01, ties broken toward 0.5) is reported as a sensitivity check only, not the headline. The tuned thresholds are picked from a single validation run (the sweep seed), and when applied to seed-averaged test probabilities they interact with the pooling step — a cutpoint tuned on one seed's distribution can land in a different place on the seed-averaged one. This interaction is flagged explicitly in §IV-C rather than hidden.

## IV. Results

### A. Seed-level performance

Table II reports the cross-seed mean of each per-run metric at argmax threshold 0.5, across ten seeds per model. The ranking is consistent across all three metrics shown: PubMedBERT > BioBERT > bert-base. Within-model standard deviations (not tabulated here; in the range of 0.003–0.007 on macro-F1) are small relative to the between-model gaps, but the biobert-vs-pubmedbert gap sits close to that noise floor.

**Table II. Cross-seed mean test metrics, argmax threshold 0.5, 10 seeds per model.**

| Model | macro-F1 | F1 (positive) | PR-AUC |
|---|---:|---:|---:|
| bert-base | 0.9200 | 0.8729 | 0.9432 |
| BioBERT | 0.9327 | 0.8932 | 0.9549 |
| PubMedBERT | 0.9377 | 0.9013 | 0.9608 |

The F1-positive column is computed as a per-model mean across the ten per-run metric JSONs; macro-F1 and PR-AUC values are read from the analysis notebook's summary table, which pools probabilities across seeds before scoring.

### B. Pairwise bootstrap CIs

Table III reports the full pairwise-gaps table — six pairs (three pairs × two metrics) — with 1000-iteration paired-bootstrap 95% CIs on pooled (seed-averaged) test predictions. Gaps are reported as `metric(model_a) − metric(model_b)`; negative gaps indicate model B outperforms model A.

**Table III. Pairwise gaps with 95% paired-bootstrap CIs on pooled test predictions (argmax 0.5).**

| Metric | model_a | model_b | Gap | 95% CI | Excludes 0? |
|---|---|---|---:|---|:---:|
| macro-F1 | bert-base | biobert | −0.0036 | [−0.0141, +0.0070] | no |
| macro-F1 | bert-base | pubmedbert | −0.0121 | [−0.0236, −0.0025] | **yes** |
| macro-F1 | biobert | pubmedbert | −0.0085 | [−0.0176, −0.0002] | **yes** |
| PR-AUC | bert-base | biobert | −0.0117 | [−0.0202, −0.0052] | **yes** |
| PR-AUC | bert-base | pubmedbert | −0.0177 | [−0.0311, −0.0067] | **yes** |
| PR-AUC | biobert | pubmedbert | −0.0060 | [−0.0165, +0.0026] | no |

Four of the six intervals exclude zero. On PR-AUC, both biomedical variants show a gap over bert-base but are not distinguishable from each other. On macro-F1, PubMedBERT is distinguishable from both other models, but BioBERT is not distinguishable from bert-base at the 0.5 cutpoint. The biobert-vs-pubmedbert macro-F1 upper bound sits at −0.0002, i.e., right on the zero line — small but statistically distinguishable, not a decisive effect.

### C. Threshold sensitivity

As a sensitivity check, the macro-F1 paired bootstrap is re-run with per-model tuned thresholds passed through, and separately per-seed macro-F1 is recomputed at those tuned thresholds. The two views disagree, and the disagreement is instructive rather than contradictory.

The per-seed view preserves the ranking from Table II: tuned-threshold mean macro-F1 is approximately 0.920 for bert-base, 0.933 for BioBERT, and 0.936 for PubMedBERT. Threshold tuning does not reshuffle model order and only modestly moves the numbers.

The pooled-bootstrap-with-tuned-threshold view does reshuffle which pairs are statistically distinguishable — `bert-base − biobert` becomes distinguishable (it was not at 0.5) and `bert-base − pubmedbert` becomes indistinguishable (it was at 0.5). This is an interaction artifact: the tuned threshold for PubMedBERT from the validation sweep sits at 0.94, and applying a 0.94 cutpoint to a seed-averaged probability vector discards true positives that any individual seed would have classified correctly at its own tuned cut. The per-seed view does not exhibit this flip because each seed is evaluated against its own non-averaged probability distribution. Both results are reported, but the argmax-0.5 bootstrap is treated as the headline and the pooled-tuned variant as a known-caveat sensitivity check.

## V. Discussion

The results are internally consistent with a simple two-part story. First, on this corpus and this split, biomedical pretraining is associated with better probability ranking on the ADE screening task relative to a general-domain baseline: both BioBERT and PubMedBERT show a PR-AUC gap over bert-base with intervals that exclude zero. Second, from-scratch biomedical pretraining (PubMedBERT) is associated with a further, smaller gap on threshold-based decision quality at the standard 0.5 cutpoint relative to continued pretraining (BioBERT), visible as a distinguishable macro-F1 gap between the two biomedical variants. BioBERT, by contrast, shows the ranking gap over bert-base but not a clear macro-F1 gap at 0.5 — its probability ordering is higher, but the 0.5 decision boundary does not capture that.

This pattern is directionally consistent with the conclusions of Gu et al. [4], with two honest qualifications. Observed effect sizes here are on the order of one percentage point macro-F1, not the larger gaps reported across the full BLURB benchmark. And the evaluation is one binary screening task on one corpus with one test split; nothing here speaks to entity extraction, relation linking, or clinical-document tasks where the balance between continued and from-scratch pretraining could plausibly shift.

## VI. Limitations

Several limitations constrain the reach of these results and are flagged as caveats rather than addressed:

- **Single test split.** The 2,090-row test partition comes from one seeded stratified 80/10/10. A cross-validated or multi-split replication would give a better read on test-sampling variance; the paired bootstrap only captures variance *within* the single test partition used here.
- **Single-seed hyperparameter sweep.** Per-model learning rate is chosen from one validation sweep run at seed 13. A more thorough protocol would repeat the sweep across seeds and pick either the seed-averaged winner or the within-seed winner before committing. The same single-seed caveat applies to the tuned threshold used in §IV-C.
- **Narrow task.** Binary sentence classification is the simplest ADE task available on this corpus. Results should not be extrapolated to span extraction, relation linking, document-level screening, or non-English clinical text without separate validation.
- **Ten seeds is a lower bound.** Ten seeds gave four-of-six distinguishable CIs at this effect size; a larger replication would tighten every interval and might resolve the currently-indistinguishable pairs.
- **No external validation set.** All test performance is measured inside the same corpus. Generalization to other ADE sources (e.g., FDA adverse-event narratives, EHR notes, social-media surveillance feeds) is not evaluated.
- **No error analysis beyond metrics.** Only aggregate scores are reported; no inspection is made of which positive sentences each model misclassifies, which would be the natural next step toward understanding *why* PubMedBERT's cutpoint behavior differs from BioBERT's.
- **Pooled-bootstrap-with-tuned-threshold interaction.** As discussed in §IV-C, applying a validation-tuned threshold on top of seed-averaged test probabilities is not a clean evaluation protocol. It is reported as a sensitivity check, not a headline.

## Acknowledgment

I acknowledge the use of Anthropic's Claude for planning, implementation, and drafting during the completion of this work. All analysis, interpretations, and conclusions remain my own.

## References

[1] P. Koehn, "Statistical significance tests for machine translation evaluation," in *Proc. 2004 Conf. Empirical Methods in Natural Language Processing (EMNLP)*, Barcelona, Spain, Jul. 2004, pp. 388–395.

[2] H. Gurulingappa, A. M. Rajput, A. Roberts, J. Fluck, M. Hofmann-Apitius, and L. Toldo, "Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports," *Journal of Biomedical Informatics*, vol. 45, no. 5, pp. 885–892, Oct. 2012, doi: 10.1016/j.jbi.2012.04.008.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. 2019 Conf. North American Chapter of the Assoc. for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, Minneapolis, MN, USA, Jun. 2019, pp. 4171–4186.

[4] Y. Gu, R. Tinn, H. Cheng, M. Lucas, N. Usuyama, X. Liu, T. Naumann, J. Gao, and H. Poon, "Domain-specific language model pretraining for biomedical natural language processing," *ACM Transactions on Computing for Healthcare*, vol. 3, no. 1, pp. 1–23, Oct. 2021, doi: 10.1145/3458754.

[5] J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, and J. Kang, "BioBERT: a pre-trained biomedical language representation model for biomedical text mining," *Bioinformatics*, vol. 36, no. 4, pp. 1234–1240, Feb. 2020, doi: 10.1093/bioinformatics/btz682.
