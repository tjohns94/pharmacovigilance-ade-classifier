"""Data loading and splitting for ADE Corpus v2.

The split is generated once, seeded, and committed to the repo so every model
in the ablation sees identical train/val/test.

Deduplication (applied before splitting):

- Same-text duplicates with matching labels: keep the first occurrence.
- Same-text duplicates with conflicting labels: drop all copies. No model can
  predict both labels for the same input, so leaving them in the test set
  would cap metrics at the conflict rate rather than measuring model quality.

Single-instance label noise is preserved — that reflects real annotation
imperfection and is part of what a deployed model has to handle. We only
resolve the specific pathology of "same text, contradictory truth".
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from datasets import Dataset


def load_ade_corpus() -> "Dataset":
    """Load ADE Corpus v2 classification split from HuggingFace.

    Returns a single Dataset with columns ('text', 'label'). The dataset only
    ships a 'train' split — splitting into train/val/test is generate_splits's job.

    Source: Gurulingappa et al. (2012), "Development of a benchmark corpus to
    support the automatic extraction of drug-related adverse effects from
    medical case reports." Distributed via HuggingFace Datasets
    (`ade_corpus_v2`). Licensing terms for the corpus are set by the dataset
    provider; check the HuggingFace dataset page before redistribution.
    """
    from datasets import load_dataset

    return load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification", split="train")


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Normalize for duplicate detection: lowercase, strip, collapse internal whitespace."""
    return _WHITESPACE_RE.sub(" ", text.lower().strip())


@dataclass(frozen=True)
class DedupeAudit:
    """Counts from the pre-split dedupe pass.

    All fields count rows (text-label pairs) in the *original* dataset.

    - ``n_total_original``: dataset size before any dedupe.
    - ``n_unique_normalized``: distinct normalized-text keys.
    - ``n_same_label_dupes_dropped``: same-text, same-label rows removed after
      keeping one per group.
    - ``n_conflicting_groups``: number of normalized-text groups whose rows
      disagreed on label.
    - ``n_conflicting_instances_dropped``: total rows across those groups.
    - ``n_after_dedupe``: rows surviving to the split step.
    """

    n_total_original: int
    n_unique_normalized: int
    n_same_label_dupes_dropped: int
    n_conflicting_groups: int
    n_conflicting_instances_dropped: int
    n_after_dedupe: int


def dedupe_and_audit(
    texts: list[str],
    labels: list[int],
) -> tuple[list[int], DedupeAudit]:
    """Dedupe by normalized text. Returns (kept indices into originals, audit).

    Policy: same-label duplicates collapse to the first occurrence;
    conflicting-label duplicates have all copies dropped.
    """
    if len(texts) != len(labels):
        raise ValueError(f"len(texts) ({len(texts)}) != len(labels) ({len(labels)})")

    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, (text, label) in enumerate(zip(texts, labels)):
        groups[_normalize_text(text)].append((i, label))

    kept: list[int] = []
    same_label_dropped = 0
    conflicting_groups = 0
    conflicting_instances_dropped = 0

    for instances in groups.values():
        if len(instances) == 1:
            kept.append(instances[0][0])
            continue

        labels_in_group = {lbl for _, lbl in instances}
        if len(labels_in_group) > 1:
            conflicting_groups += 1
            conflicting_instances_dropped += len(instances)
        else:
            kept.append(instances[0][0])
            same_label_dropped += len(instances) - 1

    audit = DedupeAudit(
        n_total_original=len(texts),
        n_unique_normalized=len(groups),
        n_same_label_dupes_dropped=same_label_dropped,
        n_conflicting_groups=conflicting_groups,
        n_conflicting_instances_dropped=conflicting_instances_dropped,
        n_after_dedupe=len(kept),
    )
    return sorted(kept), audit


def generate_splits(
    dataset: "Dataset",
    seed: int,
    ratios: tuple[float, float, float],
    out_path: Path,
    data_card_path: Path | None = None,
) -> dict[str, list[int]]:
    """Dedupe, then stratified-split and persist train/val/test indices to disk.

    Indices in the returned splits point into the *original* dataset; rows
    removed by dedupe are simply excluded from every split. Stratified on
    label. Indices sorted within each split for stable diffs.

    A data-card JSON is also written — at ``data_card_path`` if provided,
    otherwise alongside the splits file as ``<stem>.datacard.json``. The card
    contains dedupe audit stats plus per-split sizes and class balance.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1; got {ratios} -> {sum(ratios)}")
    train_r, val_r, test_r = ratios

    texts = list(dataset["text"])
    labels = list(dataset["label"])

    kept_indices, audit = dedupe_and_audit(texts, labels)
    kept_labels = [labels[i] for i in kept_indices]

    train_idx, holdout_idx, _, holdout_labels = train_test_split(
        kept_indices,
        kept_labels,
        test_size=val_r + test_r,
        random_state=seed,
        stratify=kept_labels,
    )
    val_share_within_holdout = val_r / (val_r + test_r)
    val_idx, test_idx, _, _ = train_test_split(
        holdout_idx,
        holdout_labels,
        test_size=1 - val_share_within_holdout,
        random_state=seed,
        stratify=holdout_labels,
    )

    splits = {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(splits, f)

    if data_card_path is None:
        data_card_path = out_path.with_name(f"{out_path.stem}.datacard.json")
    _write_data_card(data_card_path, splits, labels, audit)

    return splits


def _write_data_card(
    out_path: Path,
    splits: dict[str, list[int]],
    labels: list[int],
    audit: DedupeAudit,
) -> None:
    """Write a data-card JSON with dedupe audit plus per-split sizes and class balance."""
    per_split: dict[str, dict[str, float | int]] = {}
    for name, idx in splits.items():
        split_labels = [labels[i] for i in idx]
        n = len(split_labels)
        n_pos = sum(split_labels)
        per_split[name] = {
            "n": n,
            "n_positive": n_pos,
            "n_negative": n - n_pos,
            "positive_rate": n_pos / n if n else 0.0,
        }

    card = {"dedupe_audit": asdict(audit), "splits": per_split}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(card, f, indent=2)


def load_splits(path: Path) -> dict[str, list[int]]:
    """Load committed split indices."""
    with path.open() as f:
        return json.load(f)


def apply_splits(
    dataset: "Dataset",
    splits: dict[str, list[int]],
) -> dict[str, "Dataset"]:
    """Slice a dataset into the train/val/test partitions defined by `splits`."""
    return {name: dataset.select(idx) for name, idx in splits.items()}
