"""Smoke tests for data loading, deduplication, and splitting."""

import json
from pathlib import Path

import pytest
from datasets import Dataset

from pv_ade.data import (
    apply_splits,
    dedupe_and_audit,
    generate_splits,
    load_splits,
)


def _toy_dataset(n: int = 200) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [f"sentence {i}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )


# -- generate_splits ----------------------------------------------------------


def test_generate_splits_writes_disjoint_partitions(tmp_path: Path) -> None:
    ds = _toy_dataset(200)
    out = tmp_path / "splits.json"
    splits = generate_splits(ds, seed=0, ratios=(0.8, 0.1, 0.1), out_path=out)
    assert out.exists()
    assert json.loads(out.read_text()) == splits
    assert sorted(splits.keys()) == ["test", "train", "val"]
    all_idx = sum(splits.values(), [])
    assert sum(len(v) for v in splits.values()) == 200
    assert len(set(all_idx)) == 200


def test_generate_splits_reproducible(tmp_path: Path) -> None:
    ds = _toy_dataset(200)
    a = generate_splits(ds, seed=7, ratios=(0.8, 0.1, 0.1), out_path=tmp_path / "a.json")
    b = generate_splits(ds, seed=7, ratios=(0.8, 0.1, 0.1), out_path=tmp_path / "b.json")
    assert a == b


def test_load_splits_roundtrip(tmp_path: Path) -> None:
    ds = _toy_dataset(100)
    out = tmp_path / "s.json"
    expected = generate_splits(ds, seed=1, ratios=(0.8, 0.1, 0.1), out_path=out)
    assert load_splits(out) == expected


def test_apply_splits_returns_correct_sizes() -> None:
    ds = _toy_dataset(200)
    splits = {
        "train": list(range(0, 160)),
        "val": list(range(160, 180)),
        "test": list(range(180, 200)),
    }
    parts = apply_splits(ds, splits)
    assert len(parts["train"]) == 160
    assert len(parts["val"]) == 20
    assert len(parts["test"]) == 20


def test_generate_splits_emits_data_card(tmp_path: Path) -> None:
    ds = _toy_dataset(200)
    out = tmp_path / "splits.json"
    generate_splits(ds, seed=0, ratios=(0.8, 0.1, 0.1), out_path=out)

    card_path = tmp_path / "splits.datacard.json"
    assert card_path.exists()

    card = json.loads(card_path.read_text())
    assert card["dedupe_audit"]["n_total_original"] == 200
    assert card["dedupe_audit"]["n_after_dedupe"] == 200
    per_split = card["splits"]
    assert set(per_split.keys()) == {"train", "val", "test"}
    assert per_split["train"]["n"] + per_split["val"]["n"] + per_split["test"]["n"] == 200
    # Toy dataset alternates labels — positive rate should be 0.5 across splits.
    for name in ("train", "val", "test"):
        assert per_split[name]["positive_rate"] == pytest.approx(0.5, abs=0.05)


def test_generate_splits_respects_custom_data_card_path(tmp_path: Path) -> None:
    ds = _toy_dataset(100)
    out = tmp_path / "splits.json"
    card = tmp_path / "my_card.json"
    generate_splits(ds, seed=0, ratios=(0.8, 0.1, 0.1), out_path=out, data_card_path=card)
    assert card.exists()
    assert not (tmp_path / "splits.datacard.json").exists()


# -- dedupe_and_audit ---------------------------------------------------------


def test_dedupe_no_duplicates_is_identity() -> None:
    texts = ["a", "b", "c"]
    labels = [0, 1, 0]
    kept, audit = dedupe_and_audit(texts, labels)
    assert kept == [0, 1, 2]
    assert audit.n_total_original == 3
    assert audit.n_unique_normalized == 3
    assert audit.n_same_label_dupes_dropped == 0
    assert audit.n_conflicting_groups == 0
    assert audit.n_conflicting_instances_dropped == 0
    assert audit.n_after_dedupe == 3


def test_dedupe_drops_same_label_duplicates_keeps_first() -> None:
    texts = ["hello world", "Hello  World", "goodbye"]
    labels = [1, 1, 0]
    kept, audit = dedupe_and_audit(texts, labels)
    assert kept == [0, 2]
    assert audit.n_unique_normalized == 2
    assert audit.n_same_label_dupes_dropped == 1
    assert audit.n_conflicting_groups == 0
    assert audit.n_after_dedupe == 2


def test_dedupe_drops_all_copies_of_conflicting_duplicates() -> None:
    texts = ["fever is bad", "fever is bad", "no reaction"]
    labels = [1, 0, 0]
    kept, audit = dedupe_and_audit(texts, labels)
    assert kept == [2]
    assert audit.n_conflicting_groups == 1
    assert audit.n_conflicting_instances_dropped == 2
    assert audit.n_same_label_dupes_dropped == 0
    assert audit.n_after_dedupe == 1


def test_dedupe_mixed_case_and_whitespace_treated_as_duplicate() -> None:
    texts = ["Patient had fever", "patient had fever", "  PATIENT   HAD   FEVER  "]
    labels = [1, 1, 1]
    kept, audit = dedupe_and_audit(texts, labels)
    assert len(kept) == 1
    assert audit.n_same_label_dupes_dropped == 2


def test_dedupe_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="len"):
        dedupe_and_audit(["a", "b"], [0])
