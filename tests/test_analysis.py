"""Tests for the results-aggregation layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pv_ade.analysis import (
    pairwise_gaps,
    per_model_val_thresholds,
    per_seed_metrics_at_threshold,
    tune_threshold,
)


def _write_fixture(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    """Two models × 2 seeds of synthetic predictions on a 400-sample fake test set."""
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, size=n)

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()

    # model_a tracks the label with small noise → higher macro-F1 and PR-AUC.
    # model_b is closer to random → lower.
    rows = []
    for model, noise in [("model_a", 0.15), ("model_b", 0.45)]:
        for seed in (1, 2):
            prob = np.clip(
                y_true + np.random.default_rng(seed).normal(0, noise, size=n), 0.01, 0.99
            )
            pred = (prob >= 0.5).astype(int)
            np.savez_compressed(
                predictions_dir / f"{model}_seed{seed}.npz",
                y_true=y_true,
                y_pred=pred,
                y_prob_pos=prob,
            )
            rows.append(
                {
                    "model": model,
                    "checkpoint": model,
                    "seed": seed,
                    "eval_split": "test",
                    "macro_f1": 0.0,  # placeholder — pairwise_gaps doesn't read these
                    "pr_auc": 0.0,
                }
            )
    return pd.DataFrame(rows), predictions_dir


def test_pairwise_gaps_seed_is_reproducible(tmp_path: Path) -> None:
    """Same seed must give identical CI across calls; different seed must differ."""
    results, preds_dir = _write_fixture(tmp_path)
    a = pairwise_gaps(results, preds_dir, metric="macro_f1", n_iter=200, seed=7)
    b = pairwise_gaps(results, preds_dir, metric="macro_f1", n_iter=200, seed=7)
    c = pairwise_gaps(results, preds_dir, metric="macro_f1", n_iter=200, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_pairwise_gaps_supports_pr_auc(tmp_path: Path) -> None:
    """pr_auc is a ranking metric — pairwise_gaps must route probabilities, not labels."""
    results, preds_dir = _write_fixture(tmp_path)
    gaps = pairwise_gaps(results, preds_dir, metric="pr_auc", n_iter=200, seed=0)
    assert "gap_pr_auc" in gaps.columns
    # model_a was built to rank better than model_b; the gap (a − b) should be positive
    # and distinguishable from zero in this synthetic setup.
    row = gaps.iloc[0]
    assert row["model_a"] == "model_a" and row["model_b"] == "model_b"
    assert row["gap_pr_auc"] > 0
    assert row["ci_lo"] > 0


def test_tune_threshold_prefers_threshold_that_maximizes_metric() -> None:
    """Construct a prob distribution where the optimal threshold is clearly > 0.5."""
    # y_true is 1 for indices 0..9 (positives) and 0 for indices 10..29 (negatives),
    # so prevalence is 1/3 — class-imbalanced, same direction as ADE Corpus.
    y_true = np.array([1] * 10 + [0] * 20)
    # Positives score between 0.6 and 0.8 (confident); negatives between 0.4 and 0.55
    # (close to 0.5). Argmax threshold 0.5 predicts many negatives as positive;
    # raising the threshold fixes that.
    y_prob = np.concatenate([
        np.linspace(0.6, 0.8, 10),   # positives
        np.linspace(0.4, 0.55, 20),  # negatives
    ])
    picked = tune_threshold(y_true, y_prob, metric="macro_f1")
    # The optimal threshold should sit above 0.55 (separating classes fully) and
    # below the lowest positive (0.6).
    assert 0.55 < picked <= 0.6


def test_tune_threshold_ties_break_toward_0_5() -> None:
    """When the metric is flat across many thresholds, prefer one closest to argmax."""
    # Perfectly separable at any threshold in [0.45, 0.65] — metric is flat.
    y_true = np.array([1] * 5 + [0] * 5)
    y_prob = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.4, 0.4])
    picked = tune_threshold(y_true, y_prob)
    # 0.5 is inside the flat region and closest to argmax → must be picked.
    assert abs(picked - 0.5) < 1e-9


def test_per_model_val_thresholds_uses_winner_lr_file(tmp_path: Path) -> None:
    """Must load the file keyed by (model, winner LR, sweep seed) — not any other."""
    sweep_dir = tmp_path / "sweep_preds"
    sweep_dir.mkdir()
    sweep_seed = 13

    # Two candidate LRs for one model; only the winner's file should be read.
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 20 + [0] * 40)
    # Winner: well-calibrated probs. Non-winner: noise (so tune_threshold would pick differently).
    winner_prob = np.concatenate([
        rng.uniform(0.7, 0.95, 20),
        rng.uniform(0.05, 0.3, 40),
    ])
    noise_prob = rng.uniform(0.4, 0.6, 60)

    np.savez_compressed(
        sweep_dir / f"model_a_lr5e-05_seed{sweep_seed}.npz",
        y_true=y_true, y_pred=(winner_prob >= 0.5).astype(int), y_prob_pos=winner_prob,
    )
    np.savez_compressed(
        sweep_dir / f"model_a_lr2e-05_seed{sweep_seed}.npz",
        y_true=y_true, y_pred=(noise_prob >= 0.5).astype(int), y_prob_pos=noise_prob,
    )

    thresholds = per_model_val_thresholds(
        sweep_dir, {"model_a": "5e-05"}, sweep_seed=sweep_seed
    )
    # With well-separated classes, the tuned threshold should sit between the
    # highest negative (~0.3) and lowest positive (~0.7) — roughly near 0.5.
    assert 0.3 < thresholds["model_a"] < 0.7


def test_per_seed_metrics_at_threshold_reapplies_threshold(tmp_path: Path) -> None:
    """At threshold 0.5 vs 0.9, f1_pos must differ for a model whose positives are mid-confidence."""
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    rng = np.random.default_rng(1)

    y_true = np.array([1] * 30 + [0] * 70)
    # Positives with probs mostly in [0.55, 0.85]: threshold 0.9 will miss most.
    y_prob = np.concatenate([
        rng.uniform(0.55, 0.85, 30),
        rng.uniform(0.05, 0.3, 70),
    ])
    y_pred = (y_prob >= 0.5).astype(int)
    np.savez_compressed(
        preds_dir / "model_a_seed1.npz",
        y_true=y_true, y_pred=y_pred, y_prob_pos=y_prob,
    )

    low = per_seed_metrics_at_threshold(preds_dir, {"model_a": 0.5})
    high = per_seed_metrics_at_threshold(preds_dir, {"model_a": 0.9})

    # At threshold 0.9 almost no positives survive → f1_pos collapses.
    assert low["f1_pos"].iloc[0] > 0.7
    assert high["f1_pos"].iloc[0] < low["f1_pos"].iloc[0] - 0.3
    # pr_auc is threshold-independent.
    assert low["pr_auc"].iloc[0] == high["pr_auc"].iloc[0]


def test_pairwise_gaps_accepts_threshold_by_model(tmp_path: Path) -> None:
    """Per-model threshold should change labels and thus the macro_f1 gap."""
    results, preds_dir = _write_fixture(tmp_path)
    default = pairwise_gaps(results, preds_dir, metric="macro_f1", n_iter=100, seed=0)
    tuned = pairwise_gaps(
        results,
        preds_dir,
        metric="macro_f1",
        n_iter=100,
        seed=0,
        thresholds_by_model={"model_a": 0.7, "model_b": 0.3},
    )
    # The two must differ somewhere: changing thresholds changes labels and therefore the gap.
    assert not default.equals(tuned)
