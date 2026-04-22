"""Smoke tests for metric and bootstrap functions."""

import numpy as np

from pv_ade.evaluate import RANKING_METRICS, bootstrap_gap_ci, classification_metrics


def test_classification_metrics_perfect() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    m = classification_metrics(y_true, y_pred, y_prob=[0.1, 0.9, 0.2, 0.8])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["pr_auc"] == 1.0


def test_classification_metrics_all_wrong() -> None:
    m = classification_metrics([0, 1, 0, 1], [1, 0, 1, 0])
    assert m["accuracy"] == 0.0
    assert "pr_auc" not in m


def test_bootstrap_gap_ci_identical_predictions_is_zero() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    preds = rng.integers(0, 2, size=200)
    point, lo, hi = bootstrap_gap_ci(y_true, preds, preds, n_iter=200, seed=0)
    assert point == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_gap_ci_a_better_than_b() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    preds_a = y_true.copy()  # perfect
    preds_b = rng.integers(0, 2, size=500)  # random
    point, lo, hi = bootstrap_gap_ci(y_true, preds_a, preds_b, n_iter=200, seed=0)
    assert point > 0
    assert lo > 0  # CI excludes zero


def test_bootstrap_gap_ci_unknown_metric_raises() -> None:
    try:
        bootstrap_gap_ci([0, 1], [0, 1], [0, 1], metric="not_a_metric")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown metric")


def test_bootstrap_gap_ci_pr_auc_accepts_probabilities() -> None:
    """PR-AUC is a ranking metric and should accept probabilities, not labels."""
    assert "pr_auc" in RANKING_METRICS
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=300)
    # Well-calibrated: prob tracks the label, with a small noise floor.
    probs_good = np.clip(y_true + rng.normal(0, 0.15, size=300), 0.01, 0.99)
    # Random probabilities — should be worse.
    probs_random = rng.random(size=300)
    point, lo, hi = bootstrap_gap_ci(
        y_true, probs_good, probs_random, n_iter=200, metric="pr_auc", seed=0
    )
    assert point > 0
    assert lo > 0  # CI excludes zero


def test_bootstrap_gap_ci_seed_is_reproducible() -> None:
    """Same seed must produce identical CI across calls — silent seed=0 is a bug."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=300)
    preds_a = rng.integers(0, 2, size=300)
    preds_b = rng.integers(0, 2, size=300)
    out1 = bootstrap_gap_ci(y_true, preds_a, preds_b, n_iter=200, seed=7)
    out2 = bootstrap_gap_ci(y_true, preds_a, preds_b, n_iter=200, seed=7)
    out3 = bootstrap_gap_ci(y_true, preds_a, preds_b, n_iter=200, seed=8)
    assert out1 == out2
    assert out1 != out3
