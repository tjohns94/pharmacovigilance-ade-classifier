"""Evaluation metrics and bootstrap CI helpers.

Seed variance (mean ± std across seeds) and bootstrap CIs on pooled test
predictions answer different questions — keep them separate in the analysis.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict[str, float]:
    """Accuracy, per-class precision/recall/F1, macro-F1, PR-AUC (if probs given).

    PR-AUC uses the positive (ADE-related) class probability.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    p_per, r_per, f_per, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=[0, 1], zero_division=0
    )
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "precision_neg": float(p_per[0]),
        "recall_neg": float(r_per[0]),
        "f1_neg": float(f_per[0]),
        "precision_pos": float(p_per[1]),
        "recall_pos": float(r_per[1]),
        "f1_pos": float(f_per[1]),
    }
    if y_prob is not None:
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, np.asarray(y_prob)))
    return metrics


_METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "macro_f1": lambda yt, yp: float(f1_score(yt, yp, average="macro", zero_division=0)),
    "accuracy": lambda yt, yp: float(accuracy_score(yt, yp)),
    "f1_pos": lambda yt, yp: float(f1_score(yt, yp, pos_label=1, zero_division=0)),
    "pr_auc": lambda yt, yp: float(average_precision_score(yt, yp)),
}

# Metrics that operate on predicted positive-class probabilities rather than
# thresholded labels. Callers of bootstrap_gap_ci must pass probabilities —
# not labels — as `preds_a`/`preds_b` when `metric` is in this set.
RANKING_METRICS: frozenset[str] = frozenset({"pr_auc"})


def bootstrap_gap_ci(
    y_true: Sequence[int],
    preds_a: Sequence[float],
    preds_b: Sequence[float],
    n_iter: int = 1000,
    ci: float = 0.95,
    metric: str = "macro_f1",
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap on test predictions. Returns (point_estimate, lo, hi).

    Resamples test indices with replacement; for each resample computes
    metric(preds_a) − metric(preds_b). Point estimate is on the full test set.

    For threshold metrics (``macro_f1``, ``accuracy``, ``f1_pos``) pass predicted
    labels. For ranking metrics (``pr_auc``, see :data:`RANKING_METRICS`) pass
    positive-class probabilities. Degenerate single-class resamples are dropped
    before taking quantiles — rare for realistic test sizes but handled defensively
    so the function is safe at small n or extreme class imbalance.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"unknown metric {metric!r}; choose from {list(_METRIC_FNS)}")
    fn = _METRIC_FNS[metric]

    y_true_arr = np.asarray(y_true)
    a = np.asarray(preds_a)
    b = np.asarray(preds_b)
    n = len(y_true_arr)
    if not (len(a) == len(b) == n):
        raise ValueError("y_true, preds_a, preds_b must have the same length")

    point = fn(y_true_arr, a) - fn(y_true_arr, b)

    rng = np.random.default_rng(seed)
    # Ranking metrics (e.g. average_precision_score) raise on single-class
    # resamples and must be filtered up-front. Threshold metrics handle the
    # same case via ``zero_division=0`` inside sklearn, returning a finite
    # value that the ``np.isfinite`` guard below lets through — so the
    # asymmetry in filtering is intentional, not an oversight.
    needs_both_classes = metric in RANKING_METRICS
    gaps: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt_s = y_true_arr[idx]
        if needs_both_classes and len(np.unique(yt_s)) < 2:
            continue
        gap = fn(yt_s, a[idx]) - fn(yt_s, b[idx])
        if np.isfinite(gap):
            gaps.append(gap)

    if not gaps:
        raise RuntimeError(
            f"no valid bootstrap resamples for metric {metric!r} — check n_iter and data"
        )

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(np.asarray(gaps), [alpha, 1 - alpha])
    return float(point), float(lo), float(hi)
