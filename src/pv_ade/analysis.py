"""Results aggregation and figure generation. Runs locally, not in Colab."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
from sklearn.metrics import f1_score

from pv_ade.evaluate import RANKING_METRICS, bootstrap_gap_ci, classification_metrics

if TYPE_CHECKING:
    import pandas as pd


def load_run_results(metrics_dir: Path) -> "pd.DataFrame":
    """Aggregate per-run JSON files into a tidy DataFrame.

    Columns: model, checkpoint, seed, eval_split, plus one column per metric.
    """
    import pandas as pd

    rows: list[dict] = []
    for path in sorted(metrics_dir.glob("*.json")):
        record = json.loads(path.read_text())
        rows.append(
            {
                "model": record["model"],
                "checkpoint": record["checkpoint"],
                "seed": record["seed"],
                "eval_split": record["eval_split"],
                **record["metrics"],
            }
        )
    return pd.DataFrame(rows)


def summarize_by_model(
    results: "pd.DataFrame",
    metric: str = "macro_f1",
) -> "pd.DataFrame":
    """Mean ± std of `metric` per model across seeds."""
    summary = results.groupby("model")[metric].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_mean", f"{metric}_std", "n_seeds"]
    return summary.reset_index()


def pool_predictions(
    predictions_dir: Path,
    model: str,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool predictions across seeds for a model by averaging predicted probabilities.

    Pooling (rather than picking one seed) keeps the bootstrap focused on
    test-set sampling noise; seed noise is captured separately by summarize_by_model.
    The returned labels are thresholded at ``threshold`` (default 0.5 = argmax);
    pass a per-model tuned threshold when recomputing threshold metrics downstream.
    """
    files = sorted(predictions_dir.glob(f"{model}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no prediction files for model {model!r} in {predictions_dir}")
    y_true: np.ndarray | None = None
    probs = []
    for f in files:
        npz = np.load(f)
        if y_true is None:
            y_true = npz["y_true"]
        elif not np.array_equal(y_true, npz["y_true"]):
            raise ValueError(f"y_true mismatch across seeds for {model}")
        probs.append(npz["y_prob_pos"])
    pooled_prob = np.mean(np.stack(probs), axis=0)
    pooled_pred = (pooled_prob >= threshold).astype(int)
    assert y_true is not None
    return y_true, pooled_pred, pooled_prob


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "macro_f1",
    grid: np.ndarray | None = None,
) -> float:
    """Pick the threshold that maximizes ``metric`` on ``(y_true, y_prob)``.

    ``grid`` defaults to ``np.arange(0.05, 0.96, 0.01)`` — fine enough that the
    picked threshold is stable to a hundredth, coarse enough that the sweep is
    cheap. Ties are broken toward the threshold closest to 0.5 so the tuned
    value is closer to argmax when the metric is flat near the top.
    """
    if metric != "macro_f1":
        raise NotImplementedError(
            f"tune_threshold only supports macro_f1 for now, got {metric!r}"
        )
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)

    scores = np.array(
        [f1_score(y_true, (y_prob >= t).astype(int), average="macro", zero_division=0) for t in grid]
    )
    best_score = float(scores.max())
    # Among thresholds within 1e-9 of the best score, pick the one closest to 0.5.
    best_mask = scores >= best_score - 1e-9
    best_candidates = grid[best_mask]
    closest_idx = int(np.argmin(np.abs(best_candidates - 0.5)))
    return float(best_candidates[closest_idx])


def per_model_val_thresholds(
    sweep_predictions_dir: Path,
    winner_lr_by_model: Mapping[str, float],
    sweep_seed: int,
    metric: str = "macro_f1",
) -> dict[str, float]:
    """Tune one threshold per model using the sweep-winner val predictions.

    Reads ``{sweep_predictions_dir}/{model}_lr{lr}_seed{sweep_seed}.npz`` for each
    model and returns ``{model: tuned_threshold}``. The LR in the filename matches
    Python's default ``f"{lr}"`` formatting (e.g. ``1e-4`` → ``0.0001``, ``5e-5`` →
    ``5e-05``) — which is what the sweep writer used. The sweep val set was the
    vehicle for picking the LR in the first place, so re-using it for threshold
    selection is consistent — both are hyperparameters chosen before touching test.
    """
    thresholds: dict[str, float] = {}
    for model, lr in winner_lr_by_model.items():
        path = sweep_predictions_dir / f"{model}_lr{float(lr)}_seed{sweep_seed}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"sweep val predictions not found for {model} at lr={lr}: {path}"
            )
        npz = np.load(path)
        thresholds[model] = tune_threshold(npz["y_true"], npz["y_prob_pos"], metric=metric)
    return thresholds


def per_seed_metrics_at_threshold(
    predictions_dir: Path,
    thresholds_by_model: Mapping[str, float],
) -> "pd.DataFrame":
    """Recompute per-(model, seed) classification metrics at a tuned threshold.

    ``pr_auc`` is threshold-independent and reported from the same probabilities
    for consistency with the argmax table. Returns one row per (model, seed).
    """
    import pandas as pd

    rows: list[dict] = []
    for model, threshold in thresholds_by_model.items():
        files = sorted(predictions_dir.glob(f"{model}_seed*.npz"))
        if not files:
            raise FileNotFoundError(
                f"no prediction files for model {model!r} in {predictions_dir}"
            )
        for f in files:
            npz = np.load(f)
            y_true = npz["y_true"]
            y_prob = npz["y_prob_pos"]
            y_pred = (y_prob >= threshold).astype(int)
            metrics = classification_metrics(y_true, y_pred, y_prob)
            seed = int(f.stem.rsplit("_seed", 1)[-1])
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "threshold": float(threshold),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)


def pairwise_gaps(
    results: "pd.DataFrame",
    predictions_dir: Path,
    metric: str = "macro_f1",
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    thresholds_by_model: Mapping[str, float] | None = None,
) -> "pd.DataFrame":
    """Pairwise model gaps with bootstrap CIs on pooled test predictions.

    Pools predictions per model by averaging positive-class probabilities across
    seeds, then runs a paired bootstrap on the test indices. Threshold metrics
    use the thresholded pooled label; ranking metrics use the pooled probability
    directly. Model pairs are emitted in alphabetical order (``model_a`` <
    ``model_b``), and the reported gap is ``metric(a) - metric(b)``.

    Pass ``thresholds_by_model`` to threshold each model's pooled probability at
    its tuned val threshold. When omitted, pools are thresholded at 0.5 (argmax).
    Ignored for ranking metrics (e.g. ``pr_auc``), which use probabilities directly.
    """
    import pandas as pd

    models = sorted(results["model"].unique())
    if thresholds_by_model is None:
        pooled = {m: pool_predictions(predictions_dir, m) for m in models}
    else:
        missing = [m for m in models if m not in thresholds_by_model]
        if missing:
            raise ValueError(
                f"thresholds_by_model missing entries for: {missing}"
            )
        pooled = {
            m: pool_predictions(predictions_dir, m, threshold=thresholds_by_model[m])
            for m in models
        }

    rows: list[dict] = []
    for a, b in combinations(models, 2):
        y_true_a, pred_a, prob_a = pooled[a]
        y_true_b, pred_b, prob_b = pooled[b]
        if not np.array_equal(y_true_a, y_true_b):
            raise ValueError(f"y_true differs between {a} and {b}")
        if metric in RANKING_METRICS:
            input_a, input_b = prob_a, prob_b
        else:
            input_a, input_b = pred_a, pred_b
        point, lo, hi = bootstrap_gap_ci(
            y_true_a,
            input_a,
            input_b,
            n_iter=n_iter,
            ci=ci,
            metric=metric,
            seed=seed,
        )
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                f"gap_{metric}": point,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)
