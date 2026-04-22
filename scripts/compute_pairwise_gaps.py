"""Compute `results/pairwise_gaps.csv` from committed run outputs.

Runs the paired bootstrap on pooled test predictions for both macro-F1 and
PR-AUC, combines the two metric tables into a single long-format CSV, and
flags pairs whose 95% CI crosses zero. The analysis notebook reproduces the
same numbers — this script exists so you can refresh the CSV headlessly,
without executing the full notebook, after a new training run writes fresh
results/ files. No retraining involved.
"""

from pathlib import Path
import sys
sys.path.insert(0, 'src')

import pandas as pd
from pv_ade.analysis import load_run_results, pairwise_gaps

METRICS_DIR = Path('results/metrics')
PREDICTIONS_DIR = Path('results/predictions')
OUTPUT = Path('results/pairwise_gaps.csv')

results = load_run_results(METRICS_DIR)

combined = []
for metric in ('macro_f1', 'pr_auc'):
    gaps = pairwise_gaps(results, PREDICTIONS_DIR, metric=metric, n_iter=1000, ci=0.95, seed=42)
    gaps = gaps.rename(columns={f'gap_{metric}': 'point_gap'})
    gaps.insert(0, 'metric', metric)
    # True when the 95% CI includes zero — i.e. the pair is not statistically
    # distinguishable. Kept as a column so downstream consumers don't have to
    # re-derive it from ci_lo / ci_hi.
    gaps['crosses_zero'] = (gaps['ci_lo'] <= 0) & (gaps['ci_hi'] >= 0)
    combined.append(gaps)

out = pd.concat(combined, ignore_index=True)
out.to_csv(OUTPUT, index=False)
print(out.to_string(index=False))
