"""Metrics and statistical comparison utilities."""
from __future__ import annotations

from typing import Dict, Literal

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats


def compute_prf(y_true, y_pred) -> Dict[str, float]:
    """Compute precision, recall, f1 for binary labels.

    Returns a dict with keys: precision, recall, f1
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def bootstrap_compare(
    y_true,
    y_a,
    y_b,
    metric: Literal["f1", "precision", "recall"] = "f1",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap comparison of two prediction vectors.

    Returns dict with mean_diff (b - a), ci_low, ci_high, p_value, n_boot.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)

    n = len(y_true)
    if not (len(y_a) == n and len(y_b) == n):
        raise ValueError("Input lengths must match")

    def metric_fn(y_t, y_p):
        m = compute_prf(y_t, y_p)
        return m[metric]

    diffs = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        diffs[i] = metric_fn(y_true[sample_idx], y_b[sample_idx]) - metric_fn(
            y_true[sample_idx], y_a[sample_idx]
        )

    mean_diff = float(np.mean(diffs))
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))

    # Two-sided p-value for H0: diff == 0
    p_value = float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_boot": int(n_boot),
    }


def t_test_compare(y_true, y_a, y_b, metric: str = "f1") -> Dict[str, float]:
    """Paired t-test on per-sample contributions (approximate; see README).

    We approximate by using indicator of correct classification as per-sample
    contribution, which does not exactly decompose F1.
    """
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)

    # Correctness indicators
    a_correct = (y_a == y_true).astype(int)
    b_correct = (y_b == y_true).astype(int)

    t_stat, p_val = stats.ttest_rel(b_correct, a_correct)
    return {"t_stat": float(t_stat), "p_value": float(p_val)}
