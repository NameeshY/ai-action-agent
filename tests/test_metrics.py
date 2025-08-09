from __future__ import annotations

import numpy as np

from metrics import compute_prf, bootstrap_compare


def test_compute_prf_basic():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 1]
    m = compute_prf(y_true, y_pred)
    assert set(m.keys()) == {"precision", "recall", "f1"}
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0


def test_bootstrap_compare_outputs():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_a = rng.integers(0, 2, size=50)
    y_b = rng.integers(0, 2, size=50)

    res = bootstrap_compare(y_true, y_a, y_b, metric="f1", n_boot=200, seed=0)
    assert set(res.keys()) == {"mean_diff", "ci_low", "ci_high", "p_value", "n_boot"}
    assert res["ci_low"] <= res["ci_high"]
    assert res["n_boot"] == 200
