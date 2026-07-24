import numpy as np

from randomized_row_action.datasets import make_synthetic_simplex_nmf
from randomized_row_action.simplex_nmf import entropy_row_action_simplex_nmf


def test_simplex_nmf_shapes_constraints_and_finite_history():
    X, _, _ = make_synthetic_simplex_nmf(m=20, rank=3, n_samples=40, seed=1)
    result = entropy_row_action_simplex_nmf(
        X,
        rank=3,
        n_outer=3,
        line_search_mode="warmup",
        warmup_outer=2,
        seed=2,
    )
    assert result.W.shape == (20, 3)
    assert result.H.shape == (3, 40)
    assert np.min(result.W) >= 0.0
    assert np.min(result.H) >= 0.0
    assert np.allclose(np.sum(result.H, axis=0), 1.0)
    assert np.all(np.isfinite(result.history["relative_error"]))
