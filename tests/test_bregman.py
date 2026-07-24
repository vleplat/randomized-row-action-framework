import numpy as np

from randomized_row_action.bregman import (
    entropy_bregman_kaczmarz,
    local_simplex_error_bound_constants,
    make_entropy_selector_instance,
    softmax,
)


def test_softmax_returns_simplex_point():
    x = softmax(np.array([1000.0, 999.0, 998.0]))
    assert np.all(x > 0.0)
    assert np.isclose(np.sum(x), 1.0)


def test_entropy_selector_preserves_simplex_and_decreases_residual():
    instance = make_entropy_selector_instance(m=80, n=20, seed=2)
    result = entropy_bregman_kaczmarz(
        instance["A"],
        instance["b"],
        x_hat=instance["x_hat"],
        u0=instance["u0"],
        n_iterations=800,
        eta=1.0,
        seed=5,
        record_every=80,
    )
    assert np.all(result.x_last > 0.0)
    assert np.isclose(np.sum(result.x_last), 1.0)
    assert result.history["residual_sq"][-1] < result.history["residual_sq"][0]
    constants = local_simplex_error_bound_constants(
        instance["A"], instance["x_hat"], delta=0.5
    )
    assert constants["theta_delta"] > 0.0
