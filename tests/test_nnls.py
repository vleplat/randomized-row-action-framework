import numpy as np

from randomized_row_action.nnls import (
    StepSchedule,
    make_noisy_nnls_data,
    nnls_objective,
    projected_row_action_nnls,
    solve_nnls_reference,
)


def test_projected_row_action_preserves_nonnegativity_and_reduces_gap():
    A, b, *_ = make_noisy_nnls_data(m=80, n=20, seed=3)
    x_star, _ = solve_nnls_reference(A, b)
    f_star = nnls_objective(A, b, x_star)
    result = projected_row_action_nnls(
        A,
        b,
        n_iterations=800,
        schedule=StepSchedule("constant", 0.2),
        seed=4,
        x_star=x_star,
        f_star=f_star,
        record_every=80,
    )
    assert np.min(result.x_last) >= 0.0
    assert np.min(result.x_average) >= 0.0
    assert result.history["objective_gap"][-1] <= result.history["objective_gap"][0]


def test_weighted_average_mode_is_explicit_and_finite():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    result = projected_row_action_nnls(
        A,
        b,
        n_iterations=10,
        schedule=StepSchedule("diminishing_sqrt", 0.5),
        seed=1,
        record_every=5,
        average_mode="pre_update",
    )
    assert np.all(np.isfinite(result.x_average))
