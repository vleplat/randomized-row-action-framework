"""Focused tests for the final simplex-NMF comparison script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from randomized_row_action.datasets import make_synthetic_simplex_nmf, normalize_simplex_columns
from randomized_row_action.simplex_nmf import nonnegativity_violation, simplex_violation

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "experiments" / "simplex_nmf_benchmark.py"
SPEC = importlib.util.spec_from_file_location("simplex_nmf_benchmark", SCRIPT_PATH)
assert SPEC is not None
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def _small_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, W_true, H_true = make_synthetic_simplex_nmf(m=10, rank=3, n_samples=18, seed=0)
    return X, W_true, H_true


def test_same_initial_factors_are_saved_for_methods(tmp_path: Path) -> None:
    X, _W_true, _H_true = _small_problem()
    config = comparison.ComparisonConfig(seed=123, n_outer=1)
    W0, H0 = comparison.initial_factors(X, config.rank, config.seed)
    for method in ("single_row_fixed", "minibatch8_fixed"):
        result = comparison.run_method(method, X, W0.copy(), H0.copy(), config)
        comparison.save_method_outputs(method, tmp_path, W0, H0, result, config)
        saved = np.load(tmp_path / f"factors_{method}_seed123.npz")
        np.testing.assert_allclose(saved["W_initial"], W0)
        np.testing.assert_allclose(saved["H_initial"], H0)


def test_fixed_steps_and_batch_size_are_exact() -> None:
    config = comparison.ComparisonConfig(seed=123, n_outer=2)
    assert config.eta_H == 0.1
    assert config.eta_W == 0.01
    assert config.batch_size == 8
    X, _W_true, _H_true = _small_problem()
    W0, H0 = comparison.initial_factors(X, config.rank, config.seed)
    for method in ("single_row_fixed", "minibatch8_fixed"):
        result = comparison.run_method(method, X, W0.copy(), H0.copy(), config)
        np.testing.assert_allclose(result["history"]["eta_H"], 0.1)
        np.testing.assert_allclose(result["history"]["eta_W"], 0.01)


def test_optimized_single_row_trajectory_matches_original() -> None:
    X, _W_true, _H_true = _small_problem()
    config = comparison.ComparisonConfig(seed=123, n_outer=3)
    W0, H0 = comparison.initial_factors(X, config.rank, config.seed)
    original = comparison.run_single_row_fixed_original(X, W0.copy(), H0.copy(), config)
    numpy_optimized = comparison.run_single_row_fixed_numpy(X, W0.copy(), H0.copy(), config)
    selected = comparison.run_single_row_fixed(X, W0.copy(), H0.copy(), config)
    for candidate in (numpy_optimized, selected):
        np.testing.assert_allclose(candidate["W"], original["W"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(candidate["H"], original["H"], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(
            candidate["history"]["objective"],
            original["history"]["objective"],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            candidate["history"]["relative_error"],
            original["history"]["relative_error"],
            rtol=1e-12,
            atol=1e-13,
        )


def test_gram_eigenvalue_matches_squared_spectral_norm() -> None:
    rng = np.random.default_rng(7)
    for matrix in (rng.random((8, 3)), rng.random((3, 8))):
        expected = float(np.linalg.norm(matrix, 2) ** 2)
        observed = comparison._largest_gram_eigenvalue(
            matrix,
            transpose_first=matrix.shape[0] >= matrix.shape[1],
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-14)


def test_optimized_minibatch_updates_match_original() -> None:
    X, W, H = _small_problem()
    H = normalize_simplex_columns(H)
    H_original = comparison.minibatch_H_update_original(
        X,
        W,
        H,
        np.random.default_rng(7),
        eta=0.1,
        batch_size=4,
    )
    H_optimized = comparison.minibatch_H_update(
        X,
        W,
        H,
        np.random.default_rng(7),
        eta=0.1,
        batch_size=4,
    )
    np.testing.assert_allclose(H_optimized, H_original, rtol=1e-13, atol=1e-14)

    W_original = comparison.minibatch_W_update_original(
        X,
        W,
        H,
        np.random.default_rng(11),
        eta=0.01,
        batch_size=4,
    )
    W_optimized = comparison.minibatch_W_update(
        X,
        W,
        H,
        np.random.default_rng(11),
        eta=0.01,
        batch_size=4,
    )
    np.testing.assert_allclose(W_optimized, W_original, rtol=1e-13, atol=1e-14)


def test_optimized_minibatch_trajectory_matches_original() -> None:
    X, _W_true, _H_true = _small_problem()
    config = comparison.ComparisonConfig(seed=123, n_outer=3)
    W0, H0 = comparison.initial_factors(X, config.rank, config.seed)
    original = comparison.run_minibatch8_fixed_original(X, W0.copy(), H0.copy(), config)
    optimized = comparison.run_minibatch8_fixed(X, W0.copy(), H0.copy(), config)
    np.testing.assert_allclose(optimized["W"], original["W"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(optimized["H"], original["H"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        optimized["history"]["objective"],
        original["history"]["objective"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized["history"]["relative_error"],
        original["history"]["relative_error"],
        rtol=1e-12,
        atol=1e-13,
    )


def test_batched_eigensolver_matches_previous_stable_trajectory() -> None:
    X, _W_true, _H_true = _small_problem()
    config = comparison.ComparisonConfig(seed=123, n_outer=3)
    W0, H0 = comparison.initial_factors(X, config.rank, config.seed)
    previous = comparison.run_minibatch8_fixed_unbatched_eigh(
        X, W0.copy(), H0.copy(), config
    )
    integrated = comparison.run_minibatch8_fixed(X, W0.copy(), H0.copy(), config)
    np.testing.assert_array_equal(integrated["W"], previous["W"])
    np.testing.assert_array_equal(integrated["H"], previous["H"])
    np.testing.assert_array_equal(
        integrated["history"]["objective"], previous["history"]["objective"]
    )
    np.testing.assert_array_equal(
        integrated["history"]["relative_error"],
        previous["history"]["relative_error"],
    )


def test_constraints_and_finiteness() -> None:
    X, _W_true, _H_true = _small_problem()
    config = comparison.ComparisonConfig(seed=2, n_outer=2)
    W0, H0 = comparison.initial_factors(X, config.rank, config.seed)
    for method in ("single_row_fixed", "minibatch8_fixed"):
        result = comparison.run_method(method, X, W0.copy(), H0.copy(), config)
        sum_violation, h_negativity = simplex_violation(result["H"])
        assert sum_violation <= 1e-12
        assert h_negativity == 0.0
        assert nonnegativity_violation(result["W"]) == 0.0
        for values in result["history"].values():
            assert np.all(np.isfinite(values))


def test_component_alignment_is_valid_permutation() -> None:
    rng = np.random.default_rng(0)
    W_ref = rng.random((5, 3))
    H_ref = normalize_simplex_columns(rng.random((3, 8)))
    permutation = [2, 0, 1]
    factors = {
        "reference": (W_ref, H_ref),
        "single_row_fixed": (W_ref[:, permutation], H_ref[permutation]),
        "minibatch8_fixed": (W_ref[:, permutation], H_ref[permutation]),
    }
    _aligned, permutations = comparison.align_to_reference(factors)
    for method in comparison.METHODS:
        assert sorted(permutations[method]) == [0, 1, 2]


def test_saved_histories_can_be_reloaded(tmp_path: Path) -> None:
    X, _W_true, _H_true = _small_problem()
    config = comparison.ComparisonConfig(seed=123, n_outer=1)
    W0, H0 = comparison.initial_factors(X, config.rank, config.seed)
    result = comparison.run_method("single_row_fixed", X, W0.copy(), H0.copy(), config)
    comparison.save_method_outputs("single_row_fixed", tmp_path, W0, H0, result, config)
    with np.load(tmp_path / "history_single_row_fixed_seed123.npz") as data:
        assert "objective" in data.files
        assert "relative_error" in data.files
