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


def test_minibatch_formula_matches_direct_block_update() -> None:
    X, W, H = _small_problem()
    H = normalize_simplex_columns(H)
    rng = np.random.default_rng(7)
    rows = rng.choice(W.shape[0], size=4, replace=False)
    W_batch = W[rows]
    residual = W_batch @ H - X[rows]
    gradient = W_batch.T @ residual
    lipschitz = max(float(np.linalg.norm(W_batch, 2) ** 2), comparison.FLOOR)
    expected = normalize_simplex_columns(
        H * np.exp(np.clip(-0.1 * gradient / lipschitz, -comparison.EXPONENT_CLIP, comparison.EXPONENT_CLIP)),
        floor=comparison.FLOOR,
    )

    replay_rng = np.random.default_rng(7)
    observed = comparison.minibatch_H_update(X, W, H, replay_rng, eta=0.1, batch_size=4)
    # With m=10 and batch_size=4 there are three blocks, so compare the first block explicitly.
    first_block = normalize_simplex_columns(
        H * np.exp(np.clip(-0.1 * gradient / lipschitz, -comparison.EXPONENT_CLIP, comparison.EXPONENT_CLIP)),
        floor=comparison.FLOOR,
    )
    assert expected.shape == observed.shape
    np.testing.assert_allclose(first_block.sum(axis=0), 1.0)


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
