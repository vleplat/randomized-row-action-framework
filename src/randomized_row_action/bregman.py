"""Entropy Bregman--Kaczmarz selectors on the probability simplex."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ._utils import (
    FloatArray,
    arrays_from_history,
    as_float_array,
    row_sampling_probabilities,
    sampled_index_batches,
    validate_matrix_vector,
)


@dataclass
class EntropySelectorResult:
    """Output of :func:`entropy_bregman_kaczmarz`."""

    x_last: FloatArray
    u_last: FloatArray
    history: dict[str, FloatArray]


def softmax(u: object) -> FloatArray:
    """Stable softmax map from dual variables to the simplex."""
    vector = as_float_array(u, ndim=1, name="u")
    shifted = vector - np.max(vector)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential)


def kl_divergence(x_hat: object, x: object, *, floor: float = 1e-300) -> float:
    """Return ``KL(x_hat || x)`` with a numerical floor in the logarithm."""
    target = np.maximum(as_float_array(x_hat, ndim=1, name="x_hat"), floor)
    point = np.maximum(as_float_array(x, ndim=1, name="x"), floor)
    if target.shape != point.shape:
        raise ValueError("x_hat and x must have the same shape.")
    return float(np.sum(target * np.log(target / point)))


def simplex_tangent_basis(n: int) -> FloatArray:
    """Return an orthonormal basis of ``{z : 1^T z = 0}``."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    basis = np.eye(n)[:, :-1] - np.eye(n)[:, [-1]]
    q, _ = np.linalg.qr(basis)
    return q[:, : n - 1]


def restricted_simplex_singular_value(A: object) -> tuple[float, FloatArray]:
    """Compute the smallest singular value of ``A`` on the simplex tangent space."""
    matrix = as_float_array(A, ndim=2, name="A")
    q = simplex_tangent_basis(matrix.shape[1])
    singular_values = np.linalg.svd(matrix @ q, compute_uv=False)
    return float(singular_values[-1]), singular_values


def make_close_simplex_point(
    x_hat: object,
    *,
    relative_inf_radius: float = 0.25,
    seed: int = 0,
) -> FloatArray:
    """Create a simplex point near ``x_hat`` in relative infinity norm."""
    target = as_float_array(x_hat, ndim=1, name="x_hat")
    if np.any(target <= 0.0) or not np.isclose(np.sum(target), 1.0):
        raise ValueError("x_hat must lie in the relative interior of the simplex.")
    if not 0.0 <= relative_inf_radius < 1.0:
        raise ValueError("relative_inf_radius must belong to [0, 1).")

    rng = np.random.default_rng(seed)
    alpha = float(np.min(target))
    direction = rng.standard_normal(target.size)
    direction -= np.mean(direction)
    direction /= max(float(np.max(np.abs(direction))), np.finfo(float).tiny)
    point = target + relative_inf_radius * alpha * direction
    if np.min(point) <= 0.0:
        raise RuntimeError("The generated point left the relative interior of the simplex.")
    point = np.maximum(point, np.finfo(float).tiny)
    return point / np.sum(point)


def make_entropy_selector_instance(
    *,
    m: int = 1000,
    n: int = 200,
    target_spread: float = 0.25,
    initial_relative_radius: float = 0.25,
    seed: int = 7,
    minimum_restricted_sigma: float = 1e-10,
    maximum_tries: int = 100,
) -> dict[str, object]:
    """Generate the local simplex-selector instance used in the paper."""
    if m <= 0 or n < 2:
        raise ValueError("m must be positive and n must be at least 2.")
    rng = np.random.default_rng(seed)
    x_hat = softmax(target_spread * rng.standard_normal(n))

    for _ in range(maximum_tries):
        candidate = rng.standard_normal((m, n)) / np.sqrt(n)
        sigma_delta, singular_values = restricted_simplex_singular_value(candidate)
        augmented_rank = np.linalg.matrix_rank(np.vstack([candidate, np.ones((1, n))]))
        if sigma_delta > minimum_restricted_sigma and augmented_rank == n:
            A = candidate
            break
    else:
        raise RuntimeError("Failed to generate a matrix injective on the simplex tangent space.")

    b = A @ x_hat
    x0 = make_close_simplex_point(
        x_hat,
        relative_inf_radius=initial_relative_radius,
        seed=seed + 1,
    )
    u0 = np.log(x0)
    return {
        "A": A,
        "b": b,
        "x_hat": x_hat,
        "x0": x0,
        "u0": u0,
        "sigma_delta": sigma_delta,
        "tangent_singular_values": singular_values,
        "alpha": float(np.min(x_hat)),
    }


def local_simplex_error_bound_constants(
    A: object,
    x_hat: object,
    *,
    delta: float = 0.5,
) -> dict[str, object]:
    """Return the local residual-to-KL constants used in the paper."""
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta must belong to [0, 1).")
    target = as_float_array(x_hat, ndim=1, name="x_hat")
    sigma_delta, singular_values = restricted_simplex_singular_value(A)
    alpha = float(np.min(target))
    if alpha <= 0.0:
        raise ValueError("x_hat must lie in the relative interior of the simplex.")
    theta_asymptotic = 2.0 * alpha * sigma_delta**2
    theta_delta = 2.0 * (1.0 - delta) ** 2 * alpha * sigma_delta**2
    return {
        "alpha": alpha,
        "delta": float(delta),
        "sigma_delta": sigma_delta,
        "theta_asymptotic": theta_asymptotic,
        "theta_delta": theta_delta,
        "ratio_bound_asymptotic": 1.0 / theta_asymptotic,
        "ratio_bound_delta": 1.0 / theta_delta,
        "tangent_singular_values": singular_values,
    }


def entropy_bregman_kaczmarz(
    A: object,
    b: object,
    *,
    x_hat: object,
    u0: object,
    n_iterations: int,
    eta: float = 1.0,
    seed: int = 0,
    record_every: int | None = None,
    sampling_batch_size: int = 8192,
) -> EntropySelectorResult:
    """Run relaxed randomized entropy Bregman--Kaczmarz on the simplex."""
    matrix, vector = validate_matrix_vector(A, b)
    target = as_float_array(x_hat, ndim=1, name="x_hat")
    u = as_float_array(u0, ndim=1, name="u0").copy()
    if target.size != matrix.shape[1] or u.size != matrix.shape[1]:
        raise ValueError("x_hat and u0 must have one entry per column of A.")
    if np.any(target <= 0.0) or not np.isclose(np.sum(target), 1.0):
        raise ValueError("x_hat must lie in the relative interior of the simplex.")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if eta <= 0.0:
        raise ValueError("eta must be positive.")

    m = matrix.shape[0]
    if record_every is None:
        record_every = m
    if record_every <= 0:
        raise ValueError("record_every must be positive.")

    row_norm_sq, probabilities = row_sampling_probabilities(matrix)
    rng = np.random.default_rng(seed)
    x = softmax(u)
    alpha = max(float(np.min(target)), np.finfo(float).tiny)
    started = perf_counter()
    history: dict[str, list[float]] = {
        "iteration": [],
        "epoch": [],
        "time": [],
        "residual_norm": [],
        "residual_sq": [],
        "kl": [],
        "l2_error": [],
        "linf_over_alpha": [],
        "error_bound_ratio": [],
        "theta_empirical": [],
    }

    def record(iteration: int) -> None:
        residual = matrix @ x - vector
        residual_sq = float(residual @ residual)
        kl_value = kl_divergence(target, x)
        history["iteration"].append(float(iteration))
        history["epoch"].append(iteration / m)
        history["time"].append(perf_counter() - started)
        history["residual_norm"].append(float(np.sqrt(residual_sq)))
        history["residual_sq"].append(residual_sq)
        history["kl"].append(kl_value)
        history["l2_error"].append(float(np.linalg.norm(x - target)))
        history["linf_over_alpha"].append(float(np.max(np.abs(x - target)) / alpha))
        history["error_bound_ratio"].append(
            np.nan if residual_sq <= 0.0 else kl_value / residual_sq
        )
        history["theta_empirical"].append(
            np.nan if kl_value <= 0.0 else residual_sq / kl_value
        )

    record(0)
    iteration = 0
    for sampled_indices in sampled_index_batches(
        rng, probabilities, n_iterations, batch_size=sampling_batch_size
    ):
        for row_index in sampled_indices:
            iteration += 1
            row = matrix[row_index]
            residual = float(row @ x - vector[row_index])
            u -= eta * (residual / row_norm_sq[row_index]) * row
            x = softmax(u)
            if iteration % record_every == 0 or iteration == n_iterations:
                record(iteration)

    return EntropySelectorResult(
        x_last=x,
        u_last=u,
        history=arrays_from_history(history),
    )
