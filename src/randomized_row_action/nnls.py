"""Projected randomized row-action methods for noisy nonnegative least squares."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.optimize import nnls as scipy_nnls

from ._utils import (
    FloatArray,
    arrays_from_history,
    as_float_array,
    row_sampling_probabilities,
    sampled_index_batches,
    validate_matrix_vector,
)

ScheduleKind = Literal["constant", "horizon_sqrt", "diminishing_sqrt"]
AverageMode = Literal["pre_update", "post_update"]


@dataclass(frozen=True)
class StepSchedule:
    """Step-size schedule used by the projected row-action solver."""

    kind: ScheduleKind
    eta0: float

    def value(self, iteration: int, total_iterations: int) -> float:
        if self.eta0 <= 0.0:
            raise ValueError("eta0 must be positive.")
        if iteration < 1 or total_iterations < 1:
            raise ValueError("iteration and total_iterations must be positive.")
        if self.kind == "constant":
            return float(self.eta0)
        if self.kind == "horizon_sqrt":
            return float(self.eta0 / np.sqrt(total_iterations))
        if self.kind == "diminishing_sqrt":
            return float(self.eta0 / np.sqrt(iteration))
        raise ValueError(f"Unknown schedule kind: {self.kind!r}.")


@dataclass
class NNLSResult:
    """Output of :func:`projected_row_action_nnls`."""

    x_last: FloatArray
    x_average: FloatArray
    history: dict[str, FloatArray]


def project_nonnegative(x: object) -> FloatArray:
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def nnls_objective(A: object, b: object, x: object) -> float:
    matrix, vector = validate_matrix_vector(A, b)
    point = as_float_array(x, ndim=1, name="x")
    if matrix.shape[1] != point.size:
        raise ValueError("A and x have incompatible dimensions.")
    residual = matrix @ point - vector
    return 0.5 * float(residual @ residual)


def relative_residual(A: object, b: object, x: object) -> float:
    matrix, vector = validate_matrix_vector(A, b)
    point = as_float_array(x, ndim=1, name="x")
    denominator = max(float(np.linalg.norm(vector)), np.finfo(float).tiny)
    return float(np.linalg.norm(matrix @ point - vector) / denominator)


def make_noisy_nnls_data(
    *,
    m: int = 1000,
    n: int = 200,
    sparsity: float = 0.25,
    noise_level: float = 0.05,
    seed: int = 8,
    normalize_rows: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate ``b = A @ x_true + noise`` with ``x_true >= 0``.

    The noise is scaled so that ``||noise|| / ||A @ x_true|| = noise_level``.
    For positive ``noise_level``, the linear system is generically inconsistent.
    """
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive.")
    if not 0.0 < sparsity <= 1.0:
        raise ValueError("sparsity must belong to (0, 1].")
    if noise_level < 0.0:
        raise ValueError("noise_level must be nonnegative.")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    if normalize_rows:
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        A /= np.maximum(norms, np.finfo(float).tiny)

    support_size = max(1, int(round(sparsity * n)))
    support = rng.choice(n, size=support_size, replace=False)
    x_true = np.zeros(n)
    x_true[support] = rng.uniform(0.5, 2.0, size=support_size)

    clean = A @ x_true
    raw_noise = rng.standard_normal(m)
    clean_norm = max(float(np.linalg.norm(clean)), np.finfo(float).tiny)
    raw_norm = max(float(np.linalg.norm(raw_noise)), np.finfo(float).tiny)
    noise = noise_level * clean_norm * raw_noise / raw_norm
    b = clean + noise
    return A, b, x_true, clean, noise


def _fista_reference(
    A: FloatArray,
    b: FloatArray,
    *,
    max_iter: int = 30_000,
    tolerance: float = 1e-11,
) -> tuple[FloatArray, dict[str, object]]:
    """Fallback deterministic NNLS solver used if SciPy's active-set solver fails."""
    n = A.shape[1]
    lipschitz = float(np.linalg.norm(A, ord=2) ** 2)
    if lipschitz <= 0.0:
        return np.zeros(n), {"method": "fista", "iterations": 0}

    x = np.zeros(n)
    y = x.copy()
    momentum = 1.0
    previous_objective = np.inf

    for iteration in range(1, max_iter + 1):
        gradient = A.T @ (A @ y - b)
        x_next = project_nonnegative(y - gradient / lipschitz)
        objective = nnls_objective(A, b, x_next)

        if objective > previous_objective + 1e-14:
            y = x.copy()
            momentum = 1.0
            gradient = A.T @ (A @ y - b)
            x_next = project_nonnegative(y - gradient / lipschitz)
            objective = nnls_objective(A, b, x_next)

        if np.linalg.norm(x_next - x) <= tolerance * max(1.0, np.linalg.norm(x)):
            return x_next, {
                "method": "fista",
                "iterations": iteration,
                "objective": objective,
            }

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        y = x_next + ((momentum - 1.0) / next_momentum) * (x_next - x)
        x = x_next
        momentum = next_momentum
        previous_objective = objective

    return x, {"method": "fista", "iterations": max_iter, "objective": previous_objective}


def solve_nnls_reference(
    A: object,
    b: object,
    *,
    max_iter: int = 30_000,
    tolerance: float = 1e-11,
) -> tuple[FloatArray, dict[str, object]]:
    """Compute a high-accuracy NNLS reference point."""
    matrix, vector = validate_matrix_vector(A, b)
    try:
        x_star, residual_norm = scipy_nnls(matrix, vector, maxiter=max_iter)
    except (RuntimeError, ValueError):
        return _fista_reference(matrix, vector, max_iter=max_iter, tolerance=tolerance)
    return x_star, {
        "method": "scipy.optimize.nnls",
        "residual_norm": float(residual_norm),
        "objective": 0.5 * float(residual_norm**2),
    }


def projected_row_action_nnls(
    A: object,
    b: object,
    *,
    n_iterations: int,
    schedule: StepSchedule,
    seed: int = 0,
    x0: object | None = None,
    x_star: object | None = None,
    f_star: float | None = None,
    record_every: int | None = None,
    average_mode: AverageMode = "pre_update",
    sampling_batch_size: int = 8192,
) -> NNLSResult:
    """Run projected randomized row action for noisy NNLS.

    The default ``average_mode='pre_update'`` computes

    ``x_bar = sum_k eta_k x_k / sum_k eta_k``,

    exactly as stated in the paper. ``post_update`` is provided only to compare
    with the original exploratory notebook, which averaged ``x_{k+1}``.
    """
    A_array, b_array = validate_matrix_vector(A, b)
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if average_mode not in {"pre_update", "post_update"}:
        raise ValueError("average_mode must be 'pre_update' or 'post_update'.")

    m, n = A_array.shape
    row_norm_sq, probabilities = row_sampling_probabilities(A_array)
    rng = np.random.default_rng(seed)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = project_nonnegative(as_float_array(x0, ndim=1, name="x0"))
        if x.size != n:
            raise ValueError("x0 has an incompatible dimension.")

    reference = None if x_star is None else as_float_array(x_star, ndim=1, name="x_star")
    if reference is not None and reference.size != n:
        raise ValueError("x_star has an incompatible dimension.")

    if record_every is None:
        record_every = m
    if record_every <= 0:
        raise ValueError("record_every must be positive.")

    average_sum = np.zeros(n)
    eta_sum = 0.0
    started = perf_counter()
    history: dict[str, list[float]] = {
        "iteration": [],
        "epoch": [],
        "time": [],
        "eta": [],
        "objective": [],
        "objective_gap": [],
        "relative_residual": [],
        "distance_to_star": [],
        "average_objective": [],
        "average_objective_gap": [],
        "average_distance_to_star": [],
    }

    def current_average() -> FloatArray:
        return x if eta_sum == 0.0 else average_sum / eta_sum

    def record(iteration: int, eta: float) -> None:
        x_average = current_average()
        objective = nnls_objective(A_array, b_array, x)
        average_objective = nnls_objective(A_array, b_array, x_average)
        history["iteration"].append(float(iteration))
        history["epoch"].append(iteration / m)
        history["time"].append(perf_counter() - started)
        history["eta"].append(float(eta))
        history["objective"].append(objective)
        history["relative_residual"].append(relative_residual(A_array, b_array, x))
        history["average_objective"].append(average_objective)
        history["objective_gap"].append(
            np.nan if f_star is None else max(objective - float(f_star), 0.0)
        )
        history["average_objective_gap"].append(
            np.nan if f_star is None else max(average_objective - float(f_star), 0.0)
        )
        history["distance_to_star"].append(
            np.nan if reference is None else float(np.linalg.norm(x - reference))
        )
        history["average_distance_to_star"].append(
            np.nan if reference is None else float(np.linalg.norm(x_average - reference))
        )

    record(0, 0.0)
    iteration = 0

    for sampled_indices in sampled_index_batches(
        rng, probabilities, n_iterations, batch_size=sampling_batch_size
    ):
        for row_index in sampled_indices:
            iteration += 1
            eta = schedule.value(iteration, n_iterations)
            if average_mode == "pre_update":
                average_sum += eta * x
                eta_sum += eta

            row = A_array[row_index]
            residual = float(row @ x - b_array[row_index])
            x = project_nonnegative(x - eta * (residual / row_norm_sq[row_index]) * row)

            if average_mode == "post_update":
                average_sum += eta * x
                eta_sum += eta

            if iteration % record_every == 0 or iteration == n_iterations:
                record(iteration, eta)

    return NNLSResult(
        x_last=x,
        x_average=current_average().copy(),
        history=arrays_from_history(history),
    )
