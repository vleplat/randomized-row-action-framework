"""Entropy row-action updates for simplex-structured nonnegative matrix factorization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from ._utils import FloatArray, arrays_from_history, as_float_array
from .datasets import normalize_simplex_columns

LineSearchMode = Literal["always", "warmup", "none"]


@dataclass
class SimplexNMFResult:
    """Output of :func:`entropy_row_action_simplex_nmf`."""

    W: FloatArray
    H: FloatArray
    history: dict[str, FloatArray]


def nmf_loss(X: object, W: object, H: object) -> float:
    data = as_float_array(X, ndim=2, name="X")
    basis = as_float_array(W, ndim=2, name="W")
    abundance = as_float_array(H, ndim=2, name="H")
    if basis.shape[0] != data.shape[0] or abundance.shape[1] != data.shape[1]:
        raise ValueError("X, W, and H have incompatible dimensions.")
    if basis.shape[1] != abundance.shape[0]:
        raise ValueError("The inner dimensions of W and H do not agree.")
    residual = data - basis @ abundance
    return 0.5 * float(np.sum(residual * residual))


def relative_reconstruction_error(X: object, W: object, H: object) -> float:
    data = as_float_array(X, ndim=2, name="X")
    denominator = max(float(np.linalg.norm(data, "fro")), np.finfo(float).tiny)
    return float(np.linalg.norm(data - np.asarray(W) @ np.asarray(H), "fro") / denominator)


def simplex_violation(H: object) -> tuple[float, float]:
    abundance = as_float_array(H, ndim=2, name="H")
    sum_violation = float(np.max(np.abs(np.sum(abundance, axis=0) - 1.0)))
    negativity = float(max(0.0, -np.min(abundance)))
    return sum_violation, negativity


def nonnegativity_violation(W: object) -> float:
    basis = as_float_array(W, ndim=2, name="W")
    return float(max(0.0, -np.min(basis)))


def entropy_H_sweep(
    X: FloatArray,
    W: FloatArray,
    H: FloatArray,
    *,
    eta: float,
    row_order: np.ndarray,
    floor: float = 1e-15,
    exponent_clip: float = 50.0,
) -> FloatArray:
    """Apply one entropy row-action sweep to the simplex-constrained ``H`` block."""
    updated = H.copy()
    for row_index in row_order:
        row = W[row_index]
        denominator = float(row @ row) + floor
        residual = row @ updated - X[row_index]
        exponent = -eta * (row[:, None] * residual[None, :]) / denominator
        updated *= np.exp(np.clip(exponent, -exponent_clip, exponent_clip))
        updated = normalize_simplex_columns(updated, floor=floor)
    return updated


def entropy_W_sweep(
    X: FloatArray,
    W: FloatArray,
    H: FloatArray,
    *,
    eta: float,
    column_order: np.ndarray,
    floor: float = 1e-15,
    exponent_clip: float = 50.0,
) -> FloatArray:
    """Apply one positive-entropy row-action sweep to the nonnegative ``W`` block."""
    updated = W.copy()
    for column_index in column_order:
        abundance = H[:, column_index]
        denominator = float(abundance @ abundance) + floor
        residual = updated @ abundance - X[:, column_index]
        exponent = -eta * (residual[:, None] * abundance[None, :]) / denominator
        updated *= np.exp(np.clip(exponent, -exponent_clip, exponent_clip))
        updated = np.maximum(updated, floor)
    return updated


def _accepted_H_block(
    X: FloatArray,
    W: FloatArray,
    H: FloatArray,
    *,
    eta: float,
    row_order: np.ndarray,
    eta_min: float,
    decay: float,
    max_backtracks: int,
    floor: float,
    exponent_clip: float,
) -> tuple[FloatArray, float, bool, int]:
    old_loss = nmf_loss(X, W, H)
    eta_trial = eta
    for backtrack in range(max_backtracks + 1):
        candidate = entropy_H_sweep(
            X,
            W,
            H,
            eta=eta_trial,
            row_order=row_order,
            floor=floor,
            exponent_clip=exponent_clip,
        )
        if nmf_loss(X, W, candidate) <= old_loss * (1.0 + 1e-12):
            return candidate, eta_trial, True, backtrack
        eta_trial *= decay
        if eta_trial < eta_min:
            break
    return H, eta_trial, False, max_backtracks + 1


def _accepted_W_block(
    X: FloatArray,
    W: FloatArray,
    H: FloatArray,
    *,
    eta: float,
    column_order: np.ndarray,
    eta_min: float,
    decay: float,
    max_backtracks: int,
    floor: float,
    exponent_clip: float,
) -> tuple[FloatArray, float, bool, int]:
    old_loss = nmf_loss(X, W, H)
    eta_trial = eta
    for backtrack in range(max_backtracks + 1):
        candidate = entropy_W_sweep(
            X,
            W,
            H,
            eta=eta_trial,
            column_order=column_order,
            floor=floor,
            exponent_clip=exponent_clip,
        )
        if nmf_loss(X, candidate, H) <= old_loss * (1.0 + 1e-12):
            return candidate, eta_trial, True, backtrack
        eta_trial *= decay
        if eta_trial < eta_min:
            break
    return W, eta_trial, False, max_backtracks + 1


def entropy_row_action_simplex_nmf(
    X: object,
    *,
    rank: int,
    n_outer: int = 800,
    eta_H0: float = 1.0,
    eta_W0: float = 0.01,
    eta_H_max: float = 2.0,
    eta_W_max: float = 0.1,
    growth: float = 1.02,
    decay: float = 0.5,
    line_search_mode: LineSearchMode = "warmup",
    warmup_outer: int = 40,
    max_backtracks: int = 20,
    seed: int = 123,
    W0: object | None = None,
    H0: object | None = None,
    floor: float = 1e-15,
    exponent_clip: float = 50.0,
    verbose_every: int | None = None,
) -> SimplexNMFResult:
    """Alternating entropy row-action solver for simplex-structured NMF.

    The model is

    ``min 0.5 ||X - W H||_F^2`` subject to ``W >= 0`` and simplex columns in ``H``.

    Backtracking may be used throughout, only during a warmup phase, or disabled.
    The paper uses warmup backtracking and then keeps the accepted block steps fixed.
    """
    data = as_float_array(X, ndim=2, name="X")
    if np.any(data < 0.0):
        raise ValueError("X must be nonnegative.")
    if rank <= 0 or rank > min(data.shape):
        raise ValueError("rank must be positive and no larger than min(X.shape).")
    if n_outer <= 0:
        raise ValueError("n_outer must be positive.")
    if line_search_mode not in {"always", "warmup", "none"}:
        raise ValueError("line_search_mode must be 'always', 'warmup', or 'none'.")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must belong to (0, 1).")
    if growth < 1.0:
        raise ValueError("growth must be at least 1.")

    rng = np.random.default_rng(seed)
    m, n_samples = data.shape

    if (W0 is None) != (H0 is None):
        raise ValueError("W0 and H0 must be provided together.")
    if W0 is None:
        W = rng.random((m, rank)) + 1e-2
        H = rng.dirichlet(np.ones(rank), size=n_samples).T
        scale = np.linalg.norm(data, "fro") / max(np.linalg.norm(W @ H, "fro"), floor)
        W = np.maximum(scale * W, floor)
    else:
        W = np.maximum(as_float_array(W0, ndim=2, name="W0"), floor)
        H = normalize_simplex_columns(H0, floor=floor)
        if W.shape != (m, rank) or H.shape != (rank, n_samples):
            raise ValueError(
                f"Expected W0 shape {(m, rank)} and H0 shape {(rank, n_samples)}; "
                f"got {W.shape} and {H.shape}."
            )

    eta_H = float(eta_H0)
    eta_W = float(eta_W0)
    if eta_H <= 0.0 or eta_W <= 0.0:
        raise ValueError("Initial step sizes must be positive.")

    started = perf_counter()
    history: dict[str, list[float]] = {
        "outer_iteration": [],
        "loss": [],
        "relative_error": [],
        "time": [],
        "eta_H": [],
        "eta_W": [],
        "H_backtracks": [],
        "W_backtracks": [],
        "accepted_H": [],
        "accepted_W": [],
        "H_simplex_sum_violation": [],
        "H_negativity_violation": [],
        "W_negativity_violation": [],
    }

    def record(
        outer_iteration: int,
        *,
        backtracks_H: int,
        backtracks_W: int,
        accepted_H: bool,
        accepted_W: bool,
    ) -> None:
        sum_violation, H_negativity = simplex_violation(H)
        history["outer_iteration"].append(float(outer_iteration))
        history["loss"].append(nmf_loss(data, W, H))
        history["relative_error"].append(relative_reconstruction_error(data, W, H))
        history["time"].append(perf_counter() - started)
        history["eta_H"].append(eta_H)
        history["eta_W"].append(eta_W)
        history["H_backtracks"].append(float(backtracks_H))
        history["W_backtracks"].append(float(backtracks_W))
        history["accepted_H"].append(float(accepted_H))
        history["accepted_W"].append(float(accepted_W))
        history["H_simplex_sum_violation"].append(sum_violation)
        history["H_negativity_violation"].append(H_negativity)
        history["W_negativity_violation"].append(nonnegativity_violation(W))

    record(0, backtracks_H=0, backtracks_W=0, accepted_H=True, accepted_W=True)

    for outer in range(1, n_outer + 1):
        use_line_search = line_search_mode == "always" or (
            line_search_mode == "warmup" and outer <= warmup_outer
        )
        row_order = rng.permutation(m)
        if use_line_search:
            H, eta_H_used, accepted_H, backtracks_H = _accepted_H_block(
                data,
                W,
                H,
                eta=eta_H,
                row_order=row_order,
                eta_min=1e-10,
                decay=decay,
                max_backtracks=max_backtracks,
                floor=floor,
                exponent_clip=exponent_clip,
            )
            eta_H = min(growth * eta_H_used, eta_H_max) if accepted_H else max(eta_H_used, floor)
        else:
            H = entropy_H_sweep(
                data,
                W,
                H,
                eta=eta_H,
                row_order=row_order,
                floor=floor,
                exponent_clip=exponent_clip,
            )
            accepted_H = True
            backtracks_H = 0

        column_order = rng.permutation(n_samples)
        if use_line_search:
            W, eta_W_used, accepted_W, backtracks_W = _accepted_W_block(
                data,
                W,
                H,
                eta=eta_W,
                column_order=column_order,
                eta_min=1e-10,
                decay=decay,
                max_backtracks=max_backtracks,
                floor=floor,
                exponent_clip=exponent_clip,
            )
            eta_W = min(growth * eta_W_used, eta_W_max) if accepted_W else max(eta_W_used, floor)
        else:
            W = entropy_W_sweep(
                data,
                W,
                H,
                eta=eta_W,
                column_order=column_order,
                floor=floor,
                exponent_clip=exponent_clip,
            )
            accepted_W = True
            backtracks_W = 0

        record(
            outer,
            backtracks_H=backtracks_H,
            backtracks_W=backtracks_W,
            accepted_H=accepted_H,
            accepted_W=accepted_W,
        )
        if verbose_every and outer % verbose_every == 0:
            print(
                f"outer={outer:4d} | rel.err={history['relative_error'][-1]:.3e} | "
                f"eta_H={eta_H:.3e} | eta_W={eta_W:.3e} | "
                f"bt_H={backtracks_H} | bt_W={backtracks_W}"
            )

    return SimplexNMFResult(W=W, H=H, history=arrays_from_history(history))
