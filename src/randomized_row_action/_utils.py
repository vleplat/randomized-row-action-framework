"""Internal validation and sampling helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def as_float_array(value: object, *, ndim: int | None = None, name: str = "array") -> FloatArray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def validate_matrix_vector(A: object, b: object) -> tuple[FloatArray, FloatArray]:
    matrix = as_float_array(A, ndim=2, name="A")
    vector = as_float_array(b, ndim=1, name="b")
    if matrix.shape[0] != vector.size:
        raise ValueError(
            f"A and b have incompatible dimensions: A has {matrix.shape[0]} rows, "
            f"while b has length {vector.size}."
        )
    return matrix, vector


def row_sampling_probabilities(A: FloatArray) -> tuple[FloatArray, FloatArray]:
    row_norm_sq = np.einsum("ij,ij->i", A, A)
    if np.any(row_norm_sq <= 0.0):
        zero_rows = np.flatnonzero(row_norm_sq <= 0.0)
        raise ValueError(f"A contains zero rows at indices {zero_rows.tolist()}.")
    probabilities = row_norm_sq / np.sum(row_norm_sq)
    return row_norm_sq, probabilities


def sampled_index_batches(
    rng: np.random.Generator,
    probabilities: FloatArray,
    n_samples: int,
    *,
    batch_size: int = 8192,
) -> Iterator[NDArray[np.int64]]:
    """Yield weighted random indices in batches to reduce Python/RNG overhead."""
    remaining = int(n_samples)
    while remaining > 0:
        size = min(batch_size, remaining)
        yield rng.choice(probabilities.size, size=size, replace=True, p=probabilities)
        remaining -= size


def arrays_from_history(history: dict[str, list[float]]) -> dict[str, FloatArray]:
    return {key: np.asarray(values, dtype=float) for key, values in history.items()}
