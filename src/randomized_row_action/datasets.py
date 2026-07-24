"""Synthetic and hyperspectral datasets used by the experiments."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from ._utils import FloatArray, as_float_array


def normalize_simplex_columns(H: object, *, floor: float = 1e-15) -> FloatArray:
    """Normalize positive matrix columns so that each column sums to one."""
    matrix = np.maximum(as_float_array(H, ndim=2, name="H"), floor)
    sums = np.sum(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(sums, floor)


def make_synthetic_simplex_nmf(
    *,
    m: int = 80,
    rank: int = 5,
    n_samples: int = 1200,
    dirichlet_alpha: float = 0.3,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate noiseless ``X = W_true @ H_true`` with simplex columns in ``H_true``."""
    if m <= 0 or rank <= 0 or n_samples < rank:
        raise ValueError("Require m > 0, rank > 0, and n_samples >= rank.")
    if dirichlet_alpha <= 0.0:
        raise ValueError("dirichlet_alpha must be positive.")
    rng = np.random.default_rng(seed)
    W_true = rng.random((m, rank)) + 0.1
    W_true /= np.maximum(np.linalg.norm(W_true, axis=0, keepdims=True), 1e-15)
    H_true = rng.dirichlet(
        alpha=dirichlet_alpha * np.ones(rank), size=n_samples
    ).T
    H_true[:, :rank] = np.eye(rank)
    return W_true @ H_true, W_true, H_true


def load_moffett_mat(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] = (159, 2500),
) -> FloatArray:
    """Load the Moffett matrix from a MATLAB file.

    The loader searches all numeric two-dimensional variables and accepts either
    ``expected_shape`` or its transpose. MATLAB v7.3 files are read with h5py.
    """
    mat_path = Path(path)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Moffett data file not found: {mat_path}. See data/README.md for instructions."
        )

    try:
        raw = loadmat(mat_path)
    except NotImplementedError:
        raw = {}
        with h5py.File(mat_path, "r") as handle:
            for key in handle.keys():
                raw[key] = np.array(handle[key])

    target_m, target_n = expected_shape
    candidates: list[tuple[str, FloatArray]] = []
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        array = np.squeeze(np.asarray(value))
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
            continue
        if array.shape == expected_shape:
            candidates.append((key, np.asarray(array, dtype=float)))
        elif array.shape == (target_n, target_m):
            candidates.append((key, np.asarray(array.T, dtype=float)))

    if not candidates:
        shapes = {
            key: np.shape(value)
            for key, value in raw.items()
            if not key.startswith("__")
        }
        raise ValueError(
            "Could not find a numeric matrix with shape "
            f"{expected_shape} or {(target_n, target_m)}. Available variables: {shapes}."
        )

    _, X = candidates[0]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(X, 0.0)


def preprocess_hsi_global_scale(X: object, *, floor: float = 1e-12) -> tuple[FloatArray, float]:
    """Clip to nonnegativity and scale a hyperspectral matrix so that ``max(X)=1``."""
    matrix = np.maximum(as_float_array(X, ndim=2, name="X"), 0.0)
    scale = float(np.max(matrix))
    if scale <= floor:
        raise ValueError("X is zero after clipping.")
    return matrix / scale, scale
