"""Randomized row-action methods with convex constraints and Bregman geometry."""

from .bregman import (
    EntropySelectorResult,
    entropy_bregman_kaczmarz,
    kl_divergence,
    local_simplex_error_bound_constants,
    make_entropy_selector_instance,
    softmax,
)
from .datasets import (
    load_moffett_mat,
    make_synthetic_simplex_nmf,
    preprocess_hsi_global_scale,
)
from .nnls import (
    NNLSResult,
    StepSchedule,
    make_noisy_nnls_data,
    nnls_objective,
    projected_row_action_nnls,
    solve_nnls_reference,
)
from .simplex_nmf import (
    SimplexNMFResult,
    entropy_row_action_simplex_nmf,
    relative_reconstruction_error,
)

__all__ = [
    "EntropySelectorResult",
    "NNLSResult",
    "SimplexNMFResult",
    "StepSchedule",
    "entropy_bregman_kaczmarz",
    "entropy_row_action_simplex_nmf",
    "kl_divergence",
    "load_moffett_mat",
    "local_simplex_error_bound_constants",
    "make_entropy_selector_instance",
    "make_noisy_nnls_data",
    "make_synthetic_simplex_nmf",
    "nnls_objective",
    "preprocess_hsi_global_scale",
    "projected_row_action_nnls",
    "relative_reconstruction_error",
    "softmax",
    "solve_nnls_reference",
]

__version__ = "0.1.0"
