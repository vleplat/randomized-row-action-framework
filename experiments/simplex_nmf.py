"""Run the simplex-structured NMF experiment from Section 6.3."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from randomized_row_action.datasets import (
    load_moffett_mat,
    make_synthetic_simplex_nmf,
    normalize_simplex_columns,
    preprocess_hsi_global_scale,
)
from randomized_row_action.simplex_nmf import entropy_row_action_simplex_nmf

from ._common import prepare_output_dir, save_figure, save_history, save_json


@dataclass(frozen=True)
class Profile:
    outer_iterations: int
    warmup_outer: int
    verbose_every: int | None


PROFILES = {
    "quick": Profile(20, 5, 5),
    "paper": Profile(800, 40, 100),
}


def _sort_components(
    W: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norms = np.linalg.norm(W, axis=0)
    order = np.argsort(norms)
    return W[:, order], H[order], order, norms[order]


def _history_with_aliases(history: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    saved = dict(history)
    saved["rel_error"] = history["relative_error"]
    return saved


def _save_abundance_maps(
    H: np.ndarray,
    output_path: Path,
    *,
    image_shape: tuple[int, int],
    names: list[str],
    show: bool,
) -> None:
    rank, n_samples = H.shape
    if np.prod(image_shape) != n_samples:
        raise ValueError(f"image_shape={image_shape} does not match {n_samples} samples.")
    fig, axes = plt.subplots(1, rank, figsize=(3.2 * rank, 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for component, ax in enumerate(axes):
        image = H[component].reshape(image_shape, order="F")
        shown = ax.imshow(image, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(names[component])
        ax.set_xticks([])
        ax.set_yticks([])
        colorbar = fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=8)
    save_figure(fig, output_path)
    if not show:
        plt.close(fig)


def _save_spectra(W: np.ndarray, output_path: Path, names: list[str], show: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    bands = np.arange(W.shape[0])
    for component in range(W.shape[1]):
        ax.plot(bands, W[:, component], linewidth=2.0, label=names[component])
    ax.set_xlabel("Spectral band index")
    ax.set_ylabel("Scaled reflectance / intensity")
    ax.set_title("Estimated spectral signatures")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    save_figure(fig, output_path)
    if not show:
        plt.close(fig)


def _save_convergence(history: dict[str, np.ndarray], output_path: Path, show: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    axes[0].semilogy(history["outer_iteration"], history["relative_error"], linewidth=2.0)
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel(r"$\|X-WH\|_F/\|X\|_F$")
    axes[0].set_title("Error vs iteration")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogy(history["time"], history["relative_error"], linewidth=2.0)
    axes[1].set_xlabel("CPU time (seconds)")
    axes[1].set_ylabel(r"$\|X-WH\|_F/\|X\|_F$")
    axes[1].set_title("Error vs CPU time")
    axes[1].grid(True, which="both", alpha=0.3)
    save_figure(fig, output_path)
    if not show:
        plt.close(fig)


def _save_relative_error(history: dict[str, np.ndarray], output_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(history["outer_iteration"], history["relative_error"], linewidth=2.0)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"$\|X-WH\|_F / \|X\|_F$")
    ax.set_title("Moffett: relative reconstruction error")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)
    if not show:
        plt.close(fig)


def _save_loss(history: dict[str, np.ndarray], output_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(history["outer_iteration"], history["loss"], linewidth=2.0)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"$\frac{1}{2}\|X-WH\|_F^2$")
    ax.set_title("Moffett: loss")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)
    if not show:
        plt.close(fig)


def _save_relative_error_time(
    history: dict[str, np.ndarray],
    output_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(history["time"], history["relative_error"], linewidth=2.0)
    ax.set_xlabel("CPU time (seconds)")
    ax.set_ylabel(r"$\|X-WH\|_F / \|X\|_F$")
    ax.set_title("Moffett: relative error vs CPU time")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)
    if not show:
        plt.close(fig)


def _save_stepsizes(history: dict[str, np.ndarray], output_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["outer_iteration"], history["eta_H"], label=r"$\eta_H$", linewidth=2.0)
    ax.plot(history["outer_iteration"], history["eta_W"], label=r"$\eta_W$", linewidth=2.0)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Step size")
    ax.set_title("Moffett: learned block stepsizes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)
    if not show:
        plt.close(fig)


def run(
    mode: str,
    profile_name: str,
    output_dir: Path,
    data_path: Path | None,
    rank: int,
    seed: int,
    show: bool,
) -> None:
    profile = PROFILES[profile_name]
    output = prepare_output_dir(output_dir)

    if mode == "moffett":
        if data_path is None:
            raise ValueError("--data is required in moffett mode.")
        X_raw = load_moffett_mat(data_path)
        X, scale = preprocess_hsi_global_scale(X_raw)
        image_shape = (50, 50)
        component_names = ["Water", "Vegetation", "Soil"] if rank == 3 else [
            f"Component {index + 1}" for index in range(rank)
        ]
        W0 = None
        H0 = None
    else:
        X, W_true, H_true = make_synthetic_simplex_nmf(
            m=40 if profile_name == "quick" else 80,
            rank=rank,
            n_samples=200 if profile_name == "quick" else 1200,
            dirichlet_alpha=0.3,
            seed=0,
        )
        rng = np.random.default_rng(seed)
        W0 = np.maximum(W_true * (1.0 + 0.15 * rng.standard_normal(W_true.shape)), 1e-15)
        H0 = normalize_simplex_columns(H_true + 0.15 * rng.standard_normal(H_true.shape))
        scale = 1.0
        image_shape = (1, X.shape[1])
        component_names = [f"Component {index + 1}" for index in range(rank)]

    result = entropy_row_action_simplex_nmf(
        X,
        rank=rank,
        n_outer=profile.outer_iterations,
        eta_H0=1.0,
        eta_W0=0.01,
        eta_H_max=2.0,
        eta_W_max=0.1,
        growth=1.02,
        decay=0.5,
        line_search_mode="warmup",
        warmup_outer=profile.warmup_outer,
        max_backtracks=20,
        seed=seed,
        W0=W0,
        H0=H0,
        verbose_every=profile.verbose_every,
    )
    W, H, order, component_norms = _sort_components(result.W, result.H)
    history = _history_with_aliases(result.history)

    save_history(output / "history.npz", history)
    np.savez_compressed(
        output / "factors.npz",
        W=W,
        H=H,
        component_order=order,
        component_norms=component_norms,
        scale=scale,
    )
    _save_relative_error(history, output / "fig_moffett_relative_error_r3.pdf", show)
    _save_loss(history, output / "fig_moffett_loss_r3.pdf", show)
    _save_relative_error_time(history, output / "fig_moffett_relative_error_time_r3.pdf", show)
    _save_stepsizes(history, output / "fig_moffett_stepsizes_r3.pdf", show)
    _save_convergence(history, output / "fig_moffett_convergence_r3.pdf", show)
    _save_spectra(W, output / "fig_moffett_spectral_signatures_r3.pdf", component_names, show)
    if mode == "moffett":
        _save_abundance_maps(
            H,
            output / "fig_moffett_abundance_maps_r3.pdf",
            image_shape=image_shape,
            names=component_names,
            show=show,
        )

    save_json(
        output / "summary.json",
        {
            "profile": asdict(profile),
            "mode": mode,
            "rank": rank,
            "seed": seed,
            "data_path": data_path,
            "global_scale": scale,
            "component_order": order,
            "component_norms": component_norms,
            "final_relative_error": history["relative_error"][-1],
            "final_loss": history["loss"][-1],
            "final_simplex_sum_violation": history["H_simplex_sum_violation"][-1],
            "final_H_negativity_violation": history["H_negativity_violation"][-1],
            "final_W_negativity_violation": history["W_negativity_violation"][-1],
        },
    )
    if show:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "moffett"), default="synthetic")
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("results/simplex_nmf"))
    parser.add_argument("--data", type=Path, default=None, help="Path to Moffet.mat.")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.mode, args.profile, args.output_dir, args.data, args.rank, args.seed, args.show)


if __name__ == "__main__":
    main()
