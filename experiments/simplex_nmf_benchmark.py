"""Final paper simplex-NMF comparison on the Moffett Field crop.

The retained methods are:

* fixed-step single-row/column entropic row-action method;
* fixed-step mini-batch-8 entropic row-action method;
* deterministic simplex-constrained reference from the sibling ``../nmfbook`` repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from randomized_row_action.datasets import (
    load_moffett_mat,
    normalize_simplex_columns,
    preprocess_hsi_global_scale,
)
from randomized_row_action.simplex_nmf import (
    entropy_H_sweep,
    entropy_W_sweep,
    nmf_loss,
    nonnegativity_violation,
    relative_reconstruction_error,
    simplex_violation,
)

METHODS = ("single_row_fixed", "minibatch8_fixed", "reference")
DISPLAY_NAMES = {
    "single_row_fixed": "Single row",
    "minibatch8_fixed": "Mini-batch 8",
    "reference": "Deterministic reference",
}
COMPONENT_LABELS = ("Water", "Vegetation", "Soil")
FLOOR = 1e-15
EXPONENT_CLIP = 50.0


@dataclass(frozen=True)
class ComparisonConfig:
    seed: int = 123
    rank: int = 3
    n_outer: int = 200
    eta_H: float = 0.1
    eta_W: float = 0.01
    batch_size: int = 8
    data_path: str = "data/Moffet.mat"
    reference_timemax: float = 60.0
    reference_inneriter: int = 10


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_value(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_problem(data_path: Path) -> tuple[np.ndarray, float]:
    if not data_path.exists():
        raise FileNotFoundError(f"Moffett data file not found: {data_path}")
    X_raw = load_moffett_mat(data_path)
    return preprocess_hsi_global_scale(X_raw)


def initial_factors(X: np.ndarray, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W0 = rng.random((X.shape[0], rank)) + 1e-2
    H0 = rng.dirichlet(np.ones(rank), size=X.shape[1]).T
    scale = np.linalg.norm(X, "fro") / max(np.linalg.norm(W0 @ H0, "fro"), FLOOR)
    return np.maximum(scale * W0, FLOOR), H0


def paper_sampling_rng(X: np.ndarray, rank: int, seed: int) -> np.random.Generator:
    """Advance through the paper initialization draws before stochastic sweeps."""
    rng = np.random.default_rng(seed)
    rng.random((X.shape[0], rank))
    rng.dirichlet(np.ones(rank), size=X.shape[1])
    return rng


def empty_history() -> dict[str, list[float]]:
    return {
        "outer_iteration": [],
        "objective": [],
        "relative_error": [],
        "time": [],
        "H_update_time": [],
        "W_update_time": [],
        "simplex_violation": [],
        "H_negativity": [],
        "W_negativity": [],
        "eta_H": [],
        "eta_W": [],
    }


def record_history(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    history: dict[str, list[float]],
    outer: int,
    started: float,
    *,
    h_time: float,
    w_time: float,
    eta_H: float,
    eta_W: float,
) -> None:
    simplex_sum, h_negativity = simplex_violation(H)
    history["outer_iteration"].append(float(outer))
    history["objective"].append(nmf_loss(X, W, H))
    history["relative_error"].append(relative_reconstruction_error(X, W, H))
    history["time"].append(perf_counter() - started)
    history["H_update_time"].append(h_time)
    history["W_update_time"].append(w_time)
    history["simplex_violation"].append(simplex_sum)
    history["H_negativity"].append(h_negativity)
    history["W_negativity"].append(nonnegativity_violation(W))
    history["eta_H"].append(float(eta_H))
    history["eta_W"].append(float(eta_W))


def as_arrays(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(values, dtype=float) for key, values in history.items()}


def run_single_row_fixed(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    config: ComparisonConfig,
) -> dict[str, Any]:
    W = W0.copy()
    H = normalize_simplex_columns(H0.copy())
    rng = paper_sampling_rng(X, config.rank, config.seed)
    started = perf_counter()
    h_time = 0.0
    w_time = 0.0
    history = empty_history()
    record_history(
        X,
        W,
        H,
        history,
        0,
        started,
        h_time=h_time,
        w_time=w_time,
        eta_H=config.eta_H,
        eta_W=config.eta_W,
    )
    for outer in range(1, config.n_outer + 1):
        row_order = rng.permutation(X.shape[0])
        block_started = perf_counter()
        H = entropy_H_sweep(
            X,
            W,
            H,
            eta=config.eta_H,
            row_order=row_order,
            floor=FLOOR,
            exponent_clip=EXPONENT_CLIP,
        )
        h_time += perf_counter() - block_started
        column_order = rng.permutation(X.shape[1])
        block_started = perf_counter()
        W = entropy_W_sweep(
            X,
            W,
            H,
            eta=config.eta_W,
            column_order=column_order,
            floor=FLOOR,
            exponent_clip=EXPONENT_CLIP,
        )
        w_time += perf_counter() - block_started
        record_history(
            X,
            W,
            H,
            history,
            outer,
            started,
            h_time=h_time,
            w_time=w_time,
            eta_H=config.eta_H,
            eta_W=config.eta_W,
        )
    return {"W": W, "H": H, "history": as_arrays(history)}


def minibatch_H_update(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    rng: np.random.Generator,
    *,
    eta: float,
    batch_size: int,
) -> np.ndarray:
    updated = H.copy()
    n_batches = int(np.ceil(W.shape[0] / batch_size))
    for _batch in range(n_batches):
        rows = rng.choice(W.shape[0], size=min(batch_size, W.shape[0]), replace=False)
        W_batch = W[rows]
        residual = W_batch @ updated - X[rows]
        gradient = W_batch.T @ residual
        lipschitz = max(float(np.linalg.norm(W_batch, 2) ** 2), FLOOR)
        updated *= np.exp(np.clip(-eta * gradient / lipschitz, -EXPONENT_CLIP, EXPONENT_CLIP))
        updated = normalize_simplex_columns(updated, floor=FLOOR)
    return updated


def minibatch_W_update(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    rng: np.random.Generator,
    *,
    eta: float,
    batch_size: int,
) -> np.ndarray:
    updated = W.copy()
    n_batches = int(np.ceil(H.shape[1] / batch_size))
    for _batch in range(n_batches):
        columns = rng.choice(H.shape[1], size=min(batch_size, H.shape[1]), replace=False)
        H_batch = H[:, columns]
        residual = updated @ H_batch - X[:, columns]
        gradient = residual @ H_batch.T
        lipschitz = max(float(np.linalg.norm(H_batch, 2) ** 2), FLOOR)
        updated *= np.exp(np.clip(-eta * gradient / lipschitz, -EXPONENT_CLIP, EXPONENT_CLIP))
        updated = np.maximum(updated, FLOOR)
    return updated


def run_minibatch8_fixed(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    config: ComparisonConfig,
) -> dict[str, Any]:
    W = W0.copy()
    H = normalize_simplex_columns(H0.copy())
    rng = np.random.default_rng(config.seed)
    started = perf_counter()
    h_time = 0.0
    w_time = 0.0
    history = empty_history()
    record_history(
        X,
        W,
        H,
        history,
        0,
        started,
        h_time=h_time,
        w_time=w_time,
        eta_H=config.eta_H,
        eta_W=config.eta_W,
    )
    for outer in range(1, config.n_outer + 1):
        block_started = perf_counter()
        H = minibatch_H_update(X, W, H, rng, eta=config.eta_H, batch_size=config.batch_size)
        h_time += perf_counter() - block_started
        block_started = perf_counter()
        W = minibatch_W_update(X, W, H, rng, eta=config.eta_W, batch_size=config.batch_size)
        w_time += perf_counter() - block_started
        record_history(
            X,
            W,
            H,
            history,
            outer,
            started,
            h_time=h_time,
            w_time=w_time,
            eta_H=config.eta_H,
            eta_W=config.eta_W,
        )
    return {"W": W, "H": H, "history": as_arrays(history)}


def load_reference() -> tuple[Any, Any, str]:
    candidate = Path(__file__).resolve().parents[2] / "nmfbook"
    if not candidate.exists():
        raise FileNotFoundError(f"Reference repository not found: {candidate}")
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    from algorithms.minvol_nmf.minvol_nmf import MinVolNMFOptions, minvol_nmf

    return MinVolNMFOptions, minvol_nmf, "algorithms.minvol_nmf.minvol_nmf:minvol_nmf"


def run_reference(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    config: ComparisonConfig,
) -> dict[str, Any]:
    MinVolNMFOptions, minvol_nmf, import_path = load_reference()
    options = MinVolNMFOptions(
        maxiter=config.n_outer,
        timemax=config.reference_timemax,
        inneriter=config.reference_inneriter,
        random_state=config.seed,
        lam=0.0,
        model=4,
        W=W0.copy(),
        H=H0.copy(),
        display=0,
    )
    started = perf_counter()
    W, H, objective_values, time_values = minvol_nmf(X, config.rank, options)
    elapsed = perf_counter() - started
    objective_values = np.asarray(objective_values, dtype=float)
    time_values = np.asarray(time_values, dtype=float)
    if time_values.size + 1 == objective_values.size:
        time_values = np.concatenate(([0.0], time_values))
    elif time_values.size != objective_values.size:
        time_values = np.linspace(0.0, elapsed, objective_values.size)

    history = empty_history()
    initial_objective = nmf_loss(X, W0, H0)
    history["outer_iteration"].append(0.0)
    history["objective"].append(initial_objective)
    history["relative_error"].append(relative_reconstruction_error(X, W0, H0))
    history["time"].append(0.0)
    history["H_update_time"].append(np.nan)
    history["W_update_time"].append(np.nan)
    simplex_sum, h_negativity = simplex_violation(H0)
    history["simplex_violation"].append(simplex_sum)
    history["H_negativity"].append(h_negativity)
    history["W_negativity"].append(nonnegativity_violation(W0))
    history["eta_H"].append(np.nan)
    history["eta_W"].append(np.nan)

    norm_X = max(float(np.linalg.norm(X, "fro")), np.finfo(float).tiny)
    for index, objective in enumerate(objective_values, start=1):
        history["outer_iteration"].append(float(index))
        history["objective"].append(0.5 * float(objective))
        history["relative_error"].append(float(np.sqrt(max(objective, 0.0)) / norm_X))
        history["time"].append(float(time_values[index - 1]))
        history["H_update_time"].append(np.nan)
        history["W_update_time"].append(np.nan)
        simplex_sum, h_negativity = simplex_violation(H if index == objective_values.size else H0)
        history["simplex_violation"].append(simplex_sum)
        history["H_negativity"].append(h_negativity)
        history["W_negativity"].append(nonnegativity_violation(W if index == objective_values.size else W0))
        history["eta_H"].append(np.nan)
        history["eta_W"].append(np.nan)
    return {"W": W, "H": H, "history": as_arrays(history), "reference_import_path": import_path}


def run_method(
    method: str,
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    config: ComparisonConfig,
) -> dict[str, Any]:
    if method == "single_row_fixed":
        return run_single_row_fixed(X, W0, H0, config)
    if method == "minibatch8_fixed":
        return run_minibatch8_fixed(X, W0, H0, config)
    if method == "reference":
        return run_reference(X, W0, H0, config)
    raise ValueError(f"Unknown method: {method}")


def save_method_outputs(
    method: str,
    output_dir: Path,
    W0: np.ndarray,
    H0: np.ndarray,
    result: dict[str, Any],
    config: ComparisonConfig,
) -> dict[str, Any]:
    config_data = asdict(config)
    config_data["method"] = method
    if method == "reference":
        config_data["reference_import_path"] = result["reference_import_path"]
        config_data["reference_call"] = "minvol_nmf(X, rank, MinVolNMFOptions(model=4, lam=0, W=W0, H=H0))"
    save_json(output_dir / f"config_{method}_seed{config.seed}.json", config_data)
    np.savez_compressed(output_dir / f"history_{method}_seed{config.seed}.npz", **result["history"])
    np.savez_compressed(
        output_dir / f"factors_{method}_seed{config.seed}.npz",
        W_initial=W0,
        H_initial=H0,
        W_final=result["W"],
        H_final=result["H"],
    )
    history = result["history"]
    return {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "objective": float(history["objective"][-1]),
        "relative_error": float(history["relative_error"][-1]),
        "runtime": float(history["time"][-1]),
        "simplex_violation": float(history["simplex_violation"][-1]),
        "H_negativity": float(history["H_negativity"][-1]),
        "W_negativity": float(history["W_negativity"][-1]),
    }


def load_saved_factors(output_dir: Path, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    factors = {}
    for method in METHODS:
        data = np.load(output_dir / f"factors_{method}_seed{seed}.npz")
        factors[method] = (data["W_final"], data["H_final"])
    return factors


def load_saved_histories(output_dir: Path, seed: int) -> dict[str, dict[str, np.ndarray]]:
    histories = {}
    for method in METHODS:
        with np.load(output_dir / f"history_{method}_seed{seed}.npz") as data:
            histories[method] = {key: data[key] for key in data.files}
    return histories


def align_to_reference(
    factors: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[int]]]:
    W_ref, H_ref = factors["reference"]

    def normalized_columns(A: np.ndarray) -> np.ndarray:
        return A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), np.finfo(float).tiny)

    aligned = {"reference": (W_ref, H_ref)}
    permutations = {"reference": list(range(W_ref.shape[1]))}
    reference_columns = normalized_columns(W_ref)
    for method in ("single_row_fixed", "minibatch8_fixed"):
        W, H = factors[method]
        correlations = np.abs(normalized_columns(W).T @ reference_columns)
        row_index, column_index = linear_sum_assignment(-correlations)
        permutation = [int(row_index[np.where(column_index == target)[0][0]]) for target in range(W.shape[1])]
        aligned[method] = (W[:, permutation], H[permutation])
        permutations[method] = permutation
    return aligned, permutations


def save_alignment(output_dir: Path, permutations: dict[str, list[int]]) -> None:
    save_json(
        output_dir / "component_alignment_seed123.json",
        {
            "criterion": "maximize absolute spectral correlation with reference W columns",
            "component_labels": list(COMPONENT_LABELS),
            "note": "Physical material labels were validated after alignment to the reference ordering.",
            "permutations": permutations,
        },
    )


def save_abundance_figure(
    aligned: dict[str, tuple[np.ndarray, np.ndarray]],
    figure_dir: Path,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(COMPONENT_LABELS),
        len(METHODS),
        figsize=(7.2, 6.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    image_handle = None
    for col, method in enumerate(METHODS):
        _W, H = aligned[method]
        for row, label in enumerate(COMPONENT_LABELS):
            ax = axes[row, col]
            image = H[row].reshape((50, 50), order="F")
            image_handle = ax.imshow(
                image,
                cmap="gray_r",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(DISPLAY_NAMES[method], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image_handle, ax=axes, fraction=0.025, pad=0.01, label="Abundance")
    pdf_path = figure_dir / "fig_moffett_abundance_comparison_r3.pdf"
    png_path = output_dir / "fig_moffett_abundance_comparison_r3.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_spectral_figure(
    aligned: dict[str, tuple[np.ndarray, np.ndarray]],
    figure_dir: Path,
    output_dir: Path,
) -> None:
    y_max = max(float(np.max(W)) for W, _H in aligned.values())
    fig, axes = plt.subplots(1, len(METHODS), figsize=(7.2, 3.0), constrained_layout=True, sharey=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, method in zip(axes, METHODS, strict=True):
        W, _H = aligned[method]
        bands = np.arange(W.shape[0])
        for component, label in enumerate(COMPONENT_LABELS):
            ax.plot(bands, W[:, component], linewidth=1.8, color=colors[component], label=label)
        ax.set_title(DISPLAY_NAMES[method], fontsize=10)
        ax.set_xlabel("Spectral band")
        ax.set_xlim(0, W.shape[0] - 1)
        ax.set_ylim(0.0, 1.05 * y_max)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Intensity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    pdf_path = figure_dir / "fig_moffett_spectral_comparison_r3.pdf"
    png_path = output_dir / "fig_moffett_spectral_comparison_r3.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_convergence_figure(
    histories: dict[str, dict[str, np.ndarray]],
    figure_dir: Path,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for method in METHODS:
        history = histories[method]
        axes[0].semilogy(history["outer_iteration"], history["relative_error"], linewidth=1.9, label=DISPLAY_NAMES[method])
        axes[1].semilogy(history["time"], history["relative_error"], linewidth=1.9, label=DISPLAY_NAMES[method])
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel("Relative reconstruction error")
    axes[1].set_xlabel("CPU time (seconds)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    pdf_path = figure_dir / "fig_moffett_convergence_comparison_r3.pdf"
    png_path = output_dir / "fig_moffett_convergence_comparison_r3.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def regenerate_figures(output_dir: Path, figure_dir: Path, seed: int) -> None:
    factors = load_saved_factors(output_dir, seed)
    aligned, permutations = align_to_reference(factors)
    save_alignment(output_dir, permutations)
    save_abundance_figure(aligned, figure_dir, output_dir)
    save_spectral_figure(aligned, figure_dir, output_dir)
    histories = load_saved_histories(output_dir, seed)
    save_convergence_figure(histories, figure_dir, output_dir)


def run_comparison(config: ComparisonConfig, output_dir: Path, figure_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    X, scale = load_problem(Path(config.data_path))
    W0, H0 = initial_factors(X, config.rank, config.seed)
    np.savez_compressed(output_dir / f"initial_factors_seed{config.seed}.npz", W0=W0, H0=H0)
    save_json(
        output_dir / "common_config_seed123.json",
        {
            **asdict(config),
            "X_shape": X.shape,
            "global_scale": scale,
            "component_labels": list(COMPONENT_LABELS),
        },
    )
    rows = []
    for method in METHODS:
        result = run_method(method, X, W0.copy(), H0.copy(), config)
        rows.append(save_method_outputs(method, output_dir, W0, H0, result, config))
    write_csv(output_dir / "summary_seed123.csv", rows)
    regenerate_figures(output_dir, figure_dir, config.seed)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final Moffett simplex-NMF comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-outer", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/04_simplex_nmf_comparison"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/simplex_nmf_comparison"))
    parser.add_argument("--data", type=str, default="data/Moffet.mat")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ComparisonConfig(seed=args.seed, n_outer=args.n_outer, data_path=args.data)
    rows = run_comparison(config, args.output_dir, args.figure_dir)
    print("Final simplex-NMF comparison")
    for row in rows:
        print(
            f"{row['display_name']:>24s}: objective={row['objective']:.6e}, "
            f"rel.err={row['relative_error']:.6e}, time={row['runtime']:.3f}s"
        )


if __name__ == "__main__":
    main()
