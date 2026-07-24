# Randomized Row-Action Framework

## Project description

This repository contains the Python implementations and numerical experiments associated with the paper

> **A Unified Framework for Randomized Row-Action Methods for Structured Data Fitting and Constraint Geometry**  
> Valentin Leplat, Zeyu Dong, and Junfeng Yin.

The project studies randomized row-action methods for problems with linear observations, convex constraints, and non-Euclidean geometry. It implements the two algorithmic branches developed in the paper:

1. **Soft constrained data fitting:** primal projected or Bregman-proximal row steps, including noisy nonnegative least squares.
2. **Hard equality-constrained selectors:** dual Bregman--Kaczmarz steps, including the entropy selector on the simplex.

The repository also contains simplex-structured nonnegative matrix factorization experiments. They illustrate how entropy row-action updates can be used as inner block solvers in a nonconvex alternating factorization scheme, and compare a fixed-step single-row method, a fixed-step mini-batch-8 variant, and an optimized deterministic simplex-constrained NMF reference.

The numerical contributions are intentionally focused. Their purpose is to illustrate the theoretical mechanisms of the paper and to reproduce the reported figures.

---

## Paper and citation

This repository is the reproducibility companion for the paper. The manuscript source is not part of the public code release; the implementation and experiment scripts below are the reference materials for reproducing the numerical figures.

A temporary citation entry is:

```bibtex
@unpublished{leplat2026rowaction,
  title  = {Toward a Unified Framework for Randomized Row-Action Methods
            for Structured Data Fitting and Constraint Geometry},
  author = {Leplat, Valentin and Dong, Zeyu and Yin, Junfeng},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

Please update this entry after submission or publication.

---

## Repository structure

```text
randomized-row-action-framework/
|
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
|-- LICENSE
|-- CITATION.cff
|-- Makefile
|
|-- src/randomized_row_action/
|   |-- __init__.py
|   |-- nnls.py              # Soft projected row-action NNLS
|   |-- bregman.py           # Entropy Bregman--Kaczmarz selector
|   |-- simplex_nmf.py       # Entropy row-action simplex-NMF
|   |-- datasets.py          # Synthetic data and Moffett loader
|   `-- _utils.py            # Validation and weighted row sampling
|
|-- experiments/
|   |-- noisy_nnls.py
|   |-- entropy_selector.py
|   |-- simplex_nmf.py
|   |-- simplex_nmf_benchmark.py
|   `-- _common.py
|
|-- notebooks/
|   |-- 01_noisy_nnls.ipynb
|   |-- 02_entropy_selector.ipynb
|   |-- 03_simplex_nmf.ipynb
|   |-- 04_simplex_nmf_comparison.ipynb
|   `-- README.md
|
|-- tests/
|   |-- test_nnls.py
|   |-- test_bregman.py
|   |-- test_simplex_nmf.py
|   `-- test_04_simplex_nmf_comparison.py
|
|-- data/
|   |-- Moffet.mat
|   `-- README.md
|
`-- figures/
    `-- .gitkeep
```

The `src/` directory is the reusable library. The files under `experiments/` are the canonical scripts for reproducing the paper results. The notebooks preserve readable experiment workflows and are useful for inspection, but the standalone scripts are the reference path for full process reproduction.

---

## Implemented methods

### Projected randomized row action for noisy NNLS

The solver addresses

```text
minimize  0.5 ||A x - b||_2^2
subject to x >= 0.
```

At each iteration, row `i` is sampled with probability proportional to `||a_i||_2^2`, and the method applies

```text
x <- max(x - eta_k * (a_i^T x - b_i) * a_i / ||a_i||_2^2, 0).
```

The implementation records both the last iterate and the weighted average. By default, the average is

```text
x_bar = sum_k eta_k x_k / sum_k eta_k,
```

which matches the indexing used in the theorem. A legacy `post_update` option is available only to compare with the original exploratory notebook.

### Entropy Bregman--Kaczmarz selector

The hard equality branch solves

```text
minimize  sum_j x_j log(x_j)
subject to A x = b and x in the simplex.
```

The dual variable is updated row by row, and the primal point is reconstructed by the softmax map. Every primal iterate therefore remains in the relative interior of the simplex.

The implementation monitors:

- `KL(x_hat || x_k)`;
- `||A x_k - b||_2^2`;
- the local error-bound ratio `KL(x_hat || x_k) / ||A x_k - b||_2^2`;
- the relative infinity-distance used to verify the local neighborhood.

### Entropy row-action simplex-NMF

The illustrative factorization problem is

```text
minimize  0.5 ||X - W H||_F^2
subject to W >= 0 and every column of H belongs to the simplex.
```

The repository contains two randomized row-action variants:

- a single-row/column method using the fixed stepsizes `eta_H=0.1` and `eta_W=0.01`;
- a mini-batch variant using the same fixed stepsizes and blocks of size `8`.

For the abundance block, the methods use entropic updates followed by column normalization, which preserves the simplex constraint. For the basis block, they use positive-entropy multiplicative updates.

The final experiment compares these two variants with an optimized deterministic simplex-constrained NMF reference. The complete alternating problem is nonconvex; this experiment is an algorithmic illustration and the paper does not claim that its convex convergence results prove convergence of the full factorization scheme.

---

## Quick start

### Step 1: install Python

Python 3.10 or newer is recommended.

Check your version:

```bash
python3 --version
```

On Windows PowerShell, use:

```powershell
py --version
```

### Step 2: clone the repository

```bash
git clone https://github.com/vleplat/randomized-row-action-framework.git
cd randomized-row-action-framework
```

While the repository is private, use the private clone URL shown by GitHub.

### Step 3: create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 4: install the project

For the experiments only:

```bash
python -m pip install -e .
```

For development, tests, linting, and notebooks:

```bash
python -m pip install -e ".[dev,notebooks]"
```

An equivalent requirements-based installation is:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

### Step 5: run the checks

```bash
python -m pytest
python -m experiments.simplex_nmf --mode synthetic --profile quick
```

The NNLS and entropy selector scripts reproduce the full paper experiments and therefore are not used as quick smoke tests. The simplex-NMF synthetic quick profile verifies the factorization pipeline without external data.

Or use the Makefile for tests and linting:

```bash
make test
make lint
```

The full paper commands are listed below.

---

## Reproducing the paper experiments

### 1. Noisy nonnegative least squares

```bash
python -m experiments.noisy_nnls
```

Main outputs:

```text
figures/nnls_noisy/fig_nnls_noisy_weighted_average_gap.pdf
figures/nnls_noisy/fig_nnls_noisy_last_iterate_gap.pdf
figures/nnls_noisy/fig_nnls_noisy_stepsize_schedule.pdf
figures/nnls_noisy/fig_nnls_noisy_constant_eta_floor.pdf
figures/nnls_noisy/fig_nnls_noisy_horizon_rate.pdf
figures/nnls_noisy/fig_nnls_noisy_avg_horizon_rate.pdf
figures/nnls_noisy/fig_nnls_noisy_horizon_rate_last_iterate.pdf
```

The paper profile uses the dimensions, noise level, horizons, and repetition counts from the numerical section. It may take time because the longest runs contain two million randomized row updates and are repeated across several seeds.

### 2. Entropy selector on the simplex

```bash
python -m experiments.entropy_selector
```

Main outputs:

```text
figures/entropy_selector/fig_bregman_selector_kl_decay.pdf
figures/entropy_selector/fig_bregman_selector_residual_decay.pdf
figures/entropy_selector/fig_bregman_selector_error_bound_ratio.pdf
figures/entropy_selector/fig_bregman_selector_repeated_kl.pdf
```

The script prints the local theoretical constants, contraction diagnostics, repeated-run statistics, and neighborhood checks.

### 3. Simplex-structured NMF on Moffett Field

Download the Moffett MAT file and place it under `data/`; see `data/README.md`. Then run:

```bash
python -m experiments.simplex_nmf \
  --mode moffett \
  --profile paper \
  --data data/Moffet.mat \
  --rank 3 \
  --output-dir results/simplex_nmf
```

Main outputs:

```text
results/simplex_nmf/fig_moffett_abundance_maps_r3.pdf
results/simplex_nmf/fig_moffett_spectral_signatures_r3.pdf
results/simplex_nmf/fig_moffett_convergence_r3.pdf
results/simplex_nmf/fig_moffett_relative_error_r3.pdf
results/simplex_nmf/fig_moffett_loss_r3.pdf
results/simplex_nmf/fig_moffett_relative_error_time_r3.pdf
results/simplex_nmf/fig_moffett_stepsizes_r3.pdf
results/simplex_nmf/factors.npz
results/simplex_nmf/history.npz
results/simplex_nmf/summary.json
```

#### Arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--mode` | `synthetic` | Use `synthetic` without external data or `moffett` for the paper experiment. |
| `--profile` | `quick` | Short validation or full paper configuration. |
| `--data` | none | Path to `Moffet.mat`; required in Moffett mode. |
| `--rank` | `3` | Factorization rank. |
| `--seed` | `123` | Initialization and row-order seed. |
| `--output-dir` | `results/simplex_nmf` | Output directory. |
| `--show` | off | Display figures interactively. |

The paper labels the three components as water, vegetation, and soil after sorting them by increasing column norm of `W`. The component order is saved in `factors.npz` and `summary.json`. Verify these labels visually if algorithmic parameters or seeds are changed.

The contextual Moffett scene/crop image currently shown in the manuscript is not generated by this script. Before writing that *all* figures are reproducible from the repository, replace that image by one generated from distributable source data or document the required permission and generation procedure.

### 4. Final simplex-NMF comparison with deterministic reference

The fourth experiment generates the final paper comparison for the Moffett simplex-NMF study. It compares exactly three retained methods:

- fixed-step single-row/column entropic row-action updates;
- fixed-step mini-batch-8 entropic row-action updates;
- the optimized deterministic simplex-constrained NMF reference provided by the external [`nmfbook`](https://github.com/vleplat/nmfbook.git) repository.

The external competitor is not bundled here and is not installed by default. Install it next to this repository before running the reference comparison.

Install the external competitor outside this repository:

```bash
git clone https://github.com/vleplat/nmfbook.git ../nmfbook
python -m pip install -r ../nmfbook/requirements.txt
export PYTHONPATH="$(pwd)/../nmfbook:$PYTHONPATH"
```

Then run the final comparison:

```bash
python -m experiments.simplex_nmf_benchmark \
  --seed 123 \
  --n-outer 200 \
  --output-dir results/04_simplex_nmf_comparison \
  --figure-dir figures/simplex_nmf_comparison
```

The script uses the same Moffett crop and preprocessing as `experiments.simplex_nmf.py`, with rank `3`, seed `123`, and 200 outer iterations. It creates one common initialization (`W0`, `H0`) and passes exact copies to all three methods. The single-row method uses fixed stepsizes `eta_H=0.1`, `eta_W=0.01`, no warmup, and no line search. The mini-batch method uses the same fixed stepsizes with batch size `8`.

The deterministic reference uses the implementation accompanying the Python version of Nicolas Gillis's NMF book. It applies two-block coordinate descent: with one factor fixed, the other convex block subproblem is solved using a Nesterov-type accelerated projected-gradient method. Projection is performed onto the nonnegative orthant for `W` and onto the product of simplices for `H`. The implementation is called with `model=4` and `lam=0.0`, so the log-determinant volume penalty is disabled and the comparison concerns the same reconstruction objective and simplex constraints.

Main outputs:

```text
figures/simplex_nmf_comparison/fig_moffett_abundance_comparison_r3.pdf
figures/simplex_nmf_comparison/fig_moffett_spectral_comparison_r3.pdf
figures/simplex_nmf_comparison/fig_moffett_convergence_comparison_r3.pdf
results/04_simplex_nmf_comparison/summary_seed123.csv
results/04_simplex_nmf_comparison/history_single_row_fixed_seed123.npz
results/04_simplex_nmf_comparison/history_minibatch8_fixed_seed123.npz
results/04_simplex_nmf_comparison/history_reference_seed123.npz
results/04_simplex_nmf_comparison/factors_single_row_fixed_seed123.npz
results/04_simplex_nmf_comparison/factors_minibatch8_fixed_seed123.npz
results/04_simplex_nmf_comparison/factors_reference_seed123.npz
```

The matching notebook `notebooks/04_simplex_nmf_comparison.ipynb` is a fully executed reproduction of the same experiment. It shows the problem setup, formulas, method execution, numerical table, automatic component alignment, final figures, and constraint checks.


For visualization, the components returned by the two row-action methods are aligned automatically to the deterministic reference by maximizing spectral correlation. The final figures use the validated material labels: water, vegetation, and soil.

---

## Using the core functions

### Noisy NNLS

```python
from randomized_row_action import (
    StepSchedule,
    make_noisy_nnls_data,
    nnls_objective,
    projected_row_action_nnls,
    solve_nnls_reference,
)

A, b, x_true, clean, noise = make_noisy_nnls_data(
    m=1000,
    n=200,
    sparsity=0.25,
    noise_level=0.05,
    seed=8,
)

x_star, info = solve_nnls_reference(A, b)
f_star = nnls_objective(A, b, x_star)

result = projected_row_action_nnls(
    A,
    b,
    n_iterations=200_000,
    schedule=StepSchedule("constant", 0.2),
    seed=20,
    x_star=x_star,
    f_star=f_star,
    record_every=A.shape[0],
)

x_last = result.x_last
x_average = result.x_average
history = result.history
```

Available schedules are:

```python
StepSchedule("constant", eta0=0.2)
StepSchedule("horizon_sqrt", eta0=0.9)
StepSchedule("diminishing_sqrt", eta0=0.9)
```

### Entropy Bregman--Kaczmarz selector

```python
from randomized_row_action import (
    entropy_bregman_kaczmarz,
    make_entropy_selector_instance,
)

instance = make_entropy_selector_instance(m=1000, n=200, seed=7)

result = entropy_bregman_kaczmarz(
    instance["A"],
    instance["b"],
    x_hat=instance["x_hat"],
    u0=instance["u0"],
    n_iterations=200 * 1000,
    eta=1.0,
    seed=20,
    record_every=200,
)

x_last = result.x_last
history = result.history
```

### Simplex-structured NMF

```python
from randomized_row_action import entropy_row_action_simplex_nmf

result = entropy_row_action_simplex_nmf(
    X,
    rank=3,
    n_outer=800,
    eta_H0=1.0,
    eta_W0=0.01,
    eta_H_max=2.0,
    eta_W_max=0.1,
    line_search_mode="warmup",
    warmup_outer=40,
    seed=123,
)

W = result.W
H = result.H
history = result.history
```

The solver also accepts user-provided `W0` and `H0`. The input `H0` is normalized columnwise before the first iteration.

---

## Notebooks

The notebooks are designed for reading, visual inspection, and interactive reruns of the four paper experiments. The standalone scripts remain the reference path for full-process reproduction.

After activating the project environment, start JupyterLab from the repository root:

```bash
jupyter lab
```

Then open one of:

```text
notebooks/01_noisy_nnls.ipynb
notebooks/02_entropy_selector.ipynb
notebooks/03_simplex_nmf.ipynb
notebooks/04_simplex_nmf_comparison.ipynb
```

Use **Restart Kernel and Run All Cells** before trusting notebook outputs, because notebooks can contain saved output from earlier executions.

---

## Tests and development checks

Run the unit tests:

```bash
python -m pytest
```

Run the linter:

```bash
python -m ruff check src experiments tests
```

Or use:

```bash
make test
make lint
```

The current test suite checks:

- nonnegativity preservation for projected NNLS;
- simplex preservation and residual decay for the entropy selector;
- shape, positivity, simplex feasibility, and finite histories for simplex-NMF;
- common initialization across the three final simplex-NMF methods;
- preservation of the fixed stepsizes and mini-batch size;
- simplex and nonnegativity feasibility in the final comparison;
- validity of the automatic component permutation;
- saving and reloading of the fourth experiment histories.

Before public release, a continuous-integration workflow should run these checks automatically on Linux, macOS, and Windows.

---

## Reproducibility notes

- Random seeds are explicit in all experiment scripts.
- The NNLS and entropy selector scripts run the paper-sized experiments directly and save their PDFs under `figures/`.
- The simplex-NMF script keeps `quick` and `paper` profiles and saves figures, histories, factors, and `summary.json` under `results/simplex_nmf`.
- The fourth experiment reproduces the final comparison between the fixed-step single-row method, the mini-batch-8 method, and the deterministic reference. Its figures are saved under `figures/simplex_nmf_comparison/`, and its numerical histories and factors under `results/04_simplex_nmf_comparison/`.
- Figures are generated by the scripts or notebooks, not copied from exploratory outputs.

The NNLS weighted average uses the pre-update iterates `x_k`, matching the theorem indexing.

---

## Headless execution

On a server without a graphical interface, set a noninteractive Matplotlib backend.

On macOS or Linux:

```bash
export MPLBACKEND=Agg
```

On Windows PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
```

Then run the experiment scripts normally.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'randomized_row_action'`

Install the repository in editable mode from its root:

```bash
python -m pip install -e .
```

Then confirm that the active Python interpreter belongs to the intended virtual environment.

### The Moffett file cannot be found

Pass the path explicitly:

```bash
python -m experiments.simplex_nmf --mode moffett --data /full/path/to/Moffet.mat
```

The expected matrix has shape `159 x 2500`, or its transpose.

### A paper experiment is slow

The full NNLS experiment contains long randomized trajectories and repeated runs. The script batches sampled row indices to reduce overhead, but it still performs the full paper-sized computation.

### Entropy updates overflow or underflow

The softmax implementation subtracts the largest dual component. The multiplicative NMF updates clip their exponent and use a small positive floor. If warnings remain after changing parameters, inspect the step sizes rather than increasing the clipping threshold blindly.

---

## 📄 License

This project is licensed under the **MIT License** (see `LICENSE`).

**Key points (MIT):**

- ✅ **Use**: you can use this software for any purpose
- ✅ **Modify & distribute**: you can modify, distribute, and sublicense it
- ✅ **Commercial use**: permitted
- ✅ **Attribution**: include the copyright and license notice in copies
- ✅ **No warranty**: the software is provided "as is"

## 📧 Support and Contact

For questions, bug reports, or contributions, please contact:
**valentin dot leplat [at] gmail dot com**
