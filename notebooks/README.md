# Notebooks

These notebooks are concise interfaces to the reusable package under `src/`.
They replace the original exploratory notebooks, which contained duplicated solver
implementations, notebook-specific paths, and generated outputs.

Install the repository from its root:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Then open:

- `01_noisy_nnls.ipynb`;
- `02_entropy_selector.ipynb`;
- `03_simplex_nmf.ipynb`.

The standalone scripts under `experiments/` are the canonical reproducibility path.
They run from a clean process and save configurations, histories, and figures.
