# Data

The synthetic NNLS and entropy-selector experiments generate their data at runtime.
No external file is required.

The Moffett Field experiment requires a MATLAB file containing a nonnegative matrix
of size `159 x 2500` (or its transpose). The file is intentionally not committed to
this repository.

Place the file here, for example:

```text
data/Moffet.mat
```

Then run:

```bash
python -m experiments.simplex_nmf \
  --mode moffett \
  --profile paper \
  --data data/Moffet.mat \
  --output-dir results/simplex_nmf
```

The loader searches the MAT file for a numeric matrix with shape `159 x 2500` or
`2500 x 159`, clips small negative numerical artifacts, and applies one global
scaling so that the maximum entry is one.

A copy of the dataset used in the exploratory notebook was obtained from the data
folder of the public NMF-book code repository referenced in that notebook. Before
public release, verify the preferred long-term download link and the redistribution
terms of the data.
