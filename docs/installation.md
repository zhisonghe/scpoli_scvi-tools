# Installation

## Requirements

- Python 3.10–3.13
- [scvi-tools](https://scvi-tools.org) ≥ 1.5.2, < 1.6. It is installed automatically and brings
  PyTorch, Lightning and AnnData. scpolivi uses some scvi-tools internals, so it pins the minor
  version.
- scikit-learn (installed automatically)

Optional:

- **scanpy, igraph, leidenalg** (extra `leiden`): Leiden initialization of unlabeled prototypes
  (the default, `clustering="leiden"`) and `scpolivi.graph.construct_sample_graphs`. Without
  them, train with `clustering="kmeans", n_clusters=...` and build graphs yourself.

!!! note "umap-learn"
    umap-learn 0.6.0 breaks `scanpy.tl.umap` (scanpy 1.12) on data with more than a few thousand
    cells (numba `TypingError`). The `leiden` extra therefore pins `umap-learn<0.6`. This only
    affects UMAP plots, not scpolivi itself.

## Install from source

```bash
git clone https://github.com/zhisonghe/scpoli_scvi-tools.git
cd scpoli_scvi-tools
pip install ".[leiden]"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install ".[leiden]"
```

For development (editable install with test dependencies):

```bash
pip install -e ".[test]"
pytest
```

## GPU and Apple silicon

scpolivi trains with Lightning through scvi-tools, so the PyTorch build decides which hardware it
can use. Install the PyTorch wheel that matches your CUDA version first (see
[pytorch.org](https://pytorch.org/get-started/locally/)), then scpolivi. Select the device with
`model.train(accelerator="gpu")`, `accelerator="mps"` (Apple silicon) or `accelerator="cpu"`.
The default, `"auto"`, picks CUDA if it is available. scvi-tools does not pick MPS automatically.

Training runs on one device. As in scvi-tools, multi-GPU (DDP) training is not a default
feature, and scpolivi does not support it: the prototype updates at the end of each epoch run
on a single device.

## Building this documentation

```bash
pip install ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site in ./site
```

## Example data

The walkthroughs use the pancreas data of the scArches scPoli tutorial (16,382 cells, 4,000
genes, 9 studies). It is available from figshare:
<https://figshare.com/ndownloader/files/41581626>. Save it as `data/pancreas_sparse.h5ad`, or
pass another path with `--data`.
