# Tutorials

Jupyter notebooks that use the scpolivi Python API on the pancreas data of the scArches scPoli
tutorial (16,382 cells, 9 studies). Each page shows a notebook executed with its outputs, and
can be downloaded and run locally.

| Notebook | Covers |
|---|---|
| [1. Reference and query](01_reference_and_query.ipynb) | setup, training phases, query mapping, label transfer, uncertainty, joint embedding, prototypes, condition embeddings |
| [2. Several keys, hierarchy, new types](02_multiple_keys_and_hierarchy.ipynb) | two condition keys, two label levels, unknown and partial labels, finding and adding a new cell type |
| [3. Projection and trained models](03_projection_and_minification.ipynb) | projecting cells, gene matching, shot surgery, normalized expression, differential expression, minification |
| [4. Graph regularization](04_graph_regularization.ipynb) | sample graphs, graph loss from scratch, fine-tuning, query with a graph |
| [5. Moving from scArches](05_from_scarches.ipynb) | translating code, importing a scArches checkpoint, continuing in scpolivi |

## Running the notebooks

```bash
pip install ".[tutorials]"     # scanpy, seaborn, jupyterlab, ...
jupyter lab docs/tutorials
```

- **Data.** The notebooks read `data/pancreas_sparse.h5ad`, a path relative to the repository
  root. Download the file from <https://figshare.com/ndownloader/files/41581626>, and either
  start Jupyter from the repository root or change the path in the notebook.
- **Order.** Notebook 3 loads the reference model saved by notebook 1.
- **Notebook 5** needs a model saved by scArches. The one used here comes from
  `scripts/make_golden_pancreas.py`.
- **Runtime.** On a laptop CPU each notebook runs in 2–6 minutes.

To re-execute all notebooks and store their outputs (as for this documentation):

```bash
python scripts/build_tutorials.py
```
