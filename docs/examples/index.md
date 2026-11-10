# Walkthroughs

Five runnable scripts in the `examples/` folder of the repository. They all use the pancreas data
of the scArches scPoli tutorial (see [Installation](../installation.md#example-data)). Each
accepts `--data` (path to the h5ad file); most also accept `--epochs`, and a small value makes a
quick test run.

| Script | Covers |
|---|---|
| [1. Reference and query](01_reference_and_query.md) | setup, training, query mapping, label transfer, uncertainty, UMAP |
| [2. Several keys, hierarchy, new types](02_multiple_keys_and_hierarchy.md) | two condition keys, two label levels, unknown labels, partial labels, condition embeddings, adding a new cell type |
| [3. Projection and minification](03_projection_and_minification.md) | projecting cells, gene matching, shot surgery, minified AnnData |
| [4. Graph regularization](04_graph_regularization.md) | sample graphs, graph loss from scratch, fine-tuning, query |
| [5. Models from scArches](05_from_scarches.md) | importing a scArches checkpoint and continuing with it |

Walkthrough 3 uses the reference model saved by walkthrough 1.

```bash
python examples/01_reference_and_query.py --data data/pancreas_sparse.h5ad
python examples/03_projection_and_minification.py --data data/pancreas_sparse.h5ad
```
