# Example scripts

The `examples/` folder contains the tutorial workflows as plain Python scripts, for batch jobs
and clusters. They take `--data` (path to the pancreas file); most also take `--epochs`.

| Script | Corresponds to |
|---|---|
| `examples/01_reference_and_query.py` | tutorial 1 |
| `examples/02_multiple_keys_and_hierarchy.py` | tutorial 2 |
| `examples/03_projection_and_minification.py` | tutorial 3 (needs the model saved by script 1) |
| `examples/04_graph_regularization.py` | tutorial 4 |
| `examples/05_from_scarches.py` | tutorial 5 (`--scarches-model path/to/scarches_model`) |

```bash
python examples/01_reference_and_query.py --data data/pancreas_sparse.h5ad --out results/01_reference_and_query
python examples/03_projection_and_minification.py --data data/pancreas_sparse.h5ad
```
