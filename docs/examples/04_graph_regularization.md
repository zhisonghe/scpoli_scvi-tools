# 4. Graph regularization

This walkthrough compares three reference models on how well they preserve within-study
neighbourhoods. Preservation is measured as the overlap between each cell's neighbours in the
study's own kNN graph and its nearest neighbours in the latent space.

1. **Baseline**: plain scPoli.
2. **From scratch**: `set_graph` + `train(graph_weight=...)`. The graph loss starts with the
   prototype loss.
3. **Fine-tuned**: the trained baseline continues for 10 epochs with `fine_tune`.

Finally, a query model is trained with a graph of the query cells. See
[Graph regularization](../graph.md) for the options and how to choose the weight.

```bash
python examples/04_graph_regularization.py --data data/pancreas_sparse.h5ad --graph-weight 10
```

```python
--8<-- "examples/04_graph_regularization.py"
```
