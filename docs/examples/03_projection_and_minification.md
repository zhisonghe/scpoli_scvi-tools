# 3. Projection and minification

This walkthrough starts from the reference model saved by walkthrough 1.

- **Projection.** Cells from a known study are embedded and classified without any training.
  They have no cell-type column, and their genes are shuffled. `prepare_query_anndata` restores
  the model's gene order (and would zero-pad missing genes).
- **Unknown conditions.** Embedding cells of a study the model has not seen raises an error.
  The message explains how to map them.
- **Shot surgery.** The two CEL-Seq studies are mapped onto their most similar reference study
  by reconstruction loss, with no training. The script prints the mapping and the resulting
  label-transfer accuracy.
- **Minification.** `minify_adata()` replaces the expression matrix with the latent posterior.
  The saved model plus AnnData is then much smaller, and latents, classification and
  normalized expression still work.

```bash
python examples/03_projection_and_minification.py --data data/pancreas_sparse.h5ad
```

```python
--8<-- "examples/03_projection_and_minification.py"
```
