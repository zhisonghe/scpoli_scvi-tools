# 1. Reference and query

This is the scArches scPoli surgery tutorial, done with scpolivi. The pancreas data covers nine
studies. Seven form the reference, from which all **alpha cells are removed**. The two CEL-Seq
studies are the query. Alpha cells are then a cell type the reference has never seen, so they
should receive high uncertainty.

**Steps**

1. `setup_anndata` registers `study` as the condition key and `cell_type` as the label.
2. The reference trains for 80% of the epochs as a plain CVAE, then with the prototype loss
   (`eta=5`). Early stopping watches the validation prototype loss, as in the scArches tutorial.
3. `load_query_data` adds embeddings for the two query studies. Only these embeddings (and the
   dispersion) are trained; `labeled_indices=[]` keeps the query labels out of training.
4. `classify` assigns each query cell to its nearest prototype and returns an uncertainty
   score. The script prints the accuracy on known types, a per-class report, and the mean
   uncertainty of alpha cells versus known types.
5. Reference and query are embedded together and shown as a UMAP coloured by study, cell type,
   dataset and uncertainty.

**Expected results** (50 epochs; see [benchmarks](../benchmarks.md)): about 0.97 accuracy on
known types, and mean scaled uncertainty about 0.4 for alpha cells versus about 0.18 for known
types.

```bash
python examples/01_reference_and_query.py --data data/pancreas_sparse.h5ad --out results/01_reference_and_query
```

```python
--8<-- "examples/01_reference_and_query.py"
```
