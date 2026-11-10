# 2. Several condition keys, a hierarchy, new cell types

This walkthrough uses the same data with a richer setup:

- **Two condition keys.** `study` and `technology` (inDrop, Smart-seq2, CEL-Seq, …), each with
  its own embedding size (`embedding_dims=[5, 3]`). The query adds a new study *and* a new
  technology, so both keys gain embeddings.
- **Two label levels.** `cell_type` and `cell_type_coarse` (endocrine, exocrine, stromal,
  immune). Both share one prototype namespace, and `classify` returns both levels.
- **Imperfect labels.** 10% of reference cells are annotated `"unknown"` (`unknown_ct_names`),
  and only 70% count as labeled (`labeled_indices`).
- **A new cell type.** Alpha cells only exist in the query. The script finds the unlabeled
  prototypes that collect them (`get_prototypes_info("unlabeled")`), turns those into a new
  labeled prototype with `add_new_cell_type`, and checks how many alpha cells are now
  predicted as alpha.

`get_conditional_embeddings()` returns one AnnData per condition key. The technology
embeddings can be compared directly, e.g. by their distances.

```bash
python examples/02_multiple_keys_and_hierarchy.py --data data/pancreas_sparse.h5ad
```

```python
--8<-- "examples/02_multiple_keys_and_hierarchy.py"
```
