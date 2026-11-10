# scpolivi

scPoli reimplemented on scvi-tools, with no dependency on scArches. Design, parity results and
deviations from scArches are in PLAN.md.

## Reference, query mapping, projection

```python
from scpolivi import SCPOLI

SCPOLI.setup_anndata(ref, condition_keys=["study", "tech"], cell_type_keys=["ct_fine", "ct_coarse"])
model = SCPOLI(ref, embedding_dims=5)
model.train(n_epochs=50, pretraining_epochs=40, eta=5)

# query mapping (scArches surgery): new conditions get new embeddings, the rest is frozen
query = SCPOLI.load_query_data(qry, reference_model=model, labeled_indices=[])
query.train(n_epochs=50, pretraining_epochs=40, eta=10)

# projection: any cells whose conditions the model knows; cell-type columns are optional
z = query.get_latent(adata_all, mean=True)
pred = query.classify(qry, scale_uncertainties=True, use_mean=True)["ct_fine"]
```

Genes must match the model's genes in name and order; use `SCPOLI.prepare_query_anndata(adata, model)`
to pad and reorder. Projecting cells of an unknown condition raises an error (map them with
`load_query_data` or `shot_surgery` first). Models trained with scArches can be loaded with
`SCPOLI.from_scarches(dir_path, adata)`.

## Graph Laplacian regularization (from scANCVI)

An optional loss term keeps cells that are neighbours within a sample close in the latent space:
`0.5 * sum w_ij ||z_i - z_j||^2 / sum w_ij`, averaged over samples.

```python
from scpolivi.graph import construct_sample_graphs

graphs = construct_sample_graphs(ref, "sample", recompute_hvg_pca=True)  # {sample: kNN adjacency}
model.set_graph(graphs, graph_key="sample")   # or an (n_obs, n_obs) matrix / adata.obsp key

# a) train from scratch: pretraining as usual, then ELBO + prototype + graph loss
model.train(n_epochs=50, pretraining_epochs=40, eta=5, graph_weight=10)

# b) fine-tune an already trained model (no pretraining, no KL warm-up, prototypes kept)
model.fine_tune(n_epochs=10, graph_weight=10, eta=5)
```

- `graph_sampling="neighbors"` (default): each minibatch cell brings one sampled graph neighbour, so
  every cell gets a graph term at every step. `"batch"` uses only edges between cells that share a
  random minibatch, as scANCVI does; with scPoli's default batch size of 128 that is only a few edges per step.
- The graph is not saved with the model. Attach it again after loading, and attach a graph of the query
  cells to a query model.
