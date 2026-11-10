# API overview

| Object | Purpose |
|---|---|
| [`scpolivi.SCPOLI`](model.md) | The model: setup, training, query mapping, latents, classification, prototypes |
| [`scpolivi.graph`](graph.md) | Sample graphs and the graph Laplacian loss |
| [`scpolivi.SCPOLIVAE`](module.md) | The PyTorch module (encoder, decoder, prototypes, loss) |
| [`scpolivi.train`](training.md) | Training plan, data splitter, early stopping |
| [`scpolivi.data`](data.md) | AnnData field for hierarchical labels, setup helpers |

## Most used

| Call | Returns |
|---|---|
| `SCPOLI.setup_anndata(adata, condition_keys, cell_type_keys, ...)` | – (registers `adata`) |
| `SCPOLI(adata, ...)` | model |
| `model.train(...)` | – |
| `model.get_latent(adata, mean=True)` | `(n_cells, latent_dim)` array |
| `model.classify(adata, use_mean=True)` | `{level: {"preds", "uncert", "weighted_distances"}}` |
| `SCPOLI.load_query_data(query, model, labeled_indices=[])` | query model |
| `SCPOLI.shot_surgery(query, model)` | `(model, mapping)` |
| `model.get_conditional_embeddings()` | AnnData (or dict of AnnData per key) |
| `model.get_prototypes_info("labeled" / "unlabeled")` | AnnData of prototypes |
| `model.add_new_cell_type(name, obs_key, prototypes)` | – |
| `model.set_graph(graph, graph_key)`, `model.fine_tune(...)` | – |
| `model.save(dir)`, `SCPOLI.load(dir, adata)`, `SCPOLI.from_scarches(dir, adata)` | – / model |
| `model.get_normalized_expression(...)`, `model.differential_expression(...)` | inherited from scvi-tools |

Inherited scvi-tools methods (`get_normalized_expression`, `differential_expression`,
`get_latent_representation`, `posterior_predictive_sample`, `save`, `load`,
`prepare_query_anndata`, …) are documented in the
[scvi-tools API](https://docs.scvi-tools.org/en/stable/api/index.html).
