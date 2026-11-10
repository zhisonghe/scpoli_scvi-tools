# Coming from scArches

## Code changes

=== "scArches"

    ```python
    from scarches.models.scpoli import scPoli

    model = scPoli(adata=ref, condition_keys="study", cell_type_keys="cell_type",
                   embedding_dims=5, recon_loss="nb")
    model.train(n_epochs=50, pretraining_epochs=40, eta=5)
    q = scPoli.load_query_data(adata=query, reference_model=model, labeled_indices=[])
    q.train(n_epochs=50, pretraining_epochs=40, eta=10)
    q.classify(query, scale_uncertainties=True)
    ```

=== "scpolivi"

    ```python
    from scpolivi import SCPOLI

    SCPOLI.setup_anndata(ref, condition_keys="study", cell_type_keys="cell_type")
    model = SCPOLI(ref, embedding_dims=5, recon_loss="nb")
    model.train(n_epochs=50, pretraining_epochs=40, eta=5)
    q = SCPOLI.load_query_data(adata=query, reference_model=model, labeled_indices=[])
    q.train(n_epochs=50, pretraining_epochs=40, eta=10)
    q.classify(query, scale_uncertainties=True, use_mean=True)
    ```

The data arguments (`condition_keys`, `cell_type_keys`, `unknown_ct_names`, `labeled_indices`)
move from the constructor to `setup_anndata`. That is how every scvi-tools model works.
Everything else keeps its scArches name.

| scArches | scpolivi |
|---|---|
| `scPoli(adata, condition_keys, cell_type_keys, unknown_ct_names, labeled_indices, ...)` | `SCPOLI.setup_anndata(adata, ...)` + `SCPOLI(adata, ...)` |
| `hidden_layer_sizes`, `latent_dim`, `embedding_dims`, `embedding_max_norm`, `dr_rate`, `use_bn`, `use_ln`, `recon_loss`, `inject_condition` | same |
| `train(...)` arguments | same, plus `seed`, `accelerator`, graph options |
| `get_latent`, `classify`, `get_prototypes_info`, `add_new_cell_type`, `get_conditional_embeddings`, `shot_surgery`, `load_query_data` | same |
| `model.trainer.logs["val_loss"]` | `model.history["validation_loss"]` |
| `model.save(dir)` / `scPoli.load(dir, adata)` | `model.save(dir)` / `SCPOLI.load(dir, adata)` (scvi-tools format) |
| `model.model` (torch module) | `model.module` |

## Loading scArches models

```python
model = SCPOLI.from_scarches("path/to/scarches_model", adata)
```

This reads the three files scArches writes (`attr.pkl`, `model_params.pt`, `var_names.csv`)
without importing scArches. `adata` needs the model's genes in the same order and its condition
columns. The imported model gives the same latents as in scArches (difference below 10⁻⁴). It
can be trained further, mapped to queries and saved in scvi-tools format. Reference and query
checkpoints both work.

## Behaviour differences

scpolivi reproduces scArches' behaviour, with these exceptions:

1. `recon_loss="bernoulli"` crashes in scArches; it works here (BCE on `x > 0`).
2. `recon_loss="nb_dist"` crashes in scArches; it is treated as `"nb"` with a warning.
3. `use_mmd` / MMD options are dead code in scArches. They are not accepted; checkpoints with
   `use_mmd=True` are rejected.
4. `freeze_expression` and `classify(log_distance=...)` have no effect in scArches, and none here.
5. `get_prototypes_info("labeled")` fills in the prototype names. scArches has a typo that
   leaves them empty.
6. The stratified train/validation split shuffles unlabeled cells (with the seed). scArches
   takes them in order.
7. `reload_best=True` restores weights *and* prototypes. scArches keeps references to the live
   weights, so only the prototypes revert.
8. `use_stratified_sampling` works with several condition keys (weights per combined
   condition). It crashes in scArches.
9. The Poisson rate is not scaled by library size, as in scArches;
   `poisson_library_scaling=True` scales it.
10. `seed=None` by default: scpolivi uses `scvi.settings.seed` instead of reseeding with 2020.
11. Prototypes of classes without labeled cells are left uninitialized (NaN) and skipped,
    instead of producing NaN losses.
12. `classify(use_mean=False)` samples z, as scArches does; `use_mean=True` is recommended.
13. Projection checks that genes are in the model's order, and that cell-type columns are
    optional.

## Things scArches cannot do

- `shot_surgery` with several condition keys
- `get_normalized_expression(transform_batch=...)`, differential expression and the other
  scvi-tools tools that build on the decoder
- minified AnnData
- [graph regularization](graph.md) and `fine_tune`
- pandas 3: scArches' scPoli fails during model construction
