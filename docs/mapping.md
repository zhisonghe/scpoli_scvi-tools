# Query mapping and projection

There are four ways to bring new cells into a trained model. Which one fits depends on whether
the new cells come from conditions the model already knows.

| Situation | Method | Training |
|---|---|---|
| Cells from **known** conditions | `model.get_latent(adata)`, `model.classify(adata)` | none |
| **New** conditions, best quality | `SCPOLI.load_query_data(adata, model)` then `.train(...)` | embeddings + dispersion |
| New conditions, no training | `SCPOLI.shot_surgery(adata, model)` | none (optional fine-tuning) |
| Joint view of reference and query | `query_model.get_latent(concat([ref, query]))` | – |

## Projection

```python
SCPOLI.prepare_query_anndata(new_cells, model)     # gene order / missing genes
z = model.get_latent(new_cells, mean=True)
res = model.classify(new_cells, use_mean=True, scale_uncertainties=True)
res["cell_type"]["preds"], res["cell_type"]["uncert"]
```

Cell-type columns are optional. An unknown condition value raises an error that points to
`load_query_data`.

!!! tip "`use_mean=True`"
    scArches classifies one random sample of each cell's latent posterior. scpolivi keeps that
    default for comparability, but classifying the posterior mean is more accurate, especially
    for models that are not fully trained. Use `use_mean=True`.

## Surgery (`load_query_data`)

```python
query_model = SCPOLI.load_query_data(
    query,
    reference_model=model,          # model or path
    labeled_indices=[],             # [] = query labels unused; None = all labeled
    unknown_ct_names=None,
    freeze=True,                    # only embeddings + dispersion are trained
    remove_dropout=True,
)
query_model.train(n_epochs=50, pretraining_epochs=40, eta=10)
```

- Conditions shared with the reference (e.g. a known technology in a new study) keep their
  embedding. Shared embeddings are trainable, as in scArches. Reference cells of that condition
  can therefore move slightly.
- Labeled query cells of cell types absent from the reference create new prototypes. Reference
  prototypes stay fixed.
- `return_new_conditions=True` also returns the new values per condition key.

The query model contains every condition, so it can embed reference and query cells together.

## Shot surgery

```python
query_model, mapping = SCPOLI.shot_surgery(query, reference_model=model, labeled_indices=[])
```

For each new value of each condition key, scpolivi picks the reference value with the lowest
reconstruction loss and copies its embedding and dispersion. `train_epochs > 0` fine-tunes
afterwards. scArches restricts this to a single condition key; scpolivi handles several keys,
one at a time.

## Saving and sharing

```python
model.save("ref_model", save_anndata=True)
model = SCPOLI.load("ref_model")                     # or SCPOLI.load("ref_model", adata=adata)
model.minify_adata()                                 # keep only the latent posterior
```

The file format is scvi-tools'. Models saved by scArches are loaded with
[`SCPOLI.from_scarches`](scarches.md#loading-scarches-models).
