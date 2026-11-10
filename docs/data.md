# Data requirements

## What `setup_anndata` expects

```python
SCPOLI.setup_anndata(
    adata,
    condition_keys=["study", "technology"],     # required, ≥ 1 obs column
    cell_type_keys=["cell_type", "coarse"],     # optional, ≥ 0 obs columns
    unknown_ct_names=["unknown"],                # optional
    labeled_indices=None,                        # optional, default: all cells labeled
    layer=None,                                  # use adata.layers[layer] instead of X
)
```

- **Expression.** Raw counts in `X` or `layer` for `nb`, `zinb` and `poisson`. Any real values
  for `mse`. Values are binarized for `bernoulli`. Dense or sparse matrices both work.
- **Condition columns.** Any dtype; values are converted to strings, and the category order is
  the order of first appearance, as in scArches. Every cell needs a value.
- **Cell-type columns.** Strings. NaN and values in `unknown_ct_names` count as unknown.
  Only cell types of labeled cells become prototypes.

## Helper columns

`setup_anndata` writes a few `obs` columns. They start with `_scpoli_`, and you can ignore
them:

| Column | Meaning |
|---|---|
| `_scpoli_conditions_combined` | all condition values joined by `_`; selects the dispersion |
| `_scpoli_labeled` | 1 = labeled cell |
| `_scpoli_indices` | row position, used to look up graph edges |

Condition columns are converted to categoricals.

## New data for a trained model

| Use | Requirements |
|---|---|
| `get_latent`, `classify`, `get_normalized_expression` on new cells | the model's genes in the same order (`SCPOLI.prepare_query_anndata(adata, model)` reorders and zero-pads); all condition columns with **known** values; cell-type columns optional |
| `load_query_data` | the model's genes; all condition columns (new values allowed); cell-type columns optional |
| `SCPOLI.load(dir, adata)` | as for new cells |

If the genes match but are in a different order, scpolivi raises an error. scArches silently
returns wrong latents in that case.

## Size

All data is kept in memory. Training time grows linearly with the number of cells. The
end-of-epoch prototype update passes over all training cells once more. Leiden initialization
of unlabeled prototypes is slow above about 10⁶ cells; use `clustering="kmeans"` there.
After training, `minify_adata()` replaces the expression matrix with the latent posterior,
which is useful for sharing a model together with its data.
