# Concepts

## The model

scPoli is a conditional variational autoencoder (CVAE) with two additions: learned
**condition embeddings** and cell-type **prototypes**.

```
             condition embeddings (one table per condition key)
                     │                         │
counts x ──► encoder ┴──► latent z ──► decoder ┴──► NB / ZINB / Poisson / MSE / Bernoulli
                              │
                      prototypes (labeled: cell types; unlabeled: clusters)
```

**Conditions.** Each condition key (e.g. `study`, `sample`, `technology`) has an embedding
table. The embeddings of a cell's conditions are concatenated and fed into the first layer of the
encoder and/or decoder (`inject_condition`). Unlike one-hot batch covariates, embeddings put
samples in a continuous space where similar samples lie close together
(`get_conditional_embeddings`). The negative binomial dispersion is learned per combination of
all condition values (`dispersion="gene-batch"`).

**Labeled prototypes.** For every cell type, the prototype is the mean (and covariance) of the
latent codes of the labeled cells of that type. The prototype loss pulls each labeled cell
towards its prototype. Cells are classified by their nearest prototype, and the distance gives
an uncertainty score. A cell type that is missing from the reference therefore shows up as
high uncertainty.

**Hierarchies.** Several cell-type keys (e.g. fine and coarse annotation) share one prototype
namespace. Each cell contributes to the prototype of every level it is labeled at, and
`classify` predicts every level.

**Unlabeled prototypes.** At the end of pretraining, the latent space is clustered (Leiden or
k-means) and the cluster centres become unlabeled prototypes. Their loss
(`unlabeled_weight`, 0 by default) can pull unlabeled cells towards the nearest centre. They
also serve to find new cell types (`add_new_cell_type`).

## Training phases

| Epochs | Loss |
|---|---|
| `0 … pretraining_epochs-1` | ELBO (reconstruction + KL, KL weight warmed up over `alpha_epoch_anneal` epochs) |
| `pretraining_epochs …` | ELBO + `eta` × (labeled prototype loss + `unlabeled_weight` × unlabeled loss) [+ `graph_weight` × graph loss] |

At the first prototype epoch the prototypes are initialized. After every later epoch the labeled
prototypes are recomputed from the training cells, and the unlabeled prototypes take one
optimizer step. Early stopping only starts after pretraining. By default
`pretraining_epochs = 0.9 × n_epochs`.

## Query mapping

`load_query_data` extends a trained model with the conditions of new data:

- new condition values get new embedding rows, and the reference rows are kept;
- by default (`freeze=True`) only the embeddings and the dispersion are trained, so the
  reference latent space does not move;
- reference prototypes stay fixed. Cell types that only exist in labeled query cells get new
  prototypes, and unlabeled prototypes are re-initialized on the query.

This is scArches' architectural surgery. `shot_surgery` maps new conditions without training,
by copying the embedding of the reference condition that reconstructs them best.

## Labels: unknown, unlabeled, new

| Situation | How to express it |
|---|---|
| Some cells have no annotation | Value in `unknown_ct_names` (e.g. `"unknown"`), or NaN |
| Only some cells may train the labeled prototypes | `labeled_indices` in `setup_anndata` / `load_query_data` |
| The query has no labels | `load_query_data(..., labeled_indices=[])`, or no cell-type columns at all |
| A cluster of the query is a new type | `get_prototypes_info("unlabeled")`, then `add_new_cell_type` |

## Reconstruction losses

| `recon_loss` | Data | Notes |
|---|---|---|
| `"nb"` (default) | counts | negative binomial, softmax scale × library size |
| `"zinb"` | counts | adds a zero-inflation head |
| `"poisson"` | counts | rate = softmax scale; `poisson_library_scaling=True` multiplies by library size |
| `"mse"` | any real values (e.g. log-normalized) | squared error on the input, ReLU head, no `log1p` |
| `"bernoulli"` | binary / ATAC | binary cross-entropy on `x > 0` |
| `"nb_dist"` | counts | alias of `"nb"` (broken in scArches) |
