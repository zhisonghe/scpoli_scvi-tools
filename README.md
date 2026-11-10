# scpolivi

**scPoli on scvi-tools.** scpolivi reimplements [scPoli](https://doi.org/10.1038/s41592-023-02035-2)
(population-level integration of single-cell data with condition embeddings and cell-type
prototypes) on [scvi-tools](https://scvi-tools.org). It does not depend on
[scArches](https://github.com/theislab/scarches), which hosts the original implementation and is
no longer actively maintained. scpolivi reproduces scArches' scPoli numerically and adds features
scArches lacks.

The same information, organized for the web, is in the documentation site (`mkdocs serve`; see
[Documentation](#documentation)).

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Query mapping and projection](#query-mapping-and-projection)
- [Graph regularization](#graph-regularization)
- [Coming from scArches](#coming-from-scarches)
- [Tutorials](#tutorials)
- [API overview](#api-overview)
- [Benchmarks](#benchmarks)
- [Documentation](#documentation)
- [Citation](#citation)

## Features

- **Integrate many samples.** Each condition key (study, sample, technology, …) has a learned
  embedding, so similar samples become nearby vectors.
- **Transfer labels with uncertainty.** Prototypes in the latent space, on one or several
  hierarchy levels. Cell types missing from the reference show up as high uncertainty.
- **Map new data.** Extend a trained reference with query data (scArches surgery),
  map conditions without training (`shot_surgery`), or project cells of known conditions.
- **Keep local structure.** An optional graph Laplacian loss (from scANCVI) preserves
  within-sample neighbourhoods, either while training from scratch or by fine-tuning a trained
  model.
- **The scvi-tools ecosystem.** Lightning training on CPU/GPU/Apple silicon,
  `get_normalized_expression`, differential expression, minified AnnData, scvi-tools model files.
- **Compatible with scArches.** Same arguments, same model (forward pass within 10⁻⁵).
  `SCPOLI.from_scarches` loads scArches checkpoints.

## Installation

Requires Python 3.10–3.13. scvi-tools (≥ 1.5.2, < 1.6), PyTorch, AnnData and scikit-learn are
installed automatically.

```bash
git clone https://github.com/zhisonghe/scpoli_scvi-tools.git && cd scpoli_scvi-tools
pip install ".[leiden]"          # extra: scanpy + Leiden, for the default prototype init and graph building
pip install -e ".[test]" && pytest   # development
```

Without the `leiden` extra, train with `clustering="kmeans", n_clusters=...`. For GPUs,
install the PyTorch build for your CUDA version first. Choose the device with
`train(accelerator="gpu" | "mps" | "cpu")`. Training runs on one device; multi-GPU (DDP) is
opt-in in scvi-tools and not supported here.

## Quick start

```python
import scanpy as sc
from scpolivi import SCPOLI

adata = sc.read("data/pancreas_sparse.h5ad")         # https://figshare.com/ndownloader/files/41581626
reference = adata[~adata.obs.study.isin(["celseq", "celseq2"])].copy()
query = adata[adata.obs.study.isin(["celseq", "celseq2"])].copy()

# reference
SCPOLI.setup_anndata(reference, condition_keys="study", cell_type_keys="cell_type")
model = SCPOLI(reference, embedding_dims=5)
model.train(n_epochs=50, pretraining_epochs=40, eta=5)

# query: new studies get new embeddings, everything else stays frozen
query_model = SCPOLI.load_query_data(query, reference_model=model, labeled_indices=[])
query_model.train(n_epochs=50, pretraining_epochs=40, eta=10)

res = query_model.classify(query, use_mean=True, scale_uncertainties=True)["cell_type"]
query.obs["pred"], query.obs["uncert"] = res["preds"], res["uncert"]
latent = query_model.get_latent(sc.concat([reference, query]), mean=True)
```

## Concepts

scPoli is a conditional VAE.

- **Condition embeddings.** The embeddings of a cell's conditions (one table per condition key)
  are concatenated and fed into the encoder and/or decoder. The negative-binomial dispersion is
  learned per combination of condition values.
- **Labeled prototypes.** Each cell type's centroid in the latent space. The prototype loss
  pulls labeled cells towards their centroid, and `classify` assigns the nearest prototype with
  a distance-based uncertainty. Several cell-type keys (a hierarchy) share one prototype
  namespace, and every level is predicted.
- **Unlabeled prototypes.** Cluster centroids (Leiden/k-means) of the latent space. They are
  used to discover new cell types (`get_prototypes_info("unlabeled")` → `add_new_cell_type`).
- **Training phases.** The first `pretraining_epochs` (default 90%) train the plain CVAE (KL
  warm-up over `alpha_epoch_anneal` epochs). Then the loss is ELBO + `eta` × prototype loss
  (+ `graph_weight` × graph loss). Prototypes are recomputed after every epoch.

Labels: put values such as `"unknown"` in `unknown_ct_names`; restrict which cells train the
prototypes with `labeled_indices`; queries without any labels use `labeled_indices=[]` (or have
no cell-type columns at all).

| `recon_loss` | Data |
|---|---|
| `nb` (default), `zinb`, `poisson` | raw counts |
| `mse` | real values (e.g. log-normalized) |
| `bernoulli` | binary / ATAC |
| `nb_dist` | alias of `nb` |

## Query mapping and projection

| Situation | Method |
|---|---|
| Cells from known conditions | `model.get_latent(adata, mean=True)`, `model.classify(adata, use_mean=True)`; no training |
| New conditions | `SCPOLI.load_query_data(adata, model)` + `.train(...)`; only embeddings and dispersion are trained |
| New conditions, no training | `SCPOLI.shot_surgery(adata, model)`: copy the embedding of the best-reconstructing reference condition |

- **Genes.** They must match the model's genes and order; `SCPOLI.prepare_query_anndata(adata,
  model)` reorders them and zero-pads missing ones. A different gene order raises an error
  (scArches silently returns wrong latents).
- **Cell-type columns** are optional for projection and queries.
- **Unknown conditions.** Projecting cells of an unknown condition raises an error that points
  to `load_query_data`.
- **`use_mean=True`.** scArches classifies a random latent sample by default. Classifying the
  mean is more accurate, so `use_mean=True` is recommended.
- **Saving.** `model.save(dir, save_anndata=True)` and `SCPOLI.load(dir)` use the scvi-tools
  format. `model.minify_adata()` keeps only the latent posterior instead of the expression
  data; latents, classification and normalized expression still work, training does not.

## Graph regularization

The loss is taken from scANCVI:

`L = mean over samples of 0.5 · Σ w_ij ‖z_i − z_j‖² / Σ w_ij`

over the edges of each sample's own kNN graph. It complements the prototypes: prototypes align
cell types across samples, while the graph keeps each sample's local neighbourhoods. Edges never
cross samples, so the graph does not directly oppose batch mixing.

```python
from scpolivi.graph import construct_sample_graphs

graphs = construct_sample_graphs(reference, "study", recompute_hvg_pca=True)  # {sample: adjacency}
model.set_graph(graphs, graph_key="study")      # or an (n_obs, n_obs) matrix / adata.obsp key

# a) from scratch: pretraining unchanged, then ELBO + prototype + graph loss
model.train(n_epochs=50, pretraining_epochs=40, eta=5, graph_weight=10)

# b) fine-tune a trained model: no pretraining, no KL warm-up, prototypes kept, lr 1e-4
model.fine_tune(n_epochs=10, graph_weight=10, eta=5)
```

| Option | Default | Meaning |
|---|---|---|
| `graph_weight` | 0 | loss weight; 10 worked well on pancreas, 100 started to cost batch mixing and label transfer |
| `graph_sampling` | `"neighbors"` | each minibatch cell brings one sampled neighbour, which doubles the cost of epochs with the graph loss. `"batch"` (scANCVI) is almost free but only sees edges inside random minibatches, about 10–25 per step at batch size 128 |
| `graph_use_mean` | False | loss on latent means instead of samples (scANCVI uses samples; the pull on the means is the same, and samples also shrink posterior variances) |
| `graph_start_epoch` | `pretraining_epochs` | first epoch with the loss |

The graph is not saved with the model; attach it again after loading. Query models accept a
graph of the query cells. With frozen weights, only the new embeddings respond to it.

## Coming from scArches

The data arguments move from the constructor to `setup_anndata`; everything else keeps its name:

```python
# scArches: scPoli(adata, condition_keys="study", cell_type_keys="cell_type", embedding_dims=5)
SCPOLI.setup_anndata(adata, condition_keys="study", cell_type_keys="cell_type")
model = SCPOLI(adata, embedding_dims=5)
```

| scArches | scpolivi |
|---|---|
| `model.trainer.logs["val_loss"]` | `model.history["validation_loss"]` |
| `model.model` | `model.module` |
| `scPoli.load(dir, adata)` (scArches files) | `SCPOLI.from_scarches(dir, adata)` |

Behaviour differences, all deliberate:

- **Crashes fixed.** `bernoulli` and `nb_dist` work; stratified sampling works with several
  condition keys.
- **`reload_best`** restores weights *and* prototypes (scArches only reverts the prototypes).
- **Prototype names.** `get_prototypes_info("labeled")` fills them in (a scArches typo leaves
  them empty).
- **Train/validation split.** Unlabeled cells are shuffled (with the seed).
- **Seeding.** `seed=None` uses `scvi.settings.seed` instead of reseeding with 2020.
- **Uninitialized prototypes.** Prototypes without labeled cells are skipped instead of
  producing NaN losses.
- **Projection.** Genes are checked, and cell-type columns are optional.
- **Unchanged dead options.** MMD options are rejected; `freeze_expression` and `log_distance`
  have no effect, as in scArches.

The full list is in the documentation ("Coming from scArches").

## Tutorials

Jupyter notebooks in [`docs/tutorials/`](docs/tutorials), executed with their outputs, that use
the Python API on the pancreas data (GitHub renders them directly):

| Notebook | Covers |
|---|---|
| [1. Reference and query](docs/tutorials/01_reference_and_query.ipynb) | setup, training phases, query mapping, label transfer, uncertainty, joint embedding, prototypes, condition embeddings |
| [2. Several keys, hierarchy, new types](docs/tutorials/02_multiple_keys_and_hierarchy.ipynb) | two condition keys, two label levels, unknown and partial labels, finding and adding a new cell type |
| [3. Projection and trained models](docs/tutorials/03_projection_and_minification.ipynb) | projection, gene matching, shot surgery, normalized expression, differential expression, minification |
| [4. Graph regularization](docs/tutorials/04_graph_regularization.ipynb) | sample graphs, graph loss from scratch, fine-tuning, query with a graph |
| [5. Moving from scArches](docs/tutorials/05_from_scarches.ipynb) | translating code, importing a scArches checkpoint, continuing in scpolivi |

To run them locally:

```bash
pip install ".[tutorials]"
jupyter lab docs/tutorials
```

The notebooks read `data/pancreas_sparse.h5ad` relative to the repository root, and notebook 3
uses the model saved by notebook 1. `python scripts/build_tutorials.py` re-executes all of them.
The same workflows exist as command-line scripts in `examples/` for batch jobs.

## API overview

| Call | Purpose |
|---|---|
| `SCPOLI.setup_anndata(adata, condition_keys, cell_type_keys=None, unknown_ct_names=None, labeled_indices=None, layer=None)` | register data |
| `SCPOLI(adata, hidden_layer_sizes, latent_dim=10, embedding_dims=10, recon_loss="nb", inject_condition=("encoder","decoder"), dr_rate=0.05, use_bn=False, use_ln=True, ...)` | build the model |
| `model.train(n_epochs=100, pretraining_epochs=None, eta=1, lr=1e-3, early_stopping_kwargs=None, batch_size=128, graph_weight=0, seed=None, ...)` | train |
| `model.fine_tune(n_epochs=20, graph_weight=1, eta=1, lr=1e-4)` | continue training with the graph loss |
| `model.get_latent(adata=None, mean=False)` | latent codes |
| `model.classify(adata=None, use_mean=False, scale_uncertainties=False, get_prob=False)` | `{level: {"preds", "uncert", "weighted_distances"}}` |
| `SCPOLI.load_query_data(adata, reference_model, labeled_indices=None, unknown_ct_names=None, freeze=True, return_new_conditions=False)` | surgery |
| `SCPOLI.shot_surgery(adata, reference_model, train_epochs=0)` | `(model, mapping)` |
| `model.get_conditional_embeddings()` | condition embeddings as AnnData (dict for several keys) |
| `model.get_prototypes_info("labeled" / "unlabeled")` | prototypes as AnnData |
| `model.add_new_cell_type(name, obs_key, prototypes)` | promote unlabeled prototypes |
| `model.update_labeled_indices(indices)` | change labeled cells |
| `model.set_graph(graph, graph_key)` | attach a sample graph |
| `scpolivi.graph.construct_sample_graphs(adata, sample_key, ...)` | per-sample kNN graphs |
| `model.minify_adata()` | keep only the latent posterior |
| `model.save(dir)`, `SCPOLI.load(dir, adata)`, `SCPOLI.from_scarches(dir, adata)` | I/O |
| `model.get_normalized_expression(...)`, `model.differential_expression(...)` | inherited from scvi-tools |

Registry views: `condition_keys_`, `cell_type_keys_`, `conditions_`, `conditions_combined_`,
`cell_types_`, `prototypes_labeled_`, `prototypes_unlabeled_`, `labeled_indices_`, `history`.

## Benchmarks

Pancreas tutorial protocol (reference without alpha cells, CEL-Seq query), three seeds each;
the evaluation is `scripts/eval_pancreas.py`.

| | scArches | scpolivi |
|---|---|---|
| query accuracy, known types | 0.974–0.978 | 0.972–0.977 |
| kNN cell-type accuracy | 0.979–0.982 | 0.979–0.982 |
| study mixing (kNN entropy) | 0.45–0.51 | 0.46–0.48 |
| mean uncertainty, alpha / known | ≈0.42 / 0.17 | ≈0.43 / 0.19 |

The model also matches scArches numerically. Forward passes from the same weights agree within
10⁻⁵ for every loss scArches can run, and imported scArches checkpoints reproduce scArches'
latents within 10⁻⁴.

Graph regularization on pancreas (3 seeds, weight 10, defaults): within-study neighbourhood
overlap 0.238–0.240 → 0.242–0.249 and cell-type silhouette 0.803–0.809 → 0.812–0.824, with
label transfer (0.972–0.977) and study mixing unchanged. Details are in the documentation
("Benchmarks").

## Documentation

```bash
pip install ".[docs]"
mkdocs serve        # http://127.0.0.1:8000
mkdocs build        # static site in ./site
```

## Citation

- De Donno, C. et al. Population-level integration of single-cell datasets enables multi-scale
  analysis across samples. *Nature Methods* 20, 1683–1692 (2023).
- Gayoso, A. et al. A Python library for probabilistic analysis of single-cell omics data.
  *Nature Biotechnology* 40, 163–166 (2022).
- Lotfollahi, M. et al. Mapping single-cell data to reference atlases by transfer learning.
  *Nature Biotechnology* 40, 121–130 (2022).
