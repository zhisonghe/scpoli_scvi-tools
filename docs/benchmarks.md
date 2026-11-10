# Benchmarks

## Numerical equivalence with scArches

The test suite checks these against outputs written by scArches (scArches `d35cde6`):

| Check | Result |
|---|---|
| Forward pass from the same weights: latent, reconstruction loss, KL, for nb, zinb, poisson, mse; encoder/decoder injection; batch/layer norm; 1–3 hidden layers; 1–2 condition keys | difference ≤ 10⁻⁵ |
| Prototype losses and prototype updates | identical to scArches' trainer functions |
| scArches checkpoints imported with `SCPOLI.from_scarches` (reference and query) | latents within 10⁻⁴; nearest-prototype labels agree for > 99.9% of cells |
| Graph Laplacian loss (`graph_sampling="batch"`) vs. scANCVI's own function | identical |

Training trajectories are not bit-identical, because random number streams and data order
differ. Equivalence of trained models is therefore assessed statistically, below.

## Pancreas tutorial

Protocol of the scArches scPoli surgery tutorial: 7 reference studies without alpha cells
(8,634 cells), 2 CEL-Seq query studies (3,289 cells), 4,000 genes, `embedding_dims=5`, 50 epochs
(40 pretraining), `eta=5` for the reference, `eta=10` for the query, unlabeled query. Each run was
scored on the joint reference + query latent by `scripts/eval_pancreas.py`:

- **query acc / F1**: label transfer on query cells of types known to the reference
  (scArches default: classification of a sampled latent)
- **uncert α / known**: mean scaled uncertainty of the unseen alpha cells vs. known types
- **kNN acc**: 15-NN cell-type accuracy; **ASW**: cell-type silhouette rescaled to [0, 1]
- **mixing**: mean normalized entropy of studies among each cell's 30 nearest neighbours
- **NN overlap**: share of each cell's 15 nearest neighbours in a per-study expression PCA that
  are also among its 15 nearest latent neighbours within the study

### scArches vs. scpolivi (3 seeds each)

| | query acc | query F1 | uncert α / known | kNN acc | ASW | mixing | NN overlap |
|---|---|---|---|---|---|---|---|
| scArches | 0.974–0.978 | 0.871–0.941 | 0.37–0.48 / 0.17 | 0.979–0.982 | 0.810–0.815 | 0.451–0.510 | 0.234–0.238 |
| scpolivi | 0.972–0.977 | 0.869–0.941 | 0.39–0.49 / 0.18–0.23 | 0.979–0.982 | 0.803–0.809 | 0.456–0.479 | 0.238–0.240 |

The two implementations overlap on every metric, apart from a cell-type silhouette about 0.005
lower. Runtime on a laptop CPU is about 65 s (scArches) vs. 70 s (scpolivi) for the reference.

### Graph regularization (scpolivi)

Graph: per-study 15-NN connectivities on a 2,000-HVG PCA (`construct_sample_graphs`). The graph
loss is active in the 10 prototype epochs, unless noted otherwise.

| Variant | seeds | query acc | kNN acc | ASW | mixing | NN overlap |
|---|---|---|---|---|---|---|
| no graph | 3 | 0.972–0.977 | 0.979–0.982 | 0.803–0.809 | 0.456–0.479 | 0.238–0.240 |
| **weight 10, sampled z (defaults)** | 3 | 0.972–0.977 | 0.980–0.983 | 0.812–0.824 | 0.445–0.484 | 0.242–0.249 |
| weight 100, latent means | 3 | 0.961–0.969 | 0.979–0.982 | 0.827–0.831 | 0.386–0.416 | 0.244–0.250 |
| weight 100, sampled z | 1 | 0.958 | 0.979 | 0.820 | 0.424 | 0.249 |
| weight 10, latent means | 1 | 0.976 | 0.982 | 0.822 | 0.479 | 0.240 |
| weight 100, sampled z, `graph_sampling="batch"` | 1 | 0.944 | 0.971 | 0.804 | 0.474 | 0.228 |
| fine-tune 10 epochs, weight 10, sampled z | 1 | 0.974 | 0.981 | 0.798 | 0.443 | 0.246 |
| fine-tune 10 epochs, weight 100, sampled z | 1 | 0.970 | 0.979 | 0.801 | 0.453 | 0.247 |

Findings:

- **Defaults** (weight 10, sampled z, neighbour sampling): neighbourhood preservation and
  cell-type separation improve slightly but consistently. Label transfer and study mixing stay
  unchanged.
- **Stronger weights** improve separation further, at a cost in study mixing and label
  transfer.
- **`graph_sampling="batch"`** (scANCVI's estimator) did not help here. At batch size 128, too
  few edges fall into a minibatch.
- **Fine-tuning** reaches about the same neighbourhood preservation as training from scratch,
  but cell-type separation does not improve.
- The gains are small on this dataset, which has little within-study substructure.
  Datasets with more local structure (e.g. continuous trajectories, cell states) are where the
  graph term is meant to matter.

Reproduce with:

```bash
python scripts/run_pancreas_scpolivi.py --seed 0                                   # no graph
python scripts/run_pancreas_scpolivi.py --seed 0 --graph-weight 10 --graph-sampled-z
python scripts/run_pancreas_scpolivi.py --seed 0 --graph-weight 10 --graph-mode finetune --graph-sampled-z
python scripts/eval_pancreas.py data/scpolivi/pancreas_*_seed0
```

The scArches reference runs come from `scripts/make_golden_pancreas.py`, run in an environment
with scArches and `pandas<3`.

## Synthetic data

On simulated counts (2,400 cells, two condition keys, two label levels, one cell type held out of
the reference), seed 0 gave reference kNN accuracy 0.962 (scArches 0.935) and query label
transfer 0.798 (scArches 0.787). With only 600 cells, both implementations vary strongly
between seeds (accuracy 0.73–0.98).
