# Training

```python
model.train(
    n_epochs=50,
    pretraining_epochs=40,      # default: 0.9 × n_epochs
    eta=5,                      # prototype loss weight
    early_stopping_kwargs={"early_stopping_metric": "val_prototype_loss", "patience": 20},
    batch_size=128,
    seed=0,
)
```

The defaults are scArches' defaults. The most relevant options:

| Option | Default | Effect |
|---|---|---|
| `n_epochs`, `pretraining_epochs` | 100, 90 | length of the VAE-only phase and of the prototype phase |
| `eta` | 1 | weight of the prototype loss; the scPoli tutorial uses 5 (reference) and 10 (query) |
| `lr`, `eps`, `weight_decay` | 1e-3, 0.01, 0.04 | Adam settings; embeddings get no weight decay |
| `alpha_epoch_anneal` | 100 | KL warm-up length in epochs (`None`: no warm-up) |
| `unlabeled_weight` | 0 | weight of the unlabeled prototype loss |
| `clustering`, `clustering_res`, `n_clusters` | `"leiden"`, 2, None | unlabeled prototype initialization |
| `train_frac` | 0.9 | training share; the split is stratified by cell type |
| `use_stratified_sampling`, `n_samples` | False, None | sample cells inversely to their condition size |
| `use_early_stopping`, `early_stopping_kwargs`, `reload_best` | True, –, False | see below |
| `graph_weight`, … | 0 | [graph regularization](graph.md) |
| `accelerator`, `devices` | `"auto"` | Lightning device selection |

## Early stopping

Early stopping uses scArches' rules. It only starts after pretraining, and the first `patience`
checks never stop training. If `reduce_lr` is set, the learning rate is multiplied by `lr_factor`
after `lr_patience` checks without improvement. The monitored metric can be given with
scArches names (`val_unweighted_loss`, `val_prototype_loss`, …) or scvi-tools names
(`unweighted_loss_validation`, …). With `reload_best=True`, the weights and prototypes of the
best epoch are restored at the end.

## Logged metrics

`model.history` holds per-epoch curves: `train_loss`, `validation_loss`, and for both splits
`cvae_loss`, `prototype_loss`, `labeled_loss`, `unlabeled_loss`, `graph_loss` and
`unweighted_loss` (ELBO + unweighted prototype loss), e.g.
`model.history["prototype_loss_validation"]`.

## Reproducibility

`seed` sets `scvi.settings.seed`. Weight initialization happens when the model is
constructed, so set `scvi.settings.seed` before `SCPOLI(...)` for fully reproducible runs.
scArches reseeds with 2020 by default; scpolivi does not.

## Fine-tuning

`model.fine_tune(n_epochs, graph_weight, ...)` continues training a trained model: no pretraining
phase, no KL warm-up, prototypes kept, and a lower default learning rate (1e-4). It is made for
adding the [graph loss](graph.md) afterwards. With `graph_weight=0` it simply trains more epochs.
