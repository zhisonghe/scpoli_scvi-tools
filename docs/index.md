# scpolivi

**scPoli on scvi-tools.** scpolivi reimplements [scPoli](https://doi.org/10.1038/s41592-023-02035-2)
on [scvi-tools](https://scvi-tools.org). scPoli is a model for population-level integration of
single-cell data and reference mapping. The original implementation lives in
[scArches](https://github.com/theislab/scarches), which is no longer actively maintained.
scpolivi does not depend on scArches. It reproduces scArches' scPoli numerically and adds
features scArches lacks.

<div class="grid cards" markdown>

- **Integrate many samples**: one learned embedding per condition key (study, sample,
  technology, …), so samples become comparable vectors.
- **Transfer labels with uncertainty**: cell-type prototypes in the latent space, on one or
  several hierarchy levels.
- **Map new data**: extend a trained reference with query data (scArches surgery), project
  cells, or map conditions without training.
- **Keep local structure**: optional graph Laplacian loss (from scANCVI) that preserves
  within-sample neighbourhoods.

</div>

## Why scpolivi

| | scArches scPoli | scpolivi |
|---|---|---|
| Maintained dependency stack | scArches (inactive) | scvi-tools 1.5 (active) |
| Works with pandas 3 | no (fails during model construction) | yes |
| Numerically the same model | – | forward pass ≤ 1e-5, imported checkpoints ≤ 1e-4 |
| Multiple condition keys, hierarchical cell types | yes | yes |
| `shot_surgery` with several condition keys | no | yes |
| `bernoulli`, `nb_dist` losses | crash | work |
| scvi-tools tools: `get_normalized_expression`, differential expression, minified AnnData, Lightning training on CPU/GPU/MPS | – | yes |
| Graph Laplacian regularization, fine-tuning | – | yes |
| Load scArches checkpoints | – | `SCPOLI.from_scarches` |

## Minimal example

```python
from scpolivi import SCPOLI

SCPOLI.setup_anndata(reference, condition_keys="study", cell_type_keys="cell_type")
model = SCPOLI(reference, embedding_dims=5)
model.train(n_epochs=50, pretraining_epochs=40, eta=5)

query_model = SCPOLI.load_query_data(query, reference_model=model, labeled_indices=[])
query_model.train(n_epochs=50, pretraining_epochs=40, eta=10)

labels = query_model.classify(query, use_mean=True)["cell_type"]   # preds, uncert
latent = query_model.get_latent(query, mean=True)
```

## Where to go next

- [Installation](installation.md)
- [Concepts](concepts.md): what the model does and the meaning of its options.
- [Tutorials](tutorials/index.md): five executed Jupyter notebooks on the pancreas data.
- [Coming from scArches](scarches.md): argument mapping, behaviour differences, importing models.
- [API reference](api/index.md)

## Citation

If you use scpolivi, please cite the scPoli paper and scvi-tools:

- De Donno, C. et al. *Population-level integration of single-cell datasets enables multi-scale
  analysis across samples.* Nature Methods 20, 1683–1692 (2023).
- Gayoso, A. et al. *A Python library for probabilistic analysis of single-cell omics data.*
  Nature Biotechnology 40, 163–166 (2022).
- For query mapping (architectural surgery): Lotfollahi, M. et al. *Mapping single-cell data to
  reference atlases by transfer learning.* Nature Biotechnology 40, 121–130 (2022).
