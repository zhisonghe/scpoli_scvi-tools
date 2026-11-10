# 5. Models from scArches

Use a reference model trained and saved with scArches (`scPoli.save`) without installing
scArches. `SCPOLI.from_scarches` reads the checkpoint, registers the AnnData with scArches'
category order and converts the weights and prototypes. From then on the model behaves like any
scpolivi model: here it is saved in scvi-tools format and a query is mapped onto it.

```bash
python examples/05_from_scarches.py --scarches-model path/to/scarches_ref --data data/pancreas_sparse.h5ad
```

```python
--8<-- "examples/05_from_scarches.py"
```
