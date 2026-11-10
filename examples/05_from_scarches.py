"""Walkthrough 5: continue with a model trained in scArches.

Loads a reference saved by scArches ``scPoli.save`` (no scArches installation needed), checks
that it embeds cells, and maps a query onto it with scpolivi.

    python examples/05_from_scarches.py --scarches-model path/to/scarches_ref --data data/pancreas_sparse.h5ad
"""

import argparse

import scanpy as sc

from scpolivi import SCPOLI

p = argparse.ArgumentParser()
p.add_argument("--scarches-model", required=True, help="directory written by scArches scPoli.save")
p.add_argument("--data", default="data/pancreas_sparse.h5ad")
p.add_argument("--epochs", type=int, default=50)
args = p.parse_args()

adata = sc.read(args.data)
reference = adata[~adata.obs.study.isin(["celseq", "celseq2"])].copy()
reference = reference[reference.obs.cell_type != "alpha"].copy()

# The AnnData needs the model's genes (same order) and condition columns.
model = SCPOLI.from_scarches(args.scarches_model, reference)
print(model)
print("conditions:", model.conditions_)
print("cell types:", list(model.cell_types_))
z = model.get_latent(reference, mean=True)
print("reference latent:", z.shape)

# From here on it is an ordinary scpolivi model: save it in scvi-tools format, map queries.
model.save("results/05_from_scarches/reference_model", overwrite=True)
query = adata[adata.obs.study.isin(["celseq", "celseq2"])].copy()
query_model = SCPOLI.load_query_data(query, reference_model=model, labeled_indices=[])
query_model.train(n_epochs=args.epochs, pretraining_epochs=int(0.8 * args.epochs), eta=10, seed=0)
print(query_model.classify(query, use_mean=True)["cell_type"]["preds"][:10])
