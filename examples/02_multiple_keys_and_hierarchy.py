"""Walkthrough 2: several condition keys, a cell-type hierarchy, partial labels, new cell types.

Builds on the pancreas data of walkthrough 1 and adds
* a second condition key, the sequencing technology (one embedding per key);
* a coarse cell-type level (endocrine / exocrine / ...) above the fine one; both levels share
  one prototype namespace and both are predicted;
* cells annotated "unknown" and a reference that is only partly labeled;
* a cell type that appears only in the query, added as a new labeled prototype afterwards.

    python examples/02_multiple_keys_and_hierarchy.py --data data/pancreas_sparse.h5ad
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.metrics import accuracy_score

from scpolivi import SCPOLI

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/pancreas_sparse.h5ad")
p.add_argument("--out", default="results/02_multiple_keys_and_hierarchy")
p.add_argument("--epochs", type=int, default=50)
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

adata = sc.read(args.data)
technology = {
    "inDrop1": "inDrop", "inDrop2": "inDrop", "inDrop3": "inDrop", "inDrop4": "inDrop",
    "fluidigmc1": "Fluidigm C1", "smartseq2": "Smart-seq2", "smarter": "SMARTer",
    "celseq": "CEL-Seq", "celseq2": "CEL-Seq",
}
coarse = {
    "alpha": "endocrine", "beta": "endocrine", "delta": "endocrine", "gamma": "endocrine",
    "epsilon": "endocrine", "acinar": "exocrine", "ductal": "exocrine",
    "activated_stellate": "stromal", "quiescent_stellate": "stromal", "endothelial": "stromal",
    "schwann": "stromal", "mast": "immune", "macrophage": "immune", "t_cell": "immune",
}
adata.obs["technology"] = adata.obs.study.map(technology).astype(str)
adata.obs["cell_type_coarse"] = adata.obs.cell_type.map(coarse).fillna("other").astype(str)

reference = adata[~adata.obs.study.isin(["celseq", "celseq2"])].copy()
reference = reference[reference.obs.cell_type != "alpha"].copy()
query = adata[adata.obs.study.isin(["celseq", "celseq2"])].copy()

# 10% of reference cells are annotated "unknown"; these never define a prototype.
rng = np.random.default_rng(0)
reference.obs["cell_type"] = reference.obs.cell_type.astype(str)
reference.obs.loc[rng.random(reference.n_obs) < 0.1, "cell_type"] = "unknown"
# And only 70% of the reference counts as labeled (the rest only trains the VAE and the
# unlabeled prototypes).
labeled = np.flatnonzero(rng.random(reference.n_obs) < 0.7)

SCPOLI.setup_anndata(
    reference,
    condition_keys=["study", "technology"],
    cell_type_keys=["cell_type", "cell_type_coarse"],
    unknown_ct_names=["unknown"],
    labeled_indices=labeled,
)
model = SCPOLI(reference, embedding_dims=[5, 3], latent_dim=10)
print("prototypes:", model.cell_types_)
model.train(n_epochs=args.epochs, pretraining_epochs=int(0.8 * args.epochs), eta=5, seed=0)

# Condition embeddings, one AnnData per key: technologies can be compared directly.
emb = model.get_conditional_embeddings()
print("technology embeddings:\n", emb["technology"].to_df().round(2))

# Query: CEL-Seq is a new study and a new technology, so both keys get new embeddings.
query_model, new = SCPOLI.load_query_data(
    query, reference_model=model, labeled_indices=[], return_new_conditions=True
)
print("new conditions:", new)
query_model.train(n_epochs=args.epochs, pretraining_epochs=int(0.8 * args.epochs), eta=10, seed=0)

res = query_model.classify(query, scale_uncertainties=True, use_mean=True)
for level in ["cell_type", "cell_type_coarse"]:
    truth = query.obs[level].astype(str)
    keep = query.obs.cell_type != "alpha"
    print(level, "accuracy (known types):",
          round(accuracy_score(truth[keep], res[level]["preds"][keep]), 4))

# New cell type: which unlabeled prototypes collect the unseen alpha cells?
info = query_model.get_prototypes_info("unlabeled")
z = query_model.get_latent(query, mean=True)
d = ((z[:, None, :] - info.X[None]) ** 2).sum(-1)
nearest = d.argmin(1)
alpha = (query.obs.cell_type == "alpha").to_numpy()
share = np.bincount(nearest[alpha], minlength=info.n_obs) / np.maximum(np.bincount(nearest, minlength=info.n_obs), 1)
candidates = [int(i) for i in np.flatnonzero(share > 0.8)]
print("unlabeled prototypes dominated by alpha cells:", candidates)
if candidates:
    query_model.add_new_cell_type("alpha", obs_key="cell_type", prototypes=candidates, adata=query)
    res = query_model.classify(query, use_mean=True)["cell_type"]
    print("alpha cells now predicted as alpha:", round(float((res["preds"][alpha] == "alpha").mean()), 3))
query_model.save(out / "query_model", overwrite=True)
