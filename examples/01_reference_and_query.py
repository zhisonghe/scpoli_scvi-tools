"""Walkthrough 1: build a reference, map a query, transfer labels.

The scArches scPoli surgery tutorial with scpolivi. Pancreas data from 9 studies
(https://figshare.com/ndownloader/files/41581626): 7 studies form the reference, from which all
alpha cells are removed; the 2 CEL-Seq studies are the query. Alpha cells are thus a cell type
the reference has never seen, and should get high uncertainty.

    python examples/01_reference_and_query.py --data data/pancreas_sparse.h5ad --out results/01
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, classification_report

from scpolivi import SCPOLI

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/pancreas_sparse.h5ad")
p.add_argument("--out", default="results/01_reference_and_query")
p.add_argument("--epochs", type=int, default=50)
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = out

# --- 1. data -------------------------------------------------------------------------------
adata = sc.read(args.data)
reference_studies = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1", "smartseq2", "smarter"]
query_studies = ["celseq", "celseq2"]
reference = adata[adata.obs.study.isin(reference_studies)].copy()
reference = reference[reference.obs.cell_type != "alpha"].copy()  # hold out a cell type
query = adata[adata.obs.study.isin(query_studies)].copy()

# --- 2. reference model --------------------------------------------------------------------
# One condition key ("study"), one cell-type level. setup_anndata stores which columns to use.
SCPOLI.setup_anndata(reference, condition_keys="study", cell_type_keys="cell_type")
model = SCPOLI(reference, embedding_dims=5, recon_loss="nb")
# 80% of the epochs train the VAE only ("pretraining"); then the prototype loss (weight eta)
# is added. Early stopping watches the prototype loss, as in the scArches tutorial.
early_stopping = {
    "early_stopping_metric": "val_prototype_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
model.train(
    n_epochs=args.epochs,
    pretraining_epochs=int(0.8 * args.epochs),
    eta=5,
    early_stopping_kwargs=early_stopping,
    seed=0,
)
model.save(out / "reference_model", overwrite=True, save_anndata=True)

# --- 3. query mapping ----------------------------------------------------------------------
# The query studies get new embeddings; all other weights stay frozen. labeled_indices=[]
# treats every query cell as unlabeled (their labels are only used for evaluation below).
query_model = SCPOLI.load_query_data(query, reference_model=model, labeled_indices=[])
query_model.train(n_epochs=args.epochs, pretraining_epochs=int(0.8 * args.epochs), eta=10, seed=0)
query_model.save(out / "query_model", overwrite=True, save_anndata=True)

# --- 4. label transfer ---------------------------------------------------------------------
result = query_model.classify(query, scale_uncertainties=True, use_mean=True)["cell_type"]
query.obs["predicted"] = result["preds"]
query.obs["uncertainty"] = result["uncert"]
known = query.obs.cell_type != "alpha"
print("accuracy on cell types known to the reference:",
      round(accuracy_score(query.obs.cell_type[known], query.obs.predicted[known]), 4))
print(classification_report(query.obs.cell_type[known], query.obs.predicted[known], zero_division=0))
print(query.obs.groupby(query.obs.cell_type == "alpha")["uncertainty"].mean()
      .rename({True: "alpha (unseen)", False: "known types"}))

# --- 5. joint embedding --------------------------------------------------------------------
joint = sc.concat([reference, query], label="dataset", keys=["reference", "query"])
joint.obsm["X_scpoli"] = query_model.get_latent(joint, mean=True)
joint.obs["uncertainty"] = np.nan
joint.obs.loc[query.obs_names, "uncertainty"] = query.obs["uncertainty"].values
sc.pp.neighbors(joint, use_rep="X_scpoli")
sc.tl.umap(joint)
sc.pl.umap(joint, color=["study", "cell_type", "dataset", "uncertainty"], ncols=2,
           frameon=False, save="_joint.png", show=False)

# prototypes on top of the cells
protos = query_model.get_prototypes_info("labeled")
print(protos.obs[["cell_type", "cell_type_pred"]])
pd.DataFrame(query.obs[["cell_type", "predicted", "uncertainty"]]).to_csv(out / "query_predictions.csv")
print(f"figures and models written to {out}")
