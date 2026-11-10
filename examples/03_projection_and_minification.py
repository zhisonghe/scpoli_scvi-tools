"""Walkthrough 3: project new cells, map without training, keep a small model.

Uses the reference model saved by walkthrough 1 (``results/01_reference_and_query``).
* projection of cells from known conditions, without cell-type columns, with genes in another
  order;
* ``shot_surgery``: map a new study onto the most similar reference study without training;
* minification: keep only the latent posterior to share a small model + AnnData.

    python examples/03_projection_and_minification.py --data data/pancreas_sparse.h5ad
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc

from scpolivi import SCPOLI

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/pancreas_sparse.h5ad")
p.add_argument("--reference", default="results/01_reference_and_query/reference_model")
p.add_argument("--out", default="results/03_projection_and_minification")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

adata = sc.read(args.data)
model = SCPOLI.load(args.reference)  # loads the saved reference AnnData as well

# --- projection ----------------------------------------------------------------------------
# Any cells of known conditions can be embedded and classified. Cell-type columns are not
# needed; genes are matched by name after prepare_query_anndata (here: shuffled order).
new_cells = adata[adata.obs.study == "inDrop1"].copy()
new_cells.obs = new_cells.obs[["study"]].copy()
new_cells = new_cells[:, np.random.default_rng(0).permutation(new_cells.n_vars)].copy()
SCPOLI.prepare_query_anndata(new_cells, model)  # reorders (and zero-pads missing) genes
z = model.get_latent(new_cells, mean=True)
pred = model.classify(new_cells, use_mean=True)["cell_type"]["preds"]
print("projected", z.shape, "cells; predicted types:", {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))})

# Cells of an unknown condition must be mapped first:
try:
    model.get_latent(adata[adata.obs.study == "celseq"].copy())
except ValueError as e:
    print("expected error:", e)

# --- shot surgery --------------------------------------------------------------------------
# Each new study borrows the embedding of the reference study that reconstructs it best.
query = adata[adata.obs.study.isin(["celseq", "celseq2"])].copy()
shot_model, mapping = SCPOLI.shot_surgery(query, reference_model=model, labeled_indices=[])
print("shot surgery mapping:", mapping)
res = shot_model.classify(query, use_mean=True)["cell_type"]
known = query.obs.cell_type != "alpha"
print("accuracy without any query training:",
      round(float((res["preds"][known] == query.obs.cell_type[known]).mean()), 4))

# --- minification --------------------------------------------------------------------------
# Replace the expression matrix by the latent posterior: latents, classification and
# normalized expression still work, training does not.
size_before = model.adata.X.data.nbytes if hasattr(model.adata.X, "data") else model.adata.X.nbytes
model.minify_adata()
print(f"expression data: {size_before / 1e6:.1f} MB -> {model.adata.X.nnz} stored values")
model.save(out / "reference_minified", overwrite=True, save_anndata=True)
small = SCPOLI.load(out / "reference_minified")
print("minified model latent:", small.get_latent(mean=True).shape)
