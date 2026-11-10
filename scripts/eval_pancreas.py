"""Uniform metrics for pancreas runs (scArches goldens and scpolivi outputs).

    .venv/bin/python scripts/eval_pancreas.py data/golden/pancreas_seed0 data/scpolivi/pancreas_base_seed0 ...

Metrics on the joint latent (reference without alpha + query):
  knn_ct_acc        15-NN leave-one-out cell-type accuracy (bio conservation)
  ct_asw            cell-type silhouette, rescaled to [0, 1] (bio conservation)
  study_entropy     mean normalized entropy of studies among 30 NN (batch mixing)
  within_nn_overlap mean fraction of a cell's 15 within-study expression neighbours (per-study
                    HVG PCA) that are also its 15 within-study latent neighbours
  query_acc         label-transfer accuracy on known query types (from metrics.json)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1", "smartseq2", "smarter"]
QUERY = ["celseq", "celseq2"]


def load_obs():
    adata = sc.read(ROOT / "data" / "pancreas_sparse.h5ad")
    source = adata[adata.obs.study.isin(REFERENCE)].copy()
    source = source[~source.obs.cell_type.str.contains("alpha")].copy()
    target = adata[adata.obs.study.isin(QUERY)].copy()
    return sc.concat([source, target])


_EXPR_NN: dict = {}


def expression_neighbors(adata, k=15):
    if not _EXPR_NN:
        for s in adata.obs.study.unique():
            idx = np.flatnonzero(adata.obs.study.values == s)
            sub = adata[idx].copy()
            sc.pp.highly_variable_genes(sub, n_top_genes=2000, flavor="seurat")
            sc.pp.pca(sub, n_comps=30, mask_var="highly_variable")
            nn = NearestNeighbors(n_neighbors=k + 1).fit(sub.obsm["X_pca"])
            _EXPR_NN[s] = (idx, nn.kneighbors(return_distance=False)[:, :k])
    return _EXPR_NN


def evaluate(run: Path, adata) -> dict:
    o = np.load(run / "outputs.npz", allow_pickle=True)
    z = np.concatenate([o["latent_source"], o["latent_target"]])
    ct = adata.obs.cell_type.astype(str).values
    study = adata.obs.study.astype(str).values
    rng = np.random.default_rng(0)

    nn = NearestNeighbors(n_neighbors=31).fit(z)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    knn_pred = pd.DataFrame(ct[idx[:, :15]]).mode(axis=1)[0].values
    knn_acc = float((knn_pred == ct).mean())

    studies = np.unique(study)
    codes = np.searchsorted(studies, study)
    counts = np.stack([(codes[idx] == s).sum(1) for s in range(len(studies))], 1) / idx.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(counts > 0, counts * np.log(counts), 0), 1) / np.log(len(studies))

    sub = rng.choice(len(z), min(5000, len(z)), replace=False)
    asw = (silhouette_score(z[sub], ct[sub]) + 1) / 2

    overlaps = []
    for s, (cells, expr_nn) in expression_neighbors(adata).items():
        k = expr_nn.shape[1]
        lat_nn = NearestNeighbors(n_neighbors=k + 1).fit(z[cells]).kneighbors(return_distance=False)[:, 1:]
        overlaps.append(np.mean([len(set(a) & set(b)) / k for a, b in zip(expr_nn, lat_nn)]))

    m = json.loads((run / "metrics.json").read_text())
    return {
        "run": run.name,
        "query_acc": round(m["query_accuracy_known_types"], 4),
        "query_f1": round(m["query_macro_f1_known_types"], 4),
        "uncert_alpha": round(m["mean_uncert_alpha"], 3),
        "uncert_known": round(m["mean_uncert_known"], 3),
        "knn_ct_acc": round(knn_acc, 4),
        "ct_asw": round(float(asw), 4),
        "study_entropy": round(float(ent.mean()), 4),
        "within_nn_overlap": round(float(np.mean(overlaps)), 4),
        "time_ref_s": round(m["time_ref_s"], 1),
    }


if __name__ == "__main__":
    adata = load_obs()
    rows = [evaluate(Path(p), adata) for p in sys.argv[1:]]
    df = pd.DataFrame(rows).set_index("run")
    print(df.to_string())
