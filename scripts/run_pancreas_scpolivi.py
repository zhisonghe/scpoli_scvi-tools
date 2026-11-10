"""scpolivi on the scArches scPoli surgery tutorial (pancreas); counterpart of make_golden_pancreas.py.

    .venv/bin/python scripts/run_pancreas_scpolivi.py --seed 0 [--graph-weight 10 --graph-mode scratch|finetune]

Writes data/scpolivi/pancreas_<tag>_seed{seed}/ with outputs.npz and metrics.json in the same
layout as the scArches golden outputs.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score

from scpolivi import SCPOLI
from scpolivi.graph import construct_sample_graphs

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1", "smartseq2", "smarter"]
QUERY = ["celseq", "celseq2"]
EARLY_STOPPING = {
    "early_stopping_metric": "val_prototype_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
TRAIN = dict(enable_progress_bar=False, accelerator="cpu")


def sample_graphs(adata):
    # X is log-normalized in this file, hence the "seurat" HVG flavor
    return construct_sample_graphs(
        adata, "study", recompute_hvg_pca=True, n_hvg=2000, n_pcs=30, hvg_flavor="seurat"
    )


def main(seed: int, graph_weight: float, graph_mode: str, graph_sampling: str):
    tag = "base" if graph_weight == 0 else f"graph{graph_mode}_w{graph_weight:g}_{graph_sampling}"
    out = ROOT / "data" / "scpolivi" / f"pancreas_{tag}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    adata = sc.read(ROOT / "data" / "pancreas_sparse.h5ad")
    source = adata[adata.obs.study.isin(REFERENCE)].copy()
    source = source[~source.obs.cell_type.str.contains("alpha")].copy()
    target = adata[adata.obs.study.isin(QUERY)].copy()

    gkw = {}
    if graph_weight > 0:
        gkw = {"graph_weight": graph_weight, "graph_sampling": graph_sampling}

    t0 = time.time()
    SCPOLI.setup_anndata(source, condition_keys="study", cell_type_keys="cell_type")
    ref = SCPOLI(source, embedding_dims=5, recon_loss="nb")
    if graph_weight > 0:
        ref.set_graph(sample_graphs(source), graph_key="study")
    ref.train(
        n_epochs=50,
        pretraining_epochs=40,
        early_stopping_kwargs=EARLY_STOPPING,
        eta=5,
        seed=seed,
        **(gkw if graph_mode == "scratch" else {}),
        **TRAIN,
    )
    if graph_weight > 0 and graph_mode == "finetune":
        ref.fine_tune(n_epochs=10, eta=5, use_early_stopping=False, **gkw, **TRAIN)
    t_ref = time.time() - t0
    ref.save(str(out / "ref"), overwrite=True)

    t0 = time.time()
    qry = SCPOLI.load_query_data(adata=target.copy(), reference_model=ref, labeled_indices=[])
    if graph_weight > 0:
        qry.set_graph(sample_graphs(qry.adata), graph_key="study")
    qry.train(n_epochs=50, pretraining_epochs=40, eta=10, seed=seed, **gkw, **TRAIN)
    t_qry = time.time() - t0
    qry.save(str(out / "query"), overwrite=True)

    res = qry.classify(target.copy(), scale_uncertainties=True)
    res_mean = qry.classify(target.copy(), scale_uncertainties=True, use_mean=True)
    preds = res["cell_type"]["preds"]
    preds_mean = res_mean["cell_type"]["preds"]
    truth = target.obs["cell_type"].astype(str).values
    known = ~pd.Series(truth).str.contains("alpha").values

    lat_src = qry.get_latent(source.copy(), mean=True)
    lat_tgt = qry.get_latent(target.copy(), mean=True)
    emb = qry.get_conditional_embeddings()
    np.savez(
        out / "outputs.npz",
        latent_source=lat_src,
        latent_target=lat_tgt,
        preds=preds.astype(str),
        preds_mean=preds_mean.astype(str),
        uncert=res["cell_type"]["uncert"],
        cond_emb=emb.X,
        proto_labeled=qry.prototypes_labeled_["mean"].cpu().numpy(),
        proto_unlabeled=qry.prototypes_unlabeled_["mean"].cpu().numpy(),
    )
    h = ref.history
    metrics = {
        "seed": seed,
        "variant": tag,
        "time_ref_s": t_ref,
        "time_query_s": t_qry,
        "query_accuracy_known_types": float(accuracy_score(truth[known], preds[known])),
        "query_macro_f1_known_types": float(f1_score(truth[known], preds[known], average="macro")),
        "query_accuracy_known_types_mean": float(accuracy_score(truth[known], preds_mean[known])),
        "query_macro_f1_known_types_mean": float(
            f1_score(truth[known], preds_mean[known], average="macro")
        ),
        "mean_uncert_alpha": float(np.mean(res["cell_type"]["uncert"][~known])),
        "mean_uncert_known": float(np.mean(res["cell_type"]["uncert"][known])),
        "n_unlabeled_prototypes": int(qry.prototypes_unlabeled_["mean"].shape[0]),
        "ref_final_val_loss": float(np.asarray(h["validation_loss"], dtype=float).ravel()[-1]),
        "ref_epochs_run": len(h["validation_loss"]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph-weight", type=float, default=0.0)
    p.add_argument("--graph-mode", choices=["scratch", "finetune"], default="scratch")
    p.add_argument("--graph-sampling", choices=["batch", "neighbors"], default="neighbors")
    a = p.parse_args()
    main(a.seed, a.graph_weight, a.graph_mode, a.graph_sampling)
