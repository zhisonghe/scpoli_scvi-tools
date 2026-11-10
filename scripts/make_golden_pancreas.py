"""Golden outputs for the scArches scPoli surgery tutorial (pancreas), run in the parity env.

    .venv-parity/bin/python scripts/make_golden_pancreas.py [--seed 0]

Expects data/pancreas.h5ad (https://figshare.com/ndownloader/files/41581626).
Writes data/golden/pancreas_seed{seed}/ with reference/query checkpoints, latents,
predictions, uncertainties, embeddings, timings and label-transfer metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import accuracy_score, f1_score

from scarches.models.scpoli import scPoli

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


def main(seed: int):
    out = ROOT / "data" / "golden" / f"pancreas_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    adata = sc.read(ROOT / "data" / "pancreas.h5ad")
    source = adata[adata.obs.study.isin(REFERENCE)].copy()
    source = source[~source.obs.cell_type.str.contains("alpha")].copy()
    target = adata[adata.obs.study.isin(QUERY)].copy()

    t0 = time.time()
    ref = scPoli(
        adata=source,
        condition_keys="study",
        cell_type_keys="cell_type",
        embedding_dims=5,
        recon_loss="nb",
    )
    ref.train(n_epochs=50, pretraining_epochs=40, early_stopping_kwargs=EARLY_STOPPING, eta=5, seed=seed)
    t_ref = time.time() - t0
    ref.save(str(out / "ref"), overwrite=True)

    t0 = time.time()
    qry = scPoli.load_query_data(adata=target, reference_model=ref, labeled_indices=[])
    qry.train(n_epochs=50, pretraining_epochs=40, eta=10, seed=seed)
    t_qry = time.time() - t0
    qry.save(str(out / "query"), overwrite=True)

    qry.model.eval()
    res = qry.classify(target, scale_uncertainties=True)
    preds = res["cell_type"]["preds"]
    truth = target.obs["cell_type"].astype(str).values
    known = ~pd.Series(truth).str.contains("alpha").values  # alpha absent from reference

    lat_src = qry.get_latent(source, mean=True)
    lat_tgt = qry.get_latent(target, mean=True)
    emb = qry.get_conditional_embeddings()

    np.savez(
        out / "outputs.npz",
        latent_source=lat_src,
        latent_target=lat_tgt,
        preds=preds.astype(str),
        uncert=res["cell_type"]["uncert"],
        cond_emb=emb.X,
        proto_labeled=qry.prototypes_labeled_["mean"].cpu().numpy(),
        proto_unlabeled=qry.prototypes_unlabeled_["mean"].cpu().numpy(),
    )
    metrics = {
        "seed": seed,
        "time_ref_s": t_ref,
        "time_query_s": t_qry,
        "query_accuracy_known_types": float(accuracy_score(truth[known], preds[known])),
        "query_macro_f1_known_types": float(f1_score(truth[known], preds[known], average="macro")),
        "mean_uncert_alpha": float(np.mean(res["cell_type"]["uncert"][~known])),
        "mean_uncert_known": float(np.mean(res["cell_type"]["uncert"][known])),
        "n_unlabeled_prototypes": int(qry.prototypes_unlabeled_["mean"].shape[0]),
        "ref_history_keys": sorted(ref.trainer.logs.keys()),
        "ref_final_val_loss": float(ref.trainer.logs["val_loss"][-1]),
        "ref_epochs_run": len(ref.trainer.logs["val_loss"]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args().seed)
