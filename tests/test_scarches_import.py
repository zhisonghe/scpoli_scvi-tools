"""Import of trained scArches scPoli checkpoints (reference and query) without scArches."""

import json
import warnings

import anndata as ad
import numpy as np
import pytest
import torch

from scpolivi import SCPOLI
from scpolivi.module import _prototypes as P

from conftest import ACCELERATOR

warnings.filterwarnings("ignore")


def _load(golden_dir, which):
    d = golden_dir / f"trained_{which}"
    adata = ad.read_h5ad(d / f"adata_{which}.h5ad")
    model = SCPOLI.from_scarches(str(d), adata, accelerator=ACCELERATOR)
    return d, adata, model, dict(np.load(d / "outputs.npz", allow_pickle=True))


def test_reference_import_matches_scarches(golden_dir):
    d, adata, model, gold = _load(golden_dir, "ref")
    meta = json.loads((d / "meta.json").read_text())
    assert list(model.cell_types_) == list(meta["cell_types_"])
    assert model.conditions_ == meta["conditions_"]
    assert model.conditions_combined_ == meta["conditions_combined_"]
    np.testing.assert_allclose(model.get_latent(adata, mean=True), gold["latent_mean"], atol=1e-4)
    for key in ("study", "tech"):
        np.testing.assert_allclose(model.get_conditional_embeddings()[key].X, gold[f"cond_emb_{key}"])
    assert model.prototypes_unlabeled_["mean"].shape[0] == meta["n_unlabeled_prototypes"]

    # deterministic classification == nearest prototype on scArches' own latent means
    res = model.classify(adata, use_mean=True)
    protos = model.prototypes_labeled_["mean"]
    names = np.array(list(model.cell_types_))
    for level in ("ct_fine", "ct_coarse"):
        classes = torch.tensor([i for i, lv in enumerate(model.cell_types_.values()) if level in lv])
        pred, _, _ = P.classify(torch.tensor(gold["latent_mean"]), protos, classes)
        assert (res[level]["preds"] == names[pred.numpy()]).mean() > 0.999
        # scArches classifies a sampled z: compare accuracy (averaged over draws), not cells
        truth = adata.obs[level].astype(str).values
        acc = np.mean([(model.classify(adata)[level]["preds"] == truth).mean() for _ in range(10)])
        assert abs(acc - (gold[f"{level}_preds"] == truth).mean()) < 0.1
        # scaled uncertainties live in [0, 1]
        u = model.classify(adata, scale_uncertainties=True)[level]["uncert"]
        assert u.min() >= 0 and u.max() <= 1


def test_query_import_matches_scarches(golden_dir):
    d, adata, model, gold = _load(golden_dir, "query")
    meta = json.loads((d / "meta.json").read_text())
    assert list(model.cell_types_) == list(meta["cell_types_"])
    assert "NK" in model.cell_types_
    np.testing.assert_allclose(model.get_latent(adata, mean=True), gold["latent_mean"], atol=1e-4)
    ref = ad.read_h5ad(golden_dir / "trained_ref" / "adata_ref.h5ad")
    ref_model = SCPOLI.from_scarches(str(d), ref, accelerator=ACCELERATOR)
    np.testing.assert_allclose(ref_model.get_latent(ref, mean=True), gold["ref_latent_mean"], atol=1e-4)


def test_surgery_from_imported_reference(golden_dir):
    _, ref_adata, ref_model, _ = _load(golden_dir, "ref")
    full = ad.read_h5ad(golden_dir / "adata.h5ad")
    qry = full[full.obs.study == "s3"].copy()
    nk = np.where(qry.obs.ct_fine == "NK")[0][:20]
    q, new = SCPOLI.load_query_data(qry, ref_model, labeled_indices=nk, return_new_conditions=True)
    assert new == {"study": ["s3"], "tech": []}
    q.train(n_epochs=4, pretraining_epochs=2, eta=10, enable_progress_bar=False, accelerator=ACCELERATOR)
    assert "NK" in q.cell_types_
    # surgery only trains embeddings and dispersion: network weights and the embeddings of
    # reference-only conditions are unchanged (shared conditions, here "tech", may move)
    ref_sd, q_sd = ref_model.module.state_dict(), q.module.state_dict()
    for k, v in ref_sd.items():
        if k.startswith(("encoder.", "decoder.")):
            np.testing.assert_array_equal(q_sd[k].cpu().numpy(), v.cpu().numpy())
    w_ref = ref_sd["_embeddings_dict.cond_0.weight"]
    np.testing.assert_allclose(
        q_sd["_embeddings_dict.cond_0.weight"][: len(w_ref)].cpu().numpy(), w_ref.cpu().numpy()
    )
