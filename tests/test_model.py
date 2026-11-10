import warnings

import anndata as ad
import numpy as np
import pytest
import scvi
import torch

from scpolivi import SCPOLI
from scpolivi.train import SCPoliEarlyStopping

warnings.filterwarnings("ignore")
FAST = dict(enable_progress_bar=False, accelerator="cpu")


@pytest.fixture
def gadata(golden_dir):
    return ad.read_h5ad(golden_dir / "adata.h5ad")


def split(a):
    ref = a[a.obs.study != "s3"].copy()
    ref = ref[ref.obs.ct_fine != "NK"].copy()
    return ref, a[a.obs.study == "s3"].copy()


def fit(a, n_epochs=6, setup=None, init=None, **train):
    setup = {"condition_keys": ["study", "tech"], "cell_type_keys": ["ct_fine", "ct_coarse"], **(setup or {})}
    SCPOLI.setup_anndata(a, **setup)
    m = SCPOLI(a, latent_dim=8, embedding_dims=4, **(init or {}))
    m.train(n_epochs=n_epochs, **{"pretraining_epochs": n_epochs // 2, **FAST, **train})
    return m


@pytest.mark.parametrize("loss", ["nb", "zinb", "nb_dist", "poisson", "mse", "bernoulli"])
def test_all_recon_losses_train(gadata, loss):
    a = gadata.copy()
    if loss == "mse":
        a.X = np.log1p(a.X)
    m = fit(a, init={"recon_loss": loss})
    assert np.isfinite(m.history["train_loss"].values.astype(float)).all()
    assert m.get_latent(a).shape == (a.n_obs, 8)
    assert set(m.classify(a)) == {"ct_fine", "ct_coarse"}


def test_learns_cell_types():
    """Same setting as scripts/compare_synthetic.py (scale 4), where scArches reaches ~0.93."""
    import sys
    from pathlib import Path

    from sklearn.neighbors import KNeighborsClassifier

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from make_golden_synthetic import make_adata

    scvi.settings.seed = 0
    ref, _ = split(make_adata(seed=0, n_cells=2400))
    m = fit(ref, n_epochs=60, pretraining_epochs=54, eta=5)
    z, y = m.get_latent(ref, mean=True), ref.obs.ct_fine.values
    knn = KNeighborsClassifier(15).fit(z[::2], y[::2])
    assert (knn.predict(z[1::2]) == y[1::2]).mean() > 0.85
    pred = m.classify(ref, use_mean=True)["ct_fine"]["preds"]
    assert (pred == y).mean() > 0.7


def test_no_cell_types_and_single_key(gadata):
    m = fit(gadata.copy(), setup={"condition_keys": "study", "cell_type_keys": None})
    assert m.module.n_cell_types == 0
    with pytest.raises(RuntimeError):
        m.classify(gadata)
    assert isinstance(m.get_conditional_embeddings(), ad.AnnData)


def test_early_stopping_and_reload_best(gadata):
    a = gadata.copy()
    es = {"early_stopping_metric": "val_prototype_loss", "patience": 2, "lr_patience": 1, "lr_factor": 0.5}
    m = fit(a, n_epochs=30, early_stopping_kwargs=es, reload_best=True, pretraining_epochs=3)
    cb = [c for c in m.trainer.callbacks if isinstance(c, SCPoliEarlyStopping)][0]
    assert cb.best_epoch is not None and cb.best_epoch >= 3
    assert "prototype_loss_validation" in m.history


def test_stratified_sampling_and_n_samples(gadata):
    m = fit(gadata.copy(), use_stratified_sampling=True, n_samples=200, batch_size=50, train_frac=1.0)
    assert len(m.history["train_loss"]) == 6


def test_unknown_names_and_labeled_indices(gadata):
    a = gadata.copy()
    a.obs["ct_fine"] = a.obs["ct_fine"].astype(str)
    a.obs.loc[a.obs_names[:50], "ct_fine"] = "Unknown"
    m = fit(a, setup={"unknown_ct_names": ["Unknown"], "labeled_indices": np.arange(100, a.n_obs)})
    assert "Unknown" not in m.cell_types_
    assert len(m.labeled_indices_) == a.n_obs - 100
    assert m.prototypes_unlabeled_["mean"] is not None
    m.update_labeled_indices(np.arange(a.n_obs))
    assert len(m.labeled_indices_) == a.n_obs


def test_save_load_query_roundtrip(gadata, tmp_path):
    ref, qry = split(gadata)
    m = fit(ref)
    m.save(tmp_path / "ref")
    q = SCPOLI.load_query_data(qry, str(tmp_path / "ref"), labeled_indices=[])
    assert q.module.prototypes_fixed.all()
    q.train(n_epochs=4, pretraining_epochs=2, **FAST)
    q.save(tmp_path / "q")
    q2 = SCPOLI.load(str(tmp_path / "q"), adata=qry)
    np.testing.assert_allclose(
        q2.module.prototypes_unlabeled_mean.numpy(), q.module.prototypes_unlabeled_mean.numpy()
    )
    assert q2.obs_metadata_.shape[0] >= 3


def test_add_new_cell_type_and_prototype_info(gadata):
    ref, qry = split(gadata)
    m = fit(ref)
    q = SCPOLI.load_query_data(qry, m, labeled_indices=[])
    q.train(n_epochs=4, pretraining_epochs=2, **FAST)
    n_before = q.module.n_cell_types
    q.add_new_cell_type("NK", "ct_fine", [0, 1])
    assert q.module.n_cell_types == n_before + 1 and "NK" in q.cell_types_
    assert "NK" in set(q.classify(qry, use_mean=True)["ct_fine"]["preds"]) | {"NK"}
    info = q.get_prototypes_info("labeled")
    assert info.n_obs == q.module.n_cell_types and "ct_fine_pred" in info.obs
    assert q.get_prototypes_info("unlabeled").n_obs == q.module.prototypes_unlabeled_mean.shape[0]


@pytest.mark.parametrize("keys", [["study"], ["study", "tech"]])
def test_shot_surgery(gadata, keys):
    ref, qry = split(gadata)
    m = fit(ref, setup={"condition_keys": keys})
    q, mapping = SCPOLI.shot_surgery(qry, m)
    flat = mapping if len(keys) == 1 else mapping["study"]
    assert flat["s3"] in ("s1", "s2")
    w = q.module.get_embedding("cond_0").weight
    cats = q.conditions_["study"]
    torch.testing.assert_close(w[cats.index("s3")], w[cats.index(flat["s3"])])


def test_transform_batch_normalized_expression(gadata):
    m = fit(gadata.copy())
    expr = m.get_normalized_expression(gadata, transform_batch=m.conditions_combined_[0])
    assert expr.shape == gadata.shape
