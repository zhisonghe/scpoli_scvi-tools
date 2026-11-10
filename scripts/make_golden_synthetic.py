"""Generate golden (reference) outputs from scArches scPoli on small synthetic data.

Run in the parity environment only (it imports scArches):
    .venv-parity/bin/python scripts/make_golden_synthetic.py

Outputs go to tests/golden/synthetic/. They are consumed by parity tests in the runtime
environment, which never imports scArches.
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import torch

try:
    from scarches.models.scpoli import scPoli
    from scarches.trainers.scpoli.trainer import scPoliTrainer
except ImportError:  # imported for make_adata only
    scPoli = scPoliTrainer = None

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "synthetic"


def make_adata(seed: int = 0, n_cells: int = 600, n_genes: int = 120) -> ad.AnnData:
    """Counts with cell-type and study/tech effects, 2 condition keys, 2 label levels."""
    rng = np.random.default_rng(seed)
    fine = np.array(["T_CD4", "T_CD8", "B", "Mono", "NK"])
    coarse_of = {"T_CD4": "T", "T_CD8": "T", "B": "Lymph_B", "Mono": "Myeloid", "NK": "Lymph_NK"}
    studies = np.array(["s1", "s2", "s3"])
    techs = np.array(["10x", "smartseq"])

    ct = rng.choice(fine, n_cells)
    study = rng.choice(studies, n_cells)
    tech = rng.choice(techs, n_cells)

    ct_prog = {c: rng.gamma(2.0, 1.0, n_genes) for c in fine}
    study_eff = {s: np.exp(rng.normal(0, 0.3, n_genes)) for s in studies}
    tech_eff = {t: np.exp(rng.normal(0, 0.3, n_genes)) for t in techs}
    lib = rng.lognormal(np.log(800), 0.4, n_cells)

    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        mu = ct_prog[ct[i]] * study_eff[study[i]] * tech_eff[tech[i]]
        mu = mu / mu.sum() * lib[i]
        X[i] = rng.negative_binomial(5, 5 / (5 + mu))

    obs = pd.DataFrame(
        {
            "study": study,
            "tech": tech,
            "ct_fine": ct,
            "ct_coarse": [coarse_of[c] for c in ct],
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"g{j}" for j in range(n_genes)]
    return adata


FORWARD_CONFIGS = {
    # name: (scPoli kwargs)
    "nb_1key": dict(condition_keys="study", recon_loss="nb"),
    "nb_2keys": dict(condition_keys=["study", "tech"], recon_loss="nb", embedding_dims=[4, 3]),
    "zinb_2keys": dict(condition_keys=["study", "tech"], recon_loss="zinb"),
    "poisson_1key": dict(condition_keys="study", recon_loss="poisson"),
    "mse_1key": dict(condition_keys="study", recon_loss="mse"),
    "nb_dist_1key": dict(condition_keys="study", recon_loss="nb_dist"),
    "bernoulli_1key": dict(condition_keys="study", recon_loss="bernoulli"),
    "nb_enc_only": dict(condition_keys="study", recon_loss="nb", inject_condition=["encoder"]),
    "nb_dec_only": dict(condition_keys="study", recon_loss="nb", inject_condition=["decoder"]),
    "nb_bn_deep": dict(
        condition_keys=["study", "tech"],
        recon_loss="nb",
        use_bn=True,
        use_ln=False,
        hidden_layer_sizes=[48, 24],
    ),
    "nb_ln_deep3": dict(condition_keys="study", recon_loss="nb", hidden_layer_sizes=[64, 32, 16]),
}


def batch_tensors(model, adata):
    """Build the scArches batch dict for all cells, in scArches' own encoding."""
    from scarches.dataset import MultiConditionAnnotatedDataset

    m = model.model
    ds = MultiConditionAnnotatedDataset(
        adata,
        condition_keys=model.condition_keys_,
        condition_encoders=m.condition_encoders,
        conditions_combined_encoder=m.conditions_combined_encoder,
        cell_type_keys=model.cell_type_keys_,
        cell_type_encoder=m.cell_type_encoder,
    )
    idx = np.arange(len(adata))
    x = ds.data[idx] if not ds._is_sparse else torch.tensor(ds.data[idx].toarray())
    return {
        "x": x.float(),
        "batch": ds.conditions[idx],
        "combined_batch": ds.conditions_combined[idx],
        "sizefactor": ds.size_factors[idx].float(),
    }


@torch.no_grad()
def forward_golden(name, kwargs, adata, out_dir):
    torch.manual_seed(1)
    np.random.seed(1)
    a = adata.copy()
    if kwargs["recon_loss"] == "bernoulli":
        a.X = (a.X > 0).astype(np.float32)
    model = scPoli(adata=a, cell_type_keys=["ct_fine", "ct_coarse"], latent_dim=8, **kwargs)
    m = model.model
    m.eval()
    tens = batch_tensors(model, a)
    record = {"name": name, "kwargs": kwargs, "status": "ok"}

    emb = torch.hstack([m.embeddings[i](tens["batch"][:, i]) for i in range(tens["batch"].shape[1])])
    x_in = tens["x"] if m.recon_loss == "mse" else torch.log1p(tens["x"])
    enc_c = emb if "encoder" in m.inject_condition else None
    z_mean, z_log_var = m.encoder(x_in, enc_c)
    dec_c = emb if "decoder" in m.inject_condition else None
    dec_out = m.decoder(z_mean, dec_c)

    arrays = {"z_mean": z_mean.numpy(), "z_log_var": z_log_var.numpy()}
    if dec_out is None:
        record["status"] = "decoder_returned_none"
    else:
        arrays["dec_out0"] = dec_out[0].numpy()
        if m.recon_loss == "zinb":
            arrays["dec_dropout"] = dec_out[1].numpy()

    # Deterministic full forward: replace sampling by the mean.
    m.sampling = lambda mu, log_var: mu
    try:
        _, recon, kl, _ = m(**tens)
        arrays["recon_loss"] = np.array(recon.item())
        arrays["kl"] = np.array(kl.item())
    except Exception as err:  # noqa: BLE001
        record["status"] = f"forward_error: {type(err).__name__}: {err}"
        record["traceback"] = traceback.format_exc()

    d = out_dir / name
    if d.exists():
        shutil.rmtree(d)
    model.save(str(d), overwrite=True)  # model_params.pt, attr.pkl, var_names.csv
    np.savez(d / "outputs.npz", **arrays)
    (d / "record.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def prototype_function_golden(out_dir):
    """Golden values for the trainer's prototype loss/update functions on random inputs."""
    g = torch.Generator().manual_seed(3)
    n, d, k, k_unlab = 200, 8, 6, 9
    latent = torch.randn(n, d, generator=g)
    protos = torch.randn(k, d, generator=g)
    protos_cov = torch.randn(k, d, d, generator=g)
    protos_unlab = torch.randn(k_unlab, d, generator=g)
    # two label levels in a shared namespace; -1 = unknown
    lvl1 = torch.randint(0, 4, (n,), generator=g)
    lvl2 = torch.randint(4, 6, (n,), generator=g)
    lvl1[:15] = -1
    labels = torch.stack([lvl1, lvl2], dim=1)

    out = {}
    for p in (1.0, 2.0):
        fake = SimpleNamespace(device=torch.device("cpu"), p_prototype_loss=p, model=SimpleNamespace(n_cell_types=k))
        out[f"labeled_loss_p{int(p)}"] = scPoliTrainer.prototype_labeled_loss(fake, latent, protos, labels).numpy()
        ul, counts = scPoliTrainer.prototype_unlabeled_loss(fake, latent, protos_unlab)
        out[f"unlabeled_loss_p{int(p)}"] = ul.numpy()
        out[f"unlabeled_counts_p{int(p)}"] = counts.numpy()
    fake = SimpleNamespace(device=torch.device("cpu"), p_prototype_loss=2.0, model=SimpleNamespace(n_cell_types=k))
    m_all, c_all = scPoliTrainer.update_labeled_prototypes(fake, latent, labels, protos, protos_cov, None)
    m_mask, c_mask = scPoliTrainer.update_labeled_prototypes(fake, latent, labels, protos, protos_cov, [4, 5])
    out.update(
        update_mean_all=m_all.numpy(),
        update_cov_all=c_all.numpy(),
        update_mean_mask45=m_mask.numpy(),
        update_cov_mask45=c_mask.numpy(),
    )
    np.savez(
        out_dir / "prototype_functions.npz",
        latent=latent.numpy(),
        protos=protos.numpy(),
        protos_cov=protos_cov.numpy(),
        protos_unlab=protos_unlab.numpy(),
        labels=labels.numpy(),
        **out,
    )


def trained_golden(adata, out_dir):
    """Short end-to-end run: reference with prototypes, then query surgery with a new cell type."""
    torch.manual_seed(0)
    np.random.seed(0)
    ref = adata[adata.obs.study.isin(["s1", "s2"])].copy()
    qry = adata[adata.obs.study == "s3"].copy()
    # hold out NK from the reference -> novel type in query
    ref = ref[ref.obs.ct_fine != "NK"].copy()
    # make ~20% of reference cells unlabeled
    rng = np.random.default_rng(5)
    labeled = np.where(rng.random(ref.n_obs) > 0.2)[0].tolist()

    model = scPoli(
        adata=ref,
        condition_keys=["study", "tech"],
        cell_type_keys=["ct_fine", "ct_coarse"],
        labeled_indices=labeled,
        latent_dim=8,
        embedding_dims=4,
        recon_loss="nb",
    )
    model.train(n_epochs=12, pretraining_epochs=8, eta=5, use_early_stopping=False, seed=0, n_clusters=None)
    d = out_dir / "trained_ref"
    if d.exists():
        shutil.rmtree(d)
    model.save(str(d), overwrite=True)
    ref.write_h5ad(d / "adata_ref.h5ad")

    model.model.eval()
    res = model.classify(ref)
    res_scaled = model.classify(ref, scale_uncertainties=True)
    res_prob = model.classify(ref, get_prob=True)
    arrays = {"latent_mean": model.get_latent(ref, mean=True)}
    for lvl in ["ct_fine", "ct_coarse"]:
        arrays[f"{lvl}_preds"] = res[lvl]["preds"].astype(str)
        arrays[f"{lvl}_uncert"] = res[lvl]["uncert"]
        arrays[f"{lvl}_wdist"] = res[lvl]["weighted_distances"]
        arrays[f"{lvl}_uncert_scaled"] = res_scaled[lvl]["uncert"]
        arrays[f"{lvl}_prob_preds"] = res_prob[lvl]["preds"].astype(str)
        arrays[f"{lvl}_prob_uncert"] = res_prob[lvl]["uncert"]
    emb = model.get_conditional_embeddings()
    emb = emb if isinstance(emb, dict) else {model.condition_keys_[0]: emb}
    for key, e in emb.items():
        arrays[f"cond_emb_{key}"] = e.X
    np.savez(d / "outputs.npz", **arrays)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "cell_types_": model.cell_types_,
                "conditions_": model.conditions_,
                "conditions_combined_": model.conditions_combined_,
                "cell_type_encoder": model.model.cell_type_encoder,
                "n_unlabeled_prototypes": int(model.prototypes_unlabeled_["mean"].shape[0]),
                "labeled_indices": labeled,
            },
            indent=2,
        )
    )

    # Query: new study s3 (new condition), unlabeled except some NK cells labeled (new cell type)
    torch.manual_seed(0)
    nk_idx = np.where(qry.obs.ct_fine == "NK")[0][:20].tolist()
    q = scPoli.load_query_data(adata=qry, reference_model=model, labeled_indices=nk_idx, return_new_conditions=True)
    qmodel, new_conds = q
    qmodel.train(n_epochs=6, pretraining_epochs=3, eta=10, use_early_stopping=False, seed=0)
    dq = out_dir / "trained_query"
    if dq.exists():
        shutil.rmtree(dq)
    qmodel.save(str(dq), overwrite=True)
    qry.write_h5ad(dq / "adata_query.h5ad")
    qmodel.model.eval()
    qres = qmodel.classify(qry)
    np.savez(
        dq / "outputs.npz",
        latent_mean=qmodel.get_latent(qry, mean=True),
        ref_latent_mean=qmodel.get_latent(ref, mean=True),
        ct_fine_preds=qres["ct_fine"]["preds"].astype(str),
        ct_fine_uncert=qres["ct_fine"]["uncert"],
        ct_coarse_preds=qres["ct_coarse"]["preds"].astype(str),
    )
    (dq / "meta.json").write_text(
        json.dumps(
            {
                "new_conditions": {k: list(v) for k, v in new_conds.items()},
                "cell_types_": qmodel.cell_types_,
                "nk_labeled_idx": nk_idx,
                "requires_grad": {n: p.requires_grad for n, p in qmodel.model.named_parameters()},
            },
            indent=2,
        )
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adata = make_adata()
    adata.write_h5ad(OUT / "adata.h5ad")

    summary = {}
    for name, kw in FORWARD_CONFIGS.items():
        rec = forward_golden(name, kw, adata, OUT / "forward")
        summary[name] = rec["status"]
        print(f"{name:16s} {rec['status']}")
    prototype_function_golden(OUT)
    trained_golden(adata, OUT)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
