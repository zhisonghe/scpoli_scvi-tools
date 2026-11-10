"""Forward equivalence with scArches scPoli on imported weights (goldens from the parity env)."""

import json

import anndata as ad
import numpy as np
import pytest
import torch
from scvi import REGISTRY_KEYS

from scpolivi._constants import SCPOLI_REGISTRY_KEYS as SK
from scpolivi.model._scarches_io import (
    convert_state_dict,
    module_kwargs_from_attrs,
    read_scarches_checkpoint,
)
from scpolivi.module import SCPOLIVAE

CONFIGS = [
    "nb_1key", "nb_2keys", "zinb_2keys", "poisson_1key", "mse_1key", "nb_dist_1key",
    "bernoulli_1key", "nb_enc_only", "nb_dec_only", "nb_bn_deep", "nb_ln_deep3",
]


def scarches_tensors(adata, attrs):
    keys = attrs["condition_keys_"]
    conds = np.stack(
        [adata.obs[k].map({c: i for i, c in enumerate(attrs["conditions_"][k])}).values
         for k in keys], axis=1).astype(np.int64)
    if len(keys) > 1:
        comb = adata.obs[keys].astype(str).apply(lambda r: "_".join(r), axis=1)
    else:
        comb = adata.obs[keys[0]].astype(str)
    comb = comb.map({c: i for i, c in enumerate(attrs["conditions_combined_"])}).values
    return {
        REGISTRY_KEYS.X_KEY: torch.tensor(np.asarray(adata.X), dtype=torch.float32),
        SK.CONDITIONS_KEY: torch.tensor(conds),
        REGISTRY_KEYS.BATCH_KEY: torch.tensor(comb.astype(np.int64))[:, None],
    }


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_equivalence(name, golden_dir):
    d = golden_dir / "forward" / name
    rec = json.loads((d / "record.json").read_text())
    gold = np.load(d / "outputs.npz")
    attrs, state, _ = read_scarches_checkpoint(d)
    module = SCPOLIVAE(**module_kwargs_from_attrs(attrs))
    module.load_state_dict(convert_state_dict(state, attrs), strict=True)
    module.eval()

    adata = ad.read_h5ad(golden_dir / "adata.h5ad")
    if attrs["recon_loss_"] == "bernoulli":
        adata.X = (adata.X > 0).astype(np.float32)
    t = scarches_tensors(adata, attrs)
    with torch.no_grad():
        inf = module.inference(t[REGISTRY_KEYS.X_KEY], t[SK.CONDITIONS_KEY])
        np.testing.assert_allclose(inf["z_mean"].numpy(), gold["z_mean"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(inf["z_log_var"].numpy(), gold["z_log_var"], rtol=1e-5, atol=1e-5)
        if "dec_out0" not in gold:
            assert rec["status"] != "ok"  # scArches bernoulli decoder returns None
            return
        emb = module.condition_embedding(t[SK.CONDITIONS_KEY]) if "decoder" in module.inject_condition else None
        scale, drop = module.decoder(inf["z_mean"], emb)
        np.testing.assert_allclose(scale.numpy(), gold["dec_out0"], rtol=1e-5, atol=1e-6)
        if "dec_dropout" in gold:
            np.testing.assert_allclose(drop.numpy(), gold["dec_dropout"], rtol=1e-5, atol=1e-5)
        if "recon_loss" not in gold:
            assert rec["status"].startswith("forward_error")  # scArches nb_dist bug
            return
        inf["z"] = inf["z_mean"]
        gen = module.generative(inf["z"], inf["library"], t[REGISTRY_KEYS.BATCH_KEY], t[SK.CONDITIONS_KEY])
        out = module.loss(t, inf, gen)
        recon = out.reconstruction_loss["reconstruction_loss"].mean().item()
        kl = out.kl_local["kl_local"].mean().item()
    np.testing.assert_allclose(recon, float(gold["recon_loss"]), rtol=1e-5)
    np.testing.assert_allclose(kl, float(gold["kl"]), rtol=1e-5)
