"""Same-config comparison scArches vs scpolivi on the synthetic golden data.

    .venv-parity/bin/python scripts/compare_synthetic.py scarches <n_cells_scale> <epochs> <seed>
    .venv/bin/python        scripts/compare_synthetic.py scpolivi <n_cells_scale> <epochs> <seed>
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import anndata as ad, numpy as np, torch
from sklearn.neighbors import KNeighborsClassifier

impl, scale, epochs, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, "scripts")
from make_golden_synthetic import make_adata  # noqa: E402  (only numpy/pandas/anndata; scarches import guarded below)

a = make_adata(seed=0, n_cells=600 * scale)
ref = a[a.obs.study != "s3"].copy(); ref = ref[ref.obs.ct_fine != "NK"].copy()
qry = a[a.obs.study == "s3"].copy()
torch.manual_seed(seed); np.random.seed(seed)
kw = dict(condition_keys=["study", "tech"], cell_type_keys=["ct_fine", "ct_coarse"])
pre = int(0.9 * epochs)
if impl == "scarches":
    from scarches.models.scpoli import scPoli
    m = scPoli(adata=ref, latent_dim=8, embedding_dims=4, recon_loss="nb", **kw)
    m.train(n_epochs=epochs, pretraining_epochs=pre, eta=5, seed=seed)
    q = scPoli.load_query_data(adata=qry, reference_model=m, labeled_indices=[])
    q.train(n_epochs=epochs // 2, pretraining_epochs=int(0.9 * epochs // 2), eta=10, seed=seed)
    zr, zq = m.get_latent(ref, mean=True), q.get_latent(qry, mean=True)
    pq = q.classify(qry)["ct_fine"]["preds"]
else:
    import scvi
    from scpolivi import SCPOLI
    scvi.settings.seed = seed
    SCPOLI.setup_anndata(ref, **kw)
    m = SCPOLI(ref, latent_dim=8, embedding_dims=4, recon_loss="nb")
    m.train(n_epochs=epochs, pretraining_epochs=pre, eta=5, enable_progress_bar=False, accelerator="cpu")
    q = SCPOLI.load_query_data(qry, m, labeled_indices=[])
    q.train(n_epochs=epochs // 2, pretraining_epochs=int(0.9 * epochs // 2), eta=10, enable_progress_bar=False, accelerator="cpu")
    zr, zq = m.get_latent(ref, mean=True), q.get_latent(qry, mean=True)
    pq = q.classify(qry)["ct_fine"]["preds"]
y = ref.obs.ct_fine.values
knn = KNeighborsClassifier(15).fit(zr[::2], y[::2])
yq = qry.obs.ct_fine.values
known = yq != "NK"
print(json.dumps({"impl": impl, "scale": scale, "epochs": epochs, "seed": seed,
                  "ref_knn_acc": float((knn.predict(zr[1::2]) == y[1::2]).mean()),
                  "query_proto_acc_known": float((pq[known] == yq[known]).mean())}))
