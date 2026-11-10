"""Graph Laplacian loss (from scANCVI)."""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from scpolivi import SCPOLI
from scpolivi.graph import (
    NeighborBatchSampler,
    TorchGraph,
    build_cell_graph,
    inbatch_laplacian_loss,
)

from conftest import make_adata

warnings.filterwarnings("ignore")
FAST = dict(enable_progress_bar=False, accelerator="cpu")
SCANCVI_MODULE = Path(__file__).parents[1] / "external" / "scancvi" / "src" / "scancvi" / "_module.py"


def knn_graphs(adata, key, k=5, seed=0):
    """Random symmetric kNN-like graph per group, scANCVI's dict format."""
    rng = np.random.default_rng(seed)
    out = {}
    for g in adata.obs[key].astype(str).unique():
        n = int((adata.obs[key].astype(str) == g).sum())
        rows = np.repeat(np.arange(n), k)
        cols = rng.integers(0, n, n * k)
        a = sp.coo_matrix((rng.random(n * k) + 0.1, (rows, cols)), shape=(n, n)).tocsr()
        a = a.maximum(a.T)
        a.setdiag(0)
        a.eliminate_zeros()
        out[g] = a
    return out


def test_build_graph_dict_and_global_agree():
    a = make_adata(n_cells=200)
    graphs = knn_graphs(a, "study")
    g1 = build_cell_graph(a, graphs, "study")
    # same graph as one global matrix
    g2 = build_cell_graph(a, g1.adjacency, "study")
    assert (g1.adjacency != g2.adjacency).nnz == 0
    np.testing.assert_array_equal(g1.groups, g2.groups)
    # cross-group edges are dropped
    full = sp.random(200, 200, density=0.05, format="csr", random_state=0)
    g3 = build_cell_graph(a, full, "study")
    r, c = g3.adjacency.nonzero()
    assert (g3.groups[r] == g3.groups[c]).all()
    with pytest.raises(ValueError, match="expected"):
        build_cell_graph(a, {k: v[:-1, :-1] for k, v in graphs.items()}, "study")


def _scancvi_loss(z, batch_index, cells, graphs_list, index_maps):
    """Run scANCVI's own _graph_laplacian_loss on a stub module."""
    spec = importlib.util.spec_from_file_location("scancvi_module", SCANCVI_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stub = mod.SCANCVAE.__new__(mod.SCANCVAE)
    torch.nn.Module.__init__(stub)
    mod.SCANCVAE.set_graph_laplacian(stub, graphs_list, [torch.as_tensor(m) for m in index_maps])
    return mod.SCANCVAE._graph_laplacian_loss(stub, z, batch_index, cells)


@pytest.mark.skipif(not SCANCVI_MODULE.exists(), reason="scancvi sources not available")
def test_inbatch_loss_matches_scancvi():
    a = make_adata(n_cells=300)
    graphs = knn_graphs(a, "study", k=8)
    cg = build_cell_graph(a, graphs, "study")
    tg = TorchGraph(cg)
    rng = np.random.default_rng(1)
    z = torch.randn(300, 6, dtype=torch.float32)
    labels = a.obs["study"].astype(str).to_numpy()
    names = cg.group_names
    batch_index = torch.as_tensor([names.index(v) for v in labels])
    index_maps = []
    for i, g in enumerate(names):
        m = -np.ones(300, dtype=int)
        cells = np.flatnonzero(labels == g)
        m[cells] = np.arange(len(cells))
        index_maps.append(m)
    for _ in range(5):
        cells = torch.as_tensor(rng.choice(300, 120, replace=False))
        ours = inbatch_laplacian_loss(z[cells], cells, tg)
        ref = _scancvi_loss(z[cells], batch_index[cells], cells, [graphs[n] for n in names], index_maps)
        torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)


def test_neighbor_sampler_targets_same_objective():
    """E[pair estimator] equals the full-graph loss (one group)."""
    a = make_adata(n_cells=150)
    rng = np.random.default_rng(0)
    band = sp.csr_matrix(sum(rng.random() * sp.eye(150, k=k) for k in (1, 2, 5)))
    cg = build_cell_graph(a, band + band.T, None)
    z = torch.randn(150, 4)
    full = inbatch_laplacian_loss(z, torch.arange(150), TorchGraph(cg))
    s = NeighborBatchSampler(cg, np.arange(150), batch_size=150, seed=0)
    num = den = 0.0
    for _ in range(400):
        anchors, nb, w, _g = next(iter(s))
        d2 = ((z[anchors] - z[nb]) ** 2).sum(-1).numpy()
        num += (w * d2).sum()
        den += w.sum()
    np.testing.assert_allclose(0.5 * num / den, float(full), rtol=0.03)


@pytest.mark.parametrize("sampling", ["batch", "neighbors"])
def test_train_with_graph(sampling):
    a = make_adata(n_cells=400)
    SCPOLI.setup_anndata(a, condition_keys=["study", "tech"], cell_type_keys="ct_fine")
    m = SCPOLI(a, latent_dim=6, embedding_dims=4)
    with pytest.raises(ValueError, match="set_graph"):
        m.train(n_epochs=2, graph_weight=1.0, **FAST)
    m.set_graph(knn_graphs(a, "study"), graph_key="study")
    m.train(n_epochs=4, pretraining_epochs=2, graph_weight=5.0, graph_sampling=sampling,
            clustering="kmeans", n_clusters=3, **FAST)
    h = m.history
    assert np.isfinite(np.asarray(h["graph_loss_train"], dtype=float)).all()
    gl = np.asarray(h["graph_loss_train"], dtype=float).ravel()
    assert gl[:2].max() == 0 and gl[2:].min() > 0  # off during pretraining
    assert "graph_loss_validation" in h


@pytest.mark.parametrize("use_mean", [False, True])
def test_graph_reduces_neighbor_distance_and_fine_tune(use_mean):
    torch.manual_seed(0)  # weight initialization
    a = make_adata(n_cells=600, n_genes=60)
    SCPOLI.setup_anndata(a, condition_keys="study", cell_type_keys="ct_fine")
    m = SCPOLI(a, latent_dim=6, embedding_dims=4)
    m.train(n_epochs=6, pretraining_epochs=3, clustering="kmeans", n_clusters=3, seed=0, **FAST)
    graphs = knn_graphs(a, "study", k=5)
    m.set_graph(graphs, graph_key="study")
    tg = TorchGraph(m.cell_graph)

    def neighbor_loss():
        z = torch.as_tensor(m.get_latent(mean=True))
        return float(inbatch_laplacian_loss(z, torch.arange(a.n_obs), tg))

    before = neighbor_loss()
    protos = m.module.prototypes_unlabeled_mean.clone()
    m.fine_tune(n_epochs=4, graph_weight=50.0, lr=1e-3, use_early_stopping=False, seed=0,
                graph_use_mean=use_mean, **FAST)
    after = neighbor_loss()
    assert after < 0.85 * before
    # prototypes were kept, not re-clustered
    assert m.module.prototypes_unlabeled_mean.shape == protos.shape


def test_query_with_graph():
    a = make_adata(n_cells=500)
    ref, qry = a[a.obs.study != "s3"].copy(), a[a.obs.study == "s3"].copy()
    SCPOLI.setup_anndata(ref, condition_keys="study", cell_type_keys="ct_fine")
    m = SCPOLI(ref, latent_dim=6, embedding_dims=4)
    m.train(n_epochs=3, pretraining_epochs=2, clustering="kmeans", n_clusters=3, **FAST)
    q = SCPOLI.load_query_data(qry, m, labeled_indices=[])
    q.set_graph(knn_graphs(qry, "study"), graph_key="study")
    q.train(n_epochs=3, pretraining_epochs=1, graph_weight=2.0, **FAST)
    assert np.asarray(q.history["graph_loss_train"], dtype=float).ravel()[-1] > 0
