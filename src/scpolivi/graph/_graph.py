"""Within-sample cell graphs and the graph Laplacian loss (from scANCVI).

The loss keeps cells that are neighbours within a sample close in the latent space:

    L = mean over groups g of  0.5 * sum_{(i,j) in E_g} w_ij ||z_i - z_j||^2 / sum_{(i,j) in E_g} w_ij

where ``E_g`` are the (directed) edges of the graph of group ``g`` (e.g. a sample) and the
sums run over the edges available in a minibatch. Two minibatch estimators are provided:

``"batch"``
    Only edges whose two cells happen to be in the same random minibatch (scANCVI). Cheap,
    but with small minibatches few edges are seen per step.
``"neighbors"``
    Each minibatch cell (anchor) brings one neighbour drawn with probability proportional to
    the edge weight. Anchors carry all other losses, neighbours only the graph loss. Pairs
    are weighted by the anchor's degree, so that both estimators target the same objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import Dataset

from .._constants import GRAPH_GROUP_KEY, GRAPH_PAIR_WEIGHT_KEY, GRAPH_ROLE_KEY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- graph container
@dataclass
class CellGraph:
    """Weighted graph over the rows of the model's AnnData.

    Attributes
    ----------
    adjacency
        ``(n_obs, n_obs)`` CSR matrix; only edges inside a group are kept.
    groups
        ``(n_obs,)`` integer group (sample) of each cell, -1 for cells without a graph.
    group_names
        Name of each group code.
    """

    adjacency: sp.csr_matrix
    groups: np.ndarray
    group_names: list

    @property
    def n_edges(self) -> int:
        return int(self.adjacency.nnz)

    def restrict(self, indices: np.ndarray) -> sp.csr_matrix:
        """Adjacency with edges between cells outside ``indices`` removed (shape unchanged)."""
        keep = np.zeros(self.adjacency.shape[0], dtype=bool)
        keep[indices] = True
        d = sp.diags(keep.astype(np.float32))
        out = (d @ self.adjacency @ d).tocsr()
        out.eliminate_zeros()
        return out


def build_cell_graph(
    adata: AnnData,
    graph: sp.spmatrix | np.ndarray | str | dict,
    graph_key: str | None = None,
) -> CellGraph:
    """Assemble a :class:`CellGraph` from user input.

    Parameters
    ----------
    adata
        The AnnData the model was set up with (row order defines the cells).
    graph
        One of

        * a dict ``{group value: adjacency}`` with one square matrix per value of
          ``adata.obs[graph_key]``, rows ordered as the cells of that group in ``adata``
          (output of :func:`construct_sample_graphs`, or scANCVI's
          ``construct_batch_adjacency_matrices``);
        * an ``(n_obs, n_obs)`` matrix, or an ``adata.obsp`` key. With ``graph_key``, edges
          between different groups are dropped and the loss is normalized per group;
          without it the whole graph is one group.
    graph_key
        ``adata.obs`` column defining the groups (samples). Required for dict input.
    """
    n = adata.n_obs
    if graph_key is not None:
        if graph_key not in adata.obs:
            raise KeyError(f"graph_key {graph_key!r} not found in adata.obs.")
        labels = adata.obs[graph_key].astype(str).to_numpy()
        names = list(pd.unique(labels))
        code = {g: i for i, g in enumerate(names)}
        groups = np.array([code[v] for v in labels], dtype=np.int64)
    else:
        names = ["all"]
        groups = np.zeros(n, dtype=np.int64)

    if isinstance(graph, dict):
        if graph_key is None:
            raise ValueError("A dict of per-group graphs needs `graph_key`.")
        rows, cols, vals = [], [], []
        by_str = {str(k): v for k, v in graph.items()}
        missing = []
        for g, name in enumerate(names):
            cells = np.flatnonzero(groups == g)
            adj = by_str.get(name)
            if adj is None:
                missing.append(name)
                groups[cells] = -1
                continue
            adj = sp.coo_matrix(adj)
            if adj.shape != (len(cells), len(cells)):
                raise ValueError(
                    f"Graph of {graph_key}={name!r} has shape {adj.shape}, expected "
                    f"{(len(cells), len(cells))} (cells of that group in adata order)."
                )
            rows.append(cells[adj.row])
            cols.append(cells[adj.col])
            vals.append(adj.data)
        if missing:
            logger.warning("No graph given for %s=%s; these cells get no graph loss.", graph_key, missing)
        extra = set(by_str) - set(names)
        if extra:
            logger.warning("Graphs for unknown %s values ignored: %s", graph_key, sorted(extra))
        cat = (lambda x, dt: np.concatenate(x).astype(dt) if x else np.zeros(0, dt))
        adj = sp.coo_matrix(
            (cat(vals, np.float32), (cat(rows, np.int64), cat(cols, np.int64))), shape=(n, n)
        )
    else:
        if isinstance(graph, str):
            if graph not in adata.obsp:
                raise KeyError(f"{graph!r} not found in adata.obsp.")
            graph = adata.obsp[graph]
        adj = sp.coo_matrix(graph)
        if adj.shape != (n, n):
            raise ValueError(f"Graph has shape {adj.shape}, expected {(n, n)}.")
        if graph_key is not None:
            same = groups[adj.row] == groups[adj.col]
            if not same.all():
                logger.info("Dropping %d edges between different %s values.", int((~same).sum()), graph_key)
            adj = sp.coo_matrix((adj.data[same], (adj.row[same], adj.col[same])), shape=(n, n))

    adj = adj.tocsr().astype(np.float32)
    adj.sum_duplicates()
    adj.setdiag(0)
    adj.eliminate_zeros()
    if (adj.data < 0).any():
        raise ValueError("Graph weights must be non-negative.")
    return CellGraph(adjacency=adj, groups=groups, group_names=names)


def construct_sample_graphs(
    adata: AnnData,
    sample_key: str,
    embedding_key: str | None = None,
    n_neighbors: int = 15,
    use_diffmap: bool = False,
    mutual_nn: bool = False,
    recompute_hvg_pca: bool = False,
    layer_hvg: str | None = None,
    n_hvg: int = 2000,
    n_pcs: int = 30,
    hvg_flavor: str = "seurat_v3",
) -> dict[str, sp.csr_matrix]:
    """Per-sample kNN connectivities (port of scANCVI ``construct_batch_adjacency_matrices``).

    For every value of ``adata.obs[sample_key]``, builds the scanpy kNN connectivity graph of
    its cells, either on ``adata.obsm[embedding_key]`` or on a PCA of per-sample highly
    variable genes (``recompute_hvg_pca``), optionally through a diffusion map and restricted
    to mutual neighbours. Needs scanpy.

    Returns
    -------
    ``{sample: (n_cells_in_sample, n_cells_in_sample) CSR matrix}``, the input format of
    :meth:`scpolivi.SCPOLI.set_graph` with ``graph_key=sample_key``.
    """
    try:
        import scanpy as sc
    except ImportError as err:  # pragma: no cover
        raise ImportError("construct_sample_graphs needs scanpy (pip install 'scpolivi[leiden]').") from err
    if sample_key not in adata.obs:
        raise KeyError(f"{sample_key!r} not found in adata.obs.")
    if not recompute_hvg_pca and embedding_key is None:
        raise ValueError("embedding_key is required when recompute_hvg_pca=False.")
    labels = adata.obs[sample_key].astype(str).to_numpy()
    out = {}
    for g in pd.unique(labels):
        sub = adata[labels == g].copy()
        if recompute_hvg_pca:
            sc.pp.highly_variable_genes(sub, n_top_genes=n_hvg, flavor=hvg_flavor, layer=layer_hvg)
            sc.pp.pca(sub, n_comps=min(n_pcs, sub.n_obs - 1, sub.n_vars - 1))
            rep = "X_pca"
        else:
            rep = embedding_key
        k = min(n_neighbors, sub.n_obs - 1)
        sc.pp.neighbors(sub, n_neighbors=k, use_rep=rep)
        if use_diffmap:
            sc.tl.diffmap(sub, n_comps=min(n_pcs, sub.n_obs - 2))
            sc.pp.neighbors(sub, n_neighbors=k, use_rep="X_diffmap")
        adj = sp.csr_matrix(sub.obsp["connectivities"])
        if mutual_nn:
            adj = adj.multiply(adj.T > 0).tocsr()
        out[g] = adj
    return out


# ---------------------------------------------------------------------- torch side
class TorchGraph:
    """CSR graph on a device, for minibatch edge lookups.

    Parameters
    ----------
    graph
        Graph built by :func:`build_cell_graph`.
    device
        Torch device.
    """

    def __init__(self, graph: CellGraph, device: torch.device | str = "cpu"):
        a = graph.adjacency
        self.n = a.shape[0]
        self.indptr = torch.as_tensor(a.indptr, dtype=torch.long, device=device)
        self.indices = torch.as_tensor(a.indices, dtype=torch.long, device=device)
        self.data = torch.as_tensor(a.data, dtype=torch.float32, device=device)
        self.groups = torch.as_tensor(graph.groups, dtype=torch.long, device=device)
        self.n_groups = max(len(graph.group_names), 1)

    def to(self, device) -> TorchGraph:
        for k in ("indptr", "indices", "data", "groups"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    @property
    def device(self):
        return self.data.device

    def batch_edges(self, cells: torch.Tensor):
        """Edges between the cells of a minibatch.

        Returns local source/target positions, weights and the group of each edge.
        """
        cells = cells.to(self.device).long()
        starts = self.indptr[cells]
        counts = self.indptr[cells + 1] - starts
        total = int(counts.sum())
        if total == 0:
            e = torch.zeros(0, dtype=torch.long, device=self.device)
            return e, e, torch.zeros(0, device=self.device), e
        src_local = torch.repeat_interleave(torch.arange(len(cells), device=self.device), counts)
        offsets = torch.arange(total, device=self.device) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts
        )
        pos = torch.repeat_interleave(starts, counts) + offsets
        dst = self.indices[pos]
        lookup = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        lookup[cells] = torch.arange(len(cells), device=self.device)
        dst_local = lookup[dst]
        keep = dst_local >= 0
        src_local, dst_local = src_local[keep], dst_local[keep]
        return src_local, dst_local, self.data[pos][keep], self.groups[cells][src_local]


def grouped_laplacian(
    z_src: torch.Tensor,
    z_dst: torch.Tensor,
    weights: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Mean over groups of ``0.5 * sum w d^2 / sum w`` (scANCVI normalization).

    Parameters
    ----------
    z_src, z_dst
        ``(n_edges, latent_dim)`` latent codes of the two ends of each edge.
    weights
        ``(n_edges,)`` edge weights.
    groups
        ``(n_edges,)`` group (sample) code of each edge.
    n_groups
        Number of groups.

    Returns
    -------
    Scalar loss; 0 (with gradient) if there are no edges.
    """
    if weights.numel() == 0:
        return z_src.sum() * 0.0
    d2 = ((z_src - z_dst) ** 2).sum(-1)
    groups = groups.clamp(min=0)
    num = torch.zeros(n_groups, device=d2.device, dtype=d2.dtype).index_add(0, groups, weights * d2)
    den = torch.zeros(n_groups, device=d2.device, dtype=d2.dtype).index_add(0, groups, weights)
    present = den > 0
    if not present.any():
        return z_src.sum() * 0.0
    return (0.5 * num[present] / (den[present] + 1e-8)).mean()


def inbatch_laplacian_loss(z: torch.Tensor, cells: torch.Tensor, graph: TorchGraph) -> torch.Tensor:
    """Graph Laplacian loss over the edges inside a minibatch (scANCVI estimator).

    Parameters
    ----------
    z
        ``(n_cells, latent_dim)`` latent codes of the minibatch.
    cells
        Row positions of the minibatch cells in the model's AnnData.
    graph
        The graph on the same device.

    Returns
    -------
    Scalar loss over the edges whose two cells are both in the minibatch.
    """
    src, dst, w, g = graph.batch_edges(cells.reshape(-1))
    if w.numel() == 0:
        return z.sum() * 0.0
    return grouped_laplacian(z[src], z[dst], w, g, graph.n_groups)


def pair_laplacian_loss(
    z_anchor: torch.Tensor,
    z_neighbor: torch.Tensor,
    pair_weight: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Graph Laplacian loss over sampled (anchor, neighbour) pairs.

    Parameters
    ----------
    z_anchor, z_neighbor
        ``(n_pairs, latent_dim)`` latent codes of anchors and their sampled neighbours.
    pair_weight
        Pair weights (the anchor's degree; 0 for anchors without neighbours).
    groups
        Group (sample) code of each anchor.
    n_groups
        Number of groups.
    """
    pair_weight = pair_weight.reshape(-1)
    keep = pair_weight > 0
    return grouped_laplacian(
        z_anchor[keep], z_neighbor[keep], pair_weight[keep], groups.reshape(-1)[keep].long(), n_groups
    )


# ---------------------------------------------------------------------- neighbour sampling
class NeighborBatchSampler:
    """Yields ``(anchors, neighbours, pair_weights, groups)`` index batches.

    Anchors are a random permutation of ``indices`` cut into minibatches. Each anchor gets one
    neighbour among ``indices`` drawn with probability ``w_ij / deg_i``; the pair weight is
    ``deg_i`` (0 for anchors without neighbours, whose neighbour is the anchor itself).
    With these weights the expected pair loss equals the full-graph loss.

    Parameters
    ----------
    graph
        The cell graph.
    indices
        Cells to sample from (the training split); edges to other cells are ignored.
    batch_size
        Number of anchors per minibatch.
    drop_last
        Drop the last incomplete minibatch.
    seed
        Random seed.
    """

    def __init__(
        self,
        graph: CellGraph,
        indices: np.ndarray,
        batch_size: int,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        adj = graph.restrict(self.indices)
        self.indptr, self.cols = adj.indptr, adj.indices
        self.cum = np.cumsum(adj.data, dtype=np.float64)
        self.degree = np.asarray(adj.sum(1)).ravel()
        self.groups = graph.groups

    def __len__(self) -> int:
        n = len(self.indices)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def sample_neighbors(self, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        deg = self.degree[anchors]
        has = deg > 0
        nb = anchors.copy()
        if has.any():
            a = anchors[has]
            lo, hi = self.indptr[a], self.indptr[a + 1]
            base = np.where(lo > 0, self.cum[np.maximum(lo - 1, 0)], 0.0)
            target = base + self.rng.random(len(a)) * deg[has]
            pos = np.searchsorted(self.cum, target, side="right")
            pos = np.clip(pos, lo, hi - 1)
            nb[has] = self.cols[pos]
        return nb, np.where(has, deg, 0.0).astype(np.float32)

    def __iter__(self):
        perm = self.rng.permutation(self.indices)
        for i in range(len(self)):
            anchors = perm[i * self.batch_size : (i + 1) * self.batch_size]
            nb, w = self.sample_neighbors(anchors)
            yield anchors, nb, w, self.groups[anchors]


class NeighborPairDataset(Dataset):
    """Wraps an AnnTorchDataset: a batch is ``[anchors, neighbours]`` plus graph metadata."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        anchors, nb, w, groups = item
        n = len(anchors)
        out = self.dataset[np.concatenate([anchors, nb])]
        role = np.zeros((2 * n, 1), dtype=np.float32)
        role[:n] = 1.0
        pw = np.zeros((2 * n, 1), dtype=np.float32)
        pw[:n, 0] = w
        gr = np.zeros((2 * n, 1), dtype=np.int64)
        gr[:n, 0] = groups
        out[GRAPH_ROLE_KEY] = role
        out[GRAPH_PAIR_WEIGHT_KEY] = pw
        out[GRAPH_GROUP_KEY] = gr
        return out
