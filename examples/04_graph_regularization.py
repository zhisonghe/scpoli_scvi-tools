"""Walkthrough 4: graph Laplacian regularization (from scANCVI).

Keeps cells that are neighbours within a sample close in the latent space. Two ways to use it:
(a) train from scratch with the graph loss added after pretraining;
(b) fine-tune an already trained model with the graph loss.
Neighbourhood preservation is measured as the overlap between each cell's 15 nearest
neighbours in a per-study PCA and in the latent space.

    python examples/04_graph_regularization.py --data data/pancreas_sparse.h5ad
"""

import argparse

import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from scpolivi import SCPOLI
from scpolivi.graph import construct_sample_graphs

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/pancreas_sparse.h5ad")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--graph-weight", type=float, default=10.0)
args = p.parse_args()

adata = sc.read(args.data)
reference = adata[~adata.obs.study.isin(["celseq", "celseq2"])].copy()

# Per-study kNN graphs on a per-study HVG PCA ({study: adjacency}). X here is log-normalized,
# hence flavor "seurat"; for raw counts use "seurat_v3".
graphs = construct_sample_graphs(reference, "study", recompute_hvg_pca=True, n_hvg=2000, hvg_flavor="seurat")


def neighbour_overlap(z, k=15):
    scores = []
    for s in reference.obs.study.unique():
        idx = np.flatnonzero(reference.obs.study == s)
        g = graphs[s]  # expression neighbours (graph edges) of the study
        lat = NearestNeighbors(n_neighbors=k + 1).fit(z[idx]).kneighbors(return_distance=False)[:, 1:]
        for i in range(len(idx)):
            nb = g.indices[g.indptr[i]:g.indptr[i + 1]]
            if len(nb):
                scores.append(len(set(nb) & set(lat[i])) / len(nb))
    return float(np.mean(scores))


def fresh_model():
    a = reference.copy()
    SCPOLI.setup_anndata(a, condition_keys="study", cell_type_keys="cell_type")
    return SCPOLI(a, embedding_dims=5)


pre = int(0.8 * args.epochs)

# baseline
base = fresh_model()
base.train(n_epochs=args.epochs, pretraining_epochs=pre, eta=5, seed=0)
print("baseline           overlap:", round(neighbour_overlap(base.get_latent(mean=True)), 4))

# (a) from scratch: pretraining as usual, then ELBO + prototype + graph loss
scratch = fresh_model()
scratch.set_graph(graphs, graph_key="study")
scratch.train(n_epochs=args.epochs, pretraining_epochs=pre, eta=5, seed=0,
              graph_weight=args.graph_weight)
print("graph from scratch overlap:", round(neighbour_overlap(scratch.get_latent(mean=True)), 4))

# (b) fine-tune the trained baseline: no pretraining, no KL warm-up, prototypes kept
base.set_graph(graphs, graph_key="study")
base.fine_tune(n_epochs=10, graph_weight=args.graph_weight, eta=5, seed=0)
print("graph fine-tuned   overlap:", round(neighbour_overlap(base.get_latent(mean=True)), 4))

# the graph loss is logged per epoch
print(scratch.history["graph_loss_train"].tail())

# A query model can use a graph of the query cells in the same way:
query = adata[adata.obs.study.isin(["celseq", "celseq2"])].copy()
query_model = SCPOLI.load_query_data(query, reference_model=scratch, labeled_indices=[])
query_model.set_graph(construct_sample_graphs(query_model.adata, "study", recompute_hvg_pca=True,
                                              hvg_flavor="seurat"), graph_key="study")
query_model.train(n_epochs=args.epochs, pretraining_epochs=pre, eta=10, seed=0,
                  graph_weight=args.graph_weight)
