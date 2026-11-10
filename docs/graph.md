# Graph regularization

scpolivi can add the graph Laplacian loss of scANCVI to scPoli. It keeps cells that are
neighbours *within a sample* close together in the latent space:

$$
L_\text{graph} = \frac{1}{|G|}\sum_{g \in G} \frac{1}{2}\,
\frac{\sum_{(i,j)\in E_g} w_{ij}\,\lVert z_i - z_j \rVert^2}{\sum_{(i,j)\in E_g} w_{ij}}
$$

$E_g$ is the edge set of the kNN graph of sample $g$, built from that sample's own expression
data. The total loss after pretraining is
ELBO + `eta` × prototype loss + `graph_weight` × $L_\text{graph}$.

**Why combine it with scPoli.** Prototypes act globally: they align cell types across samples.
The graph acts locally: it keeps the neighbourhoods each sample has on its own. Because edges
never cross samples, the graph term does not directly oppose batch mixing. On the pancreas data,
with the defaults and weight 10, it slightly but consistently improved neighbourhood
preservation and cell-type separation, without changing label transfer or study mixing (see
[benchmarks](benchmarks.md)).

## Usage

```python
from scpolivi.graph import construct_sample_graphs

graphs = construct_sample_graphs(adata, "sample", recompute_hvg_pca=True)   # {sample: adjacency}
model.set_graph(graphs, graph_key="sample")
```

`set_graph` also accepts an `(n_obs, n_obs)` matrix or an `adata.obsp` key; with `graph_key`,
edges between samples are dropped. scANCVI's `construct_batch_adjacency_matrices` output can be
passed directly.

=== "From scratch"

    ```python
    model.train(n_epochs=50, pretraining_epochs=40, eta=5, graph_weight=10)
    ```
    Pretraining is unchanged. The graph loss starts with the prototype loss
    (change with `graph_start_epoch`).

=== "Fine-tuning a trained model"

    ```python
    model.fine_tune(n_epochs=10, graph_weight=10, eta=5)
    ```
    No pretraining, no KL warm-up, prototypes kept, learning rate 1e-4.

=== "Query model"

    ```python
    q = SCPOLI.load_query_data(query, reference_model=model, labeled_indices=[])
    q.set_graph(construct_sample_graphs(q.adata, "sample", recompute_hvg_pca=True), graph_key="sample")
    q.train(n_epochs=50, pretraining_epochs=40, eta=10, graph_weight=10)
    ```
    With the default `freeze=True`, only the query's condition embeddings can respond to the
    graph, so the effect on a query is limited.

## Options

| Option | Default | Meaning |
|---|---|---|
| `graph_weight` | 0 | weight of the loss; 0 disables it |
| `graph_sampling` | `"neighbors"` | how a minibatch sees edges (below) |
| `graph_use_mean` | False | apply the loss to latent means instead of samples |
| `graph_start_epoch` | `pretraining_epochs` | first epoch with the loss |

**`graph_sampling`.** scANCVI only counts edges whose two cells happen to land in the same random
minibatch (`"batch"`). With scPoli's default batch size of 128 and 10⁴ cells, that is about 10–25
edges per step: an unbiased but very noisy estimate. In the default `"neighbors"` mode, each
minibatch cell brings one graph neighbour, drawn in proportion to the edge weight. The
neighbours only enter the graph loss, and pair weights equal the anchor's degree, so both
modes estimate the same objective. On pancreas, `"batch"` mode did not improve neighbourhood
preservation at all. The price is compute: each minibatch carries twice as many cells, so
epochs with the graph loss take about twice as long (pancreas: 2.1 → 4.2 s per epoch on a
laptop CPU). `"batch"` mode adds almost no cost.

**`graph_use_mean`.** By default the loss uses sampled latents, as in scANCVI. For samples,
$\mathbb{E}\lVert z_i - z_j\rVert^2 = \lVert\mu_i-\mu_j\rVert^2 + \operatorname{tr}\Sigma_i + \operatorname{tr}\Sigma_j$.
The pull on the means is therefore the same as with `graph_use_mean=True`, but the sampled
version also shrinks posterior variances. On synthetic data the mean version reduced neighbour
distances more. On pancreas at weight 10, the sampled version preserved neighbourhoods better
(overlap 0.249 vs 0.240, seed 0), so it stays the default.

**Choosing the weight.** The graph loss is a mean squared distance (order 0.01–1), while the
ELBO is summed over genes (order 10³). On pancreas, weight 10 improved local structure with no
cost elsewhere, while weight 100 lowered study mixing (0.46 → 0.39–0.42) and label transfer
(0.975 → 0.965). Start around 10 and check label transfer and batch mixing alongside
neighbourhood preservation.

The graph refers to the rows of the model's AnnData and is **not saved** with the model. Attach
it again after `SCPOLI.load`.
