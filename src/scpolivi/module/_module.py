"""scPoli module: conditional VAE with per-key condition embeddings and prototypes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, EmbeddingModuleMixin, LossOutput, auto_move_data

from .. import graph as G
from .._constants import GRAPH_GROUP_KEY, GRAPH_PAIR_WEIGHT_KEY, GRAPH_ROLE_KEY
from .._constants import SCPOLI_REGISTRY_KEYS as SK
from ..nn._distributions import BinarizedBernoulli, SquaredError
from ..nn._layers import RECON_LOSSES, ReconLoss, SCPoliDecoder, SCPoliEncoder
from . import _prototypes as P

_PROTO_BUFFERS = (
    "prototypes_labeled_mean",
    "prototypes_labeled_cov",
    "prototypes_fixed",
    "prototypes_unlabeled_mean",
)


class SCPOLIVAE(EmbeddingModuleMixin, BaseModuleClass):
    """scPoli module.

    Parameters
    ----------
    n_input
        Number of features.
    n_conditions_per_key
        Number of categories of each condition key; one embedding table per key.
    n_conditions_combined
        Number of combined conditions (drives the per-condition dispersion).
    n_cell_types
        Number of labeled prototypes (cell types over all hierarchy levels).
    hidden_layer_sizes
        Widths of the encoder hidden layers (the decoder mirrors them).
    n_latent
        Latent dimensionality.
    embedding_dims
        Embedding size per condition key.
    embedding_max_norm
        ``max_norm`` of the embeddings.
    inject_condition
        Where to feed the condition embeddings: any of ``"encoder"``, ``"decoder"``.
    recon_loss
        One of ``nb``, ``zinb``, ``nb_dist`` (alias of ``nb``), ``poisson``, ``mse``, ``bernoulli``.
    dispersion
        ``"gene-batch"`` (scPoli: per combined condition) or ``"gene"``.
    dropout_rate, use_bn, use_ln
        Network options, with scPoli's meaning.
    poisson_library_scaling
        If ``False`` (scArches behavior) the Poisson rate is the softmax output itself; if
        ``True`` it is multiplied by the observed library size.
    n_unlabeled_prototypes
        Initial size of the unlabeled prototype buffer (resized on load).
    """

    def __init__(
        self,
        n_input: int,
        n_conditions_per_key: Sequence[int],
        n_conditions_combined: int,
        n_cell_types: int = 0,
        hidden_layer_sizes: Sequence[int] = (128,),
        n_latent: int = 10,
        embedding_dims: Sequence[int] | int = 10,
        embedding_max_norm: float | None = 1.0,
        inject_condition: Sequence[str] = ("encoder", "decoder"),
        recon_loss: ReconLoss = "nb",
        dispersion: str = "gene-batch",
        dropout_rate: float = 0.05,
        use_bn: bool = False,
        use_ln: bool = True,
        poisson_library_scaling: bool = False,
        n_unlabeled_prototypes: int = 0,
    ):
        super().__init__()
        if recon_loss not in RECON_LOSSES:
            raise ValueError(f"recon_loss must be one of {RECON_LOSSES}.")
        n_conditions_per_key = [int(n) for n in n_conditions_per_key]
        if isinstance(embedding_dims, int):
            embedding_dims = [embedding_dims] * len(n_conditions_per_key)
        if len(embedding_dims) != len(n_conditions_per_key):
            raise ValueError("embedding_dims must have one entry per condition key.")
        bad = set(inject_condition) - {"encoder", "decoder"}
        if bad:
            raise ValueError(f"Unknown inject_condition entries: {bad}.")

        self.n_input = n_input
        self.n_latent = n_latent
        self.n_condition_keys = len(n_conditions_per_key)
        self.n_batch = n_conditions_combined
        self.recon_loss = recon_loss
        self.dispersion = dispersion
        self.inject_condition = tuple(inject_condition)
        self.poisson_library_scaling = poisson_library_scaling
        self.log_variational = recon_loss != "mse"
        self.var_eps = 1e-4

        for i, (n, d) in enumerate(zip(n_conditions_per_key, embedding_dims, strict=True)):
            self.init_embedding(f"cond_{i}", n, embedding_dim=d, max_norm=embedding_max_norm)
        n_cond = int(sum(embedding_dims))
        hidden = [int(h) for h in hidden_layer_sizes]

        self.encoder = SCPoliEncoder(
            n_input, hidden, n_latent, n_cond if "encoder" in self.inject_condition else 0,
            use_bn, use_ln, dropout_rate,
        )
        self.decoder = SCPoliDecoder(
            n_latent, hidden, n_input, n_cond if "decoder" in self.inject_condition else 0,
            recon_loss, use_bn, use_ln, dropout_rate,
        )
        if recon_loss in ("nb", "nb_dist", "zinb"):
            if dispersion == "gene-batch":
                self.px_r = torch.nn.Parameter(torch.randn(n_input, n_conditions_combined))
            elif dispersion == "gene":
                self.px_r = torch.nn.Parameter(torch.randn(n_input))
            else:
                raise ValueError("dispersion must be 'gene-batch' or 'gene'.")
        else:
            self.px_r = None

        # Maps a combined-condition code to the per-key condition codes (for transform_batch).
        self.register_buffer(
            "combined_to_conditions",
            torch.zeros(n_conditions_combined, self.n_condition_keys, dtype=torch.long),
        )
        # Prototypes. NaN rows = not initialized. ``prototypes_fixed`` marks reference
        # prototypes that are not updated during query training.
        self.register_buffer(
            "prototypes_labeled_mean", torch.full((n_cell_types, n_latent), float("nan"))
        )
        self.register_buffer(
            "prototypes_labeled_cov",
            torch.full((n_cell_types, n_latent, n_latent), float("nan")),
        )
        self.register_buffer("prototypes_fixed", torch.zeros(n_cell_types, dtype=torch.bool))
        self.register_buffer(
            "prototypes_unlabeled_mean", torch.zeros(n_unlabeled_prototypes, n_latent)
        )

    # ------------------------------------------------------------- state handling
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Resize prototype buffers to the saved shapes before loading them."""
        for name in _PROTO_BUFFERS:
            key = prefix + name
            if key in state_dict:
                cur = getattr(self, name)
                if cur.shape != state_dict[key].shape:
                    setattr(self, name, torch.empty_like(state_dict[key], device=cur.device))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def n_cell_types(self) -> int:
        return self.prototypes_labeled_mean.shape[0]

    @property
    def labeled_prototypes_initialized(self) -> torch.Tensor:
        return ~torch.isnan(self.prototypes_labeled_mean).any(dim=1)

    # ------------------------------------------------------------- embeddings
    @auto_move_data
    def condition_embedding(self, conditions: torch.Tensor) -> torch.Tensor:
        conditions = conditions.long()
        if conditions.ndim == 1:
            conditions = conditions[:, None]
        return torch.cat(
            [self.compute_embedding(f"cond_{i}", conditions[:, i]) for i in range(self.n_condition_keys)],
            dim=-1,
        )

    # ------------------------------------------------------------- VAE interface
    def _get_inference_input(self, tensors: dict[str, torch.Tensor], **kwargs) -> dict:
        return {
            "x": tensors[REGISTRY_KEYS.X_KEY],
            "conditions": tensors[SK.CONDITIONS_KEY],
        }

    def _get_generative_input(self, tensors, inference_outputs, **kwargs) -> dict:
        return {
            "z": inference_outputs["z"],
            "library": inference_outputs["library"],
            "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
            "conditions": tensors[SK.CONDITIONS_KEY],
        }

    @auto_move_data
    def inference(self, x: torch.Tensor, conditions: torch.Tensor, n_samples: int = 1) -> dict:
        x_ = torch.log1p(x) if self.log_variational else x
        emb = self.condition_embedding(conditions) if "encoder" in self.inject_condition else None
        z_mean, z_log_var = self.encoder(x_, emb)
        qz = Normal(z_mean, torch.sqrt(torch.exp(z_log_var) + self.var_eps))
        z = qz.rsample() if n_samples == 1 else qz.rsample((n_samples,))
        lib_sum = x.sum(1, keepdim=True)
        if self.recon_loss == "mse":  # library unused; data may be non-count
            lib_sum = lib_sum.abs().clamp(min=1e-8)
        library = torch.log(lib_sum)
        if n_samples > 1:
            library = library.unsqueeze(0).expand(n_samples, *library.shape)
        return {"z": z, "qz": qz, "library": library, "z_mean": z_mean, "z_log_var": z_log_var}

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        conditions: torch.Tensor,
        transform_batch: int | None = None,
    ) -> dict:
        batch_index = batch_index.long().reshape(-1)
        if transform_batch is not None:
            batch_index = torch.full_like(batch_index, int(transform_batch))
            conditions = self.combined_to_conditions[batch_index]
        emb = None
        if "decoder" in self.inject_condition:
            emb = self.condition_embedding(conditions)
            if z.ndim == 3:
                emb = emb.unsqueeze(0).expand(z.shape[0], *emb.shape)
        scale, dropout_logits = self._decode(z, emb)
        size_factor = torch.exp(library)

        if self.recon_loss in ("nb", "nb_dist", "zinb"):
            theta = self._theta(batch_index)
            mu = scale * size_factor
            if self.recon_loss == "zinb":
                px = ZeroInflatedNegativeBinomial(
                    mu=mu, theta=theta, zi_logits=dropout_logits, scale=scale
                )
            else:
                px = NegativeBinomial(mu=mu, theta=theta, scale=scale)
        elif self.recon_loss == "poisson":
            rate = scale * size_factor if self.poisson_library_scaling else scale
            px = Poisson(rate=rate, scale=scale)
        elif self.recon_loss == "mse":
            px = SquaredError(scale)
        else:
            px = BinarizedBernoulli(scale)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return {"px": px, "pz": pz}

    def _decode(self, z, emb):
        if z.ndim == 3:  # (n_samples, n_cells, n_latent)
            s, n, d = z.shape
            flat_emb = None if emb is None else emb.reshape(s * n, -1)
            scale, drop = self.decoder(z.reshape(s * n, d), flat_emb)
            scale = scale.reshape(s, n, -1)
            drop = None if drop is None else drop.reshape(s, n, -1)
            return scale, drop
        return self.decoder(z, emb)

    def _theta(self, batch_index: torch.Tensor) -> torch.Tensor:
        if self.dispersion == "gene":
            return torch.exp(self.px_r)
        return torch.exp(self.px_r[:, batch_index].t())

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict,
        generative_outputs: dict,
        kl_weight: float = 1.0,
        prototype_weight: float = 0.0,
        unlabeled_weight: float = 0.0,
        p_prototype_loss: float = 2.0,
        use_prototypes: bool = False,
        graph=None,
        graph_weight: float = 0.0,
        graph_use_mean: bool = False,
    ) -> LossOutput:
        """ELBO plus scPoli's prototype losses (enabled with ``use_prototypes``).

        With ``graph_weight > 0`` the graph Laplacian loss (scANCVI) is added:

        * if the minibatch comes from the neighbour-sampling loader (it carries
          ``scpoli_graph_role``), anchors carry the ELBO and prototype losses and the graph
          loss runs over the sampled (anchor, neighbour) pairs;
        * otherwise ``graph`` (a :class:`~scpolivi.graph.TorchGraph`) is used to find the
          edges between cells of the minibatch.
        """
        x = tensors[REGISTRY_KEYS.X_KEY]
        px = generative_outputs["px"]
        qz = inference_outputs["qz"]
        z = inference_outputs["z"]
        reconst = -px.log_prob(x).sum(-1)
        kl_z = kl(qz, Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))).sum(-1)

        zero = torch.zeros((), device=z.device)
        graph_loss = zero
        z_graph = inference_outputs["z_mean"] if graph_use_mean else z
        if GRAPH_ROLE_KEY in tensors:
            anchor = tensors[GRAPH_ROLE_KEY].reshape(-1) > 0.5
            n_a = int(anchor.sum())
            if graph_weight > 0:
                graph_loss = G.pair_laplacian_loss(
                    z_graph[:n_a],
                    z_graph[n_a:],
                    tensors[GRAPH_PAIR_WEIGHT_KEY][:n_a],
                    tensors[GRAPH_GROUP_KEY][:n_a],
                    n_groups=graph.n_groups if graph is not None else int(tensors[GRAPH_GROUP_KEY].max()) + 1,
                )
            # all other losses on anchors only (layout: [anchors, neighbours])
            tensors = {k: v[:n_a] for k, v in tensors.items()}
            reconst, kl_z, z = reconst[:n_a], kl_z[:n_a], z[:n_a]
        elif graph_weight > 0 and graph is not None:
            graph_loss = G.inbatch_laplacian_loss(
                z_graph, tensors[REGISTRY_KEYS.INDICES_KEY], graph
            )
        cvae_loss = torch.mean(reconst + kl_weight * kl_z)

        labeled_loss, unlabeled_loss = zero, zero
        if use_prototypes:
            if self.prototypes_unlabeled_mean.shape[0] > 0:
                unlabeled_loss, _ = P.unlabeled_prototype_loss(
                    z, self.prototypes_unlabeled_mean, p=p_prototype_loss
                )
            labeled_loss = self._labeled_loss(tensors, z, p_prototype_loss)
        unweighted_proto = labeled_loss + unlabeled_weight * unlabeled_loss
        prototype_loss = prototype_weight * unweighted_proto
        loss = cvae_loss + prototype_loss + graph_weight * graph_loss

        # Metric names follow scArches' logs (val_<name> there, <name>_validation here).
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst,
            kl_local=kl_z,
            extra_metrics={
                "cvae_loss": cvae_loss.detach(),
                "prototype_loss": prototype_loss.detach(),
                "labeled_loss": labeled_loss.detach(),
                "unlabeled_loss": unlabeled_loss.detach(),
                "graph_loss": graph_loss.detach(),
                "unweighted_loss": (
                    reconst.mean() + kl_z.mean() + unweighted_proto
                ).detach(),
            },
        )

    def _labeled_loss(self, tensors, z, p) -> torch.Tensor:
        if self.n_cell_types == 0 or SK.CELL_TYPES_KEY not in tensors:
            return torch.zeros((), device=z.device)
        labeled = tensors[SK.LABELED_KEY].reshape(-1) > 0.5
        if not labeled.any():
            return torch.zeros((), device=z.device)
        labels = tensors[SK.CELL_TYPES_KEY].long()[labeled]
        # ignore classes whose prototype is not initialized (e.g. no labeled cells)
        init = self.labeled_prototypes_initialized
        labels = torch.where(init[labels.clamp(min=0)] & (labels >= 0), labels, -1)
        protos = torch.nan_to_num(self.prototypes_labeled_mean)
        return P.labeled_prototype_loss(z[labeled], protos, labels, p=p)

    @torch.inference_mode()
    def sample(self, tensors, n_samples: int = 1, **kwargs) -> torch.Tensor:
        """Sample from the posterior predictive."""
        _, gen = self.forward(
            tensors, inference_kwargs={"n_samples": n_samples}, compute_loss=False
        )
        return gen["px"].sample().cpu()
