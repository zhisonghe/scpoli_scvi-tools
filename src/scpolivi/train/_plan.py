"""Lightning training plan of scPoli (port of ``scarches.trainers.scpoli.scPoliTrainer``)."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch

from scvi.dataloaders import AnnDataLoader
from scvi.train import TrainingPlan

from .._constants import SCPOLI_REGISTRY_KEYS as SK
from ..module import _prototypes as P

logger = logging.getLogger(__name__)


class SCPoliTrainingPlan(TrainingPlan):
    """Training plan with scPoli's pretraining phase and prototype updates.

    Epoch ``e`` proceeds as in scArches:

    1. if ``e == pretraining_epochs``: initialize labeled prototypes (class means of the
       sampled latent of labeled training cells; in query mode only non-fixed classes) and
       unlabeled prototypes (Leiden or KMeans centroids of all training cells);
    2. minibatch steps with loss ``ELBO + eta * (labeled + unlabeled_weight * unlabeled)``,
       the prototype part only for ``e >= pretraining_epochs``;
    3. before validation: recompute non-fixed labeled prototypes from the training latent
       and take one Adam step on the unlabeled prototypes.

    Parameters
    ----------
    pretraining_epochs
        Number of epochs trained without prototype losses.
    eta
        Weight of the prototype loss.
    unlabeled_weight
        Weight of the unlabeled prototype loss within the prototype loss.
    p_prototype_loss
        Norm of the distances used by the prototype losses.
    prototype_training, unlabeled_prototype_training
        Switch off all prototypes / only the unlabeled ones.
    clustering, clustering_res, n_clusters
        Initialization of unlabeled prototypes: ``"leiden"`` (resolution ``clustering_res``)
        or ``"kmeans"`` with ``n_clusters`` (Leiden if ``n_clusters`` is None).
    any_unlabeled_data
        Whether unlabeled prototypes should be created (unlabeled training cells exist or
        unknown cell-type names were given).
    latent_batch_size
        Minibatch size of the full-data latent passes.
    embedding_weight_decay
        Weight decay of the condition embeddings (0 in scPoli).
    """

    def __init__(
        self,
        module,
        *,
        pretraining_epochs: int,
        eta: float = 1.0,
        unlabeled_weight: float = 0.0,
        p_prototype_loss: float = 2.0,
        prototype_training: bool = True,
        unlabeled_prototype_training: bool = True,
        clustering: Literal["leiden", "kmeans"] = "leiden",
        clustering_res: float = 2.0,
        n_clusters: int | None = None,
        any_unlabeled_data: bool = True,
        latent_batch_size: int = 1024,
        embedding_weight_decay: float = 0.0,
        lr: float = 1e-3,
        eps: float = 0.01,
        weight_decay: float = 0.04,
        n_epochs_kl_warmup: int | None = 100,
        **kwargs,
    ):
        super().__init__(
            module,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            **kwargs,
        )
        self.pretraining_epochs = int(pretraining_epochs)
        self.eta = eta
        self.unlabeled_weight = unlabeled_weight
        self.p_prototype_loss = p_prototype_loss
        self.prototype_training = prototype_training
        self.unlabeled_prototype_training = unlabeled_prototype_training
        self.clustering = clustering
        self.clustering_res = clustering_res
        self.n_clusters = n_clusters
        self.any_unlabeled_data = any_unlabeled_data
        self.latent_batch_size = latent_batch_size
        self.embedding_weight_decay = embedding_weight_decay
        self.prototype_optimizer: torch.optim.Optimizer | None = None
        self._unlabeled_param: torch.nn.Parameter | None = None
        self._updated_epoch = -1

    # ------------------------------------------------------------------ optimizers
    def configure_optimizers(self):
        """Adam with two groups: embeddings (no weight decay) and the rest (scArches)."""
        emb, rest = [], []
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                (emb if "_embeddings_dict" in name else rest).append(p)
        groups = []
        if emb:
            groups.append({"params": emb, "weight_decay": self.embedding_weight_decay})
        if rest:
            groups.append({"params": rest})
        optimizer = torch.optim.Adam(groups, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.lr_patience,
                    factor=self.lr_factor,
                    threshold=self.lr_threshold,
                    min_lr=self.lr_min,
                    threshold_mode="abs",
                ),
                "monitor": self.lr_scheduler_metric,
            }
        return config

    # ------------------------------------------------------------------ phases
    @property
    def prototypes_active(self) -> bool:
        return self.prototype_training and self.current_epoch >= self.pretraining_epochs

    def _set_loss_kwargs(self):
        self.loss_kwargs.update(
            {
                "use_prototypes": self.prototypes_active,
                "prototype_weight": self.eta,
                "unlabeled_weight": self.unlabeled_weight,
                "p_prototype_loss": self.p_prototype_loss,
            }
        )

    def training_step(self, batch, batch_idx):
        self._set_loss_kwargs()
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._set_loss_kwargs()
        return super().validation_step(batch, batch_idx)

    def on_train_epoch_start(self):
        if self.prototype_training and self.current_epoch == self.pretraining_epochs:
            self.initialize_prototypes()

    def on_validation_epoch_start(self):
        if not self.trainer.sanity_checking:
            self._end_of_epoch_update()

    def on_train_epoch_end(self):
        self._end_of_epoch_update()  # no-op if already done before validation

    # ------------------------------------------------------------------ data access
    @property
    def _datamodule(self):
        return self.trainer.datamodule

    def _registry_array(self, key: str, indices: np.ndarray) -> torch.Tensor:
        arr = np.asarray(self._datamodule.adata_manager.get_from_registry(key))
        if arr.ndim == 1:
            arr = arr[:, None]
        return torch.as_tensor(arr[indices], device=self.device)

    @torch.no_grad()
    def train_latent(self) -> torch.Tensor:
        """Sampled latent codes of all training cells (scArches ``get_latent_train``)."""
        dm = self._datamodule
        loader = AnnDataLoader(
            dm.adata_manager,
            indices=dm.train_idx,
            batch_size=self.latent_batch_size,
            shuffle=False,
            load_sparse_tensor=getattr(dm, "load_sparse_tensor", False),
        )
        out = []
        for tensors in loader:
            inf = self.module.inference(
                **self.module._get_inference_input(tensors)
            )
            out.append(inf["z"])
        return torch.cat(out).to(self.device)

    def _train_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._datamodule.train_idx
        labels = self._registry_array(SK.CELL_TYPES_KEY, idx).long()
        labeled = self._registry_array(SK.LABELED_KEY, idx).reshape(-1) > 0.5
        return labels, labeled

    # ------------------------------------------------------------------ prototypes
    @torch.no_grad()
    def initialize_prototypes(self):
        """Initialize labeled and unlabeled prototypes at the end of pretraining."""
        module = self.module
        latent = self.train_latent()
        n_ct = module.n_cell_types
        if n_ct > 0:
            labels, labeled = self._train_labels()
            if labeled.any():
                fixed = module.prototypes_fixed.to(self.device)
                prev_mean = module.prototypes_labeled_mean.to(self.device)
                prev_cov = module.prototypes_labeled_cov.to(self.device)
                mean, cv = P.update_labeled_prototypes(
                    latent[labeled], labels[labeled], prev_mean, prev_cov, n_ct, update_mask=~fixed
                )
                module.prototypes_labeled_mean = mean
                module.prototypes_labeled_cov = cv
        if self.any_unlabeled_data and self.unlabeled_prototype_training:
            centers = cluster_latent(
                latent.detach().cpu().numpy(), self.clustering, self.clustering_res, self.n_clusters
            )
            module.prototypes_unlabeled_mean = torch.as_tensor(
                centers, dtype=latent.dtype, device=self.device
            )
            self._unlabeled_param = torch.nn.Parameter(module.prototypes_unlabeled_mean.clone())
            self.prototype_optimizer = torch.optim.Adam(
                [self._unlabeled_param], lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
            )
            logger.info("Initialized %d unlabeled prototypes.", centers.shape[0])

    def _end_of_epoch_update(self):
        if self._updated_epoch == self.current_epoch or not self.prototypes_active:
            return
        self._updated_epoch = self.current_epoch
        module = self.module
        was_training = module.training
        module.eval()
        latent = self.train_latent()
        if module.n_cell_types > 0:
            labels, labeled = self._train_labels()
            if labeled.any():
                mean, cv = P.update_labeled_prototypes(
                    latent[labeled],
                    labels[labeled],
                    module.prototypes_labeled_mean,
                    module.prototypes_labeled_cov,
                    module.n_cell_types,
                    update_mask=~module.prototypes_fixed,
                )
                module.prototypes_labeled_mean = mean
                module.prototypes_labeled_cov = cv
        if self.prototype_optimizer is not None:
            with torch.enable_grad():
                self.prototype_optimizer.zero_grad()
                loss, _ = P.unlabeled_prototype_loss(
                    latent.detach(), self._unlabeled_param, p=self.p_prototype_loss
                )
                loss.backward()
                self.prototype_optimizer.step()
            module.prototypes_unlabeled_mean = self._unlabeled_param.detach().clone()
        if was_training:
            module.train()


def cluster_latent(
    latent: np.ndarray, clustering: str, resolution: float, n_clusters: int | None
) -> np.ndarray:
    """Cluster centers used to initialize unlabeled prototypes."""
    if clustering == "kmeans" and n_clusters is not None:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=n_clusters, n_init="auto").fit(latent).cluster_centers_
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Leiden initialization of unlabeled prototypes needs scanpy and leidenalg/igraph "
            "(pip install 'scpolivi[leiden]'), or use clustering='kmeans' with n_clusters."
        ) from err
    a = ad.AnnData(latent)
    sc.pp.neighbors(a, use_rep="X")
    sc.tl.leiden(a, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)
    groups = a.obs["leiden"].astype(int).to_numpy()
    return np.stack([latent[groups == g].mean(0) for g in np.unique(groups)])
