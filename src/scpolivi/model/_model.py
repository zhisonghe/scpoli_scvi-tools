"""SCPOLI model class: scPoli (De Donno et al., 2023) on scvi-tools."""

from __future__ import annotations

import contextvars
import logging
import math
import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import _SETUP_ARGS_KEY
from scvi.model._utils import parse_device_args
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.model.base import ArchesMixin, BaseMinifiedModeModelClass, RNASeqMixin, VAEMixin
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp

from .._constants import COMBINED_CONDITIONS_COL, LABELED_COL, SCPOLI_REGISTRY_KEYS as SK
from ..data import as_list, build_fields, write_helper_columns
from ..module import SCPOLIVAE
from ..module import _prototypes as P
from ..train._callbacks import SCPoliEarlyStopping
from ..train._plan import SCPoliTrainingPlan
from ..train._splitter import SCPoliDataSplitter

logger = logging.getLogger(__name__)

# Per-call overrides used while ArchesMixin re-runs setup_anndata / __init__ for a query.
_QUERY_OVERRIDES: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "scpolivi_query_overrides", default=None
)


class _RegistryAttr:
    """Read-only scArches-style attribute derived from the registry; ``setattr`` is ignored.

    scvi-tools saves every attribute ending in ``_`` and sets it back on load.
    """

    def __init__(self, fget):
        self.fget = fget
        self.__doc__ = fget.__doc__

    def __get__(self, obj, owner=None):
        return self if obj is None else self.fget(obj)

    def __set__(self, obj, value):
        pass


class SCPOLI(ArchesMixin, RNASeqMixin, VAEMixin, BaseMinifiedModeModelClass):
    """scPoli: population-level integration with condition embeddings and prototypes.

    A conditional VAE in which every condition key (e.g. study, sample, technology) gets a
    learned embedding, combined with cell-type prototypes in the latent space. Labeled
    prototypes are class centroids used for label transfer and uncertainty; unlabeled
    prototypes (cluster centroids) structure unlabeled cells. Reference models are extended to
    query data with :meth:`load_query_data` (scArches surgery), which only learns embeddings
    for the new conditions.

    Typical workflow: :meth:`setup_anndata` → ``SCPOLI(adata)`` → :meth:`train` →
    :meth:`get_latent` / :meth:`classify` → :meth:`load_query_data` on new data.

    Parameters
    ----------
    adata
        AnnData registered with :meth:`setup_anndata`.
    hidden_layer_sizes
        Encoder hidden layer widths (decoder mirrored). Default ``[ceil(sqrt(n_vars))]``.
    latent_dim
        Latent dimensionality.
    embedding_dims
        Condition embedding size, one int for all keys or one per key.
    embedding_max_norm
        Max norm of the condition embeddings.
    inject_condition
        Subset of ``("encoder", "decoder")`` receiving the condition embeddings.
    recon_loss
        ``"nb"``, ``"zinb"``, ``"nb_dist"`` (alias of ``nb``), ``"poisson"``, ``"mse"`` or ``"bernoulli"``.
    dr_rate
        Dropout rate.
    use_bn, use_ln
        Batch / layer normalization (scPoli default: layer norm).
    dispersion
        ``"gene-batch"`` (scPoli: per combined condition) or ``"gene"``.
    poisson_library_scaling
        Scale the Poisson rate by library size (scArches does not).
    share_metadata
        Store the first ``obs`` row per condition as condition metadata (``obs_metadata_``).
    obs_metadata
        Explicit condition metadata, overrides ``share_metadata``.
    **module_kwargs
        Passed to :class:`~scpolivi.module.SCPOLIVAE`.

    Examples
    --------
    >>> SCPOLI.setup_anndata(adata, condition_keys="study", cell_type_keys="cell_type")
    >>> model = SCPOLI(adata, embedding_dims=5)
    >>> model.train(n_epochs=50, pretraining_epochs=40, eta=5)
    >>> adata.obsm["X_scpoli"] = model.get_latent(mean=True)
    """

    _LATENT_QZM_KEY = "scpoli_latent_qzm"
    _LATENT_QZV_KEY = "scpoli_latent_qzv"
    _OBSERVED_LIB_SIZE_KEY = "scpoli_observed_lib_size"
    _module_cls = SCPOLIVAE
    _data_splitter_cls = SCPoliDataSplitter
    _training_plan_cls = SCPoliTrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData | None = None,
        hidden_layer_sizes: Sequence[int] | None = None,
        latent_dim: int = 10,
        embedding_dims: int | Sequence[int] = 10,
        embedding_max_norm: float | None = 1.0,
        inject_condition: Sequence[str] = ("encoder", "decoder"),
        recon_loss: Literal["nb", "zinb", "nb_dist", "poisson", "mse", "bernoulli"] = "nb",
        dr_rate: float = 0.05,
        use_bn: bool = False,
        use_ln: bool = True,
        dispersion: Literal["gene-batch", "gene"] = "gene-batch",
        poisson_library_scaling: bool = False,
        share_metadata: bool = True,
        obs_metadata: pd.DataFrame | None = None,
        **module_kwargs,
    ):
        super().__init__(adata)
        if recon_loss == "nb_dist":
            warnings.warn(
                "recon_loss='nb_dist' is treated as 'nb' (scArches' nb_dist path is broken).",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        overrides = _QUERY_OVERRIDES.get() or {}
        if "dr_rate" in overrides:
            dr_rate = overrides["dr_rate"]
        n_unlab = overrides.get("n_unlabeled_prototypes", module_kwargs.pop("n_unlabeled_prototypes", 0))

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [int(math.ceil(math.sqrt(self.summary_stats.n_vars)))]
        cond_state = self.adata_manager.get_state_registry(SK.CONDITIONS_KEY)
        n_per_key = list(cond_state["n_cats_per_key"])

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_conditions_per_key=n_per_key,
            n_conditions_combined=self.summary_stats.n_batch,
            n_cell_types=self.summary_stats[f"n_{SK.CELL_TYPES_KEY}"],
            hidden_layer_sizes=list(hidden_layer_sizes),
            n_latent=latent_dim,
            embedding_dims=embedding_dims,
            embedding_max_norm=embedding_max_norm,
            inject_condition=list(inject_condition),
            recon_loss=recon_loss,
            dispersion=dispersion,
            dropout_rate=dr_rate,
            use_bn=use_bn,
            use_ln=use_ln,
            poisson_library_scaling=poisson_library_scaling,
            n_unlabeled_prototypes=n_unlab,
            **module_kwargs,
        )
        self.module.combined_to_conditions = self._combined_to_conditions()
        self.module.minified_data_type = self.minified_data_type

        if obs_metadata is not None:
            self.obs_metadata_ = obs_metadata
        elif share_metadata:
            self.obs_metadata_ = self._condition_metadata(self.adata)
        else:
            self.obs_metadata_ = None

        self._model_summary_string = (
            f"SCPOLI model: conditions {self.condition_keys_} {n_per_key}, "
            f"cell types {self.cell_type_keys_} ({self.module.n_cell_types} prototypes), "
            f"hidden {list(hidden_layer_sizes)}, latent {latent_dim}, recon_loss {recon_loss}"
        )
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------ setup
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        condition_keys: str | Sequence[str],
        cell_type_keys: str | Sequence[str] | None = None,
        unknown_ct_names: Sequence[str] | None = None,
        labeled_indices: Sequence[int] | np.ndarray | None = None,
        layer: str | None = None,
        **kwargs,
    ):
        """Register an AnnData object for SCPOLI.

        Parameters
        ----------
        adata
            AnnData with raw counts in ``.X`` or ``layer`` (any real values for ``mse``).
        condition_keys
            ``obs`` column(s) with the conditions to embed (e.g. study, sample, technology).
        cell_type_keys
            ``obs`` column(s) with cell-type annotations; several columns form a hierarchy
            sharing one prototype namespace.
        unknown_ct_names
            Cell-type names to ignore for prototypes (e.g. ``["Unknown"]``).
        labeled_indices
            Positional indices of labeled cells (default: all cells). Not stored in the
            registry, so query data does not inherit it.
        layer
            ``adata.layers`` key to use instead of ``.X``.
        """
        condition_keys = as_list(condition_keys)
        cell_type_keys = as_list(cell_type_keys)
        if not condition_keys:
            raise ValueError("At least one condition key is required.")
        overrides = _QUERY_OVERRIDES.get() or {}
        is_transfer = kwargs.get("source_registry") is not None
        if is_transfer and "unknown_ct_names" in overrides:
            unknown_ct_names = overrides["unknown_ct_names"]
        unknown_ct_names = list(unknown_ct_names) if unknown_ct_names else None

        setup_method_args = cls._get_setup_method_args(**locals())
        setup_method_args[_SETUP_ARGS_KEY].pop("labeled_indices", None)
        write_helper_columns(
            adata,
            condition_keys,
            labeled_indices,
            keep_existing_labeled=is_transfer or overrides.get("keep_labeled", False),
            preserve_categories=overrides.get("preserve_categories", False),
        )
        fields = build_fields(layer, condition_keys, cell_type_keys, unknown_ct_names)
        minify_type = _get_adata_minify_type(adata)
        if minify_type is not None:
            fields += cls._get_fields_for_adata_minification(minify_type)
        adata_manager = AnnDataManager(fields=fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------ registry views
    @_RegistryAttr
    def condition_keys_(self) -> list[str]:
        """Condition keys."""
        return list(self.adata_manager.registry[_SETUP_ARGS_KEY]["condition_keys"])

    @_RegistryAttr
    def cell_type_keys_(self) -> list[str] | None:
        """Cell-type keys (hierarchy levels), or None."""
        keys = self.adata_manager.registry[_SETUP_ARGS_KEY]["cell_type_keys"]
        return list(keys) if keys else None

    @_RegistryAttr
    def unknown_ct_names_(self) -> list[str] | None:
        """Cell-type names ignored for prototypes."""
        st = self.adata_manager.get_state_registry(SK.CELL_TYPES_KEY)
        names = list(st.get("unknown_ct_names", []))
        return names or None

    @_RegistryAttr
    def conditions_(self) -> dict[str, list[str]]:
        """Categories of each condition key, in embedding order."""
        st = self.adata_manager.get_state_registry(SK.CONDITIONS_KEY)
        return {k: [str(c) for c in v] for k, v in st["mappings"].items()}

    @_RegistryAttr
    def conditions_combined_(self) -> list[str]:
        """Combined conditions, in dispersion order."""
        st = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        return [str(c) for c in st["categorical_mapping"]]

    @_RegistryAttr
    def cell_types_(self) -> dict[str, list[str]]:
        """Cell types (prototype order) and the hierarchy levels they belong to."""
        st = self.adata_manager.registry["field_registries"][SK.CELL_TYPES_KEY]["state_registry"]
        cts = st.get("cell_types", {})
        # the registry's frozen views sort dict keys; "mapping" carries the prototype order
        return {k: list(cts[k]) for k in st.get("mapping", [])}

    @_RegistryAttr
    def prototypes_labeled_(self) -> dict[str, torch.Tensor]:
        """Labeled prototypes ``{"mean", "cov"}`` (NaN rows: not initialized)."""
        return {
            "mean": self.module.prototypes_labeled_mean.detach().cpu(),
            "cov": self.module.prototypes_labeled_cov.detach().cpu(),
        }

    @_RegistryAttr
    def prototypes_unlabeled_(self) -> dict[str, torch.Tensor | None]:
        """Unlabeled prototypes ``{"mean"}``."""
        m = self.module.prototypes_unlabeled_mean.detach().cpu()
        return {"mean": m if m.shape[0] > 0 else None}

    @_RegistryAttr
    def labeled_indices_(self) -> np.ndarray:
        """Positional indices of labeled cells of the registered AnnData."""
        flag = np.asarray(self.adata_manager.get_from_registry(SK.LABELED_KEY)).ravel()
        return np.where(flag > 0.5)[0]

    def _combined_to_conditions(self) -> torch.Tensor:
        comb = np.asarray(self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel()
        conds = np.asarray(self.adata_manager.get_from_registry(SK.CONDITIONS_KEY))
        table = np.zeros((self.summary_stats.n_batch, conds.shape[1]), dtype=np.int64)
        _, first = np.unique(comb, return_index=True)
        for i in first:
            table[int(comb[i])] = conds[i]
        return torch.as_tensor(table)

    def _condition_metadata(self, adata: AnnData) -> pd.DataFrame:
        keys = self.condition_keys_
        obs = adata.obs.drop(columns=[c for c in adata.obs if c.startswith("_scvi")], errors="ignore")
        return obs.groupby(keys if len(keys) > 1 else keys[0], observed=True).first()

    def _level_classes(self, level: str) -> np.ndarray:
        return np.array([i for i, lv in enumerate(self.cell_types_.values()) if level in lv], dtype=int)

    # ------------------------------------------------------------------ training
    def train(
        self,
        n_epochs: int = 100,
        pretraining_epochs: int | None = None,
        eta: float = 1.0,
        lr: float = 1e-3,
        eps: float = 0.01,
        alpha_epoch_anneal: int | None = 100,
        alpha_iter_anneal: int | None = None,
        alpha_kl: float = 1.0,
        reload_best: bool = False,
        prototype_training: bool = True,
        unlabeled_prototype_training: bool = True,
        use_early_stopping: bool = True,
        early_stopping_kwargs: dict | None = None,
        batch_size: int = 128,
        train_frac: float = 0.9,
        use_stratified_sampling: bool = False,
        n_samples: int | None = None,
        weight_decay: float = 0.04,
        clip_value: float = 0.0,
        clustering: Literal["leiden", "kmeans"] = "leiden",
        clustering_res: float = 2.0,
        n_clusters: int | None = None,
        unlabeled_weight: float = 0.0,
        p_prototype_loss: float = 2.0,
        graph_weight: float = 0.0,
        graph_sampling: Literal["batch", "neighbors"] = "neighbors",
        graph_start_epoch: int | None = None,
        graph_use_mean: bool = False,
        seed: int | None = None,
        n_workers: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        plan_kwargs: dict | None = None,
        datasplitter_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """Train the model (arguments as in scArches ``scPoli.train`` / ``scPoliTrainer``).

        Parameters
        ----------
        n_epochs
            Number of epochs.
        pretraining_epochs
            Epochs without prototype losses (default ``floor(0.9 * n_epochs)``; all epochs
            when no cell-type keys are registered).
        eta
            Prototype loss weight.
        lr, eps, weight_decay
            Adam settings (embeddings get no weight decay).
        alpha_epoch_anneal, alpha_iter_anneal, alpha_kl
            KL warm-up over epochs or steps up to ``alpha_kl``.
        reload_best
            Restore the best weights and prototypes (early stopping metric) at the end.
        prototype_training, unlabeled_prototype_training
            Switch off all / the unlabeled prototypes.
        use_early_stopping, early_stopping_kwargs
            scArches early stopping (``early_stopping_metric`` may use scArches names such as
            ``"val_prototype_loss"``).
        batch_size, train_frac, use_stratified_sampling, n_samples
            Minibatching and the stratified train/validation split.
        clip_value
            Gradient value clipping (0 disables).
        clustering, clustering_res, n_clusters
            Initialization of unlabeled prototypes.
        unlabeled_weight, p_prototype_loss
            Unlabeled prototype loss weight and distance norm.
        graph_weight
            Weight of the graph Laplacian loss (scANCVI) on the graph attached with
            :meth:`set_graph`; 0 disables it.
        graph_sampling
            ``"neighbors"``: each training cell brings one sampled graph neighbour into its
            minibatch (every cell gets a graph term at every step); ``"batch"``: only edges
            between cells that share a random minibatch (scANCVI; few edges per step unless
            minibatches are large). Validation always uses ``"batch"``.
        graph_start_epoch
            First epoch with the graph loss (default: with the prototype losses, after
            ``pretraining_epochs``).
        graph_use_mean
            Apply the graph loss to the latent means instead of sampled latents (default
            ``False``, as scANCVI). Both pull neighbouring means together with the same
            expected gradient; with samples the loss also shrinks posterior variances.
        seed
            If given, sets ``scvi.settings.seed`` (scArches reseeds with 2020 by default).
        n_workers
            Dataloader worker processes.
        accelerator, devices
            Lightning accelerator (``"cpu"``, ``"gpu"``, ``"mps"``, ``"auto"``) and devices.
        plan_kwargs, datasplitter_kwargs, trainer_kwargs
            Extra arguments for the training plan, data splitter and Lightning trainer.
        """
        if self.minified_data_type is not None:
            raise ValueError(
                "This model's AnnData is minified (no expression data); training needs the full "
                "data. Load the model with the full AnnData instead."
            )
        if seed is not None:
            settings.seed = seed
        graph = self.cell_graph if graph_weight and graph_weight > 0 else None
        if graph_weight and graph_weight > 0 and graph is None:
            raise ValueError("graph_weight > 0 needs a graph: call `model.set_graph(...)` first.")
        if graph_start_epoch is None:
            graph_start_epoch = (
                pretraining_epochs if pretraining_epochs is not None else int(np.floor(n_epochs * 0.9))
            )
        if self.cell_type_keys_ is None or self.module.n_cell_types == 0:
            if prototype_training and self.cell_type_keys_ is None:
                logger.info("No cell type keys: training without prototypes.")
            pretraining_epochs = n_epochs
            prototype_training = False
        elif pretraining_epochs is None:
            pretraining_epochs = int(np.floor(n_epochs * 0.9))

        flag = np.asarray(self.adata_manager.get_from_registry(SK.LABELED_KEY)).ravel()
        any_unlabeled = bool((flag < 0.5).any()) or self.unknown_ct_names_ is not None

        plan = {
            "pretraining_epochs": pretraining_epochs,
            "eta": eta,
            "unlabeled_weight": unlabeled_weight,
            "p_prototype_loss": p_prototype_loss,
            "prototype_training": prototype_training,
            "unlabeled_prototype_training": unlabeled_prototype_training,
            "clustering": clustering,
            "clustering_res": clustering_res,
            "n_clusters": n_clusters,
            "any_unlabeled_data": any_unlabeled,
            "lr": lr,
            "eps": eps,
            "weight_decay": weight_decay,
            "n_epochs_kl_warmup": alpha_epoch_anneal,
            "n_steps_kl_warmup": alpha_iter_anneal if alpha_epoch_anneal is None else None,
            "max_kl_weight": alpha_kl,
            "graph": graph,
            "graph_weight": graph_weight,
            "graph_start_epoch": graph_start_epoch,
            "graph_use_mean": graph_use_mean,
        }
        plan.update(plan_kwargs or {})
        training_plan = self._training_plan_cls(self.module, **plan)

        ds_kwargs = {
            "train_frac": train_frac,
            "use_stratified_sampling": use_stratified_sampling,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "graph": graph,
            "graph_sampling": graph_sampling,
        }
        if n_workers is not None:
            ds_kwargs["num_workers"] = n_workers
        ds_kwargs.update(datasplitter_kwargs or {})
        data_splitter = self._data_splitter_cls(self.adata_manager, **ds_kwargs)

        callbacks = list(trainer_kwargs.pop("callbacks", []))
        if use_early_stopping and train_frac < 1:
            es = {k: v for k, v in (early_stopping_kwargs or {}).items()}
            callbacks.append(
                SCPoliEarlyStopping(pretraining_epochs=pretraining_epochs, reload_best=reload_best, **es)
            )
        elif use_early_stopping:
            logger.info("Early stopping disabled: no validation set (train_frac=1).")
        if clip_value and clip_value > 0:
            trainer_kwargs.setdefault("gradient_clip_val", clip_value)
            trainer_kwargs.setdefault("gradient_clip_algorithm", "value")
        if train_frac < 1:
            trainer_kwargs.setdefault("check_val_every_n_epoch", 1)
        trainer_kwargs["callbacks"] = callbacks

        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=n_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=False,
            **trainer_kwargs,
        )
        self.module.train()  # Lightning keeps the eval mode left by a previous run
        out = runner()
        self.module.eval()
        return out

    def fine_tune(
        self,
        n_epochs: int = 20,
        graph_weight: float = 1.0,
        eta: float = 1.0,
        lr: float = 1e-4,
        reinitialize_prototypes: bool = False,
        **train_kwargs,
    ):
        """Continue training a trained model with the graph Laplacian loss.

        Runs :meth:`train` without pretraining phase and without KL warm-up, keeping the
        current prototypes (labeled prototypes are recomputed every epoch as usual), so the
        model starts from its trained state with the loss ``ELBO + eta * prototype + graph``.

        Parameters
        ----------
        n_epochs
            Number of fine-tuning epochs.
        graph_weight
            Weight of the graph Laplacian loss (set the graph with :meth:`set_graph`).
        eta
            Prototype loss weight (use the value of the original training).
        lr
            Learning rate (lower than for training from scratch).
        reinitialize_prototypes
            Re-initialize prototypes (re-clustering for unlabeled prototypes) instead of
            continuing from the trained ones.
        train_kwargs
            Other :meth:`train` arguments (e.g. ``graph_sampling``, ``use_early_stopping``).
        """
        if not self.is_trained_:
            logger.warning("fine_tune() is meant for trained models; consider train().")
        plan_kwargs = dict(train_kwargs.pop("plan_kwargs", None) or {})
        plan_kwargs.setdefault("keep_prototypes", not reinitialize_prototypes)
        train_kwargs.setdefault("alpha_epoch_anneal", None)
        train_kwargs.setdefault("alpha_iter_anneal", None)
        return self.train(
            n_epochs=n_epochs,
            pretraining_epochs=0,
            graph_weight=graph_weight,
            graph_start_epoch=0,
            eta=eta,
            lr=lr,
            plan_kwargs=plan_kwargs,
            **train_kwargs,
        )

    # ------------------------------------------------------------------ minification
    def minify_adata(
        self,
        minified_data_type: str = ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
        use_latent_qzm_key: str = "X_latent_qzm",
        use_latent_qzv_key: str = "X_latent_qzv",
    ) -> None:
        """Replace the model's AnnData by a minified copy that keeps only the latent posterior.

        The expression matrix is dropped (unless ``minified_data_type`` is
        ``"latent_posterior_parameters_with_counts"``); latent mean and variance and the library
        size are stored instead. ``get_latent``, ``classify`` on the model's cells,
        ``get_normalized_expression`` and saving keep working on the much smaller object;
        training does not. Minification does not change how the model is trained.

        Parameters
        ----------
        minified_data_type
            ``"latent_posterior_parameters"`` (drop counts) or
            ``"latent_posterior_parameters_with_counts"``.
        use_latent_qzm_key, use_latent_qzv_key
            ``adata.obsm`` keys with the posterior mean and variance; computed here if
            missing.
        """
        if use_latent_qzm_key not in self.adata.obsm or use_latent_qzv_key not in self.adata.obsm:
            qzm, qzv = self.get_latent_representation(give_mean=True, return_dist=True)
            self.adata.obsm[use_latent_qzm_key] = qzm
            self.adata.obsm[use_latent_qzv_key] = qzv
        super().minify_adata(minified_data_type, use_latent_qzm_key, use_latent_qzv_key)

    # ------------------------------------------------------------------ graph
    @property
    def cell_graph(self):
        """The :class:`~scpolivi.graph.CellGraph` attached with :meth:`set_graph` (not saved)."""
        return getattr(self, "_graph", None)

    def set_graph(
        self,
        graph=None,
        graph_key: str | None = None,
    ) -> None:
        """Attach a within-sample cell graph for the graph Laplacian loss (scANCVI).

        Parameters
        ----------
        graph
            ``{sample: adjacency}`` (one square matrix per value of ``adata.obs[graph_key]``,
            rows in adata order; see :func:`scpolivi.graph.construct_sample_graphs`), an
            ``(n_obs, n_obs)`` matrix, or an ``adata.obsp`` key. ``None`` removes the graph.
        graph_key
            ``adata.obs`` column of the samples. Required for dict input; for a full matrix,
            edges across samples are dropped and the loss is normalized per sample.

        Notes
        -----
        The graph refers to the rows of the model's AnnData and is not saved with the model;
        attach it again after loading. For a query model (:meth:`load_query_data`), attach a
        graph of the query cells.
        """
        if graph is None:
            self._graph = None
            return
        from ..graph import build_cell_graph

        self._graph = build_cell_graph(self.adata, graph, graph_key)
        logger.info(
            "Graph with %d edges over %d groups attached.",
            self._graph.n_edges,
            len(self._graph.group_names),
        )

    # ------------------------------------------------------------------ inference
    def _validate_anndata(self, adata: AnnData | None = None, copy_if_view: bool = True):
        """Write scPoli's helper obs columns before scvi-tools transfers the setup."""
        if adata is not None and adata is not self.adata:
            if adata.is_view:
                if not copy_if_view:
                    raise ValueError("Please run `adata = adata.copy()`.")
                adata._init_as_actual(adata.copy())
            self._check_projection_input(adata)
            write_helper_columns(adata, self.condition_keys_, None, keep_existing_labeled=True)
        return super()._validate_anndata(adata, copy_if_view=copy_if_view)

    def _check_projection_input(self, adata: AnnData) -> None:
        """Checks for cells projected with a trained model (``get_latent``, ``classify``...).

        As in scArches, cell-type columns are optional (missing ones are treated as unlabeled)
        and every condition must be known to the model. Unlike scArches, genes are matched by
        name: a different gene order would otherwise silently give wrong latents.
        """
        if adata.n_vars == self.summary_stats.n_vars and self.adata is not None:
            ref_vars = self.adata.var_names
            if not adata.var_names.equals(ref_vars):
                if set(adata.var_names) == set(ref_vars):
                    raise ValueError(
                        "Genes are in a different order than in the model. Run "
                        "`SCPOLI.prepare_query_anndata(adata, model)` first."
                    )
                raise ValueError(
                    "Genes differ from the model's. Run `SCPOLI.prepare_query_anndata(adata, "
                    "model)` to pad and reorder them."
                )
        missing = [k for k in self.condition_keys_ if k not in adata.obs]
        if missing:
            raise KeyError(f"Condition keys {missing} not found in adata.obs.")
        for key, known in self.conditions_.items():
            new = sorted(set(adata.obs[key].astype(str)) - set(known))
            if new:
                raise ValueError(
                    f"Conditions {new} of {key!r} are unknown to the model. Map them first with "
                    "`SCPOLI.load_query_data` (or `shot_surgery`)."
                )
        for key in self.cell_type_keys_ or []:
            if key not in adata.obs:
                adata.obs[key] = pd.Series(np.nan, index=adata.obs_names, dtype=object)

    @torch.inference_mode()
    def _latent(
        self, adata: AnnData | None, mean: bool, batch_size: int | None = None
    ) -> torch.Tensor:
        adata = self._validate_anndata(adata)
        dl = self._make_data_loader(adata=adata, batch_size=batch_size)
        self.module.eval()
        out = []
        for tensors in dl:
            inf = self.module.inference(**self.module._get_inference_input(tensors))
            out.append((inf["z_mean"] if mean else inf["z"]).cpu())
        return torch.cat(out)

    def get_latent(self, adata: AnnData | None = None, mean: bool = False) -> np.ndarray:
        """Latent representation of cells (scArches API).

        Parameters
        ----------
        adata
            Cells to embed (default: the registered AnnData). Any AnnData with the model's
            genes and condition columns of known conditions; cell-type columns are optional.
        mean
            Return the posterior mean. ``False`` (scArches default) draws one sample from
            ``q(z|x)``; use ``True`` for downstream analysis.

        Returns
        -------
        ``(n_cells, latent_dim)`` array.
        """
        return self._latent(adata, mean).numpy()

    @torch.inference_mode()
    def classify(
        self,
        adata: AnnData | np.ndarray | torch.Tensor | None = None,
        prototype: bool = False,
        p: float = 2,
        get_prob: bool = False,
        log_distance: bool = True,
        scale_uncertainties: bool = False,
        use_mean: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Nearest-prototype classification at each hierarchy level.

        Parameters
        ----------
        adata
            Cells to classify, or a matrix of latent points if ``prototype=True``.
        prototype
            Classify latent points (e.g. prototypes) instead of cells.
        p
            Norm of the distance.
        get_prob
            Use softmax(-distance); ``uncert`` is then the max probability.
        log_distance
            Accepted for compatibility; scArches computes ``log1p`` of the distance but never
            uses it, so this has no effect either.
        scale_uncertainties
            RobustScaler followed by MinMax scaling to [0, 1] of the uncertainties.
        use_mean
            Classify the latent mean instead of a sample (scArches samples). Recommended.
        batch_size
            Minibatch size of the latent pass.

        Returns
        -------
        ``{level: {"preds", "uncert", "weighted_distances"}}``
        """
        if self.cell_type_keys_ is None or not bool(self.module.labeled_prototypes_initialized.any()):
            raise RuntimeError("Model was trained without prototypes.")
        if prototype:
            z = torch.as_tensor(np.asarray(adata), dtype=torch.float32)
        else:
            z = self._latent(adata, mean=use_mean, batch_size=batch_size)
        protos = self.module.prototypes_labeled_mean.detach().cpu()
        init = self.module.labeled_prototypes_initialized.cpu().numpy()
        names = np.array(list(self.cell_types_.keys()), dtype=object)
        results = {}
        for level in self.cell_type_keys_:
            classes = np.array([c for c in self._level_classes(level) if init[c]], dtype=int)
            preds, uncert, dists = [], [], []
            for chunk in torch.split(z, 512):
                pr, un, di = P.classify(chunk, protos, torch.as_tensor(classes), p=p, get_prob=get_prob)
                preds.append(pr)
                uncert.append(un)
                dists.append(di)
            pred = torch.cat(preds).numpy()
            unc = torch.cat(uncert).numpy()
            if scale_uncertainties:
                from sklearn.preprocessing import MinMaxScaler, RobustScaler

                unc = RobustScaler().fit_transform(unc.reshape(-1, 1))
                unc = MinMaxScaler(feature_range=(0, 1)).fit_transform(unc).reshape(-1)
            results[level] = {
                "preds": names[pred].astype(str),
                "uncert": unc,
                "weighted_distances": torch.cat(dists).numpy(),
            }
        return results

    def get_conditional_embeddings(self) -> AnnData | dict[str, AnnData]:
        """Learned condition embeddings.

        Returns
        -------
        For one condition key, an AnnData with one row per condition (``obs_names`` are the
        conditions) and the embedding in ``X``; for several keys, a dict of such AnnData keyed
        by condition key.
        """
        out = {}
        for i, (key, cats) in enumerate(self.conditions_.items()):
            w = self.module.get_embedding(f"cond_{i}").weight.detach().cpu().numpy()
            out[key] = AnnData(X=w, obs=pd.DataFrame(index=pd.Index(cats, name=key)))
        keys = self.condition_keys_
        return out[keys[0]] if len(keys) == 1 else out

    def get_prototypes_info(self, prototype_set: Literal["labeled", "unlabeled"] = "labeled") -> AnnData:
        """Prototype positions with their annotation and classification.

        Parameters
        ----------
        prototype_set
            ``"labeled"`` (cell-type prototypes) or ``"unlabeled"`` (cluster prototypes).

        Returns
        -------
        AnnData with one row per prototype (latent position in ``X``) and, per cell-type
        level, ``obs`` columns ``<level>`` (annotation), ``<level>_pred`` and
        ``<level>_uncert`` (nearest labeled prototype). Useful to plot prototypes on top of
        the cells and to decide which unlabeled prototypes form a new cell type.
        """
        if prototype_set == "labeled":
            protos = self.module.prototypes_labeled_mean.detach().cpu().numpy()
            batch_name = "prototype-Set Labeled"
        elif prototype_set == "unlabeled":
            protos = self.module.prototypes_unlabeled_mean.detach().cpu().numpy()
            batch_name = "prototype-Set Unlabeled"
        else:
            raise ValueError("prototype_set must be 'labeled' or 'unlabeled'.")
        info = AnnData(protos)
        info.obs["batch"] = batch_name
        results = self.classify(protos, prototype=True)
        for level in self.cell_type_keys_:
            if prototype_set == "labeled":
                truth = [ct if level in lv else "nan" for ct, lv in self.cell_types_.items()]
            else:
                truth = [str(i) for i in range(protos.shape[0])]
            info.obs[level] = truth
            info.obs[f"{level}_pred"] = results[level]["preds"]
            info.obs[f"{level}_uncert"] = results[level]["uncert"]
        return info

    def add_new_cell_type(
        self,
        cell_type_name: str,
        obs_key: str,
        prototypes: Sequence[int],
        adata: AnnData | None = None,
    ) -> None:
        """Turn unlabeled prototypes into a labeled prototype of a new cell type.

        The new prototype is the mean of the given unlabeled prototypes; its covariance is
        computed from the cells assigned to it. Afterwards :meth:`classify` can predict the
        new type.

        Parameters
        ----------
        cell_type_name
            Name of the new cell type.
        obs_key
            Cell-type level (one of the cell-type keys) the new type belongs to.
        prototypes
            Indices of unlabeled prototypes (see ``get_prototypes_info("unlabeled")``).
        adata
            Cells used for the covariance (default: the registered AnnData).
        """
        if cell_type_name in self.cell_types_:
            raise ValueError(f"Cell type {cell_type_name!r} already exists.")
        if obs_key not in (self.cell_type_keys_ or []):
            raise ValueError(f"{obs_key!r} is not one of the cell type keys {self.cell_type_keys_}.")
        m = self.module
        dev = m.prototypes_labeled_mean.device
        new_mean = m.prototypes_unlabeled_mean[list(prototypes)].mean(0, keepdim=True)
        m.prototypes_labeled_mean = torch.cat([m.prototypes_labeled_mean, new_mean.to(dev)])
        new_idx = m.n_cell_types - 1
        latent = self._latent(adata, mean=False).to(dev)
        classes = torch.as_tensor(
            [*[c for c in self._level_classes(obs_key)], new_idx], device=dev
        )
        pred, _, _ = P.classify(latent, m.prototypes_labeled_mean, classes)
        cells = latent[pred == new_idx]
        d = m.n_latent
        new_cov = P.cov(cells).reshape(1, d, d) if cells.shape[0] > 1 else torch.full((1, d, d), float("nan"), device=dev)
        m.prototypes_labeled_cov = torch.cat([m.prototypes_labeled_cov, new_cov])
        m.prototypes_fixed = torch.cat([m.prototypes_fixed, torch.zeros(1, dtype=torch.bool, device=dev)])
        self._extend_cell_type_registry(cell_type_name, [obs_key])

    def _extend_cell_type_registry(self, name: str, levels: list[str]) -> None:
        for manager in {id(m): m for m in [self.adata_manager, *self._get_most_recent_managers()]}.values():
            field_reg = manager.registry["field_registries"][SK.CELL_TYPES_KEY]
            st = dict(field_reg["state_registry"])
            st["mapping"] = [*st["mapping"], name]
            cts = {k: list(v) for k, v in dict(st["cell_types"]).items()}
            cts[name] = levels
            st["cell_types"] = cts
            field_reg["state_registry"] = st
            field_reg["summary_stats"] = {
                **dict(field_reg["summary_stats"]),
                f"n_{SK.CELL_TYPES_KEY}": len(st["mapping"]),
            }

    def _get_most_recent_managers(self) -> list[AnnDataManager]:
        return list(self._per_instance_manager_store.get(self.id, {}).values())

    def update_labeled_indices(self, labeled_indices: Sequence[int]) -> None:
        """Change which cells of the registered AnnData count as labeled.

        Parameters
        ----------
        labeled_indices
            Positional indices of labeled cells; all other cells become unlabeled (they still
            train the VAE and the unlabeled prototypes, but not the labeled prototypes).
        """
        flag = np.zeros(self.adata.n_obs, dtype=np.float32)
        flag[np.asarray(labeled_indices, dtype=int)] = 1.0
        self.adata.obs[LABELED_COL] = flag

    @torch.inference_mode()
    def get_recon_loss(
        self,
        adata: AnnData | None = None,
        batch_size: int = 128,
        condition_map: dict[str, dict[str, str]] | None = None,
    ) -> float:
        """Mean reconstruction loss per cell (z sampled), optionally swapping conditions.

        Parameters
        ----------
        adata
            Cells to evaluate (default: the registered AnnData).
        batch_size
            Minibatch size.
        condition_map
            ``condition_map[key][value] = other_value`` evaluates cells of ``value`` as if they
            came from ``other_value`` (embedding and dispersion), as in :meth:`shot_surgery`.

        Returns
        -------
        Mean negative log-likelihood per cell.
        """
        adata = self._validate_anndata(adata)
        conds = self.conditions_
        combined = {c: i for i, c in enumerate(self.conditions_combined_)}
        total, n = 0.0, 0
        self.module.eval()
        for tensors in self._make_data_loader(adata=adata, batch_size=batch_size):
            tensors = dict(tensors)
            if condition_map:
                c = tensors[SK.CONDITIONS_KEY].clone().long()
                for j, key in enumerate(self.condition_keys_):
                    for src, dst in condition_map.get(key, {}).items():
                        c[:, j][c[:, j] == conds[key].index(src)] = conds[key].index(dst)
                tensors[SK.CONDITIONS_KEY] = c
                values = [[conds[k][int(v)] for k, v in zip(self.condition_keys_, row, strict=True)] for row in c.tolist()]
                b = [combined.get("_".join(v)) for v in values]
                if None not in b:
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.as_tensor(b)[:, None]
            _, _, out = self.module(tensors)
            rl = out.reconstruction_loss["reconstruction_loss"]
            total += float(rl.sum())
            n += rl.shape[0]
        return total / max(n, 1)

    # ------------------------------------------------------------------ query mapping
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: str | SCPOLI,
        labeled_indices: Sequence[int] | None = None,
        unknown_ct_names: Sequence[str] | None = None,
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
        return_new_conditions: bool = False,
        accelerator: str = "auto",
        device: int | str = "auto",
        inplace_subset_query_vars: bool = False,
    ):
        """scArches surgery: extend a reference model with the conditions of a query.

        Parameters
        ----------
        adata
            Query data with the reference's condition (and optionally cell-type) columns.
        reference_model
            Trained SCPOLI model or path to a saved one.
        labeled_indices
            Labeled query cells (default: all cells, as in scArches).
        unknown_ct_names
            Cell-type names of the query to ignore for prototypes.
        freeze
            Train only the condition embeddings and the dispersion (scPoli surgery).
        freeze_expression
            Accepted for compatibility; it has no effect in scArches either.
        remove_dropout
            Set dropout to 0 in the query model.
        return_new_conditions
            Also return ``{key: [new conditions]}``.
        accelerator, device
            Where to load the model.
        inplace_subset_query_vars
            Subset the query's genes to the reference genes in place.

        Returns
        -------
        The query model (untrained), or ``(model, new_conditions)``.
        """
        if not isinstance(reference_model, str):
            ref_conditions = deepcopy(reference_model.conditions_)
            ref_n_unlab = reference_model.module.prototypes_unlabeled_mean.shape[0]
            ref_init = reference_model.module.labeled_prototypes_initialized.cpu()
            ref_meta = reference_model.obs_metadata_
        else:
            ref = cls._load_reference_summary(reference_model)
            ref_conditions, ref_n_unlab, ref_init, ref_meta = ref

        flag = np.ones(adata.n_obs, dtype=np.float32)
        if labeled_indices is not None:
            flag[:] = 0
            flag[np.asarray(labeled_indices, dtype=int)] = 1
        adata.obs[LABELED_COL] = flag

        overrides = {
            "unknown_ct_names": list(unknown_ct_names) if unknown_ct_names else None,
            "n_unlabeled_prototypes": ref_n_unlab,
            "keep_labeled": True,
        }
        if remove_dropout:
            overrides["dr_rate"] = 0.0
        token = _QUERY_OVERRIDES.set(overrides)
        try:
            model = super().load_query_data(
                adata=adata,
                reference_model=reference_model,
                inplace_subset_query_vars=inplace_subset_query_vars,
                accelerator=accelerator,
                device=device,
                unfrozen=True,
            )
        finally:
            _QUERY_OVERRIDES.reset(token)

        m = model.module
        fixed = torch.zeros(m.n_cell_types, dtype=torch.bool, device=m.prototypes_fixed.device)
        fixed[: len(ref_init)] = ref_init.to(fixed.device)
        m.prototypes_fixed = fixed
        if freeze:
            for name, par in m.named_parameters():
                par.requires_grad = "_embeddings_dict" in name or name == "px_r"
            for mod in m.modules():
                if isinstance(mod, torch.nn.BatchNorm1d):
                    mod.track_running_stats = False  # as scArches (frozen BN, batch stats)

        new_meta = model._condition_metadata(adata)
        if ref_meta is not None:
            model.obs_metadata_ = pd.concat([ref_meta, new_meta[~new_meta.index.isin(ref_meta.index)]])
        else:
            model.obs_metadata_ = new_meta
        model.is_trained_ = False
        if return_new_conditions:
            new = {
                k: [c for c in v if c not in ref_conditions.get(k, [])]
                for k, v in model.conditions_.items()
            }
            return model, new
        return model

    @staticmethod
    def _load_reference_summary(dir_path: str):
        from scvi.model.base._save_load import _load_saved_files

        attr_dict, _, state, _ = _load_saved_files(dir_path, load_adata=False, map_location="cpu")
        st = attr_dict["registry_"]["field_registries"][SK.CONDITIONS_KEY]["state_registry"]
        conds = {k: [str(c) for c in v] for k, v in st["mappings"].items()}
        unl = state.get("prototypes_unlabeled_mean")
        mean = state.get("prototypes_labeled_mean")
        init = ~torch.isnan(mean).any(dim=1) if mean is not None else torch.zeros(0, dtype=torch.bool)
        return conds, 0 if unl is None else unl.shape[0], init, attr_dict.get("obs_metadata_")

    @classmethod
    def shot_surgery(
        cls,
        adata: AnnData,
        reference_model: str | SCPOLI,
        labeled_indices: Sequence[int] | None = None,
        unknown_ct_names: Sequence[str] | None = None,
        train_epochs: int = 0,
        batch_size: int = 128,
        subsample: float = 1.0,
        **train_kwargs,
    ):
        """Map new conditions onto the closest reference condition without training.

        For each new value of each condition key, the reference value with the lowest
        reconstruction loss (other keys unchanged) is chosen and its embedding (and the
        dispersion of the matching combined condition) is copied. Optionally fine-tune.

        Parameters
        ----------
        adata
            Query data (see :meth:`load_query_data`).
        reference_model
            Trained reference model or path to a saved one.
        labeled_indices, unknown_ct_names
            As in :meth:`load_query_data`.
        train_epochs
            If > 0, fine-tune the query model (embeddings and dispersion) for this many
            epochs after the mapping.
        batch_size
            Minibatch size of the reconstruction-loss evaluation.
        subsample
            Fraction of query cells used to evaluate the reconstruction loss.
        **train_kwargs
            Passed to :meth:`train` when ``train_epochs > 0``.

        Returns
        -------
        ``(model, mapping)`` with ``mapping[key][new] = reference`` (a flat
        ``{new: reference}`` dict for a single condition key, as in scArches).
        """
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1].")
        model, new_conditions = cls.load_query_data(
            adata,
            reference_model,
            labeled_indices=labeled_indices,
            unknown_ct_names=unknown_ct_names,
            freeze=train_epochs > 0,
            return_new_conditions=True,
        )
        model.module.eval()
        conds = model.conditions_
        rng = np.random.default_rng(settings.seed)
        mapping: dict[str, dict[str, str]] = {}
        for key, new_values in new_conditions.items():
            mapping[key] = {}
            old_values = [c for c in conds[key] if c not in new_values]
            for new in new_values:
                cells = np.where(adata.obs[key].astype(str).values == new)[0]
                if subsample < 1:
                    cells = rng.choice(cells, int(subsample * len(cells)), replace=False)
                sub = adata[cells].copy()
                best, best_loss = None, None
                for old in old_values:
                    loss = model.get_recon_loss(sub, batch_size, condition_map={key: {new: old}})
                    logger.info("%s: %s -> %s, recon loss %.4f", key, new, old, loss)
                    if best_loss is None or loss < best_loss:
                        best, best_loss = old, loss
                mapping[key][new] = best

        m = model.module
        with torch.no_grad():
            for j, key in enumerate(model.condition_keys_):
                emb = m.get_embedding(f"cond_{j}").weight
                for new, old in mapping[key].items():
                    emb[conds[key].index(new)] = emb[conds[key].index(old)]
            if m.px_r is not None and m.px_r.ndim == 2:
                comb = model.conditions_combined_
                for ci, cname in enumerate(comb):
                    codes = m.combined_to_conditions[ci].tolist()
                    vals = [conds[k][c] for k, c in zip(model.condition_keys_, codes, strict=True)]
                    mapped = [mapping.get(k, {}).get(v, v) for k, v in zip(model.condition_keys_, vals, strict=True)]
                    target = "_".join(mapped)
                    if mapped != vals and target in comb:
                        m.px_r[:, ci] = m.px_r[:, comb.index(target)]
        if train_epochs > 0:
            model.train(n_epochs=train_epochs, **train_kwargs)
        if len(model.condition_keys_) == 1:
            return model, mapping[model.condition_keys_[0]]
        return model, mapping

    # ------------------------------------------------------------------ scArches import
    @classmethod
    def from_scarches(
        cls,
        dir_path: str,
        adata: AnnData,
        accelerator: str = "auto",
        device: int | str = "auto",
    ) -> SCPOLI:
        """Load a model saved by scArches ``scPoli.save`` (scArches not required).

        Reads ``attr.pkl``, ``model_params.pt`` and ``var_names.csv``, registers ``adata``
        with scArches' category orders and converts the weights and prototypes. The result
        gives the same latents as scArches and can be trained, mapped to queries and saved
        like any SCPOLI model.

        Parameters
        ----------
        dir_path
            Directory written by scArches ``scPoli.save``.
        adata
            Data with the model's genes (same order) and condition columns; cell-type
            columns are optional.
        accelerator, device
            Where to load the model.

        Returns
        -------
        SCPOLI model.
        """
        from ._scarches_io import (
            combined_to_conditions,
            convert_state_dict,
            module_kwargs_from_attrs,
            read_scarches_checkpoint,
        )

        attrs, state, var_names = read_scarches_checkpoint(dir_path)
        if list(adata.var_names) != list(var_names):
            raise ValueError("adata.var_names do not match the scArches model's var_names.")
        cond_keys = list(attrs["condition_keys_"])
        ct_keys = list(attrs["cell_type_keys_"] or [])
        for key in cond_keys:
            cats = [str(c) for c in attrs["conditions_"][key]]
            vals = adata.obs[key].astype(str)
            unknown = set(vals.unique()) - set(cats)
            if unknown:
                raise ValueError(f"{key}: conditions {sorted(unknown)} are not in the scArches model.")
            adata.obs[key] = pd.Categorical(vals, categories=cats)
        combined = [str(c) for c in attrs["conditions_combined_"]]
        for key in ct_keys:
            if key not in adata.obs:
                adata.obs[key] = pd.Series(np.nan, index=adata.obs_names, dtype=object)

        # Register with the scArches category orders; then force the cell-type namespace.
        adata.obs[COMBINED_CONDITIONS_COL] = pd.Categorical(
            np.repeat(combined[0], adata.n_obs), categories=combined
        )  # placeholder values; only the category order is used
        token = _QUERY_OVERRIDES.set({"preserve_categories": True})
        try:
            cls.setup_anndata(
                adata,
                condition_keys=cond_keys,
                cell_type_keys=ct_keys or None,
                unknown_ct_names=attrs.get("unknown_ct_names_"),
            )
        finally:
            _QUERY_OVERRIDES.reset(token)
        manager = cls._get_most_recent_anndata_manager(adata)
        cell_types = {str(k): list(v) for k, v in (attrs.get("cell_types_") or {}).items()}
        cls._force_cell_types(manager, adata, cell_types)

        kw = module_kwargs_from_attrs(attrs)
        model = cls(
            adata,
            hidden_layer_sizes=kw["hidden_layer_sizes"],
            latent_dim=kw["n_latent"],
            embedding_dims=kw["embedding_dims"],
            embedding_max_norm=kw["embedding_max_norm"],
            inject_condition=kw["inject_condition"],
            recon_loss=kw["recon_loss"],
            dr_rate=kw["dropout_rate"],
            use_bn=kw["use_bn"],
            use_ln=kw["use_ln"],
            obs_metadata=attrs.get("obs_metadata_"),
            n_unlabeled_prototypes=0,
        )
        sd = convert_state_dict(state, attrs)
        sd["combined_to_conditions"] = combined_to_conditions(attrs)
        model.module.load_state_dict(sd, strict=True)
        _, _, dev = parse_device_args(accelerator, device, return_device="torch", validate_single_device=True)
        model.to_device(dev)
        model.module.eval()
        model.is_trained_ = bool(attrs.get("is_trained_", True))
        return model

    @staticmethod
    def _force_cell_types(manager: AnnDataManager, adata: AnnData, cell_types: dict) -> None:
        """Replace the registered cell-type namespace by a given ordered one."""
        reg = manager.registry["field_registries"][SK.CELL_TYPES_KEY]
        st = dict(reg["state_registry"])
        if not st.get("field_keys"):
            return
        mapping = list(cell_types)
        index = {ct: i for i, ct in enumerate(mapping)}
        key = f"_scvi_{SK.CELL_TYPES_KEY}"
        adata.obsm[key] = pd.DataFrame(
            {
                lvl: [index.get(str(v), -1) if not pd.isna(v) else -1 for v in adata.obs[lvl]]
                for lvl in st["field_keys"]
            },
            index=adata.obs_names,
        ).astype(np.int64)
        st["mapping"] = mapping
        st["cell_types"] = cell_types
        reg["state_registry"] = st
        reg["summary_stats"] = {**dict(reg["summary_stats"]), f"n_{SK.CELL_TYPES_KEY}": len(mapping)}
