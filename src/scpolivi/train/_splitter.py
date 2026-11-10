"""Train/validation split and sampling of scPoli (``trainers/scpoli/_utils.py`` in scArches)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter

from .._constants import SCPOLI_REGISTRY_KEYS as SK


def scpoli_split(
    adata_manager: AnnDataManager, train_frac: float, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split as in scArches ``train_test_split``.

    With cell-type keys: labeled cells are split per class of the finest level (the level
    with most distinct values), unlabeled cells are split as a block (shuffled here; scArches
    does not shuffle them). Without cell-type keys: split per combined condition.
    """
    adata = adata_manager.adata
    n = adata.n_obs
    indices = np.arange(n)
    if train_frac >= 1:
        return indices, np.array([], dtype=int)

    ct_state = adata_manager.get_state_registry(SK.CELL_TYPES_KEY)
    levels = list(ct_state.get("field_keys", []))
    train, val = [], []
    if levels:
        n_cts, finest = 0, None
        for lvl in levels:  # ties resolve to the later level, like scArches
            k = adata.obs[lvl].nunique()
            if k >= n_cts:
                n_cts, finest = k, lvl
        labeled = np.asarray(adata_manager.get_from_registry(SK.LABELED_KEY)).ravel() > 0.5
        lab_idx, unl_idx = indices[labeled], indices[~labeled]
        if len(lab_idx):
            # missing labels form one stratum (NaN != NaN would otherwise drop them)
            col = adata.obs[finest].astype(object)
            values = col.where(col.notna(), "__scpoli_nan__").to_numpy()[lab_idx]
            for ct in _unique_in_order(values):
                ct_idx = lab_idx[values == ct].copy()
                rng.shuffle(ct_idx)
                k = math.ceil(train_frac * len(ct_idx))
                train.append(ct_idx[:k])
                val.append(ct_idx[k:])
        if len(unl_idx):
            unl_idx = unl_idx.copy()
            rng.shuffle(unl_idx)
            k = math.ceil(train_frac * len(unl_idx))
            train.append(unl_idx[:k])
            val.append(unl_idx[k:])
    else:
        comb = np.asarray(adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel()
        for c in _unique_in_order(comb):
            c_idx = indices[comb == c].copy()
            rng.shuffle(c_idx)
            k = math.ceil(train_frac * len(c_idx))
            train.append(c_idx[:k])
            val.append(c_idx[k:])
    return np.concatenate(train).astype(int), np.concatenate(val).astype(int)


def _unique_in_order(values: np.ndarray) -> list:
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


class SCPoliDataSplitter(DataSplitter):
    """DataSplitter with scPoli's stratified split and optional condition-balanced sampling.

    Parameters
    ----------
    train_frac
        Fraction of cells used for training (scPoli ``train_frac``).
    use_stratified_sampling
        Sample training cells with replacement, with weights inversely proportional to the
        size of their combined condition.
    n_samples
        Number of cells drawn per epoch when ``use_stratified_sampling`` (default: all).
    graph
        :class:`~scpolivi.graph.CellGraph`; with ``graph_sampling="neighbors"`` every
        training minibatch also loads one graph neighbour per cell.
    graph_sampling
        ``"batch"`` (edges inside random minibatches, scANCVI) or ``"neighbors"``.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_frac: float = 0.9,
        use_stratified_sampling: bool = False,
        n_samples: int | None = None,
        graph=None,
        graph_sampling: str = "batch",
        **kwargs,
    ):
        super().__init__(adata_manager, train_size=min(train_frac, 1.0), **kwargs)
        self.train_frac = train_frac
        self.use_stratified_sampling = use_stratified_sampling
        self.n_samples = n_samples
        if graph_sampling not in ("batch", "neighbors"):
            raise ValueError("graph_sampling must be 'batch' or 'neighbors'.")
        self.graph = graph
        self.graph_sampling = graph_sampling
        if graph is not None and graph_sampling == "neighbors" and use_stratified_sampling:
            raise ValueError("graph_sampling='neighbors' cannot be combined with stratified sampling.")

    def setup(self, stage: str | None = None):
        rng = np.random.RandomState(seed=settings.seed)
        self.train_idx, self.val_idx = scpoli_split(self.adata_manager, self.train_frac, rng)
        self.test_idx = np.array([], dtype=int)
        self.n_train, self.n_val = len(self.train_idx), len(self.val_idx)

    def _batch_size(self) -> int:
        return self.data_loader_kwargs.get("batch_size") or settings.batch_size

    def _loader(self, dataset, sampler):
        # The sampler yields index batches; batch_size=None disables torch auto-batching
        # (same trick as scvi's AnnDataLoader).
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.data_loader_kwargs.get("num_workers", settings.dl_num_workers),
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        if self.graph is not None and self.graph_sampling == "neighbors":
            from ..graph import NeighborBatchSampler, NeighborPairDataset

            sampler = NeighborBatchSampler(
                self.graph, self.train_idx, self._batch_size(), drop_last=bool(self.drop_last), seed=settings.seed
            )
            dataset = self.adata_manager.create_torch_dataset(
                indices=None, load_sparse_tensor=self.load_sparse_tensor
            )
            return self._loader(NeighborPairDataset(dataset), sampler)
        if not self.use_stratified_sampling:
            return super().train_dataloader()
        comb = np.asarray(self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel()
        comb = comb[self.train_idx]
        _, inverse, counts = np.unique(comb, return_inverse=True, return_counts=True)
        weights = (1.0 / len(comb)) / counts[inverse]
        n = self.n_samples if self.n_samples and self.n_samples < len(comb) else len(comb)
        batch_size = self._batch_size()
        sampler = BatchSampler(
            WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), n, replacement=True),
            batch_size=batch_size,
            drop_last=self.drop_last,
        )
        dataset = self.adata_manager.create_torch_dataset(
            indices=self.train_idx, load_sparse_tensor=self.load_sparse_tensor
        )
        return self._loader(dataset, sampler)
