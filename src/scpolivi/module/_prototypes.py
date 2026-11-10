"""Prototype computations of scPoli (vectorized ports of ``scPoliTrainer`` methods).

Labels are an ``(n_cells, n_levels)`` integer matrix in one shared cell-type namespace,
``-1`` meaning unknown. As in scArches, a cell contributes to the prototype of every
level it is labeled at, and duplicate (cell, label) pairs are counted twice.
"""

from __future__ import annotations

import torch


def _flatten_labels(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(cell_index, label) pairs of all known labels, in scArches ``nonzero`` order."""
    labels = labels.long()
    if labels.ndim == 1:
        labels = labels[:, None]
    cell_idx = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(labels)
    keep = labels >= 0
    return cell_idx[keep], labels[keep]


def labeled_prototype_loss(
    z: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor, p: float = 2.0
) -> torch.Tensor:
    """Sum over present classes of the mean distance of their cells to the class prototype."""
    cell_idx, lab = _flatten_labels(labels)
    if lab.numel() == 0:
        return torch.zeros((), device=z.device)
    dists = torch.cdist(z, prototypes, p=p)
    d = dists[cell_idx, lab]
    n_cls = prototypes.shape[0]
    sums = torch.zeros(n_cls, device=z.device, dtype=d.dtype).index_add_(0, lab, d)
    counts = torch.bincount(lab, minlength=n_cls).to(d.dtype)
    present = counts > 0
    return (sums[present] / counts[present]).sum()


def unlabeled_prototype_loss(
    z: torch.Tensor, prototypes: torch.Tensor, p: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean over occupied clusters of the mean distance of cells to their nearest prototype."""
    dists = torch.cdist(z, prototypes, p=p)
    min_dist, y_hat = torch.min(dists, 1)
    n = prototypes.shape[0]
    sums = torch.zeros(n, device=z.device, dtype=min_dist.dtype).index_add_(0, y_hat, min_dist)
    counts = torch.bincount(y_hat, minlength=n)
    present = counts > 0
    loss = (sums[present] / counts[present].to(sums.dtype)).mean()
    return loss, counts[present]


def cov(x: torch.Tensor) -> torch.Tensor:
    """Unbiased covariance of rows of ``x`` (scArches ``cov`` with defaults)."""
    xm = x - x.mean(0, keepdim=True)
    return (xm.t() @ xm / (x.shape[0] - 1)).squeeze()


@torch.no_grad()
def update_labeled_prototypes(
    z: torch.Tensor,
    labels: torch.Tensor,
    prev_mean: torch.Tensor | None,
    prev_cov: torch.Tensor | None,
    n_classes: int,
    update_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute prototype means/covariances from labeled latent codes.

    A class is recomputed if it has labeled cells and (``update_mask`` is None or the
    mask is True for it); otherwise its previous value is kept. Classes without cells and
    without previous values are filled with NaN, which marks them as uninitialized.
    """
    d = z.shape[1]
    cell_idx, lab = _flatten_labels(labels)
    mean = torch.full((n_classes, d), float("nan"), device=z.device, dtype=z.dtype)
    cv = torch.full((n_classes, d, d), float("nan"), device=z.device, dtype=z.dtype)
    if prev_mean is not None:
        mean[: prev_mean.shape[0]] = prev_mean.to(z.dtype)
        cv[: prev_cov.shape[0]] = prev_cov.to(z.dtype).reshape(-1, d, d)
    present = torch.unique(lab)
    for value in present.tolist():
        if update_mask is not None and not bool(update_mask[value]):
            continue
        rows = cell_idx[lab == value]
        zz = z[rows]
        mean[value] = zz.mean(0)
        cv[value] = cov(zz).reshape(d, d)
    return mean, cv


@torch.no_grad()
def classify(
    z: torch.Tensor,
    prototypes: torch.Tensor,
    classes: torch.Tensor,
    p: float = 2.0,
    get_prob: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-prototype classification among ``classes`` (scArches ``scpoli.classify``).

    Returns predictions (global class codes), uncertainty (min distance, or max softmax
    probability if ``get_prob``) and the distance (or probability) matrix.
    """
    dists = torch.cdist(z, prototypes[classes], p=p)
    if get_prob:
        dists = torch.softmax(-dists, dim=1)
        uncert, idx = torch.max(dists, dim=1)
    else:
        uncert, idx = torch.min(dists, dim=1)
    return classes[idx], uncert, dists


# ----------------------------------------------------------------- loop references
def _labeled_prototype_loss_loop(z, prototypes, labels, p=2.0):
    """Literal port of ``scPoliTrainer.prototype_labeled_loss`` (used in tests)."""
    unique_labels = torch.unique(labels, sorted=True)
    dists = torch.cdist(z, prototypes, p=p)
    loss = torch.tensor(0.0, device=z.device)
    if unique_labels.tolist() == [-1]:
        return loss
    for value in unique_labels:
        if value == -1:
            continue
        indices = labels.eq(value).nonzero(as_tuple=False)[:, 0]
        loss += dists[indices, value].sum(0) / len(indices)
    return loss
