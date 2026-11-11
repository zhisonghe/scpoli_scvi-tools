"""Helpers used by :meth:`scpolivi.SCPOLI.setup_anndata`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalObsField,
)

from .._constants import COMBINED_CONDITIONS_COL, INDICES_COL, LABELED_COL, SCPOLI_REGISTRY_KEYS
from ._fields import HierarchicalLabelsField


def as_list(keys: str | Sequence[str] | None) -> list[str]:
    """Normalize a key or a sequence of keys to a list."""
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    return list(keys)


def combine_conditions(adata: AnnData, condition_keys: list[str]) -> pd.Series:
    """scArches ``conditions_combined``: values of all condition keys joined by '_'."""
    obs = adata.obs[condition_keys].astype(str)
    if len(condition_keys) == 1:
        return obs[condition_keys[0]]
    return obs.apply(lambda row: "_".join(row), axis=1)


def write_helper_columns(
    adata: AnnData,
    condition_keys: list[str],
    labeled_indices: Sequence[int] | np.ndarray | None,
    keep_existing_labeled: bool = False,
    preserve_categories: bool = False,
) -> None:
    """Write the combined-condition and labeled-flag columns used by the fields.

    Condition columns become string categoricals whose category order is the order of first
    appearance, which reproduces scArches' condition encoders. With ``preserve_categories``
    (used when importing scArches models), pre-set categorical orders, including unused
    categories, are kept as they are.
    """
    for key in condition_keys:
        if key not in adata.obs:
            raise KeyError(f"Condition key {key!r} not found in adata.obs.")
        col = adata.obs[key]
        if preserve_categories and isinstance(col.dtype, pd.CategoricalDtype):
            continue
        values = col.astype(str)
        adata.obs[key] = pd.Categorical(values, categories=list(pd.unique(values)))

    combined = combine_conditions(adata, condition_keys)
    existing = adata.obs.get(COMBINED_CONDITIONS_COL)
    if (
        preserve_categories
        and existing is not None
        and isinstance(existing.dtype, pd.CategoricalDtype)
    ):
        # keep a pre-set category order (used when importing scArches models)
        cats = list(existing.cat.categories)
        extra = [c for c in pd.unique(combined) if c not in cats]
        adata.obs[COMBINED_CONDITIONS_COL] = pd.Categorical(combined, categories=cats + extra)
    else:
        adata.obs[COMBINED_CONDITIONS_COL] = pd.Categorical(
            combined, categories=list(pd.unique(combined))
        )

    adata.obs[INDICES_COL] = np.arange(adata.n_obs)

    if labeled_indices is not None:
        flag = np.zeros(adata.n_obs, dtype=np.float32)
        flag[np.asarray(labeled_indices, dtype=int)] = 1.0
        adata.obs[LABELED_COL] = flag
    elif not (keep_existing_labeled and LABELED_COL in adata.obs):
        adata.obs[LABELED_COL] = np.ones(adata.n_obs, dtype=np.float32)


def build_fields(
    layer: str | None,
    condition_keys: list[str],
    cell_type_keys: list[str],
    unknown_ct_names: list[str] | None,
) -> list:
    """AnnData fields of an SCPOLI model."""
    return [
        LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
        # conditions_combined drives the NB/ZINB dispersion (scArches ``theta``)
        CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, COMBINED_CONDITIONS_COL),
        CategoricalJointObsField(SCPOLI_REGISTRY_KEYS.CONDITIONS_KEY, condition_keys),
        HierarchicalLabelsField(
            SCPOLI_REGISTRY_KEYS.CELL_TYPES_KEY,
            cell_type_keys,
            unknown_ct_names=unknown_ct_names,
            labeled_obs_key=LABELED_COL,
        ),
        NumericalObsField(SCPOLI_REGISTRY_KEYS.LABELED_KEY, LABELED_COL),
        # row positions, to look up graph edges of a minibatch
        NumericalObsField(REGISTRY_KEYS.INDICES_KEY, INDICES_COL),
    ]
