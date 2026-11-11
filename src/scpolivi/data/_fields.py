"""AnnData fields for scPoli.

Conditions reuse scvi-tools' :class:`~scvi.data.fields.CategoricalJointObsField`, so they get
``extend_categories`` on query transfer for free. Cell types need a custom field: scPoli
encodes all hierarchy levels in one shared namespace (a name that appears at two levels maps
to the same prototype), restricts the namespace to labeled cells, and maps unknown names to -1.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import rich
from anndata import AnnData

from scvi import settings
from scvi.data.fields._arraylike_field import BaseJointField

from .._constants import UNKNOWN_CODE


def _appearance_unique(values: pd.Series) -> list:
    """Unique non-null values in order of first appearance (scArches uses ``Series.unique``)."""
    return [v for v in pd.unique(values) if not pd.isna(v)]


class HierarchicalLabelsField(BaseJointField):
    """Joint obs field for one or several cell-type annotation levels.

    Parameters
    ----------
    registry_key
        Key to register the field under.
    attr_keys
        ``adata.obs`` columns, one per hierarchy level (coarse/fine order is irrelevant).
    unknown_ct_names
        Names that mark cells to ignore for prototypes (encoded as -1).
    labeled_obs_key
        ``adata.obs`` column (0/1) marking labeled cells. Only names found among labeled cells
        enter the namespace, as in scArches.
    """

    MAPPING_KEY = "mapping"  # list[str], index = code
    CELL_TYPES_KEY = "cell_types"  # dict[name, list[level]]
    FIELD_KEYS_KEY = "field_keys"
    UNKNOWN_KEY = "unknown_ct_names"

    def __init__(
        self,
        registry_key: str,
        attr_keys: list[str] | None,
        unknown_ct_names: list[str] | None,
        labeled_obs_key: str,
    ) -> None:
        super().__init__(registry_key, attr_keys, field_type="obsm", required=False)
        self.unknown_ct_names = list(unknown_ct_names) if unknown_ct_names else []
        self.labeled_obs_key = labeled_obs_key
        self.count_stat_key = f"n_{registry_key}"
        self.levels_stat_key = f"n_{registry_key}_levels"

    # ------------------------------------------------------------------ helpers
    def _labeled_mask(self, adata: AnnData) -> np.ndarray:
        if self.labeled_obs_key in adata.obs:
            return np.asarray(adata.obs[self.labeled_obs_key]).astype(bool)
        return np.ones(adata.n_obs, dtype=bool)

    def _collect(self, adata: AnnData, cell_types: dict) -> dict:
        """Add names of labeled cells, level by level, to ``cell_types`` (scArches semantics).

        New names are appended with the levels they appear at; levels of names that are
        already known are not extended (as in ``scPoli.load_query_data``).
        """
        mask = self._labeled_mask(adata)
        new: dict = {}
        for level in self.attr_keys:
            for ct in _appearance_unique(adata.obs[level][mask]):
                ct = str(ct)
                if ct in self.unknown_ct_names:
                    continue
                new.setdefault(ct, []).append(level)
        out = {k: list(v) for k, v in cell_types.items()}
        for ct, levels in new.items():
            if ct not in out:
                out[ct] = levels
        return out

    def _encode(self, adata: AnnData, mapping: list[str]) -> None:
        index = {ct: i for i, ct in enumerate(mapping)}
        codes = {}
        for level in self.attr_keys:
            col = adata.obs[level]
            codes[level] = np.array(
                [UNKNOWN_CODE if pd.isna(v) else index.get(str(v), UNKNOWN_CODE) for v in col],
                dtype=np.int64,
            )
        adata.obsm[self.attr_key] = pd.DataFrame(codes, index=adata.obs_names)

    def _state(self, cell_types: dict) -> dict:
        return {
            self.MAPPING_KEY: list(cell_types.keys()),
            self.CELL_TYPES_KEY: cell_types,
            self.FIELD_KEYS_KEY: list(self.attr_keys),
            self.UNKNOWN_KEY: list(self.unknown_ct_names),
        }

    # ------------------------------------------------------------------ API
    def register_field(self, adata: AnnData) -> dict:
        """Register the field."""
        if self.is_empty:
            return self._state({})
        super().register_field(adata)
        cell_types = self._collect(adata, {})
        self._encode(adata, list(cell_types))
        return self._state(cell_types)

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: AnnData,
        extend_categories: bool = False,
        allow_missing_labels: bool = False,
        **kwargs,
    ) -> dict:
        """Transfer the field, optionally adding cell types of labeled query cells."""
        if self.is_empty:
            return self._state({})
        missing = [k for k in self.attr_keys if k not in adata_target.obs]
        if missing:
            if not allow_missing_labels:
                raise KeyError(f"Cell type keys {missing} not found in adata.obs.")
            warnings.warn(
                f"Missing cell type keys {missing}; treating these levels as unknown.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            for k in missing:
                adata_target.obs[k] = pd.Series(np.nan, index=adata_target.obs_names, dtype=object)
        super().transfer_field(state_registry, adata_target, **kwargs)
        src = state_registry[self.CELL_TYPES_KEY]
        cell_types = {k: list(src[k]) for k in state_registry[self.MAPPING_KEY]}
        if extend_categories:
            cell_types = self._collect(adata_target, cell_types)
        self._encode(adata_target, list(cell_types))
        return self._state(cell_types)

    def get_summary_stats(self, state_registry: dict) -> dict:
        """Number of prototypes (cell types over all levels) and of levels."""
        return {
            self.count_stat_key: len(state_registry.get(self.MAPPING_KEY, [])),
            self.levels_stat_key: len(self.attr_keys),
        }

    def view_state_registry(self, state_registry: dict) -> rich.table.Table | None:
        """Rich table of cell types, their codes and levels."""
        if self.is_empty:
            return None
        t = rich.table.Table(title=f"{self.registry_key} State Registry")
        t.add_column("Cell type", style="green")
        t.add_column("Code", style="dark_violet")
        t.add_column("Levels", style="dodger_blue1")
        for i, ct in enumerate(state_registry[self.MAPPING_KEY]):
            t.add_row(ct, str(i), ", ".join(state_registry[self.CELL_TYPES_KEY][ct]))
        return t
