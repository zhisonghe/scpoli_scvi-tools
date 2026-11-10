"""Registry keys and constants used by scpolivi."""

from typing import NamedTuple


class _SCPOLI_REGISTRY_KEYS_NT(NamedTuple):
    # Joint (n_cells, n_condition_keys) integer codes, one embedding per column.
    CONDITIONS_KEY: str = "scpoli_conditions"
    # Joint (n_cells, n_levels) integer codes in one shared cell-type namespace; -1 = unknown.
    CELL_TYPES_KEY: str = "scpoli_cell_types"
    # (n_cells, 1) float flag, 1 = labeled cell (used by the labeled prototype loss).
    LABELED_KEY: str = "scpoli_labeled"


SCPOLI_REGISTRY_KEYS = _SCPOLI_REGISTRY_KEYS_NT()

# obs columns written by setup_anndata
COMBINED_CONDITIONS_COL = "_scpoli_conditions_combined"
LABELED_COL = "_scpoli_labeled"

UNKNOWN_CODE = -1
